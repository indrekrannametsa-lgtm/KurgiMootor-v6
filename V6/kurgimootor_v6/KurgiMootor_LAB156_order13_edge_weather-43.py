from __future__ import annotations

"""
KurgiMootor · edge_weather-40
=============================

BASE vs FAST ARCHITECTURE VERDICT · STRICT WALK-FORWARD · READ ONLY

One question only:
Does the single FAST state discovered in LAB-39 improve practical BASE forecast
error when it is frozen exactly as-is?

Frozen FAST construction
------------------------
- one farm-wide common residual per completed harvest day
- causal expanding-median centering using only earlier strict-OOS days
- next harvested day receives exactly the previous completed day's centered anomaly
- multiplicative correction: FAST = BASE * exp(previous anomaly)
- coefficient is fixed at 1.00; there is no cap and no parameter search

Explicit exclusions
-------------------
- no weather / WIND×DRY correction
- no PI
- no LAST-2 / LAST-5 slow state
- no rotation correction
- no same-field previous-yield anchor
- no coefficient, threshold, lag or cap tuning

Evaluation lock
---------------
- development data end: 27.08.2026
- 30.08.2026 onward is reserved as untouched sequential holdout
- every BASE target row is trained only on intervals strictly before target
- target actual updates FAST only after that target has been predicted/scored
- no-harvest days are absent, never treated as zero yield
- READ ONLY: only db.get_harvest_history is called
"""
from dataclasses import dataclass
from datetime import date, datetime, timedelta
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import streamlit as st

import db

HOURS_PER_FIELD = 3.0
WEATHER_START = date(2026, 7, 1)  # BASE season origin; no weather is loaded
ABC_EPS = 0.20
MIN_BASE_TRAIN_INTERVALS = 35
MIN_FIELD_OBS = 2
BASE_FIELD_RIDGE = 1.5
BASE_SEASON_RIDGE = 0.10
BASE_MAX_ITER = 300

@dataclass
class Event:
    day: date
    field: int
    order: int
    abc: float
    interval_days: Optional[float] = None

def _d(v):
    if isinstance(v, datetime):
        return v.date()

    if isinstance(v, date):
        return v

    if v is None:
        return None

    try:
        return date.fromisoformat(
            str(v)[:10]
        )
    except Exception:
        return None

def _f(v):
    try:
        x = float(v)

        return (
            x
            if math.isfinite(x)
            else None
        )

    except Exception:
        return None

def _abc(row):
    vals = [
        _f(
            row.get(k)
        )
        for k in (
            "a",
            "b",
            "c",
        )
    ]

    if any(
        v is None
        for v in vals
    ):
        x = _f(
            row.get("_abc")
        )

        return (
            x
            if x is not None
            and x >= 0
            else None
        )

    return float(
        sum(vals)
    )

def _reliable(row):
    q = str(
        row.get("data_quality")
        or row.get("quality")
        or ""
    ).strip().lower()

    return q not in {
        "hinnanguline",
        "ligikaudne",
        "estimated",
        "approximate",
    }

def _events(rows):
    out = []

    for row in rows:
        dd = _d(
            row.get("harvest_date")
        )

        if (
            dd is None
            or not _reliable(row)
        ):
            continue

        try:
            field = int(
                row.get("field_no")
            )
        except Exception:
            continue

        if not 1 <= field <= 14:
            continue

        abc = _abc(row)

        if (
            abc is None
            or abc < 0
        ):
            continue

        try:
            order = int(
                row.get(
                    "harvest_order"
                )
                or 1
            )
        except Exception:
            order = 1

        out.append(
            Event(
                day=dd,
                field=field,
                order=order,
                abc=float(abc),
                interval_days=_f(
                    row.get(
                        "interval_days"
                    )
                ),
            )
        )

    return sorted(
        out,
        key=lambda e: (
            e.day,
            e.order,
            e.field,
        ),
    )

def _field_hist(
    events: Sequence[Event],
    field: int,
):
    return sorted(
        [
            e
            for e in events
            if e.field == field
        ],
        key=lambda e: (
            e.day,
            e.order,
            e.field,
        ),
    )

def _growth(
    prev: Event,
    cur: Event,
):
    g = float(
        (
            cur.day
            - prev.day
        ).days
    )

    g += (
        cur.order
        - prev.order
    ) * (
        HOURS_PER_FIELD
        / 24.0
    )

    return max(
        0.5,
        g,
    )

def _build_intervals(
    events: List[Event],
):
    rows = []

    for field in range(
        1,
        15,
    ):
        hist = _field_hist(
            events,
            field,
        )

        for i in range(
            1,
            len(hist),
        ):
            prev = hist[
                i - 1
            ]
            cur = hist[i]

            gap = int(
                (
                    cur.day
                    - prev.day
                ).days
            )

            if gap <= 0:
                continue

            growth = _growth(
                prev,
                cur,
            )

            days = [
                prev.day
                + timedelta(
                    days=k
                )
                for k in range(
                    1,
                    gap + 1,
                )
            ]

            rows.append({
                "target_date": cur.day,
                "start_date": prev.day,
                "field": int(field),
                "order": int(
                    cur.order
                ),
                "actual": float(
                    cur.abc
                ),
                "growth": float(
                    growth
                ),
                "days": days,
                "per_day_weight": (
                    float(growth)
                    / len(days)
                ),
            })

    df = pd.DataFrame(
        rows
    )

    if not df.empty:
        df = df.sort_values(
            [
                "target_date",
                "order",
                "field",
            ]
        ).reset_index(
            drop=True
        )

    return df


def _season_vec(
    dd: date,
):
    season = float(
        (
            dd
            - WEATHER_START
        ).days
    ) / 30.0

    return np.asarray(
        [
            1.0,
            season,
            season * season,
        ],
        dtype=float,
    )

def _base_cache(
    intervals,
):
    days = sorted({
        dd
        for interval_days
        in intervals[
            "days"
        ].tolist()
        for dd in interval_days
    })

    return {
        dd: _season_vec(
            dd
        )
        for dd in days
    }

def _predict_base_intervals(
    intervals,
    beta,
    gammas,
    cache,
):
    preds = []

    for _, row in (
        intervals.iterrows()
    ):
        X = np.vstack([
            cache[dd]
            for dd in row["days"]
        ])

        eta = (
            X @ beta
        )

        daily_prod = np.exp(
            np.clip(
                eta,
                -6.0,
                6.0,
            )
        )

        common = float(
            row[
                "per_day_weight"
            ]
            * np.sum(
                daily_prod
            )
        )

        field = int(
            row["field"]
        )

        if field == 1:
            ff = 1.0
        else:
            ff = math.exp(
                float(
                    gammas[
                        field - 2
                    ]
                )
            )

        preds.append(
            ff * common
        )

    return np.asarray(
        preds,
        dtype=float,
    )

def _fit_base(
    train,
):
    cache = _base_cache(
        train
    )

    y = train[
        "actual"
    ].to_numpy(
        dtype=float
    )

    growth = train[
        "growth"
    ].to_numpy(
        dtype=float
    )

    mean_daily = float(
        np.mean(
            y
            / np.maximum(
                growth,
                0.5,
            )
        )
    )

    beta = np.zeros(
        3,
        dtype=float,
    )

    beta[0] = math.log(
        max(
            mean_daily,
            0.05,
        )
    )

    gammas = np.zeros(
        13,
        dtype=float,
    )

    X_list = []
    w_list = []
    field_list = []

    for _, row in (
        train.iterrows()
    ):
        X_list.append(
            np.vstack([
                cache[dd]
                for dd in row[
                    "days"
                ]
            ])
        )

        w_list.append(
            float(
                row[
                    "per_day_weight"
                ]
            )
        )

        field_list.append(
            int(
                row["field"]
            )
        )

    lr = 0.035
    b1 = 0.9
    b2 = 0.999
    adam_eps = 1e-8

    mb = np.zeros_like(
        beta
    )
    vb = np.zeros_like(
        beta
    )
    mg = np.zeros_like(
        gammas
    )
    vg = np.zeros_like(
        gammas
    )

    prev_obj = None

    for step in range(
        1,
        BASE_MAX_ITER + 1,
    ):
        gb = np.zeros_like(
            beta
        )
        gg = np.zeros_like(
            gammas
        )

        obj_data = 0.0

        for i, (
            X,
            weight,
            field,
        ) in enumerate(
            zip(
                X_list,
                w_list,
                field_list,
            )
        ):
            eta = X @ beta

            prod = np.exp(
                np.clip(
                    eta,
                    -6.0,
                    6.0,
                )
            )

            common = float(
                weight
                * np.sum(
                    prod
                )
            )

            if field == 1:
                ff = 1.0
                gi = None
            else:
                gi = (
                    field - 2
                )
                ff = math.exp(
                    float(
                        gammas[gi]
                    )
                )

            pred = max(
                ff * common,
                1e-8,
            )

            resid = (
                math.log(
                    pred
                    + ABC_EPS
                )
                - math.log(
                    float(y[i])
                    + ABC_EPS
                )
            )

            obj_data += (
                resid
                * resid
            )

            shrink = (
                pred
                / (
                    pred
                    + ABC_EPS
                )
            )

            denom = max(
                float(
                    np.sum(
                        prod
                    )
                ),
                1e-12,
            )

            x_bar = (
                (
                    prod[:, None]
                    * X
                ).sum(axis=0)
                / denom
            )

            gb += (
                2.0
                * resid
                * shrink
                * x_bar
            )

            if gi is not None:
                gg[gi] += (
                    2.0
                    * resid
                    * shrink
                )

        reg = (
            BASE_SEASON_RIDGE
            * float(
                np.sum(
                    beta[1:]
                    * beta[1:]
                )
            )
            + BASE_FIELD_RIDGE
            * float(
                np.sum(
                    gammas
                    * gammas
                )
            )
        )

        gb[1:] += (
            2.0
            * BASE_SEASON_RIDGE
            * beta[1:]
        )

        gg += (
            2.0
            * BASE_FIELD_RIDGE
            * gammas
        )

        obj = (
            obj_data
            + reg
        )

        scale = max(
            len(train),
            1,
        )

        gb /= scale
        gg /= scale

        nb = float(
            np.linalg.norm(
                gb
            )
        )

        ng = float(
            np.linalg.norm(
                gg
            )
        )

        if nb > 10.0:
            gb *= (
                10.0
                / nb
            )

        if ng > 10.0:
            gg *= (
                10.0
                / ng
            )

        mb = (
            b1 * mb
            + (
                1.0
                - b1
            ) * gb
        )

        vb = (
            b2 * vb
            + (
                1.0
                - b2
            )
            * (
                gb * gb
            )
        )

        mg = (
            b1 * mg
            + (
                1.0
                - b1
            ) * gg
        )

        vg = (
            b2 * vg
            + (
                1.0
                - b2
            )
            * (
                gg * gg
            )
        )

        mbh = (
            mb
            / (
                1.0
                - b1 ** step
            )
        )

        vbh = (
            vb
            / (
                1.0
                - b2 ** step
            )
        )

        mgh = (
            mg
            / (
                1.0
                - b1 ** step
            )
        )

        vgh = (
            vg
            / (
                1.0
                - b2 ** step
            )
        )

        beta -= (
            lr
            * mbh
            / (
                np.sqrt(
                    vbh
                )
                + adam_eps
            )
        )

        gammas -= (
            lr
            * mgh
            / (
                np.sqrt(
                    vgh
                )
                + adam_eps
            )
        )

        gammas = np.clip(
            gammas,
            math.log(0.5),
            math.log(1.5),
        )

        if (
            prev_obj is not None
            and step > 80
            and abs(
                prev_obj
                - obj
            )
            < (
                1e-7
                * max(
                    1.0,
                    abs(
                        prev_obj
                    ),
                )
            )
        ):
            break

        prev_obj = obj

    return {
        "beta": beta,
        "gammas": gammas,
    }

def _predict_base(
    fit,
    test,
):
    cache = _base_cache(
        test
    )

    return _predict_base_intervals(
        test,
        fit["beta"],
        fit["gammas"],
        cache,
    )

def _strict_base_rows(
    intervals,
):
    rows = []

    for target in sorted(
        intervals[
            "target_date"
        ].unique()
    ):
        train = intervals[
            intervals[
                "target_date"
            ] < target
        ].copy()

        test = intervals[
            intervals[
                "target_date"
            ] == target
        ].copy()

        if len(train) < (
            MIN_BASE_TRAIN_INTERVALS
        ):
            continue

        counts = (
            train.groupby(
                "field"
            ).size().to_dict()
        )

        valid_idx = []

        for idx, row in (
            test.iterrows()
        ):
            if (
                counts.get(
                    int(
                        row["field"]
                    ),
                    0,
                )
                >= MIN_FIELD_OBS
            ):
                valid_idx.append(
                    idx
                )

        if not valid_idx:
            continue

        test = intervals.loc[
            valid_idx
        ].copy()

        fit = _fit_base(
            train
        )

        pred = _predict_base(
            fit,
            test,
        )

        for j, (
            idx,
            row,
        ) in enumerate(
            test.iterrows()
        ):
            rows.append({
                "target_date": target,
                "start_date": row["start_date"],
                "field": int(
                    row["field"]
                ),
                "order": int(
                    row["order"]
                ),
                "growth": float(
                    row["growth"]
                ),
                "actual": float(
                    row["actual"]
                ),
                "base": float(
                    pred[j]
                ),
                "train_n": int(
                    len(train)
                ),
            })

    return pd.DataFrame(
        rows
    )


# =====================================================================
# FAST STATE ARCHITECTURE VERDICT
# =====================================================================

FOCUS_START = date(2026, 8, 19)
FOCUS_END = date(2026, 8, 27)
DEV_END = date(2026, 8, 27)
HOLDOUT_START = date(2026, 8, 30)
CENTER_MIN_PRIOR_DAYS = 3
PRACTICAL_DIR_DEADBAND = 0.05


def _fmt_day(v):
    if isinstance(v, pd.Timestamp):
        v = v.date()
    return v.strftime('%d.%m') if isinstance(v, date) else str(v)


def _daily_fast_rows(strict):
    """Build one causal daily BASE vs FAST record.

    FAST is deliberately frozen from LAB-39:
      1) collapse the completed target day to one common median log residual,
      2) center that residual by an expanding median using only *earlier* OOS days,
      3) the next harvested day receives exactly the previous completed day's
         centered anomaly as a multiplicative log correction.

    No coefficient, no cap, no weather, no PI and no same-field yield anchor.
    """
    rows = []
    for target, g in strict.groupby('target_date', sort=True):
        if len(g) < 2:
            continue
        actual = g['actual'].to_numpy(dtype=float)
        base = g['base'].to_numpy(dtype=float)
        field_resids = np.log((actual + ABC_EPS) / (base + ABC_EPS))
        rows.append({
            'date': target,
            'fields': ','.join(str(int(v)) for v in sorted(g['field'].tolist())),
            'n_fields': int(len(g)),
            'actual': float(np.sum(actual)),
            'base': float(np.sum(base)),
            'common_resid': float(np.median(field_resids)),
            'train_n': int(g['train_n'].min()),
        })

    daily = pd.DataFrame(rows).sort_values('date').reset_index(drop=True)
    if daily.empty:
        return daily

    centers = []
    anomalies = []
    last_fast = np.nan
    fast_state = []
    fast_pred = []

    for i, row in daily.iterrows():
        # Prediction state for this target was available before target actual:
        # it is the prior completed day's previously computed anomaly.
        state = last_fast
        fast_state.append(state)
        fast_pred.append(float(row['base'] * math.exp(state)) if pd.notna(state) else np.nan)

        # Score current residual only after the day's prediction, then update state.
        prior = daily.loc[:i-1, 'common_resid'].to_numpy(dtype=float) if i > 0 else np.asarray([], dtype=float)
        if len(prior) >= CENTER_MIN_PRIOR_DAYS:
            center = float(np.median(prior))
            anomaly = float(row['common_resid'] - center)
        else:
            center = np.nan
            anomaly = np.nan
        centers.append(center)
        anomalies.append(anomaly)
        last_fast = anomaly

    daily['causal_center'] = centers
    daily['anomaly'] = anomalies
    daily['fast_state'] = fast_state
    daily['fast'] = fast_pred
    daily['base_error'] = daily['base'] - daily['actual']
    daily['fast_error'] = daily['fast'] - daily['actual']
    return daily


def _practical_metrics(df, pred_col):
    x = df[['date','actual','n_fields',pred_col]].dropna().copy()
    if x.empty:
        return {
            'N': 0, 'MAE': np.nan, 'MAE/põld': np.nan, 'MAPE %': np.nan,
            'Median AE': np.nan, 'Bias': np.nan, '±10%': np.nan, '±20%': np.nan,
            'Worst AE': np.nan, 'Direction hit %': np.nan, 'Direction N': 0,
        }

    actual = x['actual'].to_numpy(dtype=float)
    pred = x[pred_col].to_numpy(dtype=float)
    err = pred - actual
    ae = np.abs(err)
    denom = np.maximum(np.abs(actual), 1e-9)
    rel = ae / denom

    # Trend is scored in boxes/field so 2-field days do not create a false fall.
    act_pf = actual / x['n_fields'].to_numpy(dtype=float)
    pred_pf = pred / x['n_fields'].to_numpy(dtype=float)
    dir_hits = []
    for i in range(1, len(x)):
        prev_a = act_pf[i-1]
        if abs(prev_a) <= 1e-9:
            continue
        actual_change = (act_pf[i] - prev_a) / abs(prev_a)
        if abs(actual_change) < PRACTICAL_DIR_DEADBAND:
            continue
        pred_change = pred_pf[i] - pred_pf[i-1]
        actual_delta = act_pf[i] - act_pf[i-1]
        dir_hits.append(np.sign(pred_change) == np.sign(actual_delta))

    return {
        'N': int(len(x)),
        'MAE': float(np.mean(ae)),
        'MAE/põld': float(np.mean(ae / x['n_fields'].to_numpy(dtype=float))),
        'MAPE %': float(np.mean(rel) * 100.0),
        'Median AE': float(np.median(ae)),
        'Bias': float(np.mean(err)),
        '±10%': float(np.mean(rel <= 0.10) * 100.0),
        '±20%': float(np.mean(rel <= 0.20) * 100.0),
        'Worst AE': float(np.max(ae)),
        'Direction hit %': float(np.mean(dir_hits) * 100.0) if dir_hits else np.nan,
        'Direction N': int(len(dir_hits)),
    }


def _comparison_table(df):
    mb = _practical_metrics(df, 'base')
    mf = _practical_metrics(df, 'fast')
    rows = []
    for name, m in [('A · BASE', mb), ('B · BASE + FAST', mf)]:
        row = {'Variant': name, **m}
        if name.startswith('B') and pd.notna(mb['MAE']) and mb['MAE'] > 1e-12 and pd.notna(m['MAE']):
            row['Parandus vs BASE %'] = 100.0 * (mb['MAE'] - m['MAE']) / mb['MAE']
        else:
            row['Parandus vs BASE %'] = 0.0 if name.startswith('A') else np.nan
        rows.append(row)
    return pd.DataFrame(rows), mb, mf


def _half_table(df):
    x = df.dropna(subset=['fast']).sort_values('date').reset_index(drop=True)
    if x.empty:
        return pd.DataFrame()
    cut = len(x) // 2
    parts = [('I pool', x.iloc[:cut].copy()), ('II pool', x.iloc[cut:].copy())]
    rows = []
    for label, part in parts:
        if part.empty:
            continue
        base = _practical_metrics(part, 'base')
        fast = _practical_metrics(part, 'fast')
        for variant, m in [('BASE', base), ('BASE+FAST', fast)]:
            imp = (100.0 * (base['MAE'] - m['MAE']) / base['MAE']) if base['MAE'] > 1e-12 else np.nan
            rows.append({
                'Periood': label,
                'Variant': variant,
                'N': m['N'],
                'MAE': m['MAE'],
                'MAPE %': m['MAPE %'],
                '±20%': m['±20%'],
                'Bias': m['Bias'],
                'Parandus vs BASE %': 0.0 if variant == 'BASE' else imp,
            })
    return pd.DataFrame(rows)


def _dev_verdict(dev):
    common = dev.dropna(subset=['fast']).copy()
    if len(common) < 8:
        return 'INSUFFICIENT', 'Strict common sample is too small for an architecture verdict.'

    _, base, fast = _comparison_table(common)
    half = _half_table(common)
    imp = 100.0 * (base['MAE'] - fast['MAE']) / base['MAE'] if base['MAE'] > 1e-12 else np.nan

    half_fast = half[half['Variant'] == 'BASE+FAST'] if not half.empty else pd.DataFrame()
    both_halves_positive = (
        len(half_fast) == 2 and
        bool((half_fast['Parandus vs BASE %'] > 0.0).all())
    )
    practical_not_worse = (
        pd.notna(fast['±20%']) and pd.notna(base['±20%']) and fast['±20%'] >= base['±20%']
    )

    if pd.notna(imp) and imp >= 5.0 and both_halves_positive and practical_not_worse:
        return 'HISTORICAL PASS · FREEZE FAST, WAIT FOR HOLDOUT', (
            f'FAST improves development MAE by {imp:.1f}% and improves both time halves without reducing the ±20% hit rate. '
            'This is the first state architecture to pass the pre-declared historical stability gate. Do not promote it to production yet; '
            'freeze the rule exactly as-is and score new harvests from 30.08 onward as untouched holdout.'
        )
    if pd.notna(imp) and imp > 0.0:
        return 'MIXED · HISTORICAL IMPROVEMENT NOT STABLE ENOUGH', (
            f'FAST improves aggregate development MAE by {imp:.1f}%, but it does not pass every stability/practical gate. '
            'Do not promote or tune it on these days.'
        )
    return 'FAIL · KEEP BASE', (
        'The fixed one-day FAST state does not improve the strict development sample overall. Keep BASE and do not tune FAST.'
    )


def _holdout_verdict(holdout):
    x = holdout.dropna(subset=['fast']).copy()
    if x.empty:
        return None, None
    _, base, fast = _comparison_table(x)
    imp = 100.0 * (base['MAE'] - fast['MAE']) / base['MAE'] if base['MAE'] > 1e-12 else np.nan
    if len(x) < 3:
        return 'EARLY HOLDOUT', f'{len(x)} holdout day(s) available; report only, do not decide yet. FAST vs BASE MAE change {imp:+.1f}%.'
    if pd.notna(imp) and imp > 0 and fast['±20%'] >= base['±20%']:
        return 'HOLDOUT SUPPORTS FAST', f'{len(x)} untouched holdout days: FAST improves MAE by {imp:.1f}% without lowering ±20% hit rate.'
    return 'HOLDOUT DOES NOT SUPPORT FAST YET', f'{len(x)} untouched holdout days: FAST MAE change {imp:+.1f}% vs BASE.'


def main():
    st.set_page_config(page_title='KurgiMootor · BASE vs FAST', layout='wide')
    st.title('KurgiMootor · BASE vs FAST architecture verdict')
    st.caption('LAB-40 · one fixed add-on only · strict walk-forward · READ ONLY')
    st.info(
        'LAB-39 found that the previous completed harvest-day common residual was the only state proxy that improved both time halves. '
        'This LAB freezes that exact construction and asks one practical question: does BASE + FAST beat BASE in real box error? '
        'No weather, PI, LAST-2/5, rotation layer, coefficient search or correction cap is allowed.'
    )

    try:
        harvest = db.get_harvest_history(limit=5000)
        events = _events(harvest)
        intervals = _build_intervals(events)
        strict = _strict_base_rows(intervals)
        if strict.empty:
            raise RuntimeError('Strict BASE OOS rows did not form.')
        daily = _daily_fast_rows(strict)
        if daily.empty:
            raise RuntimeError('No complete strict-OOS daily rows formed.')
    except Exception as exc:
        st.exception(exc)
        st.stop()

    dev = daily[daily['date'] <= DEV_END].copy()
    holdout = daily[daily['date'] >= HOLDOUT_START].copy()
    verdict, verdict_text = _dev_verdict(dev)

    st.markdown('### 1. Otsus · kas FAST väärib päris tuleviku holdout’i?')
    if verdict.startswith('HISTORICAL PASS'):
        st.success('✅ ' + verdict + ': ' + verdict_text)
    elif verdict.startswith('FAIL'):
        st.error('⛔ ' + verdict + ': ' + verdict_text)
    else:
        st.warning('🟡 ' + verdict + ': ' + verdict_text)

    common_dev = dev.dropna(subset=['fast']).copy()
    comp, _, _ = _comparison_table(common_dev)
    st.dataframe(comp.style.format({
        'MAE': lambda v: '—' if pd.isna(v) else f'{float(v):.2f}',
        'MAE/põld': lambda v: '—' if pd.isna(v) else f'{float(v):.2f}',
        'MAPE %': lambda v: '—' if pd.isna(v) else f'{float(v):.1f}',
        'Median AE': lambda v: '—' if pd.isna(v) else f'{float(v):.2f}',
        'Bias': lambda v: '—' if pd.isna(v) else f'{float(v):+.2f}',
        '±10%': lambda v: '—' if pd.isna(v) else f'{float(v):.0f}%',
        '±20%': lambda v: '—' if pd.isna(v) else f'{float(v):.0f}%',
        'Worst AE': lambda v: '—' if pd.isna(v) else f'{float(v):.2f}',
        'Direction hit %': lambda v: '—' if pd.isna(v) else f'{float(v):.0f}%',
        'Parandus vs BASE %': lambda v: '—' if pd.isna(v) else f'{float(v):+.1f}%',
    }), use_container_width=True, hide_index=True)

    st.caption(
        'FAST correction is exactly exp(previous completed harvest-day centered common residual). Coefficient = 1.00, no cap. '
        'The current target actual is never available when its FAST state is formed.'
    )

    st.markdown('### 2. Stabiilsus · esimene pool vs teine pool')
    half = _half_table(common_dev)
    st.dataframe(half.style.format({
        'MAE': lambda v: '—' if pd.isna(v) else f'{float(v):.2f}',
        'MAPE %': lambda v: '—' if pd.isna(v) else f'{float(v):.1f}',
        '±20%': lambda v: '—' if pd.isna(v) else f'{float(v):.0f}%',
        'Bias': lambda v: '—' if pd.isna(v) else f'{float(v):+.2f}',
        'Parandus vs BASE %': lambda v: '—' if pd.isna(v) else f'{float(v):+.1f}%',
    }), use_container_width=True, hide_index=True)

    st.markdown('### 3. 19.–27.08 · päev-päevalt')
    focus = common_dev[(common_dev['date'] >= FOCUS_START) & (common_dev['date'] <= FOCUS_END)].copy()
    show = focus if not focus.empty else common_dev.tail(10).copy()
    view = show[[
        'date','fields','n_fields','actual','base','fast','base_error','fast_error',
        'fast_state','common_resid','causal_center','anomaly','train_n'
    ]].rename(columns={
        'date':'Päev','fields':'Põllud','n_fields':'N põldu','actual':'Tegelik ABC','base':'BASE',
        'fast':'BASE+FAST','base_error':'BASE viga','fast_error':'FAST viga','fast_state':'FAST state',
        'common_resid':'Ühine log-viga','causal_center':'Causal center','anomaly':'Tänane anomaalia','train_n':'BASE train N'
    })
    st.dataframe(view.style.format({
        'Päev': _fmt_day,
        'Tegelik ABC':'{:.1f}','BASE':'{:.1f}','BASE+FAST':'{:.1f}',
        'BASE viga':'{:+.1f}','FAST viga':'{:+.1f}',
        'FAST state': lambda v: '—' if pd.isna(v) else f'{float(v):+.3f}',
        'Ühine log-viga': lambda v: '—' if pd.isna(v) else f'{float(v):+.3f}',
        'Causal center': lambda v: '—' if pd.isna(v) else f'{float(v):+.3f}',
        'Tänane anomaalia': lambda v: '—' if pd.isna(v) else f'{float(v):+.3f}',
    }), use_container_width=True, hide_index=True)

    st.markdown('### 4. Tuleviku holdout · lukus alates 30.08')
    hv, ht = _holdout_verdict(holdout)
    if hv is None:
        st.info('🔒 Holdout on tühi. Reegel on nüüd lukustatud; 30.08+ uusi korjeid ei tohi kasutada FAST-i ümberhäälestamiseks.')
    else:
        st.warning('🧪 ' + hv + ': ' + ht)
        hcommon = holdout.dropna(subset=['fast']).copy()
        hcomp, _, _ = _comparison_table(hcommon)
        st.dataframe(hcomp.style.format({
            'MAE': lambda v: '—' if pd.isna(v) else f'{float(v):.2f}',
            'MAE/põld': lambda v: '—' if pd.isna(v) else f'{float(v):.2f}',
            'MAPE %': lambda v: '—' if pd.isna(v) else f'{float(v):.1f}',
            'Median AE': lambda v: '—' if pd.isna(v) else f'{float(v):.2f}',
            'Bias': lambda v: '—' if pd.isna(v) else f'{float(v):+.2f}',
            '±10%': lambda v: '—' if pd.isna(v) else f'{float(v):.0f}%',
            '±20%': lambda v: '—' if pd.isna(v) else f'{float(v):.0f}%',
            'Worst AE': lambda v: '—' if pd.isna(v) else f'{float(v):.2f}',
            'Direction hit %': lambda v: '—' if pd.isna(v) else f'{float(v):.0f}%',
            'Parandus vs BASE %': lambda v: '—' if pd.isna(v) else f'{float(v):+.1f}%',
        }), use_container_width=True, hide_index=True)

        hview = hcommon[['date','fields','actual','base','fast','base_error','fast_error','fast_state']].rename(columns={
            'date':'Päev','fields':'Põllud','actual':'Tegelik ABC','base':'BASE','fast':'BASE+FAST',
            'base_error':'BASE viga','fast_error':'FAST viga','fast_state':'FAST state'
        })
        st.dataframe(hview.style.format({
            'Päev': _fmt_day,'Tegelik ABC':'{:.1f}','BASE':'{:.1f}','BASE+FAST':'{:.1f}',
            'BASE viga':'{:+.1f}','FAST viga':'{:+.1f}',
            'FAST state': lambda v: '—' if pd.isna(v) else f'{float(v):+.3f}',
        }), use_container_width=True, hide_index=True)

    with st.expander('Kõik strict-OOS päevad'):
        st.dataframe(daily, use_container_width=True, hide_index=True)

    st.success(
        '🔒 LEAKAGE / ARCHITECTURE LOCK: BASE target is trained only on earlier intervals. FAST target state is only the previous '
        'completed strict-OOS harvest-day anomaly. Target actual updates state only after that target is scored. Development ends '
        f'{DEV_END.strftime("%d.%m.%Y")}; {HOLDOUT_START.strftime("%d.%m.%Y")}+ is untouched holdout. READ ONLY; production unchanged.'
    )


if __name__ == '__main__':
    main()
