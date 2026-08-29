from __future__ import annotations

"""
KurgiMootor · edge_weather-39
=============================

COMMON CROP-STATE / LAG AUDIT · STRICT CAUSAL TIMING · READ ONLY

One question only:
Does BASE leave behind a farm-wide residual process with useful memory on a
FAST (1–2 harvested days) or ROTATION (about 5 / 10 harvested days) horizon?

This LAB does NOT add a state correction to BASE. It first collapses strict
field-level BASE errors into one common daily log-residual, then asks whether
that common residual has causal persistence, rotation memory, momentum or
mean-reversion that was missing from BASE.

Pre-declared state probes
-------------------------
- LAST DAY: previous completed harvest-day common anomaly
- LAST 2: median of previous 2 completed harvest-day anomalies
- LAST 5: median of previous 5 completed harvest-day anomalies
  (explicit reference to the earlier slow-state idea)
- ROTATION-1: anomaly exactly 5 harvested days earlier (~one 14-field rotation)
- ROTATION-2: anomaly exactly 10 harvested days earlier (~two rotations)
- DELTA DYNAMICS: previous anomaly change versus the next anomaly change;
  both momentum and mean-reversion are reported, neither is tuned.

Important centering rule
------------------------
The target day's common anomaly is residual minus the expanding median of
ONLY earlier completed strict-OOS days. This prevents a fixed BASE bias from
masquerading as crop-state memory. The target actual is never used before its
prediction; it is used only afterwards to score that day's common residual.

Locks
-----
- every BASE target row is trained only on intervals strictly before target
- target actual is used only after prediction for scoring
- no weather, no PI, no previous-yield anchor, no fitted state coefficient
- lag horizons are fixed in advance: 1, 2, 5 and 10 harvested days
- no lag search, coefficient search, threshold search or correction size
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
# COMMON CROP-STATE / LAG AUDIT
# =====================================================================

FOCUS_START = date(2026, 8, 19)
FOCUS_END = date(2026, 8, 27)
STATE_DEADBAND = 0.03
CENTER_MIN_PRIOR_DAYS = 3
ROTATION_DAYS = 5
TWO_ROTATION_DAYS = 10


def _fmt_day(v):
    if isinstance(v, pd.Timestamp):
        v = v.date()
    return v.strftime('%d.%m') if isinstance(v, date) else str(v)


def _rank_average(values):
    s = pd.Series(np.asarray(values, dtype=float))
    return s.rank(method='average').to_numpy(dtype=float)


def _safe_spearman(a, b):
    x = pd.DataFrame({'a': pd.to_numeric(a, errors='coerce'), 'b': pd.to_numeric(b, errors='coerce')}).dropna()
    if len(x) < 3 or x['a'].nunique() < 2 or x['b'].nunique() < 2:
        return np.nan
    ra = _rank_average(x['a'].to_numpy(dtype=float))
    rb = _rank_average(x['b'].to_numpy(dtype=float))
    if np.std(ra) <= 1e-12 or np.std(rb) <= 1e-12:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def _binom_tail_half(k, n):
    if n <= 0:
        return np.nan
    k = int(k); n = int(n)
    return float(sum(math.comb(n, i) for i in range(k, n + 1)) / (2.0 ** n))


def _daily_common_rows(strict):
    rows = []
    for target, g in strict.groupby('target_date', sort=True):
        if len(g) < 2:
            continue
        field_resids = np.log((g['actual'].to_numpy(dtype=float) + ABC_EPS) / (g['base'].to_numpy(dtype=float) + ABC_EPS))
        rows.append({
            'date': target,
            'fields': ','.join(str(int(v)) for v in sorted(g['field'].tolist())),
            'n_fields': int(len(g)),
            'actual': float(g['actual'].sum()),
            'base': float(g['base'].sum()),
            'actual_minus_base': float(g['actual'].sum() - g['base'].sum()),
            'common_resid': float(np.median(field_resids)),
            'field_resid_spread': float(np.median(np.abs(field_resids - np.median(field_resids)))),
            'train_n': int(g['train_n'].min()),
        })
    daily = pd.DataFrame(rows).sort_values('date').reset_index(drop=True)
    if daily.empty:
        return daily

    # Strict causal centering. Current target residual is scored after prediction,
    # while its center uses only previously completed OOS days.
    centers = []
    anomalies = []
    for i, row in daily.iterrows():
        prior = daily.loc[:i-1, 'common_resid'].to_numpy(dtype=float) if i > 0 else np.asarray([], dtype=float)
        if len(prior) >= CENTER_MIN_PRIOR_DAYS:
            center = float(np.median(prior))
            anomaly = float(row['common_resid'] - center)
        else:
            center = np.nan
            anomaly = np.nan
        centers.append(center)
        anomalies.append(anomaly)
    daily['causal_center'] = centers
    daily['anomaly'] = anomalies

    # Pre-declared causal state probes. All values are from completed earlier days.
    for i in range(len(daily)):
        hist = daily.loc[:i-1, 'anomaly'].dropna().to_numpy(dtype=float) if i > 0 else np.asarray([], dtype=float)
        daily.loc[i, 'last1'] = hist[-1] if len(hist) >= 1 else np.nan
        daily.loc[i, 'last2_med'] = float(np.median(hist[-2:])) if len(hist) >= 2 else np.nan
        daily.loc[i, 'last5_med'] = float(np.median(hist[-5:])) if len(hist) >= 5 else np.nan
        # Exact harvested-day lags, not calendar-day lags.
        daily.loc[i, 'lag5'] = daily.loc[i-ROTATION_DAYS, 'anomaly'] if i >= ROTATION_DAYS else np.nan
        daily.loc[i, 'lag10'] = daily.loc[i-TWO_ROTATION_DAYS, 'anomaly'] if i >= TWO_ROTATION_DAYS else np.nan

        if i >= 2 and pd.notna(daily.loc[i-1, 'anomaly']) and pd.notna(daily.loc[i-2, 'anomaly']):
            daily.loc[i, 'prev_delta'] = float(daily.loc[i-1, 'anomaly'] - daily.loc[i-2, 'anomaly'])
        else:
            daily.loc[i, 'prev_delta'] = np.nan
        if i >= 1 and pd.notna(daily.loc[i, 'anomaly']) and pd.notna(daily.loc[i-1, 'anomaly']):
            daily.loc[i, 'current_delta'] = float(daily.loc[i, 'anomaly'] - daily.loc[i-1, 'anomaly'])
        else:
            daily.loc[i, 'current_delta'] = np.nan

    return daily


def _sign(v, deadband=STATE_DEADBAND):
    if pd.isna(v):
        return np.nan
    x = float(v)
    if x > deadband:
        return 1.0
    if x < -deadband:
        return -1.0
    return 0.0


def _proxy_metrics(daily, col):
    x = daily[['anomaly', col]].dropna().copy()
    if x.empty:
        return {'N': 0, 'Scored N': 0, 'Coverage %': np.nan, 'Suunahitt %': np.nan,
                'rho': np.nan, 'State MAE': np.nan, 'Zero MAE': np.nan, 'Parandus vs 0 %': np.nan,
                'Binom p≥hits': np.nan}
    x['truth'] = x['anomaly'].map(_sign)
    x['pred'] = x[col].map(_sign)
    scored = x[(x['truth'] != 0) & (x['pred'] != 0)].copy()
    hits = int((scored['truth'] == scored['pred']).sum()) if not scored.empty else 0
    n = int(len(scored))
    state_mae = float(np.mean(np.abs(x['anomaly'] - x[col])))
    zero_mae = float(np.mean(np.abs(x['anomaly'])))
    improvement = 100.0 * (zero_mae - state_mae) / zero_mae if zero_mae > 1e-12 else np.nan
    return {
        'N': int(len(x)),
        'Scored N': n,
        'Coverage %': 100.0 * n / len(x) if len(x) else np.nan,
        'Suunahitt %': 100.0 * hits / n if n else np.nan,
        'rho': _safe_spearman(x['anomaly'], x[col]),
        'State MAE': state_mae,
        'Zero MAE': zero_mae,
        'Parandus vs 0 %': improvement,
        'Binom p≥hits': _binom_tail_half(hits, n) if n else np.nan,
    }


def _delta_metrics(daily):
    x = daily[['current_delta', 'prev_delta']].dropna().copy()
    if x.empty:
        return {'N': 0, 'rho': np.nan, 'Momentum hit %': np.nan, 'Mean-reversion hit %': np.nan}
    # Tiny change is treated as abstain for sign scoring, but still remains in rho.
    x['cur_dir'] = x['current_delta'].map(_sign)
    x['prev_dir'] = x['prev_delta'].map(_sign)
    scored = x[(x['cur_dir'] != 0) & (x['prev_dir'] != 0)]
    if scored.empty:
        mom = mr = np.nan
    else:
        mom = float(np.mean(scored['cur_dir'] == scored['prev_dir']) * 100.0)
        mr = float(np.mean(scored['cur_dir'] == -scored['prev_dir']) * 100.0)
    return {
        'N': int(len(x)),
        'rho': _safe_spearman(x['current_delta'], x['prev_delta']),
        'Momentum hit %': mom,
        'Mean-reversion hit %': mr,
    }


def _half_table(daily):
    valid = daily[pd.notna(daily['anomaly'])].copy().reset_index(drop=True)
    if valid.empty:
        return pd.DataFrame()
    cut = len(valid) // 2
    parts = [('I pool', valid.iloc[:cut].copy()), ('II pool', valid.iloc[cut:].copy())]
    rows = []
    for label, part in parts:
        if part.empty:
            continue
        for col, name in [('last1','LAST DAY'), ('last2_med','LAST 2'), ('last5_med','LAST 5'), ('lag5','ROTATION-1')]:
            m = _proxy_metrics(part, col)
            rows.append({
                'Periood': label,
                'Proxy': name,
                'Päevi': int(len(part)),
                'N': m['N'],
                'Suunahitt %': m['Suunahitt %'],
                'rho': m['rho'],
                'Parandus vs 0 %': m['Parandus vs 0 %'],
            })
    return pd.DataFrame(rows)


def _lag_profile(daily):
    # Fixed audit lags only. No search and no winner is promoted automatically.
    rows = []
    for lag in [1, 2, 5, 10]:
        a = daily['anomaly']
        b = daily['anomaly'].shift(lag)
        x = pd.DataFrame({'cur': a, 'lag': b}).dropna()
        scored = x[(x['cur'].map(_sign) != 0) & (x['lag'].map(_sign) != 0)].copy()
        rows.append({
            'Lag harvested days': lag,
            'Tõlgendus': {1:'eelmine korjepäev', 2:'2 korjepäeva', 5:'~1 korjering', 10:'~2 korjeringi'}[lag],
            'N': int(len(x)),
            'Scored N': int(len(scored)),
            'rho': _safe_spearman(x['cur'], x['lag']) if not x.empty else np.nan,
            'Sama märk %': (float(np.mean(scored['cur'].map(_sign) == scored['lag'].map(_sign)) * 100.0)
                            if not scored.empty else np.nan),
        })
    return pd.DataFrame(rows)


def _verdict(daily):
    m1 = _proxy_metrics(daily, 'last1')
    m2 = _proxy_metrics(daily, 'last2_med')
    m5 = _proxy_metrics(daily, 'last5_med')
    r5 = _proxy_metrics(daily, 'lag5')
    r10 = _proxy_metrics(daily, 'lag10')
    dm = _delta_metrics(daily)

    fast = (
        m1['N'] >= 8 and pd.notna(m1['rho']) and m1['rho'] >= 0.40 and
        pd.notna(m1['Suunahitt %']) and m1['Suunahitt %'] >= 60.0
    ) or (
        m2['N'] >= 8 and pd.notna(m2['rho']) and m2['rho'] >= 0.40 and
        pd.notna(m2['Suunahitt %']) and m2['Suunahitt %'] >= 60.0
    )
    rotation = r5['N'] >= 7 and pd.notna(r5['rho']) and abs(r5['rho']) >= 0.40
    two_rotation = r10['N'] >= 5 and pd.notna(r10['rho']) and abs(r10['rho']) >= 0.45
    dynamic = dm['N'] >= 7 and pd.notna(dm['rho']) and abs(dm['rho']) >= 0.35

    if fast and rotation:
        relation = 'persistence' if r5['rho'] > 0 else 'reversal'
        return 'MULTI-SCALE STATE · FAST MEMORY + ROTATION ' + relation.upper(), (
            'The common residual persists over the previous 1–2 harvested days but also has a distinct fixed one-rotation '
            f'{relation} pattern. That is the wave-like structure we were looking for: a fast state cannot be represented by '
            'the old 5-day slow median. This is diagnostic evidence only; do not add a correction yet.'
        )
    if fast and not rotation:
        return 'FAST COMMON STATE, NOT SLOW ROTATION STATE', (
            'The common BASE residual carries usable short-memory structure over the previous 1–2 harvested days, '
            'while exact one-rotation memory is weaker. This would explain why the earlier 5-day slow-state layer lagged turns. '
            'Do not add a correction yet; preserve a future holdout for a pre-declared fast-state rule.'
        )
    if rotation or two_rotation:
        direction = 'persistence' if (pd.notna(r5['rho']) and r5['rho'] > 0) else 'reversal'
        return 'ROTATION-SCALE MEMORY PRESENT', (
            f'The common residual shows {direction} at a fixed ~one-rotation lag, with possible second-rotation structure. '
            'This is evidence for a crop-state/lag process, not yet a production correction.'
        )
    if dynamic:
        mode = 'momentum' if dm['rho'] > 0 else 'mean-reversion'
        return 'CHANGE DYNAMICS PRESENT', (
            f'Level memory is not dominant, but residual changes show {mode}. '
            'The next test should focus on turning dynamics rather than a smoothed state level.'
        )
    if pd.notna(m5['rho']) and m5['rho'] > 0.35 and (pd.isna(m1['rho']) or m5['rho'] > m1['rho']):
        return 'SLOW STATE POSSIBLE, NOT ESTABLISHED', (
            'The previous 5 harvested days retain some common-state information, but the available strict-OOS sample '
            'is too small or unstable to justify reviving the old slow-state correction.'
        )
    return 'NO STABLE CROP-STATE LAG YET', (
        'After removing the causal expanding BASE bias, the pre-declared 1/2/5/10 harvested-day probes do not show '
        'a stable enough common-state process. Do not manufacture another state layer from this evidence.'
    )


def main():
    st.set_page_config(page_title='KurgiMootor · crop-state lag audit', layout='wide')
    st.title('KurgiMootor · common crop-state / lag audit')
    st.caption('LAB-39 · BASE fixed · no weather · no fitted state coefficient · strict causal timing · READ ONLY')
    st.info(
        'LAB-36 proved the BASE miss is largely common across fields. LAB-37/38 showed WIND×DRY contains structure, '
        'but not a stable symmetric production correction. This LAB now asks whether the remaining common residual itself '
        'has 1–2 harvest-day or 1–2 rotation memory. It does not modify BASE.'
    )

    try:
        harvest = db.get_harvest_history(limit=5000)
        events = _events(harvest)
        intervals = _build_intervals(events)
        strict = _strict_base_rows(intervals)
        if strict.empty:
            raise RuntimeError('Strict BASE OOS rows did not form.')
        daily = _daily_common_rows(strict)
        if daily.empty:
            raise RuntimeError('No complete strict-OOS common daily rows formed.')
        verdict, verdict_text = _verdict(daily)
    except Exception as exc:
        st.exception(exc)
        st.stop()

    st.markdown('### 1. Otsus · kas ühisel saagivoolul on päris mälu?')
    if verdict.startswith('NO STABLE'):
        st.error('⛔ ' + verdict + ': ' + verdict_text)
    elif verdict.startswith('FAST') or verdict.startswith('ROTATION') or verdict.startswith('CHANGE'):
        st.warning('🟡 ' + verdict + ': ' + verdict_text)
    else:
        st.warning('🟡 ' + verdict + ': ' + verdict_text)

    rows = []
    for col, label in [
        ('last1','LAST DAY · eelmine korjepäev'),
        ('last2_med','LAST 2 · eelmise 2 päeva mediaan'),
        ('last5_med','LAST 5 · vana slow-state võrdlus'),
        ('lag5','ROTATION-1 · täpselt 5 korjepäeva tagasi'),
        ('lag10','ROTATION-2 · täpselt 10 korjepäeva tagasi'),
    ]:
        m = _proxy_metrics(daily, col)
        rows.append({'Proxy': label, **m})
    score = pd.DataFrame(rows)
    st.dataframe(score.style.format({
        'Coverage %': lambda v: '—' if pd.isna(v) else f'{float(v):.0f}%',
        'Suunahitt %': lambda v: '—' if pd.isna(v) else f'{float(v):.0f}%',
        'rho': lambda v: '—' if pd.isna(v) else f'{float(v):+.2f}',
        'State MAE': lambda v: '—' if pd.isna(v) else f'{float(v):.3f}',
        'Zero MAE': lambda v: '—' if pd.isna(v) else f'{float(v):.3f}',
        'Parandus vs 0 %': lambda v: '—' if pd.isna(v) else f'{float(v):+.1f}%',
        'Binom p≥hits': lambda v: '—' if pd.isna(v) else f'{float(v):.3f}',
    }), use_container_width=True, hide_index=True)

    st.caption(
        '“Zero MAE” means no common-state adjustment after a causal expanding-median centering. '
        'A proxy must beat zero and remain directionally stable to be interesting; rho alone is not enough.'
    )

    st.markdown('### 2. Fikseeritud lag-profiil · 1 / 2 / 5 / 10 korjepäeva')
    lagtab = _lag_profile(daily)
    st.dataframe(lagtab.style.format({
        'rho': lambda v: '—' if pd.isna(v) else f'{float(v):+.2f}',
        'Sama märk %': lambda v: '—' if pd.isna(v) else f'{float(v):.0f}%',
    }), use_container_width=True, hide_index=True)

    st.markdown('### 3. Muutuse dünaamika · momentum või mean-reversion?')
    dm = _delta_metrics(daily)
    dyn = pd.DataFrame([dm])
    st.dataframe(dyn.style.format({
        'rho': lambda v: '—' if pd.isna(v) else f'{float(v):+.2f}',
        'Momentum hit %': lambda v: '—' if pd.isna(v) else f'{float(v):.0f}%',
        'Mean-reversion hit %': lambda v: '—' if pd.isna(v) else f'{float(v):.0f}%',
    }), use_container_width=True, hide_index=True)

    st.markdown('### 4. Stabiilsus · esimene pool vs teine pool')
    half = _half_table(daily)
    st.dataframe(half.style.format({
        'Suunahitt %': lambda v: '—' if pd.isna(v) else f'{float(v):.0f}%',
        'rho': lambda v: '—' if pd.isna(v) else f'{float(v):+.2f}',
        'Parandus vs 0 %': lambda v: '—' if pd.isna(v) else f'{float(v):+.1f}%',
    }), use_container_width=True, hide_index=True)

    st.markdown('### 5. 19.–27.08 · kas ühine laine oli ette nähtav ainult varasema state’iga?')
    focus = daily[(daily['date'] >= FOCUS_START) & (daily['date'] <= FOCUS_END)].copy()
    show = focus if not focus.empty else daily.tail(10).copy()
    st.dataframe(show[[
        'date','fields','actual','base','actual_minus_base','common_resid','causal_center','anomaly',
        'last1','last2_med','last5_med','lag5','lag10','prev_delta','current_delta'
    ]].rename(columns={
        'date':'Päev','fields':'Põllud','actual':'Tegelik ABC','base':'BASE','actual_minus_base':'Tegelik−BASE',
        'common_resid':'Ühine log-viga','causal_center':'Varasem center','anomaly':'Ühine anomaalia',
        'last1':'LAST DAY','last2_med':'LAST 2','last5_med':'LAST 5','lag5':'ROT-1','lag10':'ROT-2',
        'prev_delta':'Eelmine Δstate','current_delta':'Praegune Δstate',
    }).style.format({
        'Päev': _fmt_day,
        'Tegelik ABC':'{:.1f}','BASE':'{:.1f}','Tegelik−BASE':'{:+.1f}',
        'Ühine log-viga': lambda v: '—' if pd.isna(v) else f'{float(v):+.3f}',
        'Varasem center': lambda v: '—' if pd.isna(v) else f'{float(v):+.3f}',
        'Ühine anomaalia': lambda v: '—' if pd.isna(v) else f'{float(v):+.3f}',
        'LAST DAY': lambda v: '—' if pd.isna(v) else f'{float(v):+.3f}',
        'LAST 2': lambda v: '—' if pd.isna(v) else f'{float(v):+.3f}',
        'LAST 5': lambda v: '—' if pd.isna(v) else f'{float(v):+.3f}',
        'ROT-1': lambda v: '—' if pd.isna(v) else f'{float(v):+.3f}',
        'ROT-2': lambda v: '—' if pd.isna(v) else f'{float(v):+.3f}',
        'Eelmine Δstate': lambda v: '—' if pd.isna(v) else f'{float(v):+.3f}',
        'Praegune Δstate': lambda v: '—' if pd.isna(v) else f'{float(v):+.3f}',
    }), use_container_width=True, hide_index=True)

    with st.expander('Kõik strict-OOS päevad'):
        st.dataframe(daily, use_container_width=True, hide_index=True)

    st.caption(
        f'Lagid on lukustatud enne tulemust: 1, 2, {ROTATION_DAYS} ja {TWO_ROTATION_DAYS} harvested days. '
        f'Anomaalia deadband ±{STATE_DEADBAND:.2f} log. Center vajab vähemalt {CENTER_MIN_PRIOR_DAYS} varasemat OOS päeva. '
        'Ühtegi lag’i ega koefitsienti ei valita tulemust vaadates.'
    )
    st.success(
        '🔒 LEAKAGE LOCK: target BASE is trained only on earlier intervals. Every state proxy uses only completed earlier '
        'strict-OOS days. Target actual enters only after prediction to score that target day. READ ONLY; production is unchanged.'
    )


if __name__ == '__main__':
    main()
