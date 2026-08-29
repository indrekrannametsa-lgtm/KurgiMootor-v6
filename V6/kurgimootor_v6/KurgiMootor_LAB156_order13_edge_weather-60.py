from __future__ import annotations

"""
KurgiMootor · edge_weather-41
=============================

SHORT-WINDOW WAVE AUDIT · STRICT WALK-FORWARD · READ ONLY

One question only
-----------------
Could a genuinely short-memory common production curve have predicted the clean
August down/up/down wave before each harvest, and does the same architecture remain
reasonable on the earlier season?

Architecture lock (declared before seeing results)
--------------------------------------------------
- LONG BASE is the existing research field + interval + smooth season reference.
- Stable FIELD identity is estimated causally from ALL prior intervals. This is the
  only long-history component kept in the short models.
- The COMMON time curve is then refit using only the last 8, 10 or 12 completed
  harvest days before each target. No target actual is visible.
- Local curve has only 3 coefficients: level + linear + quadratic time, with field
  effects frozen from prior history. Yield is still integrated over each field's
  actual growth interval, exactly like BASE.
- No single window is selected after the result. SHORT CONSENSUS is the median of
  W8 / W10 / W12 field predictions and is the pre-declared candidate.

Why this avoids the previous circle
-----------------------------------
- no PI, FAST, slow-state, previous-yield anchor or weather correction
- no lag search, coefficient search, threshold search or cap
- only three pre-declared nearby memory lengths: 8 / 10 / 12 harvest days
- the clean wave 19–24.08 is the phenomenon-to-explain, not the tuning set for a
  best K; robustness requires the three windows to tell broadly the same story
- all eligible dates before 19.08 are a backward validation set
- 26.08+ late/ageing tail is shown only as information and is not used to decide
  whether the healthy-August wave architecture works

Pass logic (pre-declared)
-------------------------
A short-window architecture is interesting only if:
1) SHORT CONSENSUS gets >=80% of the 19–24.08 day-to-day directions right,
2) its predicted focus peak is within 1 harvest day of the actual peak,
3) its focus MAE is no worse than LONG BASE,
4) on the earlier validation period its MAE is no more than 10% worse than BASE
   and direction hit is no more than 10 percentage points worse,
5) at least 2 of 3 individual windows get >=60% focus direction hit, and no
   individual window is >25% worse than BASE on earlier validation.

READ ONLY: only db.get_harvest_history is called. Production is untouched.
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
# LATENT DAILY GROWTH / INTERVAL-TOMOGRAPHY AUDIT
# =====================================================================

"""
LAB-42 design note
------------------
This is deliberately NOT another residual carry layer.

Every cucumber harvest interval overlaps several calendar days and several other
fields' harvest intervals.  Before target day T, those already completed intervals
contain information about which *past growth days* were weak or strong.  We use the
overlap to reconstruct a farm-wide daily anomaly curve, then integrate only the
portion of the target field's interval that is already in the past.

Locked architecture:
- LONG BASE remains field + interval + smooth season and is refit strictly before T.
- Each completed interval contributes its strict OOS log residual to every calendar
  growth day covered by that interval.
- Exactly the previous 10 completed harvest dates are used (~2 field rotations).
- The 10-day residual median is removed first: tomography may explain WAVE SHAPE,
  not simply repair BASE's overall bias.
- A latent day is the median centered residual of prior intervals covering that day;
  at least 2 interval observations are required.
- Target-day latent state is NEVER guessed.  The target day receives anomaly 0
  (BASE expectation); only already elapsed days in the target interval can alter it.
- TOMO = BASE * exp(mean(latent anomalies across target interval days)).
- No coefficient, cap, lag, weather, PI, FAST, state smoothing, threshold or window
  search.  If this fixed overlap reconstruction fails, do not tune 8/9/11/12 days.

Why this is genuinely different from failed LABs:
- FAST copied one whole previous harvest-day residual forward.
- SLOW STATE averaged several residual days and lagged regime turns.
- LAB-41 forced the whole recent time curve into one quadratic polynomial.
- TOMO localizes information back onto the overlapping *growth days* that produced
  the completed harvests, then recomposes the next interval from those days.

Decision sets:
- clean healthy-August wave: 19.08–24.08
- backward validation: all eligible strict-OOS days before 19.08
- 26.08+ ageing tail: information only, not part of pass/fail
"""

TOMO_HARVEST_DAYS = 10          # about two farm rotations; locked, no sensitivity search
TOMO_MIN_DAY_SUPPORT = 2        # need at least two completed interval equations per day
WAVE_START = date(2026, 8, 19)
WAVE_END = date(2026, 8, 24)
EARLIER_END = date(2026, 8, 18)
LATE_INFO_START = date(2026, 8, 26)
PRACTICAL_DIR_DEADBAND = 0.05


def _fmt_day(v):
    if isinstance(v, pd.Timestamp):
        v = v.date()
    return v.strftime('%d.%m') if isinstance(v, date) else str(v)


def _build_latent_days(history_rows):
    """Reconstruct already elapsed common growth-day anomalies from prior OOS misses.

    Every item in history_rows was predicted before its own actual was known.  At a
    later target those actuals are, of course, legitimate history.  We center the
    recent residuals first so the reconstruction targets shape rather than a global
    BASE level/bias correction.
    """
    if not history_rows:
        return {}, None, 0, 0

    prior_dates = sorted({r['target_date'] for r in history_rows})
    if len(prior_dates) < TOMO_HARVEST_DAYS:
        return {}, None, 0, len(prior_dates)

    keep = set(prior_dates[-TOMO_HARVEST_DAYS:])
    recent = [r for r in history_rows if r['target_date'] in keep]
    residuals = np.asarray([float(r['strict_log_resid']) for r in recent], dtype=float)
    center = float(np.median(residuals))

    buckets = {}
    for r in recent:
        cr = float(r['strict_log_resid']) - center
        for dd in r['days']:
            buckets.setdefault(dd, []).append(cr)

    states = {}
    for dd, vals in buckets.items():
        if len(vals) >= TOMO_MIN_DAY_SUPPORT:
            states[dd] = float(np.median(np.asarray(vals, dtype=float)))

    return states, center, len(recent), len(prior_dates)


def _target_tomo_anomaly(row, target, states):
    """Integrate only latent states knowable before target actual.

    The target calendar day itself is deliberately neutral (0). Missing historical
    days are also neutral rather than forward-filled. This is conservative and avoids
    turning tomography into another FAST carry rule.
    """
    vals = []
    support = 0
    for dd in row['days']:
        if dd >= target:
            vals.append(0.0)
            continue
        if dd in states:
            vals.append(float(states[dd]))
            support += 1
        else:
            vals.append(0.0)
    if not vals:
        return 0.0, 0, 0
    return float(np.mean(np.asarray(vals, dtype=float))), int(support), int(len(vals))


def _strict_tomography_rows(intervals):
    """Single strict chronological pass: LONG BASE vs overlap-reconstructed TOMO."""
    rows = []
    history_rows = []

    for target in sorted(intervals['target_date'].unique()):
        train = intervals[intervals['target_date'] < target].copy()
        test = intervals[intervals['target_date'] == target].copy()
        if len(train) < MIN_BASE_TRAIN_INTERVALS:
            continue

        counts = train.groupby('field').size().to_dict()
        valid_idx = [
            idx for idx, row in test.iterrows()
            if counts.get(int(row['field']), 0) >= MIN_FIELD_OBS
        ]
        if not valid_idx:
            continue
        test = intervals.loc[valid_idx].copy()

        fit = _fit_base(train)
        base_pred = _predict_base(fit, test)

        states, center, tomo_train_n, prior_day_n = _build_latent_days(history_rows)
        tomo_ready = len(states) > 0 and center is not None

        todays = []
        for j, (_, row) in enumerate(test.iterrows()):
            base = float(base_pred[j])
            if tomo_ready:
                anom, support, interval_days_n = _target_tomo_anomaly(row, target, states)
                tomo = float(base * math.exp(anom))
            else:
                anom, support, interval_days_n = np.nan, 0, int(len(row['days']))
                tomo = np.nan

            rec = {
                'target_date': target,
                'start_date': row['start_date'],
                'field': int(row['field']),
                'order': int(row['order']),
                'growth': float(row['growth']),
                'actual': float(row['actual']),
                'base': base,
                'tomo': tomo,
                'tomo_anomaly': anom,
                'tomo_support_days': int(support),
                'interval_calendar_days': int(interval_days_n),
                'tomo_center': center if center is not None else np.nan,
                'tomo_train_rows': int(tomo_train_n),
                'tomo_prior_harvest_days': int(prior_day_n),
                'base_train_n': int(len(train)),
                'days': list(row['days']),
            }
            rows.append(rec.copy())
            todays.append(rec)

        # IMPORTANT: current actual becomes history only AFTER all target predictions
        # have been formed. Residual is against the strict BASE that existed at target.
        for rec in todays:
            rec_hist = dict(rec)
            rec_hist['strict_log_resid'] = float(
                math.log(rec['actual'] + ABC_EPS) - math.log(rec['base'] + ABC_EPS)
            )
            history_rows.append(rec_hist)

    return pd.DataFrame(rows)


def _daily_rows(strict):
    rows = []
    for target, g in strict.groupby('target_date', sort=True):
        # Keep ordinary 2-field late days valid, but compare only common TOMO-ready days.
        if len(g) < 2:
            continue
        tomo_ready = bool(g['tomo'].notna().all())
        row = {
            'date': target,
            'fields': ','.join(str(int(v)) for v in sorted(g['field'].tolist())),
            'n_fields': int(len(g)),
            'actual': float(g['actual'].sum()),
            'base': float(g['base'].sum()),
            'tomo': float(g['tomo'].sum()) if tomo_ready else np.nan,
            'base_error': float(g['base'].sum() - g['actual'].sum()),
            'tomo_error': float(g['tomo'].sum() - g['actual'].sum()) if tomo_ready else np.nan,
            'tomo_anomaly': float(g['tomo_anomaly'].median()) if tomo_ready else np.nan,
            'support_days_med': float(g['tomo_support_days'].median()) if tomo_ready else np.nan,
            'interval_days_med': float(g['interval_calendar_days'].median()),
            'tomo_center': float(g['tomo_center'].median()) if tomo_ready else np.nan,
            'tomo_train_rows': int(g['tomo_train_rows'].min()) if tomo_ready else 0,
            'base_train_n': int(g['base_train_n'].min()),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values('date').reset_index(drop=True)


def _direction_stats(df, pred_col, deadband=PRACTICAL_DIR_DEADBAND):
    x = df[['date','actual','n_fields',pred_col]].dropna().sort_values('date').copy()
    if len(x) < 2:
        return np.nan, 0
    a = x['actual'].to_numpy(dtype=float) / np.maximum(x['n_fields'].to_numpy(dtype=float), 1.0)
    p = x[pred_col].to_numpy(dtype=float) / np.maximum(x['n_fields'].to_numpy(dtype=float), 1.0)
    hits = []
    for i in range(1, len(x)):
        da = (a[i] - a[i-1]) / max(abs(a[i-1]), 1e-9)
        dp = (p[i] - p[i-1]) / max(abs(p[i-1]), 1e-9)
        if abs(da) < deadband:
            continue
        hits.append(int(np.sign(da) == np.sign(dp)))
    return (100.0 * float(np.mean(hits)), len(hits)) if hits else (np.nan, 0)


def _wave_direction_stats(df, pred_col):
    x = df[['date','actual','n_fields',pred_col]].dropna().sort_values('date').copy()
    if len(x) < 2:
        return np.nan, 0
    a = x['actual'].to_numpy(dtype=float) / np.maximum(x['n_fields'].to_numpy(dtype=float), 1.0)
    p = x[pred_col].to_numpy(dtype=float) / np.maximum(x['n_fields'].to_numpy(dtype=float), 1.0)
    hits = []
    for i in range(1, len(x)):
        da = a[i] - a[i-1]
        dp = p[i] - p[i-1]
        if abs(da) < 1e-12:
            continue
        hits.append(int(np.sign(da) == np.sign(dp)))
    return (100.0 * float(np.mean(hits)), len(hits)) if hits else (np.nan, 0)


def _metrics(df, pred_col):
    x = df[['date','actual','n_fields',pred_col]].dropna().copy()
    if x.empty:
        return {
            'N':0,'MAE':np.nan,'MAE/põld':np.nan,'MAPE %':np.nan,'Median AE':np.nan,
            'Bias':np.nan,'±10%':np.nan,'±20%':np.nan,'Worst AE':np.nan,
            'Direction hit %':np.nan,'Direction N':0,
        }
    actual = x['actual'].to_numpy(dtype=float)
    pred = x[pred_col].to_numpy(dtype=float)
    ae = np.abs(pred - actual)
    ape = ae / np.maximum(np.abs(actual), 1e-9)
    fields = np.maximum(x['n_fields'].to_numpy(dtype=float), 1.0)
    dh, dn = _direction_stats(x, pred_col)
    return {
        'N':int(len(x)),
        'MAE':float(np.mean(ae)),
        'MAE/põld':float(np.mean(ae / fields)),
        'MAPE %':100.0 * float(np.mean(ape)),
        'Median AE':float(np.median(ae)),
        'Bias':float(np.mean(pred - actual)),
        '±10%':100.0 * float(np.mean(ape <= 0.10)),
        '±20%':100.0 * float(np.mean(ape <= 0.20)),
        'Worst AE':float(np.max(ae)),
        'Direction hit %':dh,
        'Direction N':int(dn),
    }


def _peak_info(df, pred_col):
    x = df[['date','actual','n_fields',pred_col]].dropna().sort_values('date').copy()
    if x.empty:
        return None, None, None
    actual_pf = x['actual'].to_numpy(dtype=float) / np.maximum(x['n_fields'].to_numpy(dtype=float), 1.0)
    pred_pf = x[pred_col].to_numpy(dtype=float) / np.maximum(x['n_fields'].to_numpy(dtype=float), 1.0)
    ai = int(np.argmax(actual_pf))
    pi = int(np.argmax(pred_pf))
    return x.iloc[ai]['date'], x.iloc[pi]['date'], int(pi - ai)


def _variant_table(df, focus=False):
    rows = []
    base = _metrics(df, 'base')
    for label, col in [('A · LONG BASE','base'),('B · TOMO overlap','tomo')]:
        m = _metrics(df, col)
        imp = (
            100.0 * (base['MAE'] - m['MAE']) / base['MAE']
            if col != 'base' and pd.notna(base['MAE']) and base['MAE'] > 1e-12 and pd.notna(m['MAE'])
            else 0.0
        )
        row = {
            'Variant':label,
            'N':m['N'],'MAE':m['MAE'],'MAE/põld':m['MAE/põld'],'MAPE %':m['MAPE %'],
            'Median AE':m['Median AE'],'Bias':m['Bias'],'±10%':m['±10%'],'±20%':m['±20%'],
            'Worst AE':m['Worst AE'],'Direction hit %':m['Direction hit %'],'Direction N':m['Direction N'],
            'Parandus vs BASE %':imp,
        }
        if focus:
            wh, wn = _wave_direction_stats(df, col)
            ap, pp, lag = _peak_info(df, col)
            row.update({
                'Wave suunahitt %':wh,'Wave suund N':wn,
                'Tegelik tipp':ap,'Ennustatud tipp':pp,'Tipu nihe korjepäeva':lag,
            })
        rows.append(row)
    return pd.DataFrame(rows)


def _half_table(earlier):
    x = earlier.dropna(subset=['tomo']).sort_values('date').copy()
    if len(x) < 6:
        return pd.DataFrame()
    cut = len(x) // 2
    parts = [('I pool', x.iloc[:cut].copy()), ('II pool', x.iloc[cut:].copy())]
    rows = []
    for period, part in parts:
        b = _metrics(part, 'base')
        for label, col in [('BASE','base'),('TOMO','tomo')]:
            m = _metrics(part, col)
            imp = 0.0 if col == 'base' else (
                100.0 * (b['MAE'] - m['MAE']) / b['MAE'] if b['MAE'] > 1e-12 else np.nan
            )
            rows.append({
                'Periood':period,'Variant':label,'N':m['N'],'MAE':m['MAE'],'MAPE %':m['MAPE %'],
                '±20%':m['±20%'],'Direction hit %':m['Direction hit %'],'Bias':m['Bias'],
                'Parandus vs BASE %':imp,
            })
    return pd.DataFrame(rows)


def _verdict(earlier, focus):
    focus = focus.dropna(subset=['tomo']).copy()
    earlier = earlier.dropna(subset=['tomo']).copy()
    if len(focus) < 5:
        return 'INSUFFICIENT FOCUS', 'Too few TOMO-ready days in the clean 19–24.08 wave.'
    if len(earlier) < 5:
        return 'INSUFFICIENT BACKTEST', 'Too few TOMO-ready days before 19.08 for an anti-overfit check.'

    fb = _metrics(focus, 'base')
    ft = _metrics(focus, 'tomo')
    wh, wn = _wave_direction_stats(focus, 'tomo')
    _, _, peak_lag = _peak_info(focus, 'tomo')

    eb = _metrics(earlier, 'base')
    et = _metrics(earlier, 'tomo')
    half = _half_table(earlier)
    tomo_halves = half[half['Variant'] == 'TOMO'] if not half.empty else pd.DataFrame()
    stable_halves = (
        len(tomo_halves) == 2
        and bool((tomo_halves['Parandus vs BASE %'] >= -10.0).all())
    )

    focus_pass = (
        pd.notna(wh) and wn >= 4 and wh >= 80.0
        and pd.notna(peak_lag) and abs(int(peak_lag)) <= 1
        and ft['MAE'] <= fb['MAE']
    )
    earlier_pass = (
        et['MAE'] <= 1.10 * eb['MAE']
        and (
            pd.isna(eb['Direction hit %']) or pd.isna(et['Direction hit %'])
            or et['Direction hit %'] >= eb['Direction hit %'] - 10.0
        )
        and stable_halves
    )

    if focus_pass and earlier_pass:
        return 'STRUCTURAL PASS · OVERLAP MODEL SEES THE WAVE', (
            'The fixed interval-overlap reconstruction reproduces the healthy-August wave without becoming materially worse on earlier data. '
            'That would be the first evidence that harvest intervals contain a recoverable latent daily production curve. Do not tune it yet; inspect the daily reconstruction first.'
        )
    if focus_pass and not earlier_pass:
        return 'FOCUS PASS ONLY · NOT GENERAL ENOUGH', (
            'TOMO sees the August wave but the same frozen overlap rule degrades earlier data too much. Do not change the 10-day horizon or support rule to rescue it.'
        )
    return 'STRUCTURAL FAIL · WAVE NOT RECOVERABLE FROM OVERLAP ALONE', (
        'The fixed overlap reconstruction does not predict the clean August wave robustly. Do not tune the horizon/support/centering. '
        'If this fails, the next useful information source would have to be genuinely exogenous/biological rather than another rearrangement of recent harvest residuals.'
    )


def _style(df, focus=False):
    fmt = {
        'MAE':lambda v:'—' if pd.isna(v) else f'{float(v):.2f}',
        'MAE/põld':lambda v:'—' if pd.isna(v) else f'{float(v):.2f}',
        'MAPE %':lambda v:'—' if pd.isna(v) else f'{float(v):.1f}',
        'Median AE':lambda v:'—' if pd.isna(v) else f'{float(v):.2f}',
        'Bias':lambda v:'—' if pd.isna(v) else f'{float(v):+.2f}',
        '±10%':lambda v:'—' if pd.isna(v) else f'{float(v):.0f}%',
        '±20%':lambda v:'—' if pd.isna(v) else f'{float(v):.0f}%',
        'Worst AE':lambda v:'—' if pd.isna(v) else f'{float(v):.2f}',
        'Direction hit %':lambda v:'—' if pd.isna(v) else f'{float(v):.0f}%',
        'Parandus vs BASE %':lambda v:'—' if pd.isna(v) else f'{float(v):+.1f}%',
    }
    if focus:
        fmt.update({
            'Wave suunahitt %':lambda v:'—' if pd.isna(v) else f'{float(v):.0f}%',
            'Tegelik tipp':_fmt_day,'Ennustatud tipp':_fmt_day,
            'Tipu nihe korjepäeva':lambda v:'—' if pd.isna(v) else f'{int(v):+d}',
        })
    return df.style.format(fmt)


def main():
    st.set_page_config(page_title='KurgiMootor · latent daily growth', layout='wide')
    st.title('KurgiMootor · latent daily growth / interval tomography')
    st.caption('LAB-42 · one structural model · strict walk-forward · READ ONLY')
    st.info(
        'Reset after the correction-layer dead end. Each completed harvest is an integral over several growth days. '
        'TOMO uses the overlap of recent field intervals to reconstruct which already elapsed calendar days were commonly weak/strong, '
        'then recomposes the next field interval. Exactly 10 prior harvest dates (~2 rotations), no parameter search, no weather, PI, FAST or slow state.'
    )

    try:
        harvest = db.get_harvest_history(limit=5000)
        events = _events(harvest)
        intervals = _build_intervals(events)
        strict = _strict_tomography_rows(intervals)
        if strict.empty:
            raise RuntimeError('TOMO strict-OOS rows did not form.')
        daily = _daily_rows(strict)
        if daily.empty:
            raise RuntimeError('No complete strict-OOS daily rows formed.')
    except Exception as exc:
        st.exception(exc)
        st.stop()

    common = daily.dropna(subset=['tomo']).copy()
    earlier = common[common['date'] <= EARLIER_END].copy()
    focus = common[(common['date'] >= WAVE_START) & (common['date'] <= WAVE_END)].copy()
    late = common[common['date'] >= LATE_INFO_START].copy()
    verdict, text = _verdict(earlier, focus)

    st.markdown('### 1. Otsus · kas kattuvad korjeintervallid sisaldasid augustilainet enne korjet?')
    if verdict.startswith('STRUCTURAL PASS'):
        st.success('✅ ' + verdict + ': ' + text)
    elif verdict.startswith('FOCUS PASS'):
        st.warning('🟡 ' + verdict + ': ' + text)
    else:
        st.error('⛔ ' + verdict + ': ' + text)

    st.caption(
        'Locked pass: 19–24.08 wave direction ≥80%, peak within ±1 harvest day and focus MAE ≤ BASE; earlier MAE may be at most 10% worse, '
        'direction at most 10 pp worse, and neither earlier half may be >10% worse. No rescue by changing the 10-day horizon, support=2 or centering.'
    )

    st.markdown('### 2. Puhas terve taime laine · 19.–24.08')
    st.dataframe(_style(_variant_table(focus, focus=True), focus=True), use_container_width=True, hide_index=True)
    fview = focus[[
        'date','fields','n_fields','actual','base','tomo','base_error','tomo_error',
        'tomo_anomaly','support_days_med','interval_days_med','tomo_train_rows'
    ]].rename(columns={
        'date':'Päev','fields':'Põllud','n_fields':'N põldu','actual':'Tegelik ABC',
        'base':'LONG BASE','tomo':'TOMO','base_error':'BASE viga','tomo_error':'TOMO viga',
        'tomo_anomaly':'Latent anomaalia','support_days_med':'Toega intervalipäevi',
        'interval_days_med':'Intervalipäevi','tomo_train_rows':'TOMO treeningridu',
    })
    st.dataframe(fview.style.format({
        'Päev':_fmt_day,'Tegelik ABC':'{:.1f}','LONG BASE':'{:.1f}','TOMO':'{:.1f}',
        'BASE viga':'{:+.1f}','TOMO viga':'{:+.1f}','Latent anomaalia':'{:+.3f}',
        'Toega intervalipäevi':'{:.1f}','Intervalipäevi':'{:.1f}',
    }), use_container_width=True, hide_index=True)

    st.markdown('### 3. Varasem kontroll · sama konstruktsioon enne 19.08')
    st.dataframe(_style(_variant_table(earlier)), use_container_width=True, hide_index=True)
    half = _half_table(earlier)
    if not half.empty:
        st.markdown('#### Varasema kontrolli esimene pool vs teine pool')
        st.dataframe(_style(half), use_container_width=True, hide_index=True)

    st.markdown('### 4. Mida TOMO tegelikult target-intervalli kohta teadis?')
    st.caption(
        'Latent anomaalia on ainult nende target-intervalli päevade ühine signaal, mida sai varasemate lõpetatud korjete kattest taastada. '
        'Target-päev ise on alati 0 ehk BASE; seda ei forward-fillita.'
    )
    diag = common[['date','fields','tomo_anomaly','support_days_med','interval_days_med','tomo_center','tomo_train_rows']].tail(20).rename(columns={
        'date':'Päev','fields':'Põllud','tomo_anomaly':'Latent anomaalia','support_days_med':'Toega päevi',
        'interval_days_med':'Intervalli päevi','tomo_center':'10 päeva residual-center','tomo_train_rows':'Katte ridu',
    })
    st.dataframe(diag.style.format({
        'Päev':_fmt_day,'Latent anomaalia':'{:+.3f}','Toega päevi':'{:.1f}','Intervalli päevi':'{:.1f}',
        '10 päeva residual-center':'{:+.3f}',
    }), use_container_width=True, hide_index=True)

    if not late.empty:
        st.markdown('### 5. 26.08+ vananev taimestik · ainult informatsioon')
        lview = late[['date','fields','actual','base','tomo','tomo_anomaly']].rename(columns={
            'date':'Päev','fields':'Põllud','actual':'Tegelik ABC','base':'LONG BASE','tomo':'TOMO','tomo_anomaly':'Latent anomaalia'
        })
        st.dataframe(lview.style.format({
            'Päev':_fmt_day,'Tegelik ABC':'{:.1f}','LONG BASE':'{:.1f}','TOMO':'{:.1f}','Latent anomaalia':'{:+.3f}'
        }), use_container_width=True, hide_index=True)

    with st.expander('Kõik strict-OOS päevad'):
        st.dataframe(daily, use_container_width=True, hide_index=True)

    st.success(
        '🔒 LEAKAGE / ARCHITECTURE LOCK: target BASE is trained only on earlier intervals. TOMO uses only strict OOS residuals from completed prior harvest dates. '
        'Exactly the previous 10 harvest dates are used; recent residual median is removed; a latent day needs ≥2 prior covering intervals; target day is neutral 0. '
        'No weather, PI, FAST, slow-state, coefficient/cap/lag/window search. READ ONLY; production unchanged.'
    )


if __name__ == '__main__':
    main()
