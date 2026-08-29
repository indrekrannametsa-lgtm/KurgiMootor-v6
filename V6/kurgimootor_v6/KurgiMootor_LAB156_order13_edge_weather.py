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
# SHORT-WINDOW WAVE AUDIT
# =====================================================================

LOCAL_WINDOWS = (8, 10, 12)
LOCAL_MAX_ITER = 220
LOCAL_TREND_RIDGE = BASE_SEASON_RIDGE
WAVE_START = date(2026, 8, 19)
WAVE_END = date(2026, 8, 24)
EARLIER_END = date(2026, 8, 18)
LATE_INFO_START = date(2026, 8, 26)
PRACTICAL_DIR_DEADBAND = 0.05


def _fmt_day(v):
    if isinstance(v, pd.Timestamp):
        v = v.date()
    return v.strftime('%d.%m') if isinstance(v, date) else str(v)


def _local_vec(dd: date, target: date):
    # Same 30-day scale as LONG BASE, but centered on the target. Centering is
    # numerical only; it does not expose target actual or future observations.
    u = float((dd - target).days) / 30.0
    return np.asarray([1.0, u, u * u], dtype=float)


def _global_to_local_beta(global_beta, target: date):
    """Express LONG BASE's polynomial around target day for stable initialization."""
    s0 = float((target - WEATHER_START).days) / 30.0
    b0, b1, b2 = [float(v) for v in global_beta]
    return np.asarray([
        b0 + b1 * s0 + b2 * s0 * s0,
        b1 + 2.0 * b2 * s0,
        b2,
    ], dtype=float)


def _field_factor(gammas, field: int):
    if int(field) == 1:
        return 1.0
    return math.exp(float(gammas[int(field) - 2]))


def _fit_local_beta(train, gammas, target: date, init_beta):
    """Fit only the 3 common time coefficients on the short causal window.

    Field effects are frozen from the all-prior causal fit. The objective and
    interval integration match the research BASE log-error formulation.
    """
    beta = np.asarray(init_beta, dtype=float).copy()
    y = train['actual'].to_numpy(dtype=float)

    X_list = []
    w_list = []
    ff_list = []
    for _, row in train.iterrows():
        X_list.append(np.vstack([_local_vec(dd, target) for dd in row['days']]))
        w_list.append(float(row['per_day_weight']))
        ff_list.append(_field_factor(gammas, int(row['field'])))

    lr = 0.030
    b1 = 0.9
    b2 = 0.999
    adam_eps = 1e-8
    mb = np.zeros_like(beta)
    vb = np.zeros_like(beta)
    prev_obj = None

    for step in range(1, LOCAL_MAX_ITER + 1):
        gb = np.zeros_like(beta)
        obj_data = 0.0

        for i, (X, weight, ff) in enumerate(zip(X_list, w_list, ff_list)):
            eta = X @ beta
            prod = np.exp(np.clip(eta, -6.0, 6.0))
            common = float(weight * np.sum(prod))
            pred = max(float(ff * common), 1e-8)
            resid = math.log(pred + ABC_EPS) - math.log(float(y[i]) + ABC_EPS)
            obj_data += resid * resid
            shrink = pred / (pred + ABC_EPS)
            denom = max(float(np.sum(prod)), 1e-12)
            x_bar = ((prod[:, None] * X).sum(axis=0) / denom)
            gb += 2.0 * resid * shrink * x_bar

        reg = LOCAL_TREND_RIDGE * float(np.sum(beta[1:] * beta[1:]))
        gb[1:] += 2.0 * LOCAL_TREND_RIDGE * beta[1:]
        obj = obj_data + reg
        gb /= max(len(train), 1)

        norm = float(np.linalg.norm(gb))
        if norm > 10.0:
            gb *= 10.0 / norm

        mb = b1 * mb + (1.0 - b1) * gb
        vb = b2 * vb + (1.0 - b2) * (gb * gb)
        mbh = mb / (1.0 - b1 ** step)
        vbh = vb / (1.0 - b2 ** step)
        beta -= lr * mbh / (np.sqrt(vbh) + adam_eps)

        if (
            prev_obj is not None
            and step > 60
            and abs(prev_obj - obj) < 1e-7 * max(1.0, abs(prev_obj))
        ):
            break
        prev_obj = obj

    return beta


def _predict_local(test, beta, gammas, target: date):
    preds = []
    for _, row in test.iterrows():
        X = np.vstack([_local_vec(dd, target) for dd in row['days']])
        prod = np.exp(np.clip(X @ beta, -6.0, 6.0))
        common = float(row['per_day_weight'] * np.sum(prod))
        preds.append(_field_factor(gammas, int(row['field'])) * common)
    return np.asarray(preds, dtype=float)


def _strict_wave_rows(intervals):
    """One strict walk-forward pass producing LONG BASE + W8/W10/W12.

    For each target:
    - all-history fit is trained strictly before target and supplies LONG BASE and
      frozen field effects;
    - each local common curve sees only its last K completed harvest dates;
    - consensus is a field-level median of W8/W10/W12, never best-K selection.
    """
    rows = []
    targets = sorted(intervals['target_date'].unique())

    for target in targets:
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

        prior_days = sorted(train['target_date'].unique())
        if len(prior_days) < max(LOCAL_WINDOWS):
            continue

        global_fit = _fit_base(train)
        base_pred = _predict_base(global_fit, test)
        local_preds = {}
        local_ns = {}

        for k in LOCAL_WINDOWS:
            selected_days = set(prior_days[-k:])
            local_train = train[train['target_date'].isin(selected_days)].copy()
            init_beta = _global_to_local_beta(global_fit['beta'], target)
            beta = _fit_local_beta(
                local_train,
                global_fit['gammas'],
                target,
                init_beta,
            )
            local_preds[k] = _predict_local(
                test, beta, global_fit['gammas'], target
            )
            local_ns[k] = int(len(local_train))

        for j, (_, row) in enumerate(test.iterrows()):
            vals = [float(local_preds[k][j]) for k in LOCAL_WINDOWS]
            rows.append({
                'target_date': target,
                'start_date': row['start_date'],
                'field': int(row['field']),
                'order': int(row['order']),
                'growth': float(row['growth']),
                'actual': float(row['actual']),
                'base': float(base_pred[j]),
                'w8': vals[0],
                'w10': vals[1],
                'w12': vals[2],
                'short': float(np.median(np.asarray(vals, dtype=float))),
                'base_train_n': int(len(train)),
                'w8_train_n': local_ns[8],
                'w10_train_n': local_ns[10],
                'w12_train_n': local_ns[12],
            })

    return pd.DataFrame(rows)


def _daily_rows(strict):
    rows = []
    for target, g in strict.groupby('target_date', sort=True):
        if len(g) < 2:
            continue
        row = {
            'date': target,
            'fields': ','.join(str(int(v)) for v in sorted(g['field'].tolist())),
            'n_fields': int(len(g)),
            'actual': float(g['actual'].sum()),
            'base': float(g['base'].sum()),
            'w8': float(g['w8'].sum()),
            'w10': float(g['w10'].sum()),
            'w12': float(g['w12'].sum()),
            # consensus is median per field, then summed; strict already stores that.
            'short': float(g['short'].sum()),
            'base_train_n': int(g['base_train_n'].min()),
            'w8_train_n': int(g['w8_train_n'].min()),
            'w10_train_n': int(g['w10_train_n'].min()),
            'w12_train_n': int(g['w12_train_n'].min()),
        }
        for col in ('base','w8','w10','w12','short'):
            row[f'{col}_error'] = float(row[col] - row['actual'])
        rows.append(row)
    return pd.DataFrame(rows).sort_values('date').reset_index(drop=True)


def _direction_stats(df, pred_col):
    x = df[['date','actual','n_fields',pred_col]].dropna().sort_values('date').copy()
    if len(x) < 2:
        return np.nan, 0
    a = x['actual'].to_numpy(dtype=float) / np.maximum(x['n_fields'].to_numpy(dtype=float), 1.0)
    p = x[pred_col].to_numpy(dtype=float) / np.maximum(x['n_fields'].to_numpy(dtype=float), 1.0)
    hits = []
    for i in range(1, len(x)):
        da = (a[i] - a[i-1]) / max(abs(a[i-1]), 1e-9)
        dp = (p[i] - p[i-1]) / max(abs(p[i-1]), 1e-9)
        if abs(da) < PRACTICAL_DIR_DEADBAND:
            continue
        hits.append(int(np.sign(da) == np.sign(dp)))
    return (100.0 * float(np.mean(hits)), len(hits)) if hits else (np.nan, 0)


def _wave_direction_stats(df, pred_col):
    """Direction score for the pre-declared clean wave: score every adjacent harvest day.

    Unlike the general practical direction metric, the clean 19–24.08 wave uses no
    deadband: the question is whether the model reproduced the observed down/up/down
    shape, including the smaller 20->21 step.
    """
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
            'N':0,'MAE':np.nan,'MAE/põld':np.nan,'MAPE %':np.nan,
            'Bias':np.nan,'±20%':np.nan,'Direction hit %':np.nan,'Direction N':0,
        }
    a = x['actual'].to_numpy(dtype=float)
    p = x[pred_col].to_numpy(dtype=float)
    err = p - a
    ae = np.abs(err)
    d_hit, d_n = _direction_stats(x, pred_col)
    return {
        'N':int(len(x)),
        'MAE':float(np.mean(ae)),
        'MAE/põld':float(np.mean(ae / np.maximum(x['n_fields'].to_numpy(dtype=float),1.0))),
        'MAPE %':100.0 * float(np.mean(ae / np.maximum(np.abs(a),1e-9))),
        'Bias':float(np.mean(err)),
        '±20%':100.0 * float(np.mean(ae / np.maximum(np.abs(a),1e-9) <= 0.20)),
        'Direction hit %':d_hit,
        'Direction N':d_n,
    }


def _peak_info(df, pred_col):
    x = df[['date','n_fields','actual',pred_col]].dropna().sort_values('date').copy()
    if x.empty:
        return None, None, np.nan
    actual_pf = x['actual'].to_numpy(dtype=float) / np.maximum(x['n_fields'].to_numpy(dtype=float),1.0)
    pred_pf = x[pred_col].to_numpy(dtype=float) / np.maximum(x['n_fields'].to_numpy(dtype=float),1.0)
    ai = int(np.argmax(actual_pf))
    pi = int(np.argmax(pred_pf))
    actual_peak = x.iloc[ai]['date']
    pred_peak = x.iloc[pi]['date']
    # Harvest-day index lag is more meaningful than calendar lag if a no-harvest day exists.
    lag = int(pi - ai)
    return actual_peak, pred_peak, lag


def _variant_table(df, include_peak=False):
    rows = []
    labels = [
        ('LONG BASE','base'),
        ('SHORT W8','w8'),
        ('SHORT W10','w10'),
        ('SHORT W12','w12'),
        ('SHORT CONSENSUS','short'),
    ]
    base_m = _metrics(df, 'base')
    for label, col in labels:
        m = _metrics(df, col)
        imp = 100.0 * (base_m['MAE'] - m['MAE']) / base_m['MAE'] if base_m['MAE'] > 1e-12 else np.nan
        row = {
            'Variant':label,
            **m,
            'Parandus vs BASE %':0.0 if col == 'base' else imp,
        }
        if include_peak:
            wave_hit, wave_n = _wave_direction_stats(df, col)
            row['Wave suunahitt %'] = wave_hit
            row['Wave suund N'] = wave_n
            apeak, ppeak, lag = _peak_info(df, col)
            row['Tegelik tipp'] = apeak
            row['Ennustatud tipp'] = ppeak
            row['Tipu nihe korjepäeva'] = lag
        rows.append(row)
    return pd.DataFrame(rows)


def _earlier_half_table(earlier):
    x = earlier.sort_values('date').reset_index(drop=True)
    if len(x) < 4:
        return pd.DataFrame()
    cut = len(x) // 2
    rows = []
    for label, part in [('I pool', x.iloc[:cut]), ('II pool', x.iloc[cut:])]:
        for variant, col in [('LONG BASE','base'),('SHORT CONSENSUS','short')]:
            m = _metrics(part, col)
            base = _metrics(part, 'base')
            imp = 100.0 * (base['MAE'] - m['MAE']) / base['MAE'] if base['MAE'] > 1e-12 else np.nan
            rows.append({
                'Periood':label,'Variant':variant,'N':m['N'],'MAE':m['MAE'],
                'MAPE %':m['MAPE %'],'Direction hit %':m['Direction hit %'],
                '±20%':m['±20%'],'Bias':m['Bias'],
                'Parandus vs BASE %':0.0 if col == 'base' else imp,
            })
    return pd.DataFrame(rows)


def _verdict(earlier, focus):
    if len(focus) < 5:
        return 'INSUFFICIENT FOCUS', '19–24.08 common focus has too few complete strict-OOS days.'
    if len(earlier) < 5:
        return 'INSUFFICIENT EARLIER VALIDATION', 'Too few earlier strict-OOS days to protect against a focus-only fit.'

    fm_base = _metrics(focus, 'base')
    fm_short = _metrics(focus, 'short')
    focus_wave_hit, focus_wave_n = _wave_direction_stats(focus, 'short')
    _, _, peak_lag = _peak_info(focus, 'short')

    indiv_hits = []
    for col in ('w8','w10','w12'):
        indiv_hits.append(_wave_direction_stats(focus, col)[0])
    two_windows_ok = sum(pd.notna(v) and v >= 60.0 for v in indiv_hits) >= 2

    em_base = _metrics(earlier, 'base')
    em_short = _metrics(earlier, 'short')
    earlier_mae_ok = em_short['MAE'] <= 1.10 * em_base['MAE']
    earlier_dir_ok = (
        pd.isna(em_base['Direction hit %']) or pd.isna(em_short['Direction hit %'])
        or em_short['Direction hit %'] >= em_base['Direction hit %'] - 10.0
    )
    no_window_catastrophe = all(
        _metrics(earlier, col)['MAE'] <= 1.25 * em_base['MAE']
        for col in ('w8','w10','w12')
    )

    focus_pass = (
        pd.notna(focus_wave_hit)
        and focus_wave_n >= 4
        and focus_wave_hit >= 80.0
        and pd.notna(peak_lag) and abs(int(peak_lag)) <= 1
        and fm_short['MAE'] <= fm_base['MAE']
        and two_windows_ok
    )
    earlier_pass = earlier_mae_ok and earlier_dir_ok and no_window_catastrophe

    if focus_pass and earlier_pass:
        return 'ROBUST SHORT-WINDOW CANDIDATE', (
            'The pre-declared W8/W10/W12 consensus predicts the clean 19–24.08 wave/peak and remains reasonable on earlier data. '
            'This supports the idea that long-history time dynamics were washing out a real short common production wave. '
            'Do not choose a best K; keep the consensus architecture for the next independent test.'
        )
    if focus_pass and not earlier_pass:
        return 'FOCUS ONLY · OVERFIT RISK', (
            'The short model reproduces the August wave, but the same frozen architecture degrades earlier validation too much. '
            'Do not tune the windows or promote it.'
        )
    return 'WAVE NOT PREDICTED ROBUSTLY', (
        'The pre-declared short-window family does not reproduce the clean August wave strongly enough under strict walk-forward. '
        'Do not search more window lengths around 8/10/12; that would start another tuning loop.'
    )


def _format_table(df, peak=False):
    fmt = {
        'MAE':lambda v:'—' if pd.isna(v) else f'{float(v):.2f}',
        'MAE/põld':lambda v:'—' if pd.isna(v) else f'{float(v):.2f}',
        'MAPE %':lambda v:'—' if pd.isna(v) else f'{float(v):.1f}',
        'Bias':lambda v:'—' if pd.isna(v) else f'{float(v):+.2f}',
        '±20%':lambda v:'—' if pd.isna(v) else f'{float(v):.0f}%',
        'Direction hit %':lambda v:'—' if pd.isna(v) else f'{float(v):.0f}%',
        'Parandus vs BASE %':lambda v:'—' if pd.isna(v) else f'{float(v):+.1f}%',
    }
    if peak:
        fmt.update({
            'Wave suunahitt %':lambda v:'—' if pd.isna(v) else f'{float(v):.0f}%',
            'Tegelik tipp':_fmt_day,
            'Ennustatud tipp':_fmt_day,
            'Tipu nihe korjepäeva':lambda v:'—' if pd.isna(v) else f'{int(v):+d}',
        })
    return df.style.format(fmt)


def main():
    st.set_page_config(page_title='KurgiMootor · short-window wave', layout='wide')
    st.title('KurgiMootor · short-window wave audit')
    st.caption('LAB-41 · 8 / 10 / 12 harvest-day common memory · strict walk-forward · READ ONLY')
    st.info(
        'One question only: could a short-memory common production curve have seen the healthy-August down/up/down wave before harvest? '
        'Field identity still learns from all prior data; only the common time curve is short. W8/W10/W12 are a sensitivity family, not a contest. '
        'The candidate is their pre-declared median consensus. No weather, PI, FAST, state correction or best-window selection.'
    )

    try:
        harvest = db.get_harvest_history(limit=5000)
        events = _events(harvest)
        intervals = _build_intervals(events)
        strict = _strict_wave_rows(intervals)
        if strict.empty:
            raise RuntimeError('Short-window strict-OOS rows did not form.')
        daily = _daily_rows(strict)
        if daily.empty:
            raise RuntimeError('No complete strict-OOS daily rows formed.')
    except Exception as exc:
        st.exception(exc)
        st.stop()

    earlier = daily[daily['date'] <= EARLIER_END].copy()
    focus = daily[(daily['date'] >= WAVE_START) & (daily['date'] <= WAVE_END)].copy()
    late = daily[daily['date'] >= LATE_INFO_START].copy()
    verdict, text = _verdict(earlier, focus)

    st.markdown('### 1. Otsus · kas lühike mälu nägi puhast augustilainet?')
    if verdict == 'ROBUST SHORT-WINDOW CANDIDATE':
        st.success('✅ ' + verdict + ': ' + text)
    elif verdict.startswith('FOCUS ONLY'):
        st.warning('🟡 ' + verdict + ': ' + text)
    else:
        st.error('⛔ ' + verdict + ': ' + text)

    st.caption(
        'Pass/fail was locked before results: focus wave-direction ≥80% across adjacent 19–24 harvest days (no deadband), peak within ±1 harvest day, focus MAE no worse than BASE; '
        'earlier validation may lose at most 10% MAE / 10 pp direction; at least 2 of 3 windows must agree and no window may be >25% worse earlier.'
    )

    st.markdown('### 2. Puhas laine · 19.–24.08')
    focus_table = _variant_table(focus, include_peak=True)
    st.dataframe(_format_table(focus_table, peak=True), use_container_width=True, hide_index=True)

    fview = focus[[
        'date','fields','n_fields','actual','base','w8','w10','w12','short',
        'base_error','short_error'
    ]].rename(columns={
        'date':'Päev','fields':'Põllud','n_fields':'N põldu','actual':'Tegelik ABC',
        'base':'LONG BASE','w8':'W8','w10':'W10','w12':'W12','short':'SHORT consensus',
        'base_error':'BASE viga','short_error':'SHORT viga',
    })
    st.dataframe(fview.style.format({
        'Päev':_fmt_day,'Tegelik ABC':'{:.1f}','LONG BASE':'{:.1f}','W8':'{:.1f}',
        'W10':'{:.1f}','W12':'{:.1f}','SHORT consensus':'{:.1f}',
        'BASE viga':'{:+.1f}','SHORT viga':'{:+.1f}',
    }), use_container_width=True, hide_index=True)

    st.markdown('### 3. Varasem kontroll · sama arhitektuur enne 19.08')
    st.caption('This is the anti-overfit check. No window is reselected or retuned for this period.')
    earlier_table = _variant_table(earlier, include_peak=False)
    st.dataframe(_format_table(earlier_table), use_container_width=True, hide_index=True)

    half = _earlier_half_table(earlier)
    if not half.empty:
        st.markdown('#### Varasema kontrolli esimene pool vs teine pool')
        st.dataframe(_format_table(half), use_container_width=True, hide_index=True)

    st.markdown('### 4. 8 / 10 / 12 akna tundlikkus · ära vali “võitjat”')
    sens = []
    base_earlier = _metrics(earlier, 'base')
    for k, col in [(8,'w8'),(10,'w10'),(12,'w12')]:
        fm = _metrics(focus, col)
        em = _metrics(earlier, col)
        wave_hit, wave_n = _wave_direction_stats(focus, col)
        _, ppeak, plag = _peak_info(focus, col)
        sens.append({
            'Aken':f'{k} korjepäeva',
            'Focus suunahitt %':wave_hit,
            'Focus suund N':wave_n,
            'Focus MAE':fm['MAE'],
            'Focus ennustatud tipp':ppeak,
            'Tipu nihe':plag,
            'Varasem MAE':em['MAE'],
            'Varasem vs BASE %':(
                100.0 * (base_earlier['MAE'] - em['MAE']) / base_earlier['MAE']
                if base_earlier['MAE'] > 1e-12 else np.nan
            ),
            'Varasem suunahitt %':em['Direction hit %'],
        })
    sens = pd.DataFrame(sens)
    st.dataframe(sens.style.format({
        'Focus suunahitt %':lambda v:'—' if pd.isna(v) else f'{float(v):.0f}%',
        'Focus MAE':lambda v:'—' if pd.isna(v) else f'{float(v):.2f}',
        'Focus ennustatud tipp':_fmt_day,
        'Tipu nihe':lambda v:'—' if pd.isna(v) else f'{int(v):+d}',
        'Varasem MAE':lambda v:'—' if pd.isna(v) else f'{float(v):.2f}',
        'Varasem vs BASE %':lambda v:'—' if pd.isna(v) else f'{float(v):+.1f}%',
        'Varasem suunahitt %':lambda v:'—' if pd.isna(v) else f'{float(v):.0f}%',
    }), use_container_width=True, hide_index=True)

    if not late.empty:
        st.markdown('### 5. 26.08+ hiline saba · ainult informatsioon')
        st.caption('Not used in the verdict because the current plant stand is already a different/ageing regime.')
        lview = late[['date','fields','actual','base','w8','w10','w12','short']].rename(columns={
            'date':'Päev','fields':'Põllud','actual':'Tegelik ABC','base':'LONG BASE',
            'w8':'W8','w10':'W10','w12':'W12','short':'SHORT consensus',
        })
        st.dataframe(lview.style.format({
            'Päev':_fmt_day,'Tegelik ABC':'{:.1f}','LONG BASE':'{:.1f}',
            'W8':'{:.1f}','W10':'{:.1f}','W12':'{:.1f}','SHORT consensus':'{:.1f}',
        }), use_container_width=True, hide_index=True)

    with st.expander('Kõik strict-OOS päevad'):
        st.dataframe(daily, use_container_width=True, hide_index=True)

    st.success(
        '🔒 LEAKAGE / ARCHITECTURE LOCK: every target uses only earlier harvest intervals. All-prior history is used only to estimate stable field identity and LONG BASE. '
        'W8/W10/W12 common time curves use exactly the previous 8/10/12 completed harvest dates; target actual is never in training. '
        'Consensus is fixed median, not selected best K. 19–24.08 is the clean-wave test; <=18.08 is backward validation; 26.08+ is informational only. READ ONLY; production unchanged.'
    )


if __name__ == '__main__':
    main()
