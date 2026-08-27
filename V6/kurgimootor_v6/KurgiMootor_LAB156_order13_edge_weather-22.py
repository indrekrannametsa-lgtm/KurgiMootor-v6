from __future__ import annotations

"""
KurgiMootor · edge_weather-22
=============================

CONSERVATIVE WEATHER CORRECTION ON INTERVAL-SUM BASE · READ ONLY

Why this LAB exists
-------------------
LAB-21 asked weather to help GENERATE daily production and it overreacted.
This LAB gives weather a smaller, more realistic role:

    strong SEASON / interval-sum BASE
        ×
    small weather correction

The BASE remains the production generator.
Weather is only allowed to explain residual deviations around that base.

Architecture
------------
For every outer target harvest date T:

1) Fit M0 SEASON interval-sum model using only intervals with target_date < T.

2) Predict all training intervals with M0 and compute:
       residual_i = log((actual_i + eps) / (base_i + eps))

3) Build interval weather exposures from the SAME growth interval:
   exposure is a BASE-production-weighted average of the daily weather features.

4) Standardize exposures using TRAINING intervals only.

5) Fit ridge residual correction:
       residual ~ weather exposure

   Ridge strength is selected from a pre-fixed grid by GCV on TRAINING data only.

6) For the target interval, apply the correction at DAILY level:
       daily_prod_corrected
          = daily_prod_base × exp(clipped_weather_effect)

   Daily weather effect is hard-limited to ±0.15 log units
   (~ ±16%) so weather cannot become the main generator.

Fixed weather blocks
--------------------
W1 · SOURCE:
    radiation
    nonlinear night-temperature block

W2 · SOURCE+WD:
    W1 + WIND×DRY

No lag search.
No 17.08 tuning.
No previous-yield anchor.
No target-date actual enters target fit.

Important
---------
This is still a measured-weather MECHANISM audit, not archived +1…+9 forecast replay.

READ ONLY:
- db.get_harvest_history
- db.get_weather_rows
- no writes
- no production snapshots
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import streamlit as st

import db


HOURS_PER_FIELD = 3.0
WEATHER_START = date(2026, 7, 1)

MIN_TRAIN_INTERVALS = 35
MIN_FIELD_OBS = 2

ABC_EPS = 0.20

# Base optimizer settings.
BASE_FIELD_RIDGE = 1.5
BASE_SEASON_RIDGE = 0.10
BASE_MAX_ITER = 300

# Weather residual correction:
# selected ONLY within the outer training sample using GCV.
RIDGE_GRID = [0.3, 1.0, 3.0, 10.0, 30.0, 100.0]

# Weather is intentionally prevented from becoming the generator.
MAX_DAILY_LOG_CORR = 0.15

BASE_FEATURES = [
    "intercept",
    "season",
    "season2",
]

WEATHER_BLOCKS = {
    "W1 · SOURCE": [
        "rad",
        "night_cool",
        "night_cool2",
        "night_warm",
        "night_heat",
    ],
    "W2 · SOURCE+WD": [
        "rad",
        "night_cool",
        "night_cool2",
        "night_warm",
        "night_heat",
        "winddry",
    ],
}


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
        return date.fromisoformat(str(v)[:10])
    except Exception:
        return None


def _f(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def _abc(r):
    vals = [_f(r.get(k)) for k in ("a", "b", "c")]

    if any(v is None for v in vals):
        x = _f(r.get("_abc"))
        return (
            x
            if x is not None and x >= 0
            else None
        )

    return float(sum(vals))


def _reliable(r):
    q = str(
        r.get("data_quality")
        or r.get("quality")
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

    for r in rows:
        dd = _d(
            r.get("harvest_date")
        )

        if dd is None or not _reliable(r):
            continue

        try:
            field = int(
                r.get("field_no")
            )
        except Exception:
            continue

        if not 1 <= field <= 14:
            continue

        abc = _abc(r)

        if abc is None or abc < 0:
            continue

        try:
            order = int(
                r.get("harvest_order")
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
                    r.get("interval_days")
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
        (cur.day - prev.day).days
    )

    g += (
        cur.order - prev.order
    ) * (
        HOURS_PER_FIELD / 24.0
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
            prev = hist[i - 1]
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
                + timedelta(days=k)
                for k in range(
                    1,
                    gap + 1,
                )
            ]

            rows.append({
                "target_date": cur.day,
                "start_date": prev.day,
                "field": int(field),
                "order": int(cur.order),
                "actual": float(cur.abc),
                "growth": float(growth),
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


# ---------------------------------------------------------------------
# Weather
# ---------------------------------------------------------------------

def _measured_weather(rows):
    out = {}

    for r in rows:
        dd = _d(
            r.get("weather_date")
        )

        if dd is None:
            continue

        if str(
            r.get("data_kind")
            or ""
        ).strip().lower() != "measured":
            continue

        if not bool(
            r.get("checked")
        ):
            continue

        night = _f(
            r.get("temp_night_avg_c")
        )
        rad = _f(
            r.get("radiation_mj_m2")
        )
        wind = _f(
            r.get("wind_avg_ms")
        )
        rh = _f(
            r.get("humidity_avg_pct")
        )

        if None in (
            night,
            rad,
            wind,
            rh,
        ):
            continue

        out[dd] = {
            "night": float(night),
            "rad": float(rad),
            "winddry": (
                float(wind)
                * (
                    100.0
                    - float(rh)
                )
            ),
        }

    return out


def _night_curve(v):
    cool = max(
        0.0,
        16.0 - v,
    )

    warm = min(
        max(
            v - 16.0,
            0.0,
        ),
        4.0,
    )

    heat = max(
        0.0,
        v - 20.0,
    )

    return {
        "night_cool": (
            cool / 5.0
        ),
        "night_cool2": (
            cool * cool / 25.0
        ),
        "night_warm": (
            warm / 4.0
        ),
        "night_heat": (
            heat / 5.0
        ),
    }


def _daily_record(
    dd: date,
    weather,
):
    w = weather.get(
        dd
    )

    if w is None:
        return None

    season = float(
        (
            dd
            - WEATHER_START
        ).days
    ) / 30.0

    rec = {
        "intercept": 1.0,
        "season": season,
        "season2": (
            season * season
        ),
        "rad": (
            float(
                w["rad"]
            ) / 20.0
        ),
        "winddry": (
            float(
                w["winddry"]
            ) / 100.0
        ),
    }

    rec.update(
        _night_curve(
            float(
                w["night"]
            )
        )
    )

    return rec


def _keep_weather_complete(
    intervals,
    weather,
):
    mask = []

    for _, row in intervals.iterrows():
        ok = all(
            _daily_record(
                dd,
                weather,
            )
            is not None
            for dd in row["days"]
        )

        mask.append(
            ok
        )

    return intervals[
        np.asarray(
            mask,
            dtype=bool,
        )
    ].reset_index(
        drop=True
    )


# ---------------------------------------------------------------------
# M0 interval-sum SEASON base
# ---------------------------------------------------------------------

def _base_feature_cache(
    intervals,
    weather,
):
    all_days = sorted({
        dd
        for days in intervals["days"].tolist()
        for dd in days
    })

    cache = {}

    for dd in all_days:
        rec = _daily_record(
            dd,
            weather,
        )

        if rec is None:
            raise RuntimeError(
                f"Puuduv ilm: {dd}"
            )

        cache[dd] = np.asarray(
            [
                rec["intercept"],
                rec["season"],
                rec["season2"],
            ],
            dtype=float,
        )

    return cache


def _predict_base_intervals(
    intervals,
    beta,
    gammas,
    cache,
):
    preds = []

    for _, row in intervals.iterrows():
        X = np.vstack([
            cache[dd]
            for dd in row["days"]
        ])

        eta = X @ beta

        daily_common = np.exp(
            np.clip(
                eta,
                -6.0,
                6.0,
            )
        )

        common_sum = float(
            row["per_day_weight"]
            * np.sum(
                daily_common
            )
        )

        field = int(
            row["field"]
        )

        field_factor = (
            1.0
            if field == 1
            else math.exp(
                float(
                    gammas[
                        field - 2
                    ]
                )
            )
        )

        preds.append(
            field_factor
            * common_sum
        )

    return np.asarray(
        preds,
        dtype=float,
    )


def _fit_base(
    train,
    weather,
):
    cache = _base_feature_cache(
        train,
        weather,
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

    interval_X = []
    interval_w = []
    interval_field = []

    for _, row in train.iterrows():
        X = np.vstack([
            cache[dd]
            for dd in row["days"]
        ])

        interval_X.append(
            X
        )
        interval_w.append(
            float(
                row[
                    "per_day_weight"
                ]
            )
        )
        interval_field.append(
            int(
                row["field"]
            )
        )

    lr = 0.035
    adam_b1 = 0.9
    adam_b2 = 0.999
    adam_eps = 1e-8

    m_beta = np.zeros_like(
        beta
    )
    v_beta = np.zeros_like(
        beta
    )
    m_gamma = np.zeros_like(
        gammas
    )
    v_gamma = np.zeros_like(
        gammas
    )

    previous = None

    for step in range(
        1,
        BASE_MAX_ITER + 1,
    ):
        grad_beta = np.zeros_like(
            beta
        )
        grad_gamma = np.zeros_like(
            gammas
        )

        data_obj = 0.0

        for i, (
            X,
            w,
            field,
        ) in enumerate(
            zip(
                interval_X,
                interval_w,
                interval_field,
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
                w
                * np.sum(
                    prod
                )
            )

            if field == 1:
                ff = 1.0
                gamma_idx = None
            else:
                gamma_idx = (
                    field - 2
                )
                ff = math.exp(
                    float(
                        gammas[
                            gamma_idx
                        ]
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

            data_obj += (
                resid * resid
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

            grad_beta += (
                2.0
                * resid
                * shrink
                * x_bar
            )

            if gamma_idx is not None:
                grad_gamma[
                    gamma_idx
                ] += (
                    2.0
                    * resid
                    * shrink
                )

        reg = (
            BASE_SEASON_RIDGE
            * (
                beta[1] * beta[1]
                + beta[2] * beta[2]
            )
            + BASE_FIELD_RIDGE
            * float(
                np.sum(
                    gammas * gammas
                )
            )
        )

        grad_beta[1:] += (
            2.0
            * BASE_SEASON_RIDGE
            * beta[1:]
        )

        grad_gamma += (
            2.0
            * BASE_FIELD_RIDGE
            * gammas
        )

        obj = (
            data_obj
            + reg
        )

        n_scale = max(
            len(train),
            1,
        )

        grad_beta /= n_scale
        grad_gamma /= n_scale

        nb = float(
            np.linalg.norm(
                grad_beta
            )
        )

        ng = float(
            np.linalg.norm(
                grad_gamma
            )
        )

        if nb > 10.0:
            grad_beta *= (
                10.0 / nb
            )

        if ng > 10.0:
            grad_gamma *= (
                10.0 / ng
            )

        m_beta = (
            adam_b1 * m_beta
            + (
                1.0
                - adam_b1
            )
            * grad_beta
        )

        v_beta = (
            adam_b2 * v_beta
            + (
                1.0
                - adam_b2
            )
            * (
                grad_beta
                * grad_beta
            )
        )

        m_gamma = (
            adam_b1 * m_gamma
            + (
                1.0
                - adam_b1
            )
            * grad_gamma
        )

        v_gamma = (
            adam_b2 * v_gamma
            + (
                1.0
                - adam_b2
            )
            * (
                grad_gamma
                * grad_gamma
            )
        )

        mbh = (
            m_beta
            / (
                1.0
                - adam_b1 ** step
            )
        )

        vbh = (
            v_beta
            / (
                1.0
                - adam_b2 ** step
            )
        )

        mgh = (
            m_gamma
            / (
                1.0
                - adam_b1 ** step
            )
        )

        vgh = (
            v_gamma
            / (
                1.0
                - adam_b2 ** step
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
            previous is not None
            and step > 80
            and abs(
                previous
                - obj
            )
            < (
                1e-7
                * max(
                    1.0,
                    abs(previous),
                )
            )
        ):
            break

        previous = obj

    return {
        "beta": beta,
        "gammas": gammas,
        "cache": cache,
        "iterations": int(step),
    }


def _predict_base_with_fit(
    fit,
    intervals,
    weather,
):
    cache = _base_feature_cache(
        intervals,
        weather,
    )

    return _predict_base_intervals(
        intervals,
        fit["beta"],
        fit["gammas"],
        cache,
    )


def _base_daily_common(
    fit,
    dd,
    weather,
):
    rec = _daily_record(
        dd,
        weather,
    )

    X = np.asarray([
        rec["intercept"],
        rec["season"],
        rec["season2"],
    ])

    return float(
        math.exp(
            float(
                np.clip(
                    X @ fit["beta"],
                    -6.0,
                    6.0,
                )
            )
        )
    )


def _field_factor(
    fit,
    field,
):
    if int(field) == 1:
        return 1.0

    return math.exp(
        float(
            fit["gammas"][
                int(field) - 2
            ]
        )
    )


# ---------------------------------------------------------------------
# Weather residual correction
# ---------------------------------------------------------------------

def _raw_weather_vector(
    dd,
    weather,
    names,
):
    rec = _daily_record(
        dd,
        weather,
    )

    return np.asarray(
        [
            float(
                rec[name]
            )
            for name in names
        ],
        dtype=float,
    )


def _training_daily_weather_stats(
    train,
    weather,
    names,
):
    days = sorted({
        dd
        for interval_days
        in train["days"].tolist()
        for dd in interval_days
    })

    X = np.vstack([
        _raw_weather_vector(
            dd,
            weather,
            names,
        )
        for dd in days
    ])

    mu = np.mean(
        X,
        axis=0,
    )

    sd = np.std(
        X,
        axis=0,
    )

    sd = np.where(
        sd < 1e-8,
        1.0,
        sd,
    )

    return mu, sd


def _interval_exposure(
    row,
    base_fit,
    weather,
    names,
    mu,
    sd,
):
    raw = np.vstack([
        _raw_weather_vector(
            dd,
            weather,
            names,
        )
        for dd in row["days"]
    ])

    z = (
        raw - mu
    ) / sd

    base_prod = np.asarray([
        _base_daily_common(
            base_fit,
            dd,
            weather,
        )
        for dd in row["days"]
    ])

    weights = (
        float(
            row["per_day_weight"]
        )
        * base_prod
    )

    denom = max(
        float(
            np.sum(
                weights
            )
        ),
        1e-12,
    )

    return (
        weights[:, None]
        * z
    ).sum(axis=0) / denom


def _exposure_matrix(
    intervals,
    base_fit,
    weather,
    names,
    mu,
    sd,
):
    return np.vstack([
        _interval_exposure(
            row,
            base_fit,
            weather,
            names,
            mu,
            sd,
        )
        for _, row in intervals.iterrows()
    ])


def _ridge_fit(
    X,
    y,
    lam,
):
    p = X.shape[1]

    lhs = (
        X.T @ X
        + float(lam)
        * np.eye(p)
    )

    rhs = (
        X.T @ y
    )

    try:
        beta = np.linalg.solve(
            lhs,
            rhs,
        )
    except np.linalg.LinAlgError:
        beta = (
            np.linalg.pinv(
                lhs
            )
            @ rhs
        )

    return beta


def _gcv_lambda(
    X,
    y,
):
    n = len(y)

    if n <= X.shape[1] + 2:
        return (
            RIDGE_GRID[-1],
            pd.DataFrame(),
        )

    rows = []

    XtX = X.T @ X

    for lam in RIDGE_GRID:
        beta = _ridge_fit(
            X,
            y,
            lam,
        )

        pred = X @ beta

        rss = float(
            np.sum(
                (
                    y - pred
                ) ** 2
            )
        )

        try:
            inv = np.linalg.inv(
                XtX
                + lam
                * np.eye(
                    X.shape[1]
                )
            )
        except np.linalg.LinAlgError:
            inv = np.linalg.pinv(
                XtX
                + lam
                * np.eye(
                    X.shape[1]
                )
            )

        df_eff = float(
            np.trace(
                XtX @ inv
            )
        )

        denom = max(
            (
                1.0
                - df_eff / n
            ) ** 2,
            1e-8,
        )

        gcv = (
            rss / n
        ) / denom

        rows.append({
            "lambda": float(lam),
            "gcv": float(gcv),
            "df_eff": df_eff,
        })

    table = pd.DataFrame(
        rows
    ).sort_values(
        [
            "gcv",
            "lambda",
        ]
    ).reset_index(
        drop=True
    )

    return (
        float(
            table.iloc[0][
                "lambda"
            ]
        ),
        table,
    )


def _fit_weather_correction(
    train,
    base_fit,
    base_pred_train,
    weather,
    names,
):
    mu, sd = (
        _training_daily_weather_stats(
            train,
            weather,
            names,
        )
    )

    X = _exposure_matrix(
        train,
        base_fit,
        weather,
        names,
        mu,
        sd,
    )

    y = (
        np.log(
            train[
                "actual"
            ].to_numpy(
                dtype=float
            )
            + ABC_EPS
        )
        - np.log(
            base_pred_train
            + ABC_EPS
        )
    )

    # No intercept: standardized weather should not be allowed to
    # re-create the baseline level.
    lam, gcv_table = (
        _gcv_lambda(
            X,
            y,
        )
    )

    beta = _ridge_fit(
        X,
        y,
        lam,
    )

    return {
        "names": list(names),
        "mu": mu,
        "sd": sd,
        "beta": beta,
        "lambda": lam,
        "gcv": gcv_table,
    }


def _predict_corrected_interval(
    row,
    base_fit,
    correction_fit,
    weather,
):
    names = correction_fit[
        "names"
    ]

    mu = correction_fit[
        "mu"
    ]

    sd = correction_fit[
        "sd"
    ]

    beta = correction_fit[
        "beta"
    ]

    corrected_common = 0.0
    base_common = 0.0
    max_abs_effect = 0.0

    for dd in row["days"]:
        base_daily = (
            _base_daily_common(
                base_fit,
                dd,
                weather,
            )
        )

        raw = _raw_weather_vector(
            dd,
            weather,
            names,
        )

        z = (
            raw - mu
        ) / sd

        raw_effect = float(
            z @ beta
        )

        effect = float(
            np.clip(
                raw_effect,
                -MAX_DAILY_LOG_CORR,
                MAX_DAILY_LOG_CORR,
            )
        )

        max_abs_effect = max(
            max_abs_effect,
            abs(effect),
        )

        weight = float(
            row[
                "per_day_weight"
            ]
        )

        base_common += (
            weight
            * base_daily
        )

        corrected_common += (
            weight
            * base_daily
            * math.exp(
                effect
            )
        )

    ff = _field_factor(
        base_fit,
        int(row["field"]),
    )

    return {
        "base": (
            ff
            * base_common
        ),
        "corrected": (
            ff
            * corrected_common
        ),
        "max_abs_effect": (
            max_abs_effect
        ),
    }


# ---------------------------------------------------------------------
# Strict walk-forward
# ---------------------------------------------------------------------

def _complete_day_map(
    intervals,
):
    out = {}

    for dd, g in intervals.groupby(
        "target_date",
        sort=True,
    ):
        if (
            len(g) == 3
            and g[
                "field"
            ].nunique() == 3
        ):
            out[dd] = (
                g.index.tolist()
            )

    return out


def _walk_forward(
    intervals,
    weather,
):
    complete = _complete_day_map(
        intervals
    )

    rows = []

    for target in sorted(
        complete
    ):
        train = intervals[
            intervals[
                "target_date"
            ] < target
        ].copy()

        test = intervals.loc[
            complete[target]
        ].copy()

        if len(train) < (
            MIN_TRAIN_INTERVALS
        ):
            continue

        counts = (
            train.groupby(
                "field"
            ).size().to_dict()
        )

        if any(
            counts.get(
                int(f),
                0,
            ) < MIN_FIELD_OBS
            for f
            in test[
                "field"
            ].tolist()
        ):
            continue

        base_fit = _fit_base(
            train,
            weather,
        )

        base_pred_train = (
            _predict_base_with_fit(
                base_fit,
                train,
                weather,
            )
        )

        rec = {
            "date": target,
            "fields": ",".join(
                str(
                    int(x)
                )
                for x
                in test.sort_values(
                    "order"
                )["field"].tolist()
            ),
            "actual": float(
                test[
                    "actual"
                ].sum()
            ),
            "train_n": int(
                len(train)
            ),
        }

        base_test = (
            _predict_base_with_fit(
                base_fit,
                test,
                weather,
            )
        )

        rec[
            "M0 · SEASON"
        ] = float(
            np.sum(
                base_test
            )
        )

        for model_name, names in (
            WEATHER_BLOCKS.items()
        ):
            corr_fit = (
                _fit_weather_correction(
                    train,
                    base_fit,
                    base_pred_train,
                    weather,
                    names,
                )
            )

            preds = []
            max_effects = []

            for _, row in (
                test.iterrows()
            ):
                out = (
                    _predict_corrected_interval(
                        row,
                        base_fit,
                        corr_fit,
                        weather,
                    )
                )

                preds.append(
                    out[
                        "corrected"
                    ]
                )

                max_effects.append(
                    out[
                        "max_abs_effect"
                    ]
                )

            rec[
                model_name
            ] = float(
                np.sum(
                    preds
                )
            )

            rec[
                model_name
                + " lambda"
            ] = float(
                corr_fit[
                    "lambda"
                ]
            )

            rec[
                model_name
                + " maxcorr%"
            ] = float(
                100.0
                * (
                    math.exp(
                        max(
                            max_effects
                            or [0.0]
                        )
                    )
                    - 1.0
                )
            )

        rows.append(
            rec
        )

    return pd.DataFrame(
        rows
    )


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def _metrics(
    df,
    model_col,
):
    use = df[
        df[
            model_col
        ].notna()
        & df[
            "M0 · SEASON"
        ].notna()
        & df[
            "actual"
        ].notna()
    ].copy()

    if use.empty:
        return {
            "n": 0,
            "base_mae": np.nan,
            "model_mae": np.nan,
            "improvement": np.nan,
            "wins": 0,
            "base_mape": np.nan,
            "model_mape": np.nan,
        }

    actual = use[
        "actual"
    ].to_numpy(
        dtype=float
    )

    base = use[
        "M0 · SEASON"
    ].to_numpy(
        dtype=float
    )

    pred = use[
        model_col
    ].to_numpy(
        dtype=float
    )

    eb = np.abs(
        base - actual
    )

    ep = np.abs(
        pred - actual
    )

    bmae = float(
        np.mean(
            eb
        )
    )

    pmae = float(
        np.mean(
            ep
        )
    )

    return {
        "n": int(
            len(use)
        ),
        "base_mae": bmae,
        "model_mae": pmae,
        "improvement": (
            100.0
            * (
                bmae - pmae
            )
            / bmae
            if bmae > 1e-9
            else np.nan
        ),
        "wins": int(
            np.sum(
                ep < eb
            )
        ),
        "base_mape": float(
            np.mean(
                eb
                / np.maximum(
                    np.abs(actual),
                    0.5,
                )
            )
            * 100.0
        ),
        "model_mape": float(
            np.mean(
                ep
                / np.maximum(
                    np.abs(actual),
                    0.5,
                )
            )
            * 100.0
        ),
    }


def _halves(
    df,
):
    days = sorted(
        df[
            "date"
        ].tolist()
    )

    cut = (
        len(days) // 2
    )

    first = set(
        days[:cut]
    )

    second = set(
        days[cut:]
    )

    return (
        df[
            df[
                "date"
            ].isin(
                first
            )
        ].copy(),
        df[
            df[
                "date"
            ].isin(
                second
            )
        ].copy(),
    )


def _metrics_table(
    df,
):
    rows = []

    for model_name in (
        WEATHER_BLOCKS
    ):
        m = _metrics(
            df,
            model_name,
        )

        rows.append({
            "Mudel": model_name,
            "N päeva": m[
                "n"
            ],
            "SEASON MAE": m[
                "base_mae"
            ],
            "Mudeli MAE": m[
                "model_mae"
            ],
            "Paranemine %": m[
                "improvement"
            ],
            "Võite": m[
                "wins"
            ],
            "SEASON MAPE %": m[
                "base_mape"
            ],
            "Mudeli MAPE %": m[
                "model_mape"
            ],
        })

    return pd.DataFrame(
        rows
    )


def main():
    st.set_page_config(
        page_title=(
            "KurgiMootor · conservative weather"
        ),
        layout="wide",
    )

    st.title(
        "Ilm kui korrigeerija, mitte generaator"
    )

    st.caption(
        "SEASON/intervallisumma BASE × väike weather correction · "
        "strict walk-forward · measured-weather mechanism audit · READ ONLY"
    )

    st.info(
        "Weather ei tohi selles LAB-is kogu produktsioonitaset ümber kirjutada. "
        "Ta õpib ainult SEASON-baasi residuali ning ühe kasvupäeva mõju on piiratud umbes ±16%-ga. "
        "Ridge tugevus valitakse igal testpäeval ainult varasemast treeningandmest GCV-ga."
    )

    try:
        harvest = db.get_harvest_history(
            limit=5000
        )

        events = _events(
            harvest
        )

        intervals = (
            _build_intervals(
                events
            )
        )

        if intervals.empty:
            st.error(
                "Korjeintervalle ei tekkinud."
            )
            st.stop()

        weather_end = max(
            dd
            for days
            in intervals[
                "days"
            ].tolist()
            for dd
            in days
        )

        weather_rows = (
            db.get_weather_rows(
                WEATHER_START,
                weather_end,
            )
        )

        weather = (
            _measured_weather(
                weather_rows
            )
        )

        intervals = (
            _keep_weather_complete(
                intervals,
                weather,
            )
        )

        oos = _walk_forward(
            intervals,
            weather,
        )

    except Exception as exc:
        st.exception(
            exc
        )
        st.stop()

    if oos.empty:
        st.error(
            "Strict OOS päevi ei tekkinud."
        )
        st.stop()

    full = _metrics_table(
        oos
    )

    first, second = _halves(
        oos
    )

    first_m = (
        _metrics_table(
            first
        )
    )

    second_m = (
        _metrics_table(
            second
        )
    )

    c1, c2, c3, c4 = st.columns(
        4
    )

    c1.metric(
        "OOS päevi",
        len(oos),
    )

    c2.metric(
        "Korjeintervalle",
        len(intervals),
    )

    c3.metric(
        "Esimene OOS",
        min(
            oos["date"]
        ).strftime(
            "%d.%m"
        ),
    )

    c4.metric(
        "Viimane OOS",
        max(
            oos["date"]
        ).strftime(
            "%d.%m"
        ),
    )

    st.markdown(
        "### 1. Kas väike weather-correction aitab?"
    )

    st.dataframe(
        full.style.format({
            "SEASON MAE": "{:.2f}",
            "Mudeli MAE": "{:.2f}",
            "Paranemine %": lambda x: (
                "—"
                if pd.isna(x)
                else f"{float(x):+.1f}%"
            ),
            "SEASON MAPE %": "{:.1f}",
            "Mudeli MAPE %": "{:.1f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    # Main conservative decision uses W2.
    w2 = _metrics(
        oos,
        "W2 · SOURCE+WD",
    )

    w2a = _metrics(
        first,
        "W2 · SOURCE+WD",
    )

    w2b = _metrics(
        second,
        "W2 · SOURCE+WD",
    )

    stable = (
        np.isfinite(
            w2["improvement"]
        )
        and np.isfinite(
            w2a["improvement"]
        )
        and np.isfinite(
            w2b["improvement"]
        )
        and w2[
            "improvement"
        ] > 0.0
        and w2a[
            "improvement"
        ] > 0.0
        and w2b[
            "improvement"
        ] > 0.0
    )

    if stable:
        st.success(
            f"✅ KONSERVATIIVNE WEATHER-KIHT TÖÖTAB: "
            f"MAE {w2['base_mae']:.2f} → {w2['model_mae']:.2f} "
            f"({w2['improvement']:+.1f}%) ja eelis püsib mõlemas ajapooles."
        )

    elif (
        np.isfinite(
            w2["improvement"]
        )
        and w2[
            "improvement"
        ] > 0.0
    ):
        st.warning(
            f"🟡 Weather-correction annab üldise eelise "
            f"({w2['improvement']:+.1f}%), kuid mitte mõlemas ajapooles."
        )

    else:
        st.error(
            f"❌ ISEGI KONSERVATIIVNE WEATHER-KORREKTSIOON EI PARANDA BASE'i: "
            f"{w2['base_mae']:.2f} → {w2['model_mae']:.2f}."
        )

    st.markdown(
        "### 2. Kõige tähtsam kontroll · kaks ajapoolt"
    )

    rows = []

    for period, mt in [
        (
            "I pool",
            first_m,
        ),
        (
            "II pool",
            second_m,
        ),
    ]:
        for _, r in (
            mt.iterrows()
        ):
            rows.append({
                "Periood": period,
                **r.to_dict(),
            })

    halves = pd.DataFrame(
        rows
    )

    st.dataframe(
        halves.style.format({
            "SEASON MAE": "{:.2f}",
            "Mudeli MAE": "{:.2f}",
            "Paranemine %": lambda x: (
                "—"
                if pd.isna(x)
                else f"{float(x):+.1f}%"
            ),
            "SEASON MAPE %": "{:.1f}",
            "Mudeli MAPE %": "{:.1f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        "### 3. Päev-päevalt · kas weather enam üle ei reageeri?"
    )

    show = oos.copy()

    for model_name in (
        WEATHER_BLOCKS
    ):
        show[
            model_name
            + " viga"
        ] = (
            show[
                model_name
            ]
            - show[
                "actual"
            ]
        )

    show[
        "SEASON viga"
    ] = (
        show[
            "M0 · SEASON"
        ]
        - show[
            "actual"
        ]
    )

    display_cols = [
        "date",
        "fields",
        "actual",
        "M0 · SEASON",
        "W1 · SOURCE",
        "W2 · SOURCE+WD",
        "SEASON viga",
        "W1 · SOURCE viga",
        "W2 · SOURCE+WD viga",
        "W1 · SOURCE lambda",
        "W2 · SOURCE+WD lambda",
        "W1 · SOURCE maxcorr%",
        "W2 · SOURCE+WD maxcorr%",
        "train_n",
    ]

    st.dataframe(
        show[
            display_cols
        ].style.format({
            "date": lambda x: (
                x.strftime(
                    "%d.%m"
                )
            ),
            "actual": "{:.1f}",
            "M0 · SEASON": "{:.1f}",
            "W1 · SOURCE": "{:.1f}",
            "W2 · SOURCE+WD": "{:.1f}",
            "SEASON viga": "{:+.1f}",
            "W1 · SOURCE viga": "{:+.1f}",
            "W2 · SOURCE+WD viga": "{:+.1f}",
            "W1 · SOURCE lambda": "{:g}",
            "W2 · SOURCE+WD lambda": "{:g}",
            "W1 · SOURCE maxcorr%": "{:.1f}",
            "W2 · SOURCE+WD maxcorr%": "{:.1f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "maxcorr% näitab suurimat ühe kasvupäeva weather-korrektsiooni target-intervallides. "
        "Arhitektuurne lagi on umbes 16%; see ei ole augusti järgi timmitud."
    )

    st.divider()

    st.caption(
        "MEHHANISMI AUDIT: kasutab realiseerunud mõõdetud ilma. "
        "Kui see kiht töötab, tuleb alles järgmises etapis kontrollida archived forecast-weather replay'd."
    )

    st.caption(
        "AUDIT LOCK: target-päeva BASE ja weather residual-model treenivad ainult intervalle target_date < target. "
        "Ridge lambda valitakse ainult outer-train GCV-ga."
    )

    st.caption(
        "READ ONLY: db.get_harvest_history + db.get_weather_rows. DB kirjutamisi ei ole."
    )


if __name__ == "__main__":
    main()
