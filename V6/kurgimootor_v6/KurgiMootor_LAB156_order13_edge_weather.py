from __future__ import annotations

"""
KurgiMootor · edge_weather-23
=============================

PHASE-WINDOW WEATHER AUDIT · STRICT OOS-ON-OOS · READ ONLY

Question
--------
If weather does not work as "same growth interval -> same production",
does it carry predictive information in a coarser DEVELOPMENT PHASE
before harvest?

Three windows are fixed BEFORE seeing results:

    EARLY : 8–12 days before harvest
    MID   : 4–7 days before harvest
    LATE  : 0–3 days before harvest

These are deliberately coarse phase windows, NOT a lag search.

Weather channels are also fixed:
    1) radiation
    2) night-temperature stress
    3) WIND×DRY = wind × (100 - RH)

Architecture
------------
A) Build a weatherless interval-sum SEASON BASE.

B) Precompute STRICT OOS BASE predictions for every historical harvest date:
       train target_date < test target_date
   This is done once.

C) For a new target date T, weather correction is trained ONLY on earlier
   STRICT-OOS BASE errors. It never learns from BASE in-sample residuals.

D) Each phase gets its OWN conservative ridge model:
       log(actual_day / base_day)
           ~ phase radiation
             + phase night stress
             + phase WIND×DRY

   Ridge lambda is chosen from a fixed grid using GCV on prior OOS days only.

E) The weather correction is common to the day's three fields and is capped:
       ±0.15 log units (~ ±16%)
   so weather cannot become the production generator.

Decision
--------
A phase is "supported" only if it improves the SEASON BASE:
    - overall
    - in chronological half 1
    - in chronological half 2

No window search.
No 17.08 tuning.
No previous-yield anchor.
No future harvest actual in its own prediction.

Important
---------
Uses MEASURED weather as a mechanism audit.
This is NOT yet archived forecast-weather replay.

READ ONLY
---------
- db.get_harvest_history
- db.get_weather_rows
- no DB writes
- no production snapshots
- no scipy
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
import math
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import streamlit as st

import db


# ---------------------------------------------------------------------
# Locked constants
# ---------------------------------------------------------------------

HOURS_PER_FIELD = 3.0
WEATHER_START = date(2026, 7, 1)

ABC_EPS = 0.20

MIN_BASE_TRAIN_INTERVALS = 35
MIN_FIELD_OBS = 2

BASE_FIELD_RIDGE = 1.5
BASE_SEASON_RIDGE = 0.10
BASE_MAX_ITER = 300

MIN_CORR_TRAIN_DAYS = 6

RIDGE_GRID = [
    1.0,
    3.0,
    10.0,
    30.0,
    100.0,
    300.0,
]

MAX_LOG_CORR = 0.15

PHASES = {
    "EARLY · 8–12p": (8, 12),
    "MID · 4–7p": (4, 7),
    "LATE · 0–3p": (0, 3),
}


@dataclass
class Event:
    day: date
    field: int
    order: int
    abc: float
    interval_days: Optional[float] = None


# ---------------------------------------------------------------------
# Basic parsing
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# Weather
# ---------------------------------------------------------------------

def _measured_weather(
    rows,
):
    out = {}

    for row in rows:
        dd = _d(
            row.get("weather_date")
        )

        if dd is None:
            continue

        if str(
            row.get("data_kind")
            or ""
        ).strip().lower() != "measured":
            continue

        if not bool(
            row.get("checked")
        ):
            continue

        night = _f(
            row.get(
                "temp_night_avg_c"
            )
        )
        rad = _f(
            row.get(
                "radiation_mj_m2"
            )
        )
        wind = _f(
            row.get(
                "wind_avg_ms"
            )
        )
        rh = _f(
            row.get(
                "humidity_avg_pct"
            )
        )

        if None in (
            night,
            rad,
            wind,
            rh,
        ):
            continue

        out[dd] = {
            "night": float(
                night
            ),
            "rad": float(
                rad
            ),
            "winddry": (
                float(wind)
                * (
                    100.0
                    - float(rh)
                )
            ),
        }

    return out


def _night_stress(
    night_c: float,
):
    """
    Fixed scalar:
    0 inside 16–20 C.
    Quadratic penalty below 16 or above 20.
    No fitted threshold.
    """
    cold = max(
        0.0,
        16.0
        - float(night_c),
    ) / 5.0

    heat = max(
        0.0,
        float(night_c)
        - 20.0,
    ) / 5.0

    return (
        cold * cold
        + heat * heat
    )


def _phase_dates(
    target: date,
    start_days_before: int,
    end_days_before: int,
):
    """
    PHASES store low/high days-before values.
    e.g. 8,12 means T-12 ... T-8 inclusive.
    """
    lo = int(
        start_days_before
    )
    hi = int(
        end_days_before
    )

    return [
        target
        - timedelta(days=k)
        for k in range(
            hi,
            lo - 1,
            -1,
        )
    ]


def _phase_vector(
    target: date,
    weather,
    phase_tuple,
):
    lo, hi = phase_tuple

    days = _phase_dates(
        target,
        lo,
        hi,
    )

    recs = []

    for dd in days:
        w = weather.get(
            dd
        )

        if w is None:
            return None

        recs.append(
            [
                float(
                    w["rad"]
                ),
                float(
                    _night_stress(
                        w["night"]
                    )
                ),
                float(
                    w["winddry"]
                ),
            ]
        )

    X = np.asarray(
        recs,
        dtype=float,
    )

    # Windows are fixed length, therefore mean and sum contain
    # the same ranking information up to a constant scale.
    return np.mean(
        X,
        axis=0,
    )


def _weather_complete_for_all_phases(
    target: date,
    weather,
):
    return all(
        _phase_vector(
            target,
            weather,
            phase,
        )
        is not None
        for phase in (
            PHASES.values()
        )
    )


# ---------------------------------------------------------------------
# Weatherless interval-sum SEASON BASE
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# Precompute strict-OOS BASE at field level
# ---------------------------------------------------------------------

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
                "field": int(
                    row["field"]
                ),
                "order": int(
                    row["order"]
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


def _complete_daily_oos(
    base_rows,
    weather,
):
    rows = []

    for target, g in (
        base_rows.groupby(
            "target_date",
            sort=True,
        )
    ):
        if (
            len(g) != 3
            or g[
                "field"
            ].nunique() != 3
        ):
            continue

        if not (
            _weather_complete_for_all_phases(
                target,
                weather,
            )
        ):
            continue

        actual = float(
            g["actual"].sum()
        )

        base = float(
            g["base"].sum()
        )

        rows.append({
            "date": target,
            "fields": ",".join(
                str(
                    int(x)
                )
                for x
                in g.sort_values(
                    "order"
                )[
                    "field"
                ].tolist()
            ),
            "actual": actual,
            "base": base,
            "base_resid": (
                math.log(
                    actual
                    + ABC_EPS
                )
                - math.log(
                    base
                    + ABC_EPS
                )
            ),
            "train_n": int(
                g[
                    "train_n"
                ].min()
            ),
        })

    return pd.DataFrame(
        rows
    ).sort_values(
        "date"
    ).reset_index(
        drop=True
    )


# ---------------------------------------------------------------------
# Conservative phase correction
# ---------------------------------------------------------------------

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
        return np.linalg.solve(
            lhs,
            rhs,
        )

    except np.linalg.LinAlgError:
        return (
            np.linalg.pinv(
                lhs
            )
            @ rhs
        )


def _gcv_lambda(
    X,
    y,
):
    n = len(y)

    XtX = (
        X.T @ X
    )

    rows = []

    for lam in RIDGE_GRID:
        beta = _ridge_fit(
            X,
            y,
            lam,
        )

        pred = (
            X @ beta
        )

        rss = float(
            np.sum(
                (
                    y
                    - pred
                )
                ** 2
            )
        )

        mat = (
            XtX
            + float(lam)
            * np.eye(
                X.shape[1]
            )
        )

        try:
            inv = np.linalg.inv(
                mat
            )
        except np.linalg.LinAlgError:
            inv = np.linalg.pinv(
                mat
            )

        df_eff = float(
            np.trace(
                XtX @ inv
            )
        )

        denom = max(
            (
                1.0
                - df_eff
                / max(
                    n,
                    1,
                )
            )
            ** 2,
            1e-8,
        )

        gcv = (
            rss
            / max(
                n,
                1,
            )
        ) / denom

        rows.append({
            "lambda": float(
                lam
            ),
            "gcv": float(
                gcv
            ),
            "df_eff": float(
                df_eff
            ),
        })

    table = (
        pd.DataFrame(
            rows
        )
        .sort_values(
            [
                "gcv",
                "lambda",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return (
        float(
            table.iloc[0][
                "lambda"
            ]
        ),
        table,
    )


def _phase_matrix(
    dates,
    weather,
    phase_tuple,
):
    return np.vstack([
        _phase_vector(
            dd,
            weather,
            phase_tuple,
        )
        for dd in dates
    ])


def _fit_phase_correction(
    train_days,
    weather,
    phase_tuple,
):
    dates = train_days[
        "date"
    ].tolist()

    X_raw = _phase_matrix(
        dates,
        weather,
        phase_tuple,
    )

    mu = np.mean(
        X_raw,
        axis=0,
    )

    sd = np.std(
        X_raw,
        axis=0,
    )

    sd = np.where(
        sd < 1e-8,
        1.0,
        sd,
    )

    X = (
        X_raw
        - mu
    ) / sd

    y = train_days[
        "base_resid"
    ].to_numpy(
        dtype=float
    )

    lam, gcv = _gcv_lambda(
        X,
        y,
    )

    beta = _ridge_fit(
        X,
        y,
        lam,
    )

    return {
        "mu": mu,
        "sd": sd,
        "beta": beta,
        "lambda": lam,
        "gcv": gcv,
    }


def _phase_effect(
    target,
    weather,
    phase_tuple,
    fit,
):
    raw = _phase_vector(
        target,
        weather,
        phase_tuple,
    )

    z = (
        raw
        - fit["mu"]
    ) / fit["sd"]

    raw_effect = float(
        z @ fit["beta"]
    )

    effect = float(
        np.clip(
            raw_effect,
            -MAX_LOG_CORR,
            MAX_LOG_CORR,
        )
    )

    return (
        effect,
        raw_effect,
    )


def _strict_phase_predictions(
    daily_oos,
    weather,
):
    rows = []

    for target in (
        daily_oos[
            "date"
        ].tolist()
    ):
        prior = daily_oos[
            daily_oos[
                "date"
            ] < target
        ].copy()

        if len(prior) < (
            MIN_CORR_TRAIN_DAYS
        ):
            continue

        target_row = daily_oos[
            daily_oos[
                "date"
            ] == target
        ].iloc[0]

        rec = {
            "date": target,
            "fields": target_row[
                "fields"
            ],
            "actual": float(
                target_row[
                    "actual"
                ]
            ),
            "base": float(
                target_row[
                    "base"
                ]
            ),
            "base_error": (
                float(
                    target_row[
                        "base"
                    ]
                )
                - float(
                    target_row[
                        "actual"
                    ]
                )
            ),
            "corr_train_days": int(
                len(prior)
            ),
        }

        for phase_name, phase_tuple in (
            PHASES.items()
        ):
            fit = _fit_phase_correction(
                prior,
                weather,
                phase_tuple,
            )

            effect, raw_effect = (
                _phase_effect(
                    target,
                    weather,
                    phase_tuple,
                    fit,
                )
            )

            pred = float(
                target_row[
                    "base"
                ]
            ) * math.exp(
                effect
            )

            rec[
                phase_name
            ] = pred

            rec[
                phase_name
                + " effect%"
            ] = (
                100.0
                * (
                    math.exp(
                        effect
                    )
                    - 1.0
                )
            )

            rec[
                phase_name
                + " lambda"
            ] = float(
                fit[
                    "lambda"
                ]
            )

            rec[
                phase_name
                + " raw_effect"
            ] = float(
                raw_effect
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
    col,
):
    use = df[
        df[col].notna()
        & df[
            "base"
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

    a = use[
        "actual"
    ].to_numpy(
        dtype=float
    )

    b = use[
        "base"
    ].to_numpy(
        dtype=float
    )

    p = use[
        col
    ].to_numpy(
        dtype=float
    )

    eb = np.abs(
        b - a
    )

    ep = np.abs(
        p - a
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
                bmae
                - pmae
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
                    np.abs(a),
                    0.5,
                )
            )
            * 100.0
        ),
        "model_mape": float(
            np.mean(
                ep
                / np.maximum(
                    np.abs(a),
                    0.5,
                )
            )
            * 100.0
        ),
    }


def _halves(
    df,
):
    dates = sorted(
        df[
            "date"
        ].tolist()
    )

    cut = (
        len(dates) // 2
    )

    first = set(
        dates[:cut]
    )

    second = set(
        dates[cut:]
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


def _summary_table(
    df,
):
    rows = []

    for phase_name in (
        PHASES
    ):
        m = _metrics(
            df,
            phase_name,
        )

        rows.append({
            "Faas": phase_name,
            "N päeva": m[
                "n"
            ],
            "BASE MAE": m[
                "base_mae"
            ],
            "Faasi MAE": m[
                "model_mae"
            ],
            "Paranemine %": m[
                "improvement"
            ],
            "Võite": m[
                "wins"
            ],
            "BASE MAPE %": m[
                "base_mape"
            ],
            "Faasi MAPE %": m[
                "model_mape"
            ],
        })

    return pd.DataFrame(
        rows
    )


def _is_supported(
    full,
    first,
    second,
    phase_name,
):
    a = _metrics(
        full,
        phase_name,
    )

    b = _metrics(
        first,
        phase_name,
    )

    c = _metrics(
        second,
        phase_name,
    )

    return (
        np.isfinite(
            a[
                "improvement"
            ]
        )
        and np.isfinite(
            b[
                "improvement"
            ]
        )
        and np.isfinite(
            c[
                "improvement"
            ]
        )
        and a[
            "improvement"
        ] > 0.0
        and b[
            "improvement"
        ] > 0.0
        and c[
            "improvement"
        ] > 0.0
    )


# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title=(
            "KurgiMootor · phase weather"
        ),
        layout="wide",
    )

    st.title(
        "Millises arengufaasis ilm üldse infot kannab?"
    )

    st.caption(
        "8–12p / 4–7p / 0–3p enne korjet · "
        "strict OOS-on-OOS · measured-weather mechanism audit · READ ONLY"
    )

    st.info(
        "See EI OLE lag-search. Kolm faasiakent ja kolm ilmakanalit on ette fikseeritud. "
        "Weather õpib ainult varasemate strict-OOS SEASON-prognooside päevavigadest, "
        "mitte BASE'i in-sample residualidest."
    )

    try:
        harvest = db.get_harvest_history(
            limit=5000
        )

        events = _events(
            harvest
        )

        intervals = _build_intervals(
            events
        )

        if intervals.empty:
            st.error(
                "Korjeintervalle ei tekkinud."
            )
            st.stop()

        first_target = min(
            intervals[
                "target_date"
            ]
        )

        last_target = max(
            intervals[
                "target_date"
            ]
        )

        weather_from = max(
            WEATHER_START,
            first_target
            - timedelta(
                days=12
            ),
        )

        weather_rows = (
            db.get_weather_rows(
                weather_from,
                last_target,
            )
        )

        weather = _measured_weather(
            weather_rows
        )

        base_rows = _strict_base_rows(
            intervals
        )

        daily_oos = _complete_daily_oos(
            base_rows,
            weather,
        )

        preds = _strict_phase_predictions(
            daily_oos,
            weather,
        )

    except Exception as exc:
        st.exception(
            exc
        )
        st.stop()

    if preds.empty:
        st.error(
            "Faasi strict-OOS prognoose ei tekkinud."
        )
        st.stop()

    first, second = _halves(
        preds
    )

    full_summary = _summary_table(
        preds
    )

    c1, c2, c3, c4 = st.columns(
        4
    )

    c1.metric(
        "Faasi OOS päevi",
        len(preds),
    )

    c2.metric(
        "Esimene testpäev",
        min(
            preds[
                "date"
            ]
        ).strftime(
            "%d.%m"
        ),
    )

    c3.metric(
        "Viimane testpäev",
        max(
            preds[
                "date"
            ]
        ).strftime(
            "%d.%m"
        ),
    )

    supported = [
        phase
        for phase in PHASES
        if _is_supported(
            preds,
            first,
            second,
            phase,
        )
    ]

    c4.metric(
        "Stabiilselt toetatud faase",
        len(
            supported
        ),
    )

    st.markdown(
        "### 1. Põhitest · milline faas aitab?"
    )

    st.dataframe(
        full_summary.style.format({
            "BASE MAE": "{:.2f}",
            "Faasi MAE": "{:.2f}",
            "Paranemine %": lambda x: (
                "—"
                if pd.isna(x)
                else f"{float(x):+.1f}%"
            ),
            "BASE MAPE %": "{:.1f}",
            "Faasi MAPE %": "{:.1f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    if len(
        supported
    ) == 1:
        st.success(
            f"✅ ÜKS FAAS ON STABIILSELT TOETATUD: {supported[0]}. "
            "See faas parandab BASE'i nii tervikuna kui mõlemas ajapooles."
        )

    elif len(
        supported
    ) > 1:
        st.warning(
            "🟡 Rohkem kui üks faas parandab BASE'i mõlemas ajapooles. "
            "See on huvitav, kuid faasispetsiifilisus pole veel puhas."
        )

    else:
        st.error(
            "❌ ÜKSKI ETTE FIKSEERITUD FAAS EI ANNA STABIILSET OOS-EELIST. "
            "Sel juhul pole aus hakata aknaid tulemuse järgi nihutama."
        )

    st.markdown(
        "### 2. Kõige tähtsam kontroll · kaks ajapoolt"
    )

    half_rows = []

    for period, part in [
        (
            "I pool",
            first,
        ),
        (
            "II pool",
            second,
        ),
    ]:
        tab = _summary_table(
            part
        )

        for _, row in (
            tab.iterrows()
        ):
            half_rows.append({
                "Periood": period,
                **row.to_dict(),
            })

    half_df = pd.DataFrame(
        half_rows
    )

    st.dataframe(
        half_df.style.format({
            "BASE MAE": "{:.2f}",
            "Faasi MAE": "{:.2f}",
            "Paranemine %": lambda x: (
                "—"
                if pd.isna(x)
                else f"{float(x):+.1f}%"
            ),
            "BASE MAPE %": "{:.1f}",
            "Faasi MAPE %": "{:.1f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        "### 3. Päev-päevalt"
    )

    show = preds.copy()

    for phase_name in (
        PHASES
    ):
        show[
            phase_name
            + " viga"
        ] = (
            show[
                phase_name
            ]
            - show[
                "actual"
            ]
        )

    cols = [
        "date",
        "fields",
        "actual",
        "base",
        "base_error",
    ]

    for phase_name in (
        PHASES
    ):
        cols.extend([
            phase_name,
            phase_name
            + " viga",
            phase_name
            + " effect%",
            phase_name
            + " lambda",
        ])

    cols.append(
        "corr_train_days"
    )

    fmt = {
        "date": lambda x: x.strftime(
            "%d.%m"
        ),
        "actual": "{:.1f}",
        "base": "{:.1f}",
        "base_error": "{:+.1f}",
        "corr_train_days": "{:.0f}",
    }

    for phase_name in (
        PHASES
    ):
        fmt[
            phase_name
        ] = "{:.1f}"

        fmt[
            phase_name
            + " viga"
        ] = "{:+.1f}"

        fmt[
            phase_name
            + " effect%"
        ] = "{:+.1f}%"

        fmt[
            phase_name
            + " lambda"
        ] = "{:g}"

    st.dataframe(
        show[
            cols
        ].style.format(
            fmt
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "effect% on kogu päeva kolme põllu ühine multiplicative weather-correction. "
        "Lagi on umbes ±16%, et faasimudel ei muutuks uueks produktsioonigeneraatoriks."
    )

    st.divider()

    st.caption(
        "AUDIT LOCK: BASE prediction for every historical day was first generated strict walk-forward. "
        "Target phase-model uses only earlier days' strict-OOS BASE errors."
    )

    st.caption(
        "Measured-weather mechanism audit only. Kui mõni faas päriselt töötab, "
        "järgmine vajalik kontroll on archived forecast-weather replay."
    )

    st.caption(
        "READ ONLY: db.get_harvest_history + db.get_weather_rows. "
        "SciPy puudub ja DB kirjutamisi ei ole."
    )


if __name__ == "__main__":
    main()
