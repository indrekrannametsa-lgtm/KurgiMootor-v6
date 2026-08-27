from __future__ import annotations

"""
KurgiMootor · edge_weather-26
=============================

STATE / WEATHER ARCHITECTURE FORK · STRICT OOS · READ ONLY

Purpose
-------
LAB-19 supported harvest-as-interval-sum.
LAB-20 showed exact daily latent timing is not identifiable.
LAB-21..23 showed weather does not work reliably as direct yield generator,
residual corrector, or fixed pre-harvest phase correction.

This LAB gives weather a different job:

    interval-sum SEASON BASE
        -> coarse sequential crop-state (STATE3)
        -> weather predicts only CHANGE in that state (WX delta)

State observation
-----------------
Each harvested field first gets a STRICT-OOS weatherless interval-sum BASE.
Each field is therefore an interval measurement:

    field_state_obs = log((actual ABC + eps) / (strict-OOS base ABC + eps))

The day's three fields are combined robustly:

    daily_state_obs = median(3 field_state_obs)

Before target day T:

    STATE3(T) = mean(last 3 consecutive earlier daily_state_obs)

Weather transition
------------------
Weather does NOT predict boxes or absolute state.
It predicts only:

    observed_state(T) - STATE3(T)

from the change between two consecutive NON-overlapping 4-day weather blocks:

    current : T-4 .. T-1
    previous: T-8 .. T-5

Fixed channels:
    delta radiation mean
    delta night-temperature stress mean
    delta WINDxDRY mean

Weather model for target T is trained only on earlier historical days whose
state observations were themselves based on strict-OOS interval predictions.
Ridge lambda is selected by GCV inside prior data only.
Weather transition is capped at +/-0.15 log units (~ +/-16%).

Compared on the SAME target days:
    BASE
    STATE3
    STATE3 + WXdelta

Decision:
Weather transition is supported only if STATE3+WXdelta beats STATE3 overall
AND in both chronological halves. STATE3 itself must also be checked against BASE.

No lag search.
No phase search.
No 17.08 tuning.
No previous same-field yield anchor.
No target-day actual enters its own prediction.

Measured-weather mechanism audit only; archived forecast replay comes later
only if this architecture works.

READ ONLY:
- db.get_harvest_history
- db.get_weather_rows
- no DB writes
- no production snapshots
- no scipy
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
import math
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import streamlit as st

import db

HOURS_PER_FIELD = 3.0
WEATHER_START = date(2026, 7, 1)
ABC_EPS = 0.20
MIN_BASE_TRAIN_INTERVALS = 35
MIN_FIELD_OBS = 2
BASE_FIELD_RIDGE = 1.5
BASE_SEASON_RIDGE = 0.10
BASE_MAX_ITER = 300

STATE_LOOKBACK_DAYS = 3
WEATHER_BLOCK_DAYS = 4
MIN_WEATHER_TRANSITIONS = 5
RIDGE_GRID = [1.0, 3.0, 10.0, 30.0, 100.0, 300.0]
MAX_WEATHER_DELTA_LOG = 0.15
MAX_STATE_LOG = 0.70


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



# ---------------------------------------------------------------------
# Weather regime delta: current 4d block vs previous 4d block
# ---------------------------------------------------------------------

def _weather_block(end_day: date, weather):
    days = [end_day - timedelta(days=k) for k in reversed(range(WEATHER_BLOCK_DAYS))]
    vals = []
    for dd in days:
        w = weather.get(dd)
        if w is None:
            return None
        vals.append([
            float(w["rad"]),
            float(_night_stress(w["night"])),
            float(w["winddry"]),
        ])
    return np.mean(np.asarray(vals, dtype=float), axis=0)


def _weather_delta(target: date, weather):
    """
    Strict pre-target weather transition.

    current  = T-4 .. T-1
    previous = T-8 .. T-5

    Target-day measured weather is excluded.
    """
    current_end = target - timedelta(days=1)
    previous_end = target - timedelta(days=WEATHER_BLOCK_DAYS + 1)

    current = _weather_block(current_end, weather)
    previous = _weather_block(previous_end, weather)

    if current is None or previous is None:
        return None

    return current - previous


# ---------------------------------------------------------------------
# Convert strict-OOS field interval errors into daily coarse state obs
# ---------------------------------------------------------------------

def _daily_state_rows(field_rows: pd.DataFrame, weather):
    rows = []
    if field_rows.empty:
        return pd.DataFrame()

    for target, g in field_rows.groupby("target_date", sort=True):
        if len(g) != 3 or g["field"].nunique() != 3:
            continue

        wx = _weather_delta(target, weather)
        if wx is None:
            continue

        states = np.log(g["actual"].to_numpy(dtype=float) + ABC_EPS) - np.log(
            g["base"].to_numpy(dtype=float) + ABC_EPS
        )

        rows.append({
            "date": target,
            "fields": ",".join(str(int(x)) for x in g.sort_values("order")["field"].tolist()),
            "actual": float(g["actual"].sum()),
            "base": float(g["base"].sum()),
            "daily_state_obs": float(np.median(states)),
            "state_dispersion": float(np.max(states) - np.min(states)),
            "wx_d_rad": float(wx[0]),
            "wx_d_nightstress": float(wx[1]),
            "wx_d_winddry": float(wx[2]),
            "base_train_n": int(g["train_n"].min()),
        })

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def _consecutive_last3(prior: pd.DataFrame, target: date):
    if len(prior) < STATE_LOOKBACK_DAYS:
        return None
    tail = prior.sort_values("date").tail(STATE_LOOKBACK_DAYS).copy()
    ds = tail["date"].tolist()
    expected = [target - timedelta(days=k) for k in (3, 2, 1)]
    if ds != expected:
        return None
    return float(np.mean(tail["daily_state_obs"].to_numpy(dtype=float)))


def _transition_training_rows(daily: pd.DataFrame, target: date):
    """
    Prior-only state-transition rows.

    Each historical row d contains:
      state3(d)        = mean observed state on d-3..d-1
      observed_state   = state revealed by harvest on d
      observed_delta   = observed_state - state3
      weather features = strict pre-target T-4..T-1 minus T-8..T-5

    The target day itself is never included in training.
    """
    past = daily[daily["date"] < target].sort_values("date").reset_index(drop=True)

    rows = []

    for j in range(STATE_LOOKBACK_DAYS, len(past)):
        dd = past.loc[j, "date"]

        prev = past.iloc[
            j - STATE_LOOKBACK_DAYS : j
        ]

        prev_dates = prev["date"].tolist()

        expected = [
            dd - timedelta(days=k)
            for k in (3, 2, 1)
        ]

        if prev_dates != expected:
            continue

        state3 = float(
            np.mean(
                prev["daily_state_obs"].to_numpy(dtype=float)
            )
        )

        observed_state = float(
            past.loc[j, "daily_state_obs"]
        )

        rows.append({
            "date": dd,
            "state3": state3,
            "observed_state": observed_state,
            "observed_delta": observed_state - state3,
            "wx_d_rad": float(past.loc[j, "wx_d_rad"]),
            "wx_d_nightstress": float(past.loc[j, "wx_d_nightstress"]),
            "wx_d_winddry": float(past.loc[j, "wx_d_winddry"]),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# PHI state persistence
# ---------------------------------------------------------------------

def _fit_phi(train: pd.DataFrame):
    """
    No intercept, no tuning grid.

    observed_state ~= phi * state3

    phi is learned ONLY from prior rows and clipped to [0, 1].
    """
    s = train["state3"].to_numpy(dtype=float)
    y = train["observed_state"].to_numpy(dtype=float)

    denom = float(
        np.sum(
            s * s
        )
    )

    if denom < 1e-10:
        return 0.0

    phi_raw = float(
        np.sum(
            s * y
        )
        / denom
    )

    return float(
        np.clip(
            phi_raw,
            0.0,
            1.0,
        )
    )


# ---------------------------------------------------------------------
# Conservative ridge weather models
# ---------------------------------------------------------------------

def _ridge_fit(X, y, lam):
    lhs = (
        X.T @ X
        + float(lam)
        * np.eye(X.shape[1])
    )

    rhs = X.T @ y

    try:
        return np.linalg.solve(
            lhs,
            rhs,
        )

    except np.linalg.LinAlgError:
        return (
            np.linalg.pinv(lhs)
            @ rhs
        )


def _gcv_lambda(X, y):
    xtx = X.T @ X
    n = len(y)

    rows = []

    for lam in RIDGE_GRID:
        beta = _ridge_fit(
            X,
            y,
            lam,
        )

        pred = X @ beta

        rss = float(
            np.sum(
                (y - pred) ** 2
            )
        )

        mat = (
            xtx
            + float(lam)
            * np.eye(X.shape[1])
        )

        try:
            inv = np.linalg.inv(mat)

        except np.linalg.LinAlgError:
            inv = np.linalg.pinv(mat)

        df_eff = float(
            np.trace(
                xtx @ inv
            )
        )

        denom = max(
            (
                1.0
                - df_eff / max(n, 1)
            ) ** 2,
            1e-8,
        )

        gcv = (
            rss / max(n, 1)
        ) / denom

        rows.append({
            "lambda": float(lam),
            "gcv": float(gcv),
            "df_eff": float(df_eff),
        })

    table = (
        pd.DataFrame(rows)
        .sort_values(
            ["gcv", "lambda"]
        )
        .reset_index(drop=True)
    )

    return (
        float(
            table.iloc[0]["lambda"]
        ),
        table,
    )


def _fit_weather_target(
    train: pd.DataFrame,
    target_col: str,
):
    """
    Weather-only ridge, no intercept.

    Features are standardized from prior rows only.
    With average weather transition, prediction tends toward zero,
    i.e. toward the weatherless BASE state.
    """
    cols = [
        "wx_d_rad",
        "wx_d_nightstress",
        "wx_d_winddry",
    ]

    Xraw = train[
        cols
    ].to_numpy(dtype=float)

    y = train[
        target_col
    ].to_numpy(dtype=float)

    mu = Xraw.mean(axis=0)

    sd = Xraw.std(axis=0)

    sd = np.where(
        sd < 1e-8,
        1.0,
        sd,
    )

    X = (
        Xraw - mu
    ) / sd

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
        "cols": cols,
        "mu": mu,
        "sd": sd,
        "lambda": lam,
        "beta": beta,
        "gcv": gcv,
    }


def _wx_for_row(
    row: pd.Series,
    fit,
):
    raw = np.asarray(
        [
            float(row[c])
            for c in fit["cols"]
        ],
        dtype=float,
    )

    z = (
        raw - fit["mu"]
    ) / fit["sd"]

    raw_value = float(
        z @ fit["beta"]
    )

    value = float(
        np.clip(
            raw_value,
            -MAX_WEATHER_DELTA_LOG,
            MAX_WEATHER_DELTA_LOG,
        )
    )

    return (
        value,
        raw_value,
    )


# ---------------------------------------------------------------------
# Strict sequential predictions
# ---------------------------------------------------------------------

def _strict_predictions(daily: pd.DataFrame):
    rows = []

    daily = (
        daily
        .sort_values("date")
        .reset_index(drop=True)
    )

    for _, target_row in daily.iterrows():
        target = target_row["date"]

        prior = daily[
            daily["date"] < target
        ].copy()

        state3 = _consecutive_last3(
            prior,
            target,
        )

        if state3 is None:
            continue

        transitions = _transition_training_rows(
            daily,
            target,
        )

        if len(transitions) < MIN_WEATHER_TRANSITIONS:
            continue

        # 1) Learn persistence from prior states only.
        phi = _fit_phi(
            transitions
        )

        transitions = transitions.copy()

        transitions[
            "phi_state"
        ] = (
            phi
            * transitions["state3"]
        )

        transitions[
            "phi_resid"
        ] = (
            transitions["observed_state"]
            - transitions["phi_state"]
        )

        # 2) WX-only asks whether weather by itself can predict state level.
        wx_only_fit = _fit_weather_target(
            transitions,
            "observed_state",
        )

        wx_only_state, wx_only_raw = _wx_for_row(
            target_row,
            wx_only_fit,
        )

        # 3) WX-residual asks whether weather adds information AFTER
        #    persistence has already explained what it can.
        wx_resid_fit = _fit_weather_target(
            transitions,
            "phi_resid",
        )

        wx_resid, wx_resid_raw = _wx_for_row(
            target_row,
            wx_resid_fit,
        )

        phi_state = float(
            np.clip(
                phi * state3,
                -MAX_STATE_LOG,
                MAX_STATE_LOG,
            )
        )

        wx_only_state_clip = float(
            np.clip(
                wx_only_state,
                -MAX_STATE_LOG,
                MAX_STATE_LOG,
            )
        )

        combined_state = float(
            np.clip(
                phi_state + wx_resid,
                -MAX_STATE_LOG,
                MAX_STATE_LOG,
            )
        )

        base = float(
            target_row["base"]
        )

        actual = float(
            target_row["actual"]
        )

        observed_state = float(
            target_row["daily_state_obs"]
        )

        observed_delta = (
            observed_state - state3
        )

        observed_phi_resid = (
            observed_state - phi_state
        )

        predicted_delta_combined = (
            combined_state - state3
        )

        pred_wx_only = (
            base
            * math.exp(
                wx_only_state_clip
            )
        )

        pred_phi = (
            base
            * math.exp(
                phi_state
            )
        )

        pred_phi_wx = (
            base
            * math.exp(
                combined_state
            )
        )

        b_only = wx_only_fit["beta"]
        b_resid = wx_resid_fit["beta"]

        rows.append({
            "date": target,
            "fields": target_row["fields"],
            "actual": actual,
            "BASE": base,
            "WX-only": pred_wx_only,
            "PHI-STATE": pred_phi,
            "PHI-STATE+WX": pred_phi_wx,

            "state3": float(state3),
            "phi": float(phi),
            "phi_state": phi_state,
            "observed_state": observed_state,
            "observed_delta": observed_delta,

            "wx_only_state": wx_only_state_clip,
            "wx_only_raw": wx_only_raw,
            "wx_resid": wx_resid,
            "wx_resid_raw": wx_resid_raw,
            "observed_phi_resid": observed_phi_resid,
            "combined_state": combined_state,
            "combined_pred_delta": predicted_delta_combined,

            "wx_only_lambda": float(
                wx_only_fit["lambda"]
            ),
            "wx_resid_lambda": float(
                wx_resid_fit["lambda"]
            ),
            "wx_train_n": int(
                len(transitions)
            ),

            "state_dispersion": float(
                target_row["state_dispersion"]
            ),

            "wx_d_rad": float(
                target_row["wx_d_rad"]
            ),
            "wx_d_nightstress": float(
                target_row["wx_d_nightstress"]
            ),
            "wx_d_winddry": float(
                target_row["wx_d_winddry"]
            ),

            "beta_only_rad": float(b_only[0]),
            "beta_only_night": float(b_only[1]),
            "beta_only_winddry": float(b_only[2]),

            "beta_resid_rad": float(b_resid[0]),
            "beta_resid_night": float(b_resid[1]),
            "beta_resid_winddry": float(b_resid[2]),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def _mae(a, p):
    return float(
        np.mean(
            np.abs(
                np.asarray(p, dtype=float)
                - np.asarray(a, dtype=float)
            )
        )
    )


def _mape(a, p):
    a = np.asarray(
        a,
        dtype=float,
    )

    p = np.asarray(
        p,
        dtype=float,
    )

    return float(
        np.mean(
            np.abs(p - a)
            / np.maximum(
                np.abs(a),
                0.5,
            )
        )
        * 100.0
    )


def _variant_metrics(
    df: pd.DataFrame,
    col: str,
):
    a = df[
        "actual"
    ].to_numpy(dtype=float)

    base = df[
        "BASE"
    ].to_numpy(dtype=float)

    pred = df[
        col
    ].to_numpy(dtype=float)

    bm = _mae(
        a,
        base,
    )

    pm = _mae(
        a,
        pred,
    )

    return {
        "n": len(df),
        "mae": pm,
        "mape": _mape(
            a,
            pred,
        ),
        "vs_base": (
            100.0
            * (bm - pm)
            / bm
            if bm > 1e-9
            else np.nan
        ),
        "wins_base": int(
            np.sum(
                np.abs(pred - a)
                < np.abs(base - a)
            )
        ),
    }


def _all_metrics(
    df: pd.DataFrame,
):
    a = df[
        "actual"
    ].to_numpy(dtype=float)

    base = df[
        "BASE"
    ].to_numpy(dtype=float)

    bm = _mae(
        a,
        base,
    )

    base_mape = _mape(
        a,
        base,
    )

    wx = _variant_metrics(
        df,
        "WX-only",
    )

    phi = _variant_metrics(
        df,
        "PHI-STATE",
    )

    combo = _variant_metrics(
        df,
        "PHI-STATE+WX",
    )

    phi_pred = df[
        "PHI-STATE"
    ].to_numpy(dtype=float)

    combo_pred = df[
        "PHI-STATE+WX"
    ].to_numpy(dtype=float)

    combo_vs_phi = (
        100.0
        * (
            phi["mae"]
            - combo["mae"]
        )
        / phi["mae"]
        if phi["mae"] > 1e-9
        else np.nan
    )

    combo_wins_phi = int(
        np.sum(
            np.abs(
                combo_pred - a
            )
            < np.abs(
                phi_pred - a
            )
        )
    )

    observed_delta = df[
        "observed_delta"
    ].to_numpy(dtype=float)

    combined_delta = df[
        "combined_pred_delta"
    ].to_numpy(dtype=float)

    observed_phi_resid = df[
        "observed_phi_resid"
    ].to_numpy(dtype=float)

    wx_resid = df[
        "wx_resid"
    ].to_numpy(dtype=float)

    return {
        "n": len(df),
        "base_mae": bm,
        "base_mape": base_mape,

        "wx_mae": wx["mae"],
        "wx_mape": wx["mape"],
        "wx_vs_base": wx["vs_base"],
        "wx_wins_base": wx["wins_base"],

        "phi_mae": phi["mae"],
        "phi_mape": phi["mape"],
        "phi_vs_base": phi["vs_base"],
        "phi_wins_base": phi["wins_base"],

        "combo_mae": combo["mae"],
        "combo_mape": combo["mape"],
        "combo_vs_base": combo["vs_base"],
        "combo_wins_base": combo["wins_base"],
        "combo_vs_phi": combo_vs_phi,
        "combo_wins_phi": combo_wins_phi,

        "combined_dir_hit": float(
            np.mean(
                np.sign(combined_delta)
                == np.sign(observed_delta)
            )
            * 100.0
        ),

        "wx_resid_dir_hit": float(
            np.mean(
                np.sign(wx_resid)
                == np.sign(observed_phi_resid)
            )
            * 100.0
        ),

        "phi_mean": float(
            df["phi"].mean()
        ),
        "phi_min": float(
            df["phi"].min()
        ),
        "phi_max": float(
            df["phi"].max()
        ),
    }


def _halves(df: pd.DataFrame):
    dates = sorted(
        df["date"].tolist()
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
            df["date"].isin(first)
        ].copy(),
        df[
            df["date"].isin(second)
        ].copy(),
    )


def _metric_row(label, df):
    m = _all_metrics(df)

    return {
        "Periood": label,
        "N": m["n"],

        "BASE MAE": m["base_mae"],

        "WX-only MAE": m["wx_mae"],
        "WX-only vs BASE %": m["wx_vs_base"],

        "PHI MAE": m["phi_mae"],
        "PHI vs BASE %": m["phi_vs_base"],

        "PHI+WX MAE": m["combo_mae"],
        "PHI+WX vs PHI %": m["combo_vs_phi"],
        "PHI+WX vs BASE %": m["combo_vs_base"],

        "PHI+WX suunahitt %": m["combined_dir_hit"],
        "WX residual suunahitt %": m["wx_resid_dir_hit"],

        "Keskm phi": m["phi_mean"],
    }


# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="KurgiMootor · state/weather fork",
        layout="wide",
    )

    st.title(
        "Kas signaal on ilm üksi või crop-state + ilm?"
    )

    st.caption(
        "BASE vs WX-only vs φ·STATE3 vs φ·STATE3+WX · strict OOS · READ ONLY"
    )

    st.info(
        "Kõik -25 lukud jäävad samaks. Weather = T-4..T-1 miinus T-8..T-5; "
        "target-päeva mõõdetud ilma ei kasutata. φ õpitakse igal testpäeval ainult varasemast infost "
        "ja piiratakse 0…1. Weather-cap jääb ±0.15 log ühikut."
    )

    try:
        harvest = db.get_harvest_history(
            limit=5000
        )

        intervals = _build_intervals(
            _events(harvest)
        )

        if intervals.empty:
            st.error(
                "Korjeintervalle ei tekkinud."
            )
            st.stop()

        earliest = min(
            intervals["target_date"]
        )

        latest = max(
            intervals["target_date"]
        )

        weather_from = max(
            WEATHER_START,
            earliest
            - timedelta(
                days=2 * WEATHER_BLOCK_DAYS
            ),
        )

        weather = _measured_weather(
            db.get_weather_rows(
                weather_from,
                latest,
            )
        )

        field_oos = _strict_base_rows(
            intervals
        )

        daily = _daily_state_rows(
            field_oos,
            weather,
        )

        preds = _strict_predictions(
            daily
        )

    except Exception as exc:
        st.exception(exc)
        st.stop()

    if preds.empty:
        st.error(
            "Strict-OOS võrdluspäevi ei tekkinud."
        )
        st.stop()

    full = _all_metrics(
        preds
    )

    first, second = _halves(
        preds
    )

    fm = _all_metrics(first)
    sm = _all_metrics(second)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "OOS päevi",
        len(preds),
    )

    c2.metric(
        "BASE MAE",
        f"{full['base_mae']:.2f}",
    )

    c3.metric(
        "WX-only MAE",
        f"{full['wx_mae']:.2f}",
        delta=(
            f"{full['base_mae'] - full['wx_mae']:+.2f} vs BASE"
        ),
    )

    c4.metric(
        "PHI+WX MAE",
        f"{full['combo_mae']:.2f}",
        delta=(
            f"{full['phi_mae'] - full['combo_mae']:+.2f} vs PHI"
        ),
    )

    st.markdown(
        "### 1. Põhitest · neli arhitektuuri"
    )

    main_tab = pd.DataFrame([
        {
            "Variant": "BASE",
            "MAE": full["base_mae"],
            "MAPE %": full["base_mape"],
            "Paranemine BASE suhtes %": 0.0,
            "Võite BASE vastu": 0,
        },
        {
            "Variant": "WX-only",
            "MAE": full["wx_mae"],
            "MAPE %": full["wx_mape"],
            "Paranemine BASE suhtes %": full["wx_vs_base"],
            "Võite BASE vastu": full["wx_wins_base"],
        },
        {
            "Variant": "φ·STATE3",
            "MAE": full["phi_mae"],
            "MAPE %": full["phi_mape"],
            "Paranemine BASE suhtes %": full["phi_vs_base"],
            "Võite BASE vastu": full["phi_wins_base"],
        },
        {
            "Variant": "φ·STATE3 + WX",
            "MAE": full["combo_mae"],
            "MAPE %": full["combo_mape"],
            "Paranemine BASE suhtes %": full["combo_vs_base"],
            "Võite BASE vastu": full["combo_wins_base"],
        },
    ])

    st.dataframe(
        main_tab.style.format({
            "MAE": "{:.2f}",
            "MAPE %": "{:.1f}",
            "Paranemine BASE suhtes %": "{:+.1f}%",
        }),
        use_container_width=True,
        hide_index=True,
    )

    wx_only_stable = (
        full["wx_vs_base"] > 0
        and fm["wx_vs_base"] > 0
        and sm["wx_vs_base"] > 0
    )

    phi_stable = (
        full["phi_vs_base"] > 0
        and fm["phi_vs_base"] > 0
        and sm["phi_vs_base"] > 0
    )

    combo_adds_stably = (
        full["combo_vs_phi"] > 0
        and fm["combo_vs_phi"] > 0
        and sm["combo_vs_phi"] > 0
    )

    combo_beats_base_stably = (
        full["combo_vs_base"] > 0
        and fm["combo_vs_base"] > 0
        and sm["combo_vs_base"] > 0
    )

    if (
        combo_adds_stably
        and combo_beats_base_stably
    ):
        st.success(
            "✅ STATE + WEATHER ARCHITEKTUUR PÜSIB: weather lisab φ·STATE3-le väärtust "
            "nii tervikuna kui mõlemas ajapooles ning kombinatsioon lööb BASE'i mõlemas pooles."
        )

    elif wx_only_stable:
        st.success(
            "✅ WEATHER-ONLY ON STABIILNE: ilm üksi lööb BASE'i tervikuna ja mõlemas ajapooles. "
            "Crop-state mälu pole selle testi järgi vajalik põhikomponent."
        )

    elif combo_adds_stably:
        st.warning(
            "🟡 Weather lisab state-mälule stabiilselt infot, kuid kogu kombinatsioon ei löö BASE'i mõlemas ajapooles."
        )

    elif phi_stable:
        st.warning(
            "🟡 Nõrgestatud crop-state mälu töötab, aga weather ei lisa sellele stabiilset väärtust."
        )

    else:
        st.error(
            "❌ ÜKSKI UUS HARU EI ANNA VAJALIKKU STABIILSET EELIST MÕLEMAS AJAPOOLES."
        )

    st.markdown(
        "### 2. Kõige tähtsam kontroll · kaks ajapoolt"
    )

    halves = pd.DataFrame([
        _metric_row(
            "I pool",
            first,
        ),
        _metric_row(
            "II pool",
            second,
        ),
    ])

    st.dataframe(
        halves.style.format({
            "BASE MAE": "{:.2f}",
            "WX-only MAE": "{:.2f}",
            "WX-only vs BASE %": "{:+.1f}%",
            "PHI MAE": "{:.2f}",
            "PHI vs BASE %": "{:+.1f}%",
            "PHI+WX MAE": "{:.2f}",
            "PHI+WX vs PHI %": "{:+.1f}%",
            "PHI+WX vs BASE %": "{:+.1f}%",
            "PHI+WX suunahitt %": "{:.0f}%",
            "WX residual suunahitt %": "{:.0f}%",
            "Keskm phi": "{:.3f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        "### 3. φ · kas vana state päriselt püsib?"
    )

    p1, p2, p3 = st.columns(3)

    p1.metric(
        "Keskmine φ",
        f"{full['phi_mean']:.3f}",
    )

    p2.metric(
        "φ min",
        f"{full['phi_min']:.3f}",
    )

    p3.metric(
        "φ max",
        f"{full['phi_max']:.3f}",
    )

    st.caption(
        "φ=1 tähendaks vana STATE3 täielikku edasikandumist. φ=0 tähendaks, et vana state ei anna järgmise päeva kohta üldse tasemeinfot."
    )

    st.markdown(
        "### 4. Päev-päevalt · mis osa tuleb state'ist ja mis ilmast?"
    )

    show = preds.copy()

    for col, outcol in [
        ("BASE", "BASE viga"),
        ("WX-only", "WX-only viga"),
        ("PHI-STATE", "PHI viga"),
        ("PHI-STATE+WX", "PHI+WX viga"),
    ]:
        show[outcol] = (
            show[col]
            - show["actual"]
        )

    cols = [
        "date",
        "fields",
        "actual",
        "BASE",
        "WX-only",
        "PHI-STATE",
        "PHI-STATE+WX",
        "BASE viga",
        "WX-only viga",
        "PHI viga",
        "PHI+WX viga",
        "state3",
        "phi",
        "phi_state",
        "observed_state",
        "observed_delta",
        "wx_only_state",
        "wx_resid",
        "observed_phi_resid",
        "combined_state",
        "combined_pred_delta",
        "wx_only_lambda",
        "wx_resid_lambda",
        "wx_train_n",
        "state_dispersion",
        "wx_d_rad",
        "wx_d_nightstress",
        "wx_d_winddry",
        "beta_resid_rad",
        "beta_resid_night",
        "beta_resid_winddry",
    ]

    st.dataframe(
        show[cols].style.format({
            "date": lambda x: x.strftime("%d.%m"),
            "actual": "{:.1f}",
            "BASE": "{:.1f}",
            "WX-only": "{:.1f}",
            "PHI-STATE": "{:.1f}",
            "PHI-STATE+WX": "{:.1f}",
            "BASE viga": "{:+.1f}",
            "WX-only viga": "{:+.1f}",
            "PHI viga": "{:+.1f}",
            "PHI+WX viga": "{:+.1f}",
            "state3": "{:+.3f}",
            "phi": "{:.3f}",
            "phi_state": "{:+.3f}",
            "observed_state": "{:+.3f}",
            "observed_delta": "{:+.3f}",
            "wx_only_state": "{:+.3f}",
            "wx_resid": "{:+.3f}",
            "observed_phi_resid": "{:+.3f}",
            "combined_state": "{:+.3f}",
            "combined_pred_delta": "{:+.3f}",
            "wx_only_lambda": "{:g}",
            "wx_resid_lambda": "{:g}",
            "wx_train_n": "{:.0f}",
            "state_dispersion": "{:.3f}",
            "wx_d_rad": "{:+.2f}",
            "wx_d_nightstress": "{:+.3f}",
            "wx_d_winddry": "{:+.1f}",
            "beta_resid_rad": "{:+.3f}",
            "beta_resid_night": "{:+.3f}",
            "beta_resid_winddry": "{:+.3f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "observed_state, observed_delta ja observed_phi_resid on nähtavad alles pärast target-päeva korjet "
        "ja on tabelis ainult diagnoosiks. Ükski neist ei lähe sama päeva prognoosi sisendisse."
    )

    st.caption(
        "LEAKAGE LOCK: WX sisend = T-4..T-1 miinus T-8..T-5. Target-päeva mõõdetud ilma ei kasutata."
    )

    st.caption(
        "WX-only ja PHI+WX weather-mudelid treenitakse eraldi: esimene ennustab state'i taset, "
        "teine ainult seda residuali, mida φ·STATE3 ei seleta."
    )

    st.caption(
        "READ ONLY: db.get_harvest_history + db.get_weather_rows. SciPy puudub ja DB kirjutamisi ei ole."
    )


if __name__ == "__main__":
    main()
