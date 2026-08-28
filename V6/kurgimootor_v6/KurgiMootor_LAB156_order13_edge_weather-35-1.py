from __future__ import annotations

"""
KurgiMootor · edge_weather-35
=============================

ARCHITECTURE VERDICT · WHOLE-SEASON STRICT WALK-FORWARD · READ ONLY

This is deliberately NOT another coefficient hunt.
It compares three fixed, simple architectures on the same chronological
strict-OOS days:

    A) BASE
       field identity + order-adjusted harvest interval + smooth season level

    B) BASE + SLOW STATE
       BASE multiplied by one common farm/crop state estimated as the median
       log residual of the previous FIVE harvested days. Five harvest days are
       roughly one full 14-field rotation. No same-field previous-yield anchor.

    C) BASE + SLOW STATE + WEATHER CHANGE
       same slow state, plus the already-audited pre-target 4d-vs-4d weather
       transition model. Weather predicts only CHANGE around the slow state;
       it does not set the crop level by itself.

A simple PREVIOUS-CYCLE benchmark is shown only as a benchmark, never as a
production recommendation.

Locks
-----
- every BASE target row is trained only on intervals strictly before target
- target actual is used only after prediction for state update/scoring
- slow state uses only previous harvested days, never target-day actual
- weather uses only T-8..T-1 measured rows; target-day measured weather excluded
- ridge lambda grid and weather cap remain the already locked values
- no alpha/window/cap/ridge search in this file
- irregular 2-field days are valid; 0-harvest days are simply absent, not zero
- READ ONLY: db.get_harvest_history + db.get_weather_rows only

Important interpretation
------------------------
This is a historical walk-forward architecture audit, not a new untouched
holdout. Its job is to answer a practical question: is one simple architecture
stable enough to be forecast-worthy across the available season, rather than
winning one tiny late-August slice by decimals?
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
    past = daily[daily["date"] < target].sort_values("date").reset_index(drop=True)
    rows = []
    for j in range(STATE_LOOKBACK_DAYS, len(past)):
        dd = past.loc[j, "date"]
        prev = past.iloc[j-STATE_LOOKBACK_DAYS:j]
        prev_dates = prev["date"].tolist()
        expected = [dd - timedelta(days=k) for k in (3, 2, 1)]
        if prev_dates != expected:
            continue
        state3 = float(np.mean(prev["daily_state_obs"].to_numpy(dtype=float)))
        rows.append({
            "date": dd,
            "target_delta": float(past.loc[j, "daily_state_obs"] - state3),
            "wx_d_rad": float(past.loc[j, "wx_d_rad"]),
            "wx_d_nightstress": float(past.loc[j, "wx_d_nightstress"]),
            "wx_d_winddry": float(past.loc[j, "wx_d_winddry"]),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Conservative ridge transition model
# ---------------------------------------------------------------------

def _ridge_fit(X, y, lam):
    lhs = X.T @ X + float(lam) * np.eye(X.shape[1])
    rhs = X.T @ y
    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(lhs) @ rhs


def _gcv_lambda(X, y):
    xtx = X.T @ X
    n = len(y)
    rows = []
    for lam in RIDGE_GRID:
        beta = _ridge_fit(X, y, lam)
        pred = X @ beta
        rss = float(np.sum((y - pred) ** 2))
        mat = xtx + float(lam) * np.eye(X.shape[1])
        try:
            inv = np.linalg.inv(mat)
        except np.linalg.LinAlgError:
            inv = np.linalg.pinv(mat)
        df_eff = float(np.trace(xtx @ inv))
        denom = max((1.0 - df_eff / max(n, 1)) ** 2, 1e-8)
        gcv = (rss / max(n, 1)) / denom
        rows.append({"lambda": float(lam), "gcv": gcv, "df_eff": df_eff})
    tab = pd.DataFrame(rows).sort_values(["gcv", "lambda"]).reset_index(drop=True)
    return float(tab.iloc[0]["lambda"]), tab


def _fit_weather_transition(train: pd.DataFrame):
    cols = ["wx_d_rad", "wx_d_nightstress", "wx_d_winddry"]
    Xraw = train[cols].to_numpy(dtype=float)
    y = train["target_delta"].to_numpy(dtype=float)
    mu = Xraw.mean(axis=0)
    sd = Xraw.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    X = (Xraw - mu) / sd
    lam, gcv = _gcv_lambda(X, y)
    beta = _ridge_fit(X, y, lam)
    return {"cols": cols, "mu": mu, "sd": sd, "lambda": lam, "beta": beta, "gcv": gcv}


def _wx_delta_for_row(row: pd.Series, fit):
    raw = np.asarray([float(row[c]) for c in fit["cols"]], dtype=float)
    z = (raw - fit["mu"]) / fit["sd"]
    raw_delta = float(z @ fit["beta"])
    delta = float(np.clip(raw_delta, -MAX_WEATHER_DELTA_LOG, MAX_WEATHER_DELTA_LOG))
    return delta, raw_delta


# ---------------------------------------------------------------------
# Forward holdout helpers
# ---------------------------------------------------------------------

def _fit_forward_base(intervals: pd.DataFrame, target: date):
    """Fit the unchanged interval-sum BASE using only rows before target."""
    train = intervals[intervals["target_date"] < target].copy()
    test = intervals[intervals["target_date"] == target].copy()

    if len(train) < MIN_BASE_TRAIN_INTERVALS:
        raise RuntimeError(
            f"{target:%d.%m}: BASE training rows only {len(train)} < {MIN_BASE_TRAIN_INTERVALS}."
        )
    if test.empty:
        raise RuntimeError(f"{target:%d.%m}: no harvested target intervals found.")

    counts = train.groupby("field").size().to_dict()
    missing = [
        int(f) for f in test["field"].tolist()
        if counts.get(int(f), 0) < MIN_FIELD_OBS
    ]
    if missing:
        raise RuntimeError(
            f"{target:%d.%m}: insufficient prior BASE observations for fields {sorted(set(missing))}."
        )

    fit = _fit_base(train)
    pred = _predict_base(fit, test)
    test = test.sort_values(["order", "field"]).copy().reset_index(drop=True)

    # _predict_base preserves row order of its input; recalc after sorting to be explicit.
    pred = _predict_base(fit, test)
    test["base_pred"] = pred
    test["calendar_interval_days"] = [
        int((dd - sd).days)
        for dd, sd in zip(test["target_date"], test["start_date"])
    ]
    return test, fit, int(len(train))


def _weather_row_for_target(target: date, weather):
    wx = _weather_delta(target, weather)
    if wx is None:
        raise RuntimeError(
            f"{target:%d.%m}: strict pre-target weather block is incomplete."
        )
    return pd.Series({
        "wx_d_rad": float(wx[0]),
        "wx_d_nightstress": float(wx[1]),
        "wx_d_winddry": float(wx[2]),
    })


def _raw_wx_for_target(target: date, weather, wx_fit):
    row = _weather_row_for_target(target, weather)
    cap_delta, raw_delta = _wx_delta_for_row(row, wx_fit)
    return {
        "target": target,
        "cap_delta": float(cap_delta),
        "raw_delta": float(raw_delta),
        "wx_d_rad": float(row["wx_d_rad"]),
        "wx_d_nightstress": float(row["wx_d_nightstress"]),
        "wx_d_winddry": float(row["wx_d_winddry"]),
    }





# =====================================================================
# Architecture verdict: fixed slow state + fixed weather-change layer
# =====================================================================

STATE_HARVEST_DAYS = 5
STATE_LEVEL_CAP_LOG = 0.40
MIN_DAY_FIELDS = 2
DIRECTION_DEADBAND = 0.05       # ignore <5% per-field daily moves
BIG_TURN_THRESHOLD = 0.15       # practical large move = >=15%


def _complete_daily_base_rows(intervals: pd.DataFrame, strict_fields: pd.DataFrame, weather) -> pd.DataFrame:
    """Build only days for which strict BASE covers every actually harvested field."""
    rows = []
    if strict_fields.empty:
        return pd.DataFrame()

    actual_counts = intervals.groupby("target_date").size().to_dict()
    for target, g in strict_fields.groupby("target_date", sort=True):
        g = g.sort_values(["order", "field"]).copy()
        expected = int(actual_counts.get(target, 0))
        if expected < MIN_DAY_FIELDS or len(g) != expected:
            continue

        states = np.log(g["actual"].to_numpy(float) + ABC_EPS) - np.log(
            g["base"].to_numpy(float) + ABC_EPS
        )
        wx = _weather_delta(target, weather)

        # Simple previous-same-field benchmark. It is deliberately NOT used by
        # any candidate architecture.
        naive_vals = []
        for ff in g["field"].astype(int).tolist():
            prev = intervals[
                (intervals["field"] == ff) & (intervals["target_date"] < target)
            ].sort_values("target_date")
            if prev.empty:
                naive_vals = []
                break
            naive_vals.append(float(prev.iloc[-1]["actual"]))

        rows.append({
            "date": target,
            "fields": ",".join(str(int(x)) for x in g["field"].tolist()),
            "n_fields": int(len(g)),
            "actual": float(g["actual"].sum()),
            "base": float(g["base"].sum()),
            "naive": float(sum(naive_vals)) if naive_vals else np.nan,
            "observed_state": float(np.median(states)),
            "state_dispersion": float(np.max(states) - np.min(states)),
            "wx_d_rad": float(wx[0]) if wx is not None else np.nan,
            "wx_d_nightstress": float(wx[1]) if wx is not None else np.nan,
            "wx_d_winddry": float(wx[2]) if wx is not None else np.nan,
            "base_train_n": int(g["train_n"].min()),
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def _slow_state_from_prior(prior: pd.DataFrame) -> Optional[float]:
    if len(prior) < STATE_HARVEST_DAYS:
        return None
    vals = prior.sort_values("date").tail(STATE_HARVEST_DAYS)["observed_state"].to_numpy(float)
    level = float(np.median(vals))
    return float(np.clip(level, -STATE_LEVEL_CAP_LOG, STATE_LEVEL_CAP_LOG))


def _slow_transition_training_rows(daily: pd.DataFrame, target: date) -> pd.DataFrame:
    """
    Historical weather rows for predicting deviation from the slow state.
    Every training row recreates the state using only days before that row.
    """
    past = daily[daily["date"] < target].sort_values("date").reset_index(drop=True)
    rows = []
    for j in range(STATE_HARVEST_DAYS, len(past)):
        current = past.iloc[j]
        wxvals = [current.get("wx_d_rad"), current.get("wx_d_nightstress"), current.get("wx_d_winddry")]
        if any(pd.isna(v) for v in wxvals):
            continue
        prior = past.iloc[:j]
        slow = _slow_state_from_prior(prior)
        if slow is None:
            continue
        rows.append({
            "date": current["date"],
            "target_delta": float(current["observed_state"] - slow),
            "wx_d_rad": float(current["wx_d_rad"]),
            "wx_d_nightstress": float(current["wx_d_nightstress"]),
            "wx_d_winddry": float(current["wx_d_winddry"]),
        })
    return pd.DataFrame(rows)


def _build_architecture_predictions(daily: pd.DataFrame) -> pd.DataFrame:
    out = []
    for j, row in daily.sort_values("date").reset_index(drop=True).iterrows():
        target = row["date"]
        prior = daily[daily["date"] < target].sort_values("date").copy()
        slow = _slow_state_from_prior(prior)

        state_pred = np.nan
        state_wx_pred = np.nan
        wx_cap = np.nan
        wx_raw = np.nan
        wx_train_n = 0
        wx_lambda = np.nan

        if slow is not None:
            state_pred = float(row["base"] * math.exp(slow))
            trans = _slow_transition_training_rows(daily, target)
            current_has_wx = not any(pd.isna(row[c]) for c in ["wx_d_rad", "wx_d_nightstress", "wx_d_winddry"])
            if len(trans) >= MIN_WEATHER_TRANSITIONS and current_has_wx:
                fit = _fit_weather_transition(trans)
                cap_delta, raw_delta = _wx_delta_for_row(row, fit)
                wx_cap = float(cap_delta)
                wx_raw = float(raw_delta)
                wx_train_n = int(len(trans))
                wx_lambda = float(fit["lambda"])
                final_state = float(np.clip(slow + wx_cap, -MAX_STATE_LOG, MAX_STATE_LOG))
                state_wx_pred = float(row["base"] * math.exp(final_state))

        rec = dict(row)
        rec.update({
            "slow_state": slow if slow is not None else np.nan,
            "state": state_pred,
            "state_wx": state_wx_pred,
            "wx_cap": wx_cap,
            "wx_raw": wx_raw,
            "wx_train_n": wx_train_n,
            "wx_lambda": wx_lambda,
        })
        out.append(rec)

    df = pd.DataFrame(out).sort_values("date").reset_index(drop=True)
    return df


def _practical_metrics(g: pd.DataFrame, col: str) -> Dict[str, float]:
    x = g[["actual", "n_fields", col]].dropna().copy()
    if x.empty:
        return {
            "N": 0, "MAE": np.nan, "MAE/põld": np.nan, "MAPE %": np.nan,
            "Median AE": np.nan, "Bias": np.nan, "±10%": np.nan, "±20%": np.nan,
            "Worst AE": np.nan, "Suunahitt %": np.nan, "N suund": 0,
            "Suure pöörde hitt %": np.nan, "N suur pööre": 0,
        }
    err = x[col].to_numpy(float) - x["actual"].to_numpy(float)
    ae = np.abs(err)
    ape = ae / np.maximum(x["actual"].to_numpy(float), 0.5)

    # Direction is scored on yield intensity (boxes/field), so 2-field days do
    # not look like a crop crash merely because fewer fields were harvested.
    dates = x.index.tolist()
    dir_hits = []
    big_hits = []
    for idx in dates:
        pos = g.index.get_loc(idx)
        if pos <= 0:
            continue
        prev = g.iloc[pos - 1]
        cur = g.loc[idx]
        if pd.isna(prev.get(col)) or pd.isna(cur.get(col)):
            continue
        prev_actual_pf = float(prev["actual"]) / max(int(prev["n_fields"]), 1)
        cur_actual_pf = float(cur["actual"]) / max(int(cur["n_fields"]), 1)
        prev_pred_pf = float(prev[col]) / max(int(prev["n_fields"]), 1)
        cur_pred_pf = float(cur[col]) / max(int(cur["n_fields"]), 1)
        actual_move = (cur_actual_pf - prev_actual_pf) / max(abs(prev_actual_pf), 0.5)
        pred_move = (cur_pred_pf - prev_pred_pf) / max(abs(prev_pred_pf), 0.5)
        if abs(actual_move) >= DIRECTION_DEADBAND:
            hit = np.sign(actual_move) == np.sign(pred_move)
            dir_hits.append(bool(hit))
            if abs(actual_move) >= BIG_TURN_THRESHOLD:
                big_hits.append(bool(hit))

    return {
        "N": int(len(x)),
        "MAE": float(np.mean(ae)),
        "MAE/põld": float(np.mean(ae / np.maximum(x["n_fields"].to_numpy(float), 1.0))),
        "MAPE %": float(np.mean(ape) * 100.0),
        "Median AE": float(np.median(ae)),
        "Bias": float(np.mean(err)),
        "±10%": float(np.mean(ape <= 0.10) * 100.0),
        "±20%": float(np.mean(ape <= 0.20) * 100.0),
        "Worst AE": float(np.max(ae)),
        "Suunahitt %": float(np.mean(dir_hits) * 100.0) if dir_hits else np.nan,
        "N suund": int(len(dir_hits)),
        "Suure pöörde hitt %": float(np.mean(big_hits) * 100.0) if big_hits else np.nan,
        "N suur pööre": int(len(big_hits)),
    }


def _summary_table(g: pd.DataFrame) -> pd.DataFrame:
    variants = [
        ("PREV-CYCLE benchmark", "naive"),
        ("A · BASE", "base"),
        ("B · BASE + slow state", "state"),
        ("C · BASE + slow state + WXΔ", "state_wx"),
    ]
    rows = []
    for name, col in variants:
        m = _practical_metrics(g, col)
        rows.append({"Variant": name, **m})
    tab = pd.DataFrame(rows)
    base_mae = float(tab.loc[tab["Variant"] == "A · BASE", "MAE"].iloc[0])
    tab["Paranemine BASE vs %"] = np.where(
        tab["MAE"].notna() & (base_mae > 0),
        (base_mae - tab["MAE"]) / base_mae * 100.0,
        np.nan,
    )
    return tab


def _half_table(g: pd.DataFrame) -> pd.DataFrame:
    if g.empty:
        return pd.DataFrame()
    gg = g.sort_values("date").reset_index(drop=True)
    split = max(1, len(gg) // 2)
    parts = [("I pool", gg.iloc[:split].copy()), ("II pool", gg.iloc[split:].copy())]
    rows = []
    for label, p in parts:
        if p.empty:
            continue
        for name, col in [
            ("BASE", "base"),
            ("SLOW STATE", "state"),
            ("STATE+WXΔ", "state_wx"),
        ]:
            m = _practical_metrics(p, col)
            rows.append({
                "Periood": label,
                "Variant": name,
                "N": m["N"],
                "MAE": m["MAE"],
                "MAPE %": m["MAPE %"],
                "±20%": m["±20%"],
                "Suunahitt %": m["Suunahitt %"],
                "Bias": m["Bias"],
            })
    return pd.DataFrame(rows)


def _fmt_day(v):
    if isinstance(v, pd.Timestamp):
        v = v.date()
    if isinstance(v, date):
        return v.strftime("%d.%m")
    return str(v)


def main():
    st.set_page_config(page_title="KurgiMootor · architecture verdict", layout="wide")
    st.title("KurgiMootor · architecture verdict")
    st.caption("Üks terviklik whole-season walk-forward · ei otsi uusi aknaid ega koefitsiente · READ ONLY")

    st.info(
        "Siin ei vaielda enam komakohtade üle. A=BASE. B lisab ühe aeglase 5-korjepäeva common-state'i. "
        "C laseb ilmal ennustada ainult selle state'i muutust. Kõik kolm ennustavad iga päeva enne selle päeva actual'i nägemist."
    )

    try:
        harvest = db.get_harvest_history(limit=5000)
        events = _events(harvest)
        intervals = _build_intervals(events)
        if intervals.empty:
            raise RuntimeError("Korjeintervalle ei tekkinud.")

        audit_end = max(intervals["target_date"])
        strict_fields = _strict_base_rows(intervals)
        if strict_fields.empty:
            raise RuntimeError("Strict BASE OOS ridu ei tekkinud.")

        earliest = min(strict_fields["target_date"])
        weather_from = max(WEATHER_START, earliest - timedelta(days=2 * WEATHER_BLOCK_DAYS + 2))
        weather = _measured_weather(db.get_weather_rows(weather_from, audit_end - timedelta(days=1)))

        daily = _complete_daily_base_rows(intervals, strict_fields, weather)
        if daily.empty:
            raise RuntimeError("Täielikke strict daily BASE päevi ei tekkinud.")

        pred = _build_architecture_predictions(daily)
        common = pred[
            pred[["base", "state", "state_wx", "naive"]].notna().all(axis=1)
        ].copy().reset_index(drop=True)
        if len(common) < 4:
            raise RuntimeError(f"Kõigile neljale variandile ühiseid päevi ainult {len(common)}.")

        summary = _summary_table(common)
        halves = _half_table(common)

    except Exception as exc:
        st.exception(exc)
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Strict BASE päevi", len(daily))
    c2.metric("Ühiseid võrdluspäevi", len(common))
    c3.metric("Algus", _fmt_day(common["date"].min()))
    c4.metric("Lõpp", _fmt_day(common["date"].max()))

    st.markdown("### 1. Otsustabel · kas meil on päriselt ennustuskõlbulik mootor?")
    st.dataframe(
        summary.style.format({
            "MAE": "{:.1f}", "MAE/põld": "{:.2f}", "MAPE %": "{:.1f}",
            "Median AE": "{:.1f}", "Bias": "{:+.1f}", "±10%": "{:.0f}%", "±20%": "{:.0f}%",
            "Worst AE": "{:.1f}", "Suunahitt %": lambda v: "—" if pd.isna(v) else f"{float(v):.0f}%",
            "Suure pöörde hitt %": lambda v: "—" if pd.isna(v) else f"{float(v):.0f}%",
            "Paranemine BASE vs %": "{:+.1f}%",
        }),
        use_container_width=True, hide_index=True,
    )

    base_row = summary[summary["Variant"] == "A · BASE"].iloc[0]
    candidates = summary[summary["Variant"].str.startswith(("B", "C"))].sort_values("MAE")
    best = candidates.iloc[0]
    gain = float(best["Paranemine BASE vs %"])
    if gain >= 5 and float(best["±20%"] or 0) >= float(base_row["±20%"] or 0):
        st.success(
            f"✅ PRAKTILINE KANDIDAAT: {best['Variant']}. MAE {base_row['MAE']:.1f} → {best['MAE']:.1f} "
            f"({gain:.0f}% parem) ja ±20% tabamus ei halvene. Vaata enne otsust ka kahte ajapoolt."
        )
    elif gain > 0:
        st.warning(
            f"🟡 PAREM, AGA MITTE VEEL SELGE: {best['Variant']} võidab BASE'i {gain:.0f}%, "
            "kuid praktiline tabamus/stabiilsus vajab ajapoolte kontrolli."
        )
    else:
        st.error(
            "❌ Ükski lisakiht ei löö BASE'i ühisel whole-season walk-forward perioodil. "
            "Siis ei ole aus state/weather kihti productionisse suruda."
        )

    st.markdown("### 2. Stabiilsus · esimene pool vs teine pool")
    st.dataframe(
        halves.style.format({
            "MAE": "{:.1f}", "MAPE %": "{:.1f}", "±20%": "{:.0f}%",
            "Suunahitt %": lambda v: "—" if pd.isna(v) else f"{float(v):.0f}%",
            "Bias": "{:+.1f}",
        }),
        use_container_width=True, hide_index=True,
    )
    st.caption(
        "Hea arhitektuur ei pea võitma iga päeva, kuid ta ei tohiks olla 'võitja' ainult hooaja ühes pooles. "
        "See tabel on tähtsam kui 0.1–0.2 kasti erinevus koond-MAE-s."
    )

    st.markdown("### 3. Päev-päevalt · trend, mitte ainult MAE")
    show = common.copy()
    for col in ["base", "state", "state_wx", "naive"]:
        show[f"{col}_err"] = show[col] - show["actual"]
        show[f"{col}_pf"] = show[col] / show["n_fields"]
    show["actual_pf"] = show["actual"] / show["n_fields"]
    cols = [
        "date", "fields", "n_fields", "actual", "base", "state", "state_wx", "naive",
        "base_err", "state_err", "state_wx_err", "slow_state", "wx_cap", "wx_train_n", "base_train_n",
    ]
    st.dataframe(
        show[cols].rename(columns={
            "date":"Päev", "fields":"Põllud", "n_fields":"N põldu", "actual":"Tegelik ABC",
            "base":"A BASE", "state":"B STATE", "state_wx":"C STATE+WXΔ", "naive":"Prev-cycle",
            "base_err":"A viga", "state_err":"B viga", "state_wx_err":"C viga",
            "slow_state":"Slow state", "wx_cap":"WXΔ", "wx_train_n":"WX train N", "base_train_n":"BASE train N",
        }).style.format({
            "Päev": _fmt_day, "Tegelik ABC":"{:.1f}", "A BASE":"{:.1f}", "B STATE":"{:.1f}",
            "C STATE+WXΔ":"{:.1f}", "Prev-cycle":"{:.1f}", "A viga":"{:+.1f}", "B viga":"{:+.1f}",
            "C viga":"{:+.1f}", "Slow state":"{:+.3f}", "WXΔ":"{:+.3f}",
        }),
        use_container_width=True, hide_index=True,
    )

    st.markdown("### 4. Kus mootor päriselt eksib? · 5 halvimat päeva")
    winner_col = "state" if str(best["Variant"]).startswith("B") else "state_wx"
    worst = common.copy()
    worst["winner_error"] = worst[winner_col] - worst["actual"]
    worst["winner_abs_error"] = worst["winner_error"].abs()
    worst = worst.sort_values("winner_abs_error", ascending=False).head(5)
    st.dataframe(
        worst[["date", "fields", "n_fields", "actual", "base", winner_col, "winner_error", "observed_state", "slow_state", "wx_cap"]]
        .rename(columns={
            "date":"Päev", "fields":"Põllud", "n_fields":"N põldu", "actual":"Tegelik",
            "base":"BASE", winner_col:"Parim kandidaat", "winner_error":"Viga",
            "observed_state":"Pärast nähtud state", "slow_state":"Enne teada slow state", "wx_cap":"WXΔ",
        }).style.format({
            "Päev":_fmt_day, "Tegelik":"{:.1f}", "BASE":"{:.1f}", "Parim kandidaat":"{:.1f}",
            "Viga":"{:+.1f}", "Pärast nähtud state":"{:+.3f}", "Enne teada slow state":"{:+.3f}", "WXΔ":"{:+.3f}",
        }),
        use_container_width=True, hide_index=True,
    )

    with st.expander("Kontroll · kõik strict BASE päevad, ka enne state/weather küpsemist"):
        st.dataframe(
            pred[["date", "fields", "n_fields", "actual", "base", "naive", "observed_state", "slow_state", "state", "wx_cap", "state_wx", "base_train_n", "wx_train_n"]]
            .style.format({
                "date":_fmt_day, "actual":"{:.1f}", "base":"{:.1f}", "naive":lambda v:"—" if pd.isna(v) else f"{float(v):.1f}",
                "observed_state":"{:+.3f}", "slow_state":lambda v:"—" if pd.isna(v) else f"{float(v):+.3f}",
                "state":lambda v:"—" if pd.isna(v) else f"{float(v):.1f}", "wx_cap":lambda v:"—" if pd.isna(v) else f"{float(v):+.3f}",
                "state_wx":lambda v:"—" if pd.isna(v) else f"{float(v):.1f}",
            }),
            use_container_width=True, hide_index=True,
        )

    st.success(
        "🔒 LEAKAGE LOCK: BASE treenib ainult varasematel intervalle; slow state näeb ainult eelnevaid korjepäevi; "
        "WXΔ näeb ainult T-8..T-1 ilma ja varem skooritud transition-ridu. Target actual läheb sisse alles pärast prognoosi."
    )
    st.caption(
        f"Fixed choices: slow state = previous {STATE_HARVEST_DAYS} harvested-day median; state safety ±{STATE_LEVEL_CAP_LOG:.2f} log; "
        f"weather cap ±{MAX_WEATHER_DELTA_LOG:.2f} log; weather blocks {WEATHER_BLOCK_DAYS}d vs {WEATHER_BLOCK_DAYS}d. "
        "PREV-CYCLE on ainult benchmark. READ ONLY."
    )


if __name__ == "__main__":
    main()
