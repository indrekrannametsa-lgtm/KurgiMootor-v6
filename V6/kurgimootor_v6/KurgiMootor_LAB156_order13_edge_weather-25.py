from __future__ import annotations

"""
KurgiMootor · edge_weather-25
=============================

COARSE CROP-STATE + PRE-TARGET WEATHER-DELTA · STRICT OOS · READ ONLY

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
# Strict sequential predictions
# ---------------------------------------------------------------------

def _strict_predictions(daily: pd.DataFrame):
    rows = []
    daily = daily.sort_values("date").reset_index(drop=True)

    for _, target_row in daily.iterrows():
        target = target_row["date"]
        prior = daily[daily["date"] < target].copy()
        state3 = _consecutive_last3(prior, target)
        if state3 is None:
            continue

        transitions = _transition_training_rows(daily, target)
        if len(transitions) < MIN_WEATHER_TRANSITIONS:
            continue

        wx_fit = _fit_weather_transition(transitions)
        wx_delta, wx_raw = _wx_delta_for_row(target_row, wx_fit)

        state3_clip = float(np.clip(state3, -MAX_STATE_LOG, MAX_STATE_LOG))
        state_wx = float(np.clip(state3 + wx_delta, -MAX_STATE_LOG, MAX_STATE_LOG))

        base = float(target_row["base"])
        actual = float(target_row["actual"])
        pred_state = base * math.exp(state3_clip)
        pred_wx = base * math.exp(state_wx)
        observed_state = float(target_row["daily_state_obs"])

        b = wx_fit["beta"]
        rows.append({
            "date": target,
            "fields": target_row["fields"],
            "actual": actual,
            "BASE": base,
            "STATE3": pred_state,
            "STATE3+WXdelta": pred_wx,
            "state3": state3,
            "observed_state": observed_state,
            "observed_delta": observed_state - state3,
            "wx_delta": wx_delta,
            "wx_raw_delta": wx_raw,
            "wx_lambda": float(wx_fit["lambda"]),
            "wx_train_n": int(len(transitions)),
            "state_dispersion": float(target_row["state_dispersion"]),
            "wx_d_rad": float(target_row["wx_d_rad"]),
            "wx_d_nightstress": float(target_row["wx_d_nightstress"]),
            "wx_d_winddry": float(target_row["wx_d_winddry"]),
            "beta_rad_std": float(b[0]),
            "beta_nightstress_std": float(b[1]),
            "beta_winddry_std": float(b[2]),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def _mae(a, p):
    return float(np.mean(np.abs(np.asarray(p, dtype=float) - np.asarray(a, dtype=float))))


def _mape(a, p):
    a = np.asarray(a, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean(np.abs(p-a) / np.maximum(np.abs(a), 0.5)) * 100.0)


def _metrics(df: pd.DataFrame):
    a = df["actual"].to_numpy(dtype=float)
    base = df["BASE"].to_numpy(dtype=float)
    state = df["STATE3"].to_numpy(dtype=float)
    wx = df["STATE3+WXdelta"].to_numpy(dtype=float)
    bm, sm, wm = _mae(a, base), _mae(a, state), _mae(a, wx)
    return {
        "n": len(df),
        "base_mae": bm,
        "state_mae": sm,
        "wx_mae": wm,
        "state_vs_base": 100.0*(bm-sm)/bm if bm > 1e-9 else np.nan,
        "wx_vs_state": 100.0*(sm-wm)/sm if sm > 1e-9 else np.nan,
        "wx_vs_base": 100.0*(bm-wm)/bm if bm > 1e-9 else np.nan,
        "state_wins": int(np.sum(np.abs(state-a) < np.abs(base-a))),
        "wx_wins": int(np.sum(np.abs(wx-a) < np.abs(state-a))),
        "base_mape": _mape(a, base),
        "state_mape": _mape(a, state),
        "wx_mape": _mape(a, wx),
        "dir_hit": float(np.mean(np.sign(df["wx_delta"].to_numpy()) == np.sign(df["observed_delta"].to_numpy())) * 100.0),
    }


def _halves(df: pd.DataFrame):
    dates = sorted(df["date"].tolist())
    cut = len(dates)//2
    return df[df["date"].isin(set(dates[:cut]))].copy(), df[df["date"].isin(set(dates[cut:]))].copy()


def _metric_row(label, df):
    m = _metrics(df)
    return {
        "Periood": label,
        "N": m["n"],
        "BASE MAE": m["base_mae"],
        "STATE3 MAE": m["state_mae"],
        "STATE3 vs BASE %": m["state_vs_base"],
        "STATE3+WXdelta MAE": m["wx_mae"],
        "WXdelta vs STATE3 %": m["wx_vs_state"],
        "WXdelta vs BASE %": m["wx_vs_base"],
        "STATE3 võite": m["state_wins"],
        "WXdelta võite": m["wx_wins"],
        "WXdelta suunahitt %": m["dir_hit"],
        "BASE MAPE %": m["base_mape"],
        "STATE3 MAPE %": m["state_mape"],
        "WXdelta MAPE %": m["wx_mape"],
    }


# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------

def main():
    st.set_page_config(page_title="KurgiMootor · coarse crop-state", layout="wide")
    st.title("Crop-state kõigepealt · ilm ennustab ainult state'i muutust")
    st.caption("interval-sum BASE -> STATE3 -> STATE3 + 4p weather-delta · strict OOS · READ ONLY")
    st.info(
        "Iga päeva state-mõõtmine tuleb kolme põllu strict-OOS intervallivigade mediaanist. "
        "STATE3 kasutab ainult kolme eelmist järjestikust päeva. Ilm ei ennusta kaste ega state'i taset: "
        "ta ennustab ainult muutust kahe järjestikuse 4-päevase ilmarežiimi vahel."
    )

    try:
        harvest = db.get_harvest_history(limit=5000)
        intervals = _build_intervals(_events(harvest))
        if intervals.empty:
            st.error("Korjeintervalle ei tekkinud.")
            st.stop()

        earliest = min(intervals["target_date"])
        latest = max(intervals["target_date"])
        weather_from = max(WEATHER_START, earliest - timedelta(days=2*WEATHER_BLOCK_DAYS))
        weather = _measured_weather(db.get_weather_rows(weather_from, latest))

        field_oos = _strict_base_rows(intervals)
        daily = _daily_state_rows(field_oos, weather)
        preds = _strict_predictions(daily)

    except Exception as exc:
        st.exception(exc)
        st.stop()

    if preds.empty:
        st.error("STATE3 + weather-transition strict-OOS päevi ei tekkinud.")
        st.stop()

    full = _metrics(preds)
    first, second = _halves(preds)
    fm, sm = _metrics(first), _metrics(second)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("OOS päevi", len(preds))
    c2.metric("BASE MAE", f"{full['base_mae']:.2f}")
    c3.metric("STATE3 MAE", f"{full['state_mae']:.2f}", delta=f"{full['base_mae']-full['state_mae']:+.2f} kasti")
    c4.metric("STATE3+WXdelta MAE", f"{full['wx_mae']:.2f}", delta=f"{full['state_mae']-full['wx_mae']:+.2f} vs STATE3")

    st.markdown("### 1. Põhitest")
    main_tab = pd.DataFrame([
        {"Variant":"BASE", "MAE":full["base_mae"], "MAPE %":full["base_mape"], "Paranemine BASE suhtes %":0.0},
        {"Variant":"STATE3", "MAE":full["state_mae"], "MAPE %":full["state_mape"], "Paranemine BASE suhtes %":full["state_vs_base"]},
        {"Variant":"STATE3+WXdelta", "MAE":full["wx_mae"], "MAPE %":full["wx_mape"], "Paranemine BASE suhtes %":full["wx_vs_base"]},
    ])
    st.dataframe(main_tab.style.format({"MAE":"{:.2f}", "MAPE %":"{:.1f}", "Paranemine BASE suhtes %":"{:+.1f}%"}), use_container_width=True, hide_index=True)

    state_ok = full["state_vs_base"] > 0
    wx_ok = full["wx_vs_state"] > 0 and fm["wx_vs_state"] > 0 and sm["wx_vs_state"] > 0

    if state_ok and wx_ok:
        st.success(
            f"✅ UUS ARHITEKTUUR ON TOETATUD: STATE3 parandab BASE'i {full['state_vs_base']:+.1f}% ja "
            f"weather-delta parandab STATE3 veel {full['wx_vs_state']:+.1f}%; weather eelis püsib mõlemas ajapooles."
        )
    elif state_ok:
        st.warning(
            f"🟡 COARSE STATE TÖÖTAB ({full['state_vs_base']:+.1f}% vs BASE), "
            "aga weather-delta ei anna stabiilset lisavõitu mõlemas ajapooles."
        )
    elif wx_ok:
        st.warning("🟡 Weather-delta parandab STATE3 mõlemas ajapooles, kuid STATE3 ise ei löö BASE'i.")
    else:
        st.error("❌ EI STATE3 EGA WEATHER-TRANSITION ANNA VAJALIKKU STABIILSET EELIST.")

    st.markdown("### 2. Kõige tähtsam kontroll · kaks ajapoolt")
    halves = pd.DataFrame([_metric_row("I pool", first), _metric_row("II pool", second)])
    st.dataframe(
        halves.style.format({
            "BASE MAE":"{:.2f}", "STATE3 MAE":"{:.2f}", "STATE3 vs BASE %":"{:+.1f}%",
            "STATE3+WXdelta MAE":"{:.2f}", "WXdelta vs STATE3 %":"{:+.1f}%", "WXdelta vs BASE %":"{:+.1f}%",
            "WXdelta suunahitt %":"{:.0f}%", "BASE MAPE %":"{:.1f}", "STATE3 MAPE %":"{:.1f}", "WXdelta MAPE %":"{:.1f}"
        }), use_container_width=True, hide_index=True
    )

    st.markdown("### 3. Päev-päevalt · kas ilm näeb state'i pööret varem?")
    show = preds.copy()
    show["BASE viga"] = show["BASE"] - show["actual"]
    show["STATE3 viga"] = show["STATE3"] - show["actual"]
    show["WXdelta viga"] = show["STATE3+WXdelta"] - show["actual"]
    cols = [
        "date","fields","actual","BASE","STATE3","STATE3+WXdelta","BASE viga","STATE3 viga","WXdelta viga",
        "state3","observed_state","observed_delta","wx_delta","wx_lambda","wx_train_n","state_dispersion",
        "wx_d_rad","wx_d_nightstress","wx_d_winddry","beta_rad_std","beta_nightstress_std","beta_winddry_std"
    ]
    st.dataframe(
        show[cols].style.format({
            "date":lambda x:x.strftime("%d.%m"), "actual":"{:.1f}", "BASE":"{:.1f}", "STATE3":"{:.1f}", "STATE3+WXdelta":"{:.1f}",
            "BASE viga":"{:+.1f}", "STATE3 viga":"{:+.1f}", "WXdelta viga":"{:+.1f}", "state3":"{:+.3f}",
            "observed_state":"{:+.3f}", "observed_delta":"{:+.3f}", "wx_delta":"{:+.3f}", "wx_lambda":"{:g}",
            "wx_train_n":"{:.0f}", "state_dispersion":"{:.3f}", "wx_d_rad":"{:+.2f}", "wx_d_nightstress":"{:+.3f}",
            "wx_d_winddry":"{:+.1f}", "beta_rad_std":"{:+.3f}", "beta_nightstress_std":"{:+.3f}", "beta_winddry_std":"{:+.3f}"
        }), use_container_width=True, hide_index=True
    )

    st.caption(
        "observed_state ja observed_delta on nähtavad alles pärast target-päeva korjet ja on tabelis ainult diagnoosiks. "
        "Prognoos STATE3 kasutab ainult T-3..T-1 varasemaid state-mõõtmisi; weather-model ainult varasemaid üleminekuid."
    )
    st.caption(
        "LEAKAGE LOCK: target-päeva mõõdetud ilma ei kasutata üheski WXdelta sisendis."
    )
    st.caption(
        "Weather delta = T-4..T-1 keskmine minus T-8..T-5 keskmine. Target-päeva mõõdetud ilm on täielikult väljas; "
        "kui see töötab, tuleb järgmine kontroll teha archived forecast-weather'iga."
    )
    st.caption("READ ONLY: db.get_harvest_history + db.get_weather_rows. SciPy puudub ja DB kirjutamisi ei ole.")


if __name__ == "__main__":
    main()
