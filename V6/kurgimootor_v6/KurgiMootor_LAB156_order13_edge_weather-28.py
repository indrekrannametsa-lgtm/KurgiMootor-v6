from __future__ import annotations

"""
KurgiMootor · edge_weather-28
=============================

26.–27.08 FORWARD HOLDOUT · INTERVAL-AWARE · LOCKED RAW WEATHER · READ ONLY

Purpose
-------
This is NOT another discovery LAB. It freezes the mechanism supported by -27
and asks whether it survives the next genuinely awkward operating case:

    25.08  no harvest at all
    26.08  two harvested fields
    27.08  two harvested fields

The test respects field-specific harvest intervals. 25.08 is NOT treated as a
zero-yield observation; it simply adds one calendar growth day to later field
intervals.

Locked mechanism from -27
-------------------------
1) Weatherless interval-sum SEASON BASE, fitted only from information available
   before each target day.
2) Coarse crop state from strict-OOS field residuals.
3) Weather predicts only CHANGE in state from fixed 4-day block deltas:
      current  = T-4 .. T-1
      previous = T-8 .. T-5
   channels are fixed: radiation, night-stress, WINDxDRY.
4) Ridge/GCV rule is unchanged.
5) RAW means NO +/-0.15 weather cap.
6) Final state safety remains +/-0.70.

Gap bridge
----------
The original STATE3 requires the previous three calendar-day state values.
Because 25.08 has no harvest, its state is unobserved. We therefore make the
smallest explicit bridge needed for a real forecast:

    state25_pred = STATE3(25) + WX_RAW(25)

This predicted latent state fills the missing 25.08 slot. No 25.08 yield is
invented.

26.08 prediction:
    STATE3_26 = mean(observed state 23.08,
                     observed state 24.08,
                     predicted state 25.08)
    FINAL_26  = BASE_26 * exp(STATE3_26 + WX_RAW_26)

27.08 is shown two ways, with NO tuning:
    LOCKED  : continue recursively without using 26.08 actual yield.
    SEQ     : after 26.08 harvest is known, replace the latent 26.08 state with
              the robust median state measured from the two harvested fields.
              This is operationally available before 27.08.

The weather transition fit itself is frozen using complete 3-field historical
state days only through 24.08. The two-field 26.08 observation is NEVER used to
refit weather coefficients; it can only update the state anchor for the SEQ
27.08 forecast.

Target-day actual A/B/C values are used only after each prediction for scoring.
Target field identities/orders are used as the test harvest plan.

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

HOLDOUT_GAP_DAY = date(2026, 8, 25)
HOLDOUT_DAYS = [date(2026, 8, 26), date(2026, 8, 27)]
DISCOVERY_CUTOFF = date(2026, 8, 24)


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


def _state_on(daily: pd.DataFrame, dd: date):
    hit = daily[daily["date"] == dd]
    if len(hit) != 1:
        raise RuntimeError(
            f"Historical complete daily state missing or duplicated for {dd:%d.%m}."
        )
    return float(hit.iloc[0]["daily_state_obs"])


def _clip_state(x: float):
    return float(np.clip(float(x), -MAX_STATE_LOG, MAX_STATE_LOG))


def _field_state_from_forward(test: pd.DataFrame):
    vals = np.log(test["actual"].to_numpy(dtype=float) + ABC_EPS) - np.log(
        test["base_pred"].to_numpy(dtype=float) + ABC_EPS
    )
    return vals


def _day_score(actual: float, pred: float):
    err = float(pred - actual)
    ape = abs(err) / max(abs(float(actual)), 0.5) * 100.0
    return err, ape


def _branch_predict(test: pd.DataFrame, state_value: float):
    factor = math.exp(_clip_state(state_value))
    out = test.copy()
    out["pred"] = out["base_pred"].astype(float) * factor
    return out, float(out["pred"].sum())


def _build_holdout(events: List[Event], intervals: pd.DataFrame, weather):
    # Historical strict-OOS state construction is hard-frozen at 24.08.
    hist_intervals = intervals[intervals["target_date"] <= DISCOVERY_CUTOFF].copy()
    field_oos = _strict_base_rows(hist_intervals)
    daily = _daily_state_rows(field_oos, weather)

    # Exact historical anchor required for the missing 25.08 bridge.
    state3_25 = _consecutive_last3(daily, HOLDOUT_GAP_DAY)
    if state3_25 is None:
        raise RuntimeError(
            "Cannot build 25.08 bridge: complete 22.–24.08 STATE3 observations are unavailable."
        )

    # Freeze weather transition fit using only information available before 25.08.
    transitions = _transition_training_rows(daily, HOLDOUT_GAP_DAY)
    if len(transitions) < MIN_WEATHER_TRANSITIONS:
        raise RuntimeError(
            f"Only {len(transitions)} historical weather transitions before 25.08."
        )
    wx_fit = _fit_weather_transition(transitions)

    s23 = _state_on(daily, date(2026, 8, 23))
    s24 = _state_on(daily, date(2026, 8, 24))

    # 25.08: no harvest. Predict latent state only; never fabricate yield.
    wx25 = _raw_wx_for_target(HOLDOUT_GAP_DAY, weather, wx_fit)
    state25_raw = _clip_state(state3_25 + wx25["raw_delta"])
    state25_cap = _clip_state(state3_25 + wx25["cap_delta"])

    # 26.08: field-specific BASE uses actual previous harvest dates and the longer interval.
    test26, _, base_n26 = _fit_forward_base(intervals, HOLDOUT_DAYS[0])
    actual26 = float(test26["actual"].sum())
    base26 = float(test26["base_pred"].sum())

    state3_26_raw = float(np.mean([s23, s24, state25_raw]))
    state3_26_cap = float(np.mean([s23, s24, state25_cap]))
    wx26 = _raw_wx_for_target(HOLDOUT_DAYS[0], weather, wx_fit)
    final_state26_raw = _clip_state(state3_26_raw + wx26["raw_delta"])
    final_state26_cap = _clip_state(state3_26_cap + wx26["cap_delta"])

    field26_raw, pred26_raw = _branch_predict(test26, final_state26_raw)
    field26_cap, pred26_cap = _branch_predict(test26, final_state26_cap)

    # Only NOW, after 26.08 prediction is fixed, derive the two-field observed state.
    field_state26 = _field_state_from_forward(test26)
    observed26 = float(np.median(field_state26))
    dispersion26 = float(np.max(field_state26) - np.min(field_state26)) if len(field_state26) else np.nan

    # 27.08 BASE is operationally allowed to learn from 26.08 because it is now prior data.
    test27, _, base_n27 = _fit_forward_base(intervals, HOLDOUT_DAYS[1])
    actual27 = float(test27["actual"].sum())
    base27 = float(test27["base_pred"].sum())
    wx27 = _raw_wx_for_target(HOLDOUT_DAYS[1], weather, wx_fit)

    # LOCKED branch: no 26.08 actual state is used.
    state3_27_raw_locked = float(np.mean([s24, state25_raw, final_state26_raw]))
    state3_27_cap_locked = float(np.mean([s24, state25_cap, final_state26_cap]))
    state27_raw_locked = _clip_state(state3_27_raw_locked + wx27["raw_delta"])
    state27_cap_locked = _clip_state(state3_27_cap_locked + wx27["cap_delta"])
    field27_raw_locked, pred27_raw_locked = _branch_predict(test27, state27_raw_locked)
    field27_cap_locked, pred27_cap_locked = _branch_predict(test27, state27_cap_locked)

    # SEQ branch: 26.08 two-field state is now known and may update the 27.08 anchor.
    state3_27_raw_seq = float(np.mean([s24, state25_raw, observed26]))
    state3_27_cap_seq = float(np.mean([s24, state25_cap, observed26]))
    state27_raw_seq = _clip_state(state3_27_raw_seq + wx27["raw_delta"])
    state27_cap_seq = _clip_state(state3_27_cap_seq + wx27["cap_delta"])
    field27_raw_seq, pred27_raw_seq = _branch_predict(test27, state27_raw_seq)
    field27_cap_seq, pred27_cap_seq = _branch_predict(test27, state27_cap_seq)

    # Day-level score table.
    rows = []
    for label, target, actual, base, cap, raw, anchor_cap, anchor_raw, wx in [
        ("26.08", HOLDOUT_DAYS[0], actual26, base26, pred26_cap, pred26_raw,
         state3_26_cap, state3_26_raw, wx26),
        ("27.08 LOCKED", HOLDOUT_DAYS[1], actual27, base27, pred27_cap_locked, pred27_raw_locked,
         state3_27_cap_locked, state3_27_raw_locked, wx27),
        ("27.08 SEQ", HOLDOUT_DAYS[1], actual27, base27, pred27_cap_seq, pred27_raw_seq,
         state3_27_cap_seq, state3_27_raw_seq, wx27),
    ]:
        be, bape = _day_score(actual, base)
        ce, cape = _day_score(actual, cap)
        re, rape = _day_score(actual, raw)
        rows.append({
            "Variantpäev": label,
            "date": target,
            "actual": actual,
            "BASE": base,
            "CAP": cap,
            "RAW": raw,
            "BASE viga": be,
            "CAP viga": ce,
            "RAW viga": re,
            "BASE APE %": bape,
            "CAP APE %": cape,
            "RAW APE %": rape,
            "state3 CAP": anchor_cap,
            "state3 RAW": anchor_raw,
            "WX cap delta": wx["cap_delta"],
            "WX raw delta": wx["raw_delta"],
        })
    day_scores = pd.DataFrame(rows)

    # Main 2-day sequential comparison: 26 forecast + 27 forecast after legal 26 update.
    actual_seq = np.asarray([actual26, actual27], dtype=float)
    base_seq = np.asarray([base26, base27], dtype=float)
    cap_seq = np.asarray([pred26_cap, pred27_cap_seq], dtype=float)
    raw_seq = np.asarray([pred26_raw, pred27_raw_seq], dtype=float)

    summary = {
        "base_mae": float(np.mean(np.abs(base_seq - actual_seq))),
        "cap_mae": float(np.mean(np.abs(cap_seq - actual_seq))),
        "raw_mae": float(np.mean(np.abs(raw_seq - actual_seq))),
        "base_mape": float(np.mean(np.abs(base_seq-actual_seq)/np.maximum(actual_seq,0.5))*100.0),
        "cap_mape": float(np.mean(np.abs(cap_seq-actual_seq)/np.maximum(actual_seq,0.5))*100.0),
        "raw_mape": float(np.mean(np.abs(raw_seq-actual_seq)/np.maximum(actual_seq,0.5))*100.0),
        "raw_wins_base": int(np.sum(np.abs(raw_seq-actual_seq) < np.abs(base_seq-actual_seq))),
        "raw_wins_cap": int(np.sum(np.abs(raw_seq-actual_seq) < np.abs(cap_seq-actual_seq))),
    }

    # Field-level operational table.
    field_rows = []
    for target, test, final_cap, final_raw, branch in [
        (HOLDOUT_DAYS[0], test26, final_state26_cap, final_state26_raw, "26.08"),
        (HOLDOUT_DAYS[1], test27, state27_cap_seq, state27_raw_seq, "27.08 SEQ"),
    ]:
        for _, r in test.sort_values(["order", "field"]).iterrows():
            basef = float(r["base_pred"])
            actualf = float(r["actual"])
            capf = basef * math.exp(final_cap)
            rawf = basef * math.exp(final_raw)
            field_rows.append({
                "Päev": branch,
                "Põld": int(r["field"]),
                "Jrk": int(r["order"]),
                "Eelmine korje": r["start_date"],
                "Kalendriintervall p": int(r["calendar_interval_days"]),
                "Order-adjusted growth p": float(r["growth"]),
                "BASE": basef,
                "CAP": capf,
                "RAW": rawf,
                "Tegelik": actualf,
                "BASE viga": basef-actualf,
                "CAP viga": capf-actualf,
                "RAW viga": rawf-actualf,
            })
    field_table = pd.DataFrame(field_rows)

    bridge = pd.DataFrame([
        {"Päev":"23.08", "Tüüp":"observed 3-field", "state":s23},
        {"Päev":"24.08", "Tüüp":"observed 3-field", "state":s24},
        {"Päev":"25.08", "Tüüp":"predicted RAW gap bridge", "state":state25_raw},
        {"Päev":"25.08", "Tüüp":"predicted CAP gap bridge", "state":state25_cap},
        {"Päev":"26.08", "Tüüp":"predicted RAW before harvest", "state":final_state26_raw},
        {"Päev":"26.08", "Tüüp":"observed 2-field after harvest", "state":observed26},
    ])

    diagnostics = {
        "daily": daily,
        "transitions": transitions,
        "wx_fit": wx_fit,
        "wx25": wx25,
        "wx26": wx26,
        "wx27": wx27,
        "state3_25": state3_25,
        "state25_raw": state25_raw,
        "state25_cap": state25_cap,
        "observed26": observed26,
        "dispersion26": dispersion26,
        "base_n26": base_n26,
        "base_n27": base_n27,
        "pred27_raw_locked": pred27_raw_locked,
        "pred27_raw_seq": pred27_raw_seq,
        "pred27_cap_locked": pred27_cap_locked,
        "pred27_cap_seq": pred27_cap_seq,
        "state27_raw_locked": state27_raw_locked,
        "state27_raw_seq": state27_raw_seq,
        "state27_cap_locked": state27_cap_locked,
        "state27_cap_seq": state27_cap_seq,
    }

    return summary, day_scores, field_table, bridge, diagnostics


# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------

def main():
    st.set_page_config(page_title="KurgiMootor · 26–27.08 forward holdout", layout="wide")
    st.title("26.–27.08 päris forward-holdout")
    st.caption("field-specific interval · 25.08 no-harvest bridge · locked -27 RAW weather · READ ONLY")

    st.info(
        "See test ei vali ühtegi uut akent, feature'it, ridge'i ega cap'i. "
        "25.08 ei ole nullsaak: see on vaatluseta kasvupäev. BASE arvutab igale 26.–27.08 põllule "
        "tema päris eelmisest korjest uue pikema intervalli. Weather-fit on külmutatud 24.08 peal."
    )

    try:
        harvest = db.get_harvest_history(limit=5000)
        events = _events(harvest)
        intervals = _build_intervals(events)
        if intervals.empty:
            st.error("Korjeintervalle ei tekkinud.")
            st.stop()

        event25 = [e for e in events if e.day == HOLDOUT_GAP_DAY]
        target_counts = {dd: len([e for e in events if e.day == dd]) for dd in HOLDOUT_DAYS}

        earliest = min(intervals["target_date"])
        # Need measured weather only through 26.08; target-day weather is excluded by construction.
        latest_weather_needed = HOLDOUT_DAYS[-1] - timedelta(days=1)
        weather_from = max(WEATHER_START, earliest - timedelta(days=2*WEATHER_BLOCK_DAYS))
        weather = _measured_weather(db.get_weather_rows(weather_from, latest_weather_needed))

        summary, day_scores, field_table, bridge, diag = _build_holdout(events, intervals, weather)

    except Exception as exc:
        st.exception(exc)
        st.stop()

    if event25:
        st.error(
            f"25.08 ei ole DB järgi tühi: leidsin {len(event25)} korjerida. Holdout'i eeldus ei vasta andmetele."
        )
    else:
        st.success("✅ 25.08 kontroll: 0 korjerida. Päeva ei käsitleta nullsaagina.")

    if target_counts[HOLDOUT_DAYS[0]] != 2 or target_counts[HOLDOUT_DAYS[1]] != 2:
        st.warning(
            f"DB korjeridade arv: 26.08 = {target_counts[HOLDOUT_DAYS[0]]}, "
            f"27.08 = {target_counts[HOLDOUT_DAYS[1]]}. Ootasime 2 + 2."
        )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Holdout päevi", "2")
    c2.metric("BASE MAE", f"{summary['base_mae']:.2f}")
    c3.metric("CAP MAE", f"{summary['cap_mae']:.2f}")
    c4.metric("RAW MAE", f"{summary['raw_mae']:.2f}", delta=f"{summary['cap_mae']-summary['raw_mae']:+.2f} vs CAP")

    st.markdown("### 1. Põhitest · 26 + 27 sequential")
    main_tab = pd.DataFrame([
        {"Variant":"BASE", "MAE":summary["base_mae"], "MAPE %":summary["base_mape"]},
        {"Variant":"WX CAP ±0.15", "MAE":summary["cap_mae"], "MAPE %":summary["cap_mape"]},
        {"Variant":"WX RAW", "MAE":summary["raw_mae"], "MAPE %":summary["raw_mape"]},
    ])
    st.dataframe(
        main_tab.style.format({"MAE":"{:.2f}", "MAPE %":"{:.1f}"}),
        use_container_width=True,
        hide_index=True,
    )

    if summary["raw_mae"] < summary["base_mae"] and summary["raw_mae"] < summary["cap_mae"]:
        st.success(
            f"✅ RAW läbis selle 2-päevase forward-kontrolli: MAE {summary['raw_mae']:.2f}; "
            f"BASE {summary['base_mae']:.2f}; CAP {summary['cap_mae']:.2f}. "
            f"RAW võidab BASE'i {summary['raw_wins_base']}/2 ja CAP-i {summary['raw_wins_cap']}/2 päeval."
        )
    elif summary["raw_mae"] < summary["base_mae"]:
        st.warning(
            "🟡 RAW lööb BASE'i, kuid ei löö CAP-i selles kahes päevas. N=2 on liiga väike uueks otsuseks."
        )
    else:
        st.error(
            "❌ RAW ei löö selles forward-kontrollis BASE'i. -27 tulemust ei tohi veel üldistada."
        )

    st.caption(
        "N=2 on meelega väike päris holdout. Seda ei kasutata ühegi parameetri ümbervalimiseks."
    )

    st.markdown("### 2. Päev-päevalt")
    st.dataframe(
        day_scores.style.format({
            "date":lambda x:x.strftime("%d.%m"),
            "actual":"{:.1f}", "BASE":"{:.1f}", "CAP":"{:.1f}", "RAW":"{:.1f}",
            "BASE viga":"{:+.1f}", "CAP viga":"{:+.1f}", "RAW viga":"{:+.1f}",
            "BASE APE %":"{:.1f}", "CAP APE %":"{:.1f}", "RAW APE %":"{:.1f}",
            "state3 CAP":"{:+.3f}", "state3 RAW":"{:+.3f}",
            "WX cap delta":"{:+.3f}", "WX raw delta":"{:+.3f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "27.08 LOCKED ei kasuta 26.08 tegelikku state'i üldse. 27.08 SEQ kasutab pärast 26.08 korjet "
        "selle kahe põllu mediaan-state'i, kuid weather-koefitsiente ei refitita."
    )

    st.markdown("### 3. Põllu kaupa · intervall on päriselt sees")
    st.dataframe(
        field_table.style.format({
            "Eelmine korje":lambda x:x.strftime("%d.%m"),
            "Order-adjusted growth p":"{:.2f}",
            "BASE":"{:.2f}", "CAP":"{:.2f}", "RAW":"{:.2f}", "Tegelik":"{:.2f}",
            "BASE viga":"{:+.2f}", "CAP viga":"{:+.2f}", "RAW viga":"{:+.2f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Kalendriintervall = target-kuupäev miinus sama põllu eelmine päris korje. "
        "Order-adjusted growth lisab -27-ga sama korjejärjekorra ajaparanduse."
    )

    st.markdown("### 4. 25.08 vaatluseta päeva bridge")
    st.dataframe(
        bridge.style.format({"state":"{:+.3f}"}),
        use_container_width=True,
        hide_index=True,
    )

    b1, b2, b3, b4 = st.columns(4)
    b1.metric("WX train N", int(len(diag["transitions"])))
    b2.metric("WX lambda", f"{diag['wx_fit']['lambda']:g}")
    b3.metric("26.08 2-field state", f"{diag['observed26']:+.3f}")
    b4.metric("26.08 state dispersion", f"{diag['dispersion26']:.3f}")

    with st.expander("Kontrolliks · frozen weather detail"):
        b = diag["wx_fit"]["beta"]
        wx_tab = pd.DataFrame([
            {
                "Target":"25.08", "raw delta":diag["wx25"]["raw_delta"],
                "Δ RAD":diag["wx25"]["wx_d_rad"], "Δ nightstress":diag["wx25"]["wx_d_nightstress"],
                "Δ WINDxDRY":diag["wx25"]["wx_d_winddry"],
            },
            {
                "Target":"26.08", "raw delta":diag["wx26"]["raw_delta"],
                "Δ RAD":diag["wx26"]["wx_d_rad"], "Δ nightstress":diag["wx26"]["wx_d_nightstress"],
                "Δ WINDxDRY":diag["wx26"]["wx_d_winddry"],
            },
            {
                "Target":"27.08", "raw delta":diag["wx27"]["raw_delta"],
                "Δ RAD":diag["wx27"]["wx_d_rad"], "Δ nightstress":diag["wx27"]["wx_d_nightstress"],
                "Δ WINDxDRY":diag["wx27"]["wx_d_winddry"],
            },
        ])
        st.dataframe(
            wx_tab.style.format({
                "raw delta":"{:+.3f}", "Δ RAD":"{:+.2f}",
                "Δ nightstress":"{:+.3f}", "Δ WINDxDRY":"{:+.1f}",
            }),
            use_container_width=True,
            hide_index=True,
        )
        st.write(
            f"Standardized beta: RAD {float(b[0]):+.3f}, nightstress {float(b[1]):+.3f}, "
            f"WINDxDRY {float(b[2]):+.3f}. Weather fit is frozen before 25.08."
        )
        st.write(
            f"27.08 RAW: LOCKED {diag['pred27_raw_locked']:.1f} vs SEQ {diag['pred27_raw_seq']:.1f}. "
            f"27.08 CAP: LOCKED {diag['pred27_cap_locked']:.1f} vs SEQ {diag['pred27_cap_seq']:.1f}."
        )

    st.caption(
        "LEAKAGE LOCK: target-day measured weather is never used. 26.08 uses weather only through 25.08; "
        "27.08 only through 26.08. Target actual A/B/C enters only scoring, except that after 26.08 is over "
        "its two-field state may legally update the separate 27.08 SEQ anchor."
    )
    st.caption(
        "READ ONLY: db.get_harvest_history + db.get_weather_rows. No DB writes, no production snapshots, no SciPy."
    )


if __name__ == "__main__":
    main()
