from __future__ import annotations

"""
KurgiMootor · edge_weather-30 FIXED
===================================

SELF-CONTAINED forward holdout test.
No sibling LAB files are required in the Streamlit repository.

Question:
Does app-128's production plant index frozen on 24.08 improve the
interval-aware BASE for the real 26.–27.08 holdout, and does the already
locked conservative weather-delta add anything on top?

Variants:
1) BASE
2) BASE × frozen PI24 (field by field)
3) BASE × frozen PI24 × WX-CAP
4) old STATE3+WX-CAP only as a reference row

Locks:
- plant index reconstructed only from 15.–24.08 locked app-128 raw anchors
- 25.08 is no-harvest, never zero yield
- weather transition mechanism/ridge/cap unchanged from edge_weather-28
- 26.–27.08 actual A+B+C used only for scoring
- no new tuning on holdout
- READ ONLY; no DB writes
"""
import json
from typing import Any, Dict, Tuple

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

# Production plant-index freeze constants (app-128 exact rule)
YEAR = 2026
PLANT_INDEX_START = date(YEAR, 8, 15)
FREEZE_DAY = date(YEAR, 8, 24)
PLANT_INDEX_ALPHA = 0.70
PLANT_INDEX_MIN = 0.50
RAW_SETTING_KEY = f"plant_index_raw_forecasts_{YEAR}"

def _i(v: Any) -> Optional[int]:
    try:
        return int(v)
    except Exception:
        return None


def _load_raw_setting() -> Tuple[Dict[str, dict], str]:
    """Read the exact app-128 raw anchor map without writing anything."""
    if not hasattr(db, "get_app_setting"):
        return {}, "app_setting API puudub"
    try:
        raw = db.get_app_setting(RAW_SETTING_KEY, "")
    except Exception as exc:
        return {}, f"app_setting lugemine ebaõnnestus: {exc}"
    if not raw:
        return {}, "app_setting on tühi"
    try:
        payload = json.loads(raw)
    except Exception as exc:
        return {}, f"app_setting JSON vigane: {exc}"
    if not isinstance(payload, dict):
        return {}, "app_setting ei ole dict"
    cleaned = {str(k): v for k, v in payload.items() if isinstance(v, dict)}
    return cleaned, "app-128 plant_index_raw_forecasts_2026"

def _normalise_harvest(rows: List[dict]) -> pd.DataFrame:
    data = []
    for r in rows or []:
        d = _d(r.get("harvest_date"))
        f = _i(r.get("field_no"))
        order = _i(r.get("harvest_order")) or 99
        total = _f(r.get("total"))
        a, b, c = _f(r.get("a")), _f(r.get("b")), _f(r.get("c"))
        if d is None or f is None or total is None or not (1 <= f <= 14):
            continue
        abc = (a + b + c) if None not in (a, b, c) else None
        data.append({
            "date": d,
            "field": f,
            "order": order,
            "total": total,
            "abc": abc,
            "quality": str(r.get("data_quality") or "").strip().lower(),
        })
    if not data:
        return pd.DataFrame(columns=["date", "field", "order", "total", "abc", "quality"])
    return pd.DataFrame(data).sort_values(["date", "order", "field"]).reset_index(drop=True)


def _reconstruct_index(hdf: pd.DataFrame, raw_map: Dict[str, dict], source_label: str):
    idx = {f: 1.0 for f in range(1, 15)}
    last_event: Dict[int, dict] = {}
    trace = []
    eligible = 0
    used = 0
    missing = []

    hist = hdf[(hdf["date"] >= PLANT_INDEX_START) & (hdf["date"] <= FREEZE_DAY)].copy()
    for _, r in hist.iterrows():
        if str(r["quality"]) in {"hinnanguline", "ligikaudne"}:
            continue
        eligible += 1
        d = r["date"]
        f = int(r["field"])
        actual_total = float(r["total"])
        key = f"{d.isoformat()}|{f}"
        rec = raw_map.get(key)
        if not isinstance(rec, dict):
            missing.append(key)
            continue
        raw_total = _f(rec.get("raw_total"))
        if raw_total is None or raw_total <= 0:
            missing.append(key)
            continue

        ratio = actual_total / raw_total
        signal = max(PLANT_INDEX_MIN, min(1.0, ratio))
        old = float(idx[f])
        new = (1.0 - PLANT_INDEX_ALPHA) * old + PLANT_INDEX_ALPHA * signal
        new = max(PLANT_INDEX_MIN, min(1.0, new))
        idx[f] = new
        used += 1
        ev = {
            "date": d,
            "field": f,
            "order": int(r["order"]),
            "actual_total": actual_total,
            "raw_total": raw_total,
            "actual/raw": ratio,
            "signal": signal,
            "index_before": old,
            "index_after": new,
            "raw_source": str(rec.get("source") or source_label),
            "captured_at": str(rec.get("captured_at") or ""),
        }
        trace.append(ev)
        last_event[f] = ev

    return idx, last_event, pd.DataFrame(trace), eligible, used, missing

def _score(actual: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return {
        "MAE": float(np.mean(np.abs(pred - actual))),
        "MAPE %": float(np.mean(np.abs(pred - actual) / np.maximum(np.abs(actual), 0.5)) * 100.0),
    }


def _fmt_day(v):
    try:
        return v.strftime("%d.%m")
    except Exception:
        return str(v)


def main():
    st.set_page_config(
        page_title="KurgiMootor · PI24 × interval-base × weather",
        layout="wide",
    )
    st.title("24.08 frozen taimeindeks × intervall × ilm")
    st.caption("26.–27.08 true forward holdout · no new tuning · READ ONLY")

    st.info(
        "Üks test, kolm varianti. BASE on sama interval-aware BASE nagu -28. "
        "PI24 on app-128 päris production taimeindeks, külmutatud 24.08. "
        "WX-CAP on sama lukustatud konservatiivne weather-delta ±0.15. "
        "STATE3 ei lähe PI-varianti juurde — muidu loeksime crop-state'i kaks korda."
    )

    try:

        # -------------------------------------------------------------
        # 1) Exact frozen production plant index through 24.08 only.
        # -------------------------------------------------------------
        harvest = db.get_harvest_history(limit=5000)
        hdf = _normalise_harvest(harvest)
        raw_map, raw_status = _load_raw_setting()
        idx, last_event, trace, eligible, used, missing = _reconstruct_index(
            hdf, raw_map, raw_status
        )

        if eligible <= 0:
            raise RuntimeError("24.08 taimeindeksi rekonstruktsioonis pole ühtegi sündmust.")
        if used != eligible or missing:
            raise RuntimeError(
                f"Exact PI reconstruction incomplete: {used}/{eligible}; missing={len(missing)}. "
                "Fallbacki selles otsustestis ei kasutata."
            )
        if not trace.empty:
            bad = [d for d in trace["date"].tolist() if d > FREEZE_DAY]
            if bad:
                raise RuntimeError(f"PI leakage lock failed: {bad}")

        # -------------------------------------------------------------
        # 2) Re-run the unchanged locked forward machinery.
        #    This gives field-specific interval BASE and the locked
        #    weather transition deltas. No new model choice here.
        # -------------------------------------------------------------
        events = _events(harvest)
        intervals = _build_intervals(events)
        if intervals.empty:
            raise RuntimeError("Korjeintervalle ei tekkinud.")

        event25 = [e for e in events if e.day == HOLDOUT_GAP_DAY]
        target_counts = {
            dd: len([e for e in events if e.day == dd])
            for dd in HOLDOUT_DAYS
        }

        earliest = min(intervals["target_date"])
        latest_weather_needed = HOLDOUT_DAYS[-1] - timedelta(days=1)
        weather_from = max(
            WEATHER_START,
            earliest - timedelta(days=2 * WEATHER_BLOCK_DAYS),
        )
        weather = _measured_weather(
            db.get_weather_rows(weather_from, latest_weather_needed)
        )

        old_summary, old_days, field_old, bridge, diag = _build_holdout(
            events, intervals, weather
        )

        if event25:
            raise RuntimeError(
                f"25.08 holdout assumption failed: DB has {len(event25)} harvest rows."
            )
        if target_counts.get(HOLDOUT_DAYS[0]) != 2 or target_counts.get(HOLDOUT_DAYS[1]) != 2:
            raise RuntimeError(
                "Holdout field count changed: "
                f"26.08={target_counts.get(HOLDOUT_DAYS[0])}, "
                f"27.08={target_counts.get(HOLDOUT_DAYS[1])}."
            )

        # -------------------------------------------------------------
        # 3) New composition only: field BASE × frozen PI24, then
        #    optional short-term weather delta. No STATE3.
        # -------------------------------------------------------------
        wx_by_day = {
            HOLDOUT_DAYS[0]: float(diag["wx26"]["cap_delta"]),
            HOLDOUT_DAYS[1]: float(diag["wx27"]["cap_delta"]),
        }

        work = field_old.copy()
        work["date"] = work["Päev"].map({
            "26.08": HOLDOUT_DAYS[0],
            "27.08 SEQ": HOLDOUT_DAYS[1],
        })
        if work["date"].isna().any():
            raise RuntimeError("locked field table has an unexpected branch label.")

        work["PI24"] = work["Põld"].map(lambda f: float(idx[int(f)]))
        work["PI last event"] = work["Põld"].map(
            lambda f: (last_event.get(int(f)) or {}).get("date")
        )
        work["PI last actual/raw"] = work["Põld"].map(
            lambda f: (last_event.get(int(f)) or {}).get("actual/raw")
        )
        work["PI pred"] = work["BASE"].astype(float) * work["PI24"].astype(float)
        work["WX cap delta"] = work["date"].map(wx_by_day).astype(float)
        work["WX factor"] = np.exp(work["WX cap delta"].astype(float))
        work["PI+WX pred"] = work["PI pred"] * work["WX factor"]

        work["BASE viga"] = work["BASE"].astype(float) - work["Tegelik"].astype(float)
        work["PI viga"] = work["PI pred"] - work["Tegelik"].astype(float)
        work["PI+WX viga"] = work["PI+WX pred"] - work["Tegelik"].astype(float)

        day = (
            work.groupby("date", as_index=False)
            .agg(
                fields=("Põld", lambda s: ",".join(str(int(x)) for x in s)),
                actual=("Tegelik", "sum"),
                BASE=("BASE", "sum"),
                PI24=("PI pred", "sum"),
                PI24_WX=("PI+WX pred", "sum"),
                wx_cap_delta=("WX cap delta", "first"),
            )
            .sort_values("date")
            .reset_index(drop=True)
        )
        day["BASE viga"] = day["BASE"] - day["actual"]
        day["PI24 viga"] = day["PI24"] - day["actual"]
        day["PI24+WX viga"] = day["PI24_WX"] - day["actual"]
        day["BASE APE %"] = np.abs(day["BASE viga"]) / np.maximum(day["actual"], 0.5) * 100.0
        day["PI24 APE %"] = np.abs(day["PI24 viga"]) / np.maximum(day["actual"], 0.5) * 100.0
        day["PI24+WX APE %"] = np.abs(day["PI24+WX viga"]) / np.maximum(day["actual"], 0.5) * 100.0

        actual = day["actual"].to_numpy(dtype=float)
        s_base = _score(actual, day["BASE"].to_numpy(dtype=float))
        s_pi = _score(actual, day["PI24"].to_numpy(dtype=float))
        s_piwx = _score(actual, day["PI24_WX"].to_numpy(dtype=float))

        summary = pd.DataFrame([
            {"Variant": "BASE", **s_base},
            {"Variant": "BASE × frozen PI24", **s_pi},
            {"Variant": "BASE × frozen PI24 × WX-CAP", **s_piwx},
            {
                "Variant": "REFERENCE: STATE3+WX-CAP",
                "MAE": float(old_summary["cap_mae"]),
                "MAPE %": float(old_summary["cap_mape"]),
            },
        ])
        summary["Paranemine BASE suhtes %"] = (
            (s_base["MAE"] - summary["MAE"]) / s_base["MAE"] * 100.0
        )

    except Exception as exc:
        st.exception(exc)
        st.stop()

    # -----------------------------------------------------------------
    # UI: deliberately short. One decision table, one day table, one
    # field audit table. No discovery kitchen sink.
    # -----------------------------------------------------------------
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PI freeze", "24.08")
    c2.metric("PI reconstruction", f"{used}/{eligible}")
    c3.metric("Holdout", "26–27.08")
    c4.metric("25.08 harvest", "0 rida")

    st.markdown("### 1. Otsustabel · kas frozen taimeindeks töötab päris holdout'is?")
    st.dataframe(
        summary.style.format({
            "MAE": "{:.2f}",
            "MAPE %": "{:.1f}",
            "Paranemine BASE suhtes %": "{:+.1f}%",
        }),
        use_container_width=True,
        hide_index=True,
    )

    pi_beats = s_pi["MAE"] < s_base["MAE"]
    wx_adds = s_piwx["MAE"] < s_pi["MAE"]

    if pi_beats and wx_adds:
        st.success(
            f"✅ FROZEN PI24 LÖÖB BASE'i JA ILM LISAB VEEL: "
            f"MAE {s_base['MAE']:.2f} → {s_pi['MAE']:.2f} → {s_piwx['MAE']:.2f}. "
            "See toetab arhitektuuri: aeglane field-state = production PI, lühike muutus = weather-delta."
        )
    elif pi_beats:
        st.success(
            f"✅ FROZEN PI24 LÖÖB BASE'i: MAE {s_base['MAE']:.2f} → {s_pi['MAE']:.2f}. "
            f"Weather-delta {'ei lisa' if not wx_adds else 'lisab'} selles 2-päevases kontrollis."
        )
    else:
        st.error(
            f"❌ FROZEN PI24 EI LÖÖ BASE'i: MAE {s_base['MAE']:.2f} → {s_pi['MAE']:.2f}. "
            "Siis ei tohi taimeindeksit uue arhitektuuri ankruks lihtsalt eeldada."
        )

    st.markdown("### 2. Päev-päevalt")
    show_day = day.rename(columns={
        "date": "Päev",
        "fields": "Põllud",
        "actual": "Tegelik ABC",
        "PI24": "BASE×PI24",
        "PI24_WX": "BASE×PI24×WX",
        "wx_cap_delta": "WX cap delta",
    })
    st.dataframe(
        show_day.style.format({
            "Päev": _fmt_day,
            "Tegelik ABC": "{:.1f}",
            "BASE": "{:.1f}",
            "BASE×PI24": "{:.1f}",
            "BASE×PI24×WX": "{:.1f}",
            "WX cap delta": "{:+.3f}",
            "BASE viga": "{:+.1f}",
            "PI24 viga": "{:+.1f}",
            "PI24+WX viga": "{:+.1f}",
            "BASE APE %": "{:.1f}",
            "PI24 APE %": "{:.1f}",
            "PI24+WX APE %": "{:.1f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### 3. Põllu kaupa · siin on intervall ja päris 24.08 indeks")
    field_show = work[[
        "date", "Põld", "Jrk", "Eelmine korje", "Kalendriintervall p",
        "Order-adjusted growth p", "BASE", "PI24", "PI last event",
        "PI last actual/raw", "PI pred", "WX cap delta", "PI+WX pred", "Tegelik",
        "BASE viga", "PI viga", "PI+WX viga",
    ]].copy()
    field_show = field_show.rename(columns={
        "date": "Päev",
        "PI pred": "BASE×PI24",
        "PI+WX pred": "BASE×PI24×WX",
    })
    st.dataframe(
        field_show.style.format({
            "Päev": _fmt_day,
            "Eelmine korje": _fmt_day,
            "Order-adjusted growth p": "{:.2f}",
            "BASE": "{:.2f}",
            "PI24": "{:.3f}",
            "PI last event": _fmt_day,
            "PI last actual/raw": lambda v: "—" if pd.isna(v) else f"{float(v):.3f}",
            "BASE×PI24": "{:.2f}",
            "WX cap delta": "{:+.3f}",
            "BASE×PI24×WX": "{:.2f}",
            "Tegelik": "{:.2f}",
            "BASE viga": "{:+.2f}",
            "PI viga": "{:+.2f}",
            "PI+WX viga": "{:+.2f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.success(
        "🔒 LEAKAGE LOCK: PI kasutab ainult 15.–24.08 sündmusi; 25.08 ei ole nullsaak; "
        "weather-fit/cap on lukustatud reegli järgi muutmata; 26.–27.08 actual kasutatakse ainult skooriks."
    )
    st.caption(
        "27.08 BASE järgib sama lukustatud operatiivset reeglit: 26.08 on selleks hetkeks juba minevik ja võib BASE fit'i sisse minna. "
        "PI24 ise jääb mõlema päeva jaoks 24.08 peale külmutatuks."
    )
    st.caption(
        "READ ONLY. See LAB ei otsi uut akent, ridge'i, cap'i, taimeindeksi alpha't ega ühtegi muud parameetrit."
    )


if __name__ == "__main__":
    main()
