from __future__ import annotations

"""
KurgiMootor · edge_weather-31
=============================

PRODUCTION TAIMEINDEKS · HISTORICAL WALK-FORWARD AUDIT · READ ONLY

One narrow question:
    Did app-128's existing production plant index improve the unchanged
    interval-aware BASE on dates that were still genuinely in the future at
    each prediction point?

Locks
-----
- exact production plant-index rule, unchanged:
      start 15.08 at 1.00
      signal = clip(actual_total / locked_raw_total, 0.50, 1.00)
      new    = 0.30 * old + 0.70 * signal
      index  = clip(new, 0.50, 1.00)
- each target day uses ONLY PI events strictly before that target day
- BASE is fitted ONLY on harvest intervals strictly before the target day
- target-day actual is used only after prediction for scoring
- weather variant uses the already locked conservative pre-target WX-delta
  mechanism and +/-0.15 cap; no new window/ridge/cap search
- 25.08 is not part of this historical audit and is never treated as zero yield
- no 26.-27.08 result is used in the primary historical score
- READ ONLY: no DB writes, no production settings changes

Important sample-size fact
--------------------------
Production PI starts only on 15.08. Therefore an honest pre-holdout audit cannot
manufacture 10-20 mature historical days. 15.-19.08 are warm-up days for the
14-field rotation; the first target day on which every target field has a prior
PI event is 20.08. The UI reports BOTH the full walk-forward period and the
mature subset separately.
"""
import json
from typing import Any, Dict, Tuple

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



# =====================================================================
# Exact production plant-index historical walk-forward
# =====================================================================

YEAR = 2026
PLANT_INDEX_START = date(YEAR, 8, 15)
AUDIT_END = date(YEAR, 8, 24)  # hard pre-holdout stop
PLANT_INDEX_ALPHA = 0.70
PLANT_INDEX_MIN = 0.50
RAW_SETTING_KEY = f"plant_index_raw_forecasts_{YEAR}"


def _i(v: Any) -> Optional[int]:
    try:
        return int(v)
    except Exception:
        return None


def _load_raw_setting() -> Tuple[Dict[str, dict], str]:
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
    return {str(k): v for k, v in payload.items() if isinstance(v, dict)}, "app-128 raw-lock setting"


def _normalise_harvest(rows: List[dict]) -> pd.DataFrame:
    data = []
    for r in rows or []:
        dd = _d(r.get("harvest_date"))
        ff = _i(r.get("field_no"))
        order = _i(r.get("harvest_order")) or 99
        total = _f(r.get("total"))
        a, b, c = _f(r.get("a")), _f(r.get("b")), _f(r.get("c"))
        if dd is None or ff is None or total is None or not (1 <= ff <= 14):
            continue
        abc = (a + b + c) if None not in (a, b, c) else None
        data.append({
            "date": dd,
            "field": int(ff),
            "order": int(order),
            "total": float(total),
            "abc": abc,
            "quality": str(r.get("data_quality") or "").strip().lower(),
        })
    cols = ["date", "field", "order", "total", "abc", "quality"]
    if not data:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(data).sort_values(["date", "order", "field"]).reset_index(drop=True)


def _capture_date(rec: dict) -> Optional[date]:
    raw = str(rec.get("captured_at") or "").strip()
    if not raw:
        return None
    try:
        return date.fromisoformat(raw[:10])
    except Exception:
        return None


def _pi_before_target(
    hdf: pd.DataFrame,
    raw_map: Dict[str, dict],
    target: date,
):
    """Rebuild exact production PI state using only events BEFORE target."""
    idx = {f: 1.0 for f in range(1, 15)}
    last_event: Dict[int, dict] = {}
    trace = []
    missing = []
    late_capture = []

    hist = hdf[
        (hdf["date"] >= PLANT_INDEX_START)
        & (hdf["date"] < target)
    ].copy()

    for _, r in hist.iterrows():
        if str(r["quality"]) in {"hinnanguline", "ligikaudne"}:
            continue
        dd = r["date"]
        ff = int(r["field"])
        key = f"{dd.isoformat()}|{ff}"
        rec = raw_map.get(key)
        if not isinstance(rec, dict):
            missing.append(key)
            continue
        raw_total = _f(rec.get("raw_total"))
        if raw_total is None or raw_total <= 0:
            missing.append(key)
            continue

        cap_day = _capture_date(rec)
        # A raw lock created after its own harvest date is suspicious. Flag it
        # loudly instead of silently accepting a hindsight anchor.
        if cap_day is not None and cap_day > dd:
            late_capture.append((key, str(rec.get("captured_at") or "")))

        actual_total = float(r["total"])
        ratio = actual_total / float(raw_total)
        signal = max(PLANT_INDEX_MIN, min(1.0, ratio))
        old = float(idx[ff])
        new = (1.0 - PLANT_INDEX_ALPHA) * old + PLANT_INDEX_ALPHA * signal
        new = max(PLANT_INDEX_MIN, min(1.0, new))
        idx[ff] = new
        ev = {
            "date": dd,
            "field": ff,
            "actual_total": actual_total,
            "raw_total": float(raw_total),
            "actual/raw": float(ratio),
            "signal": float(signal),
            "index_before": old,
            "index_after": new,
            "captured_at": str(rec.get("captured_at") or ""),
            "source": str(rec.get("source") or ""),
        }
        last_event[ff] = ev
        trace.append(ev)

    return idx, last_event, pd.DataFrame(trace), missing, late_capture


def _score(actual: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    if len(actual) == 0:
        return {"N": 0, "MAE": np.nan, "MAPE %": np.nan, "Bias": np.nan, "Wins": 0}
    err = pred - actual
    return {
        "N": int(len(actual)),
        "MAE": float(np.mean(np.abs(err))),
        "MAPE %": float(np.mean(np.abs(err) / np.maximum(np.abs(actual), 0.5)) * 100.0),
        "Bias": float(np.mean(err)),
    }


def _variant_row(name: str, g: pd.DataFrame, col: str, base_mae: Optional[float] = None) -> dict:
    sc = _score(g["actual"].to_numpy(float), g[col].to_numpy(float))
    row = {"Variant": name, **sc}
    if base_mae is not None and math.isfinite(base_mae) and base_mae > 0:
        row["Paranemine BASE suhtes %"] = (base_mae - sc["MAE"]) / base_mae * 100.0
    else:
        row["Paranemine BASE suhtes %"] = 0.0
    return row


def _fmt_day(v):
    try:
        return v.strftime("%d.%m")
    except Exception:
        return str(v)


def main():
    st.set_page_config(
        page_title="KurgiMootor · PI historical walk-forward",
        layout="wide",
    )
    st.title("Production taimeindeks · historical walk-forward")
    st.caption("15.–24.08 · exact app-128 PI rule · target-day leakage locked · READ ONLY")

    st.info(
        "See ei otsi uut alpha't, min-piiri, ilmaakent ega ridge'i. Iga sihtpäeva eel "
        "ehitatakse PI nullist ainult varasemate päris raw-lock sündmustega; BASE treenib "
        "ainult varasematel korjeintervallidel. 26.–27.08 ei lähe primary skoori sisse."
    )

    try:
        harvest = db.get_harvest_history(limit=5000)
        hdf = _normalise_harvest(harvest)
        raw_map, raw_status = _load_raw_setting()
        if not raw_map:
            raise RuntimeError(f"Production raw-lock setting puudub: {raw_status}")

        events = _events(harvest)
        intervals = _build_intervals(events)
        if intervals.empty:
            raise RuntimeError("Korjeintervalle ei tekkinud.")

        # Weather machinery is built once, but every row inside strict_base_rows
        # is already OOS relative to its own target. Per-target transition fit
        # below sees only daily rows strictly before that target.
        intervals_hist = intervals[intervals["target_date"] <= AUDIT_END].copy()
        strict_fields = _strict_base_rows(intervals_hist)
        earliest = min(intervals_hist["target_date"])
        weather_from = max(WEATHER_START, earliest - timedelta(days=2 * WEATHER_BLOCK_DAYS))
        weather = _measured_weather(
            db.get_weather_rows(weather_from, AUDIT_END - timedelta(days=1))
        )
        daily_state = _daily_state_rows(strict_fields, weather)

        target_days = sorted(
            dd for dd in intervals["target_date"].unique()
            if PLANT_INDEX_START <= dd <= AUDIT_END
        )
        if not target_days:
            raise RuntimeError("15.–24.08 target-päevi ei leitud.")

        field_rows = []
        day_rows = []
        skipped = []
        global_missing = set()
        global_late_capture = []

        for target in target_days:
            # Re-use the already-computed strict-OOS BASE rows instead of
            # refitting BASE a second time for the same target. This keeps the
            # audit materially lighter without changing a single prediction.
            raw_test = intervals_hist[intervals_hist["target_date"] == target].copy()
            btest = strict_fields[strict_fields["target_date"] == target].copy()
            if raw_test.empty:
                skipped.append({"date": target, "reason": "BASE: target interval puudub"})
                continue
            if len(btest) != len(raw_test):
                skipped.append({
                    "date": target,
                    "reason": f"BASE strict-OOS incomplete: {len(btest)}/{len(raw_test)} field rows",
                })
                continue
            test = raw_test.merge(
                btest[["target_date", "field", "order", "base", "train_n"]],
                on=["target_date", "field", "order"],
                how="left",
                validate="one_to_one",
            ).sort_values(["order", "field"]).reset_index(drop=True)
            if test["base"].isna().any():
                skipped.append({"date": target, "reason": "BASE strict-OOS merge incomplete"})
                continue
            test["base_pred"] = test["base"].astype(float)
            test["calendar_interval_days"] = [
                int((dd - sd).days) for dd, sd in zip(test["target_date"], test["start_date"])
            ]
            base_train_n = int(test["train_n"].min())

            idx, last_event, trace, missing, late_capture = _pi_before_target(hdf, raw_map, target)
            global_missing.update(missing)
            global_late_capture.extend(late_capture)
            if missing:
                skipped.append({"date": target, "reason": f"PI raw-lock missing: {len(missing)}"})
                continue
            provenance_clean = (len(late_capture) == 0)

            # Weather fit locked to information strictly before target.
            wx_cap = np.nan
            wx_raw = np.nan
            wx_train_n = 0
            wx_lambda = np.nan
            trans = _transition_training_rows(daily_state, target)
            if len(trans) >= MIN_WEATHER_TRANSITIONS:
                wx_fit = _fit_weather_transition(trans)
                wx = _raw_wx_for_target(target, weather, wx_fit)
                wx_cap = float(wx["cap_delta"])
                wx_raw = float(wx["raw_delta"])
                wx_train_n = int(len(trans))
                wx_lambda = float(wx_fit["lambda"])

            test = test.sort_values(["order", "field"]).reset_index(drop=True)
            all_seen = True
            for _, r in test.iterrows():
                ff = int(r["field"])
                pi = float(idx[ff])
                ev = last_event.get(ff)
                seen = ev is not None
                all_seen = all_seen and seen
                base = float(r["base_pred"])
                actual = float(r["actual"])
                pi_pred = base * pi
                piwx_pred = pi_pred * math.exp(wx_cap) if math.isfinite(wx_cap) else np.nan
                field_rows.append({
                    "date": target,
                    "field": ff,
                    "order": int(r["order"]),
                    "start_date": r["start_date"],
                    "calendar_interval": int(r["calendar_interval_days"]),
                    "growth": float(r["growth"]),
                    "actual": actual,
                    "BASE": base,
                    "PI": pi,
                    "PI_seen_before": bool(seen),
                    "PI_last_event": ev.get("date") if ev else None,
                    "PI_last_actual_raw": ev.get("actual/raw") if ev else np.nan,
                    "BASE_PI": pi_pred,
                    "WX_cap": wx_cap,
                    "WX_raw": wx_raw,
                    "BASE_PI_WX": piwx_pred,
                    "BASE_error": base - actual,
                    "PI_error": pi_pred - actual,
                    "PI_WX_error": piwx_pred - actual if math.isfinite(piwx_pred) else np.nan,
                    "base_train_n": base_train_n,
                    "wx_train_n": wx_train_n,
                    "wx_lambda": wx_lambda,
                })

            g = pd.DataFrame(field_rows)
            gd = g[g["date"] == target].copy()
            day_rows.append({
                "date": target,
                "fields": ",".join(str(int(x)) for x in gd.sort_values("order")["field"]),
                "actual": float(gd["actual"].sum()),
                "BASE": float(gd["BASE"].sum()),
                "BASE_PI": float(gd["BASE_PI"].sum()),
                "BASE_PI_WX": float(gd["BASE_PI_WX"].sum()) if gd["BASE_PI_WX"].notna().all() else np.nan,
                "all_target_fields_seen": bool(all_seen),
                "anchor_provenance_clean": bool(provenance_clean),
                "late_anchor_count": int(len(late_capture)),
                "mean_PI": float(gd["PI"].mean()),
                "min_PI": float(gd["PI"].min()),
                "max_PI": float(gd["PI"].max()),
                "WX_cap": wx_cap,
                "WX_raw": wx_raw,
                "base_train_n": base_train_n,
                "wx_train_n": wx_train_n,
                "wx_lambda": wx_lambda,
            })

        fields = pd.DataFrame(field_rows)
        days = pd.DataFrame(day_rows).sort_values("date").reset_index(drop=True)
        if days.empty:
            raise RuntimeError("Ühtegi walk-forward päeva ei saanud skoorida.")

        for c in ["BASE", "BASE_PI", "BASE_PI_WX"]:
            days[f"{c}_error"] = days[c] - days["actual"]
            days[f"{c}_APE"] = np.abs(days[f"{c}_error"]) / np.maximum(days["actual"], 0.5) * 100.0

        # Full target period includes unavoidable PI warm-up. Mature = every
        # target field has already had at least one earlier production PI event.
        mature = days[days["all_target_fields_seen"]].copy()
        strict_mature = mature[mature["anchor_provenance_clean"]].copy()
        wx_days = days[days["BASE_PI_WX"].notna()].copy()
        mature_wx = mature[mature["BASE_PI_WX"].notna()].copy()

        def decision_table(g: pd.DataFrame) -> pd.DataFrame:
            if g.empty:
                return pd.DataFrame()
            b = _score(g["actual"].to_numpy(float), g["BASE"].to_numpy(float))["MAE"]
            rows = [
                _variant_row("BASE", g, "BASE", b),
                _variant_row("BASE × PI", g, "BASE_PI", b),
            ]
            gw = g[g["BASE_PI_WX"].notna()].copy()
            if len(gw) == len(g) and len(gw) > 0:
                rows.append(_variant_row("BASE × PI × WX-CAP", g, "BASE_PI_WX", b))
            return pd.DataFrame(rows)

        summary_all = decision_table(days)
        summary_mature = decision_table(mature)
        summary_strict_mature = decision_table(strict_mature)

        # Field-level score is supportive only: fields within one day are not
        # independent days, but it tells us whether one lucky daily sum drove it.
        f_mature = fields[fields["date"].isin(mature["date"])].copy()
        field_summary = pd.DataFrame()
        if not f_mature.empty:
            fb = _score(f_mature["actual"].to_numpy(float), f_mature["BASE"].to_numpy(float))["MAE"]
            field_summary = pd.DataFrame([
                _variant_row("BASE", f_mature.rename(columns={"actual":"actual"}), "BASE", fb),
                _variant_row("BASE × PI", f_mature.rename(columns={"actual":"actual"}), "BASE_PI", fb),
            ])
            if f_mature["BASE_PI_WX"].notna().all():
                field_summary = pd.concat([
                    field_summary,
                    pd.DataFrame([_variant_row("BASE × PI × WX-CAP", f_mature, "BASE_PI_WX", fb)])
                ], ignore_index=True)

    except Exception as exc:
        st.exception(exc)
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Walk-forward päevi", len(days))
    c2.metric("Mature PI päevi", len(mature))
    c3.metric("Strict provenance mature", len(strict_mature))
    c4.metric("Mature field intervalle", len(f_mature))

    st.markdown("### 1. Kõige tähtsam · strict historical mature PI")
    if strict_mature.empty:
        st.error(
            "❌ Ühtegi mature päeva ei jäänud, mille KÕIK varasemad raw-ankrud oleksid captured_at järgi "
            "hiljemalt oma korjepäeval lukustatud. Siis ei tohi seda osa päris historical OOS-iks nimetada."
        )
    else:
        st.dataframe(
            summary_strict_mature.style.format({
                "MAE": "{:.2f}", "MAPE %": "{:.1f}", "Bias": "{:+.2f}",
                "Paranemine BASE suhtes %": "{:+.1f}%",
            }),
            use_container_width=True, hide_index=True,
        )
        bmae = float(summary_strict_mature.loc[summary_strict_mature["Variant"] == "BASE", "MAE"].iloc[0])
        pmae = float(summary_strict_mature.loc[summary_strict_mature["Variant"] == "BASE × PI", "MAE"].iloc[0])
        if pmae < bmae:
            st.success(
                f"✅ PI LÖÖB BASE'i mature historical OOS-is: {bmae:.2f} → {pmae:.2f} "
                f"({(bmae-pmae)/bmae*100:.1f}% parem)."
            )
        else:
            st.error(
                f"❌ PI EI LÖÖ BASE'i mature historical OOS-is: {bmae:.2f} → {pmae:.2f}."
            )

    st.markdown("### 1b. Mature PI · raw-lock state rekonstrueeritud")
    if not mature.empty:
        st.dataframe(
            summary_mature.style.format({
                "MAE": "{:.2f}", "MAPE %": "{:.1f}", "Bias": "{:+.2f}",
                "Paranemine BASE suhtes %": "{:+.1f}%",
            }),
            use_container_width=True, hide_index=True,
        )
        if len(strict_mature) < len(mature):
            st.warning(
                f"⚠️ {len(mature)-len(strict_mature)} mature päeva kasutab vähemalt üht raw-ankrut, mille captured_at "
                "on selle ankru oma korjepäevast hilisem. Tulemus on kasulik rekonstruktsioon, kuid mitte puhas historical provenance."
            )

    st.markdown("### 2. Kogu 15.–24.08 walk-forward · warm-up jääb sisse")
    st.dataframe(
        summary_all.style.format({
            "MAE": "{:.2f}", "MAPE %": "{:.1f}", "Bias": "{:+.2f}",
            "Paranemine BASE suhtes %": "{:+.1f}%",
        }),
        use_container_width=True, hide_index=True,
    )
    st.caption(
        "15.–19.08 ei ole aus 'mature PI' kontroll: 14 põllu rotatsiooni tõttu jõuavad paljud target-põllud "
        "alles oma esimese PI sündmuseni. Neid päevi ei peideta, kuid otsus tehakse mature realt."
    )

    st.markdown("### 3. Päev-päevalt")
    show = days.rename(columns={
        "date": "Päev", "fields": "Põllud", "actual": "Tegelik ABC",
        "BASE_PI": "BASE×PI", "BASE_PI_WX": "BASE×PI×WX",
        "all_target_fields_seen": "PI mature", "anchor_provenance_clean": "Anchor provenance OK",
        "mean_PI": "Keskmine PI",
        "min_PI": "Min PI", "max_PI": "Max PI",
    })
    cols = [
        "Päev", "Põllud", "PI mature", "Anchor provenance OK", "late_anchor_count",
        "Tegelik ABC", "BASE", "BASE×PI", "BASE×PI×WX",
        "BASE_error", "BASE_PI_error", "BASE_PI_WX_error", "Keskmine PI", "Min PI", "Max PI",
        "WX_cap", "base_train_n", "wx_train_n", "wx_lambda",
    ]
    st.dataframe(
        show[cols].style.format({
            "Päev": _fmt_day, "Tegelik ABC": "{:.1f}", "BASE": "{:.1f}", "BASE×PI": "{:.1f}",
            "BASE×PI×WX": lambda v: "—" if pd.isna(v) else f"{float(v):.1f}",
            "BASE_error": "{:+.1f}", "BASE_PI_error": "{:+.1f}",
            "BASE_PI_WX_error": lambda v: "—" if pd.isna(v) else f"{float(v):+.1f}",
            "Keskmine PI": "{:.3f}", "Min PI": "{:.3f}", "Max PI": "{:.3f}",
            "WX_cap": lambda v: "—" if pd.isna(v) else f"{float(v):+.3f}",
            "wx_lambda": lambda v: "—" if pd.isna(v) else f"{float(v):.0f}",
        }),
        use_container_width=True, hide_index=True,
    )

    st.markdown("### 4. Põllu kaupa · kas päevakoond peidab juhust?")
    st.dataframe(
        field_summary.style.format({
            "MAE": "{:.2f}", "MAPE %": "{:.1f}", "Bias": "{:+.2f}",
            "Paranemine BASE suhtes %": "{:+.1f}%",
        }),
        use_container_width=True, hide_index=True,
    )

    with st.expander("Kontrolliks · mature põllu read"):
        fs = f_mature[[
            "date", "field", "order", "start_date", "calendar_interval", "growth",
            "PI", "PI_last_event", "PI_last_actual_raw", "actual", "BASE", "BASE_PI", "BASE_PI_WX",
            "BASE_error", "PI_error", "PI_WX_error",
        ]].copy().rename(columns={
            "date":"Päev", "field":"Põld", "order":"Jrk", "start_date":"Eelmine korje",
            "calendar_interval":"Kalendriintervall p", "growth":"Growth p", "PI":"PI enne korjet",
            "PI_last_event":"PI viimane sündmus", "PI_last_actual_raw":"Viimane actual/raw",
            "actual":"Tegelik", "BASE_PI":"BASE×PI", "BASE_PI_WX":"BASE×PI×WX",
        })
        st.dataframe(
            fs.style.format({
                "Päev":_fmt_day, "Eelmine korje":_fmt_day, "PI viimane sündmus":_fmt_day,
                "Growth p":"{:.2f}", "PI enne korjet":"{:.3f}",
                "Viimane actual/raw":lambda v:"—" if pd.isna(v) else f"{float(v):.3f}",
                "Tegelik":"{:.2f}", "BASE":"{:.2f}", "BASE×PI":"{:.2f}",
                "BASE×PI×WX":lambda v:"—" if pd.isna(v) else f"{float(v):.2f}",
                "BASE_error":"{:+.2f}", "PI_error":"{:+.2f}",
                "PI_WX_error":lambda v:"—" if pd.isna(v) else f"{float(v):+.2f}",
            }),
            use_container_width=True, hide_index=True,
        )

    if skipped:
        with st.expander(f"Vahele jäetud päevad ({len(skipped)})"):
            st.dataframe(pd.DataFrame(skipped), use_container_width=True, hide_index=True)

    st.success(
        "🔒 LEAKAGE LOCK: target-päeva PI kasutab ainult varasemaid PI sündmusi; BASE kasutab ainult varasemaid "
        "korjeintervalle; weather kasutab ainult pre-target blokke ja varasemaid transition-ridu; 26.–27.08 ei ole primary skooris."
    )
    st.caption(
        f"Raw-lock allikas: {raw_status}. Alpha/min = {PLANT_INDEX_ALPHA:.2f}/{PLANT_INDEX_MIN:.2f}. "
        "Production PI algus 15.08 on lukus — seda ei nihutata tagasi lihtsalt suurema N saamiseks."
    )
    st.caption("READ ONLY · db.get_harvest_history · db.get_weather_rows · db.get_app_setting · no writes")


if __name__ == "__main__":
    main()
