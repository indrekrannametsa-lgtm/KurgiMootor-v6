from __future__ import annotations

"""
KurgiMootor · edge_weather-15
=============================

FINAL NARROW HOLDOUT TEST · READ ONLY

Purpose
-------
Test the already locked WIND×DRY HIGH L3–7 LEVEL+DELTA signal only on harvest
days AFTER the signal was locked (21.08.2026 onward).

This is a feature-selection holdout:
- feature set is fixed before the holdout starts;
- each target day is predicted strictly from harvests BEFORE that day;
- measured WIND×DRY values are only from L3–7 before the harvest day;
- target-day yield is evaluation only;
- no future holdout yield enters an earlier holdout prediction.

Comparison
----------
BASE ABC  = locked comparator baseline:
            previous same-field rate + season day + growth + growth delta
WD ABC    = exactly the same BASE + locked WIND×DRY HIGH L3–7 LEVEL+DELTA

Important:
This is NOT the production engine itself and previous yield is NOT proposed as a
production anchor. The purpose is only to isolate whether the fixed WD weather
signal still adds out-of-sample information after lock date.

No DB writes. No weather refresh. No research search.
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import streamlit as st

import db


# ---------------------------------------------------------------------
# Locked definition
# ---------------------------------------------------------------------

HOLDOUT_START = date(2026, 8, 21)
SEASON_START = date(2026, 6, 15)
WEATHER_START = date(2026, 7, 1)

HOURS_PER_FIELD = 3.0
RIDGE_ALPHA = 10.0
MIN_TRAIN_ROWS = 24
TARGET_EPS = 0.20

HIGH_Q = 0.75
MIN_DAYS_FOR_HIGH_THRESHOLD = 10

BASE_COLS = [
    "prev_log_rate",
    "season_day",
    "growth",
    "growth_delta",
]
LOCKED_EXTRA = [
    "l3_7_high_days",
    "l3_7_high_run",
    "d_l3_7_high_days",
    "d_l3_7_high_run",
]


@dataclass
class Event:
    day: date
    field: int
    order: int
    abc: float
    interval_days: Optional[float] = None


# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------

def _d(v) -> Optional[date]:
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


def _f(v) -> Optional[float]:
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def _abc(row: dict) -> Optional[float]:
    vals = [_f(row.get(k)) for k in ("a", "b", "c")]
    if any(v is None for v in vals):
        direct = _f(row.get("_abc"))
        return direct if direct is not None and direct >= 0 else None
    return float(sum(vals))


def _reliable(row: dict) -> bool:
    q = str(row.get("data_quality") or row.get("quality") or "").strip().lower()
    return q not in {"hinnanguline", "ligikaudne", "estimated", "approximate"}


def _prepare_events(rows: List[dict], before_day: Optional[date] = None) -> List[Event]:
    out: List[Event] = []
    for r in rows:
        dd = _d(r.get("harvest_date"))
        if dd is None:
            continue
        if before_day is not None and dd >= before_day:
            continue
        if not _reliable(r):
            continue
        try:
            field = int(r.get("field_no"))
        except Exception:
            continue
        if not 1 <= field <= 14:
            continue
        abc = _abc(r)
        if abc is None or abc < 0:
            continue
        try:
            order = int(r.get("harvest_order") or 1)
        except Exception:
            order = 1
        out.append(Event(
            dd,
            field,
            order,
            float(abc),
            _f(r.get("interval_days")),
        ))
    out.sort(key=lambda e: (e.day, e.order, e.field))
    return out


def _field_hist(events: Sequence[Event], field: int) -> List[Event]:
    return sorted(
        [e for e in events if e.field == field],
        key=lambda e: (e.day, e.order, e.field),
    )


def _growth_days(prev: Event, cur: Event) -> float:
    g = float((cur.day - prev.day).days)
    g += (cur.order - prev.order) * (HOURS_PER_FIELD / 24.0)
    return max(0.5, g)


def _complete_actual_days(harvest: List[dict]) -> Dict[date, Dict[int, dict]]:
    by_day: Dict[date, Dict[int, dict]] = {}
    for r in harvest:
        dd = _d(r.get("harvest_date"))
        if dd is None or dd < HOLDOUT_START or not _reliable(r):
            continue
        try:
            f = int(r.get("field_no"))
        except Exception:
            continue
        abc = _abc(r)
        if abc is None:
            continue
        rr = dict(r)
        rr["_abc"] = float(abc)
        by_day.setdefault(dd, {})[f] = rr

    # Strictly complete = exactly 3 unique fields.
    return {
        dd: rows
        for dd, rows in sorted(by_day.items())
        if len(rows) == 3
    }


def _ordered_actual_rows(rows: Dict[int, dict]) -> List[dict]:
    """Use recorded harvest_order if present; otherwise field order."""
    vals = list(rows.values())
    vals.sort(key=lambda r: (
        int(r.get("harvest_order") or 99),
        int(r.get("field_no") or 99),
    ))
    return vals


# ---------------------------------------------------------------------
# Measured WIND×DRY
# ---------------------------------------------------------------------

def _measured_weather(rows: List[dict]) -> Dict[date, dict]:
    out: Dict[date, dict] = {}
    for r in rows:
        dd = _d(r.get("weather_date"))
        if dd is None:
            continue
        if str(r.get("data_kind") or "").strip().lower() != "measured":
            continue
        if not bool(r.get("checked")):
            continue
        wind = _f(r.get("wind_avg_ms"))
        rh = _f(r.get("humidity_avg_pct"))
        if wind is None or rh is None:
            continue
        out[dd] = {
            "wind": float(wind),
            "rh": float(rh),
            "wind_dry": float(wind) * (100.0 - float(rh)),
        }
    return out


def _lag_dates(cur_day: date, lag_start: int = 3, lag_end: int = 7) -> List[date]:
    return [
        cur_day - timedelta(days=lag)
        for lag in range(lag_end, lag_start - 1, -1)
    ]


def _max_consecutive_true(flags: Sequence[bool]) -> int:
    best = cur = 0
    for flag in flags:
        if flag:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def _prior_high_threshold(
    measured: Dict[date, dict],
    before_day: date,
) -> Optional[float]:
    # Threshold itself is also past-only.
    vals = [
        float(rec["wind_dry"])
        for dd, rec in measured.items()
        if dd < before_day and np.isfinite(float(rec["wind_dry"]))
    ]
    if len(vals) < MIN_DAYS_FOR_HIGH_THRESHOLD:
        return None
    return float(np.quantile(np.asarray(vals, dtype=float), HIGH_Q))


def _wd_window(measured: Dict[date, dict], days: Sequence[date]):
    if not days or any(d not in measured for d in days):
        return None

    vals = np.asarray(
        [float(measured[d]["wind_dry"]) for d in days],
        dtype=float,
    )
    threshold = _prior_high_threshold(measured, min(days))
    if threshold is None:
        return None

    flags = [bool(v >= threshold) for v in vals]
    return {
        "avg": float(np.mean(vals)),
        "high_days": float(sum(flags)),
        "high_run": float(_max_consecutive_true(flags)),
        "threshold": float(threshold),
    }


def _window_features(
    cur: Event,
    prev: Event,
    prevprev: Optional[Event],
    measured: Dict[date, dict],
):
    current = _wd_window(measured, _lag_dates(cur.day, 3, 7))
    if current is None:
        return None

    previous = None
    if prevprev is not None:
        previous = _wd_window(measured, _lag_dates(prev.day, 3, 7))
        if previous is None:
            return None

    return {
        "l3_7_high_days": float(current["high_days"]),
        "l3_7_high_run": float(current["high_run"]),
        "d_l3_7_high_days": (
            float(current["high_days"] - previous["high_days"])
            if previous else 0.0
        ),
        "d_l3_7_high_run": (
            float(current["high_run"] - previous["high_run"])
            if previous else 0.0
        ),
        "l3_7_avg": float(current["avg"]),
        "threshold": float(current["threshold"]),
    }


# ---------------------------------------------------------------------
# Historical strict training table
# ---------------------------------------------------------------------

def _historical_df(
    events: List[Event],
    measured: Dict[date, dict],
) -> pd.DataFrame:
    rows = []

    for field in range(1, 15):
        hist = _field_hist(events, field)

        for i in range(1, len(hist)):
            cur = hist[i]
            prev = hist[i - 1]
            prevprev = hist[i - 2] if i >= 2 else None

            growth = _growth_days(prev, cur)

            if prevprev is not None:
                prev_growth = _growth_days(prevprev, prev)
            elif prev.interval_days is not None and prev.interval_days > 0:
                prev_growth = float(prev.interval_days)
            else:
                continue

            cur_rate = cur.abc / max(0.5, growth)
            prev_rate = prev.abc / max(0.5, prev_growth)

            if prev_rate <= 0 or cur_rate < 0:
                continue

            wx = _window_features(cur, prev, prevprev, measured)
            if wx is None:
                continue

            y = math.log(
                (cur_rate + TARGET_EPS) /
                (prev_rate + TARGET_EPS)
            )

            rows.append({
                "date": cur.day,
                "field": field,
                "prev_rate": float(prev_rate),
                "prev_log_rate": math.log(prev_rate + TARGET_EPS),
                "season_day": float((cur.day - SEASON_START).days),
                "growth": float(growth),
                "growth_delta": float(growth - prev_growth),
                "y": float(y),
                **{k: float(wx[k]) for k in LOCKED_EXTRA},
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["date", "field"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------
# Ridge
# ---------------------------------------------------------------------

def _ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd = np.where(sd < 1e-6, 1.0, sd)

    Z = (X - mu) / sd
    Xd = np.column_stack([np.ones(len(Z)), Z])

    reg = np.eye(Xd.shape[1]) * float(alpha)
    reg[0, 0] = 0.0

    try:
        beta = np.linalg.solve(
            Xd.T @ Xd + reg,
            Xd.T @ y,
        )
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(
            Xd.T @ Xd + reg
        ) @ (Xd.T @ y)

    return {
        "mu": mu,
        "sd": sd,
        "beta": beta,
    }


def _ridge_pred(model, x: Sequence[float]) -> float:
    x = np.asarray(x, dtype=float)
    z = (x - model["mu"]) / model["sd"]
    return float(np.r_[1.0, z] @ model["beta"])


def _fit_models(df: pd.DataFrame):
    if len(df) < MIN_TRAIN_ROWS:
        return None

    y = pd.to_numeric(
        df["y"],
        errors="coerce",
    ).to_numpy(dtype=float)

    xb = df[BASE_COLS].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(dtype=float)

    cand_cols = BASE_COLS + LOCKED_EXTRA
    xc = df[cand_cols].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(dtype=float)

    ok = (
        np.isfinite(y)
        & np.all(np.isfinite(xb), axis=1)
        & np.all(np.isfinite(xc), axis=1)
    )

    if int(ok.sum()) < MIN_TRAIN_ROWS:
        return None

    return (
        _ridge_fit(xb[ok], y[ok], RIDGE_ALPHA),
        _ridge_fit(xc[ok], y[ok], RIDGE_ALPHA),
        int(ok.sum()),
    )


# ---------------------------------------------------------------------
# Strict holdout prediction
# ---------------------------------------------------------------------

def _predict_field(
    target: date,
    row: dict,
    order: int,
    events_before: List[Event],
    measured: Dict[date, dict],
    base_model,
    wd_model,
):
    field = int(row.get("field_no"))
    hist = _field_hist(events_before, field)
    if not hist:
        return None

    prev = hist[-1]
    prevprev = hist[-2] if len(hist) >= 2 else None
    cur = Event(
        target,
        field,
        order,
        0.0,
        _f(row.get("interval_days")),
    )

    growth = _growth_days(prev, cur)

    if prevprev is not None:
        prev_growth = _growth_days(prevprev, prev)
    elif prev.interval_days is not None and prev.interval_days > 0:
        prev_growth = float(prev.interval_days)
    else:
        return None

    prev_rate = prev.abc / max(0.5, prev_growth)
    if prev_rate <= 0:
        return None

    wx = _window_features(cur, prev, prevprev, measured)
    if wx is None:
        return None

    rec = {
        "prev_log_rate": math.log(prev_rate + TARGET_EPS),
        "season_day": float((target - SEASON_START).days),
        "growth": float(growth),
        "growth_delta": float(growth - prev_growth),
        **{k: float(wx[k]) for k in LOCKED_EXTRA},
    }

    base_log_change = _ridge_pred(
        base_model,
        [rec[c] for c in BASE_COLS],
    )
    wd_log_change = _ridge_pred(
        wd_model,
        [rec[c] for c in BASE_COLS + LOCKED_EXTRA],
    )

    base_rate = max(
        0.0,
        (prev_rate + TARGET_EPS) * math.exp(base_log_change) - TARGET_EPS,
    )
    wd_rate = max(
        0.0,
        (prev_rate + TARGET_EPS) * math.exp(wd_log_change) - TARGET_EPS,
    )

    base_abc = float(base_rate * growth)
    wd_abc = float(wd_rate * growth)
    actual_abc = float(row["_abc"])

    return {
        "Päev": target,
        "Põld": field,
        "Order": order,
        "BASE ABC": base_abc,
        "WD ABC": wd_abc,
        "Tegelik ABC": actual_abc,
        "BASE viga": base_abc - actual_abc,
        "WD viga": wd_abc - actual_abc,
        "WD kordaja": (
            wd_abc / base_abc
            if base_abc > 1e-9 else np.nan
        ),
        "HIGH päevi": float(wx["l3_7_high_days"]),
        "HIGH jada": float(wx["l3_7_high_run"]),
        "Δ HIGH päevi": float(wx["d_l3_7_high_days"]),
        "Δ HIGH jada": float(wx["d_l3_7_high_run"]),
        "WD avg": float(wx["l3_7_avg"]),
        "HIGH lävi": float(wx["threshold"]),
    }


def _run_holdout(
    harvest: List[dict],
    measured: Dict[date, dict],
):
    complete_days = _complete_actual_days(harvest)

    daily_rows = []
    field_rows = []

    for target, actual_by_field in complete_days.items():
        # STRICT CUTOFF: no target/future harvest in training/state.
        events_before = _prepare_events(
            harvest,
            before_day=target,
        )

        train_df = _historical_df(
            events_before,
            measured,
        )

        fitted = _fit_models(train_df)

        if fitted is None:
            daily_rows.append({
                "Päev": target,
                "BASE ABC": np.nan,
                "WD ABC": np.nan,
                "Tegelik ABC": float(
                    sum(float(r["_abc"]) for r in actual_by_field.values())
                ),
                "BASE viga": np.nan,
                "WD viga": np.nan,
                "Parandus": np.nan,
                "WD võidab": None,
                "Keskm WD kordaja": np.nan,
                "Keskm HIGH päevi": np.nan,
                "Keskm Δ HIGH päevi": np.nan,
                "Treening N": len(train_df),
                "Kiht täielik": "ei",
            })
            continue

        base_model, wd_model, train_n = fitted

        rows = _ordered_actual_rows(actual_by_field)
        preds = []

        for order, row in enumerate(rows, start=1):
            p = _predict_field(
                target=target,
                row=row,
                order=order,
                events_before=events_before,
                measured=measured,
                base_model=base_model,
                wd_model=wd_model,
            )
            if p is not None:
                preds.append(p)
                field_rows.append(p)

        actual_total = float(
            sum(float(r["_abc"]) for r in rows)
        )

        if len(preds) != 3:
            daily_rows.append({
                "Päev": target,
                "BASE ABC": np.nan,
                "WD ABC": np.nan,
                "Tegelik ABC": actual_total,
                "BASE viga": np.nan,
                "WD viga": np.nan,
                "Parandus": np.nan,
                "WD võidab": None,
                "Keskm WD kordaja": np.nan,
                "Keskm HIGH päevi": np.nan,
                "Keskm Δ HIGH päevi": np.nan,
                "Treening N": train_n,
                "Kiht täielik": "ei",
            })
            continue

        base_total = float(sum(p["BASE ABC"] for p in preds))
        wd_total = float(sum(p["WD ABC"] for p in preds))

        base_err = base_total - actual_total
        wd_err = wd_total - actual_total

        daily_rows.append({
            "Päev": target,
            "BASE ABC": base_total,
            "WD ABC": wd_total,
            "Tegelik ABC": actual_total,
            "BASE viga": base_err,
            "WD viga": wd_err,
            "Parandus": abs(base_err) - abs(wd_err),
            "WD võidab": abs(wd_err) < abs(base_err),
            "Keskm WD kordaja": float(
                np.mean([p["WD kordaja"] for p in preds])
            ),
            "Keskm HIGH päevi": float(
                np.mean([p["HIGH päevi"] for p in preds])
            ),
            "Keskm Δ HIGH päevi": float(
                np.mean([p["Δ HIGH päevi"] for p in preds])
            ),
            "Treening N": train_n,
            "Kiht täielik": "jah",
        })

    return (
        pd.DataFrame(daily_rows),
        pd.DataFrame(field_rows),
    )


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def _mae(actual, pred) -> float:
    a = np.asarray(actual, dtype=float)
    p = np.asarray(pred, dtype=float)
    ok = np.isfinite(a) & np.isfinite(p)
    return float(np.mean(np.abs(p[ok] - a[ok]))) if np.any(ok) else np.nan


def _mape(actual, pred) -> float:
    a = np.asarray(actual, dtype=float)
    p = np.asarray(pred, dtype=float)
    ok = np.isfinite(a) & np.isfinite(p) & (np.abs(a) > 1e-9)
    return (
        float(np.mean(np.abs(p[ok] - a[ok]) / np.abs(a[ok])) * 100.0)
        if np.any(ok) else np.nan
    )


# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="KurgiMootor · WD holdout",
        layout="wide",
    )

    st.title("WIND×DRY · lukustamisjärgne holdout")
    st.caption(
        "21.08 → viimane täielik korjepäev · Lead 0 · "
        "feature-selection holdout · READ ONLY"
    )

    st.info(
        "See test ei küsi enam, kas WIND×DRY sobis perioodile, mille pealt ta leiti. "
        "Ta küsib: kas ENNE 21.08 lukustatud HIGH L3–7 LEVEL+DELTA annab lisainfot "
        "ka päevadel, mida tunnuse valimisel kasutada ei saanud?"
    )

    try:
        harvest = db.get_harvest_history(limit=5000)

        complete = _complete_actual_days(harvest)
        if not complete:
            st.warning(
                "21.08 järel pole veel ühtegi täielikku 3-põllu korjepäeva."
            )
            st.stop()

        holdout_end = max(complete)
        weather_rows = db.get_weather_rows(
            WEATHER_START,
            holdout_end,
        )
        measured = _measured_weather(weather_rows)

        daily, fields = _run_holdout(
            harvest,
            measured,
        )

    except Exception as exc:
        st.exception(exc)
        st.stop()

    valid = daily[
        (daily["Kiht täielik"] == "jah")
        & daily["BASE ABC"].notna()
        & daily["WD ABC"].notna()
        & daily["Tegelik ABC"].notna()
    ].copy()

    if valid.empty:
        st.error(
            "Täielikku holdout-võrdlust ei tekkinud. "
            "Kontrolli mõõdetud WIND×DRY ilma täielikkust."
        )
        st.stop()

    base_mae = _mae(
        valid["Tegelik ABC"],
        valid["BASE ABC"],
    )
    wd_mae = _mae(
        valid["Tegelik ABC"],
        valid["WD ABC"],
    )
    base_mape = _mape(
        valid["Tegelik ABC"],
        valid["BASE ABC"],
    )
    wd_mape = _mape(
        valid["Tegelik ABC"],
        valid["WD ABC"],
    )
    wins = int(valid["WD võidab"].sum())
    n = len(valid)

    improvement = (
        (base_mae - wd_mae) / base_mae
        if np.isfinite(base_mae) and base_mae > 1e-9
        else np.nan
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Holdout päevi",
        f"{n}",
    )
    c2.metric(
        "BASE MAE",
        f"{base_mae:.2f}",
    )
    c3.metric(
        "WD MAE",
        f"{wd_mae:.2f}",
        delta=f"{base_mae-wd_mae:+.2f} kasti parem",
    )
    c4.metric(
        "MAPE",
        f"{base_mape:.1f}% → {wd_mape:.1f}%",
    )
    c5.metric(
        "WD võidab",
        f"{wins}/{n} päeva",
    )

    # Deliberately conservative wording because N will still be small.
    if (
        n >= 4
        and np.isfinite(improvement)
        and improvement >= 0.15
        and wins >= max(3, math.ceil(0.60 * n))
    ):
        st.success(
            f"✅ HOLDOUT POSITIIVNE: pärast lukustamist paraneb MAE "
            f"{100.0*improvement:.0f}% ({base_mae:.2f} → {wd_mae:.2f}) "
            f"ja WD võidab {wins}/{n} päeva."
        )
    elif wd_mae < base_mae:
        st.warning(
            f"🟡 HOLDOUTIS ON EELIS, AGA TÕEND ON VEEL VÄIKE: "
            f"MAE {base_mae:.2f} → {wd_mae:.2f}, "
            f"WD võidab {wins}/{n} päeva."
        )
    else:
        st.error(
            f"❌ HOLDOUT EI KINNITA WD EELIST: "
            f"MAE {base_mae:.2f} → {wd_mae:.2f}, "
            f"WD võidab {wins}/{n} päeva."
        )

    st.markdown("### Üks põhitabel")

    show = valid[[
        "Päev",
        "BASE ABC",
        "Keskm WD kordaja",
        "WD ABC",
        "Tegelik ABC",
        "BASE viga",
        "WD viga",
        "Parandus",
        "Keskm HIGH päevi",
        "Keskm Δ HIGH päevi",
        "Treening N",
    ]].copy()

    st.dataframe(
        show.style.format({
            "Päev": lambda x: (
                x.strftime("%d.%m")
                if hasattr(x, "strftime") else str(x)
            ),
            "BASE ABC": "{:.1f}",
            "Keskm WD kordaja": "{:.3f}",
            "WD ABC": "{:.1f}",
            "Tegelik ABC": "{:.1f}",
            "BASE viga": "{:+.1f}",
            "WD viga": "{:+.1f}",
            "Parandus": "{:+.1f}",
            "Keskm HIGH päevi": "{:.1f}",
            "Keskm Δ HIGH päevi": "{:+.1f}",
            "Treening N": "{:.0f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Parandus = BASE absoluutviga − WD absoluutviga. "
        "Positiivne = lukustatud WD signaal aitas. "
        "Iga rea mudel on fititud ainult harvest_date < selle rea Päev."
    )

    with st.expander(
        "Kontrolliks põllu kaupa",
        expanded=False,
    ):
        st.dataframe(
            fields.style.format({
                "Päev": lambda x: (
                    x.strftime("%d.%m")
                    if hasattr(x, "strftime") else str(x)
                ),
                "BASE ABC": "{:.2f}",
                "WD ABC": "{:.2f}",
                "Tegelik ABC": "{:.2f}",
                "BASE viga": "{:+.2f}",
                "WD viga": "{:+.2f}",
                "WD kordaja": "{:.3f}",
                "HIGH päevi": "{:.0f}",
                "HIGH jada": "{:.0f}",
                "Δ HIGH päevi": "{:+.0f}",
                "Δ HIGH jada": "{:+.0f}",
                "WD avg": "{:.1f}",
                "HIGH lävi": "{:.1f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

    st.divider()

    st.caption(
        "TÕLGENDUSPIIR: BASE kasutab eelmise sama põllu saagikiirust ainult selleks, "
        "et mõõta WD tunnuste inkrementaalset väärtust võrdses comparatoris. "
        "See EI ole soovitus kasutada eelmist saaki productioni otsese ankruna."
    )

    st.caption(
        "AUDIT LOCK: ainult db.get_harvest_history ja db.get_weather_rows. "
        "Puuduvad save_/set_/delete_/upsert_ kutsed; ilma API-t ei värskendata."
    )


if __name__ == "__main__":
    main()
