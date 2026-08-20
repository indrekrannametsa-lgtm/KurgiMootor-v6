from __future__ import annotations

"""
KurgiMootor · app-128 vs WIND×DRY L3–7 LIVE comparator
========================================================

READ ONLY. This file never writes to Supabase.

Purpose
-------
Compare the current production app-128 forecast snapshot against exactly one locked
weather layer discovered in LAB-151..154:

    WIND×DRY = wind_avg_ms × (100 - humidity_avg_pct)
    window   = 3..7 days before target harvest
    features = HIGH level + delta vs the previous same-field harvest-cycle window

The production app-128 number already contains its current plant index. The test layer
is applied multiplicatively to production ABC only; XL is kept exactly equal to app-128.
Because app-128 applies the plant index multiplicatively, multiplying the already indexed
ABC by the weather multiplier is algebraically equivalent to applying the weather layer
before the same plant index.

The weather multiplier is not hand tuned. It is the ratio between two locked ridge
models fitted on the same historical rows:
  BASE      = previous same-field growth rate + season day + growth time
  CANDIDATE = BASE + HIGH L3–7 LEVEL+DELTA

Historical walk-forward is shown only as a sanity check. Live forecasts use the current
measured/forecast weather stored in weather_daily. No forecast is saved.
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, List, Optional, Sequence, Tuple
import math
import re
import sys

import numpy as np
import pandas as pd

TZ = ZoneInfo("Europe/Tallinn")
TODAY = datetime.now(TZ).date()
SEASON_START = date(TODAY.year, 6, 15)
WEATHER_START = date(TODAY.year, 7, 1)
VERSION = "APP128-WINDDRY-LIVE-COMPARE-V1"

HOURS_PER_FIELD = 3.0
RIDGE_ALPHA = 10.0
MIN_TRAIN_ROWS = 24
TARGET_EPS = 0.20
HIGH_Q = 0.75
MIN_DAYS_FOR_HIGH_THRESHOLD = 10

REQUIRED_WEATHER = (
    "wind_avg_ms", "humidity_avg_pct",
)
BASE_COLS = ["prev_log_rate", "season_day", "growth", "growth_delta"]
LOCKED_EXTRA = ["l3_7_high_days", "l3_7_high_run", "d_l3_7_high_days", "d_l3_7_high_run"]


@dataclass
class Event:
    day: date
    field: int
    order: int
    abc: float
    interval_days: Optional[float] = None
    source: str = "actual"


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


def _event_key(e: Event) -> Tuple[date, int, int]:
    return (e.day, e.order, e.field)


def _prepare_events(rows: List[dict]) -> List[Event]:
    out: List[Event] = []
    for r in rows:
        dd = _d(r.get("harvest_date"))
        try:
            field = int(r.get("field_no"))
        except Exception:
            continue
        if dd is None or dd > TODAY or not (1 <= field <= 14) or not _reliable(r):
            continue
        abc = _abc(r)
        if abc is None or abc < 0:
            continue
        try:
            order = int(r.get("harvest_order") or 1)
        except Exception:
            order = 1
        out.append(Event(dd, field, order, float(abc), _f(r.get("interval_days")), "actual"))
    out.sort(key=_event_key)
    return out


def _field_hist(events: Sequence[Event], field: int) -> List[Event]:
    return sorted([e for e in events if e.field == field], key=_event_key)


def _growth_days(prev: Event, cur: Event) -> float:
    g = float((cur.day - prev.day).days) + (cur.order - prev.order) * (HOURS_PER_FIELD / 24.0)
    return max(0.5, g)


def _lag_dates(cur_day: date, lag_start: int = 3, lag_end: int = 7) -> List[date]:
    return [cur_day - timedelta(days=lag) for lag in range(lag_end, lag_start - 1, -1)]


def _max_consecutive_true(flags: Sequence[bool]) -> int:
    best = cur = 0
    for flag in flags:
        if flag:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def _measured_weather(rows: List[dict]) -> Dict[date, dict]:
    out: Dict[date, dict] = {}
    for r in rows:
        dd = _d(r.get("weather_date"))
        if dd is None:
            continue
        if str(r.get("data_kind") or "").strip().lower() != "measured" or not bool(r.get("checked")):
            continue
        wind = _f(r.get("wind_avg_ms"))
        rh = _f(r.get("humidity_avg_pct"))
        if wind is None or rh is None:
            continue
        out[dd] = {
            "wind": wind,
            "rh": rh,
            "wind_dry": wind * (100.0 - rh),
            "kind": "M",
        }
    return out


def _live_weather(rows: List[dict]) -> Dict[date, dict]:
    """Prefer checked measured; otherwise use forecast for that date."""
    measured: Dict[date, dict] = {}
    forecast: Dict[date, dict] = {}
    for r in rows:
        dd = _d(r.get("weather_date"))
        if dd is None:
            continue
        wind = _f(r.get("wind_avg_ms"))
        rh = _f(r.get("humidity_avg_pct"))
        if wind is None or rh is None:
            continue
        rec = {"wind": wind, "rh": rh, "wind_dry": wind * (100.0 - rh)}
        kind = str(r.get("data_kind") or "").strip().lower()
        if kind == "measured" and bool(r.get("checked")):
            rec["kind"] = "M"
            measured[dd] = rec
        elif kind == "forecast":
            rec["kind"] = "F"
            forecast[dd] = rec
    out = dict(forecast)
    out.update(measured)
    return out


def _prior_high_threshold(measured: Dict[date, dict], before_day: date) -> Optional[float]:
    vals = [float(w["wind_dry"]) for dd, w in measured.items() if dd < before_day]
    vals = [v for v in vals if np.isfinite(v)]
    if len(vals) < MIN_DAYS_FOR_HIGH_THRESHOLD:
        return None
    return float(np.quantile(np.asarray(vals, dtype=float), HIGH_Q))


def _wd_window(weather_for_values: Dict[date, dict], measured_for_threshold: Dict[date, dict], days: Sequence[date]):
    if not days or any(d not in weather_for_values for d in days):
        return None
    wd = np.asarray([float(weather_for_values[d]["wind_dry"]) for d in days], dtype=float)
    hi = _prior_high_threshold(measured_for_threshold, min(days))
    if hi is None:
        return None
    flags = [bool(v >= hi) for v in wd]
    kinds = [str(weather_for_values[d].get("kind") or "?") for d in days]
    return {
        "avg": float(np.mean(wd)),
        "high_days": float(sum(flags)),
        "high_run": float(_max_consecutive_true(flags)),
        "threshold": float(hi),
        "kinds": kinds,
    }


def _window_features(cur: Event, prev: Event, prevprev: Optional[Event],
                     weather_for_values: Dict[date, dict], measured_for_threshold: Dict[date, dict]):
    current_days = _lag_dates(cur.day, 3, 7)
    current = _wd_window(weather_for_values, measured_for_threshold, current_days)
    if current is None:
        return None
    previous = None
    if prevprev is not None:
        previous = _wd_window(weather_for_values, measured_for_threshold, _lag_dates(prev.day, 3, 7))
        if previous is None:
            return None
    return {
        "l3_7_high_days": float(current["high_days"]),
        "l3_7_high_run": float(current["high_run"]),
        "d_l3_7_high_days": float(current["high_days"] - previous["high_days"]) if previous else 0.0,
        "d_l3_7_high_run": float(current["high_run"] - previous["high_run"]) if previous else 0.0,
        "l3_7_avg": float(current["avg"]),
        "threshold": float(current["threshold"]),
        "window_days": current_days,
        "window_kinds": list(current["kinds"]),
    }


def _historical_df(events: List[Event], measured: Dict[date, dict]) -> pd.DataFrame:
    rows = []
    for field in range(1, 15):
        hist = _field_hist(events, field)
        for i in range(1, len(hist)):
            cur, prev = hist[i], hist[i - 1]
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
            wx = _window_features(cur, prev, prevprev, measured, measured)
            if wx is None:
                continue
            y = math.log((cur_rate + TARGET_EPS) / (prev_rate + TARGET_EPS))
            rows.append({
                "date": cur.day,
                "field": field,
                "order": cur.order,
                "prev_log_rate": math.log(prev_rate + TARGET_EPS),
                "prev_rate": prev_rate,
                "season_day": float((cur.day - SEASON_START).days),
                "growth": growth,
                "growth_delta": growth - prev_growth,
                "y": y,
                "actual_abc": cur.abc,
                "prev_abc": prev.abc,
                **{k: wx[k] for k in LOCKED_EXTRA},
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["date", "order", "field"]).reset_index(drop=True)
    return df


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
        beta = np.linalg.solve(Xd.T @ Xd + reg, Xd.T @ y)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(Xd.T @ Xd + reg) @ (Xd.T @ y)
    return {"mu": mu, "sd": sd, "beta": beta}


def _ridge_pred(model, x: Sequence[float]) -> float:
    x = np.asarray(x, dtype=float)
    z = (x - model["mu"]) / model["sd"]
    return float(np.r_[1.0, z] @ model["beta"])


def _fit_locked_models(df: pd.DataFrame):
    if len(df) < MIN_TRAIN_ROWS:
        raise RuntimeError(f"Treeningridu ainult {len(df)}; vaja vähemalt {MIN_TRAIN_ROWS}.")
    base_cols = list(BASE_COLS)
    cand_cols = base_cols + list(LOCKED_EXTRA)
    y = pd.to_numeric(df["y"], errors="coerce").to_numpy(dtype=float)
    xb = df[base_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    xc = df[cand_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(y) & np.all(np.isfinite(xb), axis=1) & np.all(np.isfinite(xc), axis=1)
    if int(ok.sum()) < MIN_TRAIN_ROWS:
        raise RuntimeError(f"Täielikke treeningridu ainult {int(ok.sum())}.")
    return _ridge_fit(xb[ok], y[ok], RIDGE_ALPHA), _ridge_fit(xc[ok], y[ok], RIDGE_ALPHA), int(ok.sum())


def _walk_forward_locked(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    n = len(df)
    bp = np.full(n, np.nan)
    cp = np.full(n, np.nan)
    dates = np.array(df["date"].tolist(), dtype=object)
    y = pd.to_numeric(df["y"], errors="coerce").to_numpy(dtype=float)
    xb = df[BASE_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    xc = df[BASE_COLS + LOCKED_EXTRA].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    for dd in sorted(set(dates)):
        tr = np.where(dates < dd)[0]
        te = np.where(dates == dd)[0]
        ok = np.isfinite(y[tr]) & np.all(np.isfinite(xb[tr]), axis=1) & np.all(np.isfinite(xc[tr]), axis=1)
        tr = tr[ok]
        if len(tr) < MIN_TRAIN_ROWS:
            continue
        mb = _ridge_fit(xb[tr], y[tr], RIDGE_ALPHA)
        mc = _ridge_fit(xc[tr], y[tr], RIDGE_ALPHA)
        for j in te:
            if np.all(np.isfinite(xb[j])) and np.all(np.isfinite(xc[j])):
                bp[j] = _ridge_pred(mb, xb[j])
                cp[j] = _ridge_pred(mc, xc[j])
    return bp, cp


def _smape_yield_from_log_change(df: pd.DataFrame, pred: np.ndarray) -> float:
    actuals, preds = [], []
    for i, r in df.iterrows():
        if not np.isfinite(pred[i]):
            continue
        prev_rate = float(r["prev_rate"])
        pred_rate = max(0.0, prev_rate * math.exp(float(pred[i])))
        pred_abc = pred_rate * float(r["growth"])
        actuals.append(float(r["actual_abc"]))
        preds.append(pred_abc)
    if not actuals:
        return float("nan")
    a = np.asarray(actuals, dtype=float)
    p = np.asarray(preds, dtype=float)
    den = np.abs(a) + np.abs(p)
    m = den > 1e-9
    return float(np.mean(200.0 * np.abs(p[m] - a[m]) / den[m])) if np.any(m) else float("nan")


def _official_snapshot_batches(rows: List[dict]) -> Dict[date, List[dict]]:
    out: Dict[date, List[dict]] = {}
    by_target: Dict[date, Dict[Tuple[str, str, str], List[dict]]] = {}
    for r in rows:
        td = _d(r.get("target_date"))
        fd = _d(r.get("forecast_date"))
        if td is None or fd is None or fd > TODAY:
            continue
        mv = str(r.get("model_version") or "")
        low = mv.lower()
        if "test" in low or "lab" in low or "3h" in low or "winddry" in low:
            continue
        gen = str(r.get("generated_at") or "")
        key = (fd.isoformat(), gen, mv)
        by_target.setdefault(td, {}).setdefault(key, []).append(r)
    for td, batches in by_target.items():
        ranked = sorted(batches.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2]))
        if ranked:
            out[td] = ranked[-1][1]
    return out


def _next_field(f: int) -> int:
    return 1 if int(f) >= 14 else int(f) + 1


def _prev_field(f: int) -> int:
    return 14 if int(f) <= 1 else int(f) - 1


def _ordered_snapshot_rows(rows: List[dict]) -> List[dict]:
    if len(rows) <= 1:
        return list(rows)
    by_f = {}
    for r in rows:
        try:
            by_f[int(r.get("field_no"))] = r
        except Exception:
            pass
    fields = set(by_f)
    starts = [f for f in fields if _prev_field(f) not in fields]
    start = starts[0] if len(starts) == 1 else min(fields)
    ordered = []
    f = start
    seen = set()
    while f in fields and f not in seen:
        ordered.append(by_f[f])
        seen.add(f)
        f = _next_field(f)
    for f2 in sorted(fields - seen):
        ordered.append(by_f[f2])
    return ordered


def _plant_index_from_basis(row: dict) -> float:
    m = re.search(r"taimeindeks=([0-9]+(?:\.[0-9]+)?)", str(row.get("basis") or ""), flags=re.I)
    if not m:
        return 1.0
    try:
        return float(m.group(1))
    except Exception:
        return 1.0


def _live_feature_row(cur: Event, prev: Event, prevprev: Optional[Event],
                      live_weather: Dict[date, dict], measured: Dict[date, dict]):
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
    wx = _window_features(cur, prev, prevprev, live_weather, measured)
    if wx is None:
        return None
    rec = {
        "prev_log_rate": math.log(prev_rate + TARGET_EPS),
        "season_day": float((cur.day - SEASON_START).days),
        "growth": growth,
        "growth_delta": growth - prev_growth,
        **{k: wx[k] for k in LOCKED_EXTRA},
    }
    return rec, wx


def _live_compare(harvest_rows: List[dict], weather_rows: List[dict], saved: List[dict]):
    events = _prepare_events(harvest_rows)
    measured = _measured_weather(weather_rows)
    live_weather = _live_weather(weather_rows)
    hist_df = _historical_df(events, measured)
    base_model, cand_model, train_n = _fit_locked_models(hist_df)

    # Historical sanity check for this exact locked candidate.
    wf_base, wf_cand = _walk_forward_locked(hist_df)
    hist_base_smape = _smape_yield_from_log_change(hist_df, wf_base)
    hist_cand_smape = _smape_yield_from_log_change(hist_df, wf_cand)

    official = _official_snapshot_batches(saved)
    future_targets = [TODAY + timedelta(days=i) for i in range(1, 10)]
    missing = [d for d in future_targets if not official.get(d)]
    if missing:
        raise RuntimeError("Ametlik snapshot puudub: " + ", ".join(d.strftime("%d.%m") for d in missing))

    state: Dict[int, List[Event]] = {}
    for f in range(1, 15):
        h = _field_hist(events, f)
        if h:
            state[f] = h[-2:] if len(h) >= 2 else h[-1:]

    field_rows = []
    day_rows = []
    snapshot_dates = set()

    for lead, td in enumerate(future_targets, start=1):
        prod_rows = _ordered_snapshot_rows(official[td])
        prod_abc_sum = prod_xl_sum = prod_total_sum = test_abc_sum = 0.0
        day_complete = True
        mults = []
        plant_vals = []
        sources = set()

        for order, pr in enumerate(prod_rows, start=1):
            field = int(pr.get("field_no"))
            pabc = float(pr.get("abc_forecast"))
            pxl = float(pr.get("xl_forecast"))
            ptotal = float(pr.get("total_forecast"))
            plant = _plant_index_from_basis(pr)
            snapshot_dates.add(str(pr.get("forecast_date") or ""))

            hist_state = state.get(field, [])
            prev = hist_state[-1] if hist_state else None
            prevprev = hist_state[-2] if len(hist_state) >= 2 else None
            cur_stub = Event(td, field, order, pabc, _f(pr.get("interval_days")), "official_forecast")

            multiplier = np.nan
            test_abc = np.nan
            info = None
            if prev is not None:
                built = _live_feature_row(cur_stub, prev, prevprev, live_weather, measured)
                if built is not None:
                    rec, wx = built
                    xb = [rec[c] for c in BASE_COLS]
                    xc = [rec[c] for c in BASE_COLS + LOCKED_EXTRA]
                    b = _ridge_pred(base_model, xb)
                    c = _ridge_pred(cand_model, xc)
                    multiplier = math.exp(c - b)
                    test_abc = pabc * multiplier
                    info = (rec, wx, b, c)

            if not np.isfinite(test_abc):
                day_complete = False
                test_abc = pabc
                multiplier = 1.0

            test_total = test_abc + pxl
            prod_abc_sum += pabc
            prod_xl_sum += pxl
            prod_total_sum += ptotal
            test_abc_sum += test_abc
            mults.append(multiplier)
            plant_vals.append(plant)

            if info is not None:
                rec, wx, bpred, cpred = info
                kinds = "".join(wx.get("window_kinds") or [])
                sources.add(kinds)
                window_label = f"{wx['window_days'][0].strftime('%d.%m')}–{wx['window_days'][-1].strftime('%d.%m')}"
                high_days = rec["l3_7_high_days"]
                high_run = rec["l3_7_high_run"]
                d_days = rec["d_l3_7_high_days"]
                d_run = rec["d_l3_7_high_run"]
                wd_avg = wx["l3_7_avg"]
                threshold = wx["threshold"]
            else:
                kinds = "puudulik"
                window_label = "—"
                high_days = high_run = d_days = d_run = wd_avg = threshold = np.nan

            field_rows.append({
                "Päev": td,
                "Lead": lead,
                "Põld": field,
                "Taimeindeks": plant,
                "app-128 ABC": pabc,
                "WD kordaja": multiplier,
                "TEST ABC": test_abc,
                "ABC vahe %": 100.0 * (test_abc / pabc - 1.0) if pabc > 0 else np.nan,
                "app-128 XL": pxl,
                "app-128 kokku": ptotal,
                "TEST kokku": test_total,
                "Kokku vahe %": 100.0 * (test_total / ptotal - 1.0) if ptotal > 0 else np.nan,
                "L3–7 aken": window_label,
                "Ilm M/F": kinds,
                "WD avg": wd_avg,
                "HIGH päevi": high_days,
                "HIGH jada": high_run,
                "Δ HIGH päevi": d_days,
                "Δ HIGH jada": d_run,
                "HIGH lävi": threshold,
            })

            # IMPORTANT: state is advanced with the OFFICIAL app-128 ABC, not TEST ABC.
            # This isolates the layer and avoids feeding its own corrected yield back.
            new_hist = list(hist_state[-1:]) if hist_state else []
            new_hist.append(cur_stub)
            state[field] = new_hist[-2:]

        test_total_sum = test_abc_sum + prod_xl_sum
        day_rows.append({
            "Päev": td,
            "Lead": lead,
            "Põllud": ", ".join(str(int(r.get("field_no"))) for r in prod_rows),
            "app-128 ABC": prod_abc_sum,
            "TEST ABC": test_abc_sum,
            "ABC vahe %": 100.0 * (test_abc_sum / prod_abc_sum - 1.0) if prod_abc_sum > 0 else np.nan,
            "app-128 kokku": prod_total_sum,
            "TEST kokku": test_total_sum,
            "Kokku vahe": test_total_sum - prod_total_sum,
            "Kokku vahe %": 100.0 * (test_total_sum / prod_total_sum - 1.0) if prod_total_sum > 0 else np.nan,
            "Keskm taimeindeks": float(np.mean(plant_vals)) if plant_vals else np.nan,
            "Keskm WD kordaja": float(np.mean(mults)) if mults else np.nan,
            "Kiht täielik": "jah" if day_complete else "ei",
        })

    return pd.DataFrame(day_rows), pd.DataFrame(field_rows), {
        "train_n": train_n,
        "hist_rows": len(hist_df),
        "hist_base_smape": hist_base_smape,
        "hist_cand_smape": hist_cand_smape,
        "measured_days": len(measured),
        "live_weather_days": len(live_weather),
        "latest_measured": max(measured) if measured else None,
        "snapshot_dates": sorted(d for d in snapshot_dates if d),
    }


def _self_test():
    # Structural tests that do not touch DB.
    rows = []
    start = date(2026, 7, 1)
    for i in range(55):
        dd = start + timedelta(days=i)
        rows.append({
            "weather_date": dd.isoformat(), "data_kind": "measured", "checked": True,
            "wind_avg_ms": 3.0 + (i % 6), "humidity_avg_pct": 65.0 + (i % 20),
        })
    measured = _measured_weather(rows)
    assert len(measured) == 55
    ev = [Event(date(2026, 7, 20), 1, 1, 10.0, 5.0), Event(date(2026, 7, 25), 1, 1, 12.0), Event(date(2026, 7, 30), 1, 1, 9.0)]
    wx = _window_features(ev[2], ev[1], ev[0], measured, measured)
    assert wx is not None and "l3_7_high_days" in wx
    fake = [
        {"field_no": 14}, {"field_no": 1}, {"field_no": 2}
    ]
    assert [int(r["field_no"]) for r in _ordered_snapshot_rows(fake)] == [14, 1, 2]
    print("SELF-TEST OK")


def main():
    import streamlit as st
    import db

    st.set_page_config(page_title="KurgiMootor · WIND×DRY TEST", layout="wide")
    st.error("🧪 READ-ONLY TEST · app-128 + lukustatud WIND×DRY L3–7 LEVEL+DELTA · DB-sse EI KIRJUTA")
    st.title("app-128 vs WIND×DRY ilmakiht")
    st.caption(
        "app-128 number sisaldab sama tootmise taimeindeksit. TEST muudab ainult A+B+C osa; "
        "XL jääb täpselt app-128 omaks. Testkiht ei toida oma prognoosi järgmise korje sisendiks tagasi."
    )

    try:
        harvest_rows = db.get_harvest_history(limit=5000)
        weather_rows = db.get_weather_rows(WEATHER_START, TODAY + timedelta(days=9))
        saved = db.get_yield_forecasts(limit=5000) if db.yield_forecasts_available() else []
        daily, fields, diag = _live_compare(harvest_rows, weather_rows, saved)
    except Exception as exc:
        st.exception(exc)
        st.stop()

    if daily.empty:
        st.error("Võrdlusridu ei tekkinud.")
        st.stop()

    max_abs = float(np.nanmax(np.abs(daily["Kokku vahe %"].to_numpy(dtype=float))))
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Treeningridu", int(diag["train_n"]))
    c2.metric("LAB BASE sMAPE", f"{diag['hist_base_smape']:.1f}%" if np.isfinite(diag["hist_base_smape"]) else "—")
    c3.metric("Lukustatud kihi sMAPE", f"{diag['hist_cand_smape']:.1f}%" if np.isfinite(diag["hist_cand_smape"]) else "—")
    c4.metric("Suurim päeva muutus", f"{max_abs:.1f}%")
    lm = diag.get("latest_measured")
    c5.metric("Viimane mõõdetud ilm", lm.strftime("%d.%m") if lm else "—")

    if diag.get("snapshot_dates"):
        st.caption("Ametliku app-128 snapshoti forecast_date: " + ", ".join(diag["snapshot_dates"]))
        if TODAY.isoformat() not in set(diag["snapshot_dates"]):
            st.warning("Kõik ametlikud võrdlusread ei ole tänase forecast_date'iga. Võrdlus on siiski read-only, kuid alus võib olla eelmise jooksu app-128 snapshot.")

    st.markdown("### 9 päeva · päevade võrdlus")
    st.dataframe(
        daily.style.format({
            "Päev": lambda x: x.strftime("%d.%m") if hasattr(x, "strftime") else str(x),
            "app-128 ABC": "{:.1f}", "TEST ABC": "{:.1f}", "ABC vahe %": "{:+.1f}%",
            "app-128 kokku": "{:.1f}", "TEST kokku": "{:.1f}", "Kokku vahe": "{:+.1f}",
            "Kokku vahe %": "{:+.1f}%", "Keskm taimeindeks": "{:.2f}", "Keskm WD kordaja": "{:.3f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("Põldude kaupa · taimeindeks + WIND×DRY detail", expanded=True):
        st.dataframe(
            fields.style.format({
                "Päev": lambda x: x.strftime("%d.%m") if hasattr(x, "strftime") else str(x),
                "Taimeindeks": "{:.2f}", "app-128 ABC": "{:.2f}", "WD kordaja": "{:.3f}",
                "TEST ABC": "{:.2f}", "ABC vahe %": "{:+.1f}%", "app-128 XL": "{:.2f}",
                "app-128 kokku": "{:.2f}", "TEST kokku": "{:.2f}", "Kokku vahe %": "{:+.1f}%",
                "WD avg": "{:.1f}", "HIGH päevi": "{:.0f}", "HIGH jada": "{:.0f}",
                "Δ HIGH päevi": "{:+.0f}", "Δ HIGH jada": "{:+.0f}", "HIGH lävi": "{:.1f}",
            }, na_rep="—"),
            use_container_width=True,
            hide_index=True,
        )

    st.divider()
    st.markdown("### Kuidas seda testi lugeda")
    st.markdown(
        "- **Taimeindeks jääb täpselt app-128 sisse.** TEST ei arvuta uut taimeindeksit ega asenda seda.\n"
        "- **WD kordaja** on ainult lukustatud LAB-154 ilmamustri lisamõju ABC-le. 1.000 = muutust pole.\n"
        "- **Ilm M/F** näitab viie L3–7 päeva allikaid: M=mõõdetud, F=prognoos.\n"
        "- Kui sama põld tuleb 9 päeva sees uuesti, kasutatakse järgmise kihi state'is app-128 ametlikku prognoosi, mitte TEST väljundit. Nii ei teki testkihi enesetagasisidet.\n"
        "- See fail ei käivita Jäljeotsijat ega muuda championit."
    )
    st.caption(
        "READ-ONLY: kasutab ainult db.get_harvest_history, db.get_weather_rows ja db.get_yield_forecasts. "
        "Puuduvad save_*, set_*, delete_* ja upsert_* kutsed."
    )


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        _self_test()
    else:
        main()
