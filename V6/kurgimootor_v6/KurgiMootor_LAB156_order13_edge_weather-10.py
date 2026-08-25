from __future__ import annotations

"""
KurgiMootor LAB-158 · WD L3–7 vs CALENDAR SPLIT
================================================

Deployment
----------
Filename is intentionally kept as KurgiMootor_LAB156_order13_edge_weather.py so the
existing temporary Streamlit LAB app can be overwritten without changing Main file path.
This file replaces LAB-157 completely; no old LAB sections are executed.

Question
--------
LAB-157 unexpectedly showed that fixed calendar windows (L1–3 / L4–6 / L7–10) carried
more strict OOS information than its stripped BASE, while thermal/GDD alignment failed.
This LAB asks the narrow follow-up question:

    On the SAME rows, SAME target and SAME date-wise walk-forward,
    does CALENDAR SPLIT beat the currently locked production WD L3–7 feature block?

Locked variants
---------------
PROD-BASE:
    production-style weather-first base only.
WD-L3–7:
    PROD-BASE + the four locked production WIND×DRY HIGH L3–7 LEVEL+DELTA features.
CALENDAR:
    PROD-BASE + radiation sum and WIND×DRY load in fixed L1–3, L4–6, L7–10 windows.

No previous yield / previous rate / plant index / residual carry is an input.
Target is A+B+C, on the same positive log scale used by production.

Production parity choices
-------------------------
- base feature list copied from the active WD production app;
- WD HIGH threshold: expanding-past Q=.75, weather history from 01.07, min 10 prior days;
- WD window: target-7 ... target-3, level + delta to previous same-field WD window;
- strict date-wise walk-forward, min train 10;
- ridge alpha 10, field alpha 80, z clip 2.5, ABC log epsilon 0.05;
- when repository model_engine.py is available, its abc_growth_walk_predict is used.

Fair-comparison rule
--------------------
All three variants are trained and scored only on the common rows where BOTH locked WD
and CALENDAR features are available. This deliberately sacrifices some early rows to
prevent row-set differences from deciding the result.

Gate (fixed before seeing LAB-158 results)
------------------------------------------
CALENDAR may continue only if it:
1) beats WD-L3–7 in field MAE;
2) beats WD-L3–7 in 3-field full-day MAE;
3) beats WD-L3–7 in recent-5 full-day MAE;
4) wins >50% of comparable full days against WD-L3–7.
No lag boundaries are retuned after the result.

READ ONLY. No DB writes. No hourly API. No ECMWF replay. Production is untouched.
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo
import math
import sys

import numpy as np
import pandas as pd

try:
    from model_engine import (
        temperature_curve_features as _prod_temperature_curve_features,
        abc_growth_walk_predict as _prod_abc_growth_walk_predict,
    )
    MODEL_ENGINE_AVAILABLE = True
except Exception:
    _prod_temperature_curve_features = None
    _prod_abc_growth_walk_predict = None
    MODEL_ENGINE_AVAILABLE = False


# -----------------------------------------------------------------------------
# LOCKED CONFIG
# -----------------------------------------------------------------------------

TZ = ZoneInfo("Europe/Tallinn")
TODAY = datetime.now(TZ).date()
SEASON_START = date(2026, 6, 15)
WEATHER_START = date(2026, 7, 1)
LAB_VERSION = "LAB-158-WD-VS-CALENDAR-PARITY-V1"

MIN_TRAIN_ROWS = 10
RIDGE_ALPHA = 10.0
FIELD_ALPHA = 80.0
Z_CLIP = 2.5
ABC_LOG_EPS = 0.05

WD_HIGH_Q = 0.75
WD_MIN_THRESHOLD_DAYS = 10
WD_COLS = [
    "WD HIGH L3–7 päevad",
    "WD HIGH L3–7 jada",
    "Δ WD HIGH L3–7 päevad",
    "Δ WD HIGH L3–7 jada",
]

CALENDAR_BANDS: Tuple[Tuple[str, int, int], ...] = (
    ("1_3", 1, 3),
    ("4_6", 4, 6),
    ("7_10", 7, 10),
)
CAL_COLS = [f"CAL RAD {name}" for name, _, _ in CALENDAR_BANDS] + [
    f"CAL WD {name}" for name, _, _ in CALENDAR_BANDS
]

# Exact active-production base feature names/order from the WD production app.
BASE_COLS = [
    "Intervall p", "Hooajapäev",
    "Öö jahedus <16", "Öö jahedus² <16",
    "Öö soojus 16-20", "Öö kuumus >20",
    "Päeva jahedus <20", "Päeva jahedus² <20",
    "Päeva soojus 20-28",
    "Päeva kuumus >30", "Päeva kuumus² >30",
    "Radiatsioon Σ", "Radiatsioon/p",
    "Sademed Σ", "Niiskus kesk", "ET0 Σ", "Tuul kesk",
]

VARIANTS: Dict[str, List[str]] = {
    "PROD-BASE": [],
    "WD-L3–7": WD_COLS,
    "CALENDAR": CAL_COLS,
}


# -----------------------------------------------------------------------------
# BASIC HELPERS
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Event:
    day: date
    field: int
    order: int
    a: float
    b: float
    c: float
    quality: str

    @property
    def abc(self) -> float:
        return float(self.a + self.b + self.c)


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


def _usable_quality(v) -> bool:
    q = str(v or "").strip().lower()
    return q not in {"hinnanguline", "ligikaudne", ""}


def _prepare_events(rows: Iterable[dict]) -> List[Event]:
    dedup: Dict[Tuple[date, int], Event] = {}
    for r in rows:
        dd = _d(r.get("harvest_date"))
        ff = _f(r.get("field_no"))
        oo = _f(r.get("harvest_order"))
        aa, bb, cc = (_f(r.get(k)) for k in ("a", "b", "c"))
        if dd is None or ff is None or oo is None or None in (aa, bb, cc):
            continue
        field, order = int(ff), int(oo)
        if not (1 <= field <= 14 and 1 <= order <= 3):
            continue
        q = str(r.get("data_quality") or "").strip()
        # A non-usable quality can still serve as an interval event in production,
        # but this focused audit keeps only measured/derived numeric target events.
        if not _usable_quality(q):
            continue
        e = Event(dd, field, order, float(aa), float(bb), float(cc), q)
        dedup[(dd, field)] = e
    return sorted(dedup.values(), key=lambda e: (e.day, e.order, e.field))


def _weather_map(rows: Iterable[dict]) -> Dict[date, dict]:
    out: Dict[date, dict] = {}
    required = (
        "temp_night_avg_c", "temp_day_avg_c", "temp_min_c", "temp_max_c",
        "wind_avg_ms", "radiation_mj_m2", "humidity_avg_pct",
        "precipitation_mm", "et0_mm",
    )
    for r in rows:
        dd = _d(r.get("weather_date"))
        if dd is None or dd < WEATHER_START:
            continue
        if str(r.get("data_kind") or "").strip().lower() != "measured" or not bool(r.get("checked")):
            continue
        vals = {k: _f(r.get(k)) for k in required}
        if any(vals[k] is None for k in required):
            continue
        vals["data_kind"] = "measured"
        vals["checked"] = True
        out[dd] = vals
    return out


def _wd_value(w: dict) -> float:
    return float(w["wind_avg_ms"]) * (100.0 - float(w["humidity_avg_pct"]))


def _max_run(flags: Sequence[bool]) -> int:
    best = cur = 0
    for flag in flags:
        if flag:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def _wd_threshold(weather: Dict[date, dict], before_day: date) -> Optional[float]:
    vals = [_wd_value(w) for dd, w in weather.items() if WEATHER_START <= dd < before_day]
    if len(vals) < WD_MIN_THRESHOLD_DAYS:
        return None
    return float(np.quantile(np.asarray(vals, dtype=float), WD_HIGH_Q))


def _wd_window(target: date, weather: Dict[date, dict]) -> Optional[dict]:
    days = [target - timedelta(days=lag) for lag in range(7, 2, -1)]  # L7..L3
    rows = [weather.get(dd) for dd in days]
    if any(r is None for r in rows):
        return None
    hi = _wd_threshold(weather, min(days))
    if hi is None:
        return None
    vals = [_wd_value(r) for r in rows]
    flags = [v >= hi for v in vals]
    return {
        "high_days": float(sum(flags)),
        "high_run": float(_max_run(flags)),
        "threshold": float(hi),
        "avg": float(np.mean(vals)),
    }


def _calendar_features(target: date, weather: Dict[date, dict]) -> Optional[dict]:
    rec: Dict[str, float] = {}
    for name, lo, hi in CALENDAR_BANDS:
        rows = [weather.get(target - timedelta(days=lag)) for lag in range(lo, hi + 1)]
        if any(r is None for r in rows):
            return None
        rec[f"CAL RAD {name}"] = float(sum(float(r["radiation_mj_m2"]) for r in rows))
        rec[f"CAL WD {name}"] = float(sum(_wd_value(r) for r in rows))
    return rec


def _fallback_temperature_curve_features(nights: Sequence[float], days: Sequence[float]) -> dict:
    """Self-test fallback only. Deployment uses repository model_engine.py when present."""
    n = np.asarray(nights, dtype=float)
    d = np.asarray(days, dtype=float)
    nc = np.maximum(0.0, 16.0 - n)
    nh = np.maximum(0.0, n - 20.0)
    dc = np.maximum(0.0, 20.0 - d)
    dh = np.maximum(0.0, d - 30.0)
    return {
        "Öö jahedus <16": float(nc.mean()),
        "Öö jahedus² <16": float((nc ** 2).mean()),
        "Öö soojus 16-20": float(np.clip(n - 16.0, 0.0, 4.0).mean()),
        "Öö kuumus >20": float(nh.mean()),
        "Päeva jahedus <20": float(dc.mean()),
        "Päeva jahedus² <20": float((dc ** 2).mean()),
        "Päeva soojus 20-28": float(np.clip(d - 20.0, 0.0, 8.0).mean()),
        "Päeva kuumus >30": float(dh.mean()),
        "Päeva kuumus² >30": float((dh ** 2).mean()),
    }


def _temperature_features(nights: Sequence[float], days: Sequence[float]) -> dict:
    if MODEL_ENGINE_AVAILABLE and _prod_temperature_curve_features is not None:
        return dict(_prod_temperature_curve_features(list(nights), list(days)))
    return _fallback_temperature_curve_features(nights, days)


# -----------------------------------------------------------------------------
# RECORD BUILDING: PRODUCTION-STYLE BASE + BOTH CANDIDATE BLOCKS
# -----------------------------------------------------------------------------

def _build_records(events: List[Event], weather: Dict[date, dict]) -> pd.DataFrame:
    by_field: Dict[int, List[Event]] = {f: [] for f in range(1, 15)}
    for e in events:
        by_field[e.field].append(e)
    for f in by_field:
        by_field[f].sort(key=lambda e: (e.day, e.order))

    rows: List[dict] = []
    for field, hist in by_field.items():
        for i in range(1, len(hist)):
            prev, cur = hist[i - 1], hist[i]
            interval = (cur.day - prev.day).days
            if interval <= 0:
                continue

            # Production historical base window: previous+1 through current, inclusive.
            wrs: List[dict] = []
            dd = prev.day + timedelta(days=1)
            while dd <= cur.day:
                w = weather.get(dd)
                if w is None:
                    wrs = []
                    break
                wrs.append(w)
                dd += timedelta(days=1)
            if not wrs:
                continue

            cal = _calendar_features(cur.day, weather)
            cur_wd = _wd_window(cur.day, weather)
            if cal is None or cur_wd is None:
                continue

            nights = [float(w["temp_night_avg_c"]) for w in wrs]
            days = [float(w["temp_day_avg_c"]) for w in wrs]
            rad = [float(w["radiation_mj_m2"]) for w in wrs]
            rain = [float(w["precipitation_mm"]) for w in wrs]
            rh = [float(w["humidity_avg_pct"]) for w in wrs]
            et0 = [float(w["et0_mm"]) for w in wrs]
            wind = [float(w["wind_avg_ms"]) for w in wrs]

            rec: Dict[str, float | int | date] = {
                "date": cur.day,
                "field": field,
                "order": cur.order,
                "actual_abc": cur.abc,
                "Intervall p": float(interval),
                "Hooajapäev": float((cur.day - SEASON_START).days),
                "Radiatsioon Σ": float(sum(rad)),
                "Radiatsioon/p": float(sum(rad) / len(rad)),
                "Sademed Σ": float(sum(rain)),
                "Niiskus kesk": float(sum(rh) / len(rh)),
                "ET0 Σ": float(sum(et0)),
                "Tuul kesk": float(sum(wind) / len(wind)),
                "_wd_high_days": float(cur_wd["high_days"]),
                "_wd_high_run": float(cur_wd["high_run"]),
                "_wd_threshold": float(cur_wd["threshold"]),
            }
            rec.update(_temperature_features(nights, days))
            rec.update(cal)
            rows.append(rec)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values(["date", "field"]).reset_index(drop=True)

    # Exact production historical delta behavior: previous available WD window in same
    # field among model rows; first model row gets delta=0.
    prev_wd_by_field: Dict[int, Tuple[float, float]] = {}
    wd_seen: set[int] = set()
    out_vals = {c: [] for c in WD_COLS}
    for _, r in df.iterrows():
        field = int(r["field"])
        cur_pair = (float(r["_wd_high_days"]), float(r["_wd_high_run"]))
        prev_pair = prev_wd_by_field.get(field)
        allow_delta = field in wd_seen and prev_pair is not None
        vals = {
            "WD HIGH L3–7 päevad": cur_pair[0],
            "WD HIGH L3–7 jada": cur_pair[1],
            "Δ WD HIGH L3–7 päevad": cur_pair[0] - prev_pair[0] if allow_delta else 0.0,
            "Δ WD HIGH L3–7 jada": cur_pair[1] - prev_pair[1] if allow_delta else 0.0,
        }
        prev_wd_by_field[field] = cur_pair
        wd_seen.add(field)
        for c in WD_COLS:
            out_vals[c].append(float(vals[c]))
    for c in WD_COLS:
        df[c] = np.asarray(out_vals[c], dtype=float)

    # Common-row contract for every variant.
    ready_cols = BASE_COLS + WD_COLS + CAL_COLS + ["actual_abc"]
    finite = np.all(np.isfinite(df[ready_cols].to_numpy(dtype=float)), axis=1)
    return df.loc[finite].copy().reset_index(drop=True)


# -----------------------------------------------------------------------------
# PRODUCTION ENGINE CALL + SELF-TEST FALLBACK
# -----------------------------------------------------------------------------

def _fallback_engine_predict(
    X_base: np.ndarray,
    fields: np.ndarray,
    log_y: np.ndarray,
    extra_arrays: Sequence[np.ndarray],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> np.ndarray:
    """Local smoke-test fallback. Real Streamlit run should use model_engine.py."""
    x = np.column_stack([X_base] + [np.asarray(a, dtype=float).reshape(-1, 1) for a in extra_arrays])
    tr = np.asarray(train_idx, dtype=int)
    te = np.asarray(test_idx, dtype=int)
    ok = np.isfinite(log_y[tr]) & np.all(np.isfinite(x[tr]), axis=1)
    tr = tr[ok]
    if len(tr) < MIN_TRAIN_ROWS:
        return np.full(len(te), np.nan)
    mu = np.mean(x[tr], axis=0)
    sd = np.std(x[tr], axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    ztr = np.clip((x[tr] - mu) / sd, -Z_CLIP, Z_CLIP)
    zte = np.clip((x[te] - mu) / sd, -Z_CLIP, Z_CLIP)
    fd_tr = np.column_stack([(fields[tr] == f).astype(float) for f in range(2, 15)])
    fd_te = np.column_stack([(fields[te] == f).astype(float) for f in range(2, 15)])
    A = np.column_stack([np.ones(len(tr)), ztr, fd_tr])
    B = np.column_stack([np.ones(len(te)), zte, fd_te])
    reg = np.eye(A.shape[1]) * RIDGE_ALPHA
    reg[0, 0] = 0.0
    # Field coefficients receive stronger partial-pooling penalty.
    reg[-13:, -13:] = np.eye(13) * FIELD_ALPHA
    try:
        beta = np.linalg.solve(A.T @ A + reg, A.T @ log_y[tr])
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(A.T @ A + reg) @ (A.T @ log_y[tr])
    return np.maximum(0.0, np.exp(B @ beta))


def _engine_predict(
    X_base: np.ndarray,
    fields: np.ndarray,
    log_y: np.ndarray,
    extra_arrays: Sequence[np.ndarray],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> np.ndarray:
    if MODEL_ENGINE_AVAILABLE and _prod_abc_growth_walk_predict is not None:
        return np.asarray(
            _prod_abc_growth_walk_predict(
                X_base, fields, log_y, list(extra_arrays), train_idx, test_idx,
                min_train_rows=MIN_TRAIN_ROWS,
                alpha=RIDGE_ALPHA,
                field_alpha=FIELD_ALPHA,
                z_clip=Z_CLIP,
                log_eps=ABC_LOG_EPS,
            ),
            dtype=float,
        )
    return _fallback_engine_predict(X_base, fields, log_y, extra_arrays, train_idx, test_idx)


def _walk_forward(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    n = len(out)
    X_base = out[BASE_COLS].to_numpy(dtype=float)
    fields = out["field"].astype(int).to_numpy()
    dates = np.asarray(out["date"].tolist(), dtype=object)
    actual = out["actual_abc"].to_numpy(dtype=float)
    log_y = np.where(np.isfinite(actual) & (actual >= 0), np.log(np.maximum(actual, ABC_LOG_EPS)), np.nan)

    extra_map = {
        name: [out[c].to_numpy(dtype=float) for c in cols]
        for name, cols in VARIANTS.items()
    }
    for name in VARIANTS:
        out[f"pred_{name}"] = np.nan

    for dd in sorted(set(dates)):
        tr = np.where(dates < dd)[0]
        te = np.where(dates == dd)[0]
        if len(tr) < MIN_TRAIN_ROWS or len(te) == 0:
            continue
        # Same row indices for all three models by construction.
        for name in VARIANTS:
            pred = _engine_predict(X_base, fields, log_y, extra_map[name], tr, te)
            if len(pred) == len(te):
                out.loc[te, f"pred_{name}"] = pred

    pred_cols = [f"pred_{name}" for name in VARIANTS]
    ok = np.all(np.isfinite(out[pred_cols].to_numpy(dtype=float)), axis=1)
    return out.loc[ok].copy().reset_index(drop=True)


# -----------------------------------------------------------------------------
# METRICS
# -----------------------------------------------------------------------------

def _full_days(field_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    for dd, g in field_df.groupby("date", sort=True):
        if len(g) != 3 or g["field"].nunique() != 3:
            continue
        r = {"date": dd, "actual": float(g["actual_abc"].sum())}
        for name in VARIANTS:
            r[name] = float(g[f"pred_{name}"].sum())
            r[f"{name}_abs_err"] = abs(r[name] - r["actual"])
        rows.append(r)
    daily = pd.DataFrame(rows)
    if daily.empty:
        return daily
    for name in VARIANTS:
        direction_ok: List[float] = [np.nan]
        vals = daily[name].to_numpy(dtype=float)
        act = daily["actual"].to_numpy(dtype=float)
        for i in range(1, len(daily)):
            da, dp = act[i] - act[i - 1], vals[i] - vals[i - 1]
            if abs(da) < 1e-9:
                direction_ok.append(np.nan)
            else:
                direction_ok.append(float(np.sign(da) == np.sign(dp)))
        daily[f"{name}_direction_ok"] = direction_ok
    return daily


def _metrics(field_df: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    recent = daily.tail(min(5, len(daily))) if not daily.empty else daily
    for name in VARIANTS:
        ferr = np.abs(field_df[f"pred_{name}"].to_numpy(dtype=float) - field_df["actual_abc"].to_numpy(dtype=float))
        derr = daily[f"{name}_abs_err"].to_numpy(dtype=float) if not daily.empty else np.array([])
        rerr = recent[f"{name}_abs_err"].to_numpy(dtype=float) if not recent.empty else np.array([])
        ape = ferr / np.maximum(field_df["actual_abc"].to_numpy(dtype=float), 0.1)
        dirs = daily[f"{name}_direction_ok"].dropna().to_numpy(dtype=float) if not daily.empty else np.array([])
        rows.append({
            "Mudel": name,
            "Põllu MAE": float(np.mean(ferr)) if len(ferr) else np.nan,
            "3-põllu päeva MAE": float(np.mean(derr)) if len(derr) else np.nan,
            "Viimase 5 päeva MAE": float(np.mean(rerr)) if len(rerr) else np.nan,
            "±20% sees %": 100.0 * float(np.mean(ape <= 0.20)) if len(ape) else np.nan,
            "Lainete suund %": 100.0 * float(np.mean(dirs)) if len(dirs) else np.nan,
        })
    return pd.DataFrame(rows)


def _gate(metrics: pd.DataFrame, daily: pd.DataFrame) -> Tuple[bool, List[str]]:
    mm = metrics.set_index("Mudel")
    def better(col: str) -> bool:
        return bool(float(mm.loc["CALENDAR", col]) < float(mm.loc["WD-L3–7", col]))
    c1 = better("Põllu MAE")
    c2 = better("3-põllu päeva MAE")
    c3 = better("Viimase 5 päeva MAE")
    if daily.empty:
        win = 0.0
    else:
        win = 100.0 * float(np.mean(daily["CALENDAR_abs_err"].to_numpy(float) < daily["WD-L3–7_abs_err"].to_numpy(float)))
    c4 = win > 50.0
    reasons = [
        f"Põllu MAE CALENDAR < WD-L3–7: {'JAH' if c1 else 'EI'}",
        f"Päeva MAE CALENDAR < WD-L3–7: {'JAH' if c2 else 'EI'}",
        f"Viimase 5 päeva MAE CALENDAR < WD-L3–7: {'JAH' if c3 else 'EI'}",
        f"CALENDAR võidab >50% täispäevi: {'JAH' if c4 else 'EI'} ({win:.0f}%)",
    ]
    return bool(c1 and c2 and c3 and c4), reasons


def _worst_wd_days(daily: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    if daily.empty:
        return daily
    return daily.sort_values("WD-L3–7_abs_err", ascending=False).head(n).sort_values("date")


# -----------------------------------------------------------------------------
# SELF TEST
# -----------------------------------------------------------------------------

def _self_test() -> None:
    weather: Dict[date, dict] = {}
    start = date(2026, 7, 1)
    for i in range(90):
        dd = start + timedelta(days=i)
        night = 15.0 + 2.5 * math.sin(i / 7.0)
        dayt = 22.0 + 3.5 * math.sin(i / 6.0)
        wind = 1.8 + 1.0 * max(0.0, math.sin(i / 4.0))
        rh = 75.0 + 12.0 * math.cos(i / 5.0)
        weather[dd] = {
            "temp_night_avg_c": night, "temp_day_avg_c": dayt,
            "temp_min_c": night - 2, "temp_max_c": dayt + 2,
            "wind_avg_ms": wind, "radiation_mj_m2": 14.0 + 8.0 * max(0.0, math.sin(i / 4.5)),
            "humidity_avg_pct": rh, "precipitation_mm": max(0.0, 3.0 * math.sin(i / 3.0)),
            "et0_mm": 2.0 + 0.5 * math.sin(i / 4.0), "data_kind": "measured", "checked": True,
        }

    events: List[Event] = []
    last: Dict[int, date] = {}
    d0 = date(2026, 7, 13)
    for j in range(55):
        dd = d0 + timedelta(days=j)
        fields = [((3*j + k) % 14) + 1 for k in range(3)]
        for order, field in enumerate(fields, 1):
            if field in last:
                cal = _calendar_features(dd, weather)
                sig = 0.0 if cal is None else 0.015 * cal["CAL RAD 4_6"] - 0.0008 * cal["CAL WD 1_3"]
                abc = max(1.0, 7.0 + 0.08*field + sig + 0.3*math.sin(j/3.0))
            else:
                abc = 7.0 + 0.08*field
            events.append(Event(dd, field, order, 0.2, abc*0.45, abc*0.55-0.2, "Kinnitatud"))
            last[field] = dd

    df = _build_records(events, weather)
    assert len(df) > 30, len(df)
    wf = _walk_forward(df)
    assert len(wf) > 10, len(wf)
    daily = _full_days(wf)
    assert not daily.empty
    metrics = _metrics(wf, daily)
    assert set(metrics["Mudel"]) == set(VARIANTS)
    assert all(np.isfinite(wf[f"pred_{m}"]).all() for m in VARIANTS)
    print(f"{LAB_VERSION} SELF-TEST OK · engine={'production' if MODEL_ENGINE_AVAILABLE else 'fallback'} · records={len(df)} · oos={len(wf)} · days={len(daily)}")


# -----------------------------------------------------------------------------
# STREAMLIT UI
# -----------------------------------------------------------------------------

def main() -> None:
    import streamlit as st
    import db

    st.set_page_config(page_title="KurgiMootor LAB-158", layout="wide")
    st.error("🧪 LAB-158 · WD L3–7 vs CALENDAR SPLIT · READ ONLY")
    st.title("Kas üks L3–7 stressiaken surub olulise ajastuse kokku?")
    st.caption(
        "Üks otsustav võrdlus: aktiivse WD feature-bloki struktuur vs LAB-157-st tulnud "
        "L1–3 / L4–6 / L7–10 kalenderjaotus. Sama production-style baas, sama target, samad read, sama walk-forward."
    )

    with st.expander("Katse lukud", expanded=False):
        st.markdown(
            f"""
- **PROD-BASE:** aktiivse productioni weather-first baastunnused.
- **WD-L3–7:** PROD-BASE + täpselt 4 lukustatud WD HIGH LEVEL+DELTA tunnust; Q={WD_HIGH_Q:.2f}, min {WD_MIN_THRESHOLD_DAYS} varasemat päeva.
- **CALENDAR:** PROD-BASE + ainult `radiatsioon` ja `WIND×DRY` akendes **L1–3, L4–6, L7–10**.
- **Target:** A+B+C, productioni log-skaala; `ABC_LOG_EPS={ABC_LOG_EPS}`.
- **Previous yield/rate puudub** kõigi variantide sisenditest. Taimeindeks puudub.
- **Strict date-wise walk-forward:** min train {MIN_TRAIN_ROWS}; ridge α={RIDGE_ALPHA:g}; field α={FIELD_ALPHA:g}; z-clip={Z_CLIP:g}.
- Kõik kolm mudelit saavad **täpselt samad common feature-ready read**.
- CALENDAR aknaid ega WD läve pärast tulemust ei muudeta.
- Ainult kontrollitud mõõdetud `weather_daily`; hourly/API/ECMWF puudub. DB-sse ei kirjutata.
            """
        )

    if MODEL_ENGINE_AVAILABLE:
        st.success("✅ Repo model_engine.py leitud — walk-forward kasutab production abc_growth_walk_predict funktsiooni.")
    else:
        st.error("⛔ model_engine.py ei ole selles deploy's leitav. Ära kasuta tulemust production-parity otsuseks.")

    st.info("CPU-hoid: kaks DB lugemist + kolm väikest walk-forward mudelit. Võrgu-ilma allalaadimist ei ole.")

    if st.button("▶ Jooksuta WD vs CALENDAR kontroll", type="primary"):
        st.session_state.pop("lab158_result", None)
        try:
            harvest_rows = db.get_harvest_history(limit=5000)
            weather_rows = db.get_weather_rows(WEATHER_START, TODAY)
            events = _prepare_events(harvest_rows)
            weather = _weather_map(weather_rows)
            records = _build_records(events, weather)
            if records.empty:
                st.error("Common feature-ready ridu ei tekkinud. Kontrolli mõõdetud ilma ajalugu.")
                st.stop()
            wf = _walk_forward(records)
            daily = _full_days(wf)
            if wf.empty or daily.empty:
                st.error(f"Strict OOS jaoks pole piisavalt ridu. Common rows={len(records)}, min train={MIN_TRAIN_ROWS}.")
                st.stop()
            metrics = _metrics(wf, daily)
            passed, reasons = _gate(metrics, daily)
            st.session_state["lab158_result"] = {
                "records": records, "wf": wf, "daily": daily, "metrics": metrics,
                "passed": passed, "reasons": reasons,
                "latest_weather": max(weather) if weather else None,
            }
        except Exception as exc:
            st.exception(exc)
            st.stop()

    result = st.session_state.get("lab158_result")
    if not result:
        st.stop()

    records: pd.DataFrame = result["records"]
    wf: pd.DataFrame = result["wf"]
    daily: pd.DataFrame = result["daily"]
    metrics: pd.DataFrame = result["metrics"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Common feature-ready", len(records))
    c2.metric("Strict OOS ridu", len(wf))
    c3.metric("Täispäevi", len(daily))
    lw = result.get("latest_weather")
    c4.metric("Mõõdetud ilm kuni", lw.strftime("%d.%m") if lw else "—")

    st.markdown("### 1. Otsustav tulemus")
    st.dataframe(
        metrics.style.format({
            "Põllu MAE": "{:.3f}", "3-põllu päeva MAE": "{:.3f}",
            "Viimase 5 päeva MAE": "{:.3f}", "±20% sees %": "{:.0f}%",
            "Lainete suund %": lambda x: "—" if pd.isna(x) else f"{x:.0f}%",
        }), use_container_width=True, hide_index=True,
    )

    if result["passed"] and MODEL_ENGINE_AVAILABLE:
        st.success(
            "✅ CALENDAR VÄRAV PASS. Sama production-style raamistikus kannab ajaliselt jagatud "
            "L1–3/L4–6/L7–10 ilm rohkem OOS infot kui lukustatud WD L3–7 feature-blokk."
        )
    elif result["passed"]:
        st.warning("⚠️ Numbriline värav PASS, kuid production model_engine puudub — parity otsust ei tee.")
    else:
        st.warning(
            "⛔ CALENDAR VÄRAV EI LÄINUD LÄBI. LAB-157 hea CALENDAR tulemus ei kandunud "
            "lukustatud WD-ga samasse production-style võrdlusse. Aknaid ei häälestata ümber."
        )
    for r in result["reasons"]:
        st.write("• " + r)

    st.markdown("### 2. Viimased kuni 12 täispäeva")
    show = daily.tail(12).copy()
    show["Kuupäev"] = show["date"].map(lambda d: pd.Timestamp(d).strftime("%d.%m"))
    show["Tegelik ABC"] = show["actual"]
    for name in VARIANTS:
        show[f"{name} viga"] = show[name] - show["actual"]
    cols = ["Kuupäev", "Tegelik ABC", "PROD-BASE", "WD-L3–7", "CALENDAR", "WD-L3–7 viga", "CALENDAR viga"]
    st.dataframe(
        show[cols].style.format({c: "{:.1f}" for c in cols if c != "Kuupäev"}),
        use_container_width=True, hide_index=True,
    )

    st.markdown("### 3. WD viis halvimat päeva · kas CALENDAR päästab just suured möödapanekud?")
    bad = _worst_wd_days(daily, 5).copy()
    bad["Kuupäev"] = bad["date"].map(lambda d: pd.Timestamp(d).strftime("%d.%m"))
    bad["WD viga"] = bad["WD-L3–7"] - bad["actual"]
    bad["CAL viga"] = bad["CALENDAR"] - bad["actual"]
    bad["Δ abs viga CAL-WD"] = bad["CALENDAR_abs_err"] - bad["WD-L3–7_abs_err"]
    st.dataframe(
        bad[["Kuupäev", "actual", "WD-L3–7", "CALENDAR", "WD viga", "CAL viga", "Δ abs viga CAL-WD"]]
        .rename(columns={"actual": "Tegelik ABC"})
        .style.format({
            "Tegelik ABC": "{:.1f}", "WD-L3–7": "{:.1f}", "CALENDAR": "{:.1f}",
            "WD viga": "{:+.1f}", "CAL viga": "{:+.1f}", "Δ abs viga CAL-WD": "{:+.1f}",
        }), use_container_width=True, hide_index=True,
    )
    st.caption("Negatiivne Δ abs viga = CALENDAR parandas WD vea. See tabel on diagnostika, mitte eraldi värav.")

    st.markdown("### 4. Parity / lekkeaudit")
    st.success("✅ Previous yield / previous rate / plant index EI ole sisend")
    st.success("✅ WD Q=.75, min 10 päeva, L3–7 LEVEL+DELTA, ilma algus 01.07")
    st.success("✅ BASE feature-list ja ABC log-target on production-stiilis")
    st.success("✅ Sama päeva põllud ei õpeta üksteist: train date < test date")
    st.success("✅ Kõik variandid hinnatakse samadel common ridadel")
    st.success("✅ Ainult mõõdetud daily DB; hourly/API/ECMWF puudub")
    st.info(f"Versioon {LAB_VERSION} · model_engine={'production' if MODEL_ENGINE_AVAILABLE else 'fallback'} · failinimi hoitud vana Streamlit Main path'i pärast")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        _self_test()
    else:
        main()
