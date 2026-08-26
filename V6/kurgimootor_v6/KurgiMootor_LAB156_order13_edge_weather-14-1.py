from __future__ import annotations

"""
KurgiMootor · edge_weather-14
=============================

ONE-PURPOSE READ-ONLY AUDIT

Question:
    Would the already locked WIND×DRY HIGH L3–7 LEVEL+DELTA layer have improved
    the saved production Lead-0 forecasts on 14–20 Aug 2026?

Output:
    Production ABC -> WD-corrected ABC -> Actual ABC

Important:
- STRICT cutoff per target day: model fit uses only harvests BEFORE that day.
- Target-day actual yield is used only for evaluation, never as an input.
- WIND×DRY L3–7 uses days 3..7 before harvest, so Lead-0 uses already measured past weather.
- The HIGH threshold uses only measured weather before the evaluated L3–7 window.
- The layer is the same locked LAB-154 comparator idea:
      WIND×DRY = wind_avg_ms * (100 - humidity_avg_pct)
      HIGH level + delta vs previous same-field harvest-cycle L3–7
- The layer is applied multiplicatively to the saved production ABC.
- Production's own plant index remains inside the saved production number.
- No DB writes. No weather API refresh. No research search.
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
# Locked test constants
# ---------------------------------------------------------------------

FOCUS_START = date(2026, 8, 14)
FOCUS_END = date(2026, 8, 20)
FOCUS_DAYS = [FOCUS_START + timedelta(days=i)
              for i in range((FOCUS_END - FOCUS_START).days + 1)]

SEASON_START = date(2026, 6, 15)
WEATHER_START = date(2026, 7, 1)

DEFAULT_PROD_MODEL = "v6.5-v18-complete-daily-research-observation-snapshot"

HOURS_PER_FIELD = 3.0
RIDGE_ALPHA = 10.0
MIN_TRAIN_ROWS = 24
TARGET_EPS = 0.20
HIGH_Q = 0.75
MIN_DAYS_FOR_HIGH_THRESHOLD = 10

BASE_COLS = ["prev_log_rate", "season_day", "growth", "growth_delta"]
LOCKED_EXTRA = [
    "l3_7_high_days",
    "l3_7_high_run",
    "d_l3_7_high_days",
    "d_l3_7_high_run",
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

@dataclass
class Event:
    day: date
    field: int
    order: int
    abc: float
    interval_days: Optional[float] = None


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


def _is_old_production(row: dict) -> bool:
    mv = str(row.get("model_version") or "").strip().lower()
    if not mv:
        return False
    return not any(x in mv for x in ("lab", "test", "3h", "winddry"))


def _next_field(f: int) -> int:
    return 1 if int(f) >= 14 else int(f) + 1


def _prev_field(f: int) -> int:
    return 14 if int(f) <= 1 else int(f) - 1


def _ordered_snapshot_rows(rows: Sequence[dict]) -> List[dict]:
    """Order a normal three-field harvest block across 14->1 wrap."""
    by_f: Dict[int, dict] = {}
    for r in rows:
        try:
            by_f[int(r.get("field_no"))] = r
        except Exception:
            pass
    fields = set(by_f)
    if len(fields) <= 1:
        return list(by_f.values())

    starts = [f for f in fields if _prev_field(f) not in fields]
    start = starts[0] if len(starts) == 1 else min(fields)

    ordered = []
    seen = set()
    f = start
    while f in fields and f not in seen:
        ordered.append(by_f[f])
        seen.add(f)
        f = _next_field(f)

    for f2 in sorted(fields - seen):
        ordered.append(by_f[f2])
    return ordered


# ---------------------------------------------------------------------
# Harvest history
# ---------------------------------------------------------------------

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
            day=dd,
            field=field,
            order=order,
            abc=float(abc),
            interval_days=_f(r.get("interval_days")),
        ))
    out.sort(key=lambda e: (e.day, e.order, e.field))
    return out


def _field_hist(events: Sequence[Event], field: int) -> List[Event]:
    return sorted([e for e in events if e.field == field],
                  key=lambda e: (e.day, e.order, e.field))


def _actual_day(rows: List[dict], target: date) -> Tuple[Tuple[int, ...], Optional[float]]:
    by_field: Dict[int, float] = {}
    for r in rows:
        if _d(r.get("harvest_date")) != target or not _reliable(r):
            continue
        try:
            f = int(r.get("field_no"))
        except Exception:
            continue
        a = _abc(r)
        if a is None:
            continue
        by_field[f] = float(a)
    if not by_field:
        return tuple(), None
    fields = tuple(sorted(by_field))
    return fields, float(sum(by_field.values()))


def _growth_days(prev: Event, cur: Event) -> float:
    g = float((cur.day - prev.day).days) + (cur.order - prev.order) * (HOURS_PER_FIELD / 24.0)
    return max(0.5, g)


# ---------------------------------------------------------------------
# Locked WIND×DRY L3–7 features
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
    # chronological: oldest -> newest
    return [cur_day - timedelta(days=lag)
            for lag in range(lag_end, lag_start - 1, -1)]


def _max_consecutive_true(flags: Sequence[bool]) -> int:
    best = cur = 0
    for flag in flags:
        if flag:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def _prior_high_threshold(measured: Dict[date, dict], before_day: date) -> Optional[float]:
    # STRICT: threshold has no access to window itself or future dates.
    vals = [
        float(w["wind_dry"])
        for dd, w in measured.items()
        if dd < before_day and np.isfinite(float(w["wind_dry"]))
    ]
    if len(vals) < MIN_DAYS_FOR_HIGH_THRESHOLD:
        return None
    return float(np.quantile(np.asarray(vals, dtype=float), HIGH_Q))


def _wd_window(measured: Dict[date, dict], days: Sequence[date]):
    if not days or any(d not in measured for d in days):
        return None
    wd = np.asarray([float(measured[d]["wind_dry"]) for d in days], dtype=float)
    threshold = _prior_high_threshold(measured, min(days))
    if threshold is None:
        return None
    flags = [bool(v >= threshold) for v in wd]
    return {
        "avg": float(np.mean(wd)),
        "high_days": float(sum(flags)),
        "high_run": float(_max_consecutive_true(flags)),
        "threshold": float(threshold),
        "days": list(days),
    }


def _window_features(cur: Event, prev: Event, prevprev: Optional[Event],
                     measured: Dict[date, dict]):
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
# Locked ridge ratio layer
# ---------------------------------------------------------------------

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

            wx = _window_features(cur, prev, prevprev, measured)
            if wx is None:
                continue

            y = math.log((cur_rate + TARGET_EPS) / (prev_rate + TARGET_EPS))
            rows.append({
                "date": cur.day,
                "field": field,
                "prev_log_rate": math.log(prev_rate + TARGET_EPS),
                "season_day": float((cur.day - SEASON_START).days),
                "growth": growth,
                "growth_delta": growth - prev_growth,
                "y": y,
                **{k: wx[k] for k in LOCKED_EXTRA},
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["date", "field"]).reset_index(drop=True)
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
        return None
    base_cols = list(BASE_COLS)
    cand_cols = base_cols + list(LOCKED_EXTRA)

    y = pd.to_numeric(df["y"], errors="coerce").to_numpy(dtype=float)
    xb = df[base_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    xc = df[cand_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    ok = np.isfinite(y) & np.all(np.isfinite(xb), axis=1) & np.all(np.isfinite(xc), axis=1)
    if int(ok.sum()) < MIN_TRAIN_ROWS:
        return None

    return (
        _ridge_fit(xb[ok], y[ok], RIDGE_ALPHA),
        _ridge_fit(xc[ok], y[ok], RIDGE_ALPHA),
        int(ok.sum()),
    )


def _feature_row(target: date, order: int, field: int, interval_days: Optional[float],
                 events_before: List[Event], measured: Dict[date, dict]):
    hist = _field_hist(events_before, field)
    if not hist:
        return None

    prev = hist[-1]
    prevprev = hist[-2] if len(hist) >= 2 else None
    cur = Event(target, field, order, 0.0, interval_days)

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
        "growth": growth,
        "growth_delta": growth - prev_growth,
        **{k: wx[k] for k in LOCKED_EXTRA},
    }
    return rec, wx


# ---------------------------------------------------------------------
# Production snapshots
# ---------------------------------------------------------------------

def _choose_prod_model(saved: List[dict]) -> str:
    counts: Dict[str, int] = {}
    for r in saved:
        td = _d(r.get("target_date"))
        fd = _d(r.get("forecast_date"))
        if td not in FOCUS_DAYS or fd != td or not _is_old_production(r):
            continue
        mv = str(r.get("model_version") or "").strip()
        counts[mv] = counts.get(mv, 0) + 1

    if counts.get(DEFAULT_PROD_MODEL, 0) > 0:
        return DEFAULT_PROD_MODEL
    if not counts:
        return ""
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _lead0_batch(saved: List[dict], target: date, model_version: str) -> List[dict]:
    candidates = []
    for r in saved:
        if str(r.get("model_version") or "") != model_version:
            continue
        td = _d(r.get("target_date"))
        fd = _d(r.get("forecast_date"))
        if td != target or fd != target:
            continue
        lead = r.get("lead_days")
        try:
            lead = int(lead) if lead is not None else 0
        except Exception:
            continue
        if lead != 0:
            continue
        try:
            field = int(r.get("field_no"))
        except Exception:
            continue
        if not 1 <= field <= 14:
            continue
        if _f(r.get("abc_forecast")) is None:
            continue
        candidates.append(r)

    # Dedupe same field: keep latest generated_at.
    by_field: Dict[int, dict] = {}
    for r in candidates:
        f = int(r.get("field_no"))
        old = by_field.get(f)
        if old is None or str(r.get("generated_at") or "") >= str(old.get("generated_at") or ""):
            by_field[f] = r

    return _ordered_snapshot_rows(list(by_field.values()))


# ---------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------

def _audit(harvest: List[dict], weather_rows: List[dict], saved: List[dict]):
    measured = _measured_weather(weather_rows)
    model_version = _choose_prod_model(saved)
    if not model_version:
        raise RuntimeError("14.–20.08 Lead-0 vana production snapshotte ei leitud.")

    day_rows = []
    field_rows = []

    for target in FOCUS_DAYS:
        batch = _lead0_batch(saved, target, model_version)
        actual_fields, actual_abc = _actual_day(harvest, target)

        if len(batch) != 3 or actual_abc is None:
            day_rows.append({
                "Päev": target,
                "Plaan": "puudulik",
                "Production ABC": np.nan,
                "WD ABC": np.nan,
                "Tegelik ABC": actual_abc,
                "Production viga": np.nan,
                "WD viga": np.nan,
                "Parandus": np.nan,
                "Keskm WD kordaja": np.nan,
                "Treening N": np.nan,
                "Prod põllud": ",".join(str(int(r.get("field_no"))) for r in batch),
                "Tegelik põllud": ",".join(map(str, actual_fields)),
                "Kiht täielik": "ei",
            })
            continue

        events_before = _prepare_events(harvest, before_day=target)
        hist_df = _historical_df(events_before, measured)
        fitted = _fit_locked_models(hist_df)

        prod_fields = tuple(sorted(int(r.get("field_no")) for r in batch))
        plan_exact = prod_fields == actual_fields

        prod_sum = 0.0
        wd_sum = 0.0
        mults = []
        high_days_vals = []
        delta_days_vals = []
        complete = fitted is not None
        train_n = fitted[2] if fitted is not None else None

        for order, pr in enumerate(batch, start=1):
            field = int(pr.get("field_no"))
            pabc = float(pr.get("abc_forecast"))
            prod_sum += pabc

            multiplier = 1.0
            wx = None

            if fitted is not None:
                base_model, cand_model, _ = fitted
                built = _feature_row(
                    target=target,
                    order=order,
                    field=field,
                    interval_days=_f(pr.get("interval_days")),
                    events_before=events_before,
                    measured=measured,
                )
                if built is None:
                    complete = False
                else:
                    rec, wx = built
                    xb = [rec[c] for c in BASE_COLS]
                    xc = [rec[c] for c in BASE_COLS + LOCKED_EXTRA]
                    b = _ridge_pred(base_model, xb)
                    c = _ridge_pred(cand_model, xc)
                    multiplier = float(math.exp(c - b))
            else:
                complete = False

            if not math.isfinite(multiplier):
                multiplier = 1.0
                complete = False

            wd_abc = pabc * multiplier
            wd_sum += wd_abc
            mults.append(multiplier)

            if wx is not None:
                high_days_vals.append(float(wx["l3_7_high_days"]))
                delta_days_vals.append(float(wx["d_l3_7_high_days"]))

            field_rows.append({
                "Päev": target,
                "Põld": field,
                "Order": order,
                "Production ABC": pabc,
                "WD kordaja": multiplier,
                "WD ABC": wd_abc,
                "HIGH päevi": float(wx["l3_7_high_days"]) if wx else np.nan,
                "HIGH jada": float(wx["l3_7_high_run"]) if wx else np.nan,
                "Δ HIGH päevi": float(wx["d_l3_7_high_days"]) if wx else np.nan,
                "Δ HIGH jada": float(wx["d_l3_7_high_run"]) if wx else np.nan,
                "WD avg": float(wx["l3_7_avg"]) if wx else np.nan,
                "HIGH lävi": float(wx["threshold"]) if wx else np.nan,
            })

        prod_err = prod_sum - actual_abc
        wd_err = wd_sum - actual_abc
        prod_abs = abs(prod_err)
        wd_abs = abs(wd_err)

        day_rows.append({
            "Päev": target,
            "Plaan": "✓" if plan_exact else "EI",
            "Production ABC": prod_sum,
            "WD ABC": wd_sum,
            "Tegelik ABC": actual_abc,
            "Production viga": prod_err,
            "WD viga": wd_err,
            "Parandus": prod_abs - wd_abs,
            "Keskm WD kordaja": float(np.mean(mults)) if mults else np.nan,
            "Treening N": train_n,
            "Prod põllud": ",".join(map(str, prod_fields)),
            "Tegelik põllud": ",".join(map(str, actual_fields)),
            "Kiht täielik": "jah" if complete else "ei",
            "Keskm HIGH päevi": float(np.mean(high_days_vals)) if high_days_vals else np.nan,
            "Keskm Δ HIGH päevi": float(np.mean(delta_days_vals)) if delta_days_vals else np.nan,
        })

    return pd.DataFrame(day_rows), pd.DataFrame(field_rows), model_version, measured


def _mae(df: pd.DataFrame, col: str) -> float:
    vals = pd.to_numeric(df[col], errors="coerce").dropna()
    return float(np.mean(np.abs(vals))) if len(vals) else float("nan")


def _mape(df: pd.DataFrame, pred_col: str) -> float:
    p = pd.to_numeric(df[pred_col], errors="coerce")
    a = pd.to_numeric(df["Tegelik ABC"], errors="coerce")
    ok = p.notna() & a.notna() & (a.abs() > 1e-9)
    if not ok.any():
        return float("nan")
    return float(np.mean(np.abs(p[ok] - a[ok]) / np.abs(a[ok])) * 100.0)


# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------

def main():
    st.set_page_config(page_title="KurgiMootor · WD Lead0 audit", layout="wide")
    st.title("Production → WIND×DRY → tegelik")
    st.caption("14.–20.08 · Lead 0 · üks test · STRICT cutoff · READ ONLY")
    st.info(
        "Küsimus: kas lukustatud WIND×DRY HIGH L3–7 LEVEL+DELTA oleks vähendanud "
        "productioni 14.–20.08 ülepakkumist? Target-päeva saaki ei kasutata mudeli sisendina."
    )

    try:
        harvest = db.get_harvest_history(limit=5000)
        weather = db.get_weather_rows(WEATHER_START, FOCUS_END)
        saved = db.get_yield_forecasts(limit=5000) if db.yield_forecasts_available() else []
        daily, fields, model_version, measured = _audit(harvest, weather, saved)
    except Exception as exc:
        st.exception(exc)
        st.stop()

    valid = daily[
        daily["Production ABC"].notna()
        & daily["WD ABC"].notna()
        & daily["Tegelik ABC"].notna()
    ].copy()
    exact = valid[(valid["Plaan"] == "✓") & (valid["Kiht täielik"] == "jah")].copy()

    st.caption(f"Production snapshot model_version: {model_version}")
    st.caption(
        f"Mõõdetud WIND×DRY ilm: {min(measured).strftime('%d.%m') if measured else '—'}"
        f"–{max(measured).strftime('%d.%m') if measured else '—'} · {len(measured)} päeva"
    )

    if valid.empty:
        st.error("Võrreldavaid päevi ei tekkinud.")
        st.stop()

    # Decision metrics are exact-plan only: isolates yield model from scheduling error.
    prod_mae = _mae(exact, "Production viga")
    wd_mae = _mae(exact, "WD viga")
    prod_mape = _mape(exact, "Production ABC")
    wd_mape = _mape(exact, "WD ABC")
    wins = int((exact["WD viga"].abs() < exact["Production viga"].abs()).sum()) if len(exact) else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Exact-plan päevi", f"{len(exact)}")
    c2.metric("Production MAE", f"{prod_mae:.2f}" if np.isfinite(prod_mae) else "—")
    c3.metric(
        "WD MAE",
        f"{wd_mae:.2f}" if np.isfinite(wd_mae) else "—",
        delta=(f"{prod_mae-wd_mae:+.2f} kasti parem"
               if np.isfinite(prod_mae) and np.isfinite(wd_mae) else None),
    )
    c4.metric(
        "MAPE",
        (f"{prod_mape:.1f}% → {wd_mape:.1f}%"
         if np.isfinite(prod_mape) and np.isfinite(wd_mape) else "—"),
    )
    c5.metric("WD võidab", f"{wins}/{len(exact)} päeva" if len(exact) else "—")

    if len(exact) >= 4 and np.isfinite(prod_mae) and np.isfinite(wd_mae):
        improvement = (prod_mae - wd_mae) / prod_mae if prod_mae > 0 else 0.0
        if improvement >= 0.15 and wins >= max(3, math.ceil(0.6 * len(exact))):
            st.success(
                f"✅ NARROW TEST POSITIIVNE: exact-plan MAE paraneb "
                f"{100*improvement:.0f}% ({prod_mae:.2f} → {wd_mae:.2f}) "
                f"ja WD võidab {wins}/{len(exact)} päeva."
            )
        elif wd_mae < prod_mae:
            st.warning(
                f"🟡 WD aitab, kuid tõend on veel piiripealne: MAE "
                f"{prod_mae:.2f} → {wd_mae:.2f}, võite {wins}/{len(exact)}."
            )
        else:
            st.error(
                f"❌ See lukustatud WD kiht ei paranda seda perioodi: MAE "
                f"{prod_mae:.2f} → {wd_mae:.2f}. Productionisse viimiseks põhjust ei ole."
            )

    st.markdown("### Üks põhitabel")
    show = daily[[
        "Päev", "Plaan", "Prod põllud", "Tegelik põllud",
        "Production ABC", "Keskm WD kordaja", "WD ABC", "Tegelik ABC",
        "Production viga", "WD viga", "Parandus",
        "Keskm HIGH päevi", "Keskm Δ HIGH päevi", "Treening N", "Kiht täielik",
    ]].copy()

    st.dataframe(
        show.style.format({
            "Päev": lambda x: x.strftime("%d.%m") if hasattr(x, "strftime") else str(x),
            "Production ABC": lambda x: "—" if pd.isna(x) else f"{float(x):.1f}",
            "Keskm WD kordaja": lambda x: "—" if pd.isna(x) else f"{float(x):.3f}",
            "WD ABC": lambda x: "—" if pd.isna(x) else f"{float(x):.1f}",
            "Tegelik ABC": lambda x: "—" if pd.isna(x) else f"{float(x):.1f}",
            "Production viga": lambda x: "—" if pd.isna(x) else f"{float(x):+.1f}",
            "WD viga": lambda x: "—" if pd.isna(x) else f"{float(x):+.1f}",
            "Parandus": lambda x: "—" if pd.isna(x) else f"{float(x):+.1f}",
            "Keskm HIGH päevi": lambda x: "—" if pd.isna(x) else f"{float(x):.1f}",
            "Keskm Δ HIGH päevi": lambda x: "—" if pd.isna(x) else f"{float(x):+.1f}",
            "Treening N": lambda x: "—" if pd.isna(x) else f"{int(x)}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Parandus = production absoluutviga − WD absoluutviga. Positiivne = WD oli parem. "
        "Otsus ülal kasutab ainult täpse põlluplaaniga ja täieliku WD-kihiga päevi."
    )

    # Keep detail hidden: only for checking a suspicious day, not a second analysis.
    with st.expander("Kontrolliks põllu kaupa", expanded=False):
        st.dataframe(
            fields.style.format({
                "Päev": lambda x: x.strftime("%d.%m") if hasattr(x, "strftime") else str(x),
                "Production ABC": "{:.2f}",
                "WD kordaja": "{:.3f}",
                "WD ABC": "{:.2f}",
                "HIGH päevi": lambda x: "—" if pd.isna(x) else f"{float(x):.0f}",
                "HIGH jada": lambda x: "—" if pd.isna(x) else f"{float(x):.0f}",
                "Δ HIGH päevi": lambda x: "—" if pd.isna(x) else f"{float(x):+.0f}",
                "Δ HIGH jada": lambda x: "—" if pd.isna(x) else f"{float(x):+.0f}",
                "WD avg": lambda x: "—" if pd.isna(x) else f"{float(x):.1f}",
                "HIGH lävi": lambda x: "—" if pd.isna(x) else f"{float(x):.1f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

    st.divider()
    st.caption(
        "AUDIT LOCK: ainult db.get_harvest_history, db.get_weather_rows ja "
        "db.get_yield_forecasts. Ei ole save_/set_/delete_/upsert_ kutseid. "
        "Iga päeva ridge fit kasutab harvest_date < target day."
    )


if __name__ == "__main__":
    main()
