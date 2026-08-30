from __future__ import annotations

"""
KurgiMootor · edge_weather-47
=============================

CYCLE PRODUCTIVITY ENGINE · FIELD-LEVEL AUDIT · STRICT TIME-ORDERED REPLAY · READ ONLY

One architecture only
---------------------
The best practical short-term anchor is the same field's previous harvest and the
new harvest interval.  Instead of predicting boxes directly, this LAB predicts the
change in DAILY ABC productivity from one completed field cycle to the next.

For field f and harvest cycle t:
    rate_t = ABC_t / growth_days_t
    PERSIST_t = rate_(t-1) * growth_days_t

The learned target is:
    y_t = log(rate_t / rate_(t-1))

The model may explain y_t from only information that is structurally available at
that harvest:
    - change in cycle-average measured weather versus the previous same-field cycle
    - change in harvest interval
    - previous cycle productivity level
    - one slow season-age term

Weather is aligned to REAL field growth cycles, not arbitrary 4/5/7-day windows.
Each cycle uses calendar weather from the previous harvest date through T-1; target
harvest-day measured weather is excluded.

The architecture is now frozen from the preceding blocked-discovery LAB.  The main
score here is a STRICT EXPANDING TIME-ORDERED REPLAY.  For each harvest date T, the
coefficients are fitted only on cycle rows whose target_date is strictly before T.
No same-day or future harvest actual can enter that fit.  The current row may use its
previous same-field completed cycle and measured weather only through T-1, because
those are known before the target harvest is completed.

The only verdict here is whether the exact frozen strict replay survives at FIELD-CYCLE level, not merely after the three harvested fields are summed by day. Historical blocked CV is not used for this verdict.

Important interpretation
------------------------
- PERSIST is the practical benchmark: previous daily productivity × current interval.
- FULL MODEL is the exact fixed ridge architecture from LAB44.
- STATE-ONLY is retained only as a diagnostic comparator; FULL is the frozen candidate from LAB46.
- Same ridge lambda, minimum train rows, target and strict past-only replay are unchanged; no feature/window/lag search.
- Primary score is now each individual field-cycle row. Daily three-field sums are shown only as a control.
- The audit explicitly measures whether daily accuracy is hiding cancellation between fields.
- READ ONLY: db.get_harvest_history + db.get_weather_rows only.
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import streamlit as st

import db


# ------------------------------ locked architecture ------------------------------
HOURS_PER_FIELD = 3.0
ABC_EPS = 0.20
RIDGE_LAMBDA = 8.0              # fixed; no lambda search
BLOCK_HARVEST_DAYS = 5          # fixed retrospective validation block
MIN_TRAIN_ROWS = 24
AUG_START = date(2026, 8, 17)
AUG_END = date(2026, 8, 24)
JULY_CUTOFF = date(2026, 8, 1)

FULL_FEATURES = [
    "d_growth",
    "d_rad",
    "d_tmean",
    "d_tmin",
    "d_winddry",
    "d_precip",
    "d_et0",
    "prev_log_rate",
    "season_day",
]

STATE_FEATURES = [
    "d_growth",
    "prev_log_rate",
    "season_day",
]

# Backward-compatible alias for unchanged helper code.
FEATURES = FULL_FEATURES

WEATHER_FEATURES = {
    "d_rad",
    "d_tmean",
    "d_tmin",
    "d_winddry",
    "d_precip",
    "d_et0",
}


@dataclass
class Event:
    day: date
    field: int
    order: int
    abc: float


def _d(v: Any) -> Optional[date]:
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


def _f(v: Any) -> Optional[float]:
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def _abc(row: Dict[str, Any]) -> Optional[float]:
    vals = [_f(row.get(k)) for k in ("a", "b", "c")]
    if any(v is None for v in vals):
        return None
    out = float(sum(vals))
    return out if out >= 0 else None


def _reliable(row: Dict[str, Any]) -> bool:
    q = str(row.get("data_quality") or row.get("quality") or "").strip().lower()
    return q not in {"hinnanguline", "ligikaudne", "estimated", "approximate"}


def _events(rows: Sequence[Dict[str, Any]]) -> List[Event]:
    out: List[Event] = []
    for row in rows:
        dd = _d(row.get("harvest_date"))
        if dd is None or not _reliable(row):
            continue
        try:
            field = int(row.get("field_no"))
        except Exception:
            continue
        if not 1 <= field <= 14:
            continue
        abc = _abc(row)
        if abc is None:
            continue
        try:
            order = int(row.get("harvest_order") or 1)
        except Exception:
            order = 1
        out.append(Event(dd, field, order, float(abc)))
    return sorted(out, key=lambda e: (e.day, e.order, e.field))


def _growth(prev: Event, cur: Event) -> float:
    g = float((cur.day - prev.day).days)
    g += (cur.order - prev.order) * (HOURS_PER_FIELD / 24.0)
    return max(0.5, g)


def _weather_map(rows: Sequence[Dict[str, Any]]) -> Dict[date, Dict[str, float]]:
    out: Dict[date, Dict[str, float]] = {}
    for row in rows:
        dd = _d(row.get("weather_date"))
        if dd is None:
            continue
        if str(row.get("data_kind") or "").strip().lower() != "measured":
            continue
        if not bool(row.get("checked")):
            continue
        tmin = _f(row.get("temp_min_c"))
        tmax = _f(row.get("temp_max_c"))
        wind = _f(row.get("wind_avg_ms"))
        rad = _f(row.get("radiation_mj_m2"))
        rh = _f(row.get("humidity_avg_pct"))
        precip = _f(row.get("precipitation_mm"))
        et0 = _f(row.get("et0_mm"))
        if None in (tmin, tmax, wind, rad, rh, precip, et0):
            continue
        out[dd] = {
            "rad": float(rad),
            "tmean": (float(tmin) + float(tmax)) / 2.0,
            "tmin": float(tmin),
            "winddry": float(wind) * (100.0 - float(rh)),
            "precip": float(precip),
            "et0": float(et0),
        }
    return out


def _cycle_weather(start_day: date, target_day: date, weather: Dict[date, Dict[str, float]]) -> Optional[Dict[str, float]]:
    """Previous harvest day through T-1. Target-day measured weather is never used."""
    n = (target_day - start_day).days
    if n <= 0:
        return None
    days = [start_day + timedelta(days=k) for k in range(n)]
    if any(dd not in weather for dd in days):
        return None
    arr = {k: np.asarray([weather[dd][k] for dd in days], dtype=float) for k in ("rad", "tmean", "tmin", "winddry", "precip", "et0")}
    # All are per-calendar-day means. Precip is deliberately mean/day, not sum,
    # because interval length is already explicit in growth and d_growth.
    return {k: float(np.mean(v)) for k, v in arr.items()}


def _build_cycle_rows(events: Sequence[Event], weather: Dict[date, Dict[str, float]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    season_origin = min((e.day for e in events), default=date(2026, 7, 1))

    for field in range(1, 15):
        hist = sorted([e for e in events if e.field == field], key=lambda e: (e.day, e.order))
        intervals: List[Dict[str, Any]] = []
        for i in range(1, len(hist)):
            p, c = hist[i - 1], hist[i]
            if c.day <= p.day:
                continue
            g = _growth(p, c)
            wx = _cycle_weather(p.day, c.day, weather)
            if wx is None:
                continue
            rate = max(ABC_EPS, float(c.abc)) / g
            intervals.append({
                "field": field,
                "start_date": p.day,
                "target_date": c.day,
                "order": c.order,
                "actual": float(c.abc),
                "growth": float(g),
                "rate": float(rate),
                **wx,
            })

        for i in range(1, len(intervals)):
            prev = intervals[i - 1]
            cur = intervals[i]
            # Consecutive cycle continuity is required: previous cycle must end at
            # the current cycle start harvest.
            if prev["target_date"] != cur["start_date"]:
                continue
            prev_rate = max(ABC_EPS, float(prev["rate"]))
            cur_rate = max(ABC_EPS, float(cur["rate"]))
            row: Dict[str, Any] = {
                "field": int(field),
                "target_date": cur["target_date"],
                "prev_target_date": prev["target_date"],
                "start_date": cur["start_date"],
                "order": int(cur["order"]),
                "actual": float(cur["actual"]),
                "growth": float(cur["growth"]),
                "prev_growth": float(prev["growth"]),
                "prev_rate": float(prev_rate),
                "persist": float(prev_rate * float(cur["growth"])),
                "y": float(math.log(cur_rate / prev_rate)),
                "d_growth": float(cur["growth"] - prev["growth"]),
                "prev_log_rate": float(math.log(prev_rate)),
                "season_day": float((cur["target_date"] - season_origin).days),
            }
            for k in ("rad", "tmean", "tmin", "winddry", "precip", "et0"):
                row[f"d_{k}"] = float(cur[k] - prev[k])
            rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["target_date", "order", "field"]).reset_index(drop=True)
    return df


def _fit_ridge(train: pd.DataFrame, features: Sequence[str] = FULL_FEATURES) -> Optional[Dict[str, Any]]:
    if len(train) < MIN_TRAIN_ROWS:
        return None
    features = list(features)
    X = train[features].to_numpy(dtype=float)
    y = train["y"].to_numpy(dtype=float)
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        return None

    mu = np.mean(X, axis=0)
    sd = np.std(X, axis=0, ddof=0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    Xs = (X - mu) / sd

    # Intercept unpenalized; standardized features have the same fixed ridge penalty
    # in FULL and STATE-ONLY. No re-tuning for the ablation.
    A = np.column_stack([np.ones(len(Xs)), Xs])
    penalty = np.eye(A.shape[1], dtype=float) * RIDGE_LAMBDA
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(A.T @ A + penalty, A.T @ y)
    return {"mu": mu, "sd": sd, "beta": beta, "n": int(len(train)), "features": features}


def _predict_rows(df: pd.DataFrame, fit: Optional[Dict[str, Any]]) -> np.ndarray:
    if fit is None or df.empty:
        return np.full(len(df), np.nan, dtype=float)
    features = list(fit.get("features") or FULL_FEATURES)
    X = df[features].to_numpy(dtype=float)
    Xs = (X - fit["mu"]) / fit["sd"]
    A = np.column_stack([np.ones(len(Xs)), Xs])
    yhat = A @ fit["beta"]
    # No tuned cap. Only a broad numerical safety rail against exp overflow.
    yhat = np.clip(yhat, -1.5, 1.5)
    return df["persist"].to_numpy(dtype=float) * np.exp(yhat)


def _blocked_cv(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df.copy(), pd.DataFrame()
    days = sorted(df["target_date"].dropna().unique())
    out_parts: List[pd.DataFrame] = []
    block_rows: List[Dict[str, Any]] = []

    for b0 in range(0, len(days), BLOCK_HARVEST_DAYS):
        hold = set(days[b0:b0 + BLOCK_HARVEST_DAYS])
        if not hold:
            continue
        # Exclude targets in the block AND rows whose previous same-field cycle ends
        # in the block. This prevents a held-out actual from entering fitted state.
        train = df[(~df["target_date"].isin(hold)) & (~df["prev_target_date"].isin(hold))].copy()
        test = df[df["target_date"].isin(hold)].copy()
        fit = _fit_ridge(train)
        if fit is None or test.empty:
            continue
        test["model"] = _predict_rows(test, fit)
        test["train_n"] = int(fit["n"])
        test["block_id"] = int(b0 // BLOCK_HARVEST_DAYS + 1)
        out_parts.append(test)

        dm = _day_metrics(_daily(test), "model")
        dp = _day_metrics(_daily(test), "persist")
        block_rows.append({
            "Block": int(b0 // BLOCK_HARVEST_DAYS + 1),
            "Algus": min(hold),
            "Lõpp": max(hold),
            "N päeva": len(hold),
            "PERSIST MAE": dp["mae"],
            "MODEL MAE": dm["mae"],
            "Parandus %": _improve(dp["mae"], dm["mae"]),
            "Train N": int(fit["n"]),
        })

    pred = pd.concat(out_parts, ignore_index=True) if out_parts else pd.DataFrame()
    if not pred.empty:
        pred = pred.sort_values(["target_date", "order", "field"]).reset_index(drop=True)
    return pred, pd.DataFrame(block_rows)



def _time_ordered_replay(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Strict expanding replay for STATE-ONLY and FULL on identical past-only rows."""
    if df.empty:
        return df.copy(), pd.DataFrame()

    days = sorted(df["target_date"].dropna().unique())
    out_parts: List[pd.DataFrame] = []
    audit_rows: List[Dict[str, Any]] = []

    for dd in days:
        train = df[df["target_date"] < dd].copy()
        test = df[df["target_date"] == dd].copy()
        fit_state = _fit_ridge(train, STATE_FEATURES)
        fit_full = _fit_ridge(train, FULL_FEATURES)
        ready = fit_state is not None and fit_full is not None and not test.empty

        audit_rows.append({
            "Päev": dd,
            "Train N": int(len(train)),
            "Test N": int(len(test)),
            "Replay valmis": bool(ready),
            "STATE fit valmis": bool(fit_state is not None),
            "FULL fit valmis": bool(fit_full is not None),
            "Max train target": max(train["target_date"]) if not train.empty else None,
            "Future rows fitis": int(np.sum(train["target_date"] >= dd)) if not train.empty else 0,
        })

        if not ready:
            continue

        test["state_model"] = _predict_rows(test, fit_state)
        test["full_model"] = _predict_rows(test, fit_full)
        test["train_n"] = int(fit_full["n"])
        test["replay_day"] = dd
        test["max_train_target"] = max(train["target_date"]) if not train.empty else None
        out_parts.append(test)

    pred = pd.concat(out_parts, ignore_index=True) if out_parts else pd.DataFrame()
    if not pred.empty:
        pred = pred.sort_values(["target_date", "order", "field"]).reset_index(drop=True)
    return pred, pd.DataFrame(audit_rows)


def _replay_verdict(replay_daily: pd.DataFrame, aug: pd.DataFrame, pre_aug: pd.DataFrame) -> Tuple[str, str, str]:
    if replay_daily.empty:
        return "error", "TIME REPLAY NOT READY", "No harvest date had enough strictly earlier cycle rows to fit the frozen engine."

    p_all = _day_metrics(replay_daily, "persist")
    m_all = _day_metrics(replay_daily, "model")
    all_imp = _improve(p_all["mae"], m_all["mae"])
    _, dir_n, dir_pct = _direction(replay_daily, "model")

    pa = _day_metrics(aug, "persist")
    ma = _day_metrics(aug, "model")
    aug_imp = _improve(pa["mae"], ma["mae"])
    _, aug_n, aug_dir = _direction(aug, "model")

    pp = _day_metrics(pre_aug, "persist")
    mp = _day_metrics(pre_aug, "model")
    pre_imp = _improve(pp["mae"], mp["mae"])

    details = (
        f"Strict replay N={m_all['n']}: {all_imp:+.1f}% vs PERSIST, direction {dir_pct:.0f}% ({dir_n} changes). "
        f"Pre-Aug replay: {pre_imp:+.1f}% on N={mp['n']} days. "
        f"Healthy Aug 17–24: {aug_imp:+.1f}%, direction {aug_dir:.0f}% ({aug_n} changes)."
        if np.isfinite(all_imp) and np.isfinite(dir_pct) and np.isfinite(pre_imp) and np.isfinite(aug_imp) and np.isfinite(aug_dir)
        else "Strict replay is available, but one named period has too few eligible days for a complete score."
    )

    overall_ok = np.isfinite(all_imp) and all_imp > 0
    aug_ok = (ma["n"] >= 4 and np.isfinite(aug_imp) and aug_imp > 0 and aug_n >= 3 and aug_dir >= 60.0)
    no_disaster = np.isfinite(m_all["worst"]) and np.isfinite(p_all["worst"]) and m_all["worst"] <= max(1.5 * p_all["worst"], p_all["worst"] + 5.0)

    if overall_ok and aug_ok and no_disaster:
        return "success", "FROZEN CYCLE ENGINE SURVIVES TIME REPLAY", details
    if overall_ok and no_disaster:
        return "warning", "TIME REPLAY PROMISING, WAVE CHECK WEAKER", details
    return "error", "FROZEN CYCLE ENGINE FAILS TIME REPLAY", details

def _daily(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    agg: Dict[str, str] = {"actual": "sum", "persist": "sum"}
    for c in ("model", "state_model", "full_model"):
        if c in df.columns:
            agg[c] = "sum"
    out = df.groupby("target_date", as_index=False).agg(agg).sort_values("target_date").reset_index(drop=True)
    return out


def _metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    mask = np.isfinite(y) & np.isfinite(p)
    if not np.any(mask):
        return {"n": 0, "mae": np.nan, "mape": np.nan, "bias": np.nan, "within20": np.nan, "worst": np.nan}
    y = y[mask]
    p = p[mask]
    ae = np.abs(p - y)
    ape = ae / np.maximum(np.abs(y), ABC_EPS)
    return {
        "n": int(len(y)),
        "mae": float(np.mean(ae)),
        "mape": float(np.mean(ape) * 100.0),
        "bias": float(np.mean(p - y)),
        "within20": float(np.mean(ape <= 0.20) * 100.0),
        "worst": float(np.max(ae)),
    }


def _day_metrics(daily: pd.DataFrame, col: str) -> Dict[str, float]:
    if daily.empty or col not in daily.columns:
        return _metrics(np.asarray([]), np.asarray([]))
    return _metrics(daily["actual"].to_numpy(float), daily[col].to_numpy(float))


def _direction(daily: pd.DataFrame, col: str) -> Tuple[int, int, float]:
    if daily.empty or len(daily) < 2 or col not in daily.columns:
        return 0, 0, np.nan
    a = daily["actual"].to_numpy(float)
    p = daily[col].to_numpy(float)
    da = np.diff(a)
    dp = np.diff(p)
    mask = (np.abs(da) > 1e-9) & np.isfinite(dp)
    if not np.any(mask):
        return 0, 0, np.nan
    hit = int(np.sum(np.sign(da[mask]) == np.sign(dp[mask])))
    n = int(np.sum(mask))
    return hit, n, 100.0 * hit / n


def _improve(base: float, new: float) -> float:
    if not np.isfinite(base) or base <= 1e-12 or not np.isfinite(new):
        return np.nan
    return float(100.0 * (base - new) / base)


def _period(daily: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    if daily.empty:
        return daily.copy()
    return daily[(daily["target_date"] >= start) & (daily["target_date"] <= end)].copy()


def _strongest_pre_aug_drop(daily: pd.DataFrame) -> Tuple[Optional[date], pd.DataFrame]:
    pre = daily[daily["target_date"] < JULY_CUTOFF].copy().reset_index(drop=True)
    if len(pre) < 3:
        return None, pd.DataFrame()
    diff = pre["actual"].diff()
    idx = diff.idxmin()
    if pd.isna(idx):
        return None, pd.DataFrame()
    idx = int(idx)
    lo = max(0, idx - 2)
    hi = min(len(pre), idx + 3)
    return pre.loc[idx, "target_date"], pre.iloc[lo:hi].copy()


def _drop_sign_hit(window: pd.DataFrame, drop_day: Optional[date], col: str) -> Optional[bool]:
    if drop_day is None or window.empty or col not in window.columns:
        return None
    pos = window.index[window["target_date"] == drop_day].tolist()
    if not pos:
        return None
    i = window.index.get_loc(pos[0])
    if not isinstance(i, int) or i <= 0:
        return None
    a = float(window.iloc[i]["actual"] - window.iloc[i - 1]["actual"])
    p = float(window.iloc[i][col] - window.iloc[i - 1][col])
    return bool(a < 0 and p < 0)


def _summary_row(label: str, daily: pd.DataFrame, col: str) -> Dict[str, Any]:
    m = _day_metrics(daily, col)
    h, n, pct = _direction(daily, col)
    return {
        "Variant": label,
        "N päeva": m["n"],
        "MAE": m["mae"],
        "MAPE %": m["mape"],
        "Bias": m["bias"],
        "±20%": m["within20"],
        "Worst AE": m["worst"],
        "Suund": f"{h}/{n}" if n else "—",
        "Suund %": pct,
    }


def _fmt_table(df: pd.DataFrame, nd: int = 2) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].round(nd)
    return out


def _coef_table(fit: Optional[Dict[str, Any]]) -> pd.DataFrame:
    if fit is None:
        return pd.DataFrame()
    beta = fit["beta"][1:]
    rows = []
    labels = {
        "d_growth": "Δ interval",
        "d_rad": "Δ radiation / cycle",
        "d_tmean": "Δ mean temp / cycle",
        "d_tmin": "Δ Tmin / cycle",
        "d_winddry": "Δ WIND×DRY / cycle",
        "d_precip": "Δ rain/day / cycle",
        "d_et0": "Δ ET0/day / cycle",
        "prev_log_rate": "previous daily productivity",
        "season_day": "season age",
    }
    for name, b in zip(list(fit.get("features") or FULL_FEATURES), beta):
        rows.append({
            "Feature": labels.get(name, name),
            "Type": "weather" if name in WEATHER_FEATURES else "state",
            "Std beta": float(b),
            "|beta|": abs(float(b)),
        })
    return pd.DataFrame(rows).sort_values("|beta|", ascending=False).drop(columns="|beta|").reset_index(drop=True)


def _weather_value_verdict(daily: pd.DataFrame, aug: pd.DataFrame) -> Tuple[str, str, str]:
    if daily.empty:
        return "error", "ABLATION NOT READY", "Strict replay did not reach the minimum training size."

    p = _day_metrics(daily, "persist")
    s = _day_metrics(daily, "state_model")
    f = _day_metrics(daily, "full_model")
    pa = _day_metrics(aug, "persist")
    sa = _day_metrics(aug, "state_model")
    fa = _day_metrics(aug, "full_model")
    _, fn, fdir = _direction(daily, "full_model")
    _, fan, fadir = _direction(aug, "full_model")

    full_vs_state = _improve(s["mae"], f["mae"])
    state_vs_persist = _improve(p["mae"], s["mae"])
    full_vs_persist = _improve(p["mae"], f["mae"])
    aug_full_vs_state = _improve(sa["mae"], fa["mae"])

    details = (
        f"Strict replay N={f['n']}: STATE-only {state_vs_persist:+.1f}% vs PERSIST; "
        f"FULL {full_vs_persist:+.1f}% vs PERSIST and {full_vs_state:+.1f}% vs STATE-only. "
        f"Healthy Aug 17–24: FULL {aug_full_vs_state:+.1f}% vs STATE-only, direction {fadir:.0f}% ({fan} changes)."
    )

    if np.isfinite(full_vs_state) and full_vs_state > 10.0 and np.isfinite(aug_full_vs_state) and aug_full_vs_state > 10.0:
        return "success", "WEATHER ADDS MATERIAL VALUE", details
    if np.isfinite(full_vs_state) and full_vs_state > 0:
        return "warning", "WEATHER ADDS SOME VALUE, BUT STATE CARRIES MOST", details
    return "error", "WEATHER DOES NOT ADD OUT-OF-SAMPLE VALUE", details


def _verdict(cv_daily: pd.DataFrame, july_day: Optional[date], july_win: pd.DataFrame, aug: pd.DataFrame, blocks: pd.DataFrame) -> Tuple[str, str, str]:
    p_all = _day_metrics(cv_daily, "persist")
    m_all = _day_metrics(cv_daily, "model")
    all_imp = _improve(p_all["mae"], m_all["mae"])

    p_j = _day_metrics(july_win, "persist")
    m_j = _day_metrics(july_win, "model")
    july_imp = _improve(p_j["mae"], m_j["mae"])
    july_drop = _drop_sign_hit(july_win.reset_index(drop=True), july_day, "model")

    p_a = _day_metrics(aug, "persist")
    m_a = _day_metrics(aug, "model")
    aug_imp = _improve(p_a["mae"], m_a["mae"])
    _, aug_n, aug_dir = _direction(aug, "model")

    overall_ok = np.isfinite(all_imp) and all_imp > 0
    july_ok = np.isfinite(july_imp) and july_imp > 0 and july_drop is True
    aug_ok = np.isfinite(aug_imp) and aug_imp > 0 and aug_n >= 3 and aug_dir >= 80.0

    catastrophic = False
    if not blocks.empty:
        for _, r in blocks.iterrows():
            if float(r["PERSIST MAE"]) > 1e-9 and float(r["MODEL MAE"]) > 1.75 * float(r["PERSIST MAE"]):
                catastrophic = True
                break

    details = (
        f"Whole season blocked: {all_imp:+.1f}% vs PERSIST. "
        f"Strongest pre-Aug drop: {july_imp:+.1f}% and drop sign={'YES' if july_drop else 'NO'}. "
        f"Healthy Aug 17–24: {aug_imp:+.1f}%, direction {aug_dir:.0f}% ({aug_n} changes)."
        if np.isfinite(all_imp) and np.isfinite(july_imp) and np.isfinite(aug_imp) and np.isfinite(aug_dir)
        else "Not enough complete blocked data for all pre-declared checks."
    )

    if overall_ok and july_ok and aug_ok and not catastrophic:
        return "success", "CYCLE ENGINE SURVIVES", details + " One unchanged architecture answers both named waves and the season-wide blocked benchmark."
    if overall_ok and (july_ok or aug_ok) and not catastrophic:
        return "warning", "PROMISING, NOT YET COMPLETE", details + " Keep the architecture fixed; inspect the failing regime rather than tune windows."
    return "error", "CYCLE ENGINE FAILS", details + " Do not rescue it by changing weather windows or ridge strength in this LAB."




def _row_metrics(df: pd.DataFrame, col: str) -> Dict[str, float]:
    """Metrics on individual field-cycle rows (the primary unit in LAB47)."""
    if df.empty or col not in df.columns:
        return _metrics(np.asarray([]), np.asarray([]))
    return _metrics(df["actual"].to_numpy(float), df[col].to_numpy(float))


def _cycle_direction_rows(df: pd.DataFrame, col: str) -> Tuple[int, int, float]:
    """Did the model move the current cycle above/below PERSIST in the same direction as reality?"""
    if df.empty or col not in df.columns:
        return 0, 0, np.nan
    actual_delta = df["actual"].to_numpy(float) - df["persist"].to_numpy(float)
    pred_delta = df[col].to_numpy(float) - df["persist"].to_numpy(float)
    mask = np.isfinite(actual_delta) & np.isfinite(pred_delta) & (np.abs(actual_delta) > 1e-9)
    if not np.any(mask):
        return 0, 0, np.nan
    hit = int(np.sum(np.sign(actual_delta[mask]) == np.sign(pred_delta[mask])))
    n = int(np.sum(mask))
    return hit, n, 100.0 * hit / n


def _row_summary(label: str, df: pd.DataFrame, col: str) -> Dict[str, Any]:
    m = _row_metrics(df, col)
    h, n, pct = _cycle_direction_rows(df, col)
    return {
        "Variant": label,
        "N field-cycle": m["n"],
        "MAE / field": m["mae"],
        "MAPE %": m["mape"],
        "Bias / field": m["bias"],
        "±20%": m["within20"],
        "Worst field AE": m["worst"],
        "Cycle suund": f"{h}/{n}" if n else "—",
        "Cycle suund %": pct,
    }


def _field_breakdown(replay: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if replay.empty:
        return pd.DataFrame()
    for field in sorted(int(x) for x in replay["field"].dropna().unique()):
        f = replay[replay["field"] == field].copy()
        pm = _row_metrics(f, "persist")
        sm = _row_metrics(f, "state_model")
        fm = _row_metrics(f, "full_model")
        h, n, pct = _cycle_direction_rows(f, "full_model")
        rows.append({
            "Põld": field,
            "N": fm["n"],
            "Tegelik avg": float(f["actual"].mean()) if len(f) else np.nan,
            "PERSIST MAE": pm["mae"],
            "STATE MAE": sm["mae"],
            "FULL MAE": fm["mae"],
            "FULL vs PERSIST %": _improve(pm["mae"], fm["mae"]),
            "FULL vs STATE %": _improve(sm["mae"], fm["mae"]),
            "FULL bias": fm["bias"],
            "FULL MAPE %": fm["mape"],
            "FULL worst AE": fm["worst"],
            "FULL cycle suund": f"{h}/{n}" if n else "—",
            "FULL cycle suund %": pct,
            "FULL võidab PERSIST": bool(np.isfinite(pm["mae"]) and np.isfinite(fm["mae"]) and fm["mae"] < pm["mae"]),
            "FULL võidab STATE": bool(np.isfinite(sm["mae"]) and np.isfinite(fm["mae"]) and fm["mae"] < sm["mae"]),
        })
    return pd.DataFrame(rows)


def _cancellation_summary(replay: pd.DataFrame) -> pd.DataFrame:
    """
    Compare individual field errors with errors after same-day field sums.
    Cancellation % = 1 - sum(|daily summed error|) / sum(|individual field errors|).
    0% means daily totals do not benefit from opposite-sign field errors; higher values
    mean more cancellation. This is diagnostic only; row-level MAE remains primary.
    """
    rows: List[Dict[str, Any]] = []
    for label, col in [
        ("PERSIST", "persist"),
        ("STATE-ONLY", "state_model"),
        ("FULL + WEATHER", "full_model"),
    ]:
        if replay.empty or col not in replay.columns:
            continue
        tmp = replay[["target_date", "actual", col]].copy()
        tmp["err"] = tmp[col] - tmp["actual"]
        field_abs = float(np.abs(tmp["err"]).sum())
        by_day = tmp.groupby("target_date", as_index=False).agg(actual=("actual", "sum"), pred=(col, "sum"), field_abs=("err", lambda s: float(np.abs(s).sum())))
        by_day["day_abs"] = np.abs(by_day["pred"] - by_day["actual"])
        daily_abs = float(by_day["day_abs"].sum())
        cancel = 100.0 * (1.0 - daily_abs / field_abs) if field_abs > 1e-12 else np.nan
        rows.append({
            "Variant": label,
            "N field-cycle": int(len(tmp)),
            "Field MAE": float(np.mean(np.abs(tmp["err"]))) if len(tmp) else np.nan,
            "Daily MAE": float(np.mean(by_day["day_abs"])) if len(by_day) else np.nan,
            "Sum field AE": field_abs,
            "Sum daily AE": daily_abs,
            "Cancellation %": cancel,
        })
    return pd.DataFrame(rows)


def _daily_cancellation_detail(replay: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if replay.empty:
        return pd.DataFrame()
    for dd, g in replay.groupby("target_date", sort=True):
        row: Dict[str, Any] = {"Päev": dd, "N põldu": int(len(g)), "Tegelik ABC": float(g["actual"].sum())}
        for short, col in [("PERSIST", "persist"), ("STATE", "state_model"), ("FULL", "full_model")]:
            err = g[col].to_numpy(float) - g["actual"].to_numpy(float)
            sae = float(np.abs(err).sum())
            day_ae = abs(float(np.sum(err)))
            row[f"{short} field AE sum"] = sae
            row[f"{short} day AE"] = day_ae
            row[f"{short} cancel %"] = 100.0 * (1.0 - day_ae / sae) if sae > 1e-12 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _field_level_verdict(replay: pd.DataFrame, fields: pd.DataFrame) -> Tuple[str, str, str]:
    if replay.empty or fields.empty:
        return "error", "FIELD AUDIT NOT READY", "Strict replay has no field-cycle rows to audit."

    p = _row_metrics(replay, "persist")
    s = _row_metrics(replay, "state_model")
    f = _row_metrics(replay, "full_model")
    pdaily = _day_metrics(_daily(replay), "persist")
    fdaily = _day_metrics(_daily(replay), "full_model")

    eligible = fields[fields["N"] >= 2].copy()
    win_p = int(eligible["FULL võidab PERSIST"].sum()) if not eligible.empty else 0
    win_s = int(eligible["FULL võidab STATE"].sum()) if not eligible.empty else 0
    n_fields = int(len(eligible))
    need = n_fields // 2 + 1 if n_fields else 999

    imp_p = _improve(p["mae"], f["mae"])
    imp_s = _improve(s["mae"], f["mae"])
    day_imp = _improve(pdaily["mae"], fdaily["mae"])

    details = (
        f"Strict field-cycle N={f['n']}: FULL {imp_p:+.1f}% vs PERSIST and {imp_s:+.1f}% vs STATE-only. "
        f"Eligible fields N={n_fields}: FULL beats PERSIST on {win_p}/{n_fields} and STATE-only on {win_s}/{n_fields}. "
        f"Daily-sum control remains {day_imp:+.1f}% vs PERSIST."
    )

    if np.isfinite(imp_p) and np.isfinite(imp_s) and imp_p > 0 and imp_s > 0 and win_p >= need and win_s >= need and np.isfinite(day_imp) and day_imp > 0:
        return "success", "FIELD LEVEL SURVIVES", details + " The daily gain is supported by individual-field improvement, not only by three-field cancellation."
    if np.isfinite(imp_p) and imp_p > 0 and np.isfinite(day_imp) and day_imp > 0:
        return "warning", "FIELD LEVEL MIXED", details + " Aggregate accuracy improves, but the majority-field check is not clean enough to call the architecture field-robust."
    return "error", "FIELD LEVEL FAILS", details + " Do not promote the engine from daily totals until individual-field behavior is repaired."


def main() -> None:
    st.set_page_config(page_title="KurgiMootor · field audit", layout="wide")
    st.title("KurgiMootor · cycle engine · field-level audit · strict replay")
    st.caption("edge_weather-47 · PERSIST vs STATE-ONLY vs FULL · individual field-cycle rows · exact past-only replay · READ ONLY")

    st.info(
        "See LAB ei muuda CYCLE ENGINE'i. LAB46 näitas, et FULL + WEATHER parandab strict daily replay's STATE-only mudelit. "
        "Nüüd kontrollime, kas sama võit eksisteerib päriselt iga põllu tsükliridadel või tekib ainult kolme põllu päevase summa "
        "vastastikustest vigadest. Sama target, samad tunnused, ridge λ=8, minimum train N=24 ja target_date < T replay. "
        "Põllu taseme MAE on siin PRIMARY; päevane summa on ainult kontroll."
    )

    harvest_rows = db.get_harvest_history(limit=5000)
    events = _events(harvest_rows)
    if len(events) < 30:
        st.error("Liiga vähe usaldusväärseid korjeridu.")
        return

    wx_start = min(e.day for e in events) - timedelta(days=10)
    wx_end = max(e.day for e in events)
    weather = _weather_map(db.get_weather_rows(wx_start, wx_end))
    if not weather:
        st.error("Kontrollitud measured ilmaandmeid ei leitud.")
        return

    cycles = _build_cycle_rows(events, weather)
    if len(cycles) < MIN_TRAIN_ROWS + 4:
        st.error(f"Täielikke kahe järjestikuse tsükli ridu on ainult {len(cycles)}; vaja vähemalt {MIN_TRAIN_ROWS + 4}.")
        return

    replay, replay_audit = _time_ordered_replay(cycles)
    if replay.empty or "state_model" not in replay.columns or "full_model" not in replay.columns:
        st.error("Strict field-level replay ei jõudnud minimaalse treeningmahuni.")
        st.dataframe(replay_audit, use_container_width=True, hide_index=True)
        return

    replay_daily = _daily(replay)
    fields = _field_breakdown(replay)
    level, title, verdict = _field_level_verdict(replay, fields)

    st.header("1. Otsus · kas päevane võit püsib põllu tasemel?")
    if level == "success":
        st.success(f"✅ {title}\n\n{verdict}")
    elif level == "warning":
        st.warning(f"🟡 {title}\n\n{verdict}")
    else:
        st.error(f"⛔ {title}\n\n{verdict}")

    st.header("2. Kõik strict field-cycle read · PRIMARY SCORE")
    pm = _row_metrics(replay, "persist")
    sm = _row_metrics(replay, "state_model")
    fm = _row_metrics(replay, "full_model")
    overall = pd.DataFrame([
        {**_row_summary("A · PERSIST", replay, "persist"), "Parandus vs PERSIST %": 0.0, "FULL vs STATE %": np.nan},
        {**_row_summary("B · STATE-ONLY CYCLE", replay, "state_model"), "Parandus vs PERSIST %": _improve(pm["mae"], sm["mae"]), "FULL vs STATE %": np.nan},
        {**_row_summary("C · FULL CYCLE + WEATHER", replay, "full_model"), "Parandus vs PERSIST %": _improve(pm["mae"], fm["mae"]), "FULL vs STATE %": _improve(sm["mae"], fm["mae"])},
    ])
    st.dataframe(_fmt_table(overall), use_container_width=True, hide_index=True)
    st.caption(
        "Siin on üks vaatlus = üks konkreetne põld ühel konkreetsel korjel. Cycle suund tähendab: kas mudel sai õigesti aru, "
        "kas tegelik tsükkel tuli PERSIST-benchmarkist üles- või allapoole."
    )

    st.header("3. Põllud 1–14 eraldi")
    eligible = fields[fields["N"] >= 2].copy()
    if eligible.empty:
        st.warning("Ühelgi põllul pole vähemalt kahte strict replay rida.")
    else:
        win_p = int(eligible["FULL võidab PERSIST"].sum())
        win_s = int(eligible["FULL võidab STATE"].sum())
        st.write(
            f"Eligible põlde: **{len(eligible)}** · FULL võidab PERSIST-i **{win_p}/{len(eligible)}** põllul · "
            f"FULL võidab STATE-only mudelit **{win_s}/{len(eligible)}** põllul."
        )
        st.dataframe(_fmt_table(fields), use_container_width=True, hide_index=True)

        worst_full = fields.sort_values("FULL MAE", ascending=False).head(5)[[
            "Põld", "N", "PERSIST MAE", "STATE MAE", "FULL MAE", "FULL bias", "FULL worst AE", "FULL vs PERSIST %", "FULL vs STATE %"
        ]]
        st.subheader("Kõige nõrgemad FULL põllud · ainult audit")
        st.dataframe(_fmt_table(worst_full), use_container_width=True, hide_index=True)

    st.header("4. Kas päevane summa peidab põldude vastastikust viga?")
    cancellation = _cancellation_summary(replay)
    st.dataframe(_fmt_table(cancellation), use_container_width=True, hide_index=True)
    st.caption(
        "Cancellation % mõõdab, kui suur osa individuaalsete põlluvigade absoluutmahust kaob siis, kui sama päeva põllud kokku liita. "
        "See ei ole mudeli skoor. Oluline kontroll on, et FULL parandaks PRIMARY field-cycle MAE-d ka enne summamist."
    )
    with st.expander("Päev-päevalt cancellation detail", expanded=False):
        st.dataframe(_fmt_table(_daily_cancellation_detail(replay)), use_container_width=True, hide_index=True)

    st.header("5. Päevataseme summa · ainult kontroll LAB46 vastu")
    pday = _day_metrics(replay_daily, "persist")
    sday = _day_metrics(replay_daily, "state_model")
    fday = _day_metrics(replay_daily, "full_model")
    day_table = pd.DataFrame([
        {**_summary_row("PERSIST", replay_daily, "persist"), "Parandus vs PERSIST %": 0.0, "FULL vs STATE %": np.nan},
        {**_summary_row("STATE-ONLY", replay_daily, "state_model"), "Parandus vs PERSIST %": _improve(pday["mae"], sday["mae"]), "FULL vs STATE %": np.nan},
        {**_summary_row("FULL + WEATHER", replay_daily, "full_model"), "Parandus vs PERSIST %": _improve(pday["mae"], fday["mae"]), "FULL vs STATE %": _improve(sday["mae"], fday["mae"])},
    ])
    st.dataframe(_fmt_table(day_table), use_container_width=True, hide_index=True)

    st.header("6. Põllu detailread · kust viga tuleb?")
    detail = replay[[
        "target_date", "field", "actual", "persist", "state_model", "full_model", "growth", "prev_rate", "train_n"
    ]].copy()
    detail["PERSIST error"] = detail["persist"] - detail["actual"]
    detail["STATE error"] = detail["state_model"] - detail["actual"]
    detail["FULL error"] = detail["full_model"] - detail["actual"]
    detail["Weather võit vs STATE"] = np.abs(detail["STATE error"]) - np.abs(detail["FULL error"])
    detail = detail.rename(columns={
        "target_date": "Päev", "field": "Põld", "actual": "Tegelik ABC", "persist": "PERSIST",
        "state_model": "STATE-ONLY", "full_model": "FULL + WEATHER", "growth": "Kasvupäevi",
        "prev_rate": "Eelmine ABC/päev", "train_n": "Train N",
    })
    st.dataframe(_fmt_table(detail), use_container_width=True, hide_index=True)

    with st.expander("7. Warm-up / leakage audit", expanded=False):
        st.dataframe(replay_audit, use_container_width=True, hide_index=True)
        bad = replay_audit[(replay_audit["Replay valmis"] == True) & ((replay_audit["Future rows fitis"] != 0) | (replay_audit["Max train target"] >= replay_audit["Päev"]))]
        if bad.empty:
            st.success("Leakage lock OK: igal skooritud väljal/päeval on kogu train set rangelt minevikus; future rows fitis = 0.")
        else:
            st.error("LEAKAGE LOCK FAIL: vähemalt ühel skooritud päeval ei ole treening rangelt minevikus.")

    st.header("8. Lukud")
    st.code(
        "PRIMARY UNIT = one field-cycle row; daily sum is control only\n"
        "TARGET = log(current ABC/day ÷ previous ABC/day)\n"
        "PERSIST = previous same-field ABC/day × current growth interval\n"
        f"STATE FEATURES = {', '.join(STATE_FEATURES)}\n"
        f"FULL FEATURES = {', '.join(FULL_FEATURES)}\n"
        "WEATHER = current real field cycle mean - previous same-field cycle mean\n"
        "cycle weather days = previous harvest date .. target-1 (target measured weather excluded)\n"
        f"ridge lambda = {RIDGE_LAMBDA}; minimum train rows = {MIN_TRAIN_ROWS}\n"
        "STRICT REPLAY FIT = target_date < predicted target_date only\n"
        "same frozen architecture as LAB46; no feature/lambda/window/lag/cap search\n"
        "DB actions = get_harvest_history + get_weather_rows only"
    )


if __name__ == "__main__":
    main()
