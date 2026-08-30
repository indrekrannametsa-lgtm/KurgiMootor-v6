from __future__ import annotations

"""
KurgiMootor · edge_weather-44
=============================

CYCLE PRODUCTIVITY ENGINE · RETROSPECTIVE BLOCKED VALIDATION · READ ONLY

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

Validation is deliberately retrospective because the season is almost finished and
we want to learn one architecture from the season as a whole.  The main score is a
5-harvest-day BLOCKED leave-out test.  For each held-out block the coefficients are
trained on the rest of the season.  Training also excludes any row whose previous
same-field cycle ends inside the held-out block, so a held-out actual cannot leak
back into the fitted coefficients through the next rotation.

Important interpretation
------------------------
- PERSIST is the practical benchmark: previous daily productivity × current interval.
- MODEL is one fixed ridge architecture; no feature/window/lag search in this app.
- Full-fit reconstruction is shown only to see whether the architecture can represent
  the season after training on all available data.  It is NOT validation evidence.
- The strongest pre-August down-step is located only for a targeted diagnostic; the
  overall blocked score is computed on the whole usable season and does not depend on
  choosing that event.
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

FEATURES = [
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


def _fit_ridge(train: pd.DataFrame) -> Optional[Dict[str, Any]]:
    if len(train) < MIN_TRAIN_ROWS:
        return None
    X = train[FEATURES].to_numpy(dtype=float)
    y = train["y"].to_numpy(dtype=float)
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        return None

    mu = np.mean(X, axis=0)
    sd = np.std(X, axis=0, ddof=0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    Xs = (X - mu) / sd

    # Intercept unpenalized; standardized features have fixed ridge penalty.
    A = np.column_stack([np.ones(len(Xs)), Xs])
    penalty = np.eye(A.shape[1], dtype=float) * RIDGE_LAMBDA
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(A.T @ A + penalty, A.T @ y)
    return {"mu": mu, "sd": sd, "beta": beta, "n": int(len(train))}


def _predict_rows(df: pd.DataFrame, fit: Optional[Dict[str, Any]]) -> np.ndarray:
    if fit is None or df.empty:
        return np.full(len(df), np.nan, dtype=float)
    X = df[FEATURES].to_numpy(dtype=float)
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


def _daily(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    agg: Dict[str, str] = {"actual": "sum", "persist": "sum"}
    if "model" in df.columns:
        agg["model"] = "sum"
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
    for name, b in zip(FEATURES, beta):
        rows.append({
            "Feature": labels.get(name, name),
            "Type": "weather" if name in WEATHER_FEATURES else "state",
            "Std beta": float(b),
            "|beta|": abs(float(b)),
        })
    return pd.DataFrame(rows).sort_values("|beta|", ascending=False).drop(columns="|beta|").reset_index(drop=True)


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


def main() -> None:
    st.set_page_config(page_title="KurgiMootor · cycle engine", layout="wide")
    st.title("KurgiMootor · cycle productivity engine")
    st.caption("edge_weather-44 · previous harvest × interval anchor + cycle-to-cycle weather change · retrospective blocked validation · READ ONLY")

    st.info(
        "Üks küsimus: kas sama põllu eelmise korje päevane tootlikkus × uus intervall annab taseme, "
        "ning kas päris kasvutsüklite ilmade muutus seletab, miks päevane tootlikkus järgmisel ringil üles või alla liigub? "
        "Siin ei otsita 4/5/7-päevaseid aknaid."
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
    if len(cycles) < MIN_TRAIN_ROWS + 8:
        st.error(f"Täielikke kahe järjestikuse tsükli ridu on ainult {len(cycles)}; vaja vähemalt {MIN_TRAIN_ROWS + 8}.")
        return

    cv, blocks = _blocked_cv(cycles)
    if cv.empty or "model" not in cv.columns:
        st.error("Blocked validation ei saanud piisavalt treeningridu.")
        return
    cv_daily = _daily(cv)

    # Full-fit diagnostic only.
    full_fit = _fit_ridge(cycles)
    full = cycles.copy()
    full["model"] = _predict_rows(full, full_fit)
    full_daily = _daily(full)

    july_day, july_win = _strongest_pre_aug_drop(cv_daily)
    aug = _period(cv_daily, AUG_START, AUG_END)
    late = cv_daily[cv_daily["target_date"] >= date(2026, 8, 25)].copy()

    level, title, text = _verdict(cv_daily, july_day, july_win, aug, blocks)
    st.header("1. Otsus · kas üks tsüklimootor vastab hooajale?")
    if level == "success":
        st.success(f"✅ {title}\n\n{text}")
    elif level == "warning":
        st.warning(f"🟡 {title}\n\n{text}")
    else:
        st.error(f"⛔ {title}\n\n{text}")

    st.header("2. Kogu hooaeg · 5 korjepäeva blocked leave-out")
    p_all = _day_metrics(cv_daily, "persist")
    m_all = _day_metrics(cv_daily, "model")
    summary = pd.DataFrame([
        _summary_row("A · PERSIST = eelmine päevatootlikkus × intervall", cv_daily, "persist"),
        _summary_row("B · CYCLE MODEL", cv_daily, "model"),
    ])
    summary["Parandus vs PERSIST %"] = [0.0, _improve(p_all["mae"], m_all["mae"])]
    st.dataframe(_fmt_table(summary), use_container_width=True, hide_index=True)
    st.caption(
        "See on põhiskoor. Iga 5 korjepäeva plokk on koefitsientide treeningust väljas. "
        "Lisaks eemaldatakse treeningust järgmise rotatsiooni read, mille previous-cycle actual tuli holdout-plokist."
    )

    chart = cv_daily.set_index("target_date")[["actual", "persist", "model"]].rename(columns={
        "actual": "Tegelik ABC",
        "persist": "PERSIST",
        "model": "CYCLE MODEL",
    })
    st.line_chart(chart)

    with st.expander("Blocked plokid · kas üks periood läheb käest?", expanded=False):
        st.dataframe(_fmt_table(blocks), use_container_width=True, hide_index=True)

    st.header("3. Tugevaim langus enne augustit")
    if july_day is None or july_win.empty:
        st.warning("Enne augustit pole piisavalt päevi tugeva languse diagnostikaks.")
    else:
        st.write(f"Automaatne diagnostiline sündmus: suurim tegeliku päevasaagi allasamm enne 01.08 oli **{july_day.strftime('%d.%m')}**. Seda kuupäeva ei kasutata kogu hooaja skoori valimiseks.")
        pj = _day_metrics(july_win, "persist")
        mj = _day_metrics(july_win, "model")
        drop_p = _drop_sign_hit(july_win.reset_index(drop=True), july_day, "persist")
        drop_m = _drop_sign_hit(july_win.reset_index(drop=True), july_day, "model")
        jt = pd.DataFrame([
            {**_summary_row("PERSIST", july_win, "persist"), "Languse märk õige": bool(drop_p)},
            {**_summary_row("CYCLE MODEL", july_win, "model"), "Languse märk õige": bool(drop_m), "Parandus %": _improve(pj["mae"], mj["mae"])},
        ])
        st.dataframe(_fmt_table(jt), use_container_width=True, hide_index=True)
        show = july_win[["target_date", "actual", "persist", "model"]].copy()
        show.columns = ["Päev", "Tegelik ABC", "PERSIST", "CYCLE MODEL"]
        st.dataframe(_fmt_table(show), use_container_width=True, hide_index=True)

    st.header("4. Puhas terve taime laine · 17.–24.08")
    if aug.empty:
        st.warning("17.–24.08 blocked-päevi ei ole piisavalt.")
    else:
        pa = _day_metrics(aug, "persist")
        ma = _day_metrics(aug, "model")
        at = pd.DataFrame([
            _summary_row("PERSIST", aug, "persist"),
            {**_summary_row("CYCLE MODEL", aug, "model"), "Parandus %": _improve(pa["mae"], ma["mae"])},
        ])
        st.dataframe(_fmt_table(at), use_container_width=True, hide_index=True)
        show = aug[["target_date", "actual", "persist", "model"]].copy()
        show.columns = ["Päev", "Tegelik ABC", "PERSIST", "CYCLE MODEL"]
        st.dataframe(_fmt_table(show), use_container_width=True, hide_index=True)

    st.header("5. Mida täishooaja fit õppis? · diagnostika, mitte skoor")
    st.caption("Allolevad standardiseeritud koefitsiendid on fititud kõigile olemasolevatele tsüklitele. Need näitavad arhitektuuri kasutatud suunda ja suhtelist tugevust; blocked skoor ülal ei kasuta holdout-päeva enda targetit koefitsientide õppimiseks.")
    coefs = _coef_table(full_fit)
    st.dataframe(_fmt_table(coefs, 3), use_container_width=True, hide_index=True)

    with st.expander("Täishooaja reconstruction · ainult kas arhitektuur suudab kuju esitada", expanded=False):
        recon = full_daily.set_index("target_date")[["actual", "persist", "model"]].rename(columns={
            "actual": "Tegelik ABC",
            "persist": "PERSIST",
            "model": "FULL-FIT MODEL",
        })
        st.line_chart(recon)
        st.dataframe(_fmt_table(full_daily), use_container_width=True, hide_index=True)

    with st.expander("25.08+ vananeva taime saba · info", expanded=False):
        if late.empty:
            st.write("Pole veel piisavalt blocked-ridu.")
        else:
            st.dataframe(_fmt_table(late), use_container_width=True, hide_index=True)

    st.header("6. Leakage / arhitektuuri lukud")
    st.code(
        "PERSIST = previous same-field ABC/day × current growth interval\n"
        "TARGET = log(current ABC/day ÷ previous ABC/day)\n"
        "WEATHER = current real field cycle mean - previous same-field cycle mean\n"
        "cycle weather days = previous harvest date .. target-1 (target measured weather excluded)\n"
        f"features = {', '.join(FEATURES)}\n"
        f"ridge lambda = {RIDGE_LAMBDA} fixed; block = {BLOCK_HARVEST_DAYS} harvest days fixed\n"
        "blocked train excludes target dates in holdout AND rows whose prev_target_date is in holdout\n"
        "no PI / FAST / TOMO / 4v4 / L3-7 / cap tuning / lag search / window search\n"
        "DB actions = get_harvest_history + get_weather_rows only"
    )


if __name__ == "__main__":
    main()
