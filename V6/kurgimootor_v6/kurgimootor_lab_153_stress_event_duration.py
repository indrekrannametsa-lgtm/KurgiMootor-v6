from __future__ import annotations

"""
KurgiMootor LAB-153 · STRESS EVENT DURATION
============================================

Eesmärk
-------
LAB-151/152 leidis, et 3–7 päeva enne korjet on kõige tugevam loetav ilmastikusignaal
WIND×DRY = tuul × (100 - RH), samas ET0 toetab sama atmosfäärse nõudluse lugu.
LAB-153 EI otsi uut ajanihet. Aken L3–7 on lukustatud.

Küsimus on nüüd kitsas:
- kas piisab ühest tugevast WIND×DRY / ET0 päevast;
- kas tähtsam on 2 järjestikuse päeva koormus;
- või 3 päeva kumulatiivne koormus / sündmuse kestus?

READ ONLY
---------
- loeb ainult db.get_harvest_history() ja db.get_weather_rows();
- EI kirjuta Supabase'i;
- taimeindeksit ei kasutata;
- käsiread on vaikimisi tühjad ja elavad ainult sessioonis.

Metoodika
---------
- siht = sama põllu ABC kasvukiiruse log-muutus võrreldes eelmise korjega;
- kasvuaeg arvestab korjejärjekorda (~3 h / põld), nagu app-128;
- L3–7 = viis täispäeva 3...7 päeva enne korjet, harvest-day ilma ei kasutata;
- walk-forward: sihtpäeva mudel treenitakse ainult varasematel korjepäevadel;
- BASE = eelmine sama põllu kasvukiirus + hooajapäev + kasvuaeg;
- sündmuse kuju tunnused on ette määratud, aknaid/lävesid tulemuse järgi ei optimeerita;
- HIGH lävi = 75. protsentiil ainult selleks hetkeks juba möödunud mõõdetud ilmadest
  (enne L3–7 akna algust); see väldib tulevikuleket ja ei ole käsitsi häälestatud.

Oluline
-------
Sama korjepäeva 3 põldu jagavad sama ilmaepisoodi. Seetõttu kuvatakse lisaks
põlluridadele ka EVENT-DAY kontroll, kus üks korjepäev loetakse üheks ilmastikusündmuseks.
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, List, Optional, Sequence, Tuple
import math

import numpy as np
import pandas as pd
import streamlit as st

import db


TZ = ZoneInfo("Europe/Tallinn")
TODAY = datetime.now(TZ).date()
SEASON_START = date(TODAY.year, 6, 15)
WEATHER_START = date(TODAY.year, 7, 1)
LAB_VERSION = "LAB-153-STRESS-EVENT-DURATION-V1"

HOURS_PER_FIELD = 3.0
RIDGE_ALPHA = 10.0
MIN_TRAIN_ROWS = 24
TARGET_EPS = 0.20
BIG_DROP = -0.30
DROP_WARNING = -0.20
HIGH_Q = 0.75
MIN_DAYS_FOR_HIGH_THRESHOLD = 10

REQUIRED_WEATHER = (
    "temp_night_avg_c", "temp_day_avg_c", "temp_min_c", "temp_max_c",
    "wind_avg_ms", "radiation_mj_m2", "humidity_avg_pct",
    "precipitation_mm", "et0_mm",
)


# -----------------------------------------------------------------------------
# Üldabid
# -----------------------------------------------------------------------------

def _d(value) -> Optional[date]:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if value is None:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except Exception:
        return None


def _f(value) -> Optional[float]:
    try:
        x = float(value)
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


def _pct_from_log(x: float) -> float:
    return 100.0 * (math.exp(float(x)) - 1.0)


def _safe_smape(actual: np.ndarray, pred: np.ndarray) -> float:
    den = np.abs(actual) + np.abs(pred)
    mask = np.isfinite(actual) & np.isfinite(pred) & (den > 1e-9)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(200.0 * np.abs(pred[mask] - actual[mask]) / den[mask]))


@dataclass
class HarvestEvent:
    day: date
    field: int
    order: int
    abc: float
    interval_days: Optional[float]
    source: str


def _event_key(e: HarvestEvent) -> Tuple[date, int, int]:
    return (e.day, e.order, e.field)


# -----------------------------------------------------------------------------
# Andmed
# -----------------------------------------------------------------------------

def _prepare_events(rows: List[dict]) -> List[HarvestEvent]:
    out: List[HarvestEvent] = []
    for r in rows:
        dd = _d(r.get("harvest_date"))
        try:
            field = int(r.get("field_no"))
        except Exception:
            continue
        if dd is None or not (1 <= field <= 14) or not _reliable(r):
            continue
        abc = _abc(r)
        if abc is None or abc < 0:
            continue
        try:
            order = int(r.get("harvest_order") or 1)
        except Exception:
            order = 1
        out.append(HarvestEvent(dd, field, order, float(abc), _f(r.get("interval_days")), "DB"))
    out.sort(key=_event_key)
    return out


def _parse_manual_rows(text: str) -> List[HarvestEvent]:
    """Formaat: YYYY-MM-DD;põld;ABC;järjekord[;intervall]. Semikoolon on kindlaim."""
    out: List[HarvestEvent] = []
    for raw in (text or "").splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        sep = ";" if ";" in raw else ","
        parts = [p.strip().replace(",", ".") for p in raw.split(sep)]
        if len(parts) < 4:
            continue
        try:
            dd = date.fromisoformat(parts[0])
            field = int(parts[1])
            abc = float(parts[2])
            order = int(parts[3])
            interval = float(parts[4]) if len(parts) >= 5 and parts[4] else None
        except Exception:
            continue
        if 1 <= field <= 14 and abc >= 0:
            out.append(HarvestEvent(dd, field, order, abc, interval, "LAB käsirida"))
    return out


def _merge_manual(events: List[HarvestEvent], manual: List[HarvestEvent]) -> List[HarvestEvent]:
    by_key = {(e.day, e.field): e for e in events}
    for e in manual:
        by_key.setdefault((e.day, e.field), e)  # DB võidab, kui sama päev+põld on olemas.
    out = list(by_key.values())
    out.sort(key=_event_key)
    return out


def _measured_weather(rows: List[dict]) -> Dict[date, dict]:
    out: Dict[date, dict] = {}
    for r in rows:
        dd = _d(r.get("weather_date"))
        if dd is None:
            continue
        if str(r.get("data_kind") or "").lower() != "measured" or not bool(r.get("checked")):
            continue
        if any(_f(r.get(c)) is None for c in REQUIRED_WEATHER):
            continue
        rh = float(r["humidity_avg_pct"])
        wind = float(r["wind_avg_ms"])
        out[dd] = {
            "night": float(r["temp_night_avg_c"]),
            "tday": float(r["temp_day_avg_c"]),
            "tmin": float(r["temp_min_c"]),
            "tmax": float(r["temp_max_c"]),
            "wind": wind,
            "rad": float(r["radiation_mj_m2"]),
            "rh": rh,
            "rain": float(r["precipitation_mm"]),
            "et0": float(r["et0_mm"]),
            "wind_dry": wind * (100.0 - rh),
        }
    return out


# -----------------------------------------------------------------------------
# Sama põllu kasvukiirus
# -----------------------------------------------------------------------------

def _field_history(events: Sequence[HarvestEvent], field: int) -> List[HarvestEvent]:
    return sorted([e for e in events if e.field == field], key=_event_key)


def _growth_days(prev: HarvestEvent, cur: HarvestEvent) -> float:
    g = float((cur.day - prev.day).days) + (cur.order - prev.order) * (HOURS_PER_FIELD / 24.0)
    return max(0.5, g)


def _l3_7_dates(cur: HarvestEvent) -> List[date]:
    start = cur.day - timedelta(days=7)
    end = cur.day - timedelta(days=3)
    return [start + timedelta(days=i) for i in range(5)]


def _prior_high_threshold(weather: Dict[date, dict], before_day: date, key: str) -> Optional[float]:
    vals = [float(w[key]) for dd, w in weather.items() if dd < before_day and np.isfinite(float(w[key]))]
    if len(vals) < MIN_DAYS_FOR_HIGH_THRESHOLD:
        return None
    return float(np.quantile(np.asarray(vals, dtype=float), HIGH_Q))


def _max_consecutive_true(flags: Sequence[bool]) -> int:
    best = cur = 0
    for flag in flags:
        if flag:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def _rolling_max_sum(vals: Sequence[float], n: int) -> float:
    a = np.asarray(vals, dtype=float)
    if len(a) < n:
        return float("nan")
    return float(max(np.sum(a[i:i+n]) for i in range(len(a) - n + 1)))


def _window_shape(weather: Dict[date, dict], days: Sequence[date]) -> Optional[Dict[str, float]]:
    if len(days) != 5 or any(d not in weather for d in days):
        return None
    rows = [weather[d] for d in days]
    wd = [float(r["wind_dry"]) for r in rows]
    et0 = [float(r["et0"]) for r in rows]
    wind = [float(r["wind"]) for r in rows]
    rh = [float(r["rh"]) for r in rows]

    wd_hi = _prior_high_threshold(weather, days[0], "wind_dry")
    et0_hi = _prior_high_threshold(weather, days[0], "et0")
    wd_flags = [v >= wd_hi for v in wd] if wd_hi is not None else [False] * len(wd)
    et0_flags = [v >= et0_hi for v in et0] if et0_hi is not None else [False] * len(et0)

    return {
        # LAB-152 referents
        "wd_avg": float(np.mean(wd)),
        "et0_avg": float(np.mean(et0)),
        "wind_avg": float(np.mean(wind)),
        "rh_avg": float(np.mean(rh)),

        # 1 päeva tipp
        "wd_peak1": float(max(wd)),
        "et0_peak1": float(max(et0)),

        # 2 järjestikuse päeva suurim koormus (summa)
        "wd_run2": _rolling_max_sum(wd, 2),
        "et0_run2": _rolling_max_sum(et0, 2),

        # 3 järjestikuse päeva suurim kumulatiivne koormus
        "wd_load3": _rolling_max_sum(wd, 3),
        "et0_load3": _rolling_max_sum(et0, 3),

        # kogu 5 päeva koormus
        "wd_load5": float(sum(wd)),
        "et0_load5": float(sum(et0)),

        # dünaamiline HIGH lävi: ainult varasemast ilmast
        "wd_hi_threshold": float(wd_hi) if wd_hi is not None else float("nan"),
        "et0_hi_threshold": float(et0_hi) if et0_hi is not None else float("nan"),
        "wd_high_days": float(sum(wd_flags)) if wd_hi is not None else float("nan"),
        "et0_high_days": float(sum(et0_flags)) if et0_hi is not None else float("nan"),
        "wd_high_run": float(_max_consecutive_true(wd_flags)) if wd_hi is not None else float("nan"),
        "et0_high_run": float(_max_consecutive_true(et0_flags)) if et0_hi is not None else float("nan"),
    }


def _event_window_features(cur: HarvestEvent, prev: HarvestEvent, prevprev: Optional[HarvestEvent], weather: Dict[date, dict]) -> Optional[Dict[str, float]]:
    current = _window_shape(weather, _l3_7_dates(cur))
    if current is None:
        return None
    previous = _window_shape(weather, _l3_7_dates(prev)) if prevprev is not None else None
    out = dict(current)
    for k, v in current.items():
        if k.endswith("threshold"):
            continue
        pv = previous.get(k) if previous is not None else None
        out[f"d_{k}"] = float(v - pv) if pv is not None and np.isfinite(v) and np.isfinite(pv) else 0.0
    out["has_prev_window"] = 1.0 if previous is not None else 0.0
    return out


def _analysis_rows(events: List[HarvestEvent], weather: Dict[date, dict]) -> pd.DataFrame:
    records: List[dict] = []
    for field in range(1, 15):
        hist = _field_history(events, field)
        for i in range(1, len(hist)):
            cur, prev = hist[i], hist[i - 1]
            prevprev = hist[i - 2] if i >= 2 else None

            cur_growth = _growth_days(prev, cur)
            if prevprev is not None:
                prev_growth = _growth_days(prevprev, prev)
            elif prev.interval_days is not None and prev.interval_days > 0:
                prev_growth = float(prev.interval_days)
            else:
                continue

            cur_rate = cur.abc / max(0.5, cur_growth)
            prev_rate = prev.abc / max(0.5, prev_growth)
            if cur_rate < 0 or prev_rate <= 0:
                continue

            wx = _event_window_features(cur, prev, prevprev, weather)
            if wx is None:
                continue

            y = math.log((cur_rate + TARGET_EPS) / (prev_rate + TARGET_EPS))
            rec = {
                "date": cur.day,
                "field": field,
                "order": cur.order,
                "source": cur.source,
                "abc": cur.abc,
                "prev_abc": prev.abc,
                "growth": cur_growth,
                "prev_growth": prev_growth,
                "growth_delta": cur_growth - prev_growth,
                "cur_rate": cur_rate,
                "prev_rate": prev_rate,
                "prev_log_rate": math.log(prev_rate + TARGET_EPS),
                "season_day": float((cur.day - SEASON_START).days),
                "y": y,
                "actual_pct": _pct_from_log(y),
            }
            rec.update(wx)
            records.append(rec)

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values(["date", "order", "field"]).reset_index(drop=True)
    return df


# -----------------------------------------------------------------------------
# Walk-forward ridge
# -----------------------------------------------------------------------------

BASE_COLS = ["prev_log_rate", "season_day", "growth", "growth_delta"]


def _candidate_specs() -> Dict[str, List[str]]:
    # Kitsas, ette määratud sündmuse kuju test. Iga plokk = tase + muutus eelmise sama põllu L3–7 akna vastu.
    return {
        "WD AVG · LAB152": ["wd_avg", "d_wd_avg"],
        "WD PEAK · 1p": ["wd_peak1", "d_wd_peak1"],
        "WD RUN · 2p": ["wd_run2", "d_wd_run2"],
        "WD LOAD · 3p": ["wd_load3", "d_wd_load3"],
        "WD LOAD · 5p": ["wd_load5", "d_wd_load5"],
        "WD HIGH kestus": ["wd_high_days", "wd_high_run", "d_wd_high_days", "d_wd_high_run"],

        "ET0 AVG": ["et0_avg", "d_et0_avg"],
        "ET0 PEAK · 1p": ["et0_peak1", "d_et0_peak1"],
        "ET0 RUN · 2p": ["et0_run2", "d_et0_run2"],
        "ET0 LOAD · 3p": ["et0_load3", "d_et0_load3"],
        "ET0 LOAD · 5p": ["et0_load5", "d_et0_load5"],
        "ET0 HIGH kestus": ["et0_high_days", "et0_high_run", "d_et0_high_days", "d_et0_high_run"],

        # Väikesed kombod: kas kestus + aurumisnõudlus annavad teineteisele lisainfot?
        "WD2 + ET02": ["wd_run2", "d_wd_run2", "et0_run2", "d_et0_run2"],
        "WD3 + ET03": ["wd_load3", "d_wd_load3", "et0_load3", "d_et0_load3"],
    }


def _ridge_predict_one(X_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, alpha: float) -> float:
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    x_test = np.asarray(x_test, dtype=float)
    mu = np.nanmean(X_train, axis=0)
    sd = np.nanstd(X_train, axis=0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    Z = (X_train - mu) / sd
    zt = (x_test - mu) / sd
    Xd = np.column_stack([np.ones(len(Z)), Z])
    reg = np.eye(Xd.shape[1]) * float(alpha)
    reg[0, 0] = 0.0
    try:
        beta = np.linalg.solve(Xd.T @ Xd + reg, Xd.T @ y_train)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(Xd.T @ Xd + reg) @ (Xd.T @ y_train)
    return float(np.r_[1.0, zt] @ beta)


def _walk_forward(df: pd.DataFrame, extra_cols: Sequence[str]) -> np.ndarray:
    pred = np.full(len(df), np.nan, dtype=float)
    cols = list(BASE_COLS) + list(extra_cols)
    vals = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df["y"], errors="coerce").to_numpy(dtype=float)
    dates = np.array(df["date"].tolist(), dtype=object)
    for dd in sorted(set(dates)):
        test_idx = np.where(dates == dd)[0]
        train_idx = np.where(dates < dd)[0]
        if len(train_idx) < MIN_TRAIN_ROWS:
            continue
        ok = np.isfinite(y[train_idx]) & np.all(np.isfinite(vals[train_idx]), axis=1)
        tr = train_idx[ok]
        if len(tr) < MIN_TRAIN_ROWS:
            continue
        for j in test_idx:
            if np.isfinite(y[j]) and np.all(np.isfinite(vals[j])):
                pred[j] = _ridge_predict_one(vals[tr], y[tr], vals[j], RIDGE_ALPHA)
    return pred


def _field_metrics(df: pd.DataFrame, pred: np.ndarray) -> Dict[str, float]:
    y = df["y"].to_numpy(dtype=float)
    mask = np.isfinite(y) & np.isfinite(pred)
    if not np.any(mask):
        return {"n": 0}
    actual_pct = np.array([_pct_from_log(v) for v in y], dtype=float)
    pred_pct = np.array([_pct_from_log(v) if np.isfinite(v) else np.nan for v in pred], dtype=float)
    actual_abc = df["abc"].to_numpy(dtype=float)
    pred_rate = df["prev_rate"].to_numpy(dtype=float) * np.exp(pred)
    pred_abc = pred_rate * df["growth"].to_numpy(dtype=float)
    drop = mask & (actual_pct <= 100.0 * BIG_DROP)
    non_drop = mask & (actual_pct > 100.0 * DROP_WARNING)

    def _mae(m):
        return float(np.mean(np.abs(pred[m] - y[m]))) if np.any(m) else float("nan")

    return {
        "n": int(np.sum(mask)),
        "log_mae": _mae(mask),
        "drop_n": int(np.sum(drop)),
        "drop_log_mae": _mae(drop),
        "direction": float(np.mean(np.sign(pred[mask]) == np.sign(y[mask])) * 100.0),
        "drop_recall": float(np.mean(pred_pct[drop] <= 100.0 * DROP_WARNING) * 100.0) if np.any(drop) else float("nan"),
        "false_alarm": float(np.mean(pred_pct[non_drop] <= 100.0 * DROP_WARNING) * 100.0) if np.any(non_drop) else float("nan"),
        "smape": _safe_smape(actual_abc[mask], pred_abc[mask]),
    }


def _event_day_metrics(df: pd.DataFrame, pred: np.ndarray) -> Dict[str, float]:
    rows = []
    for dd, grp in df.assign(_pred=pred).groupby("date"):
        g = grp[np.isfinite(grp["_pred"].to_numpy(dtype=float))].copy()
        if len(g) < 3:  # ainult tavalised täielikud 3-põllu korjepäevad
            continue
        actual = g["actual_pct"].to_numpy(dtype=float)
        predicted = np.array([_pct_from_log(x) for x in g["_pred"].to_numpy(dtype=float)], dtype=float)
        actual_big = int(np.sum(actual <= 100.0 * BIG_DROP)) >= 2
        pred_warn = float(np.median(predicted)) <= 100.0 * DROP_WARNING
        rows.append((dd, actual_big, pred_warn, float(np.median(actual)), float(np.median(predicted))))
    if not rows:
        return {"event_n": 0}
    big = [r for r in rows if r[1]]
    safe = [r for r in rows if r[3] > 100.0 * DROP_WARNING]
    return {
        "event_n": len(rows),
        "major_event_n": len(big),
        "event_recall": 100.0 * sum(1 for r in big if r[2]) / len(big) if big else float("nan"),
        "event_false_alarm": 100.0 * sum(1 for r in safe if r[2]) / len(safe) if safe else float("nan"),
    }


def _run_all(df: pd.DataFrame):
    base_pred = _walk_forward(df, [])
    specs = _candidate_specs()
    results = []
    details = {}
    for name, cols in specs.items():
        if any(c not in df.columns for c in cols):
            continue
        cand = _walk_forward(df, cols)
        common = np.isfinite(base_pred) & np.isfinite(cand)
        if not np.any(common):
            continue
        bp = np.where(common, base_pred, np.nan)
        cp = np.where(common, cand, np.nan)
        bm = _field_metrics(df, bp)
        cm = _field_metrics(df, cp)
        em = _event_day_metrics(df, cp)
        if cm.get("n", 0) == 0:
            continue
        overall_gain = bm["log_mae"] - cm["log_mae"]
        drop_gain = bm["drop_log_mae"] - cm["drop_log_mae"] if np.isfinite(bm["drop_log_mae"]) and np.isfinite(cm["drop_log_mae"]) else np.nan
        score = (0.65 * drop_gain + 0.35 * overall_gain) if np.isfinite(drop_gain) else overall_gain
        results.append({
            "Kandidaat": name,
            "N": cm["n"],
            "Suuri langusi N": cm["drop_n"],
            "Kasu langustel": drop_gain,
            "Kasu kokku": overall_gain,
            "Suuna täpsus %": cm["direction"],
            "Languse tabamus %": cm["drop_recall"],
            "Valehäire %": cm["false_alarm"],
            "Saagi sMAPE %": cm["smape"],
            "Event-päevi N": em["event_n"],
            "Suuri event-päevi N": em["major_event_n"],
            "Event tabamus %": em["event_recall"],
            "Event valehäire %": em["event_false_alarm"],
            "Skoor": score,
        })
        details[name] = (bp, cp)
    res = pd.DataFrame(results)
    if not res.empty:
        res = res.sort_values(["Skoor", "Kasu langustel", "Kasu kokku"], ascending=False).reset_index(drop=True)
    return res, details, base_pred


# -----------------------------------------------------------------------------
# Kuvamise abid
# -----------------------------------------------------------------------------

def _event_day_table(df: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    rows = []
    tmp = df.assign(_pred=pred)
    for dd, g in tmp.groupby("date"):
        g = g[np.isfinite(g["_pred"].to_numpy(dtype=float))].copy()
        if g.empty:
            continue
        actual = g["actual_pct"].to_numpy(dtype=float)
        pred_pct = np.array([_pct_from_log(x) for x in g["_pred"].to_numpy(dtype=float)], dtype=float)
        rows.append({
            "Korjepäev": dd,
            "Põlde": len(g),
            "Põllud": ", ".join(str(int(x)) for x in sorted(g["field"].tolist())),
            "Tegelik mediaan %": float(np.median(actual)),
            "Tegelik min %": float(np.min(actual)),
            "Kand mediaan %": float(np.median(pred_pct)),
            "WD peak 1p": float(g["wd_peak1"].iloc[0]),
            "WD run 2p": float(g["wd_run2"].iloc[0]),
            "WD load 3p": float(g["wd_load3"].iloc[0]),
            "WD high päevi": float(g["wd_high_days"].iloc[0]) if np.isfinite(g["wd_high_days"].iloc[0]) else np.nan,
            "WD max high jada": float(g["wd_high_run"].iloc[0]) if np.isfinite(g["wd_high_run"].iloc[0]) else np.nan,
            "WD high lävi": float(g["wd_hi_threshold"].iloc[0]) if np.isfinite(g["wd_hi_threshold"].iloc[0]) else np.nan,
            "ET0 peak 1p": float(g["et0_peak1"].iloc[0]),
            "ET0 run 2p": float(g["et0_run2"].iloc[0]),
            "ET0 load 3p": float(g["et0_load3"].iloc[0]),
            "ET0 high päevi": float(g["et0_high_days"].iloc[0]) if np.isfinite(g["et0_high_days"].iloc[0]) else np.nan,
            "ET0 max high jada": float(g["et0_high_run"].iloc[0]) if np.isfinite(g["et0_high_run"].iloc[0]) else np.nan,
            "ET0 high lävi": float(g["et0_hi_threshold"].iloc[0]) if np.isfinite(g["et0_hi_threshold"].iloc[0]) else np.nan,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("Korjepäev", ascending=False).reset_index(drop=True)
    return out


def _weather_event_table(weather: Dict[date, dict], start: date, end: date) -> pd.DataFrame:
    rows = []
    d = start
    while d <= end:
        if d in weather:
            w = weather[d]
            rows.append({
                "Kuupäev": d,
                "Tuul": w["wind"],
                "RH %": w["rh"],
                "WIND×DRY": w["wind_dry"],
                "ET0": w["et0"],
                "Päev T": w["tday"],
                "Tmin": w["tmin"],
                "Rad": w["rad"],
                "Sade": w["rain"],
            })
        d += timedelta(days=1)
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Streamlit
# -----------------------------------------------------------------------------

st.set_page_config(page_title="KurgiMootor LAB-153", layout="wide")
st.title("LAB-153 · Kas stress on 1 päev, 2 päeva või 3 päeva koormus?")
st.caption(
    "READ ONLY · L3–7 aken lukustatud · WIND×DRY ja ET0 sündmuse kestus · walk-forward ainult varasematel kuupäevadel · taimeindeks puudub"
)

with st.sidebar:
    st.subheader("LAB käsiread")
    st.caption("Jäta tühjaks, kui korje on DB-s. DB-sse ei kirjutata.")
    manual_text = st.text_area(
        "YYYY-MM-DD;põld;ABC;järjekord[;intervall]",
        value="",
        height=100,
    )

try:
    harvest_raw = db.get_harvest_history(limit=1000)
    weather_raw = db.get_weather_rows(WEATHER_START, TODAY)
except Exception as exc:
    st.error(f"DB lugemine ebaõnnestus: {exc}")
    st.stop()

events = _merge_manual(_prepare_events(harvest_raw), _parse_manual_rows(manual_text))
weather = _measured_weather(weather_raw)
latest_weather = max(weather) if weather else None

c1, c2, c3 = st.columns(3)
c1.metric("Korjeread", len(events))
c2.metric("Kontrollitud mõõdetud ilmapäevi", len(weather))
c3.metric("Viimane mõõdetud ilm", latest_weather.strftime("%d.%m") if latest_weather else "—")
if latest_weather is None or latest_weather < TODAY - timedelta(days=1):
    st.warning("Mõõdetud ilma lõpp on vanem kui eilne. Värskete kontrollpunktide L3–7 aken võib siiski olla täielik, kuid kontrolli kuupäeva.")

with st.spinner("Arvutan sündmuse kestuse walk-forward kontrolli…"):
    df = _analysis_rows(events, weather)
    if df.empty:
        st.error("Analüüsiks ei tekkinud täielikke ridu.")
        st.stop()
    results, details, base_pred = _run_all(df)

if results.empty:
    st.error("Walk-forward analüüsiks ei tekkinud piisavalt ridu.")
    st.stop()

st.subheader("1. Milline sündmuse kuju kannab signaali?")
st.caption(
    "PEAK 1p = L3–7 akna üks tugevaim päev. RUN 2p = suurim kahe järjestikuse päeva summa. "
    "LOAD 3p = suurim kolme järjestikuse päeva summa. HIGH kestus kasutab 75. protsentiili läve, "
    "mis arvutatakse ainult enne seda akent juba möödunud ilmast."
)
show_cols = [
    "Kandidaat", "N", "Suuri langusi N", "Kasu langustel", "Kasu kokku",
    "Suuna täpsus %", "Languse tabamus %", "Valehäire %", "Saagi sMAPE %",
    "Event-päevi N", "Suuri event-päevi N", "Event tabamus %", "Event valehäire %",
]
st.dataframe(
    results[show_cols].style.format({
        "Kasu langustel": "{:+.3f}", "Kasu kokku": "{:+.3f}",
        "Suuna täpsus %": "{:.0f}", "Languse tabamus %": "{:.0f}", "Valehäire %": "{:.0f}",
        "Saagi sMAPE %": "{:.1f}", "Event tabamus %": "{:.0f}", "Event valehäire %": "{:.0f}",
    }, na_rep="—"),
    use_container_width=True, hide_index=True,
)

best = results.iloc[0]
st.info(
    f"Parim ette lukustatud kestuse kandidaat: **{best['Kandidaat']}** · "
    f"sMAPE **{float(best['Saagi sMAPE %']):.1f}%** · "
    f"põllurea suurte languste tabamus **{float(best['Languse tabamus %']):.0f}%** · "
    f"event-päeva tabamus **{float(best['Event tabamus %']):.0f}%**."
)

st.subheader("2. Üks korjepäev = üks ilmastikusündmus")
st.caption(
    "Sama päeva põllud ei ole kolm sõltumatut ilmaepisoodi. Siin on need kokku tõstetud üheks event-päevaks. "
    "Suur event-päev = vähemalt 2 põldu 3-st kukuvad kasvukiiruses ≥30%."
)
best_name = str(best["Kandidaat"])
_, best_pred = details[best_name]
evt = _event_day_table(df, best_pred)
recent_cut = TODAY - timedelta(days=14)
evt_focus = evt[(evt["Korjepäev"] >= recent_cut) | (evt["Tegelik mediaan %"] <= -20)].copy()
st.dataframe(
    evt_focus.style.format({
        "Tegelik mediaan %": "{:+.0f}%", "Tegelik min %": "{:+.0f}%", "Kand mediaan %": "{:+.0f}%",
        "WD peak 1p": "{:.1f}", "WD run 2p": "{:.1f}", "WD load 3p": "{:.1f}",
        "WD high päevi": "{:.0f}", "WD max high jada": "{:.0f}", "WD high lävi": "{:.1f}",
        "ET0 peak 1p": "{:.2f}", "ET0 run 2p": "{:.2f}", "ET0 load 3p": "{:.2f}",
        "ET0 high päevi": "{:.0f}", "ET0 max high jada": "{:.0f}", "ET0 high lävi": "{:.2f}",
    }, na_rep="—"),
    use_container_width=True, hide_index=True,
)

st.subheader("3. Juuli vs august · päevased stressinäitajad")
left, right = st.columns(2)
with left:
    st.markdown("**Juuli kontrollaken 05.–12.07**")
    july = _weather_event_table(weather, date(TODAY.year, 7, 5), date(TODAY.year, 7, 12))
    st.dataframe(july.style.format({
        "Tuul": "{:.1f}", "RH %": "{:.0f}", "WIND×DRY": "{:.1f}", "ET0": "{:.2f}",
        "Päev T": "{:.1f}", "Tmin": "{:.1f}", "Rad": "{:.1f}", "Sade": "{:.1f}",
    }, na_rep="—"), use_container_width=True, hide_index=True)
with right:
    st.markdown(f"**Augusti kontrollaken {(TODAY - timedelta(days=8)).strftime('%d.%m')}–{(TODAY - timedelta(days=1)).strftime('%d.%m')}**")
    aug = _weather_event_table(weather, TODAY - timedelta(days=8), TODAY - timedelta(days=1))
    st.dataframe(aug.style.format({
        "Tuul": "{:.1f}", "RH %": "{:.0f}", "WIND×DRY": "{:.1f}", "ET0": "{:.2f}",
        "Päev T": "{:.1f}", "Tmin": "{:.1f}", "Rad": "{:.1f}", "Sade": "{:.1f}",
    }, na_rep="—"), use_container_width=True, hide_index=True)

st.subheader("4. Kuidas tulemust lugeda")
st.markdown(
    """
- Kui **PEAK 1p** võidab selgelt, piisab tõenäoliselt ühest tugevast stressipäevast.
- Kui **RUN 2p** on tugevam, on oluline vähemalt kahe järjestikuse päeva koormus.
- Kui **LOAD 3p** võidab, loeb pigem mitmepäevane kumulatiivne stress kui üks tipp.
- Kui **HIGH kestus** võidab, on tähtis läve ületavate päevade arv/jada, mitte ainult summa.
- Kui **WD + ET0** kombo võidab, kannavad tuul×kuivus ja aurumisnõudlus teineteisest eraldi infot.

Põllurea tabamus ja event-päeva tabamus tuleb vaadata koos. Viimane ei loe sama ilmaepisoodi kolme põllu tõttu kolm korda.
"""
)

st.caption(f"{LAB_VERSION} · ainult lugemine · {TODAY.isoformat()}")
