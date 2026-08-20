from __future__ import annotations

"""
KurgiMootor LAB-151 · WEATHER CAUSE AUDIT
==========================================

Eesmärk
-------
Leida, milline ilmastikusignaal eelneb sama põllu A+B+C kasvukiiruse muutusele,
ilma et taime-/põlluindeks saaks põhjuseotsingus ilmaseost ära peita.

READ ONLY
---------
- loeb ainult db.get_harvest_history() ja db.get_weather_rows()
- EI salvesta prognoose, seadeid, korjeid ega ilma
- käsitsi lisatud LAB-read elavad ainult käesolevas Streamliti sessioonis

Oluline metoodika
-----------------
- siht = sama põllu ABC kasvukiiruse log-muutus võrreldes eelmise korjega;
- kasvuaeg arvestab võimalusel korjejärjekorda (~3 h / põld), nagu app-128;
- ilmaaknad lõpevad hiljemalt päev ENNE korjet, et vältida harvest-day tulevikuleket;
- walk-forward: ühe päeva ennustamisel treenitakse ainult VARASEMATEL kuupäevadel;
- BASE kasutab ainult eelmist sama põllu kasvukiirust + hooajapäeva + kasvuaega;
- taimeindeksit / põllu latentset seisundit põhjuseotsingus EI kasutata;
- VPD on päevase T + RH põhine proxy, mitte põllul mõõdetud tunnine VPD.
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import math

import numpy as np
import pandas as pd
import streamlit as st

import db


TZ = ZoneInfo("Europe/Tallinn")
TODAY = datetime.now(TZ).date()
SEASON_START = date(TODAY.year, 6, 15)
WEATHER_START = date(TODAY.year, 7, 1)
LAB_VERSION = "LAB-151-WEATHER-CAUSE-AUDIT-V1"

HOURS_PER_FIELD = 3.0
DEFAULT_GROWTH_DAYS = 14.0 / 3.0
RIDGE_ALPHA = 10.0                 # fikseeritud enne tulemust; standardiseeritud tunnused
MIN_TRAIN_ROWS = 24
TARGET_EPS = 0.20
BIG_DROP = -0.30                   # tegelik kasvukiiruse langus >=30%
DROP_WARNING = -0.20               # mudeli hoiatuslävi

REQUIRED_WEATHER = (
    "temp_night_avg_c", "temp_day_avg_c", "temp_min_c", "temp_max_c",
    "wind_avg_ms", "radiation_mj_m2", "humidity_avg_pct",
    "precipitation_mm", "et0_mm",
)

WINDOWS = {
    "GROW": "Eelmisest korjest kuni päev enne korjet",
    "L3-7": "3–7 päeva enne korjet",
    "L5-10": "5–10 päeva enne korjet",
}

TEMP_OFFSETS = (0.0, -0.5, -1.0)


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
        # LAB-i käsirea jaoks võib _abc olla otse ette antud.
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


def _sat_vp_kpa(temp_c: float) -> float:
    # FAO-56 küllastunud veeaururõhu valem.
    return 0.6108 * math.exp((17.27 * temp_c) / (temp_c + 237.3))


def _vpd_proxy_kpa(day_temp_c: float, rh_pct: float) -> float:
    rh = min(100.0, max(0.0, float(rh_pct)))
    return max(0.0, _sat_vp_kpa(float(day_temp_c)) * (1.0 - rh / 100.0))


def _temp_assim_factor(day_temp_c: float) -> float:
    """Õrn, monotonne cool-side proxy; mitte agronoomiline 'tõde'.

    10 °C -> 0, 22 °C -> 1, 22...28 °C -> 1, üle 28 °C langeb pehmelt.
    Seda kasutatakse ainult kombinatsioonitunnusena ja valideeritakse walk-forward'is.
    """
    t = float(day_temp_c)
    if t <= 10.0:
        return 0.0
    if t < 22.0:
        return (t - 10.0) / 12.0
    if t <= 28.0:
        return 1.0
    return max(0.25, 1.0 - 0.075 * (t - 28.0))


@dataclass
class HarvestEvent:
    day: date
    field: int
    order: int
    abc: float
    interval_days: Optional[float]
    reliable: bool
    source: str


def _event_key(e: HarvestEvent) -> Tuple[date, int, int]:
    return (e.day, e.order, e.field)


# -----------------------------------------------------------------------------
# Andmete laadimine / käsiread
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
        interval = _f(r.get("interval_days"))
        out.append(HarvestEvent(dd, field, order, float(abc), interval, True, "DB"))
    out.sort(key=_event_key)
    return out


def _parse_manual_rows(text: str) -> List[HarvestEvent]:
    """Formaat: YYYY-MM-DD,põld,ABC,järjekord[,intervall]."""
    out: List[HarvestEvent] = []
    for line_no, raw in enumerate((text or "").splitlines(), start=1):
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        parts = [p.strip().replace(",", ".") for p in raw.replace(";", ",").split(",")]
        # Kui kasutaja kasutab komakohta, on semikoolon kindlam. Siin toetame ka
        # standardset CSV-d, kus ABC on punktiga.
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
            out.append(HarvestEvent(dd, field, order, abc, interval, True, "LAB käsirida"))
    return out


def _merge_manual(events: List[HarvestEvent], manual: List[HarvestEvent]) -> List[HarvestEvent]:
    by_key = {(e.day, e.field): e for e in events}
    for e in manual:
        # Kui DB-s on sama päev+põld juba olemas, usaldame DB-d.
        by_key.setdefault((e.day, e.field), e)
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
        out[dd] = {
            "temp_night": float(r["temp_night_avg_c"]),
            "temp_day": float(r["temp_day_avg_c"]),
            "temp_min": float(r["temp_min_c"]),
            "temp_max": float(r["temp_max_c"]),
            "wind": float(r["wind_avg_ms"]),
            "rad": float(r["radiation_mj_m2"]),
            "rh": float(r["humidity_avg_pct"]),
            "rain": float(r["precipitation_mm"]),
            "et0": float(r["et0_mm"]),
            "source_station": str(r.get("source_station") or ""),
        }
    return out


# -----------------------------------------------------------------------------
# Kasvuaeg / sama põllu siht
# -----------------------------------------------------------------------------

def _field_history(events: Sequence[HarvestEvent], field: int) -> List[HarvestEvent]:
    return sorted([e for e in events if e.field == field], key=_event_key)


def _growth_days(prev: HarvestEvent, cur: HarvestEvent) -> float:
    g = float((cur.day - prev.day).days) + (cur.order - prev.order) * (HOURS_PER_FIELD / 24.0)
    return max(0.5, g)


def _own_growth_days(events: Sequence[HarvestEvent], idx: int) -> Optional[float]:
    e = events[idx]
    hist = [x for x in events[:idx] if x.field == e.field and x.day <= e.day]
    if hist:
        prev = hist[-1]
        return _growth_days(prev, e)
    if e.interval_days is not None and e.interval_days > 0:
        return float(e.interval_days)
    return None


# -----------------------------------------------------------------------------
# Ilma tuletised ja aknad
# -----------------------------------------------------------------------------

def _daily_metrics(raw: dict, temp_offset: float) -> Dict[str, float]:
    night = raw["temp_night"] + temp_offset
    dayt = raw["temp_day"] + temp_offset
    tmin = raw["temp_min"] + temp_offset
    tmax = raw["temp_max"] + temp_offset
    tmean = 0.5 * (night + dayt)
    wind = raw["wind"]
    rad = raw["rad"]
    rh = raw["rh"]
    et0 = raw["et0"]
    rain = raw["rain"]
    vpd = _vpd_proxy_kpa(dayt, rh)
    gdd10 = max(0.0, tmean - 10.0)
    assim = rad * _temp_assim_factor(dayt)
    cool_lowrad = (max(0.0, 20.0 - tmax) / 10.0) * (max(0.0, 12.0 - rad) / 12.0)
    return {
        "night": night,
        "tday": dayt,
        "tmin": tmin,
        "tmax": tmax,
        "tmean": tmean,
        "wind": wind,
        "rad": rad,
        "rh": rh,
        "et0": et0,
        "rain": rain,
        "vpd": vpd,
        "gdd10": gdd10,
        "rad_gdd": rad * gdd10,
        "assim": assim,
        "wind_tmax": wind * tmax,
        "wind_rad": wind * rad,
        "wind_et0": wind * et0,
        "wind_dry": wind * (100.0 - rh),
        "wind_cool": wind * max(0.0, 20.0 - dayt),
        "wind_lowrad": wind * max(0.0, 15.0 - rad),
        "vpd_wind": vpd * wind,
        "vpd_rad": vpd * rad,
        "cool14": 1.0 if night < 14.0 else 0.0,
        "cool12": 1.0 if tmin <= 12.0 else 0.0,
        "lowrad12": 1.0 if rad < 12.0 else 0.0,
        "lowrad10": 1.0 if rad < 10.0 else 0.0,
        "cool_lowrad": cool_lowrad,
    }


def _window_dates(kind: str, cur: HarvestEvent, prev: HarvestEvent) -> List[date]:
    # Kõik aknad lõpevad hiljemalt päev enne korjet -> ei kasuta harvest-day täispäeva ilma.
    if kind == "GROW":
        start = prev.day + timedelta(days=1)
        end = cur.day - timedelta(days=1)
    elif kind == "L3-7":
        start = cur.day - timedelta(days=7)
        end = cur.day - timedelta(days=3)
    elif kind == "L5-10":
        start = cur.day - timedelta(days=10)
        end = cur.day - timedelta(days=5)
    else:
        raise ValueError(kind)
    if end < start:
        return []
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def _aggregate_window(weather: Dict[date, dict], days: Sequence[date], temp_offset: float) -> Optional[Dict[str, float]]:
    if not days or any(d not in weather for d in days):
        return None
    rows = [_daily_metrics(weather[d], temp_offset) for d in days]
    keys = list(rows[0].keys())
    out = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    out["rad_sum"] = float(sum(r["rad"] for r in rows))
    out["et0_sum"] = float(sum(r["et0"] for r in rows))
    out["rain_sum"] = float(sum(r["rain"] for r in rows))
    out["n_days"] = float(len(rows))
    return out


def _event_window_features(
    cur: HarvestEvent,
    prev: HarvestEvent,
    prevprev: Optional[HarvestEvent],
    weather: Dict[date, dict],
    kind: str,
    temp_offset: float,
) -> Optional[Dict[str, float]]:
    current = _aggregate_window(weather, _window_dates(kind, cur, prev), temp_offset)
    if current is None:
        return None
    if prevprev is not None:
        previous = _aggregate_window(weather, _window_dates(kind, prev, prevprev), temp_offset)
    else:
        previous = None

    out = dict(current)
    for k, v in current.items():
        if k == "n_days":
            continue
        out[f"d_{k}"] = float(v - previous[k]) if previous is not None and k in previous else 0.0
    out["has_prev_window"] = 1.0 if previous is not None else 0.0
    return out


# -----------------------------------------------------------------------------
# Analüüsiridade ehitus
# -----------------------------------------------------------------------------

def _analysis_rows(events: List[HarvestEvent], weather: Dict[date, dict], temp_offset: float, window: str) -> pd.DataFrame:
    records: List[dict] = []
    by_field: Dict[int, List[HarvestEvent]] = {f: _field_history(events, f) for f in range(1, 15)}

    for field, hist in by_field.items():
        for i in range(1, len(hist)):
            cur = hist[i]
            prev = hist[i - 1]
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

            wx = _event_window_features(cur, prev, prevprev, weather, window, temp_offset)
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
                "is_big_drop": bool(_pct_from_log(y) <= 100.0 * BIG_DROP),
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
    return {
        "RAD": ["rad", "d_rad", "rad_sum", "d_rad_sum"],
        "TEMP": ["night", "tday", "tmin", "tmax", "cool14", "cool12", "d_night", "d_tday", "d_tmin", "d_tmax", "d_cool14", "d_cool12"],
        "WIND": ["wind", "d_wind"],
        "VPD": ["vpd", "d_vpd"],
        "RAD×TEMP": ["rad", "gdd10", "rad_gdd", "assim", "d_rad", "d_gdd10", "d_rad_gdd", "d_assim"],
        "WIND×RAD": ["wind", "rad", "wind_rad", "wind_lowrad", "d_wind", "d_rad", "d_wind_rad", "d_wind_lowrad"],
        "WIND-STRESS": ["wind_tmax", "wind_rad", "wind_et0", "wind_dry", "wind_cool", "d_wind_tmax", "d_wind_rad", "d_wind_et0", "d_wind_dry", "d_wind_cool"],
        "VPD×WIND": ["vpd", "wind", "vpd_wind", "d_vpd", "d_wind", "d_vpd_wind"],
        "VPD×RAD": ["vpd", "rad", "vpd_rad", "d_vpd", "d_rad", "d_vpd_rad"],
        "LOW-ASSIM": ["cool_lowrad", "cool14", "lowrad12", "lowrad10", "assim", "d_cool_lowrad", "d_cool14", "d_lowrad12", "d_lowrad10", "d_assim"],
        "CORE4": ["rad", "tday", "vpd", "wind", "d_rad", "d_tday", "d_vpd", "d_wind"],
    }


def _temp_sensitive(name: str) -> bool:
    return name not in {"RAD", "WIND"}


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
        train_ok = np.isfinite(y[train_idx]) & np.all(np.isfinite(vals[train_idx]), axis=1)
        tr = train_idx[train_ok]
        if len(tr) < MIN_TRAIN_ROWS:
            continue
        for j in test_idx:
            if np.isfinite(y[j]) and np.all(np.isfinite(vals[j])):
                pred[j] = _ridge_predict_one(vals[tr], y[tr], vals[j], RIDGE_ALPHA)
    return pred


def _metrics(df: pd.DataFrame, pred: np.ndarray) -> Dict[str, float]:
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

    dir_acc = float(np.mean(np.sign(pred[mask]) == np.sign(y[mask])) * 100.0)
    drop_recall = float(np.mean(pred_pct[drop] <= 100.0 * DROP_WARNING) * 100.0) if np.any(drop) else float("nan")
    false_alarm = float(np.mean(pred_pct[non_drop] <= 100.0 * DROP_WARNING) * 100.0) if np.any(non_drop) else float("nan")

    return {
        "n": int(np.sum(mask)),
        "log_mae": _mae(mask),
        "smape": _safe_smape(actual_abc[mask], pred_abc[mask]),
        "direction": dir_acc,
        "drop_n": int(np.sum(drop)),
        "drop_log_mae": _mae(drop),
        "drop_recall": drop_recall,
        "false_alarm": false_alarm,
    }


def _run_all(events: List[HarvestEvent], weather: Dict[date, dict]):
    specs = _candidate_specs()
    all_results: List[dict] = []
    detail_store: Dict[Tuple[str, str, float], Tuple[pd.DataFrame, np.ndarray, np.ndarray]] = {}

    # BASE arvutame iga akna/offseti samal analüüsirea maskil, et võrdlus oleks aus.
    for window in WINDOWS:
        for offset in TEMP_OFFSETS:
            df = _analysis_rows(events, weather, offset, window)
            if df.empty:
                continue
            base_pred = _walk_forward(df, [])

            for name, cols in specs.items():
                if offset != 0.0 and not _temp_sensitive(name):
                    continue
                missing = [c for c in cols if c not in df.columns]
                if missing:
                    continue
                cand_pred = _walk_forward(df, cols)
                common = np.isfinite(base_pred) & np.isfinite(cand_pred)
                if not np.any(common):
                    continue

                # Mõlemad mõõdikud täpselt samadel ridadel.
                base_common = np.where(common, base_pred, np.nan)
                cand_common = np.where(common, cand_pred, np.nan)
                bm = _metrics(df, base_common)
                cm = _metrics(df, cand_common)
                if cm.get("n", 0) == 0:
                    continue

                overall_gain = bm.get("log_mae", np.nan) - cm.get("log_mae", np.nan)
                if np.isfinite(bm.get("drop_log_mae", np.nan)) and np.isfinite(cm.get("drop_log_mae", np.nan)):
                    drop_gain = bm["drop_log_mae"] - cm["drop_log_mae"]
                else:
                    drop_gain = np.nan
                score = (0.65 * drop_gain + 0.35 * overall_gain) if np.isfinite(drop_gain) else overall_gain

                all_results.append({
                    "Kandidaat": name,
                    "Aken": window,
                    "T nihe °C": offset,
                    "N": cm["n"],
                    "Suuri langusi N": cm["drop_n"],
                    "BASE log-MAE": bm["log_mae"],
                    "log-MAE": cm["log_mae"],
                    "Kasu kokku": overall_gain,
                    "BASE langus-MAE": bm["drop_log_mae"],
                    "Langus-MAE": cm["drop_log_mae"],
                    "Kasu langustel": drop_gain,
                    "Suuna täpsus %": cm["direction"],
                    "Languse tabamus %": cm["drop_recall"],
                    "Valehäire %": cm["false_alarm"],
                    "Saagi sMAPE %": cm["smape"],
                    "Skoor": score,
                })
                detail_store[(name, window, offset)] = (df.copy(), base_common, cand_common)

    res = pd.DataFrame(all_results)
    if not res.empty:
        res = res.sort_values(["Skoor", "Kasu langustel", "Kasu kokku"], ascending=False).reset_index(drop=True)
    return res, detail_store


# -----------------------------------------------------------------------------
# Kuvamise abid
# -----------------------------------------------------------------------------

def _weather_table(weather: Dict[date, dict], start: date, end: date, offset: float = 0.0) -> pd.DataFrame:
    rows = []
    d = start
    while d <= end:
        if d in weather:
            raw = weather[d]
            m = _daily_metrics(raw, offset)
            rows.append({
                "Kuupäev": d.strftime("%d.%m"),
                "Öö °C": m["night"],
                "Tmin °C": m["tmin"],
                "Päev °C": m["tday"],
                "Tmax °C": m["tmax"],
                "Tuul m/s": m["wind"],
                "RH %": m["rh"],
                "VPD proxy kPa": m["vpd"],
                "ET0 mm": m["et0"],
                "Rad MJ/m²": m["rad"],
                "Sade mm": m["rain"],
            })
        d += timedelta(days=1)
    return pd.DataFrame(rows)


def _event_detail(df: pd.DataFrame, base_pred: np.ndarray, cand_pred: np.ndarray) -> pd.DataFrame:
    out = df[["date", "field", "source", "prev_abc", "abc", "prev_rate", "cur_rate", "growth", "actual_pct"]].copy()
    out["BASE %"] = [_pct_from_log(x) if np.isfinite(x) else np.nan for x in base_pred]
    out["Kandidaat %"] = [_pct_from_log(x) if np.isfinite(x) else np.nan for x in cand_pred]
    out["Viga BASE pp"] = np.abs(out["BASE %"] - out["actual_pct"])
    out["Viga kandidaat pp"] = np.abs(out["Kandidaat %"] - out["actual_pct"])
    out = out[np.isfinite(out["Kandidaat %"])]
    out = out.sort_values(["date", "field"], ascending=[False, True])
    out = out.rename(columns={
        "date": "Kuupäev", "field": "Põld", "source": "Allikas",
        "prev_abc": "Eelmine ABC", "abc": "ABC", "growth": "Kasvuaeg p",
        "actual_pct": "Tegelik muutus %", "prev_rate": "Eelm kasvukiirus", "cur_rate": "Kasvukiirus",
    })
    return out


# -----------------------------------------------------------------------------
# Streamlit
# -----------------------------------------------------------------------------

st.set_page_config(page_title="KurgiMootor LAB-151", layout="wide")
st.title("LAB-151 · Mis saaki päriselt liigutab?")
st.caption(
    "READ ONLY · ilmastiku põhjuseotsing ilma taimeindeksita · sama põllu ABC kasvukiiruse muutus · "
    "walk-forward, ainult varasemad kuupäevad"
)

with st.sidebar:
    st.subheader("LAB käsiread")
    st.caption("DB-sse EI kirjutata. Kui rida on DB-s olemas, kasutatakse DB rida.")
    manual_text = st.text_area(
        "YYYY-MM-DD,põld,ABC,järjekord[,intervall]",
        value="2026-08-20,11,4.2,1",
        height=110,
        help="Lisa hiljem näiteks uuele reale: 2026-08-20,12,7.1,2",
    )
    st.caption("20.08 põld 11 = 4,2 on siin ainult tänase LAB-i värske kontrollpunktina.")

try:
    harvest_raw = db.get_harvest_history(limit=1000)
    weather_raw = db.get_weather_rows(WEATHER_START, TODAY)
except Exception as exc:
    st.error(f"DB lugemine ebaõnnestus: {exc}")
    st.stop()

events_db = _prepare_events(harvest_raw)
manual_events = _parse_manual_rows(manual_text)
events = _merge_manual(events_db, manual_events)
weather = _measured_weather(weather_raw)

latest_weather = max(weather) if weather else None
manual_added = [e for e in events if e.source == "LAB käsirida"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Korjeread", len(events))
c2.metric("Kontrollitud mõõdetud ilmapäevi", len(weather))
c3.metric("Viimane mõõdetud ilm", latest_weather.strftime("%d.%m") if latest_weather else "—")
c4.metric("LAB käsiridu lisatud", len(manual_added))

if latest_weather is None or latest_weather < TODAY - timedelta(days=1):
    st.warning("Mõõdetud ilma lõpp on vanem kui eilne. Värske augusti põhjuseotsing võib olla puudulik.")

with st.spinner("Arvutan walk-forward ilmaseoseid…"):
    results, details = _run_all(events, weather)

if results.empty:
    st.error("Analüüsiks ei tekkinud piisavalt täielikke ridu. Kontrolli mõõdetud ilma ja korjeajaloo katvust.")
    st.stop()

# Tugevad kandidaadid: ei tohi üldpilti märgatavalt halvemaks teha.
robust = results[(results["Kasu kokku"] >= -0.01) & (results["Suuri langusi N"] >= 2)].copy()
if robust.empty:
    robust = results.copy()

st.subheader("1. TOP kandidaadid")
st.caption(
    "Kasu > 0 tähendab, et ilmablokk parandas BASE-i. Järjestus eelistab suuri langusi (65%) ja alles siis kogu ajalugu (35%). "
    "BASE = eelmine sama põllu kasvukiirus + hooajapäev + kasvuaeg; taimeindeks puudub."
)
show_cols = [
    "Kandidaat", "Aken", "T nihe °C", "N", "Suuri langusi N",
    "Kasu langustel", "Kasu kokku", "Suuna täpsus %", "Languse tabamus %",
    "Valehäire %", "Saagi sMAPE %",
]
st.dataframe(
    robust.head(12)[show_cols].style.format({
        "T nihe °C": "{:+.1f}",
        "Kasu langustel": "{:+.3f}",
        "Kasu kokku": "{:+.3f}",
        "Suuna täpsus %": "{:.0f}",
        "Languse tabamus %": "{:.0f}",
        "Valehäire %": "{:.0f}",
        "Saagi sMAPE %": "{:.1f}",
    }, na_rep="—"),
    use_container_width=True,
    hide_index=True,
)

best = robust.iloc[0]
best_key = (str(best["Kandidaat"]), str(best["Aken"]), float(best["T nihe °C"]))

st.info(
    f"Hetke parim kontrollitav ilmablokk: **{best_key[0]} · {best_key[1]} · T nihe {best_key[2]:+.1f} °C**. "
    f"Kasu suurtel langustel {float(best['Kasu langustel']):+.3f} log-MAE ja kogu ajalool {float(best['Kasu kokku']):+.3f}."
)

st.subheader("2. Suured langused ja värsked kontrollpunktid")
df_best, base_best, cand_best = details[best_key]
event_table = _event_detail(df_best, base_best, cand_best)
# Näita kõik suuri langusi + viimase 12 päeva ridu.
latest_cut = TODAY - timedelta(days=12)
mask = (event_table["Tegelik muutus %"] <= 100.0 * BIG_DROP) | (event_table["Kuupäev"] >= latest_cut)
focus = event_table[mask].copy()
st.dataframe(
    focus.style.format({
        "Eelmine ABC": "{:.1f}", "ABC": "{:.1f}", "Kasvuaeg p": "{:.2f}",
        "Eelm kasvukiirus": "{:.2f}", "Kasvukiirus": "{:.2f}",
        "Tegelik muutus %": "{:+.0f}%", "BASE %": "{:+.0f}%", "Kandidaat %": "{:+.0f}%",
        "Viga BASE pp": "{:.0f}", "Viga kandidaat pp": "{:.0f}",
    }, na_rep="—"),
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "Kui ilmablokk aitab 21.–23.07 ja praegust augusti langust, kuid ei tekita 13–14 põllule vale suurt langust, on see tugevam mehhanismikandidaat. "
    "Ülejääv süsteemne põllupõhine viga on koht, kus taime-/põlluindeks võib olla põhjendatud."
)

st.subheader("3. Juuli saagipiduri ilm vs praegune ilm")
left, right = st.columns(2)
with left:
    st.markdown("**Juuli kontrollaken 05.–12.07**")
    july = _weather_table(weather, date(TODAY.year, 7, 5), date(TODAY.year, 7, 12), 0.0)
    st.dataframe(july.style.format({
        "Öö °C": "{:.1f}", "Tmin °C": "{:.1f}", "Päev °C": "{:.1f}", "Tmax °C": "{:.1f}",
        "Tuul m/s": "{:.1f}", "RH %": "{:.0f}", "VPD proxy kPa": "{:.2f}", "ET0 mm": "{:.2f}",
        "Rad MJ/m²": "{:.1f}", "Sade mm": "{:.1f}",
    }, na_rep="—"), use_container_width=True, hide_index=True)
with right:
    st.markdown(f"**Värske kontrollaken {(TODAY - timedelta(days=8)).strftime('%d.%m')}–{(TODAY - timedelta(days=1)).strftime('%d.%m')}**")
    recent = _weather_table(weather, TODAY - timedelta(days=8), TODAY - timedelta(days=1), 0.0)
    st.dataframe(recent.style.format({
        "Öö °C": "{:.1f}", "Tmin °C": "{:.1f}", "Päev °C": "{:.1f}", "Tmax °C": "{:.1f}",
        "Tuul m/s": "{:.1f}", "RH %": "{:.0f}", "VPD proxy kPa": "{:.2f}", "ET0 mm": "{:.2f}",
        "Rad MJ/m²": "{:.1f}", "Sade mm": "{:.1f}",
    }, na_rep="—"), use_container_width=True, hide_index=True)

st.subheader("4. Kas võimalik −0,5…−1,0 °C põllumõõde muudab järeldust?")
temp_view = results[results["Kandidaat"].isin(["TEMP", "VPD", "RAD×TEMP", "VPD×WIND", "LOW-ASSIM", "CORE4"])].copy()
# Iga kandidaat+aken top offset; lisaks kõik offsetid, et näha tundlikkust.
st.dataframe(
    temp_view.head(24)[["Kandidaat", "Aken", "T nihe °C", "Kasu langustel", "Kasu kokku", "Languse tabamus %", "Valehäire %"]]
    .style.format({
        "T nihe °C": "{:+.1f}", "Kasu langustel": "{:+.3f}", "Kasu kokku": "{:+.3f}",
        "Languse tabamus %": "{:.0f}", "Valehäire %": "{:.0f}",
    }, na_rep="—"),
    use_container_width=True,
    hide_index=True,
)

with st.expander("Metoodika / mida täpselt testitakse"):
    st.markdown(
        """
- **GROW**: ilm päev pärast eelmist korjet kuni päev enne uut korjet.
- **L3-7**: 3–7 päeva enne korjet; võimalik vilja kasvukiiruse/assimileerimise viide.
- **L5-10**: 5–10 päeva enne korjet; võimalik varasema viljaalgme/koormuse viide.
- **VPD proxy**: päevase õhutemperatuuri ja päevakeskmise RH põhjal. RH ei ole põllusensor, seega seda tuleb tõlgendada proksina.
- **WIND-STRESS**: sama mõte, mida tootmise Jäljeotsija juba uurib: tuul×Tmax, tuul×radiatsioon, tuul×ET0, tuul×kuivus (+ jaheduse variant).
- **LOW-ASSIM**: madala radiatsiooni ja jaheduse koosmõju + jahedate ööde osakaal.
- **Temperatuuri nihe 0 / −0,5 / −1,0 °C** ei väida, et põld on kindlasti külmem; see on tundlikkustest rannikujaama vs põllu mikrokliimale.
- **Põllu-/taimeindeksit ei kasutata.** Kui parim ilmablokk jätab mõne põllu järjekindlalt üles/alla, on see jääk eraldi alus taime-/põlluindeksile.
        """
    )

st.caption(f"{LAB_VERSION} · ainult lugemine · {TODAY.isoformat()}")
