from __future__ import annotations

"""
KurgiMootor LAB-152 · WIND-STRESS DECOMPOSITION
==========================================

Eesmärk
-------
LAB-151 leidis tugeva 3–7 päeva eelsignaali WIND-STRESS plokis.
LAB-152 EI otsi enam uusi aknaid: lukustab L3–7 akna ja võtab WIND-STRESSi
lahti komponentideks ning teeb ablation-kontrolli, et näha, mis osa signaali kannab.

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
LAB_VERSION = "LAB-152-WIND-STRESS-DECOMP-V1"

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
    # LUKUSTATUD enne LAB-152 tulemust: LAB-151 parim aken.
    "L3-7": "3–7 päeva enne korjet",
}

# LAB-151 näitas, et 0 / -0.5 / -1.0 °C nihe järeldust sisuliselt ei muutnud.
# Siin ei lisa me uut vabadusastet: kasutame jaama mõõdetud temperatuuri.
TEMP_OFFSETS = (0.0,)


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
    # LAB-152: ainult ette lukustatud L3–7 aknas WIND-STRESSi sisu.
    # Iga üksik plokk sisaldab põhitunnust + muutust võrreldes sama põllu eelmise tsükli sama aknaga.
    full = [
        "wind_tmax", "wind_rad", "wind_et0", "wind_dry", "wind_cool",
        "d_wind_tmax", "d_wind_rad", "d_wind_et0", "d_wind_dry", "d_wind_cool",
    ]
    return {
        # kontrollid
        "WIND only": ["wind", "d_wind"],
        "TEMP+RAD no wind": ["tday", "tmax", "rad", "cool_lowrad", "d_tday", "d_tmax", "d_rad", "d_cool_lowrad"],
        "VPD only": ["vpd", "d_vpd"],

        # üksikud tuule interaktsioonid
        "WIND×COOL": ["wind", "wind_cool", "d_wind", "d_wind_cool"],
        "WIND×TMAX": ["wind", "wind_tmax", "d_wind", "d_wind_tmax"],
        "WIND×ET0": ["wind", "wind_et0", "d_wind", "d_wind_et0"],
        "WIND×DRY": ["wind", "wind_dry", "d_wind", "d_wind_dry"],
        "WIND×RAD": ["wind", "wind_rad", "d_wind", "d_wind_rad"],
        "WIND×LOWRAD": ["wind", "wind_lowrad", "d_wind", "d_wind_lowrad"],
        "VPD×WIND": ["vpd", "wind", "vpd_wind", "d_vpd", "d_wind", "d_vpd_wind"],

        # LAB-151 originaalne täisplokk
        "FULL WIND-STRESS": full,

        # Ablation: eemalda täisplokist üks komponent korraga. Kui tulemus halveneb,
        # kandis eemaldatud komponent unikaalset infot teiste kõrval.
        "FULL minus COOL": [c for c in full if c not in {"wind_cool", "d_wind_cool"}],
        "FULL minus TMAX": [c for c in full if c not in {"wind_tmax", "d_wind_tmax"}],
        "FULL minus ET0": [c for c in full if c not in {"wind_et0", "d_wind_et0"}],
        "FULL minus DRY": [c for c in full if c not in {"wind_dry", "d_wind_dry"}],
        "FULL minus RAD": [c for c in full if c not in {"wind_rad", "d_wind_rad"}],
    }


def _temp_sensitive(name: str) -> bool:
    return True


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

st.set_page_config(page_title="KurgiMootor LAB-152", layout="wide")
st.title("LAB-152 · Mis WIND-STRESSi sees päriselt töötab?")
st.caption(
    "READ ONLY · L3–7 aken on ette lukustatud LAB-151 tulemusest · sama põllu ABC kasvukiiruse muutus · "
    "walk-forward ainult varasematel kuupäevadel · taimeindeks puudub"
)

with st.sidebar:
    st.subheader("LAB käsiread")
    st.caption("DB-sse EI kirjutata. Kui rida on DB-s olemas, kasutatakse DB rida.")
    manual_text = st.text_area(
        "YYYY-MM-DD,põld,ABC,järjekord[,intervall]",
        value="",
        height=110,
        help="Lisa ainult siis, kui värske korje pole veel DB-s. Kui DB-s olemas, jäta tühjaks.",
    )

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

st.subheader("1. Üksikute komponentide tugevus · L3–7")
st.caption(
    "Kõik read kasutavad sama ette lukustatud 3–7 päeva akent ja T-nihet 0 °C. "
    "BASE = eelmine sama põllu kasvukiirus + hooajapäev + kasvuaeg. "
    "Kasu > 0 tähendab BASE-ist paremat walk-forward tulemust."
)

# Kõigepealt näitame füsioloogiliselt loetavaid üksikuid blokke; ablation eraldi all.
single_mask = ~results["Kandidaat"].str.startswith("FULL minus", na=False)
singles = results[single_mask].copy()
show_cols = [
    "Kandidaat", "N", "Suuri langusi N", "Kasu langustel", "Kasu kokku",
    "Suuna täpsus %", "Languse tabamus %", "Valehäire %", "Saagi sMAPE %",
]
st.dataframe(
    singles[show_cols].style.format({
        "Kasu langustel": "{:+.3f}", "Kasu kokku": "{:+.3f}",
        "Suuna täpsus %": "{:.0f}", "Languse tabamus %": "{:.0f}",
        "Valehäire %": "{:.0f}", "Saagi sMAPE %": "{:.1f}",
    }, na_rep="—"),
    use_container_width=True,
    hide_index=True,
)

# Parim üksik/loetav komponent, mitte ablation-rida.
readable = singles[singles["Kandidaat"] != "FULL WIND-STRESS"].copy()
best_readable = readable.iloc[0] if not readable.empty else singles.iloc[0]
full_rows = results[results["Kandidaat"] == "FULL WIND-STRESS"]
full_row = full_rows.iloc[0] if not full_rows.empty else None

if full_row is not None:
    st.info(
        f"Täisplokk FULL WIND-STRESS: sMAPE **{float(full_row['Saagi sMAPE %']):.1f}%**, "
        f"languste tabamus **{float(full_row['Languse tabamus %']):.0f}%**, "
        f"valehäire **{float(full_row['Valehäire %']):.0f}%**. "
        f"Parim loetav üksikplokk: **{best_readable['Kandidaat']}**."
    )

st.subheader("2. Ablation · mis juhtub, kui täisplokist üks osa eemaldada?")
st.caption(
    "See on tähtsam kui ainult üksikute kandidaatide edetabel. Kui mingi komponendi eemaldamine "
    "halvendab täisplokki, kannab see teistega võrreldes unikaalset infot."
)

ab = results[results["Kandidaat"].str.startswith("FULL minus", na=False)].copy()
if full_row is not None and not ab.empty:
    full_smape = float(full_row["Saagi sMAPE %"])
    full_drop_mae = float(full_row["Langus-MAE"])
    full_overall_mae = float(full_row["log-MAE"])
    ab["sMAPE halvenemine pp"] = ab["Saagi sMAPE %"] - full_smape
    ab["Langus-MAE halvenemine"] = ab["Langus-MAE"] - full_drop_mae
    ab["Kogu MAE halvenemine"] = ab["log-MAE"] - full_overall_mae
    ab["Eemaldatud osa"] = ab["Kandidaat"].str.replace("FULL minus ", "", regex=False)
    ab = ab.sort_values(["Langus-MAE halvenemine", "sMAPE halvenemine pp"], ascending=False)
    st.dataframe(
        ab[["Eemaldatud osa", "sMAPE halvenemine pp", "Langus-MAE halvenemine", "Kogu MAE halvenemine",
            "Languse tabamus %", "Valehäire %"]].style.format({
            "sMAPE halvenemine pp": "{:+.1f}",
            "Langus-MAE halvenemine": "{:+.3f}",
            "Kogu MAE halvenemine": "{:+.3f}",
            "Languse tabamus %": "{:.0f}", "Valehäire %": "{:.0f}",
        }, na_rep="—"),
        use_container_width=True, hide_index=True,
    )
else:
    st.warning("Ablation-tabelit ei saanud arvutada.")

st.subheader("3. Parima loetava komponendi kontrollpunktid")
best_key = (str(best_readable["Kandidaat"]), "L3-7", 0.0)
df_best, base_best, cand_best = details[best_key]
event_table = _event_detail(df_best, base_best, cand_best)
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
    use_container_width=True, hide_index=True,
)

st.subheader("4. L3–7 ilmaaken sündmuste taga")
st.caption(
    "Siin on sama analüüsirea toorilm. See aitab eristada, kas signaal tähendab päriselt tuult, "
    "tuul+jahedust, tuul+ET0/kuivust või tuul+kiirgust."
)
raw_cols = [
    "date", "field", "wind", "tday", "tmax", "rad", "rh", "vpd", "et0",
    "wind_cool", "wind_et0", "wind_dry", "wind_rad", "actual_pct",
]
raw = df_best[raw_cols].copy()
raw = raw[(raw["date"] >= latest_cut) | (raw["actual_pct"] <= 100.0 * BIG_DROP)]
raw = raw.sort_values(["date", "field"], ascending=[False, True])
raw = raw.rename(columns={
    "date": "Kuupäev", "field": "Põld", "wind": "Tuul",
    "tday": "Päev T", "tmax": "Tmax", "rad": "Rad", "rh": "RH",
    "vpd": "VPD", "et0": "ET0", "wind_cool": "Tuul×jahedus",
    "wind_et0": "Tuul×ET0", "wind_dry": "Tuul×kuivus", "wind_rad": "Tuul×rad",
    "actual_pct": "Tegelik muutus %",
})
st.dataframe(
    raw.style.format({
        "Tuul": "{:.2f}", "Päev T": "{:.1f}", "Tmax": "{:.1f}", "Rad": "{:.1f}",
        "RH": "{:.0f}", "VPD": "{:.2f}", "ET0": "{:.2f}",
        "Tuul×jahedus": "{:.2f}", "Tuul×ET0": "{:.2f}", "Tuul×kuivus": "{:.1f}",
        "Tuul×rad": "{:.1f}", "Tegelik muutus %": "{:+.0f}%",
    }, na_rep="—"),
    use_container_width=True, hide_index=True,
)

st.subheader("5. Juuli saagipiduri ilm vs värske august")
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

with st.expander("Metoodika / miks see test on kitsas"):
    st.markdown(
        """
- **Aken on lukustatud L3–7.** LAB-152 ei tohi enam akent valida sama tulemuse järgi, muidu hakkaksime üle sobitama.
- **Temperatuuri nihe on lukustatud 0 °C.** LAB-151 tundlikkustest näitas, et −0,5/−1,0 °C ei muutnud järeldust sisuliselt.
- **Üksik plokk** näitab, kui hästi konkreetne mehhanism BASE-ile lisandudes töötab.
- **Ablation** on rangem test: võtame LAB-151 täis WIND-STRESS ploki ja eemaldame ühe osa korraga.
- **VPD on proxy**, arvutatud päevase T ja päevakeskmise RH põhjal, mitte põllul mõõdetud tunnine VPD.
- **Taime-/põlluindeksit ei kasutata.** See LAB otsib ainult ilmastiku seletavat osa.
- Väike andmestik tähendab, et 5 suurt langusrida ei ole 5 sõltumatut ilmastikusündmust. Tulemust tuleb kinnitada järgmiste uute korjetega.
        """
    )

st.caption(f"{LAB_VERSION} · ainult lugemine · {TODAY.isoformat()}")
