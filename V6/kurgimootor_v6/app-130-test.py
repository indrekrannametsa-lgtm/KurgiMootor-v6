from __future__ import annotations

"""
KurgiMootor LAB-141 · DYNAMIC STATE kandidaat
===============================================

Eesmärk
-------
LAB-138..140 eraldustestid näitasid kolme eri rolli:
- OWN2 = sama põllu stabiilne tootmis-state;
- MEM10 = umbes 10 päeva aeglane ilmamälu;
- BLOCK1 = väga värske kogu ploki sensor, mille kasu on tugevaim +1 päeval.

LAB-141 ei otsi enam uut mälupikkust ega uut tunnust. Testis on üks fikseeritud
DYNAMIC arhitektuur ja kolm referentsi:

1) OWN2 – puhas state;
2) MEM10-ZERO – LAB-140 stiilis 10 p ilmamälu, intercept=0;
3) MEM10-CAL – sama MEM10, kuid ainult mineviku pealt õpitud kalibratsiooni-interceptiga;
4) FLAT-MEM10+BLOCK1 – LAB-140 tasane BLOCK1 referents;
5) DYNAMIC – MEM10-CAL + BLOCK1, mille mõju hääbub lead'iga fikseeritud 1 päeva poolestusajaga.

DYNAMIC valem on mõtteliselt:
  state = OWN2
  slow = MEM10 weather correction
  fast = BLOCK1_signal * 0.5 ** (lead-1)
  ABC = state_rate * exp(calibration + slow + fast) * growth_days

BLOCK1 poolestusaeg 1 päev on arhitektuurireegel, mitte replay tulemuste järgi
lead-päevade kaupa valitud lüliti. Ühtegi lead'i eraldi championiks ei valita.

Aususe reeglid
--------------
- Replay algab 22.07.2026.
- Issue-päev D näeb ainult D lõpuks teada olnud korjeid ja mõõdetud ilma.
- D+1...D+9 tuleviku ilm tuleb D arhiveeritud ECMWF IFS jooksust.
- Tuleviku tegelikust korjereast kasutatakse prognoosi ajal ainult kuupäeva,
  põldu ja korjejärjekorda; tegelik ABC kasutatakse alles skoorimiseks.
- Sama issue +1...+9 sees prognoositud saaki EI söödetata tagasi state'iks.
- OWN2 state külmutatakse issue-päeval.
- Historical correction treenitakse ainult target_day <= issue_day ridadel.
- Skoorimisse lähevad ainult täielikud 3-põllu tegelikud korjepäevad.
- Ainult A+B+C. XL ja C/B ei osale.
- READ ONLY: DB-st ainult get_harvest_history ja get_weather_rows.

Märkus õitsemise kohta
----------------------
Täpsed põllupõhised õitsemise alguskuupäevad ei ole DB-s talletatud. LAB-141 ei
leiuta neid tagantjärele. Kui päris kuupäevad lisatakse, saab vanuse lisada hiljem
eraldi ausa challenger-tunnusena.
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import json
import math
import sys

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# FIKSEERITUD KONFIGURATSIOON
# -----------------------------------------------------------------------------

TZ = ZoneInfo("Europe/Tallinn")
TODAY = datetime.now(TZ).date()
LAB_VERSION = "LAB-141-DYNAMIC-STATE-V1"

REPLAY_START = date(2026, 7, 22)
WARM_START = date(2026, 8, 1)
WEATHER_START = date(2026, 7, 1)
DEFAULT_GROWTH_DAYS = 14.0 / 3.0
EXPECTED_FIELDS_PER_DAY = 3
BLOCK_FRESH_DAYS = 2
BLOCK_LOOKBACK_DAYS = 12

FORECAST_LAT = 58.1275
FORECAST_LON = 24.49167
FORECAST_MODEL = "ecmwf_ifs"
FORECAST_RUN_HOURS_UTC = (12, 6, 0)
FORECAST_DAYS = 10

RIDGE_ALPHA = 24.0
RECENCY_HALFLIFE_DAYS = 28.0
SMALL_SAMPLE_PRIOR_ROWS = 12.0
MIN_TRAIN_ROWS = 6
TARGET_EPS = 0.25

MEASURED_REQUIRED = (
    "temp_night_avg_c", "temp_day_avg_c", "temp_min_c", "temp_max_c",
    "wind_avg_ms", "radiation_mj_m2", "humidity_avg_pct",
    "precipitation_mm", "et0_mm",
)

# Enne replay'd lukustatud arhitektuur. Uusi aknaid ega lead-champione ei valita.
MEM10_FEATURES = ["w10_gdd10", "w10_rad", "w10_et0"]
MEM10_BLOCK_FEATURES = MEM10_FEATURES + ["block1_signal"]
BLOCK_HALF_LIFE_DAYS = 1.0
MODEL_NAMES = ["OWN2", "MEM10-ZERO", "MEM10-CAL", "FLAT-MEM10+BLOCK1", "DYNAMIC"]
MEMORY_WINDOWS = (10,)


# -----------------------------------------------------------------------------
# ÜLDABID
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


def _abc_from_row(row: dict) -> Optional[float]:
    vals = [_f(row.get(k)) for k in ("a", "b", "c")]
    if any(v is None for v in vals):
        return None
    abc = float(sum(vals))
    return abc if abc > 0 else None


def _quality_reliable(row: dict) -> bool:
    q = str(row.get("data_quality") or row.get("quality") or "").strip().lower()
    return q not in {"hinnanguline", "ligikaudne", "estimated", "approximate"}


@dataclass
class Event:
    day: date
    field: int
    order: int
    abc: Optional[float]
    reliable: bool
    source: str  # actual / estimated_event / simulated_schedule


def _event_sort_key(e: Event) -> Tuple[date, int, int]:
    return (e.day, e.order, e.field)


def _copy_events(events: Iterable[Event]) -> List[Event]:
    return [Event(e.day, e.field, e.order, e.abc, e.reliable, e.source) for e in events]


# -----------------------------------------------------------------------------
# KORJE / OWN2 STATE
# -----------------------------------------------------------------------------

def _prepare_events(harvest_rows_raw: List[dict]) -> Tuple[List[Event], Dict[Tuple[date, int], dict]]:
    events: List[Event] = []
    actual_lookup: Dict[Tuple[date, int], dict] = {}
    for raw in harvest_rows_raw:
        dd = _d(raw.get("harvest_date"))
        try:
            field = int(raw.get("field_no"))
        except Exception:
            continue
        if dd is None or not (1 <= field <= 14):
            continue
        try:
            order = int(raw.get("harvest_order") or 1)
        except Exception:
            order = 1
        reliable = _quality_reliable(raw)
        abc = _abc_from_row(raw) if reliable else None
        source = "actual" if abc is not None else "estimated_event"
        events.append(Event(dd, field, order, abc, abc is not None, source))
        if abc is not None:
            rr = dict(raw)
            rr["_abc"] = float(abc)
            rr["_day"] = dd
            rr["_field"] = field
            rr["_order"] = order
            actual_lookup[(dd, field)] = rr
    events.sort(key=_event_sort_key)
    return events, actual_lookup


def _field_events_before(events: List[Event], field: int, cutoff_exclusive: date) -> List[Event]:
    return sorted(
        [e for e in events if e.field == field and e.day < cutoff_exclusive],
        key=_event_sort_key,
    )


def _growth_between(prev: Event, cur_day: date, cur_order: int) -> float:
    g = float((cur_day - prev.day).days) + (cur_order - prev.order) * (3.0 / 24.0)
    return max(0.5, g)


def _growth_for_target(schedule_events: List[Event], field: int, target_day: date, target_order: int) -> Tuple[float, Optional[Event]]:
    hist = _field_events_before(schedule_events, field, target_day)
    if hist:
        prev = hist[-1]
        return _growth_between(prev, target_day, target_order), prev
    return DEFAULT_GROWTH_DAYS, None


def _rate_history(events: List[Event], field: int, cutoff_exclusive: date) -> List[Tuple[Event, float]]:
    """Rate = ABC / täpne kasvuaeg eelmisest korjesündmusest."""
    hist = _field_events_before(events, field, cutoff_exclusive)
    out: List[Tuple[Event, float]] = []
    for i in range(1, len(hist)):
        prev, cur = hist[i - 1], hist[i]
        if not (cur.reliable and cur.abc is not None and cur.abc > 0):
            continue
        growth = _growth_between(prev, cur.day, cur.order)
        rate = float(cur.abc) / growth
        if math.isfinite(rate) and rate > 0:
            out.append((cur, rate))
    return out


def _complete_day_rate_medians(known_events: List[Event], cutoff_exclusive: date) -> List[Tuple[date, float]]:
    """Täielikud 3-põllu actual-päevad enne cutoff'i ja nende rate-mediaan."""
    rates_by_day: Dict[date, List[float]] = {}
    fields_by_day: Dict[date, set] = {}
    for field in range(1, 15):
        for ev, rate in _rate_history(known_events, field, cutoff_exclusive):
            if ev.source != "actual" or ev.day >= cutoff_exclusive:
                continue
            rates_by_day.setdefault(ev.day, []).append(float(rate))
            fields_by_day.setdefault(ev.day, set()).add(field)
    rows: List[Tuple[date, float]] = []
    for dd in sorted(rates_by_day):
        if len(fields_by_day.get(dd, set())) == EXPECTED_FIELDS_PER_DAY and len(rates_by_day[dd]) == EXPECTED_FIELDS_PER_DAY:
            rows.append((dd, float(np.median(rates_by_day[dd]))))
    return rows


def _block3_fallback(known_events: List[Event], cutoff_exclusive: date) -> Optional[float]:
    meds = _complete_day_rate_medians(known_events, cutoff_exclusive)
    if meds:
        last3 = meds[-3:]
        weights = np.asarray([1.0, 2.0, 4.0], dtype=float)[-len(last3):]
        weights /= weights.sum()
        return float(sum(w * v for w, (_dd, v) in zip(weights, last3)))

    vals: List[float] = []
    reference_day = cutoff_exclusive - timedelta(days=1)
    for field in range(1, 15):
        rh = _rate_history(known_events, field, cutoff_exclusive)
        if not rh:
            continue
        ev, rate = rh[-1]
        if 0 <= (reference_day - ev.day).days <= BLOCK_LOOKBACK_DAYS:
            vals.append(float(rate))
    return float(np.median(vals)) if vals else None


def _own2_snapshot(known_events: List[Event], field: int, cutoff_exclusive: date) -> Optional[Tuple[float, bool]]:
    """OWN2 = viimase kuni 2 rate'i mediaan; vana rate ei tohi ületada hilisemat ABC-ta korjet."""
    hist_events = _field_events_before(known_events, field, cutoff_exclusive)
    rh = _rate_history(known_events, field, cutoff_exclusive)
    own: List[float] = []
    if hist_events and rh:
        latest_event = hist_events[-1]
        latest_rate_event = rh[-1][0]
        if (latest_event.day, latest_event.order) == (latest_rate_event.day, latest_rate_event.order):
            own = [float(rate) for _ev, rate in rh]
    if own:
        return float(np.median(own[-2:])), False
    fb = _block3_fallback(known_events, cutoff_exclusive)
    if fb is None or fb <= 0:
        return None
    return float(fb), True


def _block1_signal(known_events: List[Event], cutoff_exclusive: date, own2_rate: float) -> Tuple[float, Optional[date]]:
    """Värske BLOCK1 relative signal. Üle 2 päeva vanune signaal nullitakse fikseeritult."""
    meds = _complete_day_rate_medians(known_events, cutoff_exclusive)
    if not meds or own2_rate <= 0:
        return 0.0, None
    dd, block_rate = meds[-1]
    reference_day = cutoff_exclusive - timedelta(days=1)
    age = (reference_day - dd).days
    if age < 0 or age > BLOCK_FRESH_DAYS:
        return 0.0, dd
    sig = math.log(max(1e-6, block_rate) / max(1e-6, own2_rate))
    return float(sig), dd


def _complete_actual_days(actual_lookup: Dict[Tuple[date, int], dict]) -> set:
    by_day: Dict[date, set] = {}
    for dd, field in actual_lookup:
        by_day.setdefault(dd, set()).add(field)
    return {dd for dd, fields in by_day.items() if len(fields) == EXPECTED_FIELDS_PER_DAY}


# -----------------------------------------------------------------------------
# ILM: DB mõõdetud + arhiveeritud ECMWF issue-jooks
# -----------------------------------------------------------------------------

def _measured_weather_map(rows: List[dict]) -> Dict[date, dict]:
    out: Dict[date, dict] = {}
    for r in rows:
        dd = _d(r.get("weather_date"))
        if dd is None or dd < WEATHER_START:
            continue
        if str(r.get("data_kind") or "").lower() != "measured" or not bool(r.get("checked")):
            continue
        if any(_f(r.get(c)) is None for c in MEASURED_REQUIRED):
            continue
        night = float(r["temp_night_avg_c"])
        dayt = float(r["temp_day_avg_c"])
        out[dd] = {
            "day": dd,
            "temp": 0.5 * (night + dayt),
            "temp_min": float(r["temp_min_c"]),
            "temp_max": float(r["temp_max_c"]),
            "wind": float(r["wind_avg_ms"]),
            "rad": float(r["radiation_mj_m2"]),
            "rh": float(r["humidity_avg_pct"]),
            "rain": float(r["precipitation_mm"]),
            "et0": float(r["et0_mm"]),
            "source": "measured",
        }
    return out


def _openmeteo_url(issue_day: date, run_hour_utc: int) -> str:
    hourly = ",".join([
        "temperature_2m", "relative_humidity_2m", "precipitation",
        "et0_fao_evapotranspiration", "wind_speed_10m", "shortwave_radiation",
    ])
    params = {
        "latitude": f"{FORECAST_LAT:.5f}",
        "longitude": f"{FORECAST_LON:.5f}",
        "run": f"{issue_day.isoformat()}T{run_hour_utc:02d}:00",
        "models": FORECAST_MODEL,
        "hourly": hourly,
        "wind_speed_unit": "ms",
        "timezone": "Europe/Tallinn",
        "forecast_days": str(FORECAST_DAYS),
    }
    return "https://single-runs-api.open-meteo.com/v1/forecast?" + urlencode(params)


def _fetch_json(url: str, timeout: int = 45) -> dict:
    req = Request(url, headers={"User-Agent": f"KurgiMootor-{LAB_VERSION}"})
    with urlopen(req, timeout=timeout) as response:
        payload = response.read().decode("utf-8")
    data = json.loads(payload)
    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(str(data.get("reason") or data))
    return data


def _hourly_to_daily(payload: dict) -> Dict[date, dict]:
    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or []
    keys = [
        "temperature_2m", "relative_humidity_2m", "precipitation",
        "et0_fao_evapotranspiration", "wind_speed_10m", "shortwave_radiation",
    ]
    arrays = {k: hourly.get(k) or [] for k in keys}
    if not times or any(len(arrays[k]) != len(times) for k in keys):
        raise RuntimeError("ECMWF hourly vastus on puudulik või veergude pikkused ei klapi.")

    bucket: Dict[date, List[dict]] = {}
    for i, ts in enumerate(times):
        try:
            dt = datetime.fromisoformat(str(ts))
        except Exception:
            continue
        vals = {k: _f(arrays[k][i]) for k in keys}
        if any(v is None for v in vals.values()):
            continue
        bucket.setdefault(dt.date(), []).append({"hour": dt.hour, **vals})

    out: Dict[date, dict] = {}
    for dd, rows in bucket.items():
        if len(rows) < 18:
            continue
        temps = np.asarray([r["temperature_2m"] for r in rows], dtype=float)
        rhs = np.asarray([r["relative_humidity_2m"] for r in rows], dtype=float)
        rain = np.asarray([r["precipitation"] for r in rows], dtype=float)
        et0 = np.asarray([r["et0_fao_evapotranspiration"] for r in rows], dtype=float)
        wind = np.asarray([r["wind_speed_10m"] for r in rows], dtype=float)
        rad = np.asarray([r["shortwave_radiation"] for r in rows], dtype=float)
        night_vals = [r["temperature_2m"] for r in rows if r["hour"] < 8 or r["hour"] >= 20]
        day_vals = [r["temperature_2m"] for r in rows if 8 <= r["hour"] < 20]
        if not night_vals or not day_vals:
            continue
        out[dd] = {
            "day": dd,
            "temp": 0.5 * (float(np.mean(night_vals)) + float(np.mean(day_vals))),
            "temp_min": float(temps.min()),
            "temp_max": float(temps.max()),
            "wind": float(wind.mean()),
            "rad": float(rad.sum() * 0.0036),  # W/m2-hour -> MJ/m2/day
            "rh": float(rhs.mean()),
            "rain": float(rain.sum()),
            "et0": float(et0.sum()),
            "source": "ecmwf_forecast",
        }
    return out


def _fetch_ecmwf_issue(issue_day: date) -> Tuple[Dict[date, dict], str]:
    errors = []
    for hour in FORECAST_RUN_HOURS_UTC:
        try:
            daily = _hourly_to_daily(_fetch_json(_openmeteo_url(issue_day, hour)))
            required = [issue_day + timedelta(days=i) for i in range(1, 9)]
            if all(d in daily for d in required):
                return daily, f"{issue_day.isoformat()} {hour:02d} UTC"
            errors.append(f"{hour:02d}Z: puudulik horisont ({len(daily)} päeva)")
        except Exception as exc:
            errors.append(f"{hour:02d}Z: {exc}")
    raise RuntimeError("; ".join(errors))


def _combined_weather(measured: Dict[date, dict], forecast: Dict[date, dict], issue_day: date) -> Dict[date, dict]:
    out = {d: r for d, r in measured.items() if d <= issue_day}
    for d, r in forecast.items():
        if d > issue_day:
            out[d] = r
    return out


def _range_weather(wmap: Dict[date, dict], start_day: date, end_day: date) -> Optional[List[dict]]:
    if start_day > end_day:
        return []
    rows: List[dict] = []
    cur = start_day
    while cur <= end_day:
        row = wmap.get(cur)
        if row is None:
            return None
        rows.append(row)
        cur += timedelta(days=1)
    return rows


def _agg_weather(rows: Optional[List[dict]]) -> Optional[dict]:
    if not rows:
        return None
    temp = np.asarray([r["temp"] for r in rows], dtype=float)
    rad = np.asarray([r["rad"] for r in rows], dtype=float)
    et0 = np.asarray([r["et0"] for r in rows], dtype=float)
    gdd10 = np.maximum(0.0, temp - 10.0)
    return {
        "gdd10": float(gdd10.mean()),
        "rad": float(rad.mean()),
        "et0": float(et0.mean()),
        "n": len(rows),
    }


def _memory_features(
    schedule_events: List[Event],
    field: int,
    target_day: date,
    target_order: int,
    wmap: Dict[date, dict],
) -> dict:
    """Kasvuaeg + fikseeritud 10 päeva ilmamälu.

    MEM10 kasutab ainult targetile eelnevat 10 päeva; puuduv ilmaaken jääb None.
    """
    growth, _prev = _growth_for_target(schedule_events, field, target_day, target_order)
    out = {"growth_days": float(growth)}
    for days in MEMORY_WINDOWS:
        rows = _range_weather(
            wmap,
            target_day - timedelta(days=days),
            target_day - timedelta(days=1),
        )
        mem = _agg_weather(rows) if rows is not None else None
        prefix = f"w{days}"
        if mem is None:
            out[f"{prefix}_gdd10"] = None
            out[f"{prefix}_rad"] = None
            out[f"{prefix}_et0"] = None
        else:
            out[f"{prefix}_gdd10"] = float(mem["gdd10"])
            out[f"{prefix}_rad"] = float(mem["rad"])
            out[f"{prefix}_et0"] = float(mem["et0"])
    return out

# -----------------------------------------------------------------------------
# HISTORICAL TRANSITION RECORDID
# -----------------------------------------------------------------------------

def _build_record(
    *,
    known_events: List[Event],
    schedule_events: List[Event],
    field: int,
    target_day: date,
    target_order: int,
    wmap: Dict[date, dict],
    actual_abc: Optional[float],
    frozen_own2: Optional[Tuple[float, bool]] = None,
    block_cutoff: Optional[date] = None,
) -> Optional[dict]:
    cutoff = target_day if frozen_own2 is None else None
    if frozen_own2 is None:
        own2 = _own2_snapshot(known_events, field, cutoff)
    else:
        own2 = frozen_own2
    if own2 is None:
        return None
    own2_rate, used_fallback = own2
    if own2_rate <= 0:
        return None

    wf = _memory_features(schedule_events, field, target_day, target_order, wmap)

    if block_cutoff is None:
        # Historical target: info strictly before target_day.
        block_cutoff = target_day
    block_sig, block_day = _block1_signal(known_events, block_cutoff, own2_rate)

    rec = {
        "target_day": target_day,
        "field_no": field,
        "own2_rate": float(own2_rate),
        "used_fallback": int(used_fallback),
        "block1_signal": float(block_sig),
        "block1_day": block_day,
        **wf,
        "actual_abc": float(actual_abc) if actual_abc is not None else None,
    }
    if actual_abc is not None and actual_abc > 0:
        actual_rate = float(actual_abc) / max(0.5, float(wf["growth_days"]))
        rec["actual_rate"] = actual_rate
        rec["y"] = math.log(max(1e-6, actual_rate) / max(1e-6, own2_rate))
    else:
        rec["actual_rate"] = None
        rec["y"] = None
    return rec


def _make_historical_records(
    events: List[Event],
    actual_lookup: Dict[Tuple[date, int], dict],
    measured: Dict[date, dict],
) -> List[dict]:
    records: List[dict] = []
    for (dd, field), row in sorted(actual_lookup.items()):
        # Kõik future target saagid eemaldatud; info ainult enne target päeva.
        # Mälupikkuste puuduvad väärtused jäävad rea sees None'iks ja iga variant
        # filtreerib ainult enda vajaliku akna järgi.
        prior = [e for e in events if e.day < dd]
        # Growth vajab eelmist event'i. Kui pole, pole transition-rida.
        if not _field_events_before(prior, field, dd):
            continue
        rec = _build_record(
            known_events=prior,
            schedule_events=prior,
            field=field,
            target_day=dd,
            target_order=int(row.get("_order") or 1),
            wmap=measured,
            actual_abc=float(row["_abc"]),
            frozen_own2=None,
            block_cutoff=dd,
        )
        if rec is not None:
            records.append(rec)
    records.sort(key=lambda r: (r["target_day"], r["field_no"]))
    return records


# -----------------------------------------------------------------------------
# CORRECTION RIDGE + FIKSEERITUD DYNAMIC ARHITEKTUUR
# -----------------------------------------------------------------------------

def _fit_correction(
    train_records: List[dict],
    cutoff_day: date,
    feature_names: List[str],
    *,
    allow_intercept: bool,
) -> Optional[dict]:
    clean = []
    for r in train_records:
        y = _f(r.get("y"))
        vals = [_f(r.get(c)) for c in feature_names]
        if y is None or any(v is None for v in vals):
            continue
        age = max(0.0, float((cutoff_day - r["target_day"]).days))
        w = 0.5 ** (age / RECENCY_HALFLIFE_DAYS)
        clean.append((vals, y, w))
    if len(clean) < MIN_TRAIN_ROWS:
        return None

    X = np.asarray([x for x, _y, _w in clean], dtype=float)
    y = np.asarray([yy for _x, yy, _w in clean], dtype=float)
    w = np.asarray([ww for _x, _y, ww in clean], dtype=float)
    sw = float(w.sum())
    if sw <= 0:
        return None

    mu = (X * w[:, None]).sum(axis=0) / sw
    var = ((X - mu) ** 2 * w[:, None]).sum(axis=0) / sw
    sd = np.sqrt(np.maximum(var, 1e-8))
    Z = (X - mu) / sd

    if allow_intercept:
        A = np.column_stack([np.ones(len(Z), dtype=float), Z])
        penalty = np.eye(A.shape[1], dtype=float)
        penalty[0, 0] = 0.0  # kalibratsiooni-intercept on lubatud, kuid allpool shrinkitakse koos yhat'iga
    else:
        A = Z
        penalty = np.eye(A.shape[1], dtype=float)

    ws = np.sqrt(w)[:, None]
    Aw = A * ws
    yw = y * ws[:, 0]
    lhs = Aw.T @ Aw + RIDGE_ALPHA * penalty
    rhs = Aw.T @ yw
    try:
        beta = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(lhs) @ rhs

    bounds = None
    if len(y) >= 10:
        lo, hi = np.quantile(y, [0.02, 0.98])
        pad = max(0.05, 0.12 * float(hi - lo))
        bounds = (float(lo - pad), float(hi + pad))

    return {
        "mu": mu, "sd": sd, "beta": beta, "bounds": bounds,
        "n": len(clean), "allow_intercept": bool(allow_intercept),
    }


def _predict_correction(model: Optional[dict], rec: dict, feature_names: List[str]) -> Optional[float]:
    if model is None:
        return None
    vals = [_f(rec.get(c)) for c in feature_names]
    if any(v is None for v in vals):
        return None
    x = np.asarray(vals, dtype=float)
    z = np.clip((x - model["mu"]) / model["sd"], -3.0, 3.0)
    beta = model["beta"]
    if model.get("allow_intercept"):
        yhat = float(beta[0] + z @ beta[1:])
    else:
        yhat = float(z @ beta)
    if model.get("bounds") is not None:
        lo, hi = model["bounds"]
        yhat = float(np.clip(yhat, lo, hi))
    n_train = float(model.get("n") or 0.0)
    yhat *= n_train / (n_train + SMALL_SAMPLE_PRIOR_ROWS)
    return yhat


def _abc_from_yhat(rec: dict, yhat: float) -> float:
    own2_rate = max(1e-6, float(rec["own2_rate"]))
    growth = max(0.5, float(rec["growth_days"]))
    pred_rate = own2_rate * math.exp(float(yhat))
    return float(max(0.0, pred_rate * growth))


def _own2_pred(rec: dict) -> float:
    return float(max(0.0, float(rec["own2_rate"]) * max(0.5, float(rec["growth_days"]))))


def _block_decay_weight(lead: int) -> float:
    lead = max(1, int(lead))
    return float(0.5 ** ((lead - 1) / BLOCK_HALF_LIFE_DAYS))


def _fit_issue_models(train_records: List[dict], cutoff_day: date) -> dict:
    return {
        "MEM10-ZERO": _fit_correction(train_records, cutoff_day, MEM10_FEATURES, allow_intercept=False),
        "MEM10-CAL": _fit_correction(train_records, cutoff_day, MEM10_FEATURES, allow_intercept=True),
        "FLAT-MEM10+BLOCK1": _fit_correction(train_records, cutoff_day, MEM10_BLOCK_FEATURES, allow_intercept=False),
        "DYNAMIC": _fit_correction(train_records, cutoff_day, MEM10_BLOCK_FEATURES, allow_intercept=True),
    }


def _predict_models(models: dict, rec: dict, lead: int) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {"OWN2": _own2_pred(rec)}

    y0 = _predict_correction(models.get("MEM10-ZERO"), rec, MEM10_FEATURES)
    out["MEM10-ZERO"] = None if y0 is None else _abc_from_yhat(rec, y0)

    yc = _predict_correction(models.get("MEM10-CAL"), rec, MEM10_FEATURES)
    out["MEM10-CAL"] = None if yc is None else _abc_from_yhat(rec, yc)

    yf = _predict_correction(models.get("FLAT-MEM10+BLOCK1"), rec, MEM10_BLOCK_FEATURES)
    out["FLAT-MEM10+BLOCK1"] = None if yf is None else _abc_from_yhat(rec, yf)

    # DYNAMIC: sama +1-le õpitud BLOCK1 sensor, kuid issue'ist kaugemale minnes
    # hääbub selle sisend fikseeritud poolestusajaga. MEM10 ise ei hääbu, sest
    # targeti 10p aken täitub lead'i kasvades järjest tuleviku ECMWF ilmaga.
    dyn_rec = dict(rec)
    dyn_rec["block1_signal"] = float(rec.get("block1_signal") or 0.0) * _block_decay_weight(lead)
    yd = _predict_correction(models.get("DYNAMIC"), dyn_rec, MEM10_BLOCK_FEATURES)
    out["DYNAMIC"] = None if yd is None else _abc_from_yhat(rec, yd)
    return out


# -----------------------------------------------------------------------------
# REPLAY
# -----------------------------------------------------------------------------

def _issue_dates(last_actual_day: date) -> List[date]:
    end = min(last_actual_day - timedelta(days=1), TODAY - timedelta(days=1))
    if end < REPLAY_START:
        return []
    return [REPLAY_START + timedelta(days=i) for i in range((end - REPLAY_START).days + 1)]


def _future_schedule_stubs(actual_lookup: Dict[Tuple[date, int], dict], issue_day: date) -> List[Tuple[date, int, int]]:
    out = []
    max_day = issue_day + timedelta(days=9)
    for (dd, field), row in actual_lookup.items():
        if issue_day < dd <= max_day:
            out.append((dd, field, int(row.get("_order") or 1)))
    out.sort(key=lambda x: (x[0], x[2], x[1]))
    return out


def _predict_issue(
    *,
    issue_day: date,
    all_events: List[Event],
    actual_lookup: Dict[Tuple[date, int], dict],
    measured: Dict[date, dict],
    historical_records: List[dict],
    forecast_map: Dict[date, dict],
    run_label: str,
) -> List[dict]:
    known_events = [e for e in all_events if e.day <= issue_day]
    schedule_events = _copy_events(known_events)
    wmap = _combined_weather(measured, forecast_map, issue_day)
    train = [r for r in historical_records if r["target_day"] <= issue_day]
    models = _fit_issue_models(train, issue_day)
    stubs = _future_schedule_stubs(actual_lookup, issue_day)

    frozen_own2: Dict[int, Tuple[float, bool]] = {}
    out: List[dict] = []
    for target_day, field, order in stubs:
        lead = int((target_day - issue_day).days)
        if field not in frozen_own2:
            own = _own2_snapshot(known_events, field, issue_day + timedelta(days=1))
            if own is None:
                continue
            frozen_own2[field] = own

        rec = _build_record(
            known_events=known_events,
            schedule_events=schedule_events,
            field=field,
            target_day=target_day,
            target_order=order,
            wmap=wmap,
            actual_abc=None,
            frozen_own2=frozen_own2[field],
            block_cutoff=issue_day + timedelta(days=1),
        )
        if rec is None:
            continue

        actual_row = actual_lookup.get((target_day, field))
        actual_abc = float(actual_row["_abc"]) if actual_row is not None else np.nan
        preds = _predict_models(models, rec, lead)
        row = {
            "issue_day": issue_day,
            "target_day": target_day,
            "lead": lead,
            "field": field,
            "order": order,
            "growth_days": float(rec["growth_days"]),
            "actual_abc": actual_abc,
            "fallback": int(rec["used_fallback"]),
            "block1_signal": float(rec["block1_signal"]),
            "block_weight": _block_decay_weight(lead),
            "run": run_label,
            "w10_gdd10": float(rec["w10_gdd10"]) if _f(rec.get("w10_gdd10")) is not None else np.nan,
            "w10_rad": float(rec["w10_rad"]) if _f(rec.get("w10_rad")) is not None else np.nan,
            "w10_et0": float(rec["w10_et0"]) if _f(rec.get("w10_et0")) is not None else np.nan,
        }
        for name in MODEL_NAMES:
            pred = preds.get(name)
            row[name] = np.nan if pred is None else float(pred)
        for key, model in models.items():
            row[f"{key}_train_n"] = int(model["n"]) if model is not None else 0
        out.append(row)

        # Ainult ajamärk; prognoositud ABC/state ei lähe järgmise ringi sensoriks.
        schedule_events.append(Event(target_day, field, order, None, False, "simulated_schedule"))
        schedule_events.sort(key=_event_sort_key)
    return out


def _daily_scores(field_df: pd.DataFrame, complete_days: set) -> pd.DataFrame:
    if field_df.empty:
        return pd.DataFrame()
    f = field_df[field_df["target_day"].isin(complete_days)].copy()
    if f.empty:
        return pd.DataFrame()
    agg = {
        "actual_abc": ("actual_abc", "sum"),
        "n_fields": ("field", "count"),
        "fallback_fields": ("fallback", "sum"),
        "block_weight": ("block_weight", "mean"),
    }
    for name in MODEL_NAMES:
        agg[name] = (name, "sum")
    daily = f.groupby(["issue_day", "target_day", "lead"], as_index=False).agg(**agg)
    daily = daily[daily["n_fields"] == EXPECTED_FIELDS_PER_DAY].copy()
    if daily.empty:
        return daily
    daily["fallback_pct"] = 100.0 * daily["fallback_fields"] / EXPECTED_FIELDS_PER_DAY
    for name in MODEL_NAMES:
        daily[f"{name}_err"] = daily[name] - daily["actual_abc"]
        daily[f"{name}_ape"] = daily[f"{name}_err"].abs() / daily["actual_abc"].clip(lower=0.1)
        daily[f"{name}_bias"] = 100.0 * daily[f"{name}_err"] / daily["actual_abc"].clip(lower=0.1)
    return daily


def _summary_table(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in MODEL_NAMES:
        valid = daily[np.isfinite(daily[name]) & np.isfinite(daily[f"{name}_ape"])].copy()
        if valid.empty:
            continue
        rows.append({
            "Mudel": name,
            "MAPE %": 100.0 * float(valid[f"{name}_ape"].mean()),
            "±20% sees %": 100.0 * float((valid[f"{name}_ape"] <= 0.20).mean()),
            "Bias %": float(valid[f"{name}_bias"].mean()),
            "MAE ABC": float(valid[f"{name}_err"].abs().mean()),
            "N": int(len(valid)),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    base = float(df.loc[df["Mudel"] == "OWN2", "MAPE %"].iloc[0])
    df["Δ MAPE vs OWN2"] = df["MAPE %"] - base
    return df


def _lead_table(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lead, g in daily.groupby("lead"):
        row = {
            "Lead": int(lead), "N": int(len(g)),
            "Fallback %": float(g["fallback_pct"].mean()),
            "BLOCK kaal": float(g["block_weight"].mean()),
        }
        for name in MODEL_NAMES:
            valid = g[np.isfinite(g[f"{name}_ape"])]
            row[f"{name} MAPE %"] = 100.0 * float(valid[f"{name}_ape"].mean()) if not valid.empty else np.nan
        finite = {name: row[f"{name} MAPE %"] for name in MODEL_NAMES if math.isfinite(float(row[f"{name} MAPE %"]))}
        if finite:
            best_name = min(finite, key=finite.get)
            row["Võitja"] = best_name
            row["Võitja MAPE %"] = finite[best_name]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("Lead")


def _window_summary(daily: pd.DataFrame, start_day: date) -> pd.DataFrame:
    g = daily[daily["target_day"] >= start_day].copy()
    return _summary_table(g) if not g.empty else pd.DataFrame()


# -----------------------------------------------------------------------------
# SELF TEST
# -----------------------------------------------------------------------------

def _self_test() -> None:
    measured: Dict[date, dict] = {}
    start = date(2026, 7, 1)
    for i in range(80):
        dd = start + timedelta(days=i)
        measured[dd] = {
            "day": dd, "temp": 18.0 + 0.025*i, "temp_min": 12.0, "temp_max": 25.0,
            "wind": 2.0, "rad": 17.0 + 0.07*i, "rh": 75.0, "rain": 0.4,
            "et0": 2.8 + 0.008*i, "source": "measured",
        }

    rows = []
    days = [date(2026,7,5) + timedelta(days=4*j) for j in range(12)]
    for j, dd in enumerate(days):
        for order, f in enumerate((1, 2, 3), start=1):
            abc = 6.5 + 0.32*j + 0.12*f
            rows.append({
                "harvest_date": dd.isoformat(), "field_no": f, "harvest_order": order,
                "a": 0.2, "b": abc*0.45, "c": abc*0.55-0.2, "xl": 1.0,
                "data_quality": "Kinnitatud",
            })
    events, lookup = _prepare_events(rows)
    hist = _make_historical_records(events, lookup, measured)
    assert hist and all(_f(r.get("w10_rad")) is not None for r in hist if r["target_day"] >= date(2026,7,15))

    cutoff = hist[-1]["target_day"]
    train = [r for r in hist if r["target_day"] < cutoff]
    models = _fit_issue_models(train, cutoff)
    rec0 = hist[-1]
    preds = _predict_models(models, rec0, 1)
    assert set(preds) == set(MODEL_NAMES)
    assert all(v is not None and math.isfinite(float(v)) and float(v) >= 0 for v in preds.values())
    assert abs(_block_decay_weight(1) - 1.0) < 1e-12
    assert abs(_block_decay_weight(2) - 0.5) < 1e-12
    assert abs(_block_decay_weight(3) - 0.25) < 1e-12

    issue = date(2026, 8, 2)
    fc = {d: dict(v, source="ecmwf_forecast") for d, v in measured.items() if d > issue}
    out = _predict_issue(
        issue_day=issue, all_events=events, actual_lookup=lookup, measured=measured,
        historical_records=hist, forecast_map=fc, run_label="synthetic",
    )
    assert out and all(all(name in r for name in MODEL_NAMES) for r in out)
    assert all(0.0 < float(r["block_weight"]) <= 1.0 for r in out)
    print("LAB-141 SELF-TEST OK")


# -----------------------------------------------------------------------------
# STREAMLIT
# -----------------------------------------------------------------------------

def main() -> None:
    import streamlit as st
    import db

    st.set_page_config(page_title="KurgiMootor LAB-141", layout="wide")
    st.error("🧪 LAB-141 · DYNAMIC STATE · READ-ONLY · 22.07 → +1…+9")
    st.title("KurgiMootor · OWN2 + MEM10 + hääbuv BLOCK1")
    st.caption(
        "Üks fikseeritud kandidaat: OWN2 põhi-state + 10 päeva ilmamälu + kiire BLOCK1 sensor, "
        "mille kaal poolitub iga lead-päevaga. Kõrval on ainult referentsid; lead'i järgi championit ei valita."
    )

    with st.expander("Arhitektuur ja aususe reeglid", expanded=False):
        st.markdown(
            f"""
- **OWN2:** sama põllu kuni 2 viimase ABC/kasvupäev rate'i mediaan; cold-start = BLOCK3 fallback.
- **MEM10:** targetile eelneva 10 päeva GDD10 + radiatsiooni + ET0 päevakeskmised.
- **Kalibratsioon:** MEM10-CAL ja DYNAMIC võivad õppida ainult mineviku pealt intercepti; small-sample shrink jääb alles.
- **BLOCK1:** viimase täieliku korjepäeva värske (<=2 p) suhte-signaal OWN2 vastu.
- **DYNAMIC BLOCK kaal:** `0.5 ** (lead-1)`; +1={_block_decay_weight(1):.2f}, +2={_block_decay_weight(2):.2f}, +3={_block_decay_weight(3):.2f}, +4={_block_decay_weight(4):.3f}. See reegel on enne replay'd fikseeritud.
- Tuleviku ilm = arhiveeritud ECMWF issue-jooks; issue-päevani mõõdetud ilm.
- Prognoositud ABC-d ei kasutata sama +1…+9 akna järgmise state'ina.
- Skoor ainult täielikel 3-põllu päevadel ja ainult **A+B+C**.
- Täpseid õitsemise alguskuupäevi DB-s pole; LAB ei leiuta neid tagantjärele.
            """
        )

    @st.cache_data(ttl=120, show_spinner=False)
    def _load_db_data():
        return (
            db.get_harvest_history(limit=5000),
            db.get_weather_rows(WEATHER_START, TODAY),
        )

    @st.cache_data(ttl=86400, show_spinner=False)
    def _cached_issue_weather(issue_iso: str):
        return _fetch_ecmwf_issue(date.fromisoformat(issue_iso))

    if st.button("Värskenda DB andmed", type="secondary"):
        _load_db_data.clear()
        st.session_state.pop("lab141_results", None)
        st.rerun()

    try:
        harvest_rows, weather_rows = _load_db_data()
    except Exception as exc:
        st.error(f"DB lugemine ebaõnnestus: {exc}")
        st.stop()

    events, actual_lookup = _prepare_events(harvest_rows)
    if not actual_lookup:
        st.error("Usaldusväärseid ABC ridu ei leitud.")
        st.stop()
    complete_days = _complete_actual_days(actual_lookup)
    measured = _measured_weather_map(weather_rows)
    last_actual_day = max(dd for dd, _field in actual_lookup)
    issue_days = _issue_dates(last_actual_day)
    if not issue_days:
        st.error("22.07 järel pole replay jaoks piisavalt päevi.")
        st.stop()

    required_end = min(last_actual_day, TODAY - timedelta(days=1))
    missing = []
    cur = WEATHER_START
    while cur <= required_end:
        if cur not in measured:
            missing.append(cur)
        cur += timedelta(days=1)
    if missing:
        st.error("Mõõdetud ilma ajalugu pole täielik. Puudu: " + ", ".join(d.strftime("%d.%m") for d in missing[:20]))
        st.stop()

    historical_records = _make_historical_records(events, actual_lookup, measured)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Replay algus", REPLAY_START.strftime("%d.%m.%Y"))
    c2.metric("Usaldusväärseid ABC ridu", str(len(actual_lookup)))
    c3.metric("Historical training ridu", str(len(historical_records)))
    c4.metric("Issue-päevi", str(len(issue_days)))

    run_requested = st.button("▶ Jooksuta DYNAMIC STATE replay", type="primary")
    if not run_requested and "lab141_results" not in st.session_state:
        st.info("Vajuta üks kord. ECMWF issue-jooksud on 24 h cache'is; productionit ei puuduta.")
        st.stop()

    if run_requested:
        all_rows: List[dict] = []
        errors: List[Tuple[date, str]] = []
        runs: Dict[date, str] = {}
        progress = st.progress(0.0, text="ECMWF issue-jooksud + DYNAMIC replay…")
        for i, issue in enumerate(issue_days):
            try:
                fc, run_label = _cached_issue_weather(issue.isoformat())
                runs[issue] = run_label
                all_rows.extend(_predict_issue(
                    issue_day=issue,
                    all_events=events,
                    actual_lookup=actual_lookup,
                    measured=measured,
                    historical_records=historical_records,
                    forecast_map=fc,
                    run_label=run_label,
                ))
            except Exception as exc:
                errors.append((issue, str(exc)))
            progress.progress((i + 1) / len(issue_days), text=f"{issue.strftime('%d.%m')} · {i+1}/{len(issue_days)}")
        progress.empty()
        field_df = pd.DataFrame(all_rows)
        daily = _daily_scores(field_df, complete_days)
        st.session_state["lab141_results"] = {"field": field_df, "daily": daily, "errors": errors, "runs": runs}

    result = st.session_state.get("lab141_results")
    if not result:
        st.stop()
    field_df: pd.DataFrame = result["field"]
    daily: pd.DataFrame = result["daily"]
    errors = result["errors"]

    if field_df.empty or daily.empty:
        st.error("Replay ei saanud ühtegi täielikku päeva skoorida.")
        if errors:
            st.dataframe(pd.DataFrame([{"Päev": d, "Viga": e} for d, e in errors]), use_container_width=True, hide_index=True)
        st.stop()

    overall = _summary_table(daily)
    warm = _window_summary(daily, WARM_START)
    dyn = overall.loc[overall["Mudel"] == "DYNAMIC"].iloc[0]
    own = overall.loc[overall["Mudel"] == "OWN2"].iloc[0]
    plus1 = daily[daily["lead"] == 1]
    p1 = _summary_table(plus1)
    dyn1 = p1.loc[p1["Mudel"] == "DYNAMIC"].iloc[0] if not p1.empty else None
    flat = overall.loc[overall["Mudel"] == "FLAT-MEM10+BLOCK1"].iloc[0]

    t1, t2, t3, t4 = st.columns(4)
    t1.metric("OWN2 baas · MAPE", f"{float(own['MAPE %']):.1f}%")
    t2.metric("DYNAMIC · kogu MAPE", f"{float(dyn['MAPE %']):.1f}%", f"{float(dyn['Δ MAPE vs OWN2']):+.1f} pp vs OWN2")
    t3.metric("DYNAMIC · +1p MAPE", "—" if dyn1 is None else f"{float(dyn1['MAPE %']):.1f}%")
    t4.metric("Flat BLOCK referents", f"{float(flat['MAPE %']):.1f}%")

    st.markdown("### 1. Fikseeritud arhitektuur · kogu replay")
    st.dataframe(
        overall.style.format({
            "MAPE %": "{:.1f}%", "±20% sees %": "{:.0f}%", "Bias %": "{:+.1f}%",
            "MAE ABC": "{:.1f}", "Δ MAPE vs OWN2": lambda x: f"{x:+.1f} pp",
        }),
        use_container_width=True, hide_index=True,
    )

    st.markdown("### 2. Alates 01.08 · küpsem state")
    if warm.empty:
        st.caption("01.08 järel pole skooritavaid ridu.")
    else:
        st.dataframe(
            warm.style.format({
                "MAPE %": "{:.1f}%", "±20% sees %": "{:.0f}%", "Bias %": "{:+.1f}%",
                "MAE ABC": "{:.1f}", "Δ MAPE vs OWN2": lambda x: f"{x:+.1f} pp",
            }),
            use_container_width=True, hide_index=True,
        )

    st.markdown("### 3. Lead 1–9 · sama DYNAMIC valem, ainult BLOCK kaal hääbub")
    lead_df = _lead_table(daily)
    fmts = {"Fallback %": "{:.0f}%", "BLOCK kaal": "{:.3f}", "Võitja MAPE %": "{:.1f}%"}
    for name in MODEL_NAMES:
        fmts[f"{name} MAPE %"] = "{:.1f}%"
    st.dataframe(lead_df.style.format(fmts), use_container_width=True, hide_index=True)

    st.markdown("### 4. Viimased skooritavad sihtpäevad")
    latest_targets = sorted(daily["target_day"].unique())[-7:]
    latest = daily[daily["target_day"].isin(latest_targets)].copy().sort_values(["target_day", "lead"])
    latest["Issue"] = latest["issue_day"].map(lambda d: pd.Timestamp(d).strftime("%d.%m"))
    latest["Siht"] = latest["target_day"].map(lambda d: pd.Timestamp(d).strftime("%d.%m"))
    show_cols = ["Issue", "Siht", "lead", "block_weight", "actual_abc"] + MODEL_NAMES
    st.dataframe(
        latest[show_cols].rename(columns={"lead": "Lead", "block_weight": "BLOCK kaal", "actual_abc": "Tegelik ABC"}).style.format({
            "BLOCK kaal": "{:.3f}", "Tegelik ABC": "{:.1f}", **{name: "{:.1f}" for name in MODEL_NAMES}
        }),
        use_container_width=True, hide_index=True,
    )

    if errors:
        st.warning(f"ECMWF issue-vigu: {len(errors)}. Need päevad jäid replay'st välja.")
        with st.expander("Näita ECMWF vigu", expanded=False):
            st.dataframe(pd.DataFrame([{"Päev": d, "Viga": e} for d, e in errors]), use_container_width=True, hide_index=True)

    st.divider()
    st.caption(
        f"{LAB_VERSION} · Ridge α={RIDGE_ALPHA:g} · recency half-life={RECENCY_HALFLIFE_DAYS:g} p · "
        f"small-sample prior={SMALL_SAMPLE_PRIOR_ROWS:g} · MEM10 · BLOCK half-life={BLOCK_HALF_LIFE_DAYS:g} p · READ ONLY"
    )


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        _self_test()
    else:
        main()
