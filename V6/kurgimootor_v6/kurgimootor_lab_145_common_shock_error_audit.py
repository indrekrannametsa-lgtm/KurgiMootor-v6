from __future__ import annotations

"""
KurgiMootor LAB-145 · COMMON SHOCK + ERROR AUDIT
=================================================

Eesmärk
-------
LAB-144 error audit näitas, et järelejäänud viga ei ole peamiselt põlluspetsiifiline:
- umbes 78% absoluutsest veast tekkis juhtudel, kus kõik 3 põldu eksisid samas suunas;
- 5 halvimat sihtpäeva andsid umbes poole kogu absoluutveast;
- kõrgetel päevadel oli prognoos liiga madal ja madalatel päevadel liiga kõrge.

Selles LAB-is on üks uus kandidaat: COMMON-SHOCK.

COMMON-SHOCK arhitektuur:
  state = OWN2
  slow = MEM10 correction
  shock = viimase värske täieliku 3-põllu päeva mediaan log(tegelik / range MEM10-ootus)
  ABC = state_rate * exp(slow + shock) * growth_days

Shock ei ole lihtsalt viimase päeva saagitase. Iga shock-päeva ootus arvutatakse rangelt
walk-forward: selle päeva MEM10 mudel näeb ainult varasemaid target-päevi. Seejärel
võrreldakse kolme põllu tegelikku ABC-d nende rangete ootustega ja võetakse robustne
mediaan log-jääk. Tuleviku prognoos kasutab ainult issue-päevaks juba teada olevat shock'i.

Referentsid:
1) OWN2 – sama põllu state;
2) MEM10-ZERO – LAB-140 täpne 10 p ilmamälu;
3) FLAT-MEM10+BLOCK1 – senine parim lihtne baas;
4) COMMON-SHOCK – OWN2 + MEM10 + värske ühine forecast-error shock.

Lisaks säilib vea anatoomia audit ning eraldi kontrollitakse 4.75 p kasvuaega, sest LAB-144
näitas selles rühmas ebatavaliselt suurt negatiivset bias'i.

Aususe reeglid
--------------
- Replay algab 22.07.2026.
- Issue-päev D näeb ainult D lõpuks teada olnud korjeid ja mõõdetud ilma.
- D+1...D+9 tuleviku ilm tuleb D arhiveeritud ECMWF IFS jooksust.
- COMMON SHOCK päeval S kasutab S tegelikku saaki alles issue-del S ja hiljem; shock'i
  ootus on arvutatud mudeliga, mis nägi ainult target_day < S ridu.
- Sama issue +1...+9 sees prognoositud saaki EI söödetata tagasi state'iks ega shock'iks.
- Skoorimisse lähevad ainult täielikud 3-põllu tegelikud korjepäevad.
- Ainult A+B+C. XL ja C/B ei osale.
- READ ONLY: DB-st ainult get_harvest_history ja get_weather_rows.
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
import hashlib

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# FIKSEERITUD KONFIGURATSIOON
# -----------------------------------------------------------------------------

TZ = ZoneInfo("Europe/Tallinn")
TODAY = datetime.now(TZ).date()
LAB_VERSION = "LAB-145-COMMON-SHOCK-AUDIT-V1"

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
REGIME_TREND_FEATURES = MEM10_BLOCK_FEATURES + ["regime_trend_signal"]
MODEL_NAMES = ["OWN2", "MEM10-ZERO", "FLAT-MEM10+BLOCK1", "COMMON-SHOCK"]
PARITY_TOL = 1e-9
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


def _regime_trend_signal(known_events: List[Event], cutoff_exclusive: date) -> Tuple[float, Optional[date], int]:
    """Robustne viimase kuni 3 täieliku korjepäeva log-rate trend päevas.

    Kasutab ainult cutoff'ile eelnenud täielikke 3-põllu päevi. Trend on Theil-Sen
    stiilis mediaan kõigist paarikaupa log-rate tõusudest / päevade vahest.
    Kui viimane täielik päev on vanem kui BLOCK_FRESH_DAYS, tagastab 0.
    """
    meds = _complete_day_rate_medians(known_events, cutoff_exclusive)
    if len(meds) < 2:
        return 0.0, (meds[-1][0] if meds else None), len(meds)
    last_day = meds[-1][0]
    reference_day = cutoff_exclusive - timedelta(days=1)
    age = (reference_day - last_day).days
    if age < 0 or age > BLOCK_FRESH_DAYS:
        return 0.0, last_day, min(3, len(meds))
    pts = meds[-3:]
    slopes: List[float] = []
    for i in range(len(pts)):
        d1, r1 = pts[i]
        if r1 <= 0:
            continue
        for j in range(i + 1, len(pts)):
            d2, r2 = pts[j]
            dt = (d2 - d1).days
            if dt <= 0 or r2 <= 0:
                continue
            slopes.append((math.log(r2) - math.log(r1)) / float(dt))
    if not slopes:
        return 0.0, last_day, len(pts)
    slope = float(np.median(np.asarray(slopes, dtype=float)))
    # Ainult numbriline kaitse üksiku vigase ajalookande vastu; ridge standardiseerib edasi.
    slope = float(np.clip(slope, -0.50, 0.50))
    return slope, last_day, len(pts)


def _complete_actual_days(actual_lookup: Dict[Tuple[date, int], dict]) -> set:
    by_day: Dict[date, set] = {}
    for dd, field in actual_lookup:
        by_day.setdefault(dd, set()).add(field)
    return {dd for dd, fields in by_day.items() if len(fields) == EXPECTED_FIELDS_PER_DAY}


# -----------------------------------------------------------------------------
# COMMON SHOCK · range forecast-error viimase täieliku päeva pealt
# -----------------------------------------------------------------------------

SHOCK_FRESH_DAYS = 2
SHOCK_NUMERIC_CLIP = 0.75

def _build_common_shock_map(historical_records: List[dict], complete_days: set) -> Dict[date, dict]:
    """Ehita iga täieliku päeva range MEM10 forecast-error shock.

    Päeva S ootus kasutab ainult target_day < S treeningridu. Päeva S tegelik saak
    kasutatakse ainult shock'i moodustamiseks, mida tohib kasutada issue-del S ja hiljem.
    Signaal = mediaan kolme põllu log(actual_abc / strict_mem10_pred_abc).
    """
    by_day: Dict[date, List[dict]] = {}
    for r in historical_records:
        dd = r.get("target_day")
        if dd in complete_days:
            by_day.setdefault(dd, []).append(r)

    out: Dict[date, dict] = {}
    all_rows = sorted(historical_records, key=lambda r: (r["target_day"], r["field_no"]))
    for dd in sorted(by_day):
        rows = by_day[dd]
        fields = {int(r.get("field_no")) for r in rows}
        if len(rows) != EXPECTED_FIELDS_PER_DAY or len(fields) != EXPECTED_FIELDS_PER_DAY:
            continue
        prior = [r for r in all_rows if r["target_day"] < dd]
        model = _fit_lab140_mem10_reference(prior, dd)
        residuals: List[float] = []
        actual_sum = 0.0
        expected_sum = 0.0
        for r in rows:
            actual = _f(r.get("actual_abc"))
            if actual is None or actual <= 0:
                continue
            expected = float(_predict_lab140_mem10_reference(model, r))
            if not math.isfinite(expected) or expected <= 0:
                continue
            residuals.append(math.log(max(1e-6, actual) / max(1e-6, expected)))
            actual_sum += float(actual)
            expected_sum += float(expected)
        if len(residuals) != EXPECTED_FIELDS_PER_DAY:
            continue
        raw = float(np.median(np.asarray(residuals, dtype=float)))
        signal = float(np.clip(raw, -SHOCK_NUMERIC_CLIP, SHOCK_NUMERIC_CLIP))
        out[dd] = {
            "signal": signal, "raw": raw, "n": len(residuals),
            "actual": float(actual_sum), "expected": float(expected_sum),
        }
    return out


def _latest_common_shock(shock_map: Dict[date, dict], reference_day: date) -> Tuple[float, Optional[date], Optional[dict]]:
    days = [dd for dd in shock_map if dd <= reference_day]
    if not days:
        return 0.0, None, None
    dd = max(days)
    age = (reference_day - dd).days
    meta = shock_map.get(dd)
    if age < 0 or age > SHOCK_FRESH_DAYS or not meta:
        return 0.0, dd, meta
    return float(meta.get("signal") or 0.0), dd, meta


def _predict_common_shock(mem10_model: Optional[dict], rec: dict) -> float:
    mem_hat = _mem10_yhat_or_zero(mem10_model, rec)
    shock = float(rec.get("common_shock_signal") or 0.0)
    return _abc_from_yhat(rec, float(mem_hat + shock))


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
    regime_trend, regime_trend_day, regime_trend_n = _regime_trend_signal(known_events, block_cutoff)

    rec = {
        "target_day": target_day,
        "field_no": field,
        "own2_rate": float(own2_rate),
        "used_fallback": int(used_fallback),
        "block1_signal": float(block_sig),
        "block1_day": block_day,
        "regime_trend_signal": float(regime_trend),
        "regime_trend_day": regime_trend_day,
        "regime_trend_n": int(regime_trend_n),
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
# CORRECTION RIDGE + LEARNED-BLOCK ARHITEKTUUR
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





# -----------------------------------------------------------------------------
# LEARNED BLOCK: MEM10 range walk-forward jääk -> üks scalar beta
# -----------------------------------------------------------------------------

def _mem10_yhat_or_zero(model: Optional[dict], rec: dict) -> float:
    """LAB-140 MEM10 correction log-rate skaalal. Mudeli puudumisel = 0 (OWN2)."""
    if model is None:
        return 0.0
    vals = [_f(rec.get(c)) for c in MEM10_FEATURES]
    if any(v is None for v in vals):
        return 0.0
    x = np.asarray(vals, dtype=float)
    z = np.clip((x - model["mu"]) / model["sd"], -3.0, 3.0)
    yhat = float(z @ model["beta"])
    if model.get("bounds") is not None:
        lo, hi = model["bounds"]
        yhat = float(np.clip(yhat, lo, hi))
    n_train = float(model.get("n") or 0.0)
    yhat *= n_train / (n_train + SMALL_SAMPLE_PRIOR_ROWS)
    return float(yhat)


def _target_bounds(records: List[dict]) -> Optional[Tuple[float, float]]:
    ys = np.asarray([float(r["y"]) for r in records if _f(r.get("y")) is not None], dtype=float)
    if len(ys) < 10:
        return None
    lo, hi = np.quantile(ys, [0.02, 0.98])
    pad = max(0.05, 0.12 * float(hi - lo))
    return float(lo - pad), float(hi + pad)


def _fit_learned_block_beta(all_historical_records: List[dict], cutoff_day: date) -> Optional[dict]:
    """Õpi üks beta rangelt varasemate MEM10 walk-forward jääkide pealt.

    Iga meta-rida d:
      mem10_hat(d) = MEM10 mudel, mis nägi ainult target_day < d ridu;
      residual(d) = tegelik log-rate muutus - mem10_hat(d);
      residual ~= beta * block1_signal.

    BLOCK x skaleeritakse weighted RMS-iga (ilma tsentreerimiseta), et ridge oleks
    ühikust sõltumatu ja intercepti ei tekiks.
    """
    usable = [r for r in all_historical_records if r["target_day"] <= cutoff_day and _f(r.get("y")) is not None]
    usable.sort(key=lambda r: (r["target_day"], r["field_no"]))
    meta = []
    # Cache: sama target-päeva kõik read kasutavad sama rangelt varasemat MEM10 mudelit.
    by_day_model: Dict[date, Optional[dict]] = {}
    for r in usable:
        d = r["target_day"]
        x = _f(r.get("block1_signal"))
        y = _f(r.get("y"))
        if x is None or y is None or abs(float(x)) <= 1e-12:
            continue
        if d not in by_day_model:
            prior = [q for q in usable if q["target_day"] < d]
            by_day_model[d] = _fit_lab140_mem10_reference(prior, d)
        mem_hat = _mem10_yhat_or_zero(by_day_model[d], r)
        residual = float(y) - float(mem_hat)
        age = max(0.0, float((cutoff_day - d).days))
        w = 0.5 ** (age / RECENCY_HALFLIFE_DAYS)
        meta.append((float(x), residual, float(w), d))

    if len(meta) < MIN_TRAIN_ROWS:
        return None
    x = np.asarray([a for a, _r, _w, _d in meta], dtype=float)
    resid = np.asarray([r for _a, r, _w, _d in meta], dtype=float)
    w = np.asarray([ww for _a, _r, ww, _d in meta], dtype=float)
    sw = float(w.sum())
    if sw <= 0:
        return None
    rms = math.sqrt(max(1e-10, float(np.sum(w * x * x) / sw)))
    z = x / rms
    denom = float(np.sum(w * z * z) + RIDGE_ALPHA)
    if denom <= 0:
        return None
    beta_scaled = float(np.sum(w * z * resid) / denom)
    beta_raw = float(beta_scaled / rms)
    return {
        "beta": beta_raw,
        "n": int(len(meta)),
        "rms": float(rms),
        "bounds": _target_bounds(usable),
    }


def _predict_learned_block(mem10_model: Optional[dict], block_model: Optional[dict], rec: dict) -> float:
    mem_hat = _mem10_yhat_or_zero(mem10_model, rec)
    if block_model is None:
        return _abc_from_yhat(rec, mem_hat)
    block = float(rec.get("block1_signal") or 0.0)
    yhat = float(mem_hat + float(block_model["beta"]) * block)
    if block_model.get("bounds") is not None:
        lo, hi = block_model["bounds"]
        yhat = float(np.clip(yhat, lo, hi))
    return _abc_from_yhat(rec, yhat)

# -----------------------------------------------------------------------------
# LAB-140 TÄPNE MEM10 REFERENTS + PARITEEDIKONTROLL
# -----------------------------------------------------------------------------

def _fit_lab140_mem10_reference(train_records: List[dict], cutoff_day: date) -> Optional[dict]:
    """LAB-140 _fit_model(..., MEM10_FEATURES) täpne koopia.

    Oluline: intercept=0 ja kui mudelit pole piisava treeningu tõttu, peab ennustus
    LAB-140 kombel kukkuma tagasi OWN2 peale, mitte muutuma NaN/0-ks.
    """
    clean = []
    for r in train_records:
        y = _f(r.get("y"))
        vals = [_f(r.get(c)) for c in MEM10_FEATURES]
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
    ws = np.sqrt(w)[:, None]
    Zw = Z * ws
    yw = y * ws[:, 0]
    penalty = np.eye(Z.shape[1], dtype=float)
    lhs = Zw.T @ Zw + RIDGE_ALPHA * penalty
    rhs = Zw.T @ yw
    try:
        beta = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(lhs) @ rhs

    bounds = None
    if len(y) >= 10:
        lo, hi = np.quantile(y, [0.02, 0.98])
        pad = max(0.05, 0.12 * float(hi - lo))
        bounds = (float(lo - pad), float(hi + pad))
    return {"mu": mu, "sd": sd, "beta": beta, "bounds": bounds, "n": len(clean)}


def _predict_lab140_mem10_reference(model: Optional[dict], rec: dict) -> float:
    """LAB-140 _predict_variant täpne MEM10 käitumine, sh OWN2 fallback."""
    own2_rate = float(rec["own2_rate"])
    growth = max(0.5, float(rec["growth_days"]))
    if model is None:
        return float(max(0.0, own2_rate * growth))
    vals = [_f(rec.get(c)) for c in MEM10_FEATURES]
    if any(v is None for v in vals):
        return float(max(0.0, own2_rate * growth))
    x = np.asarray(vals, dtype=float)
    z = np.clip((x - model["mu"]) / model["sd"], -3.0, 3.0)
    yhat = float(z @ model["beta"])
    if model.get("bounds") is not None:
        lo, hi = model["bounds"]
        yhat = float(np.clip(yhat, lo, hi))
    n_train = float(model.get("n") or 0.0)
    yhat *= n_train / (n_train + SMALL_SAMPLE_PRIOR_ROWS)
    pred_rate = max(0.0, own2_rate * math.exp(yhat))
    return float(pred_rate * growth)


def _predict_or_own2(model: Optional[dict], rec: dict, features: List[str]) -> float:
    """Production-safe fallback: learned correction unavailable => OWN2, never NaN/zero."""
    yhat = _predict_correction(model, rec, features)
    return _own2_pred(rec) if yhat is None else _abc_from_yhat(rec, yhat)


def _input_fingerprint(events: List[Event], measured: Dict[date, dict]) -> str:
    """Deterministic short hash so two replay screens can prove same inputs."""
    h = hashlib.sha256()
    for e in sorted(events, key=_event_sort_key):
        h.update(f"H|{e.day.isoformat()}|{e.field}|{e.order}|{e.abc}|{int(e.reliable)}|{e.source}\n".encode())
    for dd in sorted(measured):
        r = measured[dd]
        vals = [r.get(k) for k in ("temp", "rad", "et0", "rain", "rh", "wind")]
        h.update(("W|" + dd.isoformat() + "|" + "|".join(str(v) for v in vals) + "\n").encode())
    return h.hexdigest()[:16]


class ParityError(RuntimeError):
    pass


def _fit_issue_models(all_historical_records: List[dict], train_records: List[dict], cutoff_day: date) -> dict:
    return {
        "LAB140-REF": _fit_lab140_mem10_reference(train_records, cutoff_day),
        "MEM10-ZERO": _fit_correction(train_records, cutoff_day, MEM10_FEATURES, allow_intercept=False),
        "FLAT-MEM10+BLOCK1": _fit_correction(train_records, cutoff_day, MEM10_BLOCK_FEATURES, allow_intercept=False),
    }


def _predict_models(models: dict, rec: dict, lead: int) -> Dict[str, float]:
    own = _own2_pred(rec)
    out: Dict[str, float] = {"OWN2": own}

    ref140 = _predict_lab140_mem10_reference(models.get("LAB140-REF"), rec)
    zero = _predict_or_own2(models.get("MEM10-ZERO"), rec, MEM10_FEATURES)
    if abs(float(ref140) - float(zero)) > PARITY_TOL:
        raise ParityError(
            f"LAB140 parity mismatch: ref={ref140:.12f}, zero={zero:.12f}, "
            f"target={rec.get('target_day')}, field={rec.get('field_no')}"
        )
    out["MEM10-ZERO"] = float(zero)
    out["FLAT-MEM10+BLOCK1"] = _predict_or_own2(
        models.get("FLAT-MEM10+BLOCK1"), rec, MEM10_BLOCK_FEATURES
    )
    out["COMMON-SHOCK"] = _predict_common_shock(models.get("MEM10-ZERO"), rec)
    out["LAB140-REF"] = float(ref140)
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
    shock_map: Dict[date, dict],
    forecast_map: Dict[date, dict],
    run_label: str,
) -> List[dict]:
    known_events = [e for e in all_events if e.day <= issue_day]
    schedule_events = _copy_events(known_events)
    wmap = _combined_weather(measured, forecast_map, issue_day)
    train = [r for r in historical_records if r["target_day"] <= issue_day]
    models = _fit_issue_models(historical_records, train, issue_day)
    stubs = _future_schedule_stubs(actual_lookup, issue_day)
    shock_sig, shock_day, shock_meta = _latest_common_shock(shock_map, issue_day)

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
        rec["common_shock_signal"] = float(shock_sig)
        rec["common_shock_day"] = shock_day

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
            "regime_trend_signal": float(rec.get("regime_trend_signal") or 0.0),
            "common_shock_signal": float(shock_sig),
            "common_shock_day": shock_day,
            "shock_actual": float(shock_meta.get("actual")) if shock_meta else np.nan,
            "shock_expected": float(shock_meta.get("expected")) if shock_meta else np.nan,
            "run": run_label,
            "w10_gdd10": float(rec["w10_gdd10"]) if _f(rec.get("w10_gdd10")) is not None else np.nan,
            "w10_rad": float(rec["w10_rad"]) if _f(rec.get("w10_rad")) is not None else np.nan,
            "w10_et0": float(rec["w10_et0"]) if _f(rec.get("w10_et0")) is not None else np.nan,
        }
        row["LAB140-REF"] = float(preds["LAB140-REF"])
        row["parity_abs_diff"] = abs(float(preds["LAB140-REF"]) - float(preds["MEM10-ZERO"]))
        for name in MODEL_NAMES:
            row[name] = float(preds[name])
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
        "common_shock_signal": ("common_shock_signal", "mean"),
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
# ERROR AUDIT · KUS ON JÄRELEJÄÄNUD VIGA?
# -----------------------------------------------------------------------------

def _audit_stats(df: pd.DataFrame, model: str, group_col: str, label: str) -> pd.DataFrame:
    if df.empty or model not in df.columns or group_col not in df.columns:
        return pd.DataFrame()
    x = df.copy()
    x = x[np.isfinite(x[model]) & np.isfinite(x["actual_abc"]) & (x["actual_abc"] > 0)].copy()
    if x.empty:
        return pd.DataFrame()
    x["_err"] = x[model] - x["actual_abc"]
    x["_ape"] = x["_err"].abs() / x["actual_abc"].clip(lower=0.1)
    x["_pcterr"] = 100.0 * x["_err"] / x["actual_abc"].clip(lower=0.1)
    x["_abserr"] = x["_err"].abs()
    total_abs = float(x["_abserr"].sum())
    rows = []
    for key, g in x.groupby(group_col, dropna=False):
        rows.append({
            label: str(key),
            "N": int(len(g)),
            "MAPE %": 100.0 * float(g["_ape"].mean()),
            "MAE ABC": float(g["_abserr"].mean()),
            "Bias %": float(g["_pcterr"].mean()),
            "Abs vea osa %": 100.0 * float(g["_abserr"].sum()) / max(1e-9, total_abs),
        })
    return pd.DataFrame(rows).sort_values(["Abs vea osa %", "MAPE %"], ascending=[False, False])


def _error_audit_tables(field_df: pd.DataFrame, daily: pd.DataFrame, complete_days: set, model: str = "FLAT-MEM10+BLOCK1") -> dict:
    f = field_df[field_df["target_day"].isin(complete_days)].copy()
    f = f[np.isfinite(f["actual_abc"]) & (f["actual_abc"] > 0) & np.isfinite(f[model])].copy()
    if f.empty:
        return {}
    f["growth_bucket"] = (np.round(f["growth_days"].astype(float) * 4.0) / 4.0).map(lambda v: f"{v:.2f} p")
    f["trend_direction"] = np.where(
        f["regime_trend_signal"].astype(float) < 0, "Langev",
        np.where(f["regime_trend_signal"].astype(float) > 0, "Tõusev", "Trendita")
    )
    f["_err"] = f[model] - f["actual_abc"]

    # Kas ühe issue-target päeva kõik 3 põldu eksivad samas suunas?
    mode_rows = []
    for keys, g in f.groupby(["issue_day", "target_day", "lead"]):
        if len(g) != EXPECTED_FIELDS_PER_DAY:
            continue
        errs = g["_err"].to_numpy(dtype=float)
        if np.all(errs > 0):
            kind = "Kõik 3 liiga kõrge"
        elif np.all(errs < 0):
            kind = "Kõik 3 liiga madal"
        else:
            kind = "Põlluti segatud"
        for idx in g.index:
            mode_rows.append((idx, kind))
    mode_map = dict(mode_rows)
    f["error_mode"] = [mode_map.get(i, "Mittetäielik") for i in f.index]

    # Tegeliku taseme tertiilid on ainult audit, mitte mudeli sisend.
    try:
        f["actual_level"] = pd.qcut(f["actual_abc"], q=3, labels=["Madal", "Keskmine", "Kõrge"], duplicates="drop")
    except Exception:
        f["actual_level"] = "—"

    # Päevade audit: sama target-päeva eri issue-lead prognoosid koondatakse.
    d = daily.copy()
    d = d[np.isfinite(d[model]) & np.isfinite(d["actual_abc"]) & (d["actual_abc"] > 0)].copy()
    d["_ape"] = (d[model] - d["actual_abc"]).abs() / d["actual_abc"].clip(lower=0.1)
    d["_abserr"] = (d[model] - d["actual_abc"]).abs()
    by_target = d.groupby("target_day", as_index=False).agg(
        N=("lead", "count"),
        Tegelik_ABC=("actual_abc", "mean"),
        MAPE=("_ape", "mean"),
        MAE=("_abserr", "mean"),
        AbsErr=("_abserr", "sum"),
    )
    total_abs = float(by_target["AbsErr"].sum())
    by_target["MAPE %"] = 100.0 * by_target["MAPE"]
    by_target["Abs vea osa %"] = 100.0 * by_target["AbsErr"] / max(1e-9, total_abs)
    by_target["Sihtpäev"] = by_target["target_day"].map(lambda d: pd.Timestamp(d).strftime("%d.%m"))
    by_target = by_target.sort_values(["Abs vea osa %", "MAPE %"], ascending=[False, False])

    top5_share = float(by_target.head(5)["Abs vea osa %"].sum()) if not by_target.empty else float("nan")
    same_dir = f[f["error_mode"].str.startswith("Kõik 3")].copy()
    same_dir_share = 100.0 * float(same_dir["_err"].abs().sum()) / max(1e-9, float(f["_err"].abs().sum())) if not f.empty else float("nan")

    return {
        "field": _audit_stats(f, model, "field", "Põld"),
        "growth": _audit_stats(f, model, "growth_bucket", "Kasvuaeg"),
        "trend": _audit_stats(f, model, "trend_direction", "Režiim"),
        "mode": _audit_stats(f, model, "error_mode", "Vea tüüp"),
        "level": _audit_stats(f, model, "actual_level", "Tegelik tase"),
        "target": by_target[["Sihtpäev", "N", "Tegelik_ABC", "MAPE %", "MAE", "Abs vea osa %"]],
        "top5_share": top5_share,
        "same_dir_share": same_dir_share,
    }


def _target_model_compare(daily: pd.DataFrame, base: str = "FLAT-MEM10+BLOCK1", challenger: str = "COMMON-SHOCK") -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()
    rows = []
    for dd, g in daily.groupby("target_day"):
        g = g[np.isfinite(g[base]) & np.isfinite(g[challenger]) & (g["actual_abc"] > 0)].copy()
        if g.empty:
            continue
        actual = float(g["actual_abc"].mean())
        base_abs = (g[base] - g["actual_abc"]).abs()
        chal_abs = (g[challenger] - g["actual_abc"]).abs()
        rows.append({
            "Sihtpäev": pd.Timestamp(dd).strftime("%d.%m"),
            "N": int(len(g)),
            "Tegelik ABC": actual,
            "FLAT MAPE %": 100.0 * float((base_abs / g["actual_abc"].clip(lower=0.1)).mean()),
            "SHOCK MAPE %": 100.0 * float((chal_abs / g["actual_abc"].clip(lower=0.1)).mean()),
            "FLAT MAE": float(base_abs.mean()),
            "SHOCK MAE": float(chal_abs.mean()),
            "Δ MAE": float(chal_abs.mean() - base_abs.mean()),
            "FLAT abs sum": float(base_abs.sum()),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["FLAT abs sum", "FLAT MAPE %"], ascending=[False, False])


def _growth_475_audit(field_df: pd.DataFrame, complete_days: set) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if field_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    f = field_df[field_df["target_day"].isin(complete_days)].copy()
    f = f[np.isfinite(f["growth_days"]) & (np.round(f["growth_days"].astype(float) * 4.0) / 4.0 == 4.75)].copy()
    f = f[np.isfinite(f["actual_abc"]) & (f["actual_abc"] > 0)].copy()
    if f.empty:
        return pd.DataFrame(), pd.DataFrame()
    summary = []
    for name in ["FLAT-MEM10+BLOCK1", "COMMON-SHOCK"]:
        g = f[np.isfinite(f[name])].copy()
        err = g[name] - g["actual_abc"]
        summary.append({
            "Mudel": name, "N": int(len(g)),
            "MAPE %": 100.0 * float((err.abs() / g["actual_abc"].clip(lower=0.1)).mean()),
            "MAE ABC": float(err.abs().mean()),
            "Bias %": 100.0 * float((err / g["actual_abc"].clip(lower=0.1)).mean()),
        })
    detail = f.groupby(["target_day", "field"], as_index=False).agg(
        N_issue=("issue_day", "count"), actual_abc=("actual_abc", "mean"),
        flat=("FLAT-MEM10+BLOCK1", "mean"), shock=("COMMON-SHOCK", "mean"),
        order=("order", "mean"), growth=("growth_days", "mean"),
    )
    detail["Siht"] = detail["target_day"].map(lambda d: pd.Timestamp(d).strftime("%d.%m"))
    detail["FLAT abs"] = (detail["flat"] - detail["actual_abc"]).abs()
    detail["SHOCK abs"] = (detail["shock"] - detail["actual_abc"]).abs()
    detail = detail.sort_values("FLAT abs", ascending=False)
    return pd.DataFrame(summary), detail[["Siht", "field", "order", "growth", "N_issue", "actual_abc", "flat", "shock", "FLAT abs", "SHOCK abs"]]


# -----------------------------------------------------------------------------
# SELF TEST
# -----------------------------------------------------------------------------

def _self_test() -> None:
    measured: Dict[date, dict] = {}
    start = date(2026, 7, 1)
    for i in range(90):
        dd = start + timedelta(days=i)
        measured[dd] = {
            "day": dd, "temp": 18.0 + 0.025*i, "temp_min": 12.0, "temp_max": 25.0,
            "wind": 2.0, "rad": 17.0 + 0.07*i, "rh": 75.0, "rain": 0.4,
            "et0": 2.8 + 0.008*i, "source": "measured",
        }

    rows = []
    days = [date(2026,7,5) + timedelta(days=4*j) for j in range(14)]
    for j, dd in enumerate(days):
        for order, f in enumerate((1, 2, 3), start=1):
            common = 1.0 + 0.12 * math.sin(j / 2.0)
            if j == 11:
                common *= 0.65  # tugev ühine shock testiks
            abc = (6.5 + 0.28*j + 0.12*f) * common
            rows.append({
                "harvest_date": dd.isoformat(), "field_no": f, "harvest_order": order,
                "a": 0.2, "b": abc*0.45, "c": abc*0.55-0.2, "xl": 1.0,
                "data_quality": "Kinnitatud",
            })
    events, lookup = _prepare_events(rows)
    complete = _complete_actual_days(lookup)
    hist = _make_historical_records(events, lookup, measured)
    assert hist and all(_f(r.get("w10_rad")) is not None for r in hist if r["target_day"] >= date(2026,7,15))
    shock_map = _build_common_shock_map(hist, complete)
    assert shock_map and all(math.isfinite(float(v["signal"])) for v in shock_map.values())

    cutoff = hist[-1]["target_day"]
    train = [r for r in hist if r["target_day"] < cutoff]
    models = _fit_issue_models(hist, train, cutoff)
    rec0 = dict(hist[-1])
    sig, sday, _meta = _latest_common_shock(shock_map, cutoff - timedelta(days=1))
    rec0["common_shock_signal"] = sig
    rec0["common_shock_day"] = sday
    preds = _predict_models(models, rec0, 1)
    assert set(MODEL_NAMES).issubset(preds) and "LAB140-REF" in preds
    assert all(math.isfinite(float(preds[k])) and float(preds[k]) >= 0 for k in MODEL_NAMES)
    assert abs(float(preds["LAB140-REF"]) - float(preds["MEM10-ZERO"])) <= PARITY_TOL

    tiny_train = train[:2]
    tiny_models = _fit_issue_models(tiny_train, tiny_train, cutoff)
    tiny_rec = dict(rec0)
    tiny_rec["common_shock_signal"] = 0.0
    tiny_preds = _predict_models(tiny_models, tiny_rec, 1)
    assert abs(tiny_preds["MEM10-ZERO"] - _own2_pred(tiny_rec)) < 1e-12
    assert abs(tiny_preds["COMMON-SHOCK"] - tiny_preds["MEM10-ZERO"]) < 1e-12

    issue = date(2026, 8, 6)
    fc = {d: dict(v, source="ecmwf_forecast") for d, v in measured.items() if d > issue}
    out = _predict_issue(
        issue_day=issue, all_events=events, actual_lookup=lookup, measured=measured,
        historical_records=hist, shock_map=shock_map, forecast_map=fc, run_label="synthetic",
    )
    assert out and all(all(name in r for name in MODEL_NAMES) for r in out)
    assert max(float(r.get("parity_abs_diff") or 0.0) for r in out) <= PARITY_TOL
    assert all(math.isfinite(float(r["common_shock_signal"])) for r in out)
    print("LAB-145 SELF-TEST OK")

# -----------------------------------------------------------------------------
# STREAMLIT
# -----------------------------------------------------------------------------

def main() -> None:
    import streamlit as st
    import db

    st.set_page_config(page_title="KurgiMootor LAB-145", layout="wide")
    st.error("🧪 LAB-145 · COMMON SHOCK + ERROR AUDIT · READ-ONLY · 22.07 → +1…+9")
    st.title("KurgiMootor · COMMON SHOCK + vea anatoomia")
    st.caption(
        "Üks uus kandidaat kasutab viimase täieliku korjepäeva kolme põllu ranget MEM10 forecast-error jääki ühise "
        "taimikushokina. Audit kontrollib eraldi ka 4.75 p kasvuaega ja seda, kas shock parandab seni domineerinud ühiseid päevavigu."
    )

    with st.expander("Arhitektuur ja aususe reeglid", expanded=False):
        st.markdown(
            f"""
- **OWN2:** sama põllu kuni 2 viimase ABC/kasvupäev rate'i mediaan; cold-start = BLOCK3 fallback.
- **MEM10:** targetile eelneva 10 päeva GDD10 + radiatsiooni + ET0; LAB-140 täpne referents.
- **FLAT baas:** MEM10 + vana BLOCK1 suhte-signaal.
- **COMMON SHOCK:** viimase värske täieliku 3-põllu päeva mediaan `log(tegelik / range MEM10-ootus)`.
- Shock-päeva ootus kasutab ainult sellest päevast **varasemaid** target-ridu; shock'i tegelik saak jõuab sisendisse alles selle päeva lõpus.
- Shock'i kaal on selles testis fikseeritud **1.0**; lead-decay'd, uut aknaotsingut ega championivalikut ei ole.
- Shock on värske kuni {SHOCK_FRESH_DAYS} päeva; robustne mediaan ja ainult numbriline piir ±{SHOCK_NUMERIC_CLIP:.2f} log-skaalal.
- Tuleviku ilm = arhiveeritud ECMWF issue-jooks; issue-päevani mõõdetud ilm.
- Prognoositud ABC-d ei kasutata sama +1…+9 akna järgmise state'i ega shock'ina.
- Skoor ainult täielikel 3-põllu päevadel ja ainult **A+B+C**.
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
        st.session_state.pop("lab145_results", None)
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
    shock_map = _build_common_shock_map(historical_records, complete_days)
    fingerprint = _input_fingerprint(events, measured)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Replay algus", REPLAY_START.strftime("%d.%m.%Y"))
    c2.metric("Usaldusväärseid ABC ridu", str(len(actual_lookup)))
    c3.metric("Historical training ridu", str(len(historical_records)))
    c4.metric("Issue-päevi", str(len(issue_days)))
    c5.metric("Range shock-päevi", str(len(shock_map)))
    c6.metric("Input fingerprint", fingerprint)

    run_requested = st.button("▶ VERIFY LAB-140 → jooksuta COMMON SHOCK + ERROR AUDIT", type="primary")
    if not run_requested and "lab145_results" not in st.session_state:
        st.info("Vajuta üks kord. ECMWF issue-jooksud on 24 h cache'is; productionit ei puuduta.")
        st.stop()

    if run_requested:
        all_rows: List[dict] = []
        errors: List[Tuple[date, str]] = []
        runs: Dict[date, str] = {}
        progress = st.progress(0.0, text="LAB-140 parity + ECMWF + COMMON SHOCK replay…")
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
                    shock_map=shock_map,
                    forecast_map=fc,
                    run_label=run_label,
                ))
            except ParityError as exc:
                progress.empty()
                st.error(f"STOP · {issue.strftime('%d.%m')} · {exc}")
                st.stop()
            except Exception as exc:
                errors.append((issue, str(exc)))
            progress.progress((i + 1) / len(issue_days), text=f"{issue.strftime('%d.%m')} · {i+1}/{len(issue_days)}")
        progress.empty()
        field_df = pd.DataFrame(all_rows)
        max_parity = float(field_df["parity_abs_diff"].max()) if not field_df.empty else float("nan")
        if not math.isfinite(max_parity) or max_parity > PARITY_TOL:
            st.error(f"STOP: LAB-140 referents ei klapi. max |Δ| = {max_parity}")
            st.stop()
        daily = _daily_scores(field_df, complete_days)
        st.session_state["lab145_results"] = {
            "field": field_df, "daily": daily, "errors": errors, "runs": runs,
            "max_parity": max_parity, "fingerprint": fingerprint,
        }

    result = st.session_state.get("lab145_results")
    if not result:
        st.stop()
    field_df: pd.DataFrame = result["field"]
    daily: pd.DataFrame = result["daily"]
    errors = result["errors"]

    if field_df.empty or daily.empty:
        st.error("Replay ei saanud ühtegi täielikku päeva skoorida.")
        st.stop()

    max_parity = float(result.get("max_parity", float("nan")))
    if not math.isfinite(max_parity) or max_parity > PARITY_TOL:
        st.error(f"STOP: LAB-140 parity kontroll ebaõnnestus · max |Δ|={max_parity}")
        st.stop()
    st.success(
        f"✅ VERIFY PASS · LAB-140 MEM10 referents = MEM10-ZERO · max |Δ|={max_parity:.3g} · "
        f"input {result.get('fingerprint','—')}"
    )

    overall = _summary_table(daily)
    warm = _window_summary(daily, WARM_START)
    flat = overall.loc[overall["Mudel"] == "FLAT-MEM10+BLOCK1"].iloc[0]
    shock = overall.loc[overall["Mudel"] == "COMMON-SHOCK"].iloc[0]
    mem10 = overall.loc[overall["Mudel"] == "MEM10-ZERO"].iloc[0]
    plus1 = _summary_table(daily[daily["lead"] == 1])
    shock1 = plus1.loc[plus1["Mudel"] == "COMMON-SHOCK"].iloc[0] if not plus1.empty else None
    latest_issue = max(field_df["issue_day"])
    latest_rows = field_df[field_df["issue_day"] == latest_issue]
    latest_shock = float(latest_rows["common_shock_signal"].mean()) if not latest_rows.empty else 0.0

    t1, t2, t3, t4, t5 = st.columns(5)
    t1.metric("FLAT baas · MAPE", f"{float(flat['MAPE %']):.1f}%")
    t2.metric("COMMON-SHOCK · MAPE", f"{float(shock['MAPE %']):.1f}%", f"{float(shock['MAPE %']-flat['MAPE %']):+.1f} pp vs FLAT")
    t3.metric("COMMON-SHOCK · +1p", "—" if shock1 is None else f"{float(shock1['MAPE %']):.1f}%")
    t4.metric("MEM10 referents", f"{float(mem10['MAPE %']):.1f}%")
    t5.metric("Viimane ühine shock", f"{100.0*(math.exp(latest_shock)-1.0):+.0f}%")

    st.markdown("### 1. COMMON-SHOCK kandidaat · kogu replay")
    st.dataframe(
        overall.style.format({
            "MAPE %": "{:.1f}%", "±20% sees %": "{:.0f}%", "Bias %": "{:+.1f}%",
            "MAE ABC": "{:.1f}", "Δ MAPE vs OWN2": lambda x: f"{x:+.1f} pp",
        }), use_container_width=True, hide_index=True,
    )

    st.markdown("### 2. Alates 01.08 · küpsem state")
    if not warm.empty:
        st.dataframe(
            warm.style.format({
                "MAPE %": "{:.1f}%", "±20% sees %": "{:.0f}%", "Bias %": "{:+.1f}%",
                "MAE ABC": "{:.1f}", "Δ MAPE vs OWN2": lambda x: f"{x:+.1f} pp",
            }), use_container_width=True, hide_index=True,
        )

    st.markdown("### 3. Lead 1–9 · COMMON SHOCK vs baasid")
    lead_df = _lead_table(daily)
    fmts = {"Fallback %": "{:.0f}%", "Võitja MAPE %": "{:.1f}%"}
    for name in MODEL_NAMES:
        fmts[f"{name} MAPE %"] = "{:.1f}%"
    st.dataframe(lead_df.style.format(fmts), use_container_width=True, hide_index=True)

    st.markdown("### 4. Kas COMMON SHOCK tabab seda ~30% viga?")
    flat_a = _error_audit_tables(field_df, daily, complete_days, model="FLAT-MEM10+BLOCK1")
    shock_a = _error_audit_tables(field_df, daily, complete_days, model="COMMON-SHOCK")
    cmp = _target_model_compare(daily)
    if flat_a and shock_a:
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("FLAT · 5 halvima päeva veaosa", f"{float(flat_a['top5_share']):.0f}%")
        a2.metric("SHOCK · 5 halvima päeva veaosa", f"{float(shock_a['top5_share']):.0f}%")
        a3.metric("FLAT · kõik 3 sama suund", f"{float(flat_a['same_dir_share']):.0f}%")
        a4.metric("SHOCK · kõik 3 sama suund", f"{float(shock_a['same_dir_share']):.0f}%")
        if not cmp.empty:
            st.caption("Halvimad FLAT sihtpäevad · kas COMMON SHOCK vähendas viga?")
            st.dataframe(
                cmp.head(14).drop(columns=["FLAT abs sum"]).style.format({
                    "Tegelik ABC":"{:.1f}", "FLAT MAPE %":"{:.1f}%", "SHOCK MAPE %":"{:.1f}%",
                    "FLAT MAE":"{:.1f}", "SHOCK MAE":"{:.1f}", "Δ MAE":"{:+.1f}",
                }), use_container_width=True, hide_index=True,
            )
        c1, c2 = st.columns(2)
        with c1:
            st.caption("FLAT · vea tüüp")
            st.dataframe(flat_a["mode"].style.format({"MAPE %":"{:.1f}%","MAE ABC":"{:.1f}","Bias %":"{:+.1f}%","Abs vea osa %":"{:.1f}%"}), use_container_width=True, hide_index=True)
            st.caption("FLAT · tegelik saagitase")
            st.dataframe(flat_a["level"].style.format({"MAPE %":"{:.1f}%","MAE ABC":"{:.1f}","Bias %":"{:+.1f}%","Abs vea osa %":"{:.1f}%"}), use_container_width=True, hide_index=True)
        with c2:
            st.caption("COMMON SHOCK · vea tüüp")
            st.dataframe(shock_a["mode"].style.format({"MAPE %":"{:.1f}%","MAE ABC":"{:.1f}","Bias %":"{:+.1f}%","Abs vea osa %":"{:.1f}%"}), use_container_width=True, hide_index=True)
            st.caption("COMMON SHOCK · tegelik saagitase")
            st.dataframe(shock_a["level"].style.format({"MAPE %":"{:.1f}%","MAE ABC":"{:.1f}","Bias %":"{:+.1f}%","Abs vea osa %":"{:.1f}%"}), use_container_width=True, hide_index=True)

    st.markdown("### 5. 4.75 p kasvuaeg · artefakti audit")
    gsum, gdetail = _growth_475_audit(field_df, complete_days)
    if gsum.empty:
        st.caption("4.75 p skooritavaid ridu pole.")
    else:
        st.dataframe(gsum.style.format({"MAPE %":"{:.1f}%","MAE ABC":"{:.1f}","Bias %":"{:+.1f}%"}), use_container_width=True, hide_index=True)
        with st.expander("Näita 4.75 p põld/sihtpäev detaile", expanded=False):
            st.dataframe(gdetail.head(30).style.format({
                "order":"{:.0f}", "growth":"{:.2f}", "actual_abc":"{:.1f}", "flat":"{:.1f}", "shock":"{:.1f}",
                "FLAT abs":"{:.1f}", "SHOCK abs":"{:.1f}",
            }), use_container_width=True, hide_index=True)

    st.markdown("### 6. COMMON SHOCK audit · mida issue-päev päriselt teadis")
    sa = field_df.groupby("issue_day", as_index=False).agg(
        shock=("common_shock_signal", "mean"), shock_day=("common_shock_day", "first"),
        actual=("shock_actual", "mean"), expected=("shock_expected", "mean"),
    ).sort_values("issue_day")
    sa["Issue"] = sa["issue_day"].map(lambda d: pd.Timestamp(d).strftime("%d.%m"))
    sa["Shock päev"] = sa["shock_day"].map(lambda d: "—" if pd.isna(d) else pd.Timestamp(d).strftime("%d.%m"))
    sa["Shock %"] = 100.0 * (np.exp(sa["shock"].astype(float)) - 1.0)
    st.dataframe(sa[["Issue", "Shock päev", "shock", "Shock %", "actual", "expected"]].tail(18).rename(columns={
        "shock":"Shock log", "actual":"Shock päeva tegelik ABC", "expected":"Range MEM10 ootus"
    }).style.format({
        "Shock log":"{:+.3f}", "Shock %":"{:+.0f}%", "Shock päeva tegelik ABC":"{:.1f}", "Range MEM10 ootus":"{:.1f}"
    }), use_container_width=True, hide_index=True)

    st.markdown("### 7. Viimased skooritavad sihtpäevad")
    latest_targets = sorted(daily["target_day"].unique())[-7:]
    latest = daily[daily["target_day"].isin(latest_targets)].copy().sort_values(["target_day", "lead"])
    latest["Issue"] = latest["issue_day"].map(lambda d: pd.Timestamp(d).strftime("%d.%m"))
    latest["Siht"] = latest["target_day"].map(lambda d: pd.Timestamp(d).strftime("%d.%m"))
    show_cols = ["Issue", "Siht", "lead", "actual_abc"] + MODEL_NAMES
    st.dataframe(
        latest[show_cols].rename(columns={"lead":"Lead", "actual_abc":"Tegelik ABC"}).style.format({
            "Tegelik ABC":"{:.1f}", **{name:"{:.1f}" for name in MODEL_NAMES}
        }), use_container_width=True, hide_index=True,
    )

    if errors:
        st.warning(f"ECMWF issue-vigu: {len(errors)}. Need päevad jäid replay'st välja.")
        with st.expander("Näita ECMWF vigu", expanded=False):
            st.dataframe(pd.DataFrame([{"Päev": d, "Viga": e} for d, e in errors]), use_container_width=True, hide_index=True)

    st.divider()
    st.caption(
        f"{LAB_VERSION} · MEM10 LAB-140 parity · COMMON SHOCK fixed weight=1.0 · shock freshness={SHOCK_FRESH_DAYS} p · "
        f"ridge α={RIDGE_ALPHA:g} · recency half-life={RECENCY_HALFLIFE_DAYS:g} p · READ ONLY"
    )


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        _self_test()
    else:
        main()
