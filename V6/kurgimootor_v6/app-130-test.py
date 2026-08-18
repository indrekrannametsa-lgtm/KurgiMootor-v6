from __future__ import annotations

"""
KurgiMootor LAB-148 · ASYMMETRIC TURN / PLANT-HEALTH DETECTOR
================================================================

Eesmärk
-------
LAB-147 kinnitas, et kolme horisondi arhitektuur parandab üldpilti, kuid suurimad
allapöörded jäävad endiselt halvasti tabatuks. Kasutaja bioloogiline hüpotees on,
et järsk allaminek on eelkõige taimetervise / taimiku seisundi probleem, mitte lihtsalt
veel üks ilmamuster.

LAB-148 EI otsi uusi ilmamuutujaid ega vali championit. Ta testib üht konkreetset
asümmeetrilist hüpoteesi:

1) Üles- ja tavarežiimi jaoks jääb LAB-147 fikseeritud 3H arhitektuur muutmata.
2) Allapoole riski jaoks arvutatakse issue-päeval ainult MINEVIKU korjetest latentne
   PLANT-HEALTH sensor: kui paljudel põldudel on viimane kasvupäevaga normaliseeritud
   ABC-rate langenud võrreldes sama põllu varasema rate'iga ning kui suur on mediaanlangus.
3) HEALTH flag on fikseeritud enne replay'd: vähemalt 6 värsket põldu, vähemalt 60%
   neist langevad >=8% ning üle-põldude mediaanlangus on >=8%.
4) Ainult HEALTH flag'i korral lubatakse ühepoolne HEALTH-DOWN korrigeeriv mudel.
   Positiivset "health boost'i" ei ole; sensor on teadlikult ainult allapoole.

See on DETEKTORI test. Kui latentne harvest-põhine tervisesensor ei näe allapööret enne
suurt forecast-missi, ei tõesta see, et taimetervis pole põhjus. See tähendab ainult, et
korjeajalugu ei anna tervise halvenemisest piisavalt varajast signaali; siis on vaja otsest
visuaalset/haigusseisundi sisendit.

READ ONLY: ainult A+B+C; DB-st ainult get_harvest_history ja get_weather_rows.
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
LAB_VERSION = "LAB-148-ASYM-HEALTH-TURN-V1"

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

BASE_FEATURES = ["w10_gdd10", "w10_rad", "w10_et0", "block1_signal"]
D3_CORE = ["d3_gdd10", "d3_rad", "d3_et0"]
D3_FULL = D3_CORE + ["d3_rh", "d3_rain", "d3_wind"]
D5_CORE = ["d5_gdd10", "d5_rad", "d5_et0"]
DAY0 = ["d0_gdd10", "d0_rad", "d0_et0", "d0_rh", "d0_rain", "d0_wind"]
AGE = ["production_age", "production_age_sq"]

# LAB-147 ei otsi enam variante. Need kolm alusmudelit + kontrollid on fikseeritud.
SHORT_NAME = "SHORT-D3CORE"
MID_NAME = "MID-DAY0"
ROUTED_NAME = "3H-ROUTED"
HEALTH_SHORT_NAME = "SHORT-D3CORE+HEALTH"
HEALTH_MID_NAME = "MID-DAY0+HEALTH"
HEALTH_LONG_NAME = "LONG-OWN2+HEALTH"
HEALTH_ROUTED_NAME = "3H-HEALTH-DOWN"
HEALTH_FEATURES = ["health_down_feature", "health_decline_feature"]
HEALTH_MAX_AGE_DAYS = 8
HEALTH_MIN_FIELDS = 6
HEALTH_FLAG_DOWN = 0.08
HEALTH_FLAG_SHARE = 0.60

VARIANTS: Dict[str, List[str]] = {
    "OWN2": [],
    "MEM10": ["w10_gdd10", "w10_rad", "w10_et0"],
    "FLAT": BASE_FEATURES,
    SHORT_NAME: BASE_FEATURES + D3_CORE,
    MID_NAME: BASE_FEATURES + DAY0,
    HEALTH_SHORT_NAME: BASE_FEATURES + D3_CORE + HEALTH_FEATURES,
    HEALTH_MID_NAME: BASE_FEATURES + DAY0 + HEALTH_FEATURES,
    HEALTH_LONG_NAME: HEALTH_FEATURES,
}
VARIANT_NAMES = list(VARIANTS.keys())
SCORE_NAMES = VARIANT_NAMES + [ROUTED_NAME, HEALTH_ROUTED_NAME]
MEMORY_WINDOWS = (10,)

def _route_name(lead: int) -> str:
    lead = int(lead)
    if 1 <= lead <= 4:
        return SHORT_NAME
    if 5 <= lead <= 7:
        return MID_NAME
    return "OWN2"


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


def _health_snapshot(known_events: List[Event], cutoff_exclusive: date) -> dict:
    """Issue-hetke latentne taimiku tervise/languse sensor ainult varasematest korjetest.

    Iga põllu jaoks võrreldakse viimast kasvupäevaga normaliseeritud ABC-rate'i
    sama põllu 1-2 eelmise rate'i mediaaniga. Kasutame ainult värskeid põlde.
    Positiivne `health_down` tähendab languse suurust; tõusu ei premeerita.
    """
    reference_day = cutoff_exclusive - timedelta(days=1)
    changes: List[float] = []
    for field in range(1, 15):
        rh = _rate_history(known_events, field, cutoff_exclusive)
        if len(rh) < 2:
            continue
        latest_ev, latest_rate = rh[-1]
        age = (reference_day - latest_ev.day).days
        if age < 0 or age > HEALTH_MAX_AGE_DAYS:
            continue
        prev_rates = [float(rate) for _ev, rate in rh[max(0, len(rh)-3):-1]]
        if not prev_rates:
            continue
        baseline = float(np.median(prev_rates))
        if baseline <= 0 or latest_rate <= 0:
            continue
        changes.append(float(math.log(latest_rate / baseline)))

    n = len(changes)
    if not changes:
        return {
            "health_median_log": 0.0,
            "health_down": 0.0,
            "health_decline_share": 0.0,
            "health_down_feature": 0.0,
            "health_decline_feature": 0.0,
            "health_fields_n": 0,
            "health_flag": 0,
        }
    arr = np.asarray(changes, dtype=float)
    med = float(np.median(arr))
    down = max(0.0, -med)
    decline_share = float(np.mean(arr <= -HEALTH_FLAG_DOWN))
    enough = n >= HEALTH_MIN_FIELDS
    flag = int(
        enough
        and down >= HEALTH_FLAG_DOWN
        and decline_share >= HEALTH_FLAG_SHARE
    )
    return {
        "health_median_log": med,
        "health_down": float(down),
        "health_decline_share": decline_share,
        # Learned HEALTH haru ei õpi 1-5 põllu hõredast varasest sensorist.
        "health_down_feature": float(down if enough else 0.0),
        "health_decline_feature": float(decline_share if enough else 0.0),
        "health_fields_n": int(n),
        "health_flag": flag,
    }


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
            required = [issue_day + timedelta(days=i) for i in range(1, 10)]
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
    rh = np.asarray([r["rh"] for r in rows], dtype=float)
    rain = np.asarray([r["rain"] for r in rows], dtype=float)
    wind = np.asarray([r["wind"] for r in rows], dtype=float)
    tmin = np.asarray([r["temp_min"] for r in rows], dtype=float)
    tmax = np.asarray([r["temp_max"] for r in rows], dtype=float)
    gdd10 = np.maximum(0.0, temp - 10.0)
    return {
        "gdd10": float(gdd10.mean()),
        "rad": float(rad.mean()),
        "et0": float(et0.mean()),
        "rh": float(rh.mean()),
        "rain": float(rain.mean()),
        "wind": float(wind.mean()),
        "tmin": float(tmin.min()),
        "tmax": float(tmax.max()),
        "n": len(rows),
    }


def _first_reliable_harvest_day(known_events: List[Event], field: int, target_day: date) -> Optional[date]:
    days = [e.day for e in known_events if e.field == field and e.day < target_day and e.reliable and e.abc is not None]
    return min(days) if days else None


def _delta_window(
    wmap: Dict[date, dict],
    target_day: date,
    days: int,
) -> Optional[dict]:
    recent_rows = _range_weather(
        wmap, target_day - timedelta(days=days), target_day - timedelta(days=1)
    )
    prev_rows = _range_weather(
        wmap, target_day - timedelta(days=2*days), target_day - timedelta(days=days+1)
    )
    recent = _agg_weather(recent_rows) if recent_rows is not None else None
    prev = _agg_weather(prev_rows) if prev_rows is not None else None
    if recent is None or prev is None:
        return None
    return {
        "gdd10": recent["gdd10"] - prev["gdd10"],
        "rad": recent["rad"] - prev["rad"],
        "et0": recent["et0"] - prev["et0"],
        "rh": recent["rh"] - prev["rh"],
        "rain": recent["rain"] - prev["rain"],
        "wind": recent["wind"] - prev["wind"],
    }


def _memory_features(
    known_events: List[Event],
    schedule_events: List[Event],
    field: int,
    target_day: date,
    target_order: int,
    wmap: Dict[date, dict],
) -> dict:
    """Senine MEM10 + pöörde-eelsed exploratory sensorid.

    Kõik aknad lõpevad target-1, välja arvatud DAY0, mis kasutab target-päeva
    forecasti. Vanus on päevad esimesest issue-hetkeks teada olevast usaldusväärsest
    sama põllu korjest; see on ainult andmepõhine bioloogilise kella proxy.
    """
    growth, _prev = _growth_for_target(schedule_events, field, target_day, target_order)
    out = {"growth_days": float(growth)}

    rows10 = _range_weather(wmap, target_day - timedelta(days=10), target_day - timedelta(days=1))
    mem10 = _agg_weather(rows10) if rows10 is not None else None
    for suffix in ("gdd10", "rad", "et0"):
        out[f"w10_{suffix}"] = None if mem10 is None else float(mem10[suffix])

    for days in (3, 5):
        delta = _delta_window(wmap, target_day, days)
        for suffix in ("gdd10", "rad", "et0", "rh", "rain", "wind"):
            out[f"d{days}_{suffix}"] = None if delta is None else float(delta[suffix])

    day0_row = wmap.get(target_day)
    if day0_row is None:
        for suffix in ("gdd10", "rad", "et0", "rh", "rain", "wind"):
            out[f"d0_{suffix}"] = None
    else:
        out["d0_gdd10"] = max(0.0, float(day0_row["temp"]) - 10.0)
        out["d0_rad"] = float(day0_row["rad"])
        out["d0_et0"] = float(day0_row["et0"])
        out["d0_rh"] = float(day0_row["rh"])
        out["d0_rain"] = float(day0_row["rain"])
        out["d0_wind"] = float(day0_row["wind"])

    first_day = _first_reliable_harvest_day(known_events, field, target_day)
    if first_day is None:
        out["production_age"] = None
        out["production_age_sq"] = None
    else:
        age = max(0.0, float((target_day - first_day).days))
        out["production_age"] = age
        out["production_age_sq"] = (age / 30.0) ** 2

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
    frozen_health: Optional[dict] = None,
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

    wf = _memory_features(known_events, schedule_events, field, target_day, target_order, wmap)

    if block_cutoff is None:
        # Historical target: info strictly before target_day.
        block_cutoff = target_day
    block_sig, block_day = _block1_signal(known_events, block_cutoff, own2_rate)
    health = dict(frozen_health) if frozen_health is not None else _health_snapshot(known_events, block_cutoff)

    rec = {
        "target_day": target_day,
        "field_no": field,
        "own2_rate": float(own2_rate),
        "used_fallback": int(used_fallback),
        "block1_signal": float(block_sig),
        "block1_day": block_day,
        **health,
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
# ÜKS RIDGE RAAMISTIK, ERINEVAD FEATURE-LADDERID
# -----------------------------------------------------------------------------

def _fit_model(train_records: List[dict], cutoff_day: date, feature_names: List[str]) -> Optional[dict]:
    if not feature_names:
        return {"persistence": True, "n": len(train_records)}

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
    # OWN2 on baasankur: intercept on teadlikult 0, et iga aste testiks ainult
    # feature'i kõrvalekalde lisainfot, mitte üldist taseme ümberkalibreerimist.
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
    return {"persistence": False, "mu": mu, "sd": sd, "beta": beta, "bounds": bounds, "n": len(clean)}


def _predict_variant(model: Optional[dict], rec: dict, feature_names: List[str]) -> Optional[float]:
    own2_rate = float(rec["own2_rate"])
    growth = max(0.5, float(rec["growth_days"]))
    if not feature_names:
        return own2_rate * growth
    if model is None or model.get("persistence"):
        return own2_rate * growth
    vals = [_f(rec.get(c)) for c in feature_names]
    if any(v is None for v in vals):
        return None
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
    models = {name: _fit_model(train, issue_day, feats) for name, feats in VARIANTS.items()}
    stubs = _future_schedule_stubs(actual_lookup, issue_day)

    # OWN2/BLOCK1 ja HEALTH sensor külmutatakse issue-päeval. HEALTH on kogu ploki ühine.
    frozen_own2: Dict[int, Tuple[float, bool]] = {}
    frozen_health = _health_snapshot(known_events, issue_day + timedelta(days=1))
    out: List[dict] = []
    for target_day, field, order in stubs:
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
            frozen_health=frozen_health,
        )
        if rec is None:
            continue

        actual_row = actual_lookup.get((target_day, field))
        actual_abc = float(actual_row["_abc"]) if actual_row is not None else np.nan
        row = {
            "issue_day": issue_day,
            "target_day": target_day,
            "lead": int((target_day - issue_day).days),
            "field": field,
            "order": order,
            "growth_days": float(rec["growth_days"]),
            "actual_abc": actual_abc,
            "fallback": int(rec["used_fallback"]),
            "block1_signal": float(rec["block1_signal"]),
            "health_median_log": float(rec.get("health_median_log", 0.0)),
            "health_down": float(rec.get("health_down", 0.0)),
            "health_decline_share": float(rec.get("health_decline_share", 0.0)),
            "health_fields_n": int(rec.get("health_fields_n", 0)),
            "health_flag": int(rec.get("health_flag", 0)),
            "run": run_label,
        }
        audit_feature_keys = sorted(set(sum(VARIANTS.values(), [])))
        for key in audit_feature_keys:
            val = _f(rec.get(key))
            row[key] = np.nan if val is None else float(val)
        # Search-kandidaadi learned kiht ei tohi puuduvate tunnuste / vähese treeningu
        # korral muutuda NaN->0 päevakoondiks ega kukkuda OWN2 peale. Kandidandid on
        # defineeritud FLAT baasi lisana, seega fallback on täpselt FLAT.
        flat_pred = _predict_variant(models["FLAT"], rec, VARIANTS["FLAT"])
        if flat_pred is None:
            flat_pred = _predict_variant(models["MEM10"], rec, VARIANTS["MEM10"])
        if flat_pred is None:
            flat_pred = _predict_variant(models["OWN2"], rec, VARIANTS["OWN2"])
        for name, feats in VARIANTS.items():
            if name in {"OWN2", "MEM10", "FLAT"}:
                pred = _predict_variant(models[name], rec, feats)
            else:
                vals_ok = all(_f(rec.get(c)) is not None for c in feats)
                pred = _predict_variant(models[name], rec, feats) if (models[name] is not None and vals_ok) else flat_pred
            row[name] = float(flat_pred if pred is None else pred)
            row[f"{name}_train_n"] = int(models[name]["n"]) if models[name] is not None else 0

        route = _route_name(row["lead"])
        row["route"] = route
        row[ROUTED_NAME] = float(row[route])
        row[f"{ROUTED_NAME}_train_n"] = int(row.get(f"{route}_train_n", 0))

        if int(row.get("health_flag", 0)):
            if 1 <= int(row["lead"]) <= 4:
                hroute = HEALTH_SHORT_NAME
            elif 5 <= int(row["lead"]) <= 7:
                hroute = HEALTH_MID_NAME
            else:
                hroute = HEALTH_LONG_NAME
        else:
            hroute = route
        row["health_route"] = hroute
        row[HEALTH_ROUTED_NAME] = float(row[hroute])
        row[f"{HEALTH_ROUTED_NAME}_train_n"] = int(row.get(f"{hroute}_train_n", 0))
        out.append(row)

        # Ainult ajamärk, mitte prognoositud saak/state.
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
        "health_down": ("health_down", "first"),
        "health_decline_share": ("health_decline_share", "first"),
        "health_fields_n": ("health_fields_n", "first"),
        "health_flag": ("health_flag", "first"),
    }
    for name in SCORE_NAMES:
        agg[name] = (name, "sum")
    daily = f.groupby(["issue_day", "target_day", "lead"], as_index=False).agg(**agg)
    daily = daily[daily["n_fields"] == EXPECTED_FIELDS_PER_DAY].copy()
    if daily.empty:
        return daily
    daily["fallback_pct"] = 100.0 * daily["fallback_fields"] / EXPECTED_FIELDS_PER_DAY
    for name in SCORE_NAMES:
        daily[f"{name}_err"] = daily[name] - daily["actual_abc"]
        daily[f"{name}_ape"] = daily[f"{name}_err"].abs() / daily["actual_abc"].clip(lower=0.1)
        daily[f"{name}_bias"] = 100.0 * daily[f"{name}_err"] / daily["actual_abc"].clip(lower=0.1)
    return daily


def _summary_table(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in SCORE_NAMES:
        rows.append({
            "Variant": name,
            "MAPE %": 100.0 * float(daily[f"{name}_ape"].mean()),
            "±20% sees %": 100.0 * float((daily[f"{name}_ape"] <= 0.20).mean()),
            "Bias %": float(daily[f"{name}_bias"].mean()),
            "MAE ABC": float(daily[f"{name}_err"].abs().mean()),
            "N": int(len(daily)),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    base = float(df.loc[df["Variant"] == "OWN2", "MAPE %"].iloc[0])
    df["Δ MAPE vs OWN2"] = df["MAPE %"] - base
    return df

def _lead_table(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lead, g in daily.groupby("lead"):
        lead_i = int(lead)
        row = {
            "Lead": lead_i,
            "N": int(len(g)),
            "Fallback %": float(g["fallback_pct"].mean()),
            "Route": _route_name(lead_i),
        }
        for name in ("OWN2", "FLAT", SHORT_NAME, MID_NAME, ROUTED_NAME):
            row[f"{name} MAPE %"] = 100.0 * float(g[f"{name}_ape"].mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("Lead")


def _bucket_table(daily: pd.DataFrame) -> pd.DataFrame:
    buckets = [
        ("SHORT · +1…+4", 1, 4, SHORT_NAME),
        ("MID · +5…+7", 5, 7, MID_NAME),
        ("LONG · +8…+9", 8, 9, "OWN2"),
    ]
    rows = []
    for label, lo, hi, route in buckets:
        g = daily[(daily["lead"] >= lo) & (daily["lead"] <= hi)]
        if g.empty:
            continue
        flat_mape = 100.0 * float(g["FLAT_ape"].mean())
        routed_mape = 100.0 * float(g[f"{ROUTED_NAME}_ape"].mean())
        rows.append({
            "Horisont": label,
            "Route": route,
            "N": int(len(g)),
            "FLAT MAPE %": flat_mape,
            "3H MAPE %": routed_mape,
            "Δ vs FLAT": routed_mape - flat_mape,
            "3H ±20% sees %": 100.0 * float((g[f"{ROUTED_NAME}_ape"] <= 0.20).mean()),
            "3H Bias %": float(g[f"{ROUTED_NAME}_bias"].mean()),
        })
    return pd.DataFrame(rows)

def _window_summary(daily: pd.DataFrame, start_day: date) -> pd.DataFrame:
    g = daily[daily["target_day"] >= start_day].copy()
    return _summary_table(g) if not g.empty else pd.DataFrame()


# -----------------------------------------------------------------------------
# SELF TEST
# -----------------------------------------------------------------------------

def _self_test() -> None:
    # Piisavalt pikk sünteetiline ilm, et MEM10, delta-aknad ja DAY0 saaksid treenida.
    measured: Dict[date, dict] = {}
    start = date(2026, 7, 1)
    for i in range(75):
        dd = start + timedelta(days=i)
        measured[dd] = {
            "day": dd, "temp": 18.0 + 0.025*i, "temp_min": 12.0, "temp_max": 25.0,
            "wind": 2.0, "rad": 17.0 + 0.07*i, "rh": 75.0, "rain": 0.4,
            "et0": 2.8 + 0.008*i, "source": "measured",
        }

    rows = []
    days = [date(2026,7,5) + timedelta(days=4*j) for j in range(11)]
    for j, dd in enumerate(days):
        for order, f in enumerate((1, 2, 3), start=1):
            abc = 6.5 + 0.32*j + 0.12*f
            rows.append({
                "harvest_date": dd.isoformat(), "field_no": f, "harvest_order": order,
                "a": 0.2, "b": abc*0.45, "c": abc*0.55-0.2, "xl": 1.0,
                "data_quality": "Kinnitatud",
            })
    events, lookup = _prepare_events(rows)
    assert len(lookup) == 33
    hist = _make_historical_records(events, lookup, measured)
    assert hist and all("y" in r for r in hist)

    # Viimase historical rea peal peavad kõik variandid olema numbriliselt ennustatavad.
    rec0 = hist[-1]
    train = [r for r in hist if r["target_day"] < rec0["target_day"]]
    for name, feats in VARIANTS.items():
        model = _fit_model(train, rec0["target_day"], feats)
        pred = _predict_variant(model, rec0, feats)
        assert pred is not None and math.isfinite(pred) and pred >= 0, name

    # E2E issue: actualid lookupis on ainult skooriks; recordisse neid ei anta.
    issue = date(2026, 8, 2)
    fc = {d: dict(v, source="ecmwf_forecast") for d, v in measured.items() if d > issue}
    out = _predict_issue(
        issue_day=issue, all_events=events, actual_lookup=lookup, measured=measured,
        historical_records=hist, forecast_map=fc, run_label="synthetic",
    )
    assert out and all(all(name in r for name in SCORE_NAMES) for r in out)
    assert all(r["route"] == _route_name(r["lead"]) for r in out)
    assert all(abs(float(r[ROUTED_NAME]) - float(r[r["route"]])) < 1e-12 for r in out)
    assert all("health_flag" in r and "health_down" in r for r in out)
    assert all(abs(float(r[HEALTH_ROUTED_NAME]) - float(r[r["health_route"]])) < 1e-12 for r in out)
    assert all(math.isfinite(float(r["growth_days"])) for r in out)
    hs = _health_snapshot([e for e in events if e.day <= issue], issue + timedelta(days=1))
    assert 0.0 <= float(hs["health_decline_share"]) <= 1.0
    assert float(hs["health_down"]) >= 0.0
    print("LAB-148 SELF-TEST OK")

# -----------------------------------------------------------------------------
# STREAMLIT
# -----------------------------------------------------------------------------

def _bad_day_compare(daily: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[date]]:
    """Top-5 halvimad FLAT sihtpäevad defineeritakse ainult baasi vea järgi.

    See grupp on audit, mitte ühegi kandidaadi sisend ega training filter.
    """
    if daily.empty:
        return pd.DataFrame(), pd.DataFrame(), []
    base = daily.copy()
    base["_flat_abs"] = (base["FLAT"] - base["actual_abc"]).abs()
    by_target = base.groupby("target_day", as_index=False).agg(abs_err=("_flat_abs", "sum"))
    bad_days = list(by_target.sort_values("abs_err", ascending=False).head(5)["target_day"])
    bad = daily[daily["target_day"].isin(bad_days)].copy()
    rest = daily[~daily["target_day"].isin(bad_days)].copy()
    return _summary_table(bad), _summary_table(rest), bad_days


def _level_bias_table(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()
    # Tertiilid ainult auditiks; tegelik saak EI ole ühegi mudeli sisend.
    x = daily.copy()
    try:
        x["Tase"] = pd.qcut(x["actual_abc"], 3, labels=["Madal", "Keskmine", "Kõrge"], duplicates="drop")
    except Exception:
        return pd.DataFrame()
    rows = []
    for level, g in x.groupby("Tase", observed=False):
        row = {"Tase": str(level), "N": int(len(g))}
        for name in SCORE_NAMES:
            row[f"{name} bias %"] = float(g[f"{name}_bias"].mean())
            row[f"{name} MAPE %"] = 100.0 * float(g[f"{name}_ape"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    import streamlit as st
    import db

    st.set_page_config(page_title="KurgiMootor LAB-148", layout="wide")
    st.error("🧪 LAB-148 · ASYMMETRIC TURN / PLANT-HEALTH DETECTOR · READ-ONLY")
    st.title("KurgiMootor · kas allapööre on taimetervise signaalina varem nähtav?")
    st.caption(
        "Üks fikseeritud arhitektuur: +1…+4 SHORT = FLAT+D3CORE; +5…+7 MID = FLAT+DAY0; "
        "+8…+9 LONG = OWN2. LAB ei vali lead'i kaupa võitjat ega otsi uusi tunnuseid."
    )
    st.warning(
        "Aususe märkus: 3H baas pärineb LAB-147-st. HEALTH detectori piirid on enne replay'd fikseeritud: "
        ">=6 värsket põldu, >=60% langevad vähemalt 8% ja mediaanlangus >=8%. LAB neid piire tulemuse järgi ümber ei vali."
    )

    with st.expander("Fikseeritud HEALTH-DOWN hüpotees", expanded=False):
        st.markdown(
            """
- Kontrollbaas on muutmata **LAB-147 3H**: SHORT 1–4 = D3CORE, MID 5–7 = DAY0, LONG 8–9 = OWN2.
- Iga issue-päev arvutame kõigi värskete põldude **ABC / kasvupäev** rate'i muutuse sama põllu varasema taseme suhtes.
- **HEALTH flag**: vähemalt 6 värsket põldu; vähemalt 60% neist on langenud >=8%; mediaanlangus >=8%.
- Kui flag puudub, `3H-HEALTH-DOWN` = täpselt tavaline `3H-ROUTED`.
- Kui flag on olemas, kasutatakse samas horisondis mudelit, millel on lisaks ainult kaks ühepoolset health-tunnust: languse mediaan ja languse levik põldudel.
- Positiivset health-boost'i pole. Tuleviku saaki state'iks tagasi ei söödetata. Ainult **A+B+C**.
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
        st.session_state.pop("lab148_results", None)
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

    run_requested = st.button("▶ Jooksuta HEALTH TURN replay", type="primary")
    if not run_requested and "lab148_results" not in st.session_state:
        st.info("Vajuta üks kord. ECMWF issue-jooksud on 24 h cache'is; productionit see ei puuduta.")
        st.stop()

    if run_requested:
        all_rows: List[dict] = []
        errors: List[Tuple[date, str]] = []
        runs: Dict[date, str] = {}
        progress = st.progress(0.0, text="ECMWF + HEALTH TURN replay…")
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
        st.session_state["lab148_results"] = {
            "field": field_df, "daily": daily, "errors": errors, "runs": runs,
        }

    result = st.session_state.get("lab148_results")
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
    buckets = _bucket_table(daily)
    bad, rest, bad_days = _bad_day_compare(daily)

    flat_row = overall.loc[overall["Variant"] == "FLAT"].iloc[0]
    routed_row = overall.loc[overall["Variant"] == ROUTED_NAME].iloc[0]
    health_row = overall.loc[overall["Variant"] == HEALTH_ROUTED_NAME].iloc[0]

    down_miss = daily[f"{ROUTED_NAME}_bias"] > 20.0
    flagged = daily["health_flag"] > 0
    tp = int((down_miss & flagged).sum())
    fp = int((~down_miss & flagged).sum())
    fn = int((down_miss & ~flagged).sum())
    precision = 100.0 * tp / max(1, tp + fp)
    recall = 100.0 * tp / max(1, tp + fn)

    t1, t2, t3, t4, t5 = st.columns(5)
    t1.metric("3H baas · MAPE", f"{float(routed_row['MAPE %']):.1f}%")
    t2.metric(
        "3H + HEALTH-DOWN", f"{float(health_row['MAPE %']):.1f}%",
        f"{float(health_row['MAPE %'] - routed_row['MAPE %']):+.1f} pp vs 3H"
    )
    t3.metric("HEALTH flag ridu", f"{int(flagged.sum())}/{len(daily)}")
    t4.metric("Down-miss recall", f"{recall:.0f}%")
    t5.metric("Flag precision", f"{precision:.0f}%")

    st.markdown("### 1. Kas ühepoolne HEALTH-DOWN haru parandab 3H baasi?")
    show_order = ["OWN2", "FLAT", ROUTED_NAME, HEALTH_SHORT_NAME, HEALTH_MID_NAME, HEALTH_LONG_NAME, HEALTH_ROUTED_NAME]
    overall_show = overall.set_index("Variant").loc[show_order].reset_index()
    overall_show["Δ MAPE vs 3H"] = overall_show["MAPE %"] - float(routed_row["MAPE %"])
    st.dataframe(
        overall_show.style.format({
            "MAPE %": "{:.1f}%", "±20% sees %": "{:.0f}%", "Bias %": "{:+.1f}%",
            "MAE ABC": "{:.1f}", "Δ MAPE vs OWN2": lambda x: f"{x:+.1f} pp",
            "Δ MAPE vs 3H": lambda x: f"{x:+.1f} pp",
        }),
        use_container_width=True, hide_index=True,
    )

    st.markdown("### 2. Detector audit · kas flag näeb ette, et 3H jääb liiga kõrgeks?")
    st.caption("Down-miss = 3H prognoos on tegelikust ABC-st >20% kõrgem. See on ainult auditi siht, mitte detectori sisend.")
    det_rows = []
    for label, mask in [
        ("Kõik", pd.Series(True, index=daily.index)),
        ("SHORT +1…+4", daily["lead"].between(1,4)),
        ("MID +5…+7", daily["lead"].between(5,7)),
        ("LONG +8…+9", daily["lead"].between(8,9)),
    ]:
        g = daily[mask].copy()
        if g.empty:
            continue
        dm = g[f"{ROUTED_NAME}_bias"] > 20.0
        fl = g["health_flag"] > 0
        tpi = int((dm & fl).sum()); fpi = int((~dm & fl).sum()); fni = int((dm & ~fl).sum())
        det_rows.append({
            "Horisont": label, "N": len(g), "HEALTH flag %": 100.0*float(fl.mean()),
            "Down-miss %": 100.0*float(dm.mean()),
            "Recall %": 100.0*tpi/max(1,tpi+fni), "Precision %": 100.0*tpi/max(1,tpi+fpi),
            "3H MAPE %": 100.0*float(g[f"{ROUTED_NAME}_ape"].mean()),
            "HEALTH MAPE %": 100.0*float(g[f"{HEALTH_ROUTED_NAME}_ape"].mean()),
        })
    det_df = pd.DataFrame(det_rows)
    st.dataframe(det_df.style.format({
        "HEALTH flag %":"{:.0f}%", "Down-miss %":"{:.0f}%", "Recall %":"{:.0f}%", "Precision %":"{:.0f}%",
        "3H MAPE %":"{:.1f}%", "HEALTH MAPE %":"{:.1f}%",
    }), use_container_width=True, hide_index=True)

    st.markdown("### 3. Issue-päeva latentne taimiku tervis · kuupäeviti")
    issue_health = daily.groupby("issue_day", as_index=False).agg(
        health_down=("health_down","first"), health_decline_share=("health_decline_share","first"),
        health_fields_n=("health_fields_n","first"), health_flag=("health_flag","first"),
    ).sort_values("issue_day")
    issue_health["Issue"] = issue_health["issue_day"].map(lambda d: pd.Timestamp(d).strftime("%d.%m"))
    issue_health["Mediaanlangus %"] = 100.0*(1.0-np.exp(-issue_health["health_down"]))
    issue_health["Langevaid põlde %"] = 100.0*issue_health["health_decline_share"]
    st.dataframe(issue_health[["Issue","Mediaanlangus %","Langevaid põlde %","health_fields_n","health_flag"]].rename(columns={
        "health_fields_n":"Värskeid põlde", "health_flag":"HEALTH flag"
    }).style.format({"Mediaanlangus %":"{:.1f}%","Langevaid põlde %":"{:.0f}%"}), use_container_width=True, hide_index=True)

    st.markdown("### 4. 3H suurimad allamissid · kas HEALTH flag oli enne olemas?")
    audit = daily.copy()
    audit["3H kõrge viga %"] = audit[f"{ROUTED_NAME}_bias"]
    worst_down = audit.sort_values("3H kõrge viga %", ascending=False).head(18).copy()
    worst_down["Issue"] = worst_down["issue_day"].map(lambda d: pd.Timestamp(d).strftime("%d.%m"))
    worst_down["Siht"] = worst_down["target_day"].map(lambda d: pd.Timestamp(d).strftime("%d.%m"))
    worst_down["Mediaanlangus %"] = 100.0*(1.0-np.exp(-worst_down["health_down"]))
    worst_down["Langevaid põlde %"] = 100.0*worst_down["health_decline_share"]
    st.dataframe(worst_down[["Issue","Siht","lead","actual_abc",ROUTED_NAME,HEALTH_ROUTED_NAME,"3H kõrge viga %","Mediaanlangus %","Langevaid põlde %","health_fields_n","health_flag"]].rename(columns={
        "lead":"Lead","actual_abc":"Tegelik ABC",ROUTED_NAME:"3H ABC",HEALTH_ROUTED_NAME:"HEALTH ABC","health_fields_n":"Värskeid põlde","health_flag":"Flag"
    }).style.format({"Tegelik ABC":"{:.1f}","3H ABC":"{:.1f}","HEALTH ABC":"{:.1f}","3H kõrge viga %":"{:+.1f}%","Mediaanlangus %":"{:.1f}%","Langevaid põlde %":"{:.0f}%"}), use_container_width=True, hide_index=True)

    st.markdown("### 5. Alates 01.08 · küpsem state")
    if not warm.empty:
        warm_show = warm[warm["Variant"].isin(["OWN2", "FLAT", ROUTED_NAME, HEALTH_ROUTED_NAME])].copy()
        st.dataframe(
            warm_show.style.format({
                "MAPE %": "{:.1f}%", "±20% sees %": "{:.0f}%", "Bias %": "{:+.1f}%",
                "MAE ABC": "{:.1f}", "Δ MAPE vs OWN2": lambda x: f"{x:+.1f} pp",
            }), use_container_width=True, hide_index=True,
        )

    st.markdown("### 6. Viimased skooritavad sihtpäevad")
    latest_targets = sorted(daily["target_day"].unique())[-7:]
    latest = daily[daily["target_day"].isin(latest_targets)].copy().sort_values(["target_day", "lead"])
    latest["Issue"] = latest["issue_day"].map(lambda d: pd.Timestamp(d).strftime("%d.%m"))
    latest["Siht"] = latest["target_day"].map(lambda d: pd.Timestamp(d).strftime("%d.%m"))
    latest["Route"] = latest["lead"].map(_route_name)
    show_cols = ["Issue", "Siht", "lead", "Route", "actual_abc", ROUTED_NAME, HEALTH_ROUTED_NAME, "health_down", "health_decline_share", "health_fields_n", "health_flag"]
    st.dataframe(
        latest[show_cols].rename(columns={"lead": "Lead", "actual_abc": "Tegelik ABC"}).style.format({
            "Tegelik ABC": "{:.1f}", ROUTED_NAME: "{:.1f}", HEALTH_ROUTED_NAME: "{:.1f}",
            "health_down": "{:.3f}", "health_decline_share": "{:.0%}",
        }), use_container_width=True, hide_index=True,
    )

    if errors:
        st.warning(f"ECMWF issue-vigu: {len(errors)}. Need päevad jäid replay'st välja.")
        with st.expander("Näita ECMWF vigu", expanded=False):
            st.dataframe(pd.DataFrame([{"Päev": d, "Viga": e} for d, e in errors]), use_container_width=True, hide_index=True)

    st.divider()
    st.caption(
        f"{LAB_VERSION} · Ridge α={RIDGE_ALPHA:g} · recency half-life={RECENCY_HALFLIFE_DAYS:g} p · "
        f"small-sample prior={SMALL_SAMPLE_PRIOR_ROWS:g} · fixed 3H + one-sided HEALTH-DOWN detector · READ ONLY"
    )


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        _self_test()
    else:
        main()
