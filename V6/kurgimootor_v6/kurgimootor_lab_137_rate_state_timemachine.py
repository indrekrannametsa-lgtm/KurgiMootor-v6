from __future__ import annotations

"""
KurgiMootor LAB-137 RATE-STATE AJAMASIN
=================================

Eesmärk
-------
Üks fikseeritud, read-only A+B+C (ABC) saagimootor, mida jooksutatakse ajas tagasi
alates 22.07.2026. Iga replay-päeva õhtul tehakse +1 ... +9 päeva prognoos.

Aususe reeglid
--------------
1) Prognoosipäev D näeb ainult D lõpuks teada olnud KINNITATUD/TULETATUD korje-ABC-d.
2) Mõõdetud ilm on lubatud ainult kuni päevani D.
3) Päevad D+1 ... D+9 saavad ilma ECMWF IFS HRES 9 km arhiveeritud üksikjooksust,
   eelistatult D 12 UTC jooksust (õhtuks kättesaadav); vajadusel 06/00 UTC fallback.
4) Tuleviku tegelikust korjereast kasutatakse replay-plaanina ainult kuupäeva, põldu
   ja korjejärjekorda. A/B/C/XL/total ei anta ennustajale.
5) Sama issue-päeva +1 ... +9 prognoosi jooksul mudeli koefitsiente ei treenita ümber.
   Tulevased ennustatud sama põllu korjed võivad olla järgmise tulevase korje state.
6) Hinnanguline/ligikaudne saak ei ole õppimise siht ega state-ABC sensor, kuid selle
   korjesündmus võib anda teada, et põld sel päeval korjati.
7) LAB ei kirjuta andmebaasi ega puuduta production-mootorit.

Mudel
-----
RATE-STATE + ridge transition:
- latentne state = A+B+C tootmiskiirus (ABC / täpne kasvupäev), mitte eelmise korje kastide arv;
- sama põllu viimane teadaolev rate on state-anchor; puuduliku ajaloo korral kasutatakse
  värsket kogu ploki robustset rate-taset;
- mudeli target = log(rate_actual / rate_anchor);
- sisendid on teadlikult väikesed: state-tase, sama põllu rate-trend, täpne kasvuaeg,
  kasvuperioodi GDD/radiatsioon/ET0/RH, 14 p radiatsioonimälu, ploki rate-režiim ja õitsemisvanus;
- +1...+9 sees ei muudeta prognoositud ABC-d uueks sensoriks. Tulevase korje järel
  kandub edasi ainult latentne prognoositud rate; järgmise korje ABC = rate × uus kasvuaeg.

Õitsemise telg
--------------
Täpsed õitsemiskuupäevad pole DB-s. Et feature oleks olemas ilma tuleviku saaki
vaatamata, kasutame ainult teada olevat järjestust/bioloogilise vanuse vahet:
Põld 1 = 14.06.2026 ja iga järgmine põld +1 päev. See annab 1...14 vahel 13 päeva
vanusevahe. Kuna mudelis on ainult lineaarne "päevi õitsemisest" feature, on selle
absoluutse nullpunkti mõju peamiselt interceptis; tähtis on põldude suhteline vanus.
Kui päris õitsemiskuupäevad on olemas, muuda ainult FLOWERING_STARTS kaarti.
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
# FIKSEERITUD LAB KONFIGURATSIOON
# -----------------------------------------------------------------------------

TZ = ZoneInfo("Europe/Tallinn")
TODAY = datetime.now(TZ).date()
LAB_VERSION = "LAB-137-RATE-STATE-TIMEMACHINE-V1"

REPLAY_START = date(2026, 7, 22)
WEATHER_START = date(2026, 7, 1)
SEASON_START = date(2026, 6, 15)

FORECAST_LAT = 58.1275
FORECAST_LON = 24.49167
FORECAST_MODEL = "ecmwf_ifs"  # Open-Meteo model id: ECMWF IFS HRES 9 km
FORECAST_RUN_HOURS_UTC = (12, 6, 0)  # kõik on issue-päeva jooksul kättesaadavaks saavad jooksud
FORECAST_DAYS = 10  # issue-päev + järgmised 9 kalendripäeva

RIDGE_ALPHA = 20.0
RECENCY_HALFLIFE_DAYS = 24.0
MIN_TRAIN_ROWS = 6
SMALL_SAMPLE_PRIOR_ROWS = 12.0  # cold-start: learned transition shrinks toward state persistence
TARGET_EPS = 0.25
DEFAULT_GROWTH_DAYS = 14.0 / 3.0  # 3 põldu päevas, 14 põldu => 4.667 päeva
POOL_LOOKBACK_DAYS = 10

FLOWERING_STARTS: Dict[int, date] = {
    f: date(2026, 6, 14) + timedelta(days=f - 1) for f in range(1, 15)
}

MEASURED_REQUIRED = (
    "temp_night_avg_c",
    "temp_day_avg_c",
    "temp_min_c",
    "temp_max_c",
    "wind_avg_ms",
    "radiation_mj_m2",
    "humidity_avg_pct",
    "precipitation_mm",
    "et0_mm",
)

# Üks mudelipere. Ei mingit champion/challenger valikut replay tulemuse järgi.
# Teadlikult väike feature-set: vähe ridu => vähe vabalt õpitavaid seoseid.
FEATURES = [
    "log_anchor_rate",
    "same_rate_trend",
    "has_rate_trend",
    "growth_days",
    "gw_gdd10",
    "gw_rad",
    "gw_et0",
    "gw_rh",
    "w14_rad",
    "regime3",
    "flower_age_days",
]


# -----------------------------------------------------------------------------
# ABIFUNKTSIOONID
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
    return float(sum(vals))


def _quality_reliable(row: dict) -> bool:
    q = str(row.get("data_quality") or row.get("quality") or "").strip().lower()
    return q not in {"hinnanguline", "ligikaudne", "estimated", "approximate"}


def _safe_log_ratio(a: float, b: float) -> float:
    return math.log((max(0.0, a) + TARGET_EPS) / (max(0.0, b) + TARGET_EPS))


@dataclass
class Event:
    day: date
    field: int
    order: int
    abc: Optional[float]
    reliable: bool
    source: str  # actual / estimated_event / simulated


def _event_sort_key(e: Event) -> Tuple[date, int, int]:
    return (e.day, e.order, e.field)


def _copy_events(events: Iterable[Event]) -> List[Event]:
    return [Event(e.day, e.field, e.order, e.abc, e.reliable, e.source) for e in events]


# -----------------------------------------------------------------------------
# KORJEANDMED / RATE-STATE
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
        if abc is not None and abc <= 0:
            abc = None

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


def _field_events_before(events: List[Event], field: int, target_day: date) -> List[Event]:
    return sorted(
        [e for e in events if e.field == field and e.day < target_day],
        key=_event_sort_key,
    )


def _growth_between(prev: Event, cur_day: date, cur_order: int) -> float:
    g = float((cur_day - prev.day).days) + (cur_order - prev.order) * (3.0 / 24.0)
    return max(0.5, g)


def _growth_info(events: List[Event], field: int, target_day: date, target_order: int) -> dict:
    hist = _field_events_before(events, field, target_day)
    prev = hist[-1] if hist else None
    prev2 = hist[-2] if len(hist) >= 2 else None
    if prev is not None:
        growth = _growth_between(prev, target_day, target_order)
        known = 1.0
    else:
        growth = DEFAULT_GROWTH_DAYS
        known = 0.0
    if prev is not None and prev2 is not None:
        prev_growth = _growth_between(prev2, prev.day, prev.order)
    else:
        prev_growth = growth
    return {
        "growth": max(0.5, float(growth)),
        "prev_growth": max(0.5, float(prev_growth)),
        "known": known,
        "prev_event": prev,
        "prev2_event": prev2,
    }


def _field_rate_history(events: List[Event], field: int, target_day: date) -> List[Tuple[Event, float]]:
    """Usaldusväärsed ABC/päev rate'id; rate vajab eelmist korjesündmust kui ajamärki."""
    hist = _field_events_before(events, field, target_day)
    out: List[Tuple[Event, float]] = []
    for i in range(1, len(hist)):
        prev, cur = hist[i - 1], hist[i]
        if not (cur.reliable and cur.abc is not None and cur.abc > 0):
            continue
        growth = _growth_between(prev, cur.day, cur.order)
        out.append((cur, float(cur.abc) / growth))
    return out


def _pool_rate_state(events: List[Event], target_day: date) -> Optional[Tuple[float, float]]:
    vals: List[float] = []
    ages: List[float] = []
    for field in range(1, 15):
        rh = _field_rate_history(events, field, target_day)
        if not rh:
            continue
        ev, rate = rh[-1]
        # Kui pärast seda oli sama põllu teadmata ABC-ga korje, pole vana rate enam puhas state-sensor.
        later_events = [e for e in _field_events_before(events, field, target_day) if e.day > ev.day]
        if later_events:
            continue
        age = (target_day - ev.day).days
        if 0 < age <= POOL_LOOKBACK_DAYS and rate > 0:
            vals.append(rate)
            ages.append(float(age))
    if not vals:
        return None
    return float(np.median(vals)), float(np.median(ages))


def _rate_state(events: List[Event], field: int, target_day: date) -> Optional[dict]:
    hist = _field_events_before(events, field, target_day)
    latest_event = hist[-1] if hist else None
    rh = _field_rate_history(events, field, target_day)

    own_ok = False
    if rh and latest_event is not None:
        own_ok = rh[-1][0].day == latest_event.day and rh[-1][0].order == latest_event.order

    if own_ok:
        anchor_event, anchor_rate = rh[-1]
        anchor_day = anchor_event.day
        anchor_is_field = 1.0
    else:
        pooled = _pool_rate_state(events, target_day)
        if pooled is None:
            return None
        anchor_rate, pool_age = pooled
        anchor_day = target_day - timedelta(days=max(1, int(round(pool_age))))
        anchor_is_field = 0.0

    if len(rh) >= 2:
        trend = math.log(max(1e-6, rh[-1][1]) / max(1e-6, rh[-2][1]))
        has_trend = 1.0
    else:
        trend = 0.0
        has_trend = 0.0

    return {
        "anchor_rate": float(anchor_rate),
        "anchor_day": anchor_day,
        "anchor_is_field": anchor_is_field,
        "anchor_age_days": float(max(1, (target_day - anchor_day).days)),
        "same_rate_trend": float(trend),
        "has_rate_trend": has_trend,
    }


def _rate_regime_stats(events: List[Event], cutoff_day: date) -> dict:
    """Ploki värske rate-muutus. Ainult päris actualid; prognoosid ei tohi režiimi tagasi sööta."""
    by_day: Dict[date, List[float]] = {}
    for field in range(1, 15):
        rh = _field_rate_history([e for e in events if e.day < cutoff_day and e.source != "simulated_schedule"], field, cutoff_day)
        for i in range(1, len(rh)):
            prev_ev, prev_rate = rh[i - 1]
            cur_ev, cur_rate = rh[i]
            if prev_ev.source != "actual" or cur_ev.source != "actual":
                continue
            sig = math.log(max(1e-6, cur_rate) / max(1e-6, prev_rate))
            by_day.setdefault(cur_ev.day, []).append(sig)

    signals = [(dd, float(np.median(v))) for dd, v in sorted(by_day.items()) if v]
    if not signals:
        return {"regime3": 0.0}
    last3 = signals[-3:]
    weights = np.asarray([1.0, 2.0, 4.0], dtype=float)[-len(last3):]
    weights /= weights.sum()
    smooth = float(sum(w * sig for w, (_dd, sig) in zip(weights, last3)))
    return {"regime3": smooth}

# -----------------------------------------------------------------------------
# ILM: MÕÕDETUD DB + ARHIVEERITUD ECMWF IFS HRES
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
        temp = 0.5 * (night + dayt)
        out[dd] = {
            "day": dd,
            "temp": temp,
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
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "et0_fao_evapotranspiration",
        "wind_speed_10m",
        "shortwave_radiation",
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
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "et0_fao_evapotranspiration",
        "wind_speed_10m",
        "shortwave_radiation",
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
            # 3/6h native samm võib Open-Meteos olla interpoleeritud; tavaliselt 24.
            # Alla 18 tunni ei kasuta päeva, et osalist päeva mitte vaikides läbi lasta.
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
        temp = 0.5 * (float(np.mean(night_vals)) + float(np.mean(day_vals)))
        # shortwave_radiation on W/m² keskmine eelmise tunni kohta -> MJ/m² päevas.
        rad_mj = float(rad.sum() * 0.0036)
        out[dd] = {
            "day": dd,
            "temp": temp,
            "temp_min": float(temps.min()),
            "temp_max": float(temps.max()),
            "wind": float(wind.mean()),
            "rad": rad_mj,
            "rh": float(rhs.mean()),
            "rain": float(rain.sum()),
            "et0": float(et0.sum()),
            "source": "ecmwf_forecast",
        }
    return out


def _fetch_ecmwf_issue(issue_day: date) -> Tuple[Dict[date, dict], str]:
    errors = []
    for hour in FORECAST_RUN_HOURS_UTC:
        url = _openmeteo_url(issue_day, hour)
        try:
            payload = _fetch_json(url)
            daily = _hourly_to_daily(payload)
            # Vajame vähemalt issue+1 ... issue+8, sest +9 saagi kasvuilm lõpeb target-1.
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
    rows = []
    d0 = start_day
    while d0 <= end_day:
        row = wmap.get(d0)
        if row is None:
            return None
        rows.append(row)
        d0 += timedelta(days=1)
    return rows


def _agg_weather(rows: Optional[List[dict]]) -> Optional[dict]:
    if not rows:
        return None
    temp = np.asarray([r["temp"] for r in rows], dtype=float)
    rad = np.asarray([r["rad"] for r in rows], dtype=float)
    et0 = np.asarray([r["et0"] for r in rows], dtype=float)
    rh = np.asarray([r["rh"] for r in rows], dtype=float)
    wind = np.asarray([r["wind"] for r in rows], dtype=float)
    rain = np.asarray([r["rain"] for r in rows], dtype=float)
    gdd10 = np.maximum(0.0, temp - 10.0)
    return {
        "temp": float(temp.mean()),
        "rad": float(rad.mean()),
        "gdd10": float(gdd10.mean()),
        "et0": float(et0.mean()),
        "rh": float(rh.mean()),
        "wind": float(wind.mean()),
        "rain": float(rain.mean()),
        "n": len(rows),
    }


# -----------------------------------------------------------------------------
# RECORD / RATE-STATE TRANSITION FEATURE'D
# -----------------------------------------------------------------------------


def _weather_features(
    events: List[Event],
    field: int,
    target_day: date,
    target_order: int,
    wmap: Dict[date, dict],
) -> Optional[dict]:
    gi = _growth_info(events, field, target_day, target_order)
    prev = gi["prev_event"]
    if prev is not None:
        gw_start = prev.day + timedelta(days=1)
    else:
        gw_start = target_day - timedelta(days=max(1, int(round(DEFAULT_GROWTH_DAYS))))
    gw_end = target_day - timedelta(days=1)
    gw_rows = _range_weather(wmap, gw_start, gw_end)
    gw = _agg_weather(gw_rows) if gw_rows is not None else None
    if gw is None:
        return None

    memory14_rows = _range_weather(wmap, target_day - timedelta(days=14), target_day - timedelta(days=1))
    memory14 = _agg_weather(memory14_rows) if memory14_rows is not None else None
    if memory14 is None:
        return None

    return {
        "growth_days": float(gi["growth"]),
        "gw_gdd10": float(gw["gdd10"]),
        "gw_rad": float(gw["rad"]),
        "gw_et0": float(gw["et0"]),
        "gw_rh": float(gw["rh"]),
        "w14_rad": float(memory14["rad"]),
    }


def _build_rate_record(
    *,
    events: List[Event],
    field: int,
    target_day: date,
    target_order: int,
    wmap: Dict[date, dict],
    regime_cutoff: date,
    actual_abc: Optional[float],
    forced_anchor_rate: Optional[float] = None,
    frozen_rate_trend: Optional[Tuple[float, float]] = None,
) -> Optional[dict]:
    rs = _rate_state(events, field, target_day)
    if rs is None and forced_anchor_rate is None:
        return None
    wf = _weather_features(events, field, target_day, target_order, wmap)
    if wf is None:
        return None

    anchor_rate = float(forced_anchor_rate) if forced_anchor_rate is not None else float(rs["anchor_rate"])
    if anchor_rate <= 0:
        return None

    if frozen_rate_trend is not None:
        rate_trend, has_rate_trend = frozen_rate_trend
    elif rs is not None:
        rate_trend = float(rs["same_rate_trend"])
        has_rate_trend = float(rs["has_rate_trend"])
    else:
        rate_trend, has_rate_trend = 0.0, 0.0

    flower_start = FLOWERING_STARTS.get(field)
    if flower_start is None:
        return None
    regime = _rate_regime_stats(events, regime_cutoff)

    rec = {
        "target_day": target_day,
        "field_no": field,
        "target_order": target_order,
        "log_anchor_rate": math.log(anchor_rate + 1e-6),
        "same_rate_trend": float(rate_trend),
        "has_rate_trend": float(has_rate_trend),
        **wf,
        **regime,
        "flower_age_days": float((target_day - flower_start).days),
        # diagnostika
        "anchor_rate": anchor_rate,
        "anchor_abc_equiv": anchor_rate * float(wf["growth_days"]),
        "anchor_is_field": float(rs["anchor_is_field"]) if rs is not None else 1.0,
        "actual_abc": float(actual_abc) if actual_abc is not None else None,
    }

    if actual_abc is not None and actual_abc > 0:
        actual_rate = float(actual_abc) / max(0.5, float(wf["growth_days"]))
        rec["actual_rate"] = actual_rate
        rec["y"] = math.log(max(1e-6, actual_rate) / max(1e-6, anchor_rate))
    else:
        rec["actual_rate"] = None
        rec["y"] = None
    return rec


def _make_historical_records(events: List[Event], actual_lookup: Dict[Tuple[date, int], dict], measured: Dict[date, dict]) -> List[dict]:
    records: List[dict] = []
    for (dd, field), row in sorted(actual_lookup.items()):
        if dd < WEATHER_START + timedelta(days=14):
            continue
        prior_events = [e for e in events if e.day < dd]
        # Rate-target vajab eelmist korjesündmust; esimene korje ei ole transition-target.
        if not _field_events_before(prior_events, field, dd):
            continue
        rec = _build_rate_record(
            events=prior_events,
            field=field,
            target_day=dd,
            target_order=int(row.get("_order") or 1),
            wmap=measured,
            regime_cutoff=dd,
            actual_abc=float(row["_abc"]),
        )
        if rec is not None:
            records.append(rec)
    records.sort(key=lambda r: (r["target_day"], r["field_no"]))
    return records

# -----------------------------------------------------------------------------
# RIDGE STATE-TRANSITION
# -----------------------------------------------------------------------------


def _fit_model(train_records: List[dict], cutoff_day: date) -> Optional[dict]:
    clean = []
    for r in train_records:
        y = _f(r.get("y"))
        vals = [_f(r.get(c)) for c in FEATURES]
        if y is None or any(v is None for v in vals):
            continue
        age = max(0.0, float((cutoff_day - r["target_day"]).days))
        weight = 0.5 ** (age / RECENCY_HALFLIFE_DAYS)
        clean.append((vals, y, weight))

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

    D = np.column_stack([np.ones(len(Z)), Z])
    ws = np.sqrt(w)[:, None]
    Dw = D * ws
    yw = y * ws[:, 0]
    penalty = np.eye(D.shape[1], dtype=float)
    penalty[0, 0] = 0.0
    lhs = Dw.T @ Dw + RIDGE_ALPHA * penalty
    rhs = Dw.T @ yw
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
        "mu": mu,
        "sd": sd,
        "beta": beta,
        "bounds": bounds,
        "n": len(clean),
    }


def _predict_rate_with_model(model: dict, rec: dict) -> Optional[Tuple[float, float]]:
    vals = [_f(rec.get(c)) for c in FEATURES]
    if any(v is None for v in vals):
        return None
    x = np.asarray(vals, dtype=float)
    z = (x - model["mu"]) / model["sd"]
    z = np.clip(z, -3.0, 3.0)
    yhat = float(np.r_[1.0, z] @ model["beta"])
    if model.get("bounds") is not None:
        lo, hi = model["bounds"]
        yhat = float(np.clip(yhat, lo, hi))

    n_train = float(model.get("n") or 0.0)
    learn_fraction = n_train / (n_train + SMALL_SAMPLE_PRIOR_ROWS)
    yhat *= learn_fraction

    anchor_rate = float(rec["anchor_rate"])
    pred_rate = max(0.0, anchor_rate * math.exp(yhat))
    pred_abc = pred_rate * max(0.5, float(rec["growth_days"]))
    return float(pred_rate), float(pred_abc)


# -----------------------------------------------------------------------------
# REPLAY
# -----------------------------------------------------------------------------


def _issue_dates(last_actual_day: date) -> List[date]:
    end = min(last_actual_day - timedelta(days=1), TODAY - timedelta(days=1))
    if end < REPLAY_START:
        return []
    n = (end - REPLAY_START).days
    return [REPLAY_START + timedelta(days=i) for i in range(n + 1)]


def _future_schedule_stubs(
    actual_lookup: Dict[Tuple[date, int], dict], issue_day: date
) -> List[Tuple[date, int, int]]:
    """Ainult kuupäev, põld, järjekord. Ühtegi tuleviku saaginumbrit ei tagasta."""
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

    train = [r for r in historical_records if r["target_day"] <= issue_day]
    model = _fit_model(train, issue_day)
    if model is None:
        return []

    wmap = _combined_weather(measured, forecast_map, issue_day)
    stubs = _future_schedule_stubs(actual_lookup, issue_day)
    out = []

    # Latentne rate kandub tulevikus edasi. ABC ise EI muutu järgmise korje sensoriks.
    latent_rate: Dict[int, float] = {}
    frozen_trend: Dict[int, Tuple[float, float]] = {}

    for target_day, field, order in stubs:
        if field not in latent_rate:
            rs0 = _rate_state(known_events, field, target_day)
            if rs0 is None:
                continue
            latent_rate[field] = float(rs0["anchor_rate"])
            frozen_trend[field] = (float(rs0["same_rate_trend"]), float(rs0["has_rate_trend"]))

        rec = _build_rate_record(
            events=schedule_events,
            field=field,
            target_day=target_day,
            target_order=order,
            wmap=wmap,
            regime_cutoff=issue_day + timedelta(days=1),
            actual_abc=None,
            forced_anchor_rate=latent_rate[field],
            frozen_rate_trend=frozen_trend[field],
        )
        if rec is None:
            continue
        predicted = _predict_rate_with_model(model, rec)
        if predicted is None:
            continue
        pred_rate, pred_abc = predicted

        actual_row = actual_lookup.get((target_day, field))
        actual_abc = float(actual_row["_abc"]) if actual_row is not None else np.nan
        lead = (target_day - issue_day).days
        persist_abc = float(rec["anchor_rate"]) * float(rec["growth_days"])
        out.append({
            "issue_day": issue_day,
            "target_day": target_day,
            "lead": lead,
            "field": field,
            "order": order,
            "pred_abc": float(pred_abc),
            "persist_abc": float(persist_abc),
            "actual_abc": actual_abc,
            "anchor_rate": float(rec["anchor_rate"]),
            "pred_rate": float(pred_rate),
            "anchor_abc": float(rec["anchor_abc_equiv"]),
            "anchor_is_field": int(rec["anchor_is_field"]),
            "growth_days": float(rec["growth_days"]),
            "regime3": float(rec["regime3"]),
            "flower_age_days": float(rec["flower_age_days"]),
            "gw_rad": float(rec["gw_rad"]),
            "gw_gdd10": float(rec["gw_gdd10"]),
            "run": run_label,
            "train_n": int(model["n"]),
        })

        # Edasi läheb ainult latentne rate; saak ise pole state-sensor.
        latent_rate[field] = float(pred_rate)
        schedule_events.append(Event(target_day, field, order, None, False, "simulated_schedule"))
        schedule_events.sort(key=_event_sort_key)

    return out


def _daily_scores(field_df: pd.DataFrame) -> pd.DataFrame:
    if field_df.empty:
        return pd.DataFrame()
    daily = field_df.groupby(["issue_day", "target_day", "lead"], as_index=False).agg(
        pred_abc=("pred_abc", "sum"),
        persist_abc=("persist_abc", "sum"),
        actual_abc=("actual_abc", "sum"),
        n_fields=("field", "count"),
        train_n=("train_n", "max"),
    )
    daily["err"] = daily["pred_abc"] - daily["actual_abc"]
    daily["ape"] = daily["err"].abs() / daily["actual_abc"].clip(lower=0.1)
    daily["bias_pct"] = 100.0 * daily["err"] / daily["actual_abc"].clip(lower=0.1)
    daily["persist_ape"] = (daily["persist_abc"] - daily["actual_abc"]).abs() / daily["actual_abc"].clip(lower=0.1)
    return daily


def _lead_scores(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()
    rows = []
    for lead, g in daily.groupby("lead"):
        rows.append({
            "Lead": int(lead),
            "N": int(len(g)),
            "RATE-STATE MAPE %": 100.0 * float(g["ape"].mean()),
            "Persistence MAPE %": 100.0 * float(g["persist_ape"].mean()),
            "±20% sees %": 100.0 * float((g["ape"] <= 0.20).mean()),
            "Bias %": float(g["bias_pct"].mean()),
            "MAE ABC": float((g["pred_abc"] - g["actual_abc"]).abs().mean()),
        })
    return pd.DataFrame(rows).sort_values("Lead")


# -----------------------------------------------------------------------------
# SELF TEST
# -----------------------------------------------------------------------------


def _self_test() -> None:
    start = date(2026, 7, 1)
    measured = {}
    for i in range(60):
        dd = start + timedelta(days=i)
        measured[dd] = {
            "day": dd, "temp": 18.0 + 0.02*i, "temp_min": 12.0, "temp_max": 24.0,
            "wind": 2.0, "rad": 18.0 + 0.05*i, "rh": 75.0, "rain": 0.5,
            "et0": 3.0, "source": "measured",
        }

    events = [
        Event(date(2026,7,5), 1, 1, 7.0, True, "actual"),
        Event(date(2026,7,10), 1, 1, 8.0, True, "actual"),
        Event(date(2026,7,15), 1, 1, 9.0, True, "actual"),
        Event(date(2026,7,20), 1, 1, 7.5, True, "actual"),
        Event(date(2026,7,6), 2, 1, 8.0, True, "actual"),
        Event(date(2026,7,11), 2, 1, 9.0, True, "actual"),
        Event(date(2026,7,16), 2, 1, 10.0, True, "actual"),
        Event(date(2026,7,21), 2, 1, 8.0, True, "actual"),
    ]
    events.sort(key=_event_sort_key)

    rs = _rate_state(events, 1, date(2026,7,25))
    assert rs is not None and rs["anchor_rate"] > 0

    fc = {d: dict(v, source="ecmwf_forecast") for d, v in measured.items() if d > date(2026,7,22)}
    cm = _combined_weather(measured, fc, date(2026,7,22))
    rec = _build_rate_record(
        events=events, field=1, target_day=date(2026,7,25), target_order=1,
        wmap=cm, regime_cutoff=date(2026,7,23), actual_abc=None,
    )
    assert rec is not None and rec["y"] is None and all(k in rec for k in FEATURES)

    fake = []
    for i in range(16):
        rr = dict(rec)
        rr["target_day"] = date(2026,7,1) + timedelta(days=i)
        rr["y"] = 0.015 * math.sin(i/3.0)
        for c in FEATURES:
            rr[c] = float(rr[c]) + 0.0005*i
        fake.append(rr)
    model = _fit_model(fake, date(2026,7,22))
    assert model is not None
    pr = _predict_rate_with_model(model, rec)
    assert pr is not None and all(math.isfinite(x) and x >= 0 for x in pr)

    lookup = {
        (date(2026,7,23), 1): {"_order": 1, "_abc": 99.0},
        (date(2026,7,28), 1): {"_order": 1, "_abc": 88.0},
    }
    stubs = _future_schedule_stubs(lookup, date(2026,7,22))
    assert stubs == [(date(2026,7,23),1,1),(date(2026,7,28),1,1)]

    # Rekursiooni põhikontroll: tuleviku schedule-eventil abc=None; latent rate on eraldi state.
    schedule = _copy_events([e for e in events if e.day <= date(2026,7,22)])
    schedule.append(Event(date(2026,7,23), 1, 1, None, False, "simulated_schedule"))
    gi = _growth_info(schedule, 1, date(2026,7,28), 1)
    assert 4.5 <= gi["growth"] <= 5.5

    print("LAB-137 SELF-TEST OK")


# -----------------------------------------------------------------------------
# STREAMLIT UI
# -----------------------------------------------------------------------------


def main() -> None:
    import streamlit as st
    import db

    st.set_page_config(page_title="KurgiMootor LAB-137", layout="wide")
    st.error("🧪 LAB-137 RATE-STATE AJAMASIN · READ-ONLY · 22.07 → +1…+9")
    st.title("KurgiMootor · RATE-STATE ajamasin")
    st.caption(
        "Üks fikseeritud ABC-mudel. State = ABC kasvupäeva kohta; ilm = state'i muutus. "
        "Replay algab 22.07.2026 ja kasutab tuleviku jaoks arhiveeritud ECMWF IFS HRES 9 km prognoosijookse."
    )

    with st.expander("Katse reeglid ja lukud", expanded=False):
        st.markdown(
            """
- **Prognoosi hetk:** iga replay-päeva õhtu; sama päeva teadaolev korje ja mõõdetud ilm võivad olla sees.
- **Tuleviku ilm:** ECMWF IFS HRES 9 km, eelistus 12 UTC issue-päeva jooks; fallback 06/00 UTC.
- **+1…+9:** üks issue-mudel külmutatakse; tuleviku tegelik saak ei õpeta sama issue prognoosi. Edasi kandub latentne rate, mitte prognoositud ABC.
- **Korjeplaan:** ajaloolise rea kuupäev/põld/järjekord on testplaan; A/B/C/XL/total eemaldatakse enne predictionit.
- **Siht:** ainult **A+B+C**. XL ja C/B on teadlikult sellest katsest väljas.
- **Õitsemine:** praegu suhteline vanusetelg: põld 1 = 14.06 ja iga järgmine põld +1 päev (13 päeva vahe 1→14).
- **DB:** ainult lugemine; production prognoose ega ilma ei kirjutata/uuendata.
            """
        )

    @st.cache_data(ttl=120, show_spinner=False)
    def _load_db_data():
        harvests = db.get_harvest_history(limit=5000)
        weather = db.get_weather_rows(WEATHER_START, TODAY)
        return harvests, weather

    @st.cache_data(ttl=86400, show_spinner=False)
    def _cached_issue_weather(issue_iso: str):
        return _fetch_ecmwf_issue(date.fromisoformat(issue_iso))

    if st.button("Värskenda DB andmed", type="secondary"):
        _load_db_data.clear()
        st.rerun()

    try:
        harvest_rows, weather_rows = _load_db_data()
    except Exception as exc:
        st.error(f"DB lugemine ebaõnnestus: {exc}")
        st.stop()

    events, actual_lookup = _prepare_events(harvest_rows)
    if not actual_lookup:
        st.error("Usaldusväärseid A+B+C korjeridu ei leitud.")
        st.stop()

    measured = _measured_weather_map(weather_rows)
    last_actual_day = max(dd for dd, _field in actual_lookup.keys())
    issue_days = _issue_dates(last_actual_day)
    if not issue_days:
        st.error("Replay jaoks pole 22.07 järel piisavalt tegelikke päevi.")
        st.stop()

    # Range completeness: historical recordid vajavad 14p ilmamälu.
    required_end = min(last_actual_day, TODAY - timedelta(days=1))
    missing_measured = []
    d0 = WEATHER_START
    while d0 <= required_end:
        if d0 not in measured:
            missing_measured.append(d0)
        d0 += timedelta(days=1)
    if missing_measured:
        st.error(
            "Mõõdetud ilma ajalugu ei ole 100% täielik. Puudu: "
            + ", ".join(d.strftime("%d.%m") for d in missing_measured[:20])
        )
        st.stop()

    historical_records = _make_historical_records(events, actual_lookup, measured)

    top1, top2, top3, top4 = st.columns(4)
    top1.metric("Replay algus", REPLAY_START.strftime("%d.%m.%Y"))
    top2.metric("Usaldusväärseid ABC ridu", str(len(actual_lookup)))
    top3.metric("Historical training recordeid", str(len(historical_records)))
    top4.metric("Issue-päevi", str(len(issue_days)))

    st.markdown("### 1. Jooksuta ajamasin")
    st.caption(
        "Esimene käivitus tõmbab iga issue-päeva arhiveeritud ECMWF jooksu ühe korra. "
        "Seejärel hoiab Streamlit need cache'is 24 h; arvutus ise on väike."
    )

    run_requested = st.button("▶ Jooksuta 22.07 → tänane replay", type="primary")
    if not run_requested and "lab137_results" not in st.session_state:
        st.info("Vajuta üks kord nuppu. Productionit see ei puuduta.")
        st.stop()

    if run_requested:
        all_rows = []
        weather_errors = []
        run_used = {}
        progress = st.progress(0.0, text="ECMWF ajalooliste jooksude laadimine…")
        for i, issue_day in enumerate(issue_days):
            try:
                fc_map, run_label = _cached_issue_weather(issue_day.isoformat())
                run_used[issue_day] = run_label
                rows = _predict_issue(
                    issue_day=issue_day,
                    all_events=events,
                    actual_lookup=actual_lookup,
                    measured=measured,
                    historical_records=historical_records,
                    forecast_map=fc_map,
                    run_label=run_label,
                )
                all_rows.extend(rows)
            except Exception as exc:
                weather_errors.append((issue_day, str(exc)))
            progress.progress((i + 1) / len(issue_days), text=f"{issue_day.strftime('%d.%m')} · {i+1}/{len(issue_days)}")
        progress.empty()

        field_df = pd.DataFrame(all_rows)
        daily_df = _daily_scores(field_df)
        st.session_state["lab137_results"] = {
            "field": field_df,
            "daily": daily_df,
            "errors": weather_errors,
            "runs": run_used,
        }

    result = st.session_state.get("lab137_results")
    if not result:
        st.stop()

    field_df: pd.DataFrame = result["field"]
    daily: pd.DataFrame = result["daily"]
    errors = result["errors"]
    runs = result["runs"]

    if field_df.empty or daily.empty:
        st.error("Replay ei saanud ühtegi skooritavat prognoosi. Vaata ECMWF/API vigu allpool.")
        if errors:
            st.dataframe(pd.DataFrame([{"Päev": d, "Viga": e} for d, e in errors]), use_container_width=True, hide_index=True)
        st.stop()

    st.markdown("### 2. Tulemus lead'i kaupa")
    lead_df = _lead_scores(daily)
    overall_mape = 100.0 * float(daily["ape"].mean())
    overall_hit = 100.0 * float((daily["ape"] <= 0.20).mean())
    persist_mape = 100.0 * float(daily["persist_ape"].mean())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("RATE-STATE · MAPE", f"{overall_mape:.1f}%")
    c2.metric("Persistence · MAPE", f"{persist_mape:.1f}%")
    for col, lead in zip((c3, c4), (1, 5)):
        row = lead_df[lead_df["Lead"] == lead]
        col.metric(f"+{lead} päeva MAPE", "—" if row.empty else f"{float(row.iloc[0]['RATE-STATE MAPE %']):.1f}%")

    st.dataframe(
        lead_df.style.format({
            "RATE-STATE MAPE %": "{:.1f}%",
            "Persistence MAPE %": "{:.1f}%",
            "±20% sees %": "{:.0f}%",
            "Bias %": "{:+.1f}%",
            "MAE ABC": "{:.1f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### 3. Ühe issue-päeva 9 päeva prognoos")
    available_issues = sorted(daily["issue_day"].unique())
    default_idx = len(available_issues) - 1
    chosen = st.selectbox(
        "Prognoos tehtud",
        available_issues,
        index=default_idx,
        format_func=lambda d: pd.Timestamp(d).strftime("%d.%m.%Y"),
    )
    issue_table = daily[daily["issue_day"] == chosen].copy().sort_values("lead")
    issue_table["Prognoos tehtud"] = issue_table["issue_day"].map(lambda d: pd.Timestamp(d).strftime("%d.%m"))
    issue_table["Sihtpäev"] = issue_table["target_day"].map(lambda d: pd.Timestamp(d).strftime("%d.%m"))
    issue_table["Lead"] = issue_table["lead"].map(lambda x: f"+{int(x)}")
    issue_table["RATE-STATE ABC"] = issue_table["pred_abc"]
    issue_table["Persistence ABC"] = issue_table["persist_abc"]
    issue_table["Tegelik ABC"] = issue_table["actual_abc"]
    issue_table["Viga %"] = 100.0 * issue_table["ape"]
    issue_table["Treeningridu"] = issue_table["train_n"]
    st.dataframe(
        issue_table[["Prognoos tehtud", "Sihtpäev", "Lead", "RATE-STATE ABC", "Persistence ABC", "Tegelik ABC", "Viga %", "Treeningridu"]]
        .style.format({"RATE-STATE ABC": "{:.1f}", "Persistence ABC": "{:.1f}", "Tegelik ABC": "{:.1f}", "Viga %": "{:.1f}%"}),
        use_container_width=True,
        hide_index=True,
    )

    run_label = runs.get(pd.Timestamp(chosen).date() if isinstance(chosen, pd.Timestamp) else chosen)
    if run_label:
        st.caption(f"Selle issue ilma jooks: ECMWF IFS HRES · {run_label}")

    st.markdown("### 4. Sama sihtpäev eri kauguselt")
    target_options = sorted(daily["target_day"].unique())
    target_default = len(target_options) - 1
    target_choice = st.selectbox(
        "Sihtpäev",
        target_options,
        index=target_default,
        format_func=lambda d: pd.Timestamp(d).strftime("%d.%m.%Y"),
        key="target_choice",
    )
    tdf = daily[daily["target_day"] == target_choice].copy().sort_values("lead", ascending=False)
    tdf["Prognoos tehtud"] = tdf["issue_day"].map(lambda d: pd.Timestamp(d).strftime("%d.%m"))
    tdf["Lead"] = tdf["lead"].map(lambda x: f"+{int(x)}")
    tdf["RATE-STATE ABC"] = tdf["pred_abc"]
    tdf["Persistence ABC"] = tdf["persist_abc"]
    tdf["Tegelik ABC"] = tdf["actual_abc"]
    tdf["Viga %"] = 100.0 * tdf["ape"]
    st.dataframe(
        tdf[["Prognoos tehtud", "Lead", "RATE-STATE ABC", "Persistence ABC", "Tegelik ABC", "Viga %"]]
        .style.format({"RATE-STATE ABC": "{:.1f}", "Persistence ABC": "{:.1f}", "Tegelik ABC": "{:.1f}", "Viga %": "{:.1f}%"}),
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("Põldude detail · mida RATE-STATE nägi"):
        fshow = field_df.copy()
        fshow["Prognoos"] = fshow["issue_day"].map(lambda d: pd.Timestamp(d).strftime("%d.%m"))
        fshow["Siht"] = fshow["target_day"].map(lambda d: pd.Timestamp(d).strftime("%d.%m"))
        fshow["Lead"] = fshow["lead"].map(lambda x: f"+{int(x)}")
        fshow["Põld"] = fshow["field"]
        fshow["RATE-STATE ABC"] = fshow["pred_abc"]
        fshow["Persistence ABC"] = fshow["persist_abc"]
        fshow["Anchor rate"] = fshow["anchor_rate"]
        fshow["Pred rate"] = fshow["pred_rate"]
        fshow["Tegelik ABC"] = fshow["actual_abc"]
        fshow["Anchor"] = fshow["anchor_abc"]
        fshow["Oma põld anchor"] = fshow["anchor_is_field"].map({1: "jah", 0: "plokk"})
        fshow["Kasvuaeg p"] = fshow["growth_days"]
        fshow["Režiim3 %"] = 100.0 * (np.exp(fshow["regime3"]) - 1.0)
        fshow["Õitsemisest p"] = fshow["flower_age_days"]
        st.dataframe(
            fshow[[
                "Prognoos", "Siht", "Lead", "Põld", "RATE-STATE ABC", "Persistence ABC", "Tegelik ABC",
                "Anchor rate", "Pred rate", "Kasvuaeg p", "Režiim3 %", "Õitsemisest p", "train_n",
            ]].style.format({
                "RATE-STATE ABC": "{:.1f}", "Persistence ABC": "{:.1f}", "Tegelik ABC": "{:.1f}",
                "Anchor rate": "{:.2f}", "Pred rate": "{:.2f}",
                "Kasvuaeg p": "{:.2f}", "Režiim3 %": "{:+.1f}%",
                "Õitsemisest p": "{:.0f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("Lekkeaudit / tehniline kontroll"):
        st.success("✅ Train target_day ≤ issue_day")
        st.success("✅ Mõõdetud ilm ainult ≤ issue_day; tulevik ECMWF arhiveeritud issue-jooksust")
        st.success("✅ Tuleviku schedule stub = ainult kuupäev + põld + order")
        st.success("✅ Sama issue +1…+9 mudeli koefitsiendid on külmutatud")
        st.success("✅ Prognoositud ABC ei muutu uueks sensoriks; edasi kandub ainult latentne rate")
        st.info(f"Mudel: {FORECAST_MODEL} · ridge α={RIDGE_ALPHA:g} · recency half-life={RECENCY_HALFLIFE_DAYS:g} p · min train={MIN_TRAIN_ROWS} · cold-start prior={SMALL_SAMPLE_PRIOR_ROWS:g} rida")
        if errors:
            st.warning(f"ECMWF jooksu ei saadud {len(errors)} issue-päeval.")
            st.dataframe(pd.DataFrame([{"Päev": d.strftime("%d.%m"), "Viga": e} for d, e in errors]), use_container_width=True, hide_index=True)
        else:
            st.success("✅ Kõigi issue-päevade ECMWF jooksud saadi kätte.")

    st.caption(
        f"{LAB_VERSION} · read-only · replay {REPLAY_START.isoformat()} → {last_actual_day.isoformat()} · "
        "ABC only · latent rate-state · ECMWF IFS HRES single-run archive"
    )


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        _self_test()
    else:
        main()
