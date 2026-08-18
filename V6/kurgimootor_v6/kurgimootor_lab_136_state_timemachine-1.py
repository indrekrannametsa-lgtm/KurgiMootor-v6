from __future__ import annotations

"""
KurgiMootor LAB-136 STATE AJAMASIN
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
STATE anchor + ridge transition:
- state-anchor = viimase sama põllu usaldusväärne ABC, kui viimane korjesündmus on
  usaldusväärse ABC-ga; muidu värske kogu ploki robustne state-tase;
- mudeli target = log(ABC_actual / state_anchor);
- sisendid = sama põllu trend, ploki värske režiim, täpne kasvuaeg, kasvuperioodi ilm,
  ilmamuutus, 7p ilmamuutus, 14p ilmamälu ja bioloogiline vanus õitsemisest;
- ridge on tugevalt regulariseeritud ja recency-weighted.

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
LAB_VERSION = "LAB-136-STATE-TIMEMACHINE-V1"

REPLAY_START = date(2026, 7, 22)
WEATHER_START = date(2026, 7, 1)
SEASON_START = date(2026, 6, 15)

FORECAST_LAT = 58.1275
FORECAST_LON = 24.49167
FORECAST_MODEL = "ecmwf_ifs"  # Open-Meteo model id: ECMWF IFS HRES 9 km
FORECAST_RUN_HOURS_UTC = (12, 6, 0)  # kõik on issue-päeva jooksul kättesaadavaks saavad jooksud
FORECAST_DAYS = 10  # issue-päev + järgmised 9 kalendripäeva

RIDGE_ALPHA = 28.0
RECENCY_HALFLIFE_DAYS = 18.0
MIN_TRAIN_ROWS = 4
SMALL_SAMPLE_PRIOR_ROWS = 8.0  # cold-start: learned transition shrinks toward state persistence
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
FEATURES = [
    # State / oma põllu mälu
    "log_anchor",
    "anchor_is_field",
    "anchor_age_days",
    "has_lag2",
    "same_trend1",
    "has_lag3",
    "same_trend2",
    # Kasvuaeg
    "growth_days",
    "prev_growth_days",
    "growth_delta",
    "log_growth_ratio",
    "growth_known",
    # Kasvuperioodi ilm
    "gw_temp",
    "gw_rad",
    "gw_gdd10",
    "gw_et0",
    "gw_rh",
    "gw_wind",
    "gw_rain",
    # Muutus eelmise sama põllu kasvuperioodi suhtes
    "dw_temp",
    "dw_rad",
    "dw_gdd10",
    "dw_et0",
    "dw_rh",
    "dw_wind",
    "dw_rain",
    "has_prev_weather",
    # 7 päeva ilmamuutus ning 14 päeva state-mälu
    "d7_temp",
    "d7_rad",
    "d7_gdd10",
    "d7_et0",
    "d7_rh",
    "d7_wind",
    "w14_temp",
    "w14_rad",
    "w14_gdd10",
    "w14_et0",
    "w14_rh",
    "w14_wind",
    "w14_rain",
    # Kogu ploki värske režiimisensor
    "regime1",
    "regime3",
    # Bioloogiline vanus
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
# KORJEANDMED / STATE
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

        # Hindamiseks hoiame ainult usaldusväärse tegeliku ABC-ga ridu.
        if abc is not None:
            rr = dict(raw)
            rr["_abc"] = float(abc)
            rr["_day"] = dd
            rr["_field"] = field
            rr["_order"] = order
            actual_lookup[(dd, field)] = rr

    events.sort(key=_event_sort_key)
    return events, actual_lookup


def _events_before(events: List[Event], cutoff: date, inclusive: bool = False) -> List[Event]:
    if inclusive:
        return [e for e in events if e.day <= cutoff]
    return [e for e in events if e.day < cutoff]


def _field_events_before(events: List[Event], field: int, target_day: date) -> List[Event]:
    return [e for e in events if e.field == field and e.day < target_day]


def _latest_reliable_per_field(events: List[Event], target_day: date) -> Dict[int, Event]:
    out: Dict[int, Event] = {}
    for e in events:
        if e.day >= target_day or e.abc is None or not e.reliable:
            continue
        old = out.get(e.field)
        if old is None or _event_sort_key(e) > _event_sort_key(old):
            out[e.field] = e
    return out


def _pool_state(events: List[Event], target_day: date) -> Optional[Tuple[float, float]]:
    """Värske robustne kogu ploki state.

    Võtame iga põllu viimase usaldusväärse/simuleeritud ABC ainult ühe korra, et
    tihedamini esinev põld plokki ei domineeriks. Vanem kui 10 päeva jääb välja.
    Tagastab (mediaan ABC, mediaan vanus päevades).
    """
    latest = _latest_reliable_per_field(events, target_day)
    vals = []
    ages = []
    for e in latest.values():
        age = (target_day - e.day).days
        if 0 < age <= POOL_LOOKBACK_DAYS and e.abc is not None and e.abc > 0:
            vals.append(float(e.abc))
            ages.append(float(age))
    if not vals:
        return None
    return float(np.median(vals)), float(np.median(ages))


def _state_anchor(events: List[Event], field: int, target_day: date) -> Optional[dict]:
    hist = _field_events_before(events, field, target_day)
    latest_event = hist[-1] if hist else None

    # Oma põllu anchor on lubatud ainult siis, kui kõige värskem korjesündmus ise
    # sisaldab usaldusväärset/predicted ABC-d. Nii ei ankurduta üle teadmata korje.
    if latest_event is not None and latest_event.reliable and latest_event.abc is not None and latest_event.abc > 0:
        anchor = float(latest_event.abc)
        anchor_day = latest_event.day
        is_field = 1.0
    else:
        pooled = _pool_state(events, target_day)
        if pooled is None:
            return None
        anchor, pool_age = pooled
        anchor_day = target_day - timedelta(days=max(1, int(round(pool_age))))
        is_field = 0.0

    reliable_hist = [e for e in hist if e.reliable and e.abc is not None and e.abc > 0]
    lag1 = reliable_hist[-1] if reliable_hist else None
    lag2 = reliable_hist[-2] if len(reliable_hist) >= 2 else None
    lag3 = reliable_hist[-3] if len(reliable_hist) >= 3 else None

    has2 = 1.0 if lag1 is not None and lag2 is not None else 0.0
    has3 = 1.0 if lag2 is not None and lag3 is not None else 0.0
    trend1 = _safe_log_ratio(float(lag1.abc), float(lag2.abc)) if has2 else 0.0
    trend2 = _safe_log_ratio(float(lag2.abc), float(lag3.abc)) if has3 else 0.0

    return {
        "anchor": anchor,
        "anchor_day": anchor_day,
        "anchor_is_field": is_field,
        "anchor_age_days": float(max(1, (target_day - anchor_day).days)),
        "has_lag2": has2,
        "same_trend1": trend1,
        "has_lag3": has3,
        "same_trend2": trend2,
        "lag1_event": lag1,
        "lag2_event": lag2,
        "lag3_event": lag3,
        "latest_event": latest_event,
    }


def _growth_info(events: List[Event], field: int, target_day: date, target_order: int) -> dict:
    hist = _field_events_before(events, field, target_day)
    prev = hist[-1] if hist else None
    prev2 = hist[-2] if len(hist) >= 2 else None

    if prev is not None:
        growth = float((target_day - prev.day).days) + (target_order - prev.order) * (3.0 / 24.0)
        growth_known = 1.0
    else:
        growth = DEFAULT_GROWTH_DAYS
        growth_known = 0.0

    if prev is not None and prev2 is not None:
        prev_growth = float((prev.day - prev2.day).days) + (prev.order - prev2.order) * (3.0 / 24.0)
        if prev_growth <= 0:
            prev_growth = growth
    else:
        prev_growth = growth

    growth = max(0.5, growth)
    prev_growth = max(0.5, prev_growth)
    return {
        "growth": growth,
        "prev_growth": prev_growth,
        "known": growth_known,
        "prev_event": prev,
        "prev2_event": prev2,
    }


def _regime_stats(events: List[Event], cutoff_day: date) -> dict:
    """Kogu ploki värske state-muutus ainult enne cutoff_day toimunud actualitest.

    Päevasignaal = mediaan log(ABC_now / ABC_prev), ainult kui eelmine sama põllu
    usaldusväärne ABC on olemas ja nende vahel pole teadmata-ABC korjesündmust.
    Päev ei pea sisaldama täpselt 3 põldu; varase hõreda ajaloo tõttu piisab 1+
    ausast sama-põllu üleminekust.
    """
    by_field: Dict[int, List[Event]] = {f: [] for f in range(1, 15)}
    for e in events:
        if e.day < cutoff_day:
            by_field[e.field].append(e)
    for f in by_field:
        by_field[f].sort(key=_event_sort_key)

    by_day: Dict[date, List[float]] = {}
    for items in by_field.values():
        for i in range(1, len(items)):
            prev = items[i - 1]
            cur = items[i]
            if not (prev.reliable and cur.reliable and prev.abc and cur.abc):
                continue
            if cur.source == "simulated" or prev.source == "simulated":
                # Režiimisensor on pärisandur, mitte enda prognooside kaja.
                continue
            by_day.setdefault(cur.day, []).append(_safe_log_ratio(float(cur.abc), float(prev.abc)))

    signals = [(dd, float(np.median(vals))) for dd, vals in sorted(by_day.items()) if vals]
    if not signals:
        return {"regime1": 0.0, "regime3": 0.0}

    last = signals[-1][1]
    last3 = signals[-3:]
    weights = np.asarray([1.0, 2.0, 4.0], dtype=float)[-len(last3):]
    weights /= weights.sum()
    smooth = float(sum(w * sig for w, (_dd, sig) in zip(weights, last3)))
    return {"regime1": float(last), "regime3": smooth}


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
# RECORD / STATE-TRANSITION FEATURE'D
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
    prev2 = gi["prev2_event"]

    if prev is not None:
        gw_start = prev.day + timedelta(days=1)
    else:
        gw_start = target_day - timedelta(days=max(1, int(round(DEFAULT_GROWTH_DAYS))))
    gw_end = target_day - timedelta(days=1)
    gw_rows = _range_weather(wmap, gw_start, gw_end)
    gw = _agg_weather(gw_rows) if gw_rows is not None else None
    if gw is None:
        return None

    pgw = None
    if prev is not None and prev2 is not None:
        p_rows = _range_weather(wmap, prev2.day + timedelta(days=1), prev.day - timedelta(days=1))
        pgw = _agg_weather(p_rows) if p_rows is not None else None
    has_prev_weather = 1.0 if pgw is not None else 0.0
    if pgw is None:
        pgw = dict(gw)

    recent7_rows = _range_weather(wmap, target_day - timedelta(days=7), target_day - timedelta(days=1))
    previous7_rows = _range_weather(wmap, target_day - timedelta(days=14), target_day - timedelta(days=8))
    memory14_rows = _range_weather(wmap, target_day - timedelta(days=14), target_day - timedelta(days=1))
    recent7 = _agg_weather(recent7_rows) if recent7_rows is not None else None
    previous7 = _agg_weather(previous7_rows) if previous7_rows is not None else None
    memory14 = _agg_weather(memory14_rows) if memory14_rows is not None else None
    if recent7 is None or previous7 is None or memory14 is None:
        return None

    return {
        "growth_days": gi["growth"],
        "prev_growth_days": gi["prev_growth"],
        "growth_delta": gi["growth"] - gi["prev_growth"],
        "log_growth_ratio": math.log(gi["growth"] / gi["prev_growth"]),
        "growth_known": gi["known"],
        "gw_temp": gw["temp"],
        "gw_rad": gw["rad"],
        "gw_gdd10": gw["gdd10"],
        "gw_et0": gw["et0"],
        "gw_rh": gw["rh"],
        "gw_wind": gw["wind"],
        "gw_rain": gw["rain"],
        "dw_temp": gw["temp"] - pgw["temp"],
        "dw_rad": gw["rad"] - pgw["rad"],
        "dw_gdd10": gw["gdd10"] - pgw["gdd10"],
        "dw_et0": gw["et0"] - pgw["et0"],
        "dw_rh": gw["rh"] - pgw["rh"],
        "dw_wind": gw["wind"] - pgw["wind"],
        "dw_rain": gw["rain"] - pgw["rain"],
        "has_prev_weather": has_prev_weather,
        "d7_temp": recent7["temp"] - previous7["temp"],
        "d7_rad": recent7["rad"] - previous7["rad"],
        "d7_gdd10": recent7["gdd10"] - previous7["gdd10"],
        "d7_et0": recent7["et0"] - previous7["et0"],
        "d7_rh": recent7["rh"] - previous7["rh"],
        "d7_wind": recent7["wind"] - previous7["wind"],
        "w14_temp": memory14["temp"],
        "w14_rad": memory14["rad"],
        "w14_gdd10": memory14["gdd10"],
        "w14_et0": memory14["et0"],
        "w14_rh": memory14["rh"],
        "w14_wind": memory14["wind"],
        "w14_rain": memory14["rain"],
    }


def _build_record(
    *,
    events: List[Event],
    field: int,
    target_day: date,
    target_order: int,
    wmap: Dict[date, dict],
    regime_cutoff: date,
    actual_abc: Optional[float],
) -> Optional[dict]:
    anchor_info = _state_anchor(events, field, target_day)
    if anchor_info is None:
        return None
    wf = _weather_features(events, field, target_day, target_order, wmap)
    if wf is None:
        return None

    regime = _regime_stats(events, regime_cutoff)
    flower_start = FLOWERING_STARTS.get(field)
    if flower_start is None:
        return None

    rec = {
        "target_day": target_day,
        "field_no": field,
        "target_order": target_order,
        "log_anchor": math.log(float(anchor_info["anchor"]) + TARGET_EPS),
        "anchor_is_field": anchor_info["anchor_is_field"],
        "anchor_age_days": anchor_info["anchor_age_days"],
        "has_lag2": anchor_info["has_lag2"],
        "same_trend1": anchor_info["same_trend1"],
        "has_lag3": anchor_info["has_lag3"],
        "same_trend2": anchor_info["same_trend2"],
        **wf,
        **regime,
        "flower_age_days": float((target_day - flower_start).days),
        # diagnostika
        "anchor_abc": float(anchor_info["anchor"]),
        "actual_abc": float(actual_abc) if actual_abc is not None else None,
    }
    if actual_abc is not None and actual_abc > 0:
        rec["y"] = _safe_log_ratio(float(actual_abc), float(anchor_info["anchor"]))
    else:
        rec["y"] = None
    return rec


def _make_historical_records(events: List[Event], actual_lookup: Dict[Tuple[date, int], dict], measured: Dict[date, dict]) -> List[dict]:
    records: List[dict] = []
    # Iga targeti record ehitatakse ainult talle eelnenud sündmustest.
    for (dd, field), row in sorted(actual_lookup.items()):
        if dd < WEATHER_START + timedelta(days=14):
            continue
        prior_events = [e for e in events if e.day < dd]
        rec = _build_record(
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


def _predict_with_model(model: dict, rec: dict) -> Optional[float]:
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

    # Cold-start prior: väga väikese treeninghulga korral ei tohi 3–4 ajaloolist
    # üleminekut sundida kogu süsteemi korduva eksponentsiaalse kukkumise/tõusu sisse.
    # Prior on state persistence (y=0); selle mõju hääbub automaatselt andmete lisandudes.
    n_train = float(model.get("n") or 0.0)
    learn_fraction = n_train / (n_train + SMALL_SAMPLE_PRIOR_ROWS)
    yhat *= learn_fraction

    anchor = float(rec["anchor_abc"])
    pred = (anchor + TARGET_EPS) * math.exp(yhat) - TARGET_EPS
    return max(0.0, float(pred))


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
    # Issue päeva õhtu: selle päeva sündmused on teada, kuid ainult usaldusväärne ABC on state-sensor.
    known_events = [e for e in all_events if e.day <= issue_day]
    state_events = _copy_events(known_events)

    # Train ainult issue-päevani teada olnud tegelike targetitega.
    train = [r for r in historical_records if r["target_day"] <= issue_day]
    model = _fit_model(train, issue_day)
    if model is None:
        return []

    wmap = _combined_weather(measured, forecast_map, issue_day)
    stubs = _future_schedule_stubs(actual_lookup, issue_day)
    out = []

    # Üks issue = üks külmutatud mudel. Rekursioonis lisame state'i ainult enda predicted ABC.
    for target_day, field, order in stubs:
        rec = _build_record(
            events=state_events,
            field=field,
            target_day=target_day,
            target_order=order,
            wmap=wmap,
            regime_cutoff=issue_day + timedelta(days=1),  # lubab actual signalid kuni issue-päevani
            actual_abc=None,
        )
        if rec is None:
            continue
        pred = _predict_with_model(model, rec)
        if pred is None:
            continue

        # Alles PÄRAST predictionit loeme actual_lookupist hindamise numbri.
        actual_row = actual_lookup.get((target_day, field))
        actual_abc = float(actual_row["_abc"]) if actual_row is not None else np.nan
        lead = (target_day - issue_day).days
        out.append({
            "issue_day": issue_day,
            "target_day": target_day,
            "lead": lead,
            "field": field,
            "order": order,
            "pred_abc": float(pred),
            "actual_abc": actual_abc,
            "anchor_abc": float(rec["anchor_abc"]),
            "anchor_is_field": int(rec["anchor_is_field"]),
            "growth_days": float(rec["growth_days"]),
            "regime1": float(rec["regime1"]),
            "regime3": float(rec["regime3"]),
            "flower_age_days": float(rec["flower_age_days"]),
            "gw_rad": float(rec["gw_rad"]),
            "gw_gdd10": float(rec["gw_gdd10"]),
            "run": run_label,
            "train_n": int(model["n"]),
        })

        # Simuleeritud tuleviku korje muutub järgmise sama issue-horisondi state'iks.
        state_events.append(Event(target_day, field, order, float(pred), True, "simulated"))
        state_events.sort(key=_event_sort_key)

    return out


def _daily_scores(field_df: pd.DataFrame) -> pd.DataFrame:
    if field_df.empty:
        return pd.DataFrame()
    daily = field_df.groupby(["issue_day", "target_day", "lead"], as_index=False).agg(
        pred_abc=("pred_abc", "sum"),
        actual_abc=("actual_abc", "sum"),
        n_fields=("field", "count"),
        train_n=("train_n", "max"),
    )
    daily["err"] = daily["pred_abc"] - daily["actual_abc"]
    daily["ape"] = daily["err"].abs() / daily["actual_abc"].clip(lower=0.1)
    daily["bias_pct"] = 100.0 * daily["err"] / daily["actual_abc"].clip(lower=0.1)
    return daily


def _lead_scores(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()
    rows = []
    for lead, g in daily.groupby("lead"):
        rows.append({
            "Lead": int(lead),
            "N": int(len(g)),
            "MAPE %": 100.0 * float(g["ape"].mean()),
            "±20% sees %": 100.0 * float((g["ape"] <= 0.20).mean()),
            "Bias %": float(g["bias_pct"].mean()),
            "MAE ABC": float((g["pred_abc"] - g["actual_abc"]).abs().mean()),
        })
    return pd.DataFrame(rows).sort_values("Lead")


# -----------------------------------------------------------------------------
# SELF TEST
# -----------------------------------------------------------------------------


def _self_test() -> None:
    # 60 päeva täielikku sünteetilist mõõdetud ilma.
    start = date(2026, 7, 1)
    measured = {}
    for i in range(60):
        dd = start + timedelta(days=i)
        measured[dd] = {
            "day": dd,
            "temp": 18.0 + 0.02 * i,
            "temp_min": 12.0,
            "temp_max": 24.0,
            "wind": 2.0,
            "rad": 18.0 + 0.05 * i,
            "rh": 75.0,
            "rain": 0.5,
            "et0": 3.0,
            "source": "measured",
        }

    # Piisavalt evente ja transition-targeteid, sh üks teadmata ABC event.
    events = [
        Event(date(2026, 7, 10), 1, 1, 8.0, True, "actual"),
        Event(date(2026, 7, 11), 2, 1, 9.0, True, "actual"),
        Event(date(2026, 7, 15), 1, 1, 9.0, True, "actual"),
        Event(date(2026, 7, 16), 2, 1, 10.0, True, "actual"),
        Event(date(2026, 7, 20), 1, 1, 7.5, True, "actual"),
        Event(date(2026, 7, 21), 2, 1, 8.0, True, "actual"),
        Event(date(2026, 7, 22), 1, 1, None, False, "estimated_event"),
    ]
    events.sort(key=_event_sort_key)

    # State peab pärast teadmata ABC eventi langema pooled-anchorile, mitte kasutama vana field-ABC-d.
    ai = _state_anchor(events, 1, date(2026, 7, 27))
    assert ai is not None and ai["anchor_is_field"] == 0.0

    # Kombineeritud ilm: issue-päev measured, tulevik forecast.
    fc = {d: dict(v, source="ecmwf_forecast") for d, v in measured.items() if d > date(2026, 7, 22)}
    cm = _combined_weather(measured, fc, date(2026, 7, 22))
    assert cm[date(2026, 7, 22)]["source"] == "measured"
    assert cm[date(2026, 7, 23)]["source"] == "ecmwf_forecast"

    # Record ei vaja actual targetit ning flower-age on olemas.
    rec = _build_record(
        events=events,
        field=2,
        target_day=date(2026, 7, 26),
        target_order=1,
        wmap=cm,
        regime_cutoff=date(2026, 7, 23),
        actual_abc=None,
    )
    assert rec is not None
    assert rec["y"] is None
    assert rec["flower_age_days"] > 0
    assert all(k in rec for k in FEATURES)

    # Ridge numbriline smoke-test.
    fake = []
    for i in range(12):
        rr = dict(rec)
        rr["target_day"] = date(2026, 7, 1) + timedelta(days=i)
        rr["y"] = 0.02 * math.sin(i / 3.0)
        for c in FEATURES:
            rr[c] = float(rr[c]) + 0.0005 * i
        fake.append(rr)
    model = _fit_model(fake, date(2026, 7, 22))
    assert model is not None and model["n"] == 12
    pred = _predict_with_model(model, rec)
    assert pred is not None and math.isfinite(pred) and pred >= 0

    # Schedule stub ei tohi sisaldada saaginumbreid.
    lookup = {(date(2026, 7, 23), 3): {"_order": 1, "_abc": 99.0}}
    stubs = _future_schedule_stubs(lookup, date(2026, 7, 22))
    assert stubs == [(date(2026, 7, 23), 3, 1)]

    print("LAB-136 SELF-TEST OK")


# -----------------------------------------------------------------------------
# STREAMLIT UI
# -----------------------------------------------------------------------------


def main() -> None:
    import streamlit as st
    import db

    st.set_page_config(page_title="KurgiMootor LAB-136", layout="wide")
    st.error("🧪 LAB-136 STATE AJAMASIN · READ-ONLY · 22.07 → +1…+9")
    st.title("KurgiMootor · STATE ajamasin")
    st.caption(
        "Üks fikseeritud ABC-mudel. Korje = taime state-sensor; ilm = state'i muutus. "
        "Replay algab 22.07.2026 ja kasutab tuleviku jaoks arhiveeritud ECMWF IFS HRES 9 km prognoosijookse."
    )

    with st.expander("Katse reeglid ja lukud", expanded=False):
        st.markdown(
            """
- **Prognoosi hetk:** iga replay-päeva õhtu; sama päeva teadaolev korje ja mõõdetud ilm võivad olla sees.
- **Tuleviku ilm:** ECMWF IFS HRES 9 km, eelistus 12 UTC issue-päeva jooks; fallback 06/00 UTC.
- **+1…+9:** üks issue-mudel külmutatakse; tuleviku tegelik saak ei õpeta sama issue prognoosi.
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
    if not run_requested and "lab136_results" not in st.session_state:
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
        st.session_state["lab136_results"] = {
            "field": field_df,
            "daily": daily_df,
            "errors": weather_errors,
            "runs": run_used,
        }

    result = st.session_state.get("lab136_results")
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
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Kõik lead'id · MAPE", f"{overall_mape:.1f}%")
    c2.metric("Kõik lead'id · ±20%", f"{overall_hit:.0f}%")
    for col, lead in zip((c3, c4), (1, 5)):
        row = lead_df[lead_df["Lead"] == lead]
        col.metric(f"+{lead} päeva MAPE", "—" if row.empty else f"{float(row.iloc[0]['MAPE %']):.1f}%")

    st.dataframe(
        lead_df.style.format({
            "MAPE %": "{:.1f}%",
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
    issue_table["STATE ABC"] = issue_table["pred_abc"]
    issue_table["Tegelik ABC"] = issue_table["actual_abc"]
    issue_table["Viga %"] = 100.0 * issue_table["ape"]
    issue_table["Treeningridu"] = issue_table["train_n"]
    st.dataframe(
        issue_table[["Prognoos tehtud", "Sihtpäev", "Lead", "STATE ABC", "Tegelik ABC", "Viga %", "Treeningridu"]]
        .style.format({"STATE ABC": "{:.1f}", "Tegelik ABC": "{:.1f}", "Viga %": "{:.1f}%"}),
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
    tdf["STATE ABC"] = tdf["pred_abc"]
    tdf["Tegelik ABC"] = tdf["actual_abc"]
    tdf["Viga %"] = 100.0 * tdf["ape"]
    st.dataframe(
        tdf[["Prognoos tehtud", "Lead", "STATE ABC", "Tegelik ABC", "Viga %"]]
        .style.format({"STATE ABC": "{:.1f}", "Tegelik ABC": "{:.1f}", "Viga %": "{:.1f}%"}),
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("Põldude detail · mida STATE nägi"):
        fshow = field_df.copy()
        fshow["Prognoos"] = fshow["issue_day"].map(lambda d: pd.Timestamp(d).strftime("%d.%m"))
        fshow["Siht"] = fshow["target_day"].map(lambda d: pd.Timestamp(d).strftime("%d.%m"))
        fshow["Lead"] = fshow["lead"].map(lambda x: f"+{int(x)}")
        fshow["Põld"] = fshow["field"]
        fshow["STATE ABC"] = fshow["pred_abc"]
        fshow["Tegelik ABC"] = fshow["actual_abc"]
        fshow["Anchor"] = fshow["anchor_abc"]
        fshow["Oma põld anchor"] = fshow["anchor_is_field"].map({1: "jah", 0: "plokk"})
        fshow["Kasvuaeg p"] = fshow["growth_days"]
        fshow["Režiim1 %"] = 100.0 * (np.exp(fshow["regime1"]) - 1.0)
        fshow["Režiim3 %"] = 100.0 * (np.exp(fshow["regime3"]) - 1.0)
        fshow["Õitsemisest p"] = fshow["flower_age_days"]
        st.dataframe(
            fshow[[
                "Prognoos", "Siht", "Lead", "Põld", "STATE ABC", "Tegelik ABC", "Anchor",
                "Oma põld anchor", "Kasvuaeg p", "Režiim1 %", "Režiim3 %", "Õitsemisest p", "train_n",
            ]].style.format({
                "STATE ABC": "{:.1f}", "Tegelik ABC": "{:.1f}", "Anchor": "{:.1f}",
                "Kasvuaeg p": "{:.2f}", "Režiim1 %": "{:+.1f}%", "Režiim3 %": "{:+.1f}%",
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
        st.success("✅ Simuleeritud tuleviku saak võib olla järgmise tulevase korje state, actual ei või")
        st.info(f"Mudel: {FORECAST_MODEL} · ridge α={RIDGE_ALPHA:g} · recency half-life={RECENCY_HALFLIFE_DAYS:g} p · min train={MIN_TRAIN_ROWS} · cold-start prior={SMALL_SAMPLE_PRIOR_ROWS:g} rida")
        if errors:
            st.warning(f"ECMWF jooksu ei saadud {len(errors)} issue-päeval.")
            st.dataframe(pd.DataFrame([{"Päev": d.strftime("%d.%m"), "Viga": e} for d, e in errors]), use_container_width=True, hide_index=True)
        else:
            st.success("✅ Kõigi issue-päevade ECMWF jooksud saadi kätte.")

    st.caption(
        f"{LAB_VERSION} · read-only · replay {REPLAY_START.isoformat()} → {last_actual_day.isoformat()} · "
        "ABC only · ECMWF IFS HRES single-run archive"
    )


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        _self_test()
    else:
        main()
