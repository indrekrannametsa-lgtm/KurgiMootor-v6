# BUILD: PARNU_WEATHER_ET0_2026_08_07
from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from statistics import mean
from math import acos, cos, exp, log, pi, sin, sqrt, tan
from typing import Any, Dict, Iterable, List
from zoneinfo import ZoneInfo

import requests

import db


ESTONIA = ZoneInfo("Europe/Tallinn")
OFFICIAL_HOURLY = "https://keskkonnaandmed.envir.ee/f_kliima_tund"
OFFICIAL_DAILY = "https://keskkonnaandmed.envir.ee/f_kliima_paev"
OPEN_METEO = "https://api.open-meteo.com/v1/forecast"
HEADERS = {"Accept-Profile": "apijahiala", "Accept": "application/json"}
FARM_LAT = 58.13
FARM_LON = 24.50
FARM_ELEVATION_M = 5.0


class WeatherError(RuntimeError):
    pass


class WeatherService:
    def __init__(self, http: Any = None):
        self.http = http or requests.Session()

    @staticmethod
    def _month_ranges(start_day: date, end_day: date) -> Iterable[tuple[int, int]]:
        current = date(start_day.year, start_day.month, 1)
        final = date(end_day.year, end_day.month, 1)
        while current <= final:
            yield current.year, current.month
            current = date(current.year + (current.month == 12), 1 if current.month == 12 else current.month + 1, 1)

    def _get_json(self, url: str, params: Any, headers: Dict[str, str] | None = None) -> Any:
        response = self.http.get(url, params=params, headers=headers, timeout=45)
        if not response.ok:
            raise WeatherError(f"API {response.status_code}: {response.text}")
        return response.json()

    def _official_rows(self, url: str, station: str, element_code: str, start_day: date, end_day: date) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        for year, month in self._month_ranges(start_day, end_day):
            params = {
                "jaam_nimi": f"ilike.*{station}*",
                "element_kood": f"eq.{element_code}",
                "aasta": f"eq.{year}",
                "kuu": f"eq.{month}",
                "select": "jaam_nimi,aasta,kuu,paev,tund,vaartus,element_kood",
                "order": "paev.asc,tund.asc",
            }
            payload = self._get_json(url, params, HEADERS)
            if not isinstance(payload, list):
                raise WeatherError("Keskkonnaandmete API ei tagastanud loendit.")
            result.extend(payload)
        return result

    @staticmethod
    def _as_float(value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _hourly_local_day(row: Dict[str, Any]) -> date | None:
        try:
            utc_dt = datetime(
                int(row["aasta"]), int(row["kuu"]), int(row["paev"]), int(row.get("tund") or 0),
                tzinfo=timezone.utc,
            )
            return utc_dt.astimezone(ESTONIA).date()
        except Exception:
            return None

    def _parnu_daily(self, start_day: date, end_day: date) -> Dict[str, Dict[str, float]]:
        temp_rows = self._official_rows(OFFICIAL_HOURLY, "Pärnu", "TA", start_day - timedelta(days=1), end_day)
        wind_rows = self._official_rows(OFFICIAL_HOURLY, "Pärnu", "WS10M", start_day - timedelta(days=1), end_day)

        temps: Dict[date, List[float]] = defaultdict(list)
        winds: Dict[date, List[float]] = defaultdict(list)
        for row in temp_rows:
            d = self._hourly_local_day(row)
            value = self._as_float(row.get("vaartus"))
            if d and value is not None and start_day <= d <= end_day:
                temps[d].append(value)
        for row in wind_rows:
            d = self._hourly_local_day(row)
            value = self._as_float(row.get("vaartus"))
            if d and value is not None and start_day <= d <= end_day:
                winds[d].append(value)

        result: Dict[str, Dict[str, float]] = {}
        d = start_day
        while d <= end_day:
            item: Dict[str, float] = {}
            if temps.get(d):
                item["t_min"] = min(temps[d])
                item["t_max"] = max(temps[d])
            if winds.get(d):
                item["wind_avg"] = mean(winds[d])
            result[d.isoformat()] = item
            d += timedelta(days=1)
        return result

    def _parnu_hourly_daily_element(
        self,
        element_code: str,
        start_day: date,
        end_day: date,
        reducer: str = "mean",
    ) -> Dict[str, float]:
        """Koondab Pärnu ametliku tunni-elemendi kohaliku kalendripäeva kaupa.

        Toetatud reducerid:
        - mean: tunni hetkväärtuste keskmine (nt RH)
        - sum: tunni summade summa (nt PR1H, SDUR1H)
        """
        rows = self._official_rows(
            OFFICIAL_HOURLY,
            "Pärnu",
            element_code,
            start_day - timedelta(days=1),
            end_day,
        )
        values: Dict[date, List[float]] = defaultdict(list)
        for row in rows:
            d = self._hourly_local_day(row)
            value = self._as_float(row.get("vaartus"))
            if d and value is not None and start_day <= d <= end_day:
                values[d].append(value)

        result: Dict[str, float] = {}
        d = start_day
        while d <= end_day:
            vals = values.get(d) or []
            if vals:
                result[d.isoformat()] = float(sum(vals) if reducer == "sum" else mean(vals))
            d += timedelta(days=1)
        return result

    @staticmethod
    def _radiation_from_sunshine_hours(day: date, sunshine_hours: float | None) -> float | None:
        """Ajutine Rs hinnang ametlikust SDUR1H päikesepaistest (FAO-56 Ångström–Prescott).

        Rs = (0.25 + 0.50 * n/N) * Ra
        Kus n on mõõdetud päikesepaiste kestus ja N astronoomiline päevapikkus.
        Kasutatakse ainult siis, kui valideeritud DRQS pole veel avaldatud.
        """
        if sunshine_hours is None:
            return None
        n = max(0.0, float(sunshine_hours))
        j = day.timetuple().tm_yday
        phi = FARM_LAT * pi / 180.0
        dr = 1.0 + 0.033 * cos((2.0 * pi / 365.0) * j)
        solar_declination = 0.409 * sin((2.0 * pi / 365.0) * j - 1.39)
        sunset_angle = acos(max(-1.0, min(1.0, -tan(phi) * tan(solar_declination))))
        ra = (
            (24.0 * 60.0 / pi)
            * 0.0820
            * dr
            * (
                sunset_angle * sin(phi) * sin(solar_declination)
                + cos(phi) * cos(solar_declination) * sin(sunset_angle)
            )
        )
        N = 24.0 * sunset_angle / pi
        if N <= 0:
            return None
        n = min(n, N)
        rs = (0.25 + 0.50 * (n / N)) * ra
        return round(max(0.0, rs), 3)

    def _parnu_daily_element(self, element_code: str, start_day: date, end_day: date) -> Dict[str, float]:
        """Loeb Pärnu valideeritud ööpäevaelemendi (nt DRQS, DRH08, DPREC)."""
        rows: List[Dict[str, Any]] = []
        for year, month in self._month_ranges(start_day, end_day):
            params = {
                "jaam_kood": "eq.AJPARN01",
                "element_kood": f"eq.{element_code}",
                "aasta": f"eq.{year}",
                "kuu": f"eq.{month}",
                "select": "aasta,kuu,paev,vaartus,element_kood",
                "order": "paev.asc",
            }
            payload = self._get_json(OFFICIAL_DAILY, params, HEADERS)
            if not isinstance(payload, list):
                raise WeatherError("Keskkonnaandmete päeva-API ei tagastanud loendit.")
            rows.extend(payload)
        result: Dict[str, float] = {}
        for row in rows:
            try:
                d = date(int(row["aasta"]), int(row["kuu"]), int(row["paev"]))
            except Exception:
                continue
            value = self._as_float(row.get("vaartus"))
            if value is not None and start_day <= d <= end_day:
                result[d.isoformat()] = value
        return result


    @staticmethod
    def calculate_et0_mm(
        day: date,
        t_min: float | None,
        t_max: float | None,
        wind_avg_10m: float | None,
        radiation_mj_m2: float | None,
        humidity_avg_pct: float | None,
    ) -> float | None:
        """FAO-56 Penman-Monteith reference ET0 (mm/day).

        Inputs: daily Tmin/Tmax (C), average 10 m wind (m/s), global shortwave
        radiation (MJ/m2/day) and average relative humidity (%). The 10 m wind
        is converted to the FAO reference height of 2 m. With only daily mean
        RH available, actual vapour pressure is approximated from RHmean and
        mean saturation vapour pressure.
        """
        values = (t_min, t_max, wind_avg_10m, radiation_mj_m2, humidity_avg_pct)
        if any(v is None for v in values):
            return None

        tmin = float(t_min)
        tmax = float(t_max)
        u10 = max(0.0, float(wind_avg_10m))
        rs = max(0.0, float(radiation_mj_m2))
        rh = min(100.0, max(0.0, float(humidity_avg_pct)))
        tmean = (tmin + tmax) / 2.0

        def e0(temp_c: float) -> float:
            return 0.6108 * exp((17.27 * temp_c) / (temp_c + 237.3))

        es = (e0(tmax) + e0(tmin)) / 2.0
        ea = (rh / 100.0) * es
        delta = 4098.0 * e0(tmean) / ((tmean + 237.3) ** 2)

        pressure = 101.3 * (((293.0 - 0.0065 * FARM_ELEVATION_M) / 293.0) ** 5.26)
        gamma = 0.000665 * pressure

        # Convert wind measured at 10 m to the standard 2 m height.
        u2 = u10 * 4.87 / log(67.8 * 10.0 - 5.42)

        j = day.timetuple().tm_yday
        phi = FARM_LAT * pi / 180.0
        dr = 1.0 + 0.033 * cos((2.0 * pi / 365.0) * j)
        solar_declination = 0.409 * sin((2.0 * pi / 365.0) * j - 1.39)
        sunset_angle = acos(max(-1.0, min(1.0, -tan(phi) * tan(solar_declination))))
        ra = (
            (24.0 * 60.0 / pi)
            * 0.0820
            * dr
            * (
                sunset_angle * sin(phi) * sin(solar_declination)
                + cos(phi) * cos(solar_declination) * sin(sunset_angle)
            )
        )
        rso = (0.75 + 2e-5 * FARM_ELEVATION_M) * ra
        rns = (1.0 - 0.23) * rs
        cloud_term = 0.05 if rso <= 0 else max(0.05, 1.35 * min(rs / rso, 1.0) - 0.35)
        rnl = (
            4.903e-9
            * (((tmax + 273.16) ** 4 + (tmin + 273.16) ** 4) / 2.0)
            * (0.34 - 0.14 * sqrt(max(ea, 0.0)))
            * cloud_term
        )
        rn = rns - rnl

        numerator = (
            0.408 * delta * rn
            + gamma * (900.0 / (tmean + 273.0)) * u2 * max(es - ea, 0.0)
        )
        denominator = delta + gamma * (1.0 + 0.34 * u2)
        if denominator <= 0:
            return None
        return round(max(0.0, numerator / denominator), 3)

    @staticmethod
    def _validate_measured(
        t_min: float | None,
        t_max: float | None,
        wind_avg: float | None,
        radiation: float | None,
        humidity_avg: float | None,
        precipitation: float | None,
        et0_mm: float | None,
    ) -> List[str]:
        problems: List[str] = []
        if t_min is None:
            problems.append("Pärnu miinimumtemperatuur puudub")
        elif not -50 <= t_min <= 50:
            problems.append("miinimumtemperatuur on ebarealistlik")
        if t_max is None:
            problems.append("Pärnu maksimumtemperatuur puudub")
        elif not -50 <= t_max <= 50:
            problems.append("maksimumtemperatuur on ebarealistlik")
        if t_min is not None and t_max is not None and t_min > t_max:
            problems.append("temperatuuri min/max on vahetuses")
        if wind_avg is None:
            problems.append("Pärnu tuul puudub")
        elif not 0 <= wind_avg <= 50:
            problems.append("tuul on ebarealistlik")
        if radiation is None:
            problems.append("Pärnu radiatsioon puudub")
        elif not 0 <= radiation <= 45:
            problems.append("radiatsioon on ebarealistlik")
        if humidity_avg is None:
            problems.append("Pärnu õhuniiskus puudub")
        elif not 0 <= humidity_avg <= 100:
            problems.append("õhuniiskus on ebarealistlik")
        if precipitation is None:
            problems.append("Pärnu sademed puuduvad")
        elif not 0 <= precipitation <= 500:
            problems.append("sademed on ebarealistlikud")
        if et0_mm is None:
            problems.append("ET0 ei arvutunud")
        elif not 0 <= et0_mm <= 20:
            problems.append("ET0 on ebarealistlik")
        return problems

    def refresh_measured(self, start_day: date, end_day: date) -> Dict[str, int]:
        parnu = self._parnu_daily(start_day, end_day)
        radiation_daily = self._parnu_daily_element("DRQS", start_day, end_day)
        humidity_daily = self._parnu_daily_element("DRH08", start_day, end_day)
        precipitation_daily = self._parnu_daily_element("DPREC", start_day, end_day)

        saved = checked = temporary = 0
        d = start_day
        while d <= end_day:
            key = d.isoformat()
            local = parnu.get(key, {})
            t_min = local.get("t_min")
            t_max = local.get("t_max")
            wind_avg = local.get("wind_avg")

            rad = radiation_daily.get(key)
            humidity_avg = humidity_daily.get(key)
            precipitation_mm = precipitation_daily.get(key)

            # Ainult selle kuupäeva ENNE säilitatud forecast võib täita veel
            # avaldamata Pärnu päevaelemente. Minevikku uut forecast'i ei küsita.
            forecast_snapshot = db.get_weather_forecast_snapshot(d)
            forecast_parts: List[str] = []
            if forecast_snapshot:
                if rad is None and forecast_snapshot.get("radiation_mj_m2") is not None:
                    rad = self._as_float(forecast_snapshot.get("radiation_mj_m2"))
                    forecast_parts.append("radiatsioon")
                if humidity_avg is None and forecast_snapshot.get("humidity_avg_pct") is not None:
                    humidity_avg = self._as_float(forecast_snapshot.get("humidity_avg_pct"))
                    forecast_parts.append("niiskus")
                if precipitation_mm is None and forecast_snapshot.get("precipitation_mm") is not None:
                    precipitation_mm = self._as_float(forecast_snapshot.get("precipitation_mm"))
                    forecast_parts.append("sademed")

            # ET0 arvutame hübriidrea enda sisenditest: mõõdetud Tmin/Tmax/tuul +
            # vajadusel varem salvestatud forecast'i rad/RH.
            et0_mm = self.calculate_et0_mm(d, t_min, t_max, wind_avg, rad, humidity_avg)

            raw_problems = self._validate_measured(
                t_min, t_max, wind_avg,
                radiation_daily.get(key),
                humidity_daily.get(key),
                precipitation_daily.get(key),
                self.calculate_et0_mm(
                    d, t_min, t_max, wind_avg,
                    radiation_daily.get(key),
                    humidity_daily.get(key),
                ),
            )

            # "checked" tähendab nüüd ainult täielikult ametlikku Pärnu päeva.
            fully_official = (
                t_min is not None
                and t_max is not None
                and wind_avg is not None
                and key in radiation_daily
                and key in humidity_daily
                and key in precipitation_daily
                and not raw_problems
            )

            usable_problems = self._validate_measured(
                t_min, t_max, wind_avg, rad, humidity_avg, precipitation_mm, et0_mm
            )

            if fully_official:
                check_message = "Kontrollitud · valideeritud Pärnu päevaelemendid"
            elif not usable_problems and forecast_parts:
                temporary += 1
                check_message = (
                    "Ajutine · Pärnu T/tuul + varem salvestatud prognoos: "
                    + ", ".join(forecast_parts)
                )
            else:
                check_message = "; ".join(usable_problems) if usable_problems else "Puudulik"

            # DB-kihis on põhiline regressioonikaitse. Siin jääb payload
            # teadlikult päeva hetkeallikate järgi ausaks; puudulik uus vastus
            # ei tohi db.upsert_weather kaudu head checked=True rida rikkuda.
            db.upsert_weather({
                "weather_date": key,
                "data_kind": "measured",
                "temp_min_c": t_min,
                "temp_max_c": t_max,
                "wind_avg_ms": wind_avg,
                "radiation_mj_m2": rad,
                "humidity_avg_pct": humidity_avg,
                "precipitation_mm": precipitation_mm,
                "et0_mm": et0_mm,
                "source_station": "Pärnu",
                "radiation_station": (
                    "Pärnu" if key in radiation_daily
                    else ("Open-Meteo (varasem salvestatud prognoos)" if "radiatsioon" in forecast_parts else "Pärnu")
                ),
                "checked": bool(fully_official),
                "check_message": check_message,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            })
            saved += 1
            checked += int(fully_official)
            d += timedelta(days=1)

        return {"saved": saved, "checked": checked, "temporary": temporary}

    def refresh_forecast(self, start_day: date, days: int = 10) -> Dict[str, int]:
        end_day = start_day + timedelta(days=days - 1)
        params = {
            "latitude": FARM_LAT,
            "longitude": FARM_LON,
            "timezone": "Europe/Tallinn",
            "start_date": start_day.isoformat(),
            "end_date": end_day.isoformat(),
            "daily": "temperature_2m_min,temperature_2m_max,shortwave_radiation_sum,precipitation_sum",
            "hourly": "wind_speed_10m,relative_humidity_2m",
            "wind_speed_unit": "ms",
        }
        payload = self._get_json(OPEN_METEO, params)
        daily = payload.get("daily") or {}
        hourly = payload.get("hourly") or {}
        hourly_wind: Dict[str, List[float]] = defaultdict(list)
        hourly_humidity: Dict[str, List[float]] = defaultdict(list)
        for ts, value in zip(hourly.get("time", []), hourly.get("wind_speed_10m", [])):
            if value is not None:
                hourly_wind[str(ts)[:10]].append(float(value))
        for ts, value in zip(hourly.get("time", []), hourly.get("relative_humidity_2m", [])):
            if value is not None:
                hourly_humidity[str(ts)[:10]].append(float(value))
        saved = 0
        dates = daily.get("time", [])
        for i, key in enumerate(dates):
            t_min = daily.get("temperature_2m_min", [None] * len(dates))[i]
            t_max = daily.get("temperature_2m_max", [None] * len(dates))[i]
            rad = daily.get("shortwave_radiation_sum", [None] * len(dates))[i]
            precipitation_mm = daily.get("precipitation_sum", [None] * len(dates))[i]
            wind = mean(hourly_wind[key]) if hourly_wind.get(key) else None
            humidity_avg = mean(hourly_humidity[key]) if hourly_humidity.get(key) else None
            et0_mm = self.calculate_et0_mm(
                date.fromisoformat(key), t_min, t_max, wind, rad, humidity_avg
            )
            problems = []
            if t_min is None or t_max is None: problems.append("temperatuuri prognoos puudub")
            if wind is None: problems.append("tuule prognoos puudub")
            if rad is None: problems.append("radiatsiooni prognoos puudub")
            if humidity_avg is None: problems.append("õhuniiskuse prognoos puudub")
            if precipitation_mm is None: problems.append("sademete prognoos puudub")
            if et0_mm is None: problems.append("ET0 prognoos ei arvutunud")
            forecast_payload = {
                "weather_date": key,
                "data_kind": "forecast",
                "temp_min_c": t_min,
                "temp_max_c": t_max,
                "wind_avg_ms": wind,
                "radiation_mj_m2": rad,
                "humidity_avg_pct": humidity_avg,
                "precipitation_mm": precipitation_mm,
                "et0_mm": et0_mm,
                "source_station": "Open-Meteo",
                "radiation_station": "Open-Meteo",
                "checked": not problems,
                "check_message": "; ".join(problems) if problems else "Prognoos olemas",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }

            # Säilita forecast eraldi enne weather_daily upsert'i. Nii ei kao
            # eilse päeva prognoos ära, kui mõõdetud Pärnu rida sama kuupäeva üle kirjutab.
            db.save_weather_forecast_snapshot(date.fromisoformat(key), forecast_payload)
            db.upsert_weather(forecast_payload)
            saved += 1
        return {"saved": saved}


    def test_sources(self, today: date) -> Dict[str, Any]:
        """Kontrollib Pärnu mõõteallikaid ja Open-Meteo prognoosi ilma Supabase'i kirjutamata."""
        measured_day = today - timedelta(days=1)
        temp_rows = self._official_rows(OFFICIAL_HOURLY, "Pärnu", "TA", measured_day - timedelta(days=1), measured_day)
        wind_rows = self._official_rows(OFFICIAL_HOURLY, "Pärnu", "WS10M", measured_day - timedelta(days=1), measured_day)
        rh_rows = self._official_rows(OFFICIAL_HOURLY, "Pärnu", "RH", measured_day - timedelta(days=1), measured_day)
        pr1h_rows = self._official_rows(OFFICIAL_HOURLY, "Pärnu", "PR1H", measured_day - timedelta(days=1), measured_day)
        sdur_rows = self._official_rows(OFFICIAL_HOURLY, "Pärnu", "SDUR1H", measured_day - timedelta(days=1), measured_day)

        temp_rows_for_day = [r for r in temp_rows if self._hourly_local_day(r) == measured_day]
        wind_rows_for_day = [r for r in wind_rows if self._hourly_local_day(r) == measured_day]
        rh_rows_for_day = [r for r in rh_rows if self._hourly_local_day(r) == measured_day]
        pr1h_rows_for_day = [r for r in pr1h_rows if self._hourly_local_day(r) == measured_day]
        sdur_rows_for_day = [r for r in sdur_rows if self._hourly_local_day(r) == measured_day]

        radiation = self._parnu_daily_element("DRQS", measured_day, measured_day)
        humidity = self._parnu_daily_element("DRH08", measured_day, measured_day)
        precipitation = self._parnu_daily_element("DPREC", measured_day, measured_day)
        key = measured_day.isoformat()

        forecast_dates: List[str] = []
        forecast_error = None
        try:
            forecast = self._get_json(OPEN_METEO, {
                "latitude": FARM_LAT,
                "longitude": FARM_LON,
                "timezone": "Europe/Tallinn",
                "start_date": today.isoformat(),
                "end_date": today.isoformat(),
                "daily": "temperature_2m_min,temperature_2m_max,shortwave_radiation_sum,precipitation_sum",
                "hourly": "wind_speed_10m,relative_humidity_2m",
                "wind_speed_unit": "ms",
            })
            forecast_dates = list((forecast.get("daily") or {}).get("time") or [])
        except Exception as exc:
            forecast_error = str(exc)

        temp_values = [self._as_float(r.get("vaartus")) for r in temp_rows_for_day]
        temp_values = [v for v in temp_values if v is not None]
        wind_values = [self._as_float(r.get("vaartus")) for r in wind_rows_for_day]
        wind_values = [v for v in wind_values if v is not None]
        test_et0 = self.calculate_et0_mm(
            measured_day,
            min(temp_values) if temp_values else None,
            max(temp_values) if temp_values else None,
            mean(wind_values) if wind_values else None,
            radiation.get(key),
            humidity.get(key),
        )
        measured_ok = bool(
            temp_rows_for_day
            and wind_rows_for_day
            and key in radiation
            and key in humidity
            and key in precipitation
            and test_et0 is not None
        )
        return {
            "measured_day": key,
            "parnu_temperature_rows": len(temp_rows_for_day),
            "parnu_wind_rows": len(wind_rows_for_day),
            "parnu_rh_hourly_rows": len(rh_rows_for_day),
            "parnu_pr1h_hourly_rows": len(pr1h_rows_for_day),
            "parnu_sdur1h_hourly_rows": len(sdur_rows_for_day),
            "parnu_radiation_rows": int(key in radiation),
            "parnu_humidity_rows": int(key in humidity),
            "parnu_precipitation_rows": int(key in precipitation),
            "et0_mm": test_et0,
            "forecast_days": len(forecast_dates),
            "forecast_error": forecast_error,
            "measured_sources_ok": measured_ok,
            "ok": bool(measured_ok and forecast_dates),
        }

    def _refresh_measured_incremental(self, today: date) -> Dict[str, Any]:
        season_start = date(today.year, 6, 15)
        # Ametlikud päevad võivad saabuda viitega. Automaatika loeb kuni eilseni.
        target_end = today - timedelta(days=1)
        if target_end < season_start:
            return {"saved": 0, "checked": 0, "ranges": []}

        missing = db.get_incomplete_measured_dates(season_start, target_end)

        # Viimased 3 lõppenud päeva kontrollime alati uuesti. Nii saab ajutine salvestatud prognoos
        # järgmisel käivitamisel automaatselt asenduda DRQS/DRH08/DPREC valideeritud
        # päevaelementidega, ilma et kasutaja peaks midagi käsitsi puhastama.
        recent_start = max(season_start, target_end - timedelta(days=2))
        recent_days: List[date] = []
        cursor = recent_start
        while cursor <= target_end:
            recent_days.append(cursor)
            cursor += timedelta(days=1)

        missing = sorted(set(missing + recent_days))
        if not missing:
            return {"saved": 0, "checked": 0, "ranges": []}

        # Koondame järjestikused puuduvad/uuesti kontrollitavad päevad vahemikeks, et API-kutseid oleks vähe,
        # kuid juba kontrollitud ridu ei kirjutataks uuesti üle.
        ranges: List[tuple[date, date]] = []
        range_start = range_end = missing[0]
        for current in missing[1:]:
            if current == range_end + timedelta(days=1):
                range_end = current
            else:
                ranges.append((range_start, range_end))
                range_start = range_end = current
        ranges.append((range_start, range_end))

        saved = checked = 0
        labels: List[str] = []
        for start_day, end_day in ranges:
            result = self.refresh_measured(start_day, end_day)
            saved += int(result.get("saved", 0))
            checked += int(result.get("checked", 0))
            labels.append(f"{start_day.isoformat()}…{end_day.isoformat()}")
        return {"saved": saved, "checked": checked, "ranges": labels}

    def refresh_all(self, today: date) -> Dict[str, Any]:
        errors: List[str] = []
        measured: Dict[str, Any] = {"saved": 0, "checked": 0, "range": None}
        forecast: Dict[str, Any] = {"saved": 0}

        try:
            measured = self._refresh_measured_incremental(today)
        except Exception as exc:
            errors.append(f"Mõõdetud ilm: {exc}")

        try:
            forecast = self.refresh_forecast(today, 10)
        except Exception as exc:
            errors.append(f"Prognoos: {exc}")

        now = datetime.now(ESTONIA).isoformat(timespec="seconds")
        db.set_app_setting("weather_last_refresh_at", now)
        db.set_app_setting("weather_last_refresh_date", today.isoformat())
        db.set_app_setting("weather_last_error", " | ".join(errors))
        db.set_app_setting(
            "weather_last_result",
            f"Mõõdetud {measured.get('saved', 0)} päeva; prognoos {forecast.get('saved', 0)} päeva",
        )
        result: Dict[str, Any] = {"measured": measured, "forecast": forecast}
        if errors:
            result["error"] = " | ".join(errors)
        return result

    def auto_refresh_if_needed(self, today: date) -> Dict[str, Any]:
        """Uuendab ilma kõige rohkem üks kord kohaliku kalendripäeva jooksul."""
        if db.get_app_setting("weather_last_refresh_date", "") == today.isoformat():
            return {"skipped": True, "reason": "already_refreshed_today"}
        return self.safe_refresh_all(today)

    def safe_refresh_all(self, today: date) -> Dict[str, Any]:
        try:
            return self.refresh_all(today)
        except Exception as exc:
            # Märgime katse tehtuks, et iga Streamliti rerun ei hakkaks sama API-t uuesti pommitama.
            db.set_app_setting("weather_last_refresh_date", today.isoformat())
            db.set_app_setting("weather_last_error", str(exc))
            return {"error": str(exc)}

