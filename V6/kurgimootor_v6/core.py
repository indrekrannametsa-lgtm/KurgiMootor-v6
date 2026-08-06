from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from statistics import mean
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

    def _get_json(self, url: str, params: Dict[str, Any], headers: Dict[str, str] | None = None) -> Any:
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

    def _haademeeste_daily(self, start_day: date, end_day: date) -> Dict[str, Dict[str, float]]:
        temp_rows = self._official_rows(OFFICIAL_HOURLY, "Häädemeeste", "TA", start_day - timedelta(days=1), end_day)
        wind_rows = self._official_rows(OFFICIAL_HOURLY, "Häädemeeste", "WS10M", start_day - timedelta(days=1), end_day)
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

    def _parnu_radiation(self, start_day: date, end_day: date) -> Dict[str, float]:
        rows: List[Dict[str, Any]] = []
        for year, month in self._month_ranges(start_day, end_day):
            params = {
                "jaam_nimi": "like.Pärnu",
                "element_kood": "eq.DRQS",
                "aasta": f"eq.{year}",
                "kuu": f"eq.{month}",
                "select": "aasta,kuu,paev,vaartus,element_kood",
                "order": "paev.asc",
            }
            payload = self._get_json(OFFICIAL_DAILY, params, HEADERS)
            if isinstance(payload, list):
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
    def _validate_measured(t_min: float | None, t_max: float | None, wind_avg: float | None, radiation: float | None) -> List[str]:
        problems: List[str] = []
        if t_min is None:
            problems.append("Häädemeeste miinimumtemperatuur puudub")
        elif not -50 <= t_min <= 50:
            problems.append("miinimumtemperatuur on ebarealistlik")
        if t_max is None:
            problems.append("Häädemeeste maksimumtemperatuur puudub")
        elif not -50 <= t_max <= 50:
            problems.append("maksimumtemperatuur on ebarealistlik")
        if t_min is not None and t_max is not None and t_min > t_max:
            problems.append("temperatuuri min/max on vahetuses")
        if wind_avg is None:
            problems.append("Häädemeeste tuul puudub")
        elif not 0 <= wind_avg <= 50:
            problems.append("tuul on ebarealistlik")
        if radiation is None:
            problems.append("Pärnu radiatsioon puudub")
        elif not 0 <= radiation <= 45:
            problems.append("radiatsioon on ebarealistlik")
        return problems

    def refresh_measured(self, start_day: date, end_day: date) -> Dict[str, int]:
        hdm = self._haademeeste_daily(start_day, end_day)
        radiation = self._parnu_radiation(start_day, end_day)
        saved = checked = 0
        d = start_day
        while d <= end_day:
            key = d.isoformat()
            local = hdm.get(key, {})
            t_min = local.get("t_min")
            t_max = local.get("t_max")
            wind_avg = local.get("wind_avg")
            rad = radiation.get(key)
            problems = self._validate_measured(t_min, t_max, wind_avg, rad)
            db.upsert_weather({
                "weather_date": key,
                "data_kind": "measured",
                "temp_min_c": t_min,
                "temp_max_c": t_max,
                "wind_avg_ms": wind_avg,
                "radiation_mj_m2": rad,
                "source_station": "Häädemeeste",
                "radiation_station": "Pärnu",
                "checked": not problems,
                "check_message": "; ".join(problems) if problems else "Kontrollitud",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            })
            saved += 1
            checked += int(not problems)
            d += timedelta(days=1)
        return {"saved": saved, "checked": checked}

    def refresh_forecast(self, start_day: date, days: int = 9) -> Dict[str, int]:
        end_day = start_day + timedelta(days=days - 1)
        params = {
            "latitude": FARM_LAT,
            "longitude": FARM_LON,
            "timezone": "Europe/Tallinn",
            "start_date": start_day.isoformat(),
            "end_date": end_day.isoformat(),
            "daily": "temperature_2m_min,temperature_2m_max,shortwave_radiation_sum",
            "hourly": "wind_speed_10m",
            "wind_speed_unit": "ms",
        }
        payload = self._get_json(OPEN_METEO, params)
        daily = payload.get("daily") or {}
        hourly = payload.get("hourly") or {}
        hourly_wind: Dict[str, List[float]] = defaultdict(list)
        for ts, value in zip(hourly.get("time", []), hourly.get("wind_speed_10m", [])):
            if value is not None:
                hourly_wind[str(ts)[:10]].append(float(value))
        saved = 0
        dates = daily.get("time", [])
        for i, key in enumerate(dates):
            t_min = daily.get("temperature_2m_min", [None] * len(dates))[i]
            t_max = daily.get("temperature_2m_max", [None] * len(dates))[i]
            rad = daily.get("shortwave_radiation_sum", [None] * len(dates))[i]
            wind = mean(hourly_wind[key]) if hourly_wind.get(key) else None
            problems = []
            if t_min is None or t_max is None: problems.append("temperatuuri prognoos puudub")
            if wind is None: problems.append("tuule prognoos puudub")
            if rad is None: problems.append("radiatsiooni prognoos puudub")
            db.upsert_weather({
                "weather_date": key,
                "data_kind": "forecast",
                "temp_min_c": t_min,
                "temp_max_c": t_max,
                "wind_avg_ms": wind,
                "radiation_mj_m2": rad,
                "source_station": "Open-Meteo",
                "radiation_station": "Open-Meteo",
                "checked": not problems,
                "check_message": "; ".join(problems) if problems else "Prognoos olemas",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            })
            saved += 1
        return {"saved": saved}

    def test_sources(self, today: date) -> Dict[str, Any]:
        """Kontrollib nelja vajalikku allikat ilma Supabase'i kirjutamata."""
        measured_day = today - timedelta(days=1)
        temp_rows = self._official_rows(OFFICIAL_HOURLY, "Häädemeeste", "TA", measured_day, measured_day)
        wind_rows = self._official_rows(OFFICIAL_HOURLY, "Häädemeeste", "WS10M", measured_day, measured_day)
        radiation_rows = []
        for year, month in self._month_ranges(measured_day, measured_day):
            radiation_rows.extend(self._get_json(OFFICIAL_DAILY, {
                "jaam_nimi": "like.Pärnu",
                "element_kood": "eq.DRQS",
                "aasta": f"eq.{year}",
                "kuu": f"eq.{month}",
                "paev": f"eq.{measured_day.day}",
                "select": "aasta,kuu,paev,vaartus,element_kood",
                "order": "paev.asc",
            }, HEADERS))

        forecast = self._get_json(OPEN_METEO, {
            "latitude": FARM_LAT,
            "longitude": FARM_LON,
            "timezone": "Europe/Tallinn",
            "start_date": today.isoformat(),
            "end_date": today.isoformat(),
            "daily": "temperature_2m_min,temperature_2m_max,shortwave_radiation_sum",
            "hourly": "wind_speed_10m",
            "wind_speed_unit": "ms",
        })
        forecast_dates = list((forecast.get("daily") or {}).get("time") or [])

        return {
            "measured_day": measured_day.isoformat(),
            "haademeeste_temperature_rows": len(temp_rows),
            "haademeeste_wind_rows": len(wind_rows),
            "parnu_radiation_rows": len(radiation_rows),
            "forecast_days": len(forecast_dates),
            "ok": bool(temp_rows and wind_rows and radiation_rows and forecast_dates),
        }

    def refresh_all(self, today: date) -> Dict[str, Any]:
        measured_start = date(today.year, 7, 1)
        measured_end = today - timedelta(days=1)
        measured = self.refresh_measured(measured_start, measured_end) if measured_end >= measured_start else {"saved": 0, "checked": 0}
        forecast = self.refresh_forecast(today, 9)
        db.set_app_setting("weather_last_refresh_at", datetime.now(ESTONIA).isoformat(timespec="seconds"))
        db.set_app_setting("weather_last_error", "")
        return {"measured": measured, "forecast": forecast}

    def safe_refresh_all(self, today: date) -> Dict[str, Any]:
        try:
            return self.refresh_all(today)
        except Exception as exc:
            db.set_app_setting("weather_last_error", str(exc))
            return {"error": str(exc)}
