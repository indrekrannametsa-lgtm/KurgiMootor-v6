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
HYDRO_OBSERVATIONS = "https://keskkonnaandmed.envir.ee/f_hydroseire"
HAADEMEESTE_STATION_CODE = "86031"
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

    @staticmethod
    def _normalise_name(value: Any) -> str:
        text = str(value or "").strip().lower()
        return (text.replace("õ", "o").replace("ä", "a").replace("ö", "o")
                    .replace("ü", "u").replace("š", "s").replace("ž", "z"))

    @classmethod
    def _series_score(cls, name: str, kind: str) -> int:
        n = cls._normalise_name(name)
        if kind == "temperature":
            if not any(token in n for token in ("ohutemp", "air temp", "temperature air", " ta")):
                return -1
            if any(token in n for token in ("vesi", "water", "pinnas", "soil")):
                return -1
            score = 10
            if "ohutemperatuur" in n:
                score += 10
            if "10 min" in n or "10 minuti" in n:
                score += 2
            return score

        if not any(token in n for token in ("tuule kiirus", "wind speed", "windspeed")):
            return -1
        if any(token in n for token in ("suund", "direction", "puhang", "gust", "maks", "max")):
            return -1
        score = 10
        if "10 minuti keskmine" in n or "10 min avg" in n or "10 minute average" in n:
            score += 10
        elif "2 minuti keskmine" in n or "2 min avg" in n or "2 minute average" in n:
            score += 6
        elif "keskmine" in n or "avg" in n or "average" in n:
            score += 4
        return score

    def _hydro_rows(self, start_day: date, end_day: date) -> List[Dict[str, Any]]:
        # Kohalik päev teisendatakse UTC piirideks, sest teenuse ajatempel on UTC-s.
        start_local = datetime.combine(start_day, datetime.min.time(), tzinfo=ESTONIA)
        end_local = datetime.combine(end_day + timedelta(days=1), datetime.min.time(), tzinfo=ESTONIA)
        start_utc = start_local.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        end_utc = end_local.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        params = [
            ("jaam_kood", f"eq.{HAADEMEESTE_STATION_CODE}"),
            ("timeline_ts_utc", f"gte.{start_utc}"),
            ("timeline_ts_utc", f"lt.{end_utc}"),
            ("select", "timeline_ts_utc,jaam_kood,jaam_nimi,aegrida_nimi,vaartus"),
            ("order", "timeline_ts_utc.asc"),
            ("limit", "20000"),
        ]
        payload = self._get_json(HYDRO_OBSERVATIONS, params, HEADERS)
        if not isinstance(payload, list):
            raise WeatherError("Hüdroseire API ei tagastanud loendit.")
        return payload

    @classmethod
    def _choose_series(cls, rows: List[Dict[str, Any]], kind: str) -> str | None:
        names = sorted({str(row.get("aegrida_nimi") or "").strip() for row in rows if row.get("aegrida_nimi")})
        ranked = sorted(((cls._series_score(name, kind), name) for name in names), reverse=True)
        return ranked[0][1] if ranked and ranked[0][0] >= 0 else None

    @staticmethod
    def _hydro_local_day(row: Dict[str, Any]) -> date | None:
        try:
            ts = str(row.get("timeline_ts_utc") or "").replace("Z", "+00:00")
            return datetime.fromisoformat(ts).astimezone(ESTONIA).date()
        except Exception:
            return None

    def _haademeeste_daily(self, start_day: date, end_day: date) -> Dict[str, Dict[str, float]]:
        rows = self._hydro_rows(start_day, end_day)
        temp_series = self._choose_series(rows, "temperature")
        wind_series = self._choose_series(rows, "wind")
        if not temp_series or not wind_series:
            available = sorted({str(r.get("aegrida_nimi") or "") for r in rows if r.get("aegrida_nimi")})
            raise WeatherError(
                "Häädemeeste jaama temperatuuri või tuule aegrida ei leitud. "
                f"Saadaval: {', '.join(available[:30])}"
            )

        temps: Dict[date, List[float]] = defaultdict(list)
        winds: Dict[date, List[float]] = defaultdict(list)
        for row in rows:
            d = self._hydro_local_day(row)
            value = self._as_float(row.get("vaartus"))
            if not d or value is None or not start_day <= d <= end_day:
                continue
            series = str(row.get("aegrida_nimi") or "").strip()
            if series == temp_series:
                temps[d].append(value)
            elif series == wind_series:
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
        hydro_rows = self._hydro_rows(measured_day, measured_day)
        temp_series = self._choose_series(hydro_rows, "temperature")
        wind_series = self._choose_series(hydro_rows, "wind")
        temp_rows = [r for r in hydro_rows if str(r.get("aegrida_nimi") or "").strip() == temp_series]
        wind_rows = [r for r in hydro_rows if str(r.get("aegrida_nimi") or "").strip() == wind_series]
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

        forecast_dates = []
        forecast_error = None
        try:
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
        except Exception as exc:
            forecast_error = str(exc)

        return {
            "measured_day": measured_day.isoformat(),
            "haademeeste_station_code": HAADEMEESTE_STATION_CODE,
            "haademeeste_temperature_series": temp_series,
            "haademeeste_temperature_rows": len(temp_rows),
            "haademeeste_wind_series": wind_series,
            "haademeeste_wind_rows": len(wind_rows),
            "haademeeste_available_series": sorted({str(r.get("aegrida_nimi") or "") for r in hydro_rows if r.get("aegrida_nimi")}),
            "parnu_radiation_rows": len(radiation_rows),
            "forecast_days": len(forecast_dates),
            "forecast_error": forecast_error,
            "measured_sources_ok": bool(temp_rows and wind_rows and radiation_rows),
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
