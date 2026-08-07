from __future__ import annotations

from datetime import date
from typing import Any, Dict, List

import streamlit as st
from supabase import Client, create_client


class DatabaseError(RuntimeError):
    pass


def _client() -> Client:
    try:
        cfg = st.secrets["supabase"]
        url = str(cfg["url"]).strip()
        key = str(cfg.get("secret_key") or cfg.get("service_role_key")).strip()
    except Exception as exc:
        raise DatabaseError(
            "Streamliti Secrets peab sisaldama [supabase] url ja secret_key."
        ) from exc
    if not url or not key:
        raise DatabaseError("Supabase URL või secret key puudub.")
    try:
        return create_client(url, key)
    except Exception as exc:
        raise DatabaseError(f"Supabase kliendi loomine ebaõnnestus: {exc}") from exc


def _execute(query: Any) -> Any:
    try:
        return query.execute()
    except Exception as exc:
        raise DatabaseError(str(exc)) from exc


def get_all_fields() -> List[Dict[str, Any]]:
    response = _execute(_client().table("fields").select("id,name").order("id"))
    return list(response.data or [])


def get_plan_for_day(day: date) -> List[Dict[str, Any]]:
    response = _execute(_client().rpc("get_daily_plan", {"p_plan_date": day.isoformat()}))
    return list(response.data or [])


def ensure_default_plan(day: date) -> None:
    _execute(_client().rpc("ensure_default_daily_plan", {"p_plan_date": day.isoformat()}))


def add_plan_field(day: date, field_id: int) -> None:
    _execute(_client().rpc("add_daily_plan_field", {"p_plan_date": day.isoformat(), "p_field_id": int(field_id)}))


def remove_plan_field(day: date, field_id: int) -> None:
    _execute(_client().rpc("remove_daily_plan_field", {"p_plan_date": day.isoformat(), "p_field_id": int(field_id)}))


def save_harvest(harvest_date: date, field_id: int, interval_days: int, a: float, b: float, c: float, xl: float) -> None:
    payload = {
        "harvest_date": harvest_date.isoformat(),
        "field_id": int(field_id),
        "interval_days": int(interval_days),
        "a": float(a), "b": float(b), "c": float(c), "xl": float(xl),
    }
    _execute(_client().table("harvests").upsert(payload, on_conflict="harvest_date,field_id"))


def get_harvest_for_day(day: date) -> List[Dict[str, Any]]:
    response = _execute(_client().table("harvests").select("field_id,a,b,c,xl,interval_days").eq("harvest_date", day.isoformat()))
    return list(response.data or [])


def get_harvest_history(limit: int = 300) -> List[Dict[str, Any]]:
    response = _execute(_client().rpc("get_harvest_history", {"p_limit": int(limit)}))
    return list(response.data or [])


def upsert_weather(payload: Dict[str, Any]) -> None:
    _execute(_client().table("weather_daily").upsert(payload, on_conflict="weather_date"))


def get_weather_rows(start_day: date, end_day: date) -> List[Dict[str, Any]]:
    response = _execute(
        _client().table("weather_daily").select("*")
        .gte("weather_date", start_day.isoformat())
        .lte("weather_date", end_day.isoformat())
        .order("weather_date")
    )
    return list(response.data or [])


def get_weather_counts() -> Dict[str, int]:
    rows = get_weather_rows(date(2020, 1, 1), date(2100, 1, 1))
    return {
        "measured": sum(1 for r in rows if r.get("data_kind") == "measured"),
        "checked": sum(1 for r in rows if r.get("data_kind") == "measured" and bool(r.get("checked"))),
        "forecast": sum(1 for r in rows if r.get("data_kind") == "forecast"),
    }



def get_latest_checked_measured_date() -> date | None:
    response = _execute(
        _client().table("weather_daily")
        .select("weather_date")
        .eq("data_kind", "measured")
        .eq("checked", True)
        .order("weather_date", desc=True)
        .limit(1)
    )
    rows = list(response.data or [])
    if not rows:
        return None
    try:
        return date.fromisoformat(str(rows[0]["weather_date"]))
    except (KeyError, TypeError, ValueError):
        return None


def get_incomplete_measured_dates(start_day: date, end_day: date) -> List[date]:
    rows = get_weather_rows(start_day, end_day)
    by_day = {str(r.get("weather_date")): r for r in rows if r.get("data_kind") == "measured"}
    missing: List[date] = []
    current = start_day
    while current <= end_day:
        row = by_day.get(current.isoformat())
        required_values = (
            "temp_min_c",
            "temp_max_c",
            "wind_avg_ms",
            "radiation_mj_m2",
            "humidity_avg_pct",
            "precipitation_mm",
            "et0_mm",
        )
        if (
            not row
            or not bool(row.get("checked"))
            or any(row.get(field) is None for field in required_values)
        ):
            missing.append(current)
        current += date.resolution
    return missing

def set_app_setting(key: str, value: str) -> None:
    _execute(_client().table("app_settings").upsert({"key": key, "value": str(value)}, on_conflict="key"))


def get_app_setting(key: str, default: str = "") -> str:
    response = _execute(_client().table("app_settings").select("value").eq("key", key).limit(1))
    rows = list(response.data or [])
    return str(rows[0]["value"]) if rows else default
