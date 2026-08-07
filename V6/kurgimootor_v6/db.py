from __future__ import annotations

from datetime import date, datetime, timezone
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


def _previous_harvest_date(harvest_date: date, field_no: int) -> date | None:
    response = _execute(
        _client().table("harvests")
        .select("harvest_date")
        .eq("field_no", int(field_no))
        .lt("harvest_date", harvest_date.isoformat())
        .order("harvest_date", desc=True)
        .limit(1)
    )
    rows = list(response.data or [])
    if not rows:
        return None
    try:
        return date.fromisoformat(str(rows[0]["harvest_date"]))
    except (KeyError, TypeError, ValueError):
        return None


def save_harvest(
    harvest_date: date,
    field_id: int,
    interval_days: int,
    a: float,
    b: float,
    c: float,
    xl: float,
    harvest_order: int | None = None,
) -> None:
    field_no = int(field_id)
    prev = _previous_harvest_date(harvest_date, field_no)
    calculated_interval = (harvest_date - prev).days if prev else int(interval_days)
    payload = {
        "harvest_date": harvest_date.isoformat(),
        "field_no": field_no,
        "harvest_order": int(harvest_order) if harvest_order is not None else None,
        "a": float(a),
        "b": float(b),
        "c": float(c),
        "xl": float(xl),
        "total": round(float(a) + float(b) + float(c) + float(xl), 3),
        "previous_harvest_date": prev.isoformat() if prev else None,
        "interval_days": calculated_interval,
        "data_quality": "Kinnitatud",
        "note": "Sisestatud KurgiMootor V6.2 äpis",
    }
    _execute(_client().table("harvests").upsert(payload, on_conflict="harvest_date,field_no"))


def get_harvest_for_day(day: date) -> List[Dict[str, Any]]:
    response = _execute(
        _client().table("harvests")
        .select("field_no,harvest_order,a,b,c,xl,total,interval_days")
        .eq("harvest_date", day.isoformat())
        .order("harvest_order")
    )
    rows = list(response.data or [])
    # Hoidame app.py vana field_id liidese ühilduvana.
    for row in rows:
        row["field_id"] = row.get("field_no")
    return rows


def get_harvest_history(limit: int = 500) -> List[Dict[str, Any]]:
    response = _execute(
        _client().table("harvests")
        .select("harvest_date,field_no,harvest_order,interval_days,a,b,c,xl,total,data_quality,note")
        .order("harvest_date", desc=True)
        .order("harvest_order")
        .limit(int(limit))
    )
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


def save_yield_forecasts(rows: List[Dict[str, Any]]) -> int:
    """Save/update one forecast snapshot per forecast day + target day + field + model.

    Re-running the app on the same day updates the same snapshot instead of creating
    duplicates. A new calendar day creates a new lead-time snapshot automatically.
    """
    if not rows:
        return 0
    generated_at = datetime.now(timezone.utc).isoformat()
    payloads: List[Dict[str, Any]] = []
    for row in rows:
        payload = {
            "forecast_date": str(row["forecast_date"]),
            "target_date": str(row["target_date"]),
            "field_no": int(row["field_no"]),
            "lead_days": int(row["lead_days"]),
            "abc_forecast": float(row["abc_forecast"]),
            "xl_forecast": float(row["xl_forecast"]),
            "total_forecast": float(row["total_forecast"]),
            "interval_days": int(row["interval_days"]) if row.get("interval_days") is not None else None,
            "basis": str(row.get("basis") or ""),
            "estimated_weather_days": str(row.get("estimated_weather_days") or ""),
            "model_version": str(row.get("model_version") or "v6.3-abc-xl"),
            "generated_at": generated_at,
        }
        payloads.append(payload)
    _execute(
        _client().table("yield_forecasts").upsert(
            payloads,
            on_conflict="forecast_date,target_date,field_no,model_version",
        )
    )
    return len(payloads)


def get_yield_forecasts(limit: int = 1000) -> List[Dict[str, Any]]:
    response = _execute(
        _client().table("yield_forecasts")
        .select("forecast_date,target_date,field_no,lead_days,abc_forecast,xl_forecast,total_forecast,interval_days,basis,estimated_weather_days,model_version,generated_at")
        .order("target_date", desc=True)
        .order("field_no")
        .order("forecast_date", desc=True)
        .limit(int(limit))
    )
    return list(response.data or [])


def yield_forecasts_available() -> bool:
    try:
        _execute(_client().table("yield_forecasts").select("id").limit(1))
        return True
    except DatabaseError:
        return False
