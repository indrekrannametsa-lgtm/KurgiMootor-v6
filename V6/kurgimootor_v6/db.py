from __future__ import annotations

import json

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


def delete_harvest(harvest_date: date, field_id: int) -> None:
    """Kustutab ühe konkreetse kuupäeva + põllu korjerea."""
    _execute(
        _client().table("harvests")
        .delete()
        .eq("harvest_date", harvest_date.isoformat())
        .eq("field_no", int(field_id))
    )


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


WEATHER_VALUE_FIELDS = (
    "temp_min_c",
    "temp_max_c",
    "wind_avg_ms",
    "radiation_mj_m2",
    "humidity_avg_pct",
    "precipitation_mm",
    "et0_mm",
)


def _merge_measured_weather(existing: Dict[str, Any] | None, incoming: Dict[str, Any]) -> Dict[str, Any]:
    """Kaitseb olemasolevat head mõõdetud ilma regressiooni eest.

    Reeglid:
    - forecast-ridu ei muudeta;
    - mõõdetud rea None ei tohi kustutada varem olemas olnud numbrilist väärtust;
    - täielikult kontrollitud ametlikku rida ei langetata ajutiseks/puudulikuks
      ainult seetõttu, et järgmine API-kutse tagastab mõne elemendi puudu;
    - uus täielikult kontrollitud ametlik rida võib vana ametlikku rida uuendada.
    """
    merged = dict(incoming)
    if str(incoming.get("data_kind") or "") != "measured":
        return merged
    if not existing or str(existing.get("data_kind") or "") != "measured":
        return merged

    existing_checked = bool(existing.get("checked"))
    incoming_checked = bool(incoming.get("checked"))

    # Kui meil on juba täielikult ametlik rida ja uus fetch on ajutiselt puudulik,
    # hoia vana ametlik rida puutumatuna. Ainult updated_at võib uueneda.
    if existing_checked and not incoming_checked:
        protected = dict(existing)
        if incoming.get("updated_at") is not None:
            protected["updated_at"] = incoming.get("updated_at")
        return protected

    # Muul juhul täida ainult uue rea päriselt puuduvad väärtused olemasolevast reast.
    for field in WEATHER_VALUE_FIELDS:
        if merged.get(field) is None and existing.get(field) is not None:
            merged[field] = existing.get(field)

    # Ära kaota allikakirjeldust, kui uus osaline payload seda ei anna.
    for field in ("source_station", "radiation_station"):
        if not merged.get(field) and existing.get(field):
            merged[field] = existing.get(field)

    return merged


def upsert_weather(payload: Dict[str, Any]) -> None:
    incoming = dict(payload)
    key = str(incoming.get("weather_date") or "").strip()

    if incoming.get("data_kind") == "measured" and key:
        response = _execute(
            _client().table("weather_daily")
            .select("*")
            .eq("weather_date", key)
            .limit(1)
        )
        rows = list(response.data or [])
        existing = rows[0] if rows else None
        incoming = _merge_measured_weather(existing, incoming)

    _execute(_client().table("weather_daily").upsert(incoming, on_conflict="weather_date"))


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


def save_weather_forecast_snapshot(day: date, payload: Dict[str, Any]) -> None:
    """Säilitab päeva forecast'i eraldi, et minevikku jõudes ei kaoks see measured upsert'iga."""
    key = f"weather_forecast_snapshot_{day.isoformat()}"
    compact = {
        "weather_date": day.isoformat(),
        "temp_min_c": payload.get("temp_min_c"),
        "temp_max_c": payload.get("temp_max_c"),
        "wind_avg_ms": payload.get("wind_avg_ms"),
        "radiation_mj_m2": payload.get("radiation_mj_m2"),
        "humidity_avg_pct": payload.get("humidity_avg_pct"),
        "precipitation_mm": payload.get("precipitation_mm"),
        "et0_mm": payload.get("et0_mm"),
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    set_app_setting(key, json.dumps(compact, ensure_ascii=False))


def get_weather_forecast_snapshot(day: date) -> Dict[str, Any] | None:
    key = f"weather_forecast_snapshot_{day.isoformat()}"
    raw = get_app_setting(key, "")
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except Exception:
        return None
    return value if isinstance(value, dict) else None


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
            "cb_forecast": float(row["cb_forecast"]) if row.get("cb_forecast") is not None else None,
            "xl_forecast": float(row["xl_forecast"]),
            "total_forecast": float(row["total_forecast"]),
            "interval_days": int(row["interval_days"]) if row.get("interval_days") is not None else None,
            "basis": str(row.get("basis") or ""),
            "estimated_weather_days": str(row.get("estimated_weather_days") or ""),
            "model_version": str(row.get("model_version") or "v6.3-abc-xl"),
            "generated_at": generated_at,
        }
        payloads.append(payload)
    try:
        _execute(
            _client().table("yield_forecasts").upsert(
                payloads,
                on_conflict="forecast_date,target_date,field_no,model_version",
            )
        )
    except DatabaseError as exc:
        # Tagasiühilduvus: kui cb_forecast veergu pole veel migreeritud,
        # salvesta ülejäänud prognoosid ikkagi. UI annab eraldi märku, et
        # C/B ajalugu hakkab salvestuma pärast ALTER TABLE migratsiooni.
        if "cb_forecast" not in str(exc).lower():
            raise
        fallback = [{k: v for k, v in row.items() if k != "cb_forecast"} for row in payloads]
        _execute(
            _client().table("yield_forecasts").upsert(
                fallback,
                on_conflict="forecast_date,target_date,field_no,model_version",
            )
        )
    return len(payloads)


def get_yield_forecasts(limit: int = 1000) -> List[Dict[str, Any]]:
    base = _client().table("yield_forecasts")
    try:
        response = _execute(
            base.select("forecast_date,target_date,field_no,lead_days,abc_forecast,cb_forecast,xl_forecast,total_forecast,interval_days,basis,estimated_weather_days,model_version,generated_at")
            .order("target_date", desc=True)
            .order("field_no")
            .order("forecast_date", desc=True)
            .limit(int(limit))
        )
    except DatabaseError as exc:
        if "cb_forecast" not in str(exc).lower():
            raise
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
