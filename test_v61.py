from pathlib import Path
import py_compile

ROOT = Path(__file__).parent
py_compile.compile(str(ROOT / "app.py"), doraise=True)
py_compile.compile(str(ROOT / "db.py"), doraise=True)

schema = (ROOT / "supabase_schema.sql").read_text(encoding="utf-8")
required = [
    "create table if not exists public.fields",
    "create table if not exists public.daily_plan",
    "create table if not exists public.plan_days",
    "create table if not exists public.harvests",
    "create table if not exists public.weather_daily",
    "ensure_default_daily_plan",
    "get_field_interval",
    "get_daily_plan",
    "add_daily_plan_field",
    "remove_daily_plan_field",
    "get_harvest_history",
    "get_weather_status",
    "generate_series(1, 14)",
]
for item in required:
    assert item in schema.lower(), item

app = (ROOT / "app.py").read_text(encoding="utf-8")
assert "sqlite" not in app.lower()
assert "ilmaandmed" in app.lower()
assert "forecast" not in app.lower()

db = (ROOT / "db.py").read_text(encoding="utf-8")
assert "sqlite" not in db.lower()
assert "create_client" in db

print("V6.1 kontroll läbitud.")
