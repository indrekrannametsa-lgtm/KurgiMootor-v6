import json
import sqlite3
import unicodedata
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

BASE_INTERVALS = [5, 5, 4]


class KurgiDB:
    def __init__(self, db_path):
        self.db_path = Path(db_path)
        self.init_db()

    def conn(self):
        c = sqlite3.connect(self.db_path)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA foreign_keys = ON")
        return c

    def init_db(self):
        with self.conn() as c:
            c.executescript(
                """
                CREATE TABLE IF NOT EXISTS fields (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE
                );
                CREATE TABLE IF NOT EXISTS harvests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    harvest_date TEXT NOT NULL,
                    field_id INTEGER NOT NULL,
                    interval_days INTEGER NOT NULL,
                    a REAL NOT NULL DEFAULT 0,
                    b REAL NOT NULL DEFAULT 0,
                    c REAL NOT NULL DEFAULT 0,
                    xl REAL NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    UNIQUE(harvest_date, field_id),
                    FOREIGN KEY(field_id) REFERENCES fields(id)
                );
                CREATE TABLE IF NOT EXISTS plan_days (
                    plan_date TEXT PRIMARY KEY,
                    initialized_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS daily_plan (
                    plan_date TEXT NOT NULL,
                    field_id INTEGER NOT NULL,
                    position INTEGER NOT NULL,
                    PRIMARY KEY(plan_date, field_id),
                    FOREIGN KEY(plan_date) REFERENCES plan_days(plan_date) ON DELETE CASCADE,
                    FOREIGN KEY(field_id) REFERENCES fields(id)
                );
                CREATE TABLE IF NOT EXISTS weather_days (
                    weather_date TEXT PRIMARY KEY,
                    data_kind TEXT NOT NULL DEFAULT 'measured',
                    t_avg REAL,
                    wind_avg REAL,
                    radiation REAL,
                    temp_source TEXT,
                    wind_source TEXT,
                    radiation_source TEXT,
                    checked INTEGER NOT NULL DEFAULT 0,
                    check_message TEXT,
                    imported_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS engine_notes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    title TEXT NOT NULL,
                    body TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'uus'
                );
                """
            )
            self._migrate_weather_table(c)
            for i in range(1, 15):
                c.execute(
                    "INSERT OR IGNORE INTO fields(id, name) VALUES (?, ?)",
                    (i, f"Põld {i}"),
                )
            c.commit()

    @staticmethod
    def _migrate_weather_table(c):
        cols = {r[1] for r in c.execute("PRAGMA table_info(weather_days)").fetchall()}
        additions = {
            "data_kind": "TEXT NOT NULL DEFAULT 'measured'",
            "wind_avg": "REAL",
            "temp_source": "TEXT",
            "wind_source": "TEXT",
            "radiation_source": "TEXT",
            "check_message": "TEXT",
        }
        for name, definition in additions.items():
            if name not in cols:
                c.execute(f"ALTER TABLE weather_days ADD COLUMN {name} {definition}")
        if "source" in cols:
            c.execute(
                """
                UPDATE weather_days
                SET temp_source=COALESCE(temp_source, source),
                    radiation_source=COALESCE(radiation_source, source)
                """
            )

    def get_setting(self, key, default=None):
        with self.conn() as c:
            row = c.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default

    def set_setting(self, key, value):
        with self.conn() as c:
            c.execute(
                "INSERT INTO settings(key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, str(value)),
            )
            c.commit()

    def all_fields(self):
        with self.conn() as c:
            return c.execute("SELECT id, name FROM fields ORDER BY id").fetchall()

    def last_harvest_date(self, field_id, before_day=None):
        sql = "SELECT MAX(harvest_date) AS d FROM harvests WHERE field_id=?"
        args = [field_id]
        if before_day:
            sql += " AND harvest_date < ?"
            args.append(before_day)
        with self.conn() as c:
            row = c.execute(sql, args).fetchone()
        return row["d"] if row and row["d"] else None

    def interval_days(self, field_id, target_day, fallback=None):
        last = self.last_harvest_date(field_id, before_day=target_day)
        if last:
            return (date.fromisoformat(target_day) - date.fromisoformat(last)).days
        return fallback

    def plan_initialized(self, day):
        with self.conn() as c:
            return (
                c.execute("SELECT 1 FROM plan_days WHERE plan_date=?", (day,)).fetchone()
                is not None
            )

    def plan_for(self, day):
        with self.conn() as c:
            return c.execute(
                """
                SELECT p.field_id, f.name, p.position
                FROM daily_plan p JOIN fields f ON f.id=p.field_id
                WHERE p.plan_date=? ORDER BY p.position
                """,
                (day,),
            ).fetchall()

    def ensure_default_plan(self, day):
        if self.plan_initialized(day):
            return
        scored = []
        for f in self.all_fields():
            last = self.last_harvest_date(f["id"], before_day=day)
            scored.append((last or "0000-00-00", f["id"]))
        scored.sort(key=lambda x: (x[0], x[1]))
        self.replace_plan(day, [field_id for _, field_id in scored[:3]])

    def replace_plan(self, day, field_ids):
        if len(field_ids) != len(set(field_ids)):
            raise ValueError("Sama põld ei tohi olla plaanis kaks korda.")
        valid = {r["id"] for r in self.all_fields()}
        if any(fid not in valid for fid in field_ids):
            raise ValueError("Tundmatu põld.")
        with self.conn() as c:
            c.execute(
                "INSERT OR IGNORE INTO plan_days(plan_date, initialized_at) VALUES (?, ?)",
                (day, datetime.now().isoformat(timespec="seconds")),
            )
            c.execute("DELETE FROM daily_plan WHERE plan_date=?", (day,))
            for pos, field_id in enumerate(field_ids, start=1):
                c.execute(
                    "INSERT INTO daily_plan(plan_date, field_id, position) VALUES (?, ?, ?)",
                    (day, field_id, pos),
                )
            c.commit()

    def save_harvest(self, day, field_id, interval, a, b, c_value, xl):
        values = [float(a), float(b), float(c_value), float(xl)]
        if any(v < 0 for v in values):
            raise ValueError("Korje kogus ei saa olla negatiivne.")
        if interval is None or int(interval) < 1:
            raise ValueError("Korjeintervall peab olema vähemalt 1 päev.")
        with self.conn() as c:
            c.execute(
                """
                INSERT INTO harvests(harvest_date, field_id, interval_days, a, b, c, xl, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(harvest_date, field_id) DO UPDATE SET
                    interval_days=excluded.interval_days,
                    a=excluded.a, b=excluded.b, c=excluded.c, xl=excluded.xl,
                    created_at=excluded.created_at
                """,
                (
                    day,
                    field_id,
                    int(interval),
                    *values,
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )
            c.commit()

    def harvest_for(self, day, field_id):
        with self.conn() as c:
            return c.execute(
                "SELECT * FROM harvests WHERE harvest_date=? AND field_id=?",
                (day, field_id),
            ).fetchone()

    def harvest_rows(self, limit=500):
        with self.conn() as c:
            return c.execute(
                """
                SELECT h.harvest_date, f.name AS field, h.interval_days,
                       h.a, h.b, h.c, h.xl, (h.a+h.b+h.c+h.xl) AS total
                FROM harvests h JOIN fields f ON f.id=h.field_id
                ORDER BY h.harvest_date DESC, h.field_id LIMIT ?
                """,
                (limit,),
            ).fetchall()

    def import_weather(self, payload):
        """Backward-compatible manual import for already checked daily data."""
        data = json.loads(payload)
        if isinstance(data, dict):
            data = data["data"] if isinstance(data.get("data"), list) else [data]
        if not isinstance(data, list) or not data:
            raise ValueError("JSON peab sisaldama vähemalt üht kirjet.")
        count = 0
        for i, row in enumerate(data, start=1):
            d = str(row.get("date") or row.get("weather_date") or "")[:10]
            if not d:
                raise ValueError(f"Kirjel {i} puudub kuupäev.")
            date.fromisoformat(d)
            t = row.get("t_avg")
            wind = row.get("wind_avg")
            rad = row.get("radiation")
            if t is None or rad is None:
                raise ValueError(f"Kirjel {i} peavad olema t_avg ja radiation.")
            t = float(t)
            rad = float(rad)
            wind = float(wind) if wind is not None else None
            if not -50 <= t <= 50:
                raise ValueError(f"Kirje {i}: temperatuur on ebarealistlik.")
            if wind is not None and not 0 <= wind <= 50:
                raise ValueError(f"Kirje {i}: tuul on ebarealistlik.")
            if not 0 <= rad <= 45:
                raise ValueError(f"Kirje {i}: radiatsioon on ebarealistlik.")
            self.upsert_weather({
                "weather_date": d,
                "data_kind": "measured",
                "t_avg": t,
                "wind_avg": wind,
                "radiation": rad,
                "temp_source": "Käsitsi kontrollitud JSON",
                "wind_source": "Käsitsi kontrollitud JSON" if wind is not None else None,
                "radiation_source": "Käsitsi kontrollitud JSON",
                "checked": wind is not None,
                "check_message": "Kontrollitud" if wind is not None else "Tuul puudub",
            })
            count += 1
        return count

    def upsert_weather(self, row):
        with self.conn() as c:
            c.execute(
                """
                INSERT INTO weather_days(
                    weather_date, data_kind, t_avg, wind_avg, radiation,
                    temp_source, wind_source, radiation_source,
                    checked, check_message, imported_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(weather_date) DO UPDATE SET
                    data_kind=excluded.data_kind,
                    t_avg=excluded.t_avg,
                    wind_avg=excluded.wind_avg,
                    radiation=excluded.radiation,
                    temp_source=excluded.temp_source,
                    wind_source=excluded.wind_source,
                    radiation_source=excluded.radiation_source,
                    checked=excluded.checked,
                    check_message=excluded.check_message,
                    imported_at=excluded.imported_at
                """,
                (
                    row["weather_date"],
                    row["data_kind"],
                    row.get("t_avg"),
                    row.get("wind_avg"),
                    row.get("radiation"),
                    row.get("temp_source"),
                    row.get("wind_source"),
                    row.get("radiation_source"),
                    int(bool(row.get("checked"))),
                    row.get("check_message"),
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )
            c.commit()

    def weather_rows(self, start_day, end_day):
        with self.conn() as c:
            return c.execute(
                """
                SELECT * FROM weather_days
                WHERE weather_date BETWEEN ? AND ?
                ORDER BY weather_date
                """,
                (start_day, end_day),
            ).fetchall()

    def weather_status(self):
        with self.conn() as c:
            measured = c.execute(
                "SELECT COUNT(*) n FROM weather_days WHERE data_kind='measured'"
            ).fetchone()["n"]
            checked = c.execute(
                "SELECT COUNT(*) n FROM weather_days WHERE data_kind='measured' AND checked=1"
            ).fetchone()["n"]
            forecast = c.execute(
                "SELECT COUNT(*) n FROM weather_days WHERE data_kind='forecast'"
            ).fetchone()["n"]
        return measured, checked, forecast


class WeatherService:
    OFFICIAL_BASE = "https://keskkonnaandmed.envir.ee/f_kliima_paev"
    OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
    HEADERS = {"Accept-Profile": "apijahiala","Accept": "application/json"}

    # Farm-area coordinates; forecast is not presented as a station measurement.
    FORECAST_LAT = 58.13
    FORECAST_LON = 24.50

    def __init__(self, db, http=None):
        self.db = db
        if http is None:
            import requests

            http = requests.Session()
        self.http = http

    @staticmethod
    def _norm(text):
        text = unicodedata.normalize("NFKD", str(text or ""))
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
        return " ".join(text.lower().replace("_", " ").split())

    @staticmethod
    def _month_ranges(start_day, end_day):
        current = date(start_day.year, start_day.month, 1)
        while current <= end_day:
            next_month = (
                date(current.year + 1, 1, 1)
                if current.month == 12
                else date(current.year, current.month + 1, 1)
            )
            yield current.year, current.month
            current = next_month

    def _official_rows(self, station_fragment, start_day, end_day):
        result = []
        for year, month in self._month_ranges(start_day, end_day):
            params = {
                "jaam_nimi": f"like.{station_fragment}",
                "aasta": f"eq.{year}",
                "kuu": f"eq.{month}",
                "select": (
                    "jaam_kood,jaam_nimi,aasta,kuu,paev,vaartus,"
                    "element_kood"
                ),
                "order": "paev.asc,element_kood.asc",
            }
            response = self.http.get(
                self.OFFICIAL_BASE,
                params=params,
                headers=self.HEADERS,
                timeout=30,
            )
            if not response.ok: raise RuntimeError(f"Keskkonnaandmete API{response.status_code}:{response.text}")
            payload = response.json()
            if not isinstance(payload, list):
                raise ValueError("Ametlik ilma-API ei tagastanud kirjete loendit.")
            result.extend(payload)
        return result

    def _pick_daily_values(self, rows, kind, start_day, end_day):
        candidates = defaultdict(list)
        for row in rows:
            try:
                d = date(int(row["aasta"]), int(row["kuu"]), int(row["paev"]))
                value = float(row["vaartus"])
            except (KeyError, TypeError, ValueError):
                continue
            if not (start_day <= d <= end_day):
                continue
            code = self._norm(row.get("element_kood"))
            name = self._norm(row.get("element_nimi"))
            unit = self._norm(row.get("element_yhik"))
            score = 0
            if kind == "temperature":
                if code == "dta08":
                    score += 100
                if "temperature" in name and ("daily avg" in name or "average" in name):
                    score += 70
                if "ohutemperatuur" in name and ("keskm" in name or "oopaeva" in name):
                    score += 70
                if "c" in unit or "°c" in str(row.get("element_yhik", "")):
                    score += 5
            elif kind == "wind":
                if "wind speed" in name and ("daily avg" in name or "average" in name):
                    score += 70
                if "tuule" in name and "kiirus" in name and "keskm" in name:
                    score += 70
                if "m/s" in str(row.get("element_yhik", "")).lower():
                    score += 5
            elif kind == "radiation":
                if code == "drqs":
                    score += 100
                if "global radiation" in name and ("sum" in name or "daily" in name):
                    score += 70
                if "globaalradiatsioon" in name:
                    score += 70
                if "mj" in unit:
                    score += 10
            if score:
                candidates[d.isoformat()].append((score, value, row))

        picked = {}
        for d, values in candidates.items():
            values.sort(key=lambda x: x[0], reverse=True)
            picked[d] = values[0][1]
        return picked

    @staticmethod
    def _validate_measured(t_avg, wind_avg, radiation):
        problems = []
        if t_avg is None:
            problems.append("Häädemeeste temperatuur puudub")
        elif not -50 <= t_avg <= 50:
            problems.append("temperatuur on ebarealistlik")
        if wind_avg is None:
            problems.append("Häädemeeste tuul puudub")
        elif not 0 <= wind_avg <= 50:
            problems.append("tuul on ebarealistlik")
        if radiation is None:
            problems.append("Pärnu radiatsioon puudub")
        elif not 0 <= radiation <= 45:
            problems.append("radiatsioon on ebarealistlik")
        return problems

    def refresh_measured(self, start_day, end_day):
        if end_day < start_day:
            return {"saved": 0, "checked": 0, "errors": 0}
        haade = self._official_rows("Häädemeeste", start_day, end_day)
        parnu = self._official_rows("Pärnu", start_day, end_day)
        temps = self._pick_daily_values(haade, "temperature", start_day, end_day)
        winds = self._pick_daily_values(haade, "wind", start_day, end_day)
        radiation = self._pick_daily_values(parnu, "radiation", start_day, end_day)

        saved = checked = errors = 0
        d = start_day
        while d <= end_day:
            key = d.isoformat()
            problems = self._validate_measured(
                temps.get(key), winds.get(key), radiation.get(key)
            )
            self.db.upsert_weather(
                {
                    "weather_date": key,
                    "data_kind": "measured",
                    "t_avg": temps.get(key),
                    "wind_avg": winds.get(key),
                    "radiation": radiation.get(key),
                    "temp_source": "KAUR Häädemeeste",
                    "wind_source": "KAUR Häädemeeste",
                    "radiation_source": "KAUR Pärnu",
                    "checked": not problems,
                    "check_message": "; ".join(problems) if problems else "Kontrollitud",
                }
            )
            saved += 1
            checked += int(not problems)
            errors += int(bool(problems))
            d += timedelta(days=1)
        return {"saved": saved, "checked": checked, "errors": errors}

    def refresh_forecast(self, today, days=9):
        params = {
            "latitude": self.FORECAST_LAT,
            "longitude": self.FORECAST_LON,
            "hourly": "temperature_2m,wind_speed_10m",
            "daily": "shortwave_radiation_sum",
            "forecast_days": days,
            "timezone": "Europe/Tallinn",
            "wind_speed_unit": "ms",
        }
        response = self.http.get(self.OPEN_METEO_URL, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
        hourly = payload.get("hourly", {})
        daily = payload.get("daily", {})

        times = hourly.get("time", [])
        temperatures = hourly.get("temperature_2m", [])
        winds = hourly.get("wind_speed_10m", [])
        if not (len(times) == len(temperatures) == len(winds)):
            raise ValueError("Prognoosi tunniandmed on mittetäielikud.")

        buckets = defaultdict(lambda: {"t": [], "w": []})
        for timestamp, temp, wind in zip(times, temperatures, winds):
            d = str(timestamp)[:10]
            if temp is not None:
                buckets[d]["t"].append(float(temp))
            if wind is not None:
                buckets[d]["w"].append(float(wind))

        rad_by_date = dict(
            zip(daily.get("time", []), daily.get("shortwave_radiation_sum", []))
        )
        saved = 0
        for offset in range(days):
            d = today + timedelta(days=offset)
            key = d.isoformat()
            temps = buckets[key]["t"]
            ws = buckets[key]["w"]
            rad = rad_by_date.get(key)
            problems = []
            if len(temps) < 20:
                problems.append("temperatuuri tunniandmeid alla 20")
            if len(ws) < 20:
                problems.append("tuule tunniandmeid alla 20")
            if rad is None:
                problems.append("radiatsioon puudub")
            t_avg = sum(temps) / len(temps) if temps else None
            w_avg = sum(ws) / len(ws) if ws else None
            if t_avg is not None and not -50 <= t_avg <= 50:
                problems.append("temperatuur ebarealistlik")
            if w_avg is not None and not 0 <= w_avg <= 50:
                problems.append("tuul ebarealistlik")
            if rad is not None and not 0 <= float(rad) <= 45:
                problems.append("radiatsioon ebarealistlik")
            self.db.upsert_weather(
                {
                    "weather_date": key,
                    "data_kind": "forecast",
                    "t_avg": t_avg,
                    "wind_avg": w_avg,
                    "radiation": float(rad) if rad is not None else None,
                    "temp_source": "Open-Meteo prognoos",
                    "wind_source": "Open-Meteo prognoos",
                    "radiation_source": "Open-Meteo prognoos",
                    "checked": not problems,
                    "check_message": "; ".join(problems) if problems else "Prognoos olemas",
                }
            )
            saved += 1
        return {"saved": saved}

    def refresh_all(self, today=None, force=False):
        today = today or date.today()
        last_refresh = self.db.get_setting("weather_last_refresh")
        if not force and last_refresh == today.isoformat():
            return {"skipped": True, "message": "Täna juba uuendatud"}

        season_start = date(today.year, 7, 1)
        yesterday = today - timedelta(days=1)
        result = {
            "skipped": False,
            "measured": self.refresh_measured(season_start, yesterday),
            "forecast": self.refresh_forecast(today, 9),
        }
        self.db.set_setting("weather_last_refresh", today.isoformat())
        self.db.set_setting(
            "weather_last_refresh_at", datetime.now().isoformat(timespec="seconds")
        )
        self.db.set_setting("weather_last_error", "")
        return result

    def safe_refresh_all(self, today=None, force=False):
        try:
            return self.refresh_all(today=today, force=force)
        except Exception as exc:
            self.db.set_setting("weather_last_error", str(exc))
            return {"skipped": False, "error": str(exc)}
