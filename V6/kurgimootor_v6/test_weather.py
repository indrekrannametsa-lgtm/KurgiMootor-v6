import tempfile
import unittest
from datetime import date
from pathlib import Path

from core import KurgiDB, WeatherService


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeHTTP:
    def get(self, url, params=None, headers=None, timeout=None):
        if "keskkonnaandmed" in url:
            station = params.get("jaam_nimi", "")
            month = int(params["kuu"].split(".")[-1])
            if "Häädemeeste" in station:
                return FakeResponse([
                    {"jaam_nimi":"Häädemeeste","aasta":2026,"kuu":month,"paev":1,"vaartus":18.5,"element_kood":"DTA08","element_nimi":"Air temperature (daily avg)","element_yhik":"°C"},
                    {"jaam_nimi":"Häädemeeste","aasta":2026,"kuu":month,"paev":1,"vaartus":3.2,"element_kood":"DWS","element_nimi":"Wind speed (daily avg)","element_yhik":"m/s"},
                    {"jaam_nimi":"Häädemeeste","aasta":2026,"kuu":month,"paev":2,"vaartus":19.0,"element_kood":"DTA08","element_nimi":"Air temperature (daily avg)","element_yhik":"°C"},
                ])
            return FakeResponse([
                {"jaam_nimi":"Pärnu","aasta":2026,"kuu":month,"paev":1,"vaartus":22.1,"element_kood":"DRQS","element_nimi":"Global radiation daily sum","element_yhik":"MJ/m²"},
                {"jaam_nimi":"Pärnu","aasta":2026,"kuu":month,"paev":2,"vaartus":20.5,"element_kood":"DRQS","element_nimi":"Global radiation daily sum","element_yhik":"MJ/m²"},
            ])
        times=[]; temps=[]; winds=[]
        for day in range(3,12):
            for hour in range(24):
                times.append(f"2026-08-{day:02d}T{hour:02d}:00")
                temps.append(20.0 + hour/100)
                winds.append(3.0)
        return FakeResponse({
            "hourly":{"time":times,"temperature_2m":temps,"wind_speed_10m":winds},
            "daily":{"time":[f"2026-08-{d:02d}" for d in range(3,12)],"shortwave_radiation_sum":[20.0]*9},
        })


class WeatherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = KurgiDB(Path(self.tmp.name)/"test.db")
        self.service = WeatherService(self.db, http=FakeHTTP())

    def tearDown(self):
        self.tmp.cleanup()

    def test_measured_only_complete_day_is_green(self):
        result = self.service.refresh_measured(date(2026,8,1), date(2026,8,2))
        self.assertEqual(result["saved"], 2)
        rows = [dict(r) for r in self.db.weather_rows("2026-08-01","2026-08-02")]
        self.assertEqual(rows[0]["checked"], 1)
        self.assertEqual(rows[0]["wind_avg"], 3.2)
        self.assertEqual(rows[1]["checked"], 0)
        self.assertIn("tuul puudub", rows[1]["check_message"])

    def test_forecast_has_exactly_nine_days(self):
        self.service.refresh_forecast(date(2026,8,3), 9)
        rows = [dict(r) for r in self.db.weather_rows("2026-08-03","2026-08-11")]
        self.assertEqual(len(rows), 9)
        self.assertTrue(all(r["data_kind"] == "forecast" for r in rows))
        self.assertTrue(all(r["checked"] == 1 for r in rows))


if __name__ == "__main__":
    unittest.main()
