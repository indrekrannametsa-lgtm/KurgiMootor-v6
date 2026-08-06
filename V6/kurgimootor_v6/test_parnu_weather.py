from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import types
sys.modules["db"] = types.SimpleNamespace()
import core


class FakeHTTP:
    def get(self, url, params=None, headers=None, timeout=None):
        params = params or {}
        elem = str(params.get("element_kood", ""))
        if "f_kliima_tund" in url and elem == "eq.TA":
            payload = [
                {"jaam_nimi": "Pärnu", "aasta": 2026, "kuu": 8, "paev": 5, "tund": h, "vaartus": v, "element_kood": "TA"}
                for h, v in [(0, 14.0), (6, 12.5), (12, 23.0), (18, 19.0)]
            ]
        elif "f_kliima_tund" in url and elem == "eq.WS10M":
            payload = [
                {"jaam_nimi": "Pärnu", "aasta": 2026, "kuu": 8, "paev": 5, "tund": h, "vaartus": v, "element_kood": "WS10M"}
                for h, v in [(0, 2.0), (6, 3.0), (12, 5.0), (18, 4.0)]
            ]
        elif "f_kliima_paev" in url and elem == "eq.DRQS":
            payload = [{"jaam_nimi": "Pärnu", "aasta": 2026, "kuu": 8, "paev": 5, "vaartus": 18.7, "element_kood": "DRQS"}]
        elif "open-meteo" in url:
            payload = {
                "daily": {"time": ["2026-08-07"], "temperature_2m_min": [13.0], "temperature_2m_max": [24.0], "shortwave_radiation_sum": [19.2]},
                "hourly": {"time": ["2026-08-07T00:00", "2026-08-07T12:00"], "wind_speed_10m": [2.0, 4.0]},
            }
        else:
            raise AssertionError((url, params))
        return SimpleNamespace(ok=True, status_code=200, text="", json=lambda: payload)


def run() -> None:
    service = core.WeatherService(http=FakeHTTP())
    measured = service._parnu_daily(date(2026, 8, 5), date(2026, 8, 5))
    assert measured["2026-08-05"]["t_min"] == 12.5
    assert measured["2026-08-05"]["t_max"] == 23.0
    assert measured["2026-08-05"]["wind_avg"] == 3.5
    radiation = service._parnu_radiation(date(2026, 8, 5), date(2026, 8, 5))
    assert radiation == {"2026-08-05": 18.7}
    tested = service.test_sources(date(2026, 8, 7))
    assert tested["measured_day"] == "2026-08-05"
    assert tested["parnu_temperature_rows"] == 4
    assert tested["parnu_wind_rows"] == 4
    assert tested["parnu_radiation_rows"] == 1
    assert tested["forecast_days"] == 1
    assert tested["ok"] is True
    print("PARNU WEATHER TESTS OK")


if __name__ == "__main__":
    run()
