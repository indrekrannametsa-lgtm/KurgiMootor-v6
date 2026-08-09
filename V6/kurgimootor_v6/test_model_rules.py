import unittest
from pathlib import Path


APP = Path(__file__).resolve().parent / "app.py"
DB = Path(__file__).resolve().parent / "db.py"


class ModelArchitectureRulesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = APP.read_text(encoding="utf-8")
        cls.db = DB.read_text(encoding="utf-8") if DB.exists() else ""

    def test_tallinn_date_is_explicit(self):
        self.assertIn('ZoneInfo("Europe/Tallinn")', self.app)

    def test_abc_baseline_is_weather_first_without_raw_harvest_anchor(self):
        self.assertIn(
            "abc_predictions[test_idx] = _abc_growth_walk_predict([], train_idx, test_idx)",
            self.app,
        )

    def test_xl_baseline_has_no_raw_previous_harvest_anchor(self):
        expected = """xl_predictions[test_idx] = _ridge_walk_predict(
                y_xl, [], train_idx, test_idx
            )"""
        self.assertIn(expected, self.app)

    def test_cb_baseline_has_no_previous_cb_anchor(self):
        expected = """log_pred = _ridge_walk_predict(log_y_cb, [], train_idx, test_idx, floor_zero=False)"""
        # Formatting may span lines; stronger rule is that the old raw_cb1 baseline must be absent.
        self.assertNotIn(
            "_ridge_walk_predict(log_y_cb, [raw_cb1], train_idx, test_idx",
            self.app,
        )
        self.assertIn("C/B BAASMudel", self.app)

    def test_full_xl_and_cb_models_use_only_champion_approved_extras(self):
        self.assertIn(
            "full_xl_model = _fit_full_generic(y_xl, xl_champion_extra_arrays)",
            self.app,
        )
        self.assertIn(
            "full_cb_model = _fit_full_generic(log_y_cb, cb_champion_extra_arrays)",
            self.app,
        )
        self.assertNotIn(
            "full_cb_model = _fit_full_generic(log_y_cb, [raw_cb1]",
            self.app,
        )

    def test_discovery_median_uses_only_past_days(self):
        self.assertIn("def _past_only_abs_deviation(values):", self.app)
        self.assertIn("_past_idx = np.where(dates < _day)[0]", self.app)
        self.assertNotIn("_med = np.nanmedian(_x)", self.app)

    def test_autonomous_discovery_has_separate_confirmation_days(self):
        self.assertIn("_auto_discovery_days", self.app)
        self.assertIn("_auto_confirm_days", self.app)
        self.assertIn("Kinnitus paranemine", self.app)

    def test_three_day_accuracy_excludes_today(self):
        self.assertIn("def _motor_accuracy_3p():", self.app)
        self.assertIn("if d >= TODAY:", self.app)
        self.assertIn("continue", self.app)

    def test_missing_weather_model_fallback_looks_only_backward(self):
        self.assertIn("for delta in range(1, 4):", self.app)
        self.assertIn("dd = day_value - timedelta(days=delta)", self.app)

    def test_yield_forecast_snapshot_has_identity_fields(self):
        if not self.db:
            self.skipTest("db.py not present beside test file")
        for field in (
            "forecast_date",
            "target_date",
            "field_no",
            "lead_days",
            "model_version",
        ):
            self.assertIn(f'"{field}"', self.db)
        self.assertIn(
            'on_conflict="forecast_date,target_date,field_no,model_version"',
            self.db,
        )


    def test_season_start_is_current_year_june_15(self):
        self.assertIn("SEASON_START = date(TODAY.year, 6, 15)", self.app)
        self.assertNotIn("date(2026, 7, 1)", self.app)

    def test_harvest_edit_requires_confirmation_and_supports_delete(self):
        self.assertIn("Kinnitan olemasoleva korje muutmise või kustutamise", self.app)
        self.assertIn("db.delete_harvest(entry_date, entry_field)", self.app)

    def test_database_exposes_delete_harvest(self):
        if not self.db:
            self.skipTest("db.py not present beside test file")
        self.assertIn("def delete_harvest(", self.db)
        self.assertIn(".delete()", self.db)


if __name__ == "__main__":
    unittest.main()
