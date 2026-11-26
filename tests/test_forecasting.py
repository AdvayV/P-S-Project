"""Regression coverage for chronological next-session forecasts."""

import unittest

import numpy as np
import pandas as pd

from forecasting import evaluate_next_session


class NextSessionForecastTests(unittest.TestCase):
    def setUp(self):
        dates = pd.bdate_range("2025-01-03", periods=30)
        self.frame = pd.DataFrame({"Date": dates, "Close": 100.0 + np.arange(len(dates)) * 2})

    def test_forecast_uses_latest_row_and_targets_are_labeled_with_next_session(self):
        result = evaluate_next_session(self.frame)
        self.assertEqual(result["as_of"], self.frame["Date"].iloc[-1])
        self.assertEqual(result["last_close"], 158.0)
        self.assertAlmostEqual(result["prediction"], 160.0)
        holdout = result["holdout"]
        self.assertEqual(holdout["target_date"].iloc[0], self.frame["Date"].iloc[-6])
        self.assertEqual(holdout["target_date"].iloc[-1], result["as_of"])
        self.assertAlmostEqual(result["rmse"], 0.0)
        self.assertAlmostEqual(result["persistence_rmse"], 2.0)
        np.testing.assert_allclose(holdout["persistence"], holdout["actual"] - 2)

    def test_holdout_targets_do_not_change_the_evaluation_model(self):
        original = evaluate_next_session(self.frame)
        self.frame.loc[self.frame.index[-1], "Close"] += 40
        changed = evaluate_next_session(self.frame)
        np.testing.assert_allclose(changed["holdout"]["regression"], original["holdout"]["regression"])
        self.assertGreater(changed["rmse"], original["rmse"])
        self.assertNotEqual(changed["prediction"], original["prediction"])

    def test_latest_invalid_features_do_not_fall_back_to_older_rows(self):
        self.frame["RSI_14"] = 50.0
        self.frame.loc[self.frame.index[-1], "RSI_14"] = np.nan
        with self.assertRaisesRegex(ValueError, "most recent session"):
            evaluate_next_session(self.frame)

    def test_dates_must_not_repeat(self):
        self.frame.loc[self.frame.index[-1], "Date"] = self.frame["Date"].iloc[-2]
        with self.assertRaisesRegex(ValueError, "unique dates"):
            evaluate_next_session(self.frame)


if __name__ == "__main__":
    unittest.main()
