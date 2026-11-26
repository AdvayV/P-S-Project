"""Offline smoke test for the two forecast views."""

from io import BytesIO
from pathlib import Path
import unittest
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "GOOGL_5y_1d.csv"


class ForecastDashboardTests(unittest.TestCase):
    def test_bundled_csv_renders_both_forecasts(self):
        def sample_upload(*args, **kwargs):
            if kwargs.get("key") == "csv_stock_1":
                csv = BytesIO(SAMPLE.read_bytes())
                csv.name = SAMPLE.name
                return csv
            return None

        with patch("streamlit.file_uploader", side_effect=sample_upload):
            app = AppTest.from_file(str(ROOT / "main.py")).run(timeout=40)

        self.assertEqual(len(app.tabs), 7)
        self.assertFalse(app.exception, [error.message for error in app.exception])
        baselines = [item for item in app.metric if item.label == "Persistence RMSE"]
        self.assertEqual(len(baselines), 2)
        latest_date = pd.to_datetime(pd.read_csv(SAMPLE)["Date"]).max()
        self.assertTrue(any(f"As of {latest_date:%Y-%m-%d}" in item.value for item in app.caption))
