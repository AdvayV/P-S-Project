"""Chronological evaluation and next-session forecasts for daily OHLC data."""

from math import ceil

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

FEATURES = (
    "Open", "High", "Low", "Close", "Volume", "Expected_High_68", "Expected_Low_68", "RSI_14",
)


def evaluate_next_session(frame, test_fraction=0.2):
    """Wait until session close, then predict the next observation's close."""
    dates = frame["Date"]
    if dates.empty or dates.isna().any() or not dates.is_monotonic_increasing or not dates.is_unique:
        raise ValueError("Use a daily CSV with unique dates in chronological order.")
    if not 0 < test_fraction < 1:
        raise ValueError("The holdout fraction must be between zero and one.")

    features = [column for column in FEATURES if column in frame.columns]
    inputs = frame[features].replace([np.inf, -np.inf], np.nan)
    latest_inputs = inputs.iloc[[-1]]
    if latest_inputs.isna().any(axis=None):
        raise ValueError("The most recent session is missing a complete set of model inputs.")

    targets = frame["Close"].shift(-1).replace([np.inf, -np.inf], np.nan)
    target_dates = dates.shift(-1)
    usable = inputs.notna().all(axis=1) & targets.notna()
    inputs = inputs.loc[usable]
    targets = targets.loc[usable]
    target_dates = target_dates.loc[usable]

    test_count = max(2, ceil(len(inputs) * test_fraction))
    train_count = len(inputs) - test_count
    if train_count <= len(features):
        raise ValueError("Upload more daily observations so rolling features, training and holdout are available.")

    train_inputs, test_inputs = inputs.iloc[:train_count], inputs.iloc[train_count:]
    train_targets, test_targets = targets.iloc[:train_count], targets.iloc[train_count:]
    model = LinearRegression().fit(train_inputs, train_targets)
    predictions = model.predict(test_inputs)
    persistence = test_inputs["Close"]

    holdout = pd.DataFrame({
        "target_date": target_dates.iloc[train_count:],
        "actual": test_targets,
        "regression": predictions,
        "persistence": persistence,
    })
    coefficients = pd.Series(model.coef_, index=features, name="Coefficient")

    # The forecast can use every target already known as of the latest session.
    forecast_model = LinearRegression().fit(inputs, targets)
    return {
        "as_of": dates.iloc[-1],
        "last_close": float(latest_inputs["Close"].iloc[0]),
        "prediction": float(forecast_model.predict(latest_inputs)[0]),
        "holdout": holdout,
        "train_count": train_count,
        "rmse": float(np.sqrt(mean_squared_error(test_targets, predictions))),
        "persistence_rmse": float(np.sqrt(mean_squared_error(test_targets, persistence))),
        "r2": float(r2_score(test_targets, predictions)),
        "coefficients": coefficients,
    }
