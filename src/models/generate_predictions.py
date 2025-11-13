"""Generate predictions using trained models and store in database."""

import logging
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Optional

import tensorflow as tf
from tensorflow import keras

from src.database.db_utils import get_connection, get_or_create_symbol, insert_predictions, query_features_and_targets
from src.models.build_datasets import build_tabular_dataset
from src.models.sequence_dataset import build_sequence_dataset
from src.config import MODELS_DIR, LSTM_LOOKBACK_WINDOW

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_baseline_predictions(ticker: str, model_name: str, start_date: Optional[str] = None, end_date: Optional[str] = None):
    """Generate predictions using baseline model."""
    model_path = MODELS_DIR / f"{model_name}.pkl"
    
    if not model_path.exists():
        logger.warning(f"Model not found: {model_path}")
        return
    
    model = joblib.load(model_path)
    
    conn = get_connection()
    df = query_features_and_targets(conn, ticker, start_date, end_date)
    conn.close()
    
    if df.empty:
        logger.warning(f"No data found for {ticker}")
        return
    
    feature_cols = [
        "return_1d", "return_5d", "volatility_10d", "volatility_20d",
        "sma_10", "sma_20", "sma_50", "rsi_14",
        "macd", "macd_signal", "macd_histogram",
        "lag_return_1", "lag_return_2", "lag_return_5"
    ]
    
    df = df.dropna(subset=feature_cols)
    X = df[feature_cols]
    
    predictions = model.predict(X)
    
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        if probabilities.shape[1] == 3:
            prob_up = probabilities[:, 2]
            prob_flat = probabilities[:, 1]
            prob_down = probabilities[:, 0]
        else:
            prob_up = None
            prob_flat = None
            prob_down = None
    else:
        prob_up = None
        prob_flat = None
        prob_down = None
    
    predictions_df = pd.DataFrame({
        "date": df["date"],
        "predicted_direction": predictions,
        "prob_up": prob_up,
        "prob_flat": prob_flat,
        "prob_down": prob_down
    })
    
    conn = get_connection()
    symbol_id = get_or_create_symbol(conn, ticker)
    insert_predictions(conn, symbol_id, predictions_df, model_name)
    conn.close()
    
    logger.info(f"Generated {len(predictions_df)} predictions for {ticker}")


def generate_lstm_predictions(ticker: str, start_date: Optional[str] = None, end_date: Optional[str] = None):
    """Generate predictions using LSTM model."""
    model_path = MODELS_DIR / "lstm_model.h5"
    
    if not model_path.exists():
        logger.warning(f"Model not found: {model_path}")
        return
    
    model = keras.models.load_model(model_path)
    
    X_train, y_train, X_test, y_test = build_sequence_dataset(ticker, start_date, end_date, lookback=LSTM_LOOKBACK_WINDOW)
    
    if len(X_test) == 0:
        logger.warning(f"No test sequences for {ticker}")
        return
    
    y_test_shifted = y_test + 1
    predictions_proba = model.predict(X_test)
    predictions = np.argmax(predictions_proba, axis=1) - 1
    
    conn = get_connection()
    df = query_features_and_targets(conn, ticker, start_date, end_date)
    conn.close()
    
    if df.empty:
        logger.warning(f"No data found for {ticker}")
        return
    
    df = df.sort_values("date").reset_index(drop=True)
    feature_cols = [
        "return_1d", "return_5d", "volatility_10d", "volatility_20d",
        "sma_10", "sma_20", "sma_50", "rsi_14",
        "macd", "macd_signal", "macd_histogram",
        "lag_return_1", "lag_return_2", "lag_return_5"
    ]
    df = df.dropna(subset=feature_cols + ["direction_label"])
    
    start_idx = LSTM_LOOKBACK_WINDOW
    dates = df["date"].iloc[start_idx:start_idx+len(predictions)].values
    
    predictions_df = pd.DataFrame({
        "date": dates,
        "predicted_direction": predictions,
        "prob_up": predictions_proba[:, 2],
        "prob_flat": predictions_proba[:, 1],
        "prob_down": predictions_proba[:, 0]
    })
    
    conn = get_connection()
    symbol_id = get_or_create_symbol(conn, ticker)
    insert_predictions(conn, symbol_id, predictions_df, "lstm_model")
    conn.close()
    
    logger.info(f"Generated {len(predictions_df)} LSTM predictions for {ticker}")


if __name__ == "__main__":
    from src.config import DEFAULT_TICKERS, TEST_START_DATE, TEST_END_DATE
    
    for ticker in DEFAULT_TICKERS[:3]:
        logger.info(f"Generating predictions for {ticker}")
        generate_baseline_predictions(ticker, "logistic_regression", TEST_START_DATE, TEST_END_DATE)
        generate_baseline_predictions(ticker, "random_forest", TEST_START_DATE, TEST_END_DATE)
        generate_lstm_predictions(ticker, TEST_START_DATE, TEST_END_DATE)

