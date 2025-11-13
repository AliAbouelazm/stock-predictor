"""Build sequence datasets for LSTM/GRU models."""

import logging
import pandas as pd
import numpy as np
from typing import Tuple, Optional

from src.database.db_utils import get_connection, query_features_and_targets
from src.config import LSTM_LOOKBACK_WINDOW

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_sequence_dataset(
    ticker: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    lookback: int = LSTM_LOOKBACK_WINDOW,
    train_split_date: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build sequence dataset for LSTM/GRU models.
    
    Returns:
        X_train_seq, y_train, X_test_seq, y_test
    """
    conn = get_connection()
    df = query_features_and_targets(conn, ticker, start_date, end_date)
    conn.close()
    
    if df.empty:
        raise ValueError("No data found for given parameters")
    
    feature_cols = [
        "return_1d", "return_5d", "volatility_10d", "volatility_20d",
        "sma_10", "sma_20", "sma_50", "rsi_14",
        "macd", "macd_signal", "macd_histogram",
        "lag_return_1", "lag_return_2", "lag_return_5"
    ]
    
    df = df.sort_values("date").reset_index(drop=True)
    df = df.dropna(subset=feature_cols + ["direction_label"])
    
    X_features = df[feature_cols].values
    y_labels = df["direction_label"].values
    
    sequences = []
    labels = []
    
    for i in range(lookback, len(X_features)):
        sequences.append(X_features[i-lookback:i])
        labels.append(y_labels[i])
    
    X_seq = np.array(sequences)
    y_seq = np.array(labels)
    
    if train_split_date:
        split_idx = len(df[df["date"] < pd.to_datetime(train_split_date)]) - lookback
    else:
        split_idx = int(len(X_seq) * 0.8)
    
    X_train = X_seq[:split_idx]
    y_train = y_seq[:split_idx]
    X_test = X_seq[split_idx:]
    y_test = y_seq[split_idx:]
    
    logger.info(f"Train sequences: {len(X_train)}, Test sequences: {len(X_test)}")
    logger.info(f"Sequence shape: {X_train.shape}")
    
    return X_train, y_train, X_test, y_test

