"""Build datasets for model training from database."""

import logging
import pandas as pd
import numpy as np
from typing import Tuple, Optional

from src.database.db_utils import get_connection, query_features_and_targets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_tabular_dataset(
    ticker: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    train_split_date: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Build tabular dataset for baseline models.
    
    Returns:
        X_train, y_train, X_test, y_test
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
    
    df = df.dropna(subset=feature_cols + ["direction_label"])
    
    X = df[feature_cols].copy()
    y = df["direction_label"].copy()
    
    if train_split_date:
        train_mask = df["date"] < pd.to_datetime(train_split_date)
        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[~train_mask]
        y_test = y[~train_mask]
    else:
        split_idx = int(len(X) * 0.8)
        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_test = y.iloc[split_idx:]
    
    logger.info(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")
    
    return X_train, y_train, X_test, y_test

