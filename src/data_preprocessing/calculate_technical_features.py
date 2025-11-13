"""Calculate technical indicators and features."""

import logging
import pandas as pd
import numpy as np
from typing import Optional

from src.database.db_utils import get_connection, query_features_and_targets, insert_features, get_or_create_symbol
from src.config import DB_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
    """Calculate MACD, signal line, and histogram."""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_histogram = macd - macd_signal
    return macd, macd_signal, macd_histogram


def calculate_technical_features(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators from price data.
    
    Args:
        prices_df: DataFrame with columns: date, close, adjusted_close, volume
    
    Returns:
        DataFrame with technical features
    """
    df = prices_df.copy()
    df = df.sort_values("date").reset_index(drop=True)
    
    close = df["adjusted_close"]
    
    df["return_1d"] = close.pct_change()
    df["return_5d"] = close.pct_change(5)
    
    df["volatility_10d"] = df["return_1d"].rolling(window=10).std()
    df["volatility_20d"] = df["return_1d"].rolling(window=20).std()
    
    df["sma_10"] = close.rolling(window=10).mean()
    df["sma_20"] = close.rolling(window=20).mean()
    df["sma_50"] = close.rolling(window=50).mean()
    
    df["rsi_14"] = calculate_rsi(close, period=14)
    
    macd, macd_signal, macd_histogram = calculate_macd(close)
    df["macd"] = macd
    df["macd_signal"] = macd_signal
    df["macd_histogram"] = macd_histogram
    
    df["lag_return_1"] = df["return_1d"].shift(1)
    df["lag_return_2"] = df["return_1d"].shift(2)
    df["lag_return_5"] = df["return_1d"].shift(5)
    
    return df[["date", "return_1d", "return_5d", "volatility_10d", "volatility_20d",
               "sma_10", "sma_20", "sma_50", "rsi_14", "macd", "macd_signal",
               "macd_histogram", "lag_return_1", "lag_return_2", "lag_return_5"]]


def compute_and_store_features(ticker: Optional[str] = None) -> None:
    """Compute features for all symbols or a specific ticker and store in database."""
    conn = get_connection()
    
    query = """
        SELECT s.ticker, s.id as symbol_id, p.date, p.close, p.adjusted_close, p.volume
        FROM prices p
        JOIN symbols s ON p.symbol_id = s.id
    """
    
    if ticker:
        query += " WHERE s.ticker = ?"
        params = (ticker,)
    else:
        params = ()
    
    query += " ORDER BY s.ticker, p.date"
    
    prices_df = pd.read_sql_query(query, conn, params=params)
    
    if prices_df.empty:
        logger.warning("No price data found")
        conn.close()
        return
    
    prices_df["date"] = pd.to_datetime(prices_df["date"])
    
    for ticker_name in prices_df["ticker"].unique():
        ticker_data = prices_df[prices_df["ticker"] == ticker_name].copy()
        logger.info(f"Computing features for {ticker_name}")
        
        features_df = calculate_technical_features(ticker_data)
        symbol_id = get_or_create_symbol(conn, ticker_name)
        insert_features(conn, symbol_id, features_df)
        logger.info(f"Stored {len(features_df)} feature rows for {ticker_name}")
    
    conn.close()


if __name__ == "__main__":
    compute_and_store_features()

