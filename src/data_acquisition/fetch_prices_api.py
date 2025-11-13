"""Fetch stock prices from API (placeholder structure)."""

import logging
import os
import pandas as pd
from typing import List, Optional
from datetime import datetime

from src.database.db_utils import get_connection, get_or_create_symbol, insert_prices

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_prices_alpha_vantage(ticker: str, api_key: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch daily prices from Alpha Vantage API.
    
    Args:
        ticker: Stock ticker symbol
        api_key: API key (or from environment variable ALPHA_VANTAGE_API_KEY)
    
    Returns:
        DataFrame with columns: date, open, high, low, close, adjusted_close, volume
    """
    if api_key is None:
        api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    
    if not api_key:
        logger.warning("No API key provided. Returning empty DataFrame.")
        logger.info("Set ALPHA_VANTAGE_API_KEY environment variable or pass api_key parameter")
        return pd.DataFrame()
    
    import requests
    
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": ticker,
        "apikey": api_key,
        "outputsize": "full",
        "datatype": "csv"
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        if "Error Message" in response.text or "Invalid API" in response.text:
            logger.error(f"API error for {ticker}: {response.text[:200]}")
            return pd.DataFrame()
        
        df = pd.read_csv(url + "?" + "&".join([f"{k}={v}" for k, v in params.items()]))
        
        df = df.rename(columns={
            "timestamp": "date",
            "adjusted_close": "adjusted_close",
            "close": "close"
        })
        
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        
        if "adjusted_close" not in df.columns:
            df["adjusted_close"] = df["close"]
        
        return df[["date", "open", "high", "low", "close", "adjusted_close", "volume"]]
    
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()


def fetch_and_store_prices(tickers: List[str], api_key: Optional[str] = None) -> None:
    """Fetch prices for multiple tickers and store in database."""
    conn = get_connection()
    
    for ticker in tickers:
        logger.info(f"Fetching prices for {ticker}")
        df = fetch_prices_alpha_vantage(ticker, api_key)
        
        if df.empty:
            logger.warning(f"No data fetched for {ticker}")
            continue
        
        symbol_id = get_or_create_symbol(conn, ticker)
        insert_prices(conn, symbol_id, df)
        logger.info(f"Stored {len(df)} rows for {ticker}")
    
    conn.close()


if __name__ == "__main__":
    from src.config import DEFAULT_TICKERS
    
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    fetch_and_store_prices(DEFAULT_TICKERS, api_key)

