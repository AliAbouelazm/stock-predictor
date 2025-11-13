"""Load stock prices from CSV files into database."""

import logging
import pandas as pd
from pathlib import Path
from typing import Optional

from src.database.db_utils import get_connection, get_or_create_symbol, insert_prices
from src.config import RAW_DATA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def normalize_price_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to match schema."""
    column_mapping = {
        "Date": "date",
        "DATE": "date",
        "timestamp": "date",
        "Open": "open",
        "OPEN": "open",
        "High": "high",
        "HIGH": "high",
        "Low": "low",
        "LOW": "low",
        "Close": "close",
        "CLOSE": "close",
        "Adj Close": "adjusted_close",
        "Adjusted Close": "adjusted_close",
        "ADJ_CLOSE": "adjusted_close",
        "Volume": "volume",
        "VOLUME": "volume"
    }
    
    df = df.rename(columns=column_mapping)
    
    if "adjusted_close" not in df.columns and "close" in df.columns:
        df["adjusted_close"] = df["close"]
    
    return df


def load_prices_from_csv(file_path: Path, ticker: str) -> None:
    """
    Load prices from CSV file and store in database.
    
    Args:
        file_path: Path to CSV file
        ticker: Stock ticker symbol
    """
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return
    
    df = pd.read_csv(file_path)
    df = normalize_price_columns(df)
    
    if "date" not in df.columns:
        logger.error(f"CSV must contain a date column. Found: {df.columns.tolist()}")
        return
    
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    
    required_cols = ["date", "open", "high", "low", "close", "adjusted_close", "volume"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        logger.warning(f"Missing columns: {missing_cols}. Filling with NaN.")
        for col in missing_cols:
            if col == "adjusted_close":
                df[col] = df.get("close", pd.Series())
            else:
                df[col] = None
    
    conn = get_connection()
    symbol_id = get_or_create_symbol(conn, ticker)
    insert_prices(conn, symbol_id, df[required_cols])
    conn.close()
    
    logger.info(f"Loaded {len(df)} rows for {ticker}")


if __name__ == "__main__":
    csv_file = RAW_DATA_DIR / "sample_prices.csv"
    if csv_file.exists():
        load_prices_from_csv(csv_file, "AAPL")
    else:
        logger.info(f"Place CSV file at {csv_file} to load prices")

