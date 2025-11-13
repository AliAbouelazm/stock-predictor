"""Clean and normalize price data."""

import logging
import pandas as pd
from typing import Optional

from src.database.db_utils import get_connection, query_features_and_targets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_price_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean price DataFrame.
    
    Args:
        df: DataFrame with price data
    
    Returns:
        Cleaned DataFrame
    """
    df = df.copy()
    
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
    
    df = df.drop_duplicates(subset=["date"], keep="first")
    
    critical_cols = ["close", "adjusted_close"]
    df = df.dropna(subset=critical_cols)
    
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    
    for col in ["open", "high", "low", "close", "adjusted_close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df[df[col] > 0]
    
    return df.reset_index(drop=True)


if __name__ == "__main__":
    conn = get_connection()
    df = query_features_and_targets(conn)
    if not df.empty:
        cleaned = clean_price_dataframe(df)
        logger.info(f"Cleaned data: {len(df)} -> {len(cleaned)} rows")

