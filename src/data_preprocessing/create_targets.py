"""Create target variables (next day return and direction labels)."""

import logging
import pandas as pd
from typing import Optional

from src.database.db_utils import get_connection, get_or_create_symbol, insert_targets
from src.config import DIRECTION_THRESHOLD_UP, DIRECTION_THRESHOLD_DOWN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_targets_from_prices(
    prices_df: pd.DataFrame,
    threshold_up: float = DIRECTION_THRESHOLD_UP,
    threshold_down: float = DIRECTION_THRESHOLD_DOWN
) -> pd.DataFrame:
    """
    Create target variables from price data.
    
    Args:
        prices_df: DataFrame with date and adjusted_close columns
        threshold_up: Threshold for "Up" label
        threshold_down: Threshold for "Down" label
    
    Returns:
        DataFrame with date, next_day_return, direction_label
    """
    df = prices_df.copy()
    df = df.sort_values("date").reset_index(drop=True)
    
    close = df["adjusted_close"]
    df["next_day_return"] = close.shift(-1) / close - 1
    
    df["direction_label"] = 0
    df.loc[df["next_day_return"] > threshold_up, "direction_label"] = 1
    df.loc[df["next_day_return"] < threshold_down, "direction_label"] = -1
    
    df = df[["date", "next_day_return", "direction_label"]].dropna()
    
    return df


def compute_and_store_targets(ticker: Optional[str] = None) -> None:
    """Compute targets for all symbols or a specific ticker and store in database."""
    conn = get_connection()
    
    query = """
        SELECT s.ticker, s.id as symbol_id, p.date, p.adjusted_close
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
        logger.info(f"Computing targets for {ticker_name}")
        
        targets_df = create_targets_from_prices(ticker_data)
        symbol_id = get_or_create_symbol(conn, ticker_name)
        insert_targets(conn, symbol_id, targets_df)
        logger.info(f"Stored {len(targets_df)} target rows for {ticker_name}")
    
    conn.close()


if __name__ == "__main__":
    compute_and_store_targets()

