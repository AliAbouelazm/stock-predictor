"""Create sample stock data for testing."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

from src.database.db_utils import get_connection, initialize_schema, get_or_create_symbol, insert_prices
from src.data_preprocessing.calculate_technical_features import compute_and_store_features
from src.data_preprocessing.create_targets import compute_and_store_targets
from src.config import DEFAULT_TICKERS

np.random.seed(42)

conn = get_connection()
initialize_schema(conn)

start_date = datetime(2020, 1, 1)
dates = pd.date_range(start_date, periods=1000, freq="D")

for ticker in DEFAULT_TICKERS[:3]:
    print(f"Creating sample data for {ticker}...")
    
    prices = 100 + np.cumsum(np.random.randn(len(dates)) * 2)
    prices = np.maximum(prices, 10)
    
    prices_df = pd.DataFrame({
        "date": dates,
        "open": prices * (1 + np.random.randn(len(dates)) * 0.01),
        "high": prices * (1 + np.abs(np.random.randn(len(dates)) * 0.02)),
        "low": prices * (1 - np.abs(np.random.randn(len(dates)) * 0.02)),
        "close": prices,
        "adjusted_close": prices,
        "volume": np.random.randint(1000000, 10000000, len(dates))
    })
    
    symbol_id = get_or_create_symbol(conn, ticker)
    insert_prices(conn, symbol_id, prices_df)
    print(f"  Inserted {len(prices_df)} price rows")

conn.close()

print("\nComputing features...")
compute_and_store_features()

print("Computing targets...")
compute_and_store_targets()

print("\n✅ Sample data created!")

