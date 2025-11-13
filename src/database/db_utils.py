"""Database utilities for SQLite operations."""

import sqlite3
import logging
from pathlib import Path
from typing import Optional, List, Tuple
import pandas as pd

from src.config import DB_PATH, PROJECT_ROOT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Get connection to SQLite database."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def initialize_schema(conn: Optional[sqlite3.Connection] = None) -> None:
    """Initialize database schema from schema.sql."""
    if conn is None:
        conn = get_connection()
    
    schema_path = PROJECT_ROOT / "src" / "database" / "schema.sql"
    
    with open(schema_path, "r") as f:
        schema_sql = f.read()
    
    conn.executescript(schema_sql)
    conn.commit()
    logger.info("Database schema initialized")


def get_or_create_symbol(conn: sqlite3.Connection, ticker: str, name: Optional[str] = None) -> int:
    """Get symbol ID or create if doesn't exist."""
    cursor = conn.execute("SELECT id FROM symbols WHERE ticker = ?", (ticker,))
    row = cursor.fetchone()
    
    if row:
        return row["id"]
    
    conn.execute("INSERT INTO symbols (ticker, name) VALUES (?, ?)", (ticker, name))
    conn.commit()
    symbol_id = cursor.lastrowid if cursor.lastrowid else conn.lastrowid
    return symbol_id


def insert_prices(conn: sqlite3.Connection, symbol_id: int, prices_df: pd.DataFrame) -> None:
    """Insert or replace price data."""
    for _, row in prices_df.iterrows():
        conn.execute("""
            INSERT OR REPLACE INTO prices 
            (symbol_id, date, open, high, low, close, adjusted_close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            symbol_id,
            row["date"],
            row.get("open"),
            row.get("high"),
            row.get("low"),
            row.get("close"),
            row.get("adjusted_close", row.get("close")),
            row.get("volume")
        ))
    conn.commit()


def insert_features(conn: sqlite3.Connection, symbol_id: int, features_df: pd.DataFrame) -> None:
    """Insert or replace feature data."""
    feature_cols = [
        "date", "return_1d", "return_5d", "volatility_10d", "volatility_20d",
        "sma_10", "sma_20", "sma_50", "rsi_14", "macd", "macd_signal",
        "macd_histogram", "lag_return_1", "lag_return_2", "lag_return_5"
    ]
    
    for _, row in features_df.iterrows():
        values = [symbol_id, row["date"]] + [row.get(col) for col in feature_cols[1:]]
        placeholders = ",".join(["?"] * len(values))
        cols = "symbol_id,date," + ",".join(feature_cols[1:])
        
        conn.execute(f"""
            INSERT OR REPLACE INTO features ({cols})
            VALUES ({placeholders})
        """, values)
    conn.commit()


def insert_targets(conn: sqlite3.Connection, symbol_id: int, targets_df: pd.DataFrame) -> None:
    """Insert or replace target data."""
    for _, row in targets_df.iterrows():
        conn.execute("""
            INSERT OR REPLACE INTO targets 
            (symbol_id, date, next_day_return, direction_label)
            VALUES (?, ?, ?, ?)
        """, (
            symbol_id,
            row["date"],
            row["next_day_return"],
            row["direction_label"]
        ))
    conn.commit()


def insert_predictions(conn: sqlite3.Connection, symbol_id: int, predictions_df: pd.DataFrame, model_name: str) -> None:
    """Insert or replace prediction data."""
    for _, row in predictions_df.iterrows():
        conn.execute("""
            INSERT OR REPLACE INTO predictions 
            (symbol_id, date, model_name, predicted_direction, predicted_return,
             prob_up, prob_flat, prob_down)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            symbol_id,
            row["date"],
            model_name,
            row.get("predicted_direction"),
            row.get("predicted_return"),
            row.get("prob_up"),
            row.get("prob_flat"),
            row.get("prob_down")
        ))
    conn.commit()


def query_features_and_targets(
    conn: sqlite3.Connection,
    ticker: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """Query features and targets joined together."""
    query = """
        SELECT 
            s.ticker,
            f.date,
            f.return_1d, f.return_5d, f.volatility_10d, f.volatility_20d,
            f.sma_10, f.sma_20, f.sma_50, f.rsi_14,
            f.macd, f.macd_signal, f.macd_histogram,
            f.lag_return_1, f.lag_return_2, f.lag_return_5,
            t.next_day_return, t.direction_label
        FROM features f
        JOIN symbols s ON f.symbol_id = s.id
        JOIN targets t ON f.symbol_id = t.symbol_id AND f.date = t.date
        WHERE 1=1
    """
    
    params = []
    if ticker:
        query += " AND s.ticker = ?"
        params.append(ticker)
    if start_date:
        query += " AND f.date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND f.date <= ?"
        params.append(end_date)
    
    query += " ORDER BY s.ticker, f.date"
    
    df = pd.read_sql_query(query, conn, params=params)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df

