"""Backtesting for time series predictions."""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional

from src.database.db_utils import get_connection, query_features_and_targets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def backtest_strategy(
    predictions: pd.Series,
    actual_returns: pd.Series,
    initial_capital: float = 10000.0
) -> Dict:
    """
    Backtest trading strategy based on predictions.
    
    Args:
        predictions: Series of predicted directions (-1, 0, 1)
        actual_returns: Series of actual next-day returns
        initial_capital: Starting capital
    
    Returns:
        Dictionary with backtest results
    """
    positions = predictions.copy()
    positions[positions == 0] = 0
    positions[positions == 1] = 1
    positions[positions == -1] = -1
    
    strategy_returns = positions * actual_returns
    
    cumulative_returns = (1 + strategy_returns).cumprod()
    cumulative_returns.iloc[0] = 1.0
    
    portfolio_value = initial_capital * cumulative_returns
    
    buy_hold_returns = (1 + actual_returns).cumprod()
    buy_hold_returns.iloc[0] = 1.0
    buy_hold_value = initial_capital * buy_hold_returns
    
    total_return = (portfolio_value.iloc[-1] / initial_capital - 1) * 100
    buy_hold_return = (buy_hold_value.iloc[-1] / initial_capital - 1) * 100
    
    sharpe_ratio = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std() if strategy_returns.std() > 0 else 0
    
    max_drawdown = ((cumulative_returns / cumulative_returns.expanding().max()) - 1).min() * 100
    
    results = {
        "total_return": total_return,
        "buy_hold_return": buy_hold_return,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "strategy_returns": strategy_returns,
        "cumulative_returns": cumulative_returns,
        "portfolio_value": portfolio_value,
        "buy_hold_value": buy_hold_value
    }
    
    return results


def backtest_model(
    ticker: str,
    model_name: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Dict:
    """Backtest a specific model on a ticker."""
    conn = get_connection()
    
    query = """
        SELECT 
            p.date,
            p.predicted_direction,
            t.next_day_return,
            t.direction_label
        FROM predictions p
        JOIN targets t ON p.symbol_id = t.symbol_id AND p.date = t.date
        JOIN symbols s ON p.symbol_id = s.id
        WHERE s.ticker = ? AND p.model_name = ?
    """
    
    params = [ticker, model_name]
    if start_date:
        query += " AND p.date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND p.date <= ?"
        params.append(end_date)
    
    query += " ORDER BY p.date"
    
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    
    if df.empty:
        logger.warning(f"No predictions found for {ticker} with model {model_name}")
        return {}
    
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    
    results = backtest_strategy(
        df["predicted_direction"],
        df["next_day_return"]
    )
    
    results["dates"] = df["date"].values
    results["ticker"] = ticker
    results["model_name"] = model_name
    
    logger.info(f"Backtest results for {ticker} ({model_name}):")
    logger.info(f"  Total Return: {results['total_return']:.2f}%")
    logger.info(f"  Buy & Hold: {results['buy_hold_return']:.2f}%")
    logger.info(f"  Sharpe Ratio: {results['sharpe_ratio']:.4f}")
    
    return results

