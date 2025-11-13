"""Plot price data with prediction signals in pixel style."""

import logging
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional

from src.database.db_utils import get_connection
from src.visualization.style_pixel_theme import apply_pixel_style, plot_pixel_line, PIXEL_COLORS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_price_with_signals(
    ticker: str,
    model_name: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    save_path: Optional[str] = None
):
    """Plot price chart with prediction signals."""
    conn = get_connection()
    
    query = """
        SELECT 
            p.date,
            pr.close,
            pr.adjusted_close,
            p.predicted_direction
        FROM predictions p
        JOIN prices pr ON p.symbol_id = pr.symbol_id AND p.date = pr.date
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
        logger.warning(f"No data found for {ticker}")
        return
    
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    apply_pixel_style(fig, ax)
    
    plot_pixel_line(ax, df["date"], df["adjusted_close"], PIXEL_COLORS["price"], label="Price")
    
    up_mask = df["predicted_direction"] == 1
    down_mask = df["predicted_direction"] == -1
    flat_mask = df["predicted_direction"] == 0
    
    if up_mask.any():
        ax.scatter(df.loc[up_mask, "date"], df.loc[up_mask, "adjusted_close"],
                  color=PIXEL_COLORS["up"], marker="s", s=50, label="Up", zorder=5)
    if down_mask.any():
        ax.scatter(df.loc[down_mask, "date"], df.loc[down_mask, "adjusted_close"],
                  color=PIXEL_COLORS["down"], marker="s", s=50, label="Down", zorder=5)
    if flat_mask.any():
        ax.scatter(df.loc[flat_mask, "date"], df.loc[flat_mask, "adjusted_close"],
                  color=PIXEL_COLORS["flat"], marker="s", s=50, label="Flat", zorder=5)
    
    ax.set_xlabel("Date", fontsize=12, fontweight="bold")
    ax.set_ylabel("Price", fontsize=12, fontweight="bold")
    ax.set_title(f"{ticker} - Price with {model_name} Signals", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", facecolor=PIXEL_COLORS["background"], edgecolor=PIXEL_COLORS["grid"])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, facecolor=PIXEL_COLORS["background"], dpi=150)
        logger.info(f"Saved plot to {save_path}")
    else:
        plt.show()
    
    plt.close()

