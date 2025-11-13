"""Plot backtest performance in pixel style."""

import logging
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Optional

from src.visualization.style_pixel_theme import apply_pixel_style, plot_pixel_line, PIXEL_COLORS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_backtest_performance(
    backtest_results: Dict,
    save_path: Optional[str] = None
):
    """Plot cumulative returns comparison."""
    if not backtest_results:
        logger.warning("No backtest results to plot")
        return
    
    dates = pd.to_datetime(backtest_results["dates"])
    strategy_value = backtest_results["portfolio_value"]
    buy_hold_value = backtest_results["buy_hold_value"]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    apply_pixel_style(fig, ax)
    
    plot_pixel_line(ax, dates, strategy_value, PIXEL_COLORS["up"], 
                   label=f"Strategy ({backtest_results['model_name']})")
    plot_pixel_line(ax, dates, buy_hold_value, PIXEL_COLORS["price"], 
                   label="Buy & Hold")
    
    ax.set_xlabel("Date", fontsize=12, fontweight="bold")
    ax.set_ylabel("Portfolio Value ($)", fontsize=12, fontweight="bold")
    ax.set_title(f"{backtest_results['ticker']} - Strategy vs Buy & Hold", 
                fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", facecolor=PIXEL_COLORS["background"], 
             edgecolor=PIXEL_COLORS["grid"])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, facecolor=PIXEL_COLORS["background"], dpi=150)
        logger.info(f"Saved plot to {save_path}")
    else:
        plt.show()
    
    plt.close()

