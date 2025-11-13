"""Streamlit app for StockOracle with retro pixel style."""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, date
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.database.db_utils import get_connection, initialize_schema
from src.models.time_series_backtest import backtest_model
from src.visualization.plot_price_and_signals import plot_price_with_signals
from src.visualization.plot_performance import plot_backtest_performance
from src.visualization.style_pixel_theme import PIXEL_COLORS

st.set_page_config(page_title="StockOracle", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap');
    
    * {
        font-family: 'Courier New', monospace;
    }
    
    .main {
        background-color: #000000;
    }
    
    .stApp {
        background-color: #000000;
    }
    
    h1, h2, h3 {
        color: #00FF00;
        font-family: 'Courier New', monospace;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    .stButton>button {
        background-color: #00FF00;
        color: #000000;
        border: 3px solid #00FF00;
        font-family: 'Courier New', monospace;
        font-weight: bold;
        padding: 0.5rem 1rem;
        border-radius: 0;
    }
    
    .stButton>button:hover {
        background-color: #00CC00;
        border-color: #00CC00;
    }
    
    .stSelectbox label, .stDateInput label {
        color: #00FF00;
        font-family: 'Courier New', monospace;
        font-weight: bold;
    }
    
    .stMetric {
        background-color: #111111;
        border: 2px solid #00FF00;
        padding: 1rem;
    }
    
    .stMetric label {
        color: #00FF00;
        font-family: 'Courier New', monospace;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        color: #00FF00;
        font-family: 'Courier New', monospace;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.title("STOCKORACLE")

conn = get_connection()
try:
    symbols_df = pd.read_sql_query("SELECT ticker FROM symbols ORDER BY ticker", conn)
    tickers = symbols_df["ticker"].tolist() if not symbols_df.empty else []
except:
    tickers = []

if not tickers:
    st.error("No tickers found in database. Please load data first.")
    st.stop()

with st.sidebar:
    st.header("CONTROLS")
    selected_ticker = st.selectbox("TICKER", tickers)
    model_choice = st.selectbox("MODEL", ["Logistic Regression", "Random Forest", "LSTM"])
    
    start_date = st.date_input("START DATE", value=date(2024, 1, 1))
    end_date = st.date_input("END DATE", value=date.today())
    
    run_button = st.button("RUN ANALYSIS", type="primary")

if run_button:
    model_map = {
        "Logistic Regression": "logistic_regression",
        "Random Forest": "random_forest",
        "LSTM": "lstm_model"
    }
    model_name = model_map[model_choice]
    
    try:
        with st.spinner("Running backtest..."):
            results = backtest_model(
                selected_ticker,
                model_name,
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d")
            )
        
        if results:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("TOTAL RETURN", f"{results['total_return']:.2f}%")
            with col2:
                st.metric("BUY & HOLD", f"{results['buy_hold_return']:.2f}%")
            with col3:
                st.metric("SHARPE RATIO", f"{results['sharpe_ratio']:.4f}")
            with col4:
                st.metric("MAX DRAWDOWN", f"{results['max_drawdown']:.2f}%")
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("PRICE & SIGNALS")
                fig1, ax1 = plt.subplots(figsize=(10, 6))
                ax1.set_facecolor(PIXEL_COLORS["background"])
                fig1.patch.set_facecolor(PIXEL_COLORS["background"])
                
                dates = pd.to_datetime(results["dates"])
                prices_df = pd.read_sql_query("""
                    SELECT date, adjusted_close FROM prices p
                    JOIN symbols s ON p.symbol_id = s.id
                    WHERE s.ticker = ? AND date >= ? AND date <= ?
                    ORDER BY date
                """, conn, params=[selected_ticker, start_date, end_date])
                
                if not prices_df.empty:
                    prices_df["date"] = pd.to_datetime(prices_df["date"])
                    ax1.plot(prices_df["date"], prices_df["adjusted_close"], 
                            color=PIXEL_COLORS["price"], linewidth=3, marker="s", markersize=3)
                    ax1.set_facecolor(PIXEL_COLORS["background"])
                    ax1.tick_params(colors=PIXEL_COLORS["text"])
                    ax1.set_xlabel("Date", color=PIXEL_COLORS["text"])
                    ax1.set_ylabel("Price", color=PIXEL_COLORS["text"])
                    ax1.set_title(f"{selected_ticker} Price", color=PIXEL_COLORS["text"], fontweight="bold")
                    plt.tight_layout()
                    st.pyplot(fig1)
            
            with col2:
                st.subheader("PERFORMANCE")
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                ax2.set_facecolor(PIXEL_COLORS["background"])
                fig2.patch.set_facecolor(PIXEL_COLORS["background"])
                
                ax2.plot(dates, results["portfolio_value"], 
                        color=PIXEL_COLORS["up"], linewidth=3, marker="s", markersize=3, label="Strategy")
                ax2.plot(dates, results["buy_hold_value"], 
                        color=PIXEL_COLORS["price"], linewidth=3, marker="s", markersize=3, label="Buy & Hold")
                ax2.set_facecolor(PIXEL_COLORS["background"])
                ax2.tick_params(colors=PIXEL_COLORS["text"])
                ax2.set_xlabel("Date", color=PIXEL_COLORS["text"])
                ax2.set_ylabel("Portfolio Value", color=PIXEL_COLORS["text"])
                ax2.set_title("Cumulative Returns", color=PIXEL_COLORS["text"], fontweight="bold")
                ax2.legend(facecolor=PIXEL_COLORS["background"], edgecolor=PIXEL_COLORS["grid"])
                plt.tight_layout()
                st.pyplot(fig2)
        
    except Exception as e:
        st.error(f"Error: {str(e)}")

conn.close()

