"""Streamlit app for stockly with retro pixel style."""

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

st.set_page_config(page_title="stockly", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=JetBrains+Mono:wght@100;200;300;400&family=Orbitron:wght@400;500;600;700;800;900&display=swap');
    
    * {
        font-family: 'JetBrains Mono', 'Space Mono', monospace;
        font-weight: 200;
        letter-spacing: 1.5px;
    }
    
    .main {
        background-color: #1a1a1a;
    }
    
    .stApp {
        background-color: #1a1a1a;
    }
    
    .logo-header {
        background: #1a1a1a;
        border-bottom: 1px solid #333333;
        padding: 1.5rem 0;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .logo-text {
        font-family: 'Orbitron', sans-serif;
        font-weight: 700;
        font-size: 3.5rem;
        color: #000000;
        letter-spacing: 8px;
        text-transform: uppercase;
        margin: 0;
    }
    
    h1, h2, h3 {
        color: #00FF00;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 200;
        letter-spacing: 4px;
        text-transform: lowercase;
        font-size: 2.5rem;
    }
    
    h2 {
        font-size: 1.8rem;
        letter-spacing: 3px;
    }
    
    h3 {
        font-size: 1.3rem;
        letter-spacing: 2px;
    }
    
    .stButton>button {
        background-color: #00FF00;
        color: #000000;
        border: 2px solid #00FF00;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 200;
        padding: 0.4rem 1.2rem;
        border-radius: 0;
        letter-spacing: 2px;
        font-size: 0.9rem;
    }
    
    .stButton>button:hover {
        background-color: #00CC00;
        border-color: #00CC00;
    }
    
    .stSelectbox label, .stDateInput label {
        color: #00FF00;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 200;
        letter-spacing: 1px;
        font-size: 0.85rem;
    }
    
    .stMetric {
        background-color: #0a0a0a;
        border: 1px solid #00FF00;
        padding: 1rem;
    }
    
    .stMetric label {
        color: #00FF00;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 200;
        letter-spacing: 1px;
        font-size: 0.75rem;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        color: #00FF00;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 200;
        letter-spacing: 1px;
        font-size: 1.5rem;
    }
    
    body, p, div, span {
        font-weight: 200;
        letter-spacing: 1px;
    }
    
    .sidebar .sidebar-content {
        background-color: #1a1a1a;
    }
    
    .sidebar h1, .sidebar h2, .sidebar h3 {
        color: #00FF00;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 200;
        letter-spacing: 2px;
        text-transform: lowercase;
    }
    
    .sidebar label {
        color: #00FF00;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 200;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="logo-header">
    <div class="logo-text">stockly</div>
</div>
""", unsafe_allow_html=True)

conn = get_connection()
try:
    initialize_schema(conn)
    symbols_df = pd.read_sql_query("SELECT ticker FROM symbols ORDER BY ticker", conn)
    tickers = symbols_df["ticker"].tolist() if not symbols_df.empty else []
except Exception as e:
    tickers = []
    st.warning(f"Database issue: {e}")

if not tickers:
    st.info("No tickers found. Creating sample data...")
    with st.spinner("Initializing database and creating sample data..."):
        try:
            from src.database.db_utils import initialize_schema
            initialize_schema(conn)
            import subprocess
            import sys
            result = subprocess.run([sys.executable, "create_sample_data.py"], 
                                  capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent)
            if result.returncode == 0:
                st.success("Sample data created! Please refresh the page.")
                st.rerun()
            else:
                st.error(f"Error creating sample data: {result.stderr}")
        except Exception as e:
            st.error(f"Error: {e}")
    st.stop()

with st.sidebar:
    st.header("controls")
    selected_ticker = st.selectbox("ticker", tickers)
    model_choice = st.selectbox("model", ["Logistic Regression", "Random Forest", "LSTM"])
    
    start_date = st.date_input("start date", value=date(2024, 1, 1))
    end_date = st.date_input("end date", value=date.today())
    
    run_button = st.button("run analysis", type="primary")

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
                st.metric("total return", f"{results['total_return']:.2f}%")
            with col2:
                st.metric("buy & hold", f"{results['buy_hold_return']:.2f}%")
            with col3:
                st.metric("sharpe ratio", f"{results['sharpe_ratio']:.4f}")
            with col4:
                st.metric("max drawdown", f"{results['max_drawdown']:.2f}%")
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("price & signals")
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
                st.subheader("performance")
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

