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

st.set_page_config(page_title="stockly", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=JetBrains+Mono:wght@100;200;300;400&family=Orbitron:wght@400;500;600;700;800;900&display=swap');
    
    * {
        font-family: 'JetBrains Mono', 'Space Mono', monospace;
        font-weight: 200;
        letter-spacing: 1.5px;
    }
    
    .main {
        background-color: #ffffff;
    }
    
    .stApp {
        background-color: #ffffff;
    }
    
    .logo-header {
        background: #ffffff;
        border-bottom: 1px solid #cccccc;
        padding: 1rem 0;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .logo-text {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 300;
        font-size: 2rem;
        color: #000000;
        letter-spacing: 4px;
        text-transform: lowercase;
        margin: 0;
    }
    
    h1, h2, h3, h4 {
        color: #000000;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 200;
        letter-spacing: 2px;
        text-transform: lowercase;
    }
    
    h1 {
        font-size: 2rem;
        letter-spacing: 3px;
    }
    
    h2 {
        font-size: 1.5rem;
        letter-spacing: 2px;
    }
    
    h3, h4 {
        font-size: 1.2rem;
        letter-spacing: 1.5px;
    }
    
    .stButton>button {
        background-color: #000000;
        color: #ffffff;
        border: 2px solid #000000;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 200;
        padding: 0.4rem 1.2rem;
        border-radius: 0;
        letter-spacing: 2px;
        font-size: 0.9rem;
    }
    
    .stButton>button:hover {
        background-color: #333333;
        border-color: #333333;
    }
    
    .stSelectbox label, .stDateInput label {
        color: #000000;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 200;
        letter-spacing: 1px;
        font-size: 0.85rem;
    }
    
    .stSelectbox>div>div {
        background-color: #ffffff;
        border: 1px solid #cccccc;
        color: #000000;
    }
    
    .stSelectbox>div>div>div {
        color: #000000;
    }
    
    .stSelectbox input {
        color: #000000;
    }
    
    .stDateInput>div>div {
        background-color: #ffffff;
        border: 1px solid #cccccc;
        color: #000000;
    }
    
    .stDateInput input {
        color: #000000;
    }
    
    .stDateInput>div>div>div {
        color: #000000;
    }
    
    .stMetric {
        background-color: #f5f5f5;
        border: 1px solid #cccccc;
        padding: 1rem;
    }
    
    .stMetric label {
        color: #000000;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 200;
        letter-spacing: 1px;
        font-size: 0.75rem;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        color: #000000;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 200;
        letter-spacing: 1px;
        font-size: 1.5rem;
    }
    
    body, p, div, span {
        font-weight: 200;
        letter-spacing: 1px;
    }
    
    .stAlert {
        color: #000000;
    }
    
    .stAlert [data-baseweb="notification"] {
        color: #000000;
        background-color: #f5f5f5;
        border: 1px solid #cccccc;
    }
    
    .element-container {
        color: #000000;
    }
    
    .stSelectbox [data-baseweb="select"] {
        color: #000000 !important;
    }
    
    .stSelectbox [data-baseweb="select"] > div {
        color: #000000 !important;
    }
    
    [data-baseweb="select"] {
        color: #000000 !important;
    }
    
    [data-baseweb="select"] input {
        color: #000000 !important;
    }
    
    [data-baseweb="select"] > div {
        color: #000000 !important;
    }
    
    .stSelectbox [data-baseweb="popover"] {
        color: #000000 !important;
    }
    
    .stSelectbox [data-baseweb="popover"] > div {
        color: #000000 !important;
    }
    
    .stDateInput [data-baseweb="input"] {
        color: #000000 !important;
    }
    
    .stDateInput [data-baseweb="input"] input {
        color: #000000 !important;
    }
    
    .stAlert p {
        color: #000000 !important;
        font-weight: 400 !important;
    }
    
    .stAlert [data-baseweb="notification"] p {
        color: #000000 !important;
        font-weight: 400 !important;
    }
    
    .stWarning {
        color: #000000 !important;
    }
    
    .stError {
        color: #000000 !important;
    }
    
    .stInfo {
        color: #000000 !important;
    }
    
    .sidebar .sidebar-content {
        background-color: #ffffff;
    }
    
    .sidebar h1, .sidebar h2, .sidebar h3 {
        color: #000000;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 200;
        letter-spacing: 2px;
        text-transform: lowercase;
    }
    
    .sidebar label {
        color: #000000;
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

st.write("**App Version:** v3.0 - Enhanced Debug Mode")

from src.config import DB_PATH
import os

db_path_str = str(DB_PATH.resolve())
st.write(f"**Database location:** `{db_path_str}`")
st.write(f"**Database exists:** {os.path.exists(db_path_str)}")

if os.path.exists(db_path_str):
    db_size = os.path.getsize(db_path_str)
    st.write(f"**Database size:** {db_size:,} bytes")

conn = get_connection()
try:
    initialize_schema(conn)
    symbols_df = pd.read_sql_query("SELECT ticker FROM symbols ORDER BY ticker", conn)
    tickers = symbols_df["ticker"].tolist() if not symbols_df.empty else []
    st.write(f"**Tickers in database:** {', '.join(tickers) if tickers else 'None'}")
except Exception as e:
    tickers = []
    st.error(f"Database issue: {e}")
    import traceback
    st.code(traceback.format_exc())

has_predictions = False
try:
    total_preds = pd.read_sql_query("SELECT COUNT(*) as cnt FROM predictions", conn)
    pred_count = total_preds.iloc[0]['cnt']
    st.write(f"**Total predictions in DB:** {pred_count}")
    
    if pred_count > 0:
        predictions_check = pd.read_sql_query("""
            SELECT DISTINCT s.ticker, p.model_name 
            FROM predictions p
            JOIN symbols s ON p.symbol_id = s.id
        """, conn)
        has_predictions = not predictions_check.empty
        if has_predictions:
            tickers_found = sorted(predictions_check['ticker'].unique())
            models_found = sorted(predictions_check['model_name'].unique())
            st.success(f"✅ Found predictions for tickers: {', '.join(tickers_found)}")
            st.info(f"Available models: {', '.join(models_found)}")
        else:
            st.warning("Query returned empty - checking symbols table...")
            symbols_check = pd.read_sql_query("SELECT COUNT(*) as cnt FROM symbols", conn)
            st.write(f"Symbols in DB: {symbols_check.iloc[0]['cnt']}")
    else:
        st.warning("Predictions table is empty!")
except Exception as e:
    has_predictions = False
    import traceback
    st.error(f"Error checking predictions: {e}")
    st.code(traceback.format_exc())
finally:
    if conn:
        conn.close()

if not tickers:
    st.info("No tickers found. Creating sample data...")
    with st.spinner("Initializing database and creating sample data..."):
        try:
            from src.database.db_utils import initialize_schema, get_connection
            conn2 = get_connection()
            initialize_schema(conn2)
            import subprocess
            import sys
            result = subprocess.run(
                [sys.executable, "create_sample_data.py"], 
                capture_output=True, 
                text=True, 
                cwd=Path(__file__).parent.parent.parent,
                timeout=120
            )
            if result.returncode == 0:
                st.success("Sample data created! Please refresh the page.")
                st.rerun()
            else:
                st.error(f"Error creating sample data: {result.stderr}")
                st.code(result.stdout)
            conn2.close()
        except subprocess.TimeoutExpired:
            st.error("Sample data creation timed out. Please try again.")
        except Exception as e:
            st.error(f"Error: {e}")
            import traceback
            st.code(traceback.format_exc())
    st.stop()

if not has_predictions and tickers:
    st.warning("⚠️ **No predictions found in database.**")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 Generate Predictions Now", type="primary", use_container_width=True):
            with st.spinner("Training models and generating predictions... This may take a few minutes."):
                try:
                    import subprocess
                    import sys
                    from pathlib import Path
                    import os
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    steps = [
                        ("Creating sample data...", ["create_sample_data.py"]),
                        ("Training baseline models...", ["src/models/train_baseline_models.py"]),
                        ("Training LSTM model...", ["src/models/train_lstm.py"]),
                        ("Generating predictions...", ["src/models/generate_predictions.py"])
                    ]
                    
                    project_root = Path(__file__).parent.parent.parent
                    env = os.environ.copy()
                    env['PYTHONPATH'] = str(project_root)
                    
                    for i, (step_name, script) in enumerate(steps):
                        status_text.text(f"Step {i+1}/{len(steps)}: {step_name}")
                        progress_bar.progress((i + 1) / len(steps))
                        
                        result = subprocess.run(
                            [sys.executable] + script,
                            capture_output=True,
                            text=True,
                            cwd=project_root,
                            timeout=600,
                            env=env
                        )
                        if result.returncode != 0:
                            st.error(f"Error in {step_name}")
                            st.code(result.stderr)
                            st.code(result.stdout)
                            st.stop()
                        else:
                            st.success(f"✅ {step_name} completed")
                    
                    progress_bar.progress(1.0)
                    status_text.text("Complete!")
                    st.success("🎉 All predictions generated! Refreshing...")
                    st.rerun()
                    
                except subprocess.TimeoutExpired:
                    st.error("Process timed out. Please try again.")
                except Exception as e:
                    st.error(f"Error: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    
    st.markdown("""
    **Or run manually in terminal:**
    ```bash
    python create_sample_data.py
    python src/models/train_baseline_models.py
    python src/models/train_lstm.py
    python src/models/generate_predictions.py
    ```
    """)

col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    selected_ticker = st.selectbox("ticker", tickers, key="ticker_select")

with col2:
    date_range = st.date_input("date range", value=(date(2022, 7, 31), date(2022, 8, 11)), key="date_range")

with col3:
    st.write("")
    st.write("")
    run_button = st.button("analyze", type="primary", use_container_width=True)

if run_button and len(date_range) == 2:
    start_date, end_date = date_range[0], date_range[1]
    model_name = "lstm_model"
    
    st.write(f"**Testing:** {selected_ticker} from {start_date} to {end_date} with {model_name}")
    
    try:
        with st.spinner("running analysis..."):
            results = backtest_model(
                selected_ticker,
                model_name,
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d")
            )
            
            if not results:
                st.error("No results returned from backtest_model")
                st.info("Checking database directly...")
                conn_debug = get_connection()
                try:
                    debug_df = pd.read_sql_query("""
                        SELECT COUNT(*) as cnt FROM predictions p
                        JOIN symbols s ON p.symbol_id = s.id
                        WHERE s.ticker = ? AND p.model_name = ?
                        AND p.date >= ? AND p.date <= ?
                    """, conn_debug, params=[selected_ticker, model_name, 
                                             start_date.strftime("%Y-%m-%d"), 
                                             end_date.strftime("%Y-%m-%d")])
                    st.write(f"Found {debug_df.iloc[0]['cnt']} predictions in database for this query")
                except Exception as e2:
                    st.error(f"Debug query error: {e2}")
                finally:
                    conn_debug.close()
        
        if results:
            metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
            
            with metrics_col1:
                st.metric("total return", f"{results['total_return']:.2f}%")
            with metrics_col2:
                st.metric("buy & hold", f"{results['buy_hold_return']:.2f}%")
            with metrics_col3:
                st.metric("sharpe ratio", f"{results['sharpe_ratio']:.4f}")
            with metrics_col4:
                st.metric("max drawdown", f"{results['max_drawdown']:.2f}%")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            dates = pd.to_datetime(results["dates"])
            prices_df = pd.read_sql_query("""
                SELECT date, adjusted_close FROM prices p
                JOIN symbols s ON p.symbol_id = s.id
                WHERE s.ticker = ? AND date >= ? AND date <= ?
                ORDER BY date
            """, conn, params=[selected_ticker, start_date, end_date])
            
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                st.markdown("#### price & signals")
                fig1, ax1 = plt.subplots(figsize=(10, 6))
                ax1.set_facecolor("#ffffff")
                fig1.patch.set_facecolor("#ffffff")
                
                if not prices_df.empty:
                    prices_df["date"] = pd.to_datetime(prices_df["date"])
                    ax1.plot(prices_df["date"], prices_df["adjusted_close"], 
                            color="#000000", linewidth=2, marker="s", markersize=3)
                    ax1.set_facecolor("#ffffff")
                    ax1.tick_params(colors="#000000")
                    ax1.set_xlabel("date", color="#000000", fontfamily="JetBrains Mono")
                    ax1.set_ylabel("price", color="#000000", fontfamily="JetBrains Mono")
                    ax1.set_title(f"{selected_ticker}", color="#000000", fontfamily="JetBrains Mono", fontweight=200)
                    ax1.grid(True, color="#cccccc", linestyle="-", linewidth=0.5)
                    plt.tight_layout()
                    st.pyplot(fig1)
            
            with chart_col2:
                st.markdown("#### performance")
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                ax2.set_facecolor("#ffffff")
                fig2.patch.set_facecolor("#ffffff")
                
                ax2.plot(dates, results["portfolio_value"], 
                        color="#000000", linewidth=2, marker="s", markersize=3, label="strategy")
                ax2.plot(dates, results["buy_hold_value"], 
                        color="#666666", linewidth=2, marker="s", markersize=3, label="buy & hold")
                ax2.set_facecolor("#ffffff")
                ax2.tick_params(colors="#000000")
                ax2.set_xlabel("date", color="#000000", fontfamily="JetBrains Mono")
                ax2.set_ylabel("portfolio value", color="#000000", fontfamily="JetBrains Mono")
                ax2.set_title("cumulative returns", color="#000000", fontfamily="JetBrains Mono", fontweight=200)
                ax2.legend(facecolor="#ffffff", edgecolor="#cccccc", prop={"family": "JetBrains Mono"})
                ax2.grid(True, color="#cccccc", linestyle="-", linewidth=0.5)
                plt.tight_layout()
                st.pyplot(fig2)
        else:
            st.warning("no results found for this ticker and date range")
            st.info("""
            **Possible reasons:**
            - Predictions don't exist for this ticker (run `python src/models/generate_predictions.py`)
            - Date range doesn't match available predictions
            - Model 'lstm_model' not found in predictions table
            
            Check available predictions:
            """)
            conn_check2 = get_connection()
            try:
                available = pd.read_sql_query("""
                    SELECT DISTINCT s.ticker, p.model_name, 
                           MIN(p.date) as min_date, MAX(p.date) as max_date
                    FROM predictions p
                    JOIN symbols s ON p.symbol_id = s.id
                    GROUP BY s.ticker, p.model_name
                """, conn_check2)
                if not available.empty:
                    st.dataframe(available, use_container_width=True)
                else:
                    st.write("No predictions found in database.")
            except Exception as e2:
                st.write(f"Error checking predictions: {e2}")
            finally:
                conn_check2.close()
    
    except Exception as e:
        st.error(f"error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        st.info("ensure models are trained and predictions are generated")

conn.close()

