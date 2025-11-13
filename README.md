# stockly

A complete stock market prediction and backtesting system with SQL-based data storage, multiple ML models (baseline + LSTM), and a retro pixel-style visualizer.

## 🚀 Live Demo

Try the interactive Streamlit app: **[https://sstockly.streamlit.app/](https://sstockly.streamlit.app/)**

## Technologies

- **Python** - Primary language
- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **scikit-learn** - Baseline models (Logistic Regression, Random Forest)
- **TensorFlow/Keras** - LSTM/GRU deep learning models
- **SQLite** - Relational database for data storage
- **matplotlib** - Visualization with retro pixel theme
- **Streamlit** - Interactive demo app

## Project Structure

```
stock-predictor/
├── data/
│   ├── raw/              # Raw CSV files
│   ├── market.db         # SQLite database
│   └── processed/        # Exported datasets
├── notebooks/
│   ├── 01_exploration.ipynb
│   └── 02_model_prototyping.ipynb
├── src/
│   ├── config.py
│   ├── database/
│   │   ├── schema.sql
│   │   └── db_utils.py
│   ├── data_acquisition/
│   │   ├── fetch_prices_api.py
│   │   └── load_prices_csv.py
│   ├── data_preprocessing/
│   │   ├── clean_prices.py
│   │   ├── calculate_technical_features.py
│   │   └── create_targets.py
│   ├── models/
│   │   ├── build_datasets.py
│   │   ├── sequence_dataset.py
│   │   ├── train_baseline_models.py
│   │   ├── train_lstm.py
│   │   └── time_series_backtest.py
│   ├── visualization/
│   │   ├── style_pixel_theme.py
│   │   ├── plot_price_and_signals.py
│   │   └── plot_performance.py
│   └── app/
│       └── streamlit_app.py
├── models/               # Saved trained models
├── tests/
└── requirements.txt
```

## Database Schema

The SQLite database (`data/market.db`) contains:

- **symbols** - Stock ticker information
- **prices** - Historical OHLCV data
- **features** - Engineered technical indicators
- **targets** - Next-day returns and direction labels
- **predictions** - Model predictions

## Setup & Installation

### 1. Clone Repository

```bash
git clone https://github.com/AliAbouelazm/stock-predictor.git
cd stock-predictor
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Data Acquisition

### Option 1: API (Alpha Vantage)

1. Get API key from [Alpha Vantage](https://www.alphavantage.co/support/#api-key)
2. Set environment variable:
   ```bash
   export ALPHA_VANTAGE_API_KEY="your_key_here"
   ```
3. Fetch data:
   ```bash
   python src/data_acquisition/fetch_prices_api.py
   ```

### Option 2: CSV Files

Place CSV files in `data/raw/` with columns: date, open, high, low, close, adjusted_close, volume

Then load:
```bash
python src/data_acquisition/load_prices_csv.py
```

### Option 3: Sample Data

Create sample data for testing:
```bash
python create_sample_data.py
```

## Pipeline Steps

### 1. Initialize Database

```bash
python -c "from src.database.db_utils import get_connection, initialize_schema; conn = get_connection(); initialize_schema(conn); conn.close()"
```

### 2. Load Prices

Use one of the data acquisition methods above.

### 3. Compute Features

```bash
python src/data_preprocessing/calculate_technical_features.py
```

### 4. Create Targets

```bash
python src/data_preprocessing/create_targets.py
```

### 5. Train Models

**Baseline Models:**
```bash
python src/models/train_baseline_models.py
```

**LSTM Model:**
```bash
python src/models/train_lstm.py
```

### 6. Run Streamlit App

```bash
streamlit run src/app/streamlit_app.py
```

## Models

### Baseline Models

- **Logistic Regression** - Multinomial classification for direction prediction
- **Random Forest** - Ensemble classifier

### Deep Learning

- **LSTM** - Long Short-Term Memory network for sequence-based prediction
  - Uses 30-day lookback window
  - 2-layer LSTM architecture
  - Predicts direction (-1, 0, 1)

## Features

Technical indicators computed:
- Daily returns (1d, 5d)
- Rolling volatility (10d, 20d)
- Simple Moving Averages (10, 20, 50 days)
- RSI (14-period)
- MACD and signal line
- Lagged returns

## Backtesting

The system includes backtesting functionality:
- Time-based train/test splits
- Strategy performance vs buy-and-hold
- Metrics: Total return, Sharpe ratio, Max drawdown

## Visual Style

Charts use a **retro pixel aesthetic**:
- Blocky square markers
- Bold color palette (green/red/yellow/blue on black)
- Thick gridlines
- Step-style plots
- Monospace fonts

## Limitations

- No transaction costs modeled
- Overfitting risk with limited data
- No fundamental data or news sentiment
- Educational purposes only - not financial advice

## License

This project is for educational purposes.

## Author

**AliAbouelazm**

