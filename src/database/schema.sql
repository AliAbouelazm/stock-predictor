CREATE TABLE IF NOT EXISTS symbols (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT UNIQUE NOT NULL,
    name TEXT
);

CREATE TABLE IF NOT EXISTS prices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol_id INTEGER NOT NULL,
    date DATE NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    adjusted_close REAL,
    volume REAL,
    FOREIGN KEY (symbol_id) REFERENCES symbols(id),
    UNIQUE(symbol_id, date)
);

CREATE TABLE IF NOT EXISTS features (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol_id INTEGER NOT NULL,
    date DATE NOT NULL,
    return_1d REAL,
    return_5d REAL,
    volatility_10d REAL,
    volatility_20d REAL,
    sma_10 REAL,
    sma_20 REAL,
    sma_50 REAL,
    rsi_14 REAL,
    macd REAL,
    macd_signal REAL,
    macd_histogram REAL,
    lag_return_1 REAL,
    lag_return_2 REAL,
    lag_return_5 REAL,
    FOREIGN KEY (symbol_id) REFERENCES symbols(id),
    UNIQUE(symbol_id, date)
);

CREATE TABLE IF NOT EXISTS targets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol_id INTEGER NOT NULL,
    date DATE NOT NULL,
    next_day_return REAL,
    direction_label INTEGER NOT NULL,
    FOREIGN KEY (symbol_id) REFERENCES symbols(id),
    UNIQUE(symbol_id, date)
);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol_id INTEGER NOT NULL,
    date DATE NOT NULL,
    model_name TEXT NOT NULL,
    predicted_direction INTEGER,
    predicted_return REAL,
    prob_up REAL,
    prob_flat REAL,
    prob_down REAL,
    FOREIGN KEY (symbol_id) REFERENCES symbols(id),
    UNIQUE(symbol_id, date, model_name)
);

CREATE INDEX IF NOT EXISTS idx_prices_symbol_date ON prices(symbol_id, date);
CREATE INDEX IF NOT EXISTS idx_features_symbol_date ON features(symbol_id, date);
CREATE INDEX IF NOT EXISTS idx_targets_symbol_date ON targets(symbol_id, date);
CREATE INDEX IF NOT EXISTS idx_predictions_symbol_date ON predictions(symbol_id, date);

