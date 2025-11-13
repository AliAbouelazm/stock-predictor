"""Train LSTM/GRU model for time series prediction."""

import logging
import numpy as np
from pathlib import Path
from typing import Tuple
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.models.sequence_dataset import build_sequence_dataset
from src.config import MODELS_DIR, LSTM_LOOKBACK_WINDOW, LSTM_BATCH_SIZE, LSTM_EPOCHS, LSTM_HIDDEN_UNITS, RANDOM_SEED

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

tf.random.set_seed(RANDOM_SEED)


def build_lstm_model(input_shape: Tuple[int, int], num_classes: int = 3) -> keras.Model:
    """
    Build LSTM model for direction prediction.
    
    Args:
        input_shape: (lookback, num_features)
        num_classes: Number of classes (3 for -1, 0, 1)
    
    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        layers.LSTM(LSTM_HIDDEN_UNITS, return_sequences=True, input_shape=input_shape),
        layers.Dropout(0.2),
        layers.LSTM(LSTM_HIDDEN_UNITS // 2, return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dense(num_classes, activation="softmax")
    ])
    
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model


def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray
) -> keras.Model:
    """Train LSTM model."""
    logger.info("Training LSTM model...")
    logger.info(f"Input shape: {X_train.shape}")
    
    y_train_shifted = y_train + 1
    y_val_shifted = y_val + 1
    
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train,
        y_train_shifted,
        batch_size=LSTM_BATCH_SIZE,
        epochs=LSTM_EPOCHS,
        validation_data=(X_val, y_val_shifted),
        callbacks=[early_stopping],
        verbose=1
    )
    
    val_pred = model.predict(X_val)
    val_pred_classes = np.argmax(val_pred, axis=1) - 1
    
    from sklearn.metrics import accuracy_score, classification_report
    accuracy = accuracy_score(y_val, val_pred_classes)
    logger.info(f"LSTM Validation Accuracy: {accuracy:.4f}")
    logger.info("\nClassification Report:\n" + classification_report(y_val, val_pred_classes))
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODELS_DIR / "lstm_model.h5")
    logger.info(f"Model saved to {MODELS_DIR / 'lstm_model.h5'}")
    
    return model


if __name__ == "__main__":
    from src.config import TRAIN_END_DATE
    
    X_train, y_train, X_test, y_test = build_sequence_dataset(train_split_date=TRAIN_END_DATE)
    train_lstm(X_train, y_train, X_test, y_test)

