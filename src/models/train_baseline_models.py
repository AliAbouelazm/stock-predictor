"""Train baseline models (Logistic Regression, RandomForest)."""

import logging
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pathlib import Path

from src.models.build_datasets import build_tabular_dataset
from src.config import MODELS_DIR, RANDOM_SEED

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> LogisticRegression:
    """Train Logistic Regression model."""
    logger.info("Training Logistic Regression...")
    
    model = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_SEED,
        multi_class="multinomial",
        solver="lbfgs"
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    logger.info(f"Logistic Regression Accuracy: {accuracy:.4f}")
    logger.info("\nClassification Report:\n" + classification_report(y_test, y_pred))
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / "logistic_regression.pkl")
    logger.info(f"Model saved to {MODELS_DIR / 'logistic_regression.pkl'}")
    
    return model


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> RandomForestClassifier:
    """Train Random Forest model."""
    logger.info("Training Random Forest...")
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    logger.info(f"Random Forest Accuracy: {accuracy:.4f}")
    logger.info("\nClassification Report:\n" + classification_report(y_test, y_pred))
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / "random_forest.pkl")
    logger.info(f"Model saved to {MODELS_DIR / 'random_forest.pkl'}")
    
    return model


if __name__ == "__main__":
    from src.config import TRAIN_END_DATE
    
    X_train, y_train, X_test, y_test = build_tabular_dataset(train_split_date=TRAIN_END_DATE)
    
    train_logistic_regression(X_train, y_train, X_test, y_test)
    train_random_forest(X_train, y_train, X_test, y_test)

