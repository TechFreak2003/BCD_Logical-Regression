import logging
import os
from typing import Tuple
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    RocCurveDisplay
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def train_logistic_model(X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
    """
    Trains a Logistic Regression model.
    """
    logging.info("Initializing Logistic Regression model...")
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train, y_train)
    logging.info("Model training complete.")
    return model

def save_model(model: LogisticRegression, scaler, model_dir="outputs/model/") -> None:
    """
    Saves the model and scaler to disk.
    """
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "logistic_regression.pkl"))
    if scaler is not None:
        joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
    logging.info("Model and scaler saved.")

def save_metrics(metrics: dict, output_path="outputs/model/model_metrics.json") -> None:
    """
    Saves evaluation metrics as JSON.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=4)
    logging.info("Metrics saved to JSON.")

def save_figure(fig, filename: str, folder: str = "outputs/figures/") -> None:
    """
    Saves a matplotlib figure to the outputs folder.
    """
    os.makedirs(folder, exist_ok=True)
    fig.savefig(os.path.join(folder, filename), bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Figure saved: {filename}")

def evaluate_model(model: LogisticRegression, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """
    Evaluates the trained model, prints metrics, and saves outputs.
    """
    logging.info("Making predictions on test set...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    logging.info("Evaluating model performance...")
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_test, y_proba)

    logging.info(f"Accuracy: {acc:.4f}")
    logging.info(f"Confusion Matrix:\n{cm}")
    logging.info("Classification Report:\n" + classification_report(y_test, y_pred))

    # Save metrics
    metrics = {
        "accuracy": acc,
        "roc_auc": roc_auc,
        "classification_report": report
    }
    save_metrics(metrics)

    # Save confusion matrix plot
    fig_cm, ax = plt.subplots()
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax)
    save_figure(fig_cm, "confusion_matrix.png")

    # Save ROC curve
    fig_roc, ax = plt.subplots()
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    save_figure(fig_roc, "roc_curve.png")
