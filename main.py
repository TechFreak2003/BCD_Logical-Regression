import logging
from src.data_loader import load_train_test_split
from src.preprocess import scale_features
from src.model import train_logistic_model, evaluate_model
from src.utils import save_model, save_metrics, ensure_dir

import os
import json
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def main():
    logging.info("Starting Breast Cancer Diagnosis Pipeline...")

    # === Step 1: Load and Split Data ===
    X_train, X_test, y_train, y_test = load_train_test_split()

    # === Step 2: Preprocess ===
    X_train_scaled = scale_features(X_train)
    X_test_scaled = scale_features(X_test)

    # === Step 3: Train Model ===
    model = train_logistic_model(X_train_scaled, y_train)

    # === Step 4: Evaluate Model ===
    logging.info("Evaluating model and preparing outputs...")
    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # === Step 5: Save Outputs ===
    ensure_dir("outputs/model")
    save_model(model, "outputs/model/logistic_model.pkl")

    metrics = {
        "accuracy": accuracy,
        "confusion_matrix": cm.tolist(),
        "classification_report": report
    }
    save_metrics(metrics, "outputs/model/metrics.json")

    logging.info("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
