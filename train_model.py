"""
Orchestrates the full offline pipeline: generate data if needed, clean it,
engineer features, train the anomaly detector and the risk classifier,
evaluate them, and persist everything the Streamlit app needs to load at
start up.

Run this once before launching the dashboard:
    python train_model.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

from data.generate_dataset import generate_dataset
from src.config import (
    ANOMALY_MODEL_PATH,
    METRICS_PATH,
    MODEL_PATH,
    RAW_DATA_PATH,
    SCALER_PATH,
)
from src.data_processing import DataQualityAgent, prepare_features
from src.risk_model import RiskPredictionModel, save_metrics
import pickle


def main():
    print("Step 1: Loading or generating the sensor dataset")
    if not os.path.exists(RAW_DATA_PATH):
        df = generate_dataset()
        df.to_csv(RAW_DATA_PATH, index=False)
        print(f"  Generated {len(df)} new records at {RAW_DATA_PATH}")
    else:
        df = pd.read_csv(RAW_DATA_PATH)
        print(f"  Loaded {len(df)} existing records from {RAW_DATA_PATH}")

    print("Step 2: Cleaning data and engineering features")
    enriched, feature_matrix = prepare_features(df)
    print(f"  Feature matrix shape: {feature_matrix.shape}")

    print("Step 3: Fitting the data quality (anomaly detection) agent")
    quality_agent = DataQualityAgent()
    quality_agent.fit(enriched)
    with open(ANOMALY_MODEL_PATH, "wb") as f:
        pickle.dump(quality_agent.detector, f)
    anomaly_flags = quality_agent.flag_anomalies(enriched)
    print(f"  Flagged {int(anomaly_flags.sum())} anomalous readings out of {len(enriched)}")

    print("Step 4: Training the risk classification model")
    model = RiskPredictionModel()
    metrics = model.train(feature_matrix, enriched["risk_label"])

    print("Step 5: Evaluation results")
    for key in ("accuracy", "precision", "recall", "f1_score", "roc_auc"):
        print(f"  {key}: {metrics[key]}")

    print("Step 6: Saving model artifacts")
    model.save(MODEL_PATH, SCALER_PATH)
    save_metrics(metrics, METRICS_PATH)
    print(f"  Model saved to {MODEL_PATH}")
    print(f"  Scaler saved to {SCALER_PATH}")
    print(f"  Metrics saved to {METRICS_PATH}")

    print("\nTraining complete. Launch the dashboard with: streamlit run app.py")


if __name__ == "__main__":
    main()
