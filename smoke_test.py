"""
Runs the full pipeline outside Streamlit to prove every module actually
works together: data generation, cleaning, feature engineering, anomaly
detection, model training and inference, composite scoring, report
generation, and a dry run email dispatch.

Run with: python smoke_test.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

from data.generate_dataset import generate_dataset
from src.data_processing import DataQualityAgent, prepare_features
from src.email_dispatch import MunicipalEmailDispatcher
from src.report_generator import RoadRiskReportAgent
from src.risk_model import CompositeRiskEngine, RiskPredictionModel
from src.config import MODEL_PATH, SCALER_PATH


def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}")
    if not condition:
        raise SystemExit(1)


def main():
    print("Running end to end smoke test\n")

    df = generate_dataset(n_records=600)
    check("Dataset generated", len(df) > 0)
    check("All four risk tiers present", set(df["risk_label"].unique()) == {"Low", "Medium", "High", "Critical"})
    check("Cities include Bangalore, Mumbai, Delhi", {"Bangalore", "Mumbai", "Delhi"}.issubset(set(df["city"])))

    enriched, feature_matrix = prepare_features(df)
    check("No missing values after cleaning", feature_matrix.isna().sum().sum() == 0)

    quality_agent = DataQualityAgent()
    quality_agent.fit(enriched)
    anomaly_flags = quality_agent.flag_anomalies(enriched)
    check("Anomaly detector runs", len(anomaly_flags) == len(enriched))

    check("Trained model artifact exists", os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH))

    model = RiskPredictionModel()
    model.load(MODEL_PATH, SCALER_PATH)
    probabilities = model.predict_proba(feature_matrix)
    check("Model produces probabilities for every record", len(probabilities) == len(feature_matrix))

    engine = CompositeRiskEngine()
    scored = engine.score_batch(enriched, probabilities)
    check("Composite scores are within 0 to 100", scored["composite_risk_score"].between(0, 100).all())
    check("Every record has a risk tier", scored["risk_tier"].notna().all())

    agent = RoadRiskReportAgent(prefer_local_llm=True)
    top_record = scored.sort_values("composite_risk_score", ascending=False).iloc[0].to_dict()
    report = agent.generate_report(top_record)
    check("Report has a non empty subject", len(report.subject) > 0)
    check("Report has a non empty body", len(report.body) > 0)

    dispatcher = MunicipalEmailDispatcher(dry_run=True)
    result = dispatcher.send_report(top_record["city"], top_record["ward"], report.subject, report.body)
    check("Dry run dispatch succeeds", result.success is True and result.dry_run is True)

    print("\nAll checks passed. The pipeline is fully functional.")


if __name__ == "__main__":
    main()
