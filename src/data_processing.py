"""
Data cleaning and feature engineering for the road risk pipeline.

The functions here are deliberately split into small, testable steps
(clean, engineer, encode) rather than one large function, because the
Streamlit dashboard needs to run the same transformation on a freshly
uploaded CSV that train_model.py runs on the training set. Keeping both
callers pointed at this single module is what actually makes the accuracy
numbers on the dashboard match the ones from training.
"""

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.config import FEATURE_COLUMNS, RANDOM_STATE

RAW_SENSOR_COLUMNS = [
    "rainfall_mm",
    "humidity_percent",
    "traffic_density_vph",
    "road_age_years",
    "vehicle_load_tons",
    "drainage_quality_score",
]


class DataQualityAgent:
    """Flags sensor dropouts and statistically implausible readings.

    This mirrors the first stage of the pipeline described in the project
    report: before any risk score is trusted, the raw feed has to be
    checked for missing values and anomalous spikes (a stuck sensor
    reporting a five times normal traffic count, for instance).
    """

    def __init__(self, contamination: float = 0.015):
        self.contamination = contamination
        self.detector = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=RANDOM_STATE,
        )
        self._is_fitted = False

    def fit(self, df: pd.DataFrame) -> "DataQualityAgent":
        numeric = df[RAW_SENSOR_COLUMNS].fillna(df[RAW_SENSOR_COLUMNS].median())
        self.detector.fit(numeric)
        self._is_fitted = True
        return self

    def flag_anomalies(self, df: pd.DataFrame) -> pd.Series:
        if not self._is_fitted:
            self.fit(df)
        numeric = df[RAW_SENSOR_COLUMNS].fillna(df[RAW_SENSOR_COLUMNS].median())
        predictions = self.detector.predict(numeric)
        return pd.Series(predictions == -1, index=df.index, name="is_anomalous")


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing sensor values with ward level medians.

    A plain global median would blur the difference between, say, a dry
    interior ward and a coastal ward with a naturally higher baseline
    rainfall, so imputation happens within each ward group where possible
    and falls back to the city, then global, median only if a ward has too
    few readings to compute its own.
    """
    df = df.copy()

    for col in RAW_SENSOR_COLUMNS:
        if col not in df.columns:
            continue
        df[col] = df.groupby("ward")[col].transform(lambda s: s.fillna(s.median()))
        df[col] = df.groupby("city")[col].transform(lambda s: s.fillna(s.median()))
        df[col] = df[col].fillna(df[col].median())

    df = df.drop_duplicates(subset=[c for c in df.columns if c != "record_id"])
    df = df.reset_index(drop=True)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the composite engineering indices used by the risk model.

    Each index encodes a real failure mechanism rather than an arbitrary
    product of columns: monsoon intensity captures water damage risk,
    structural fatigue captures long term wear from heavy vehicles on an
    aging surface, and traffic stress captures short term load on the
    surface irrespective of its age.
    """
    df = df.copy()

    df["monsoon_intensity_index"] = (
        df["rainfall_mm"] * (df["humidity_percent"] / 100.0)
    ).round(2)

    df["structural_fatigue_index"] = (
        df["road_age_years"] * (df["vehicle_load_tons"] / 10.0)
    ).round(2)

    df["traffic_stress_index"] = (
        (df["traffic_density_vph"] / 1000.0) * (df["vehicle_load_tons"] / 10.0)
    ).round(2)

    return df


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Full pipeline: clean, engineer, then slice out the model matrix.

    Returns the enriched dataframe (kept for display in the dashboard) and
    the numeric feature matrix (used for model training or inference).
    """
    cleaned = clean_dataset(df)
    enriched = engineer_features(cleaned)
    feature_matrix = enriched[FEATURE_COLUMNS].copy()
    feature_matrix = feature_matrix.replace([np.inf, -np.inf], np.nan)
    feature_matrix = feature_matrix.fillna(feature_matrix.median())
    return enriched, feature_matrix
