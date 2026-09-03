import json
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize

from src.config import (
    FEATURE_COLUMNS,
    MODEL_PATH,
    RANDOM_STATE,
    RISK_LABELS,
    SCALER_PATH,
    TRAIN_TEST_SPLIT,
)


class RiskPredictionModel:
    """Random forest classifier over the four tier risk label."""

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=350,
            max_depth=14,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self.classes_: List[str] = RISK_LABELS

    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TRAIN_TEST_SPLIT,
            random_state=RANDOM_STATE,
            stratify=y,
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.model.fit(X_train_scaled, y_train)
        predictions = self.model.predict(X_test_scaled)
        probabilities = self.model.predict_proba(X_test_scaled)

        y_test_binarized = label_binarize(y_test, classes=self.model.classes_)
        try:
            roc_auc = roc_auc_score(
                y_test_binarized, probabilities, average="macro", multi_class="ovr"
            )
        except ValueError:
            roc_auc = float("nan")

        metrics = {
            "accuracy": round(float(accuracy_score(y_test, predictions)) * 100, 2),
            "precision": round(
                float(precision_score(y_test, predictions, average="macro", zero_division=0)) * 100, 2
            ),
            "recall": round(
                float(recall_score(y_test, predictions, average="macro", zero_division=0)) * 100, 2
            ),
            "f1_score": round(
                float(f1_score(y_test, predictions, average="macro", zero_division=0)) * 100, 2
            ),
            "roc_auc": round(float(roc_auc), 4) if not np.isnan(roc_auc) else None,
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test)),
        }

        importances = dict(zip(FEATURE_COLUMNS, self.model.feature_importances_.round(4)))
        metrics["feature_importance"] = {
            k: float(v) for k, v in sorted(importances.items(), key=lambda kv: kv[1], reverse=True)
        }

        return metrics

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """R
        """
        X_scaled = self.scaler.transform(X[FEATURE_COLUMNS])
        probabilities = self.model.predict_proba(X_scaled)
        return pd.DataFrame(probabilities, columns=self.model.classes_, index=X.index)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict risk tier labels for input samples.

        Args:
            X: DataFrame containing feature columns defined in FEATURE_COLUMNS.

        Returns:
            Array of predicted risk tier labels.
        """
        X_scaled = self.scaler.transform(X[FEATURE_COLUMNS])
        return self.model.predict(X_scaled)

    def save(self, model_path: str = MODEL_PATH, scaler_path: str = SCALER_PATH) -> None:
        """Persist the trained model and scaler to disk.

        Args:
            model_path: Path to save the trained RandomForest model.
            scaler_path: Path to save the fitted StandardScaler.
        """
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)
        with open(scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)

    def load(self, model_path: str = MODEL_PATH, scaler_path: str = SCALER_PATH) -> "RiskPredictionModel":
        """Load a trained model and scaler from disk.

        Args:
            model_path: Path to the saved RandomForest model.
            scaler_path: Path to the saved StandardScaler.

        Returns:
            Self with loaded model and scaler for method chaining.
        """
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
        self.classes_ = list(self.model.classes_)
        return self


@dataclass
class RiskAssessment:
    composite_score: float
    tier: str
    ml_component: float
    engineering_component: float
    top_factors: List[Tuple[str, float]] = field(default_factory=list)


class CompositeRiskEngine:
    """Blends ML class probabilities with explainable engineering terms.

    ml_weight controls how much the trained classifier influences the
    final 0 to 100 score versus the transparent rule based term. Keeping
    both halves visible (see explain_row) is what lets the report
    generator justify a score instead of just stating it.
    """

    TIER_ORDER = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}

    FACTOR_WEIGHTS = {
        "rainfall_mm": ("Heavy rainfall exposure", 0.24, 120.0),
        "humidity_percent": ("Sustained high humidity", 0.08, 100.0),
        "traffic_density_vph": ("High traffic density", 0.14, 3000.0),
        "road_age_years": ("Advanced road age", 0.18, 30.0),
        "vehicle_load_tons": ("Heavy vehicle loading", 0.14, 40.0),
        "drainage_quality_score": ("Poor drainage quality", 0.10, 100.0),
        "monsoon_intensity_index": ("Combined rain and humidity stress", 0.06, 60.0),
        "structural_fatigue_index": ("Structural fatigue buildup", 0.06, 40.0),
    }

    def __init__(self, ml_weight: float = 0.6):
        self.ml_weight = ml_weight

    def _engineering_score(self, row: pd.Series) -> Tuple[float, List[Tuple[str, float]]]:
        contributions = []
        total = 0.0
        for column, (label, weight, scale) in self.FACTOR_WEIGHTS.items():
            if column not in row:
                continue
            value = row[column]
            if column == "drainage_quality_score":
                normalized = max(0.0, 1 - (value / scale))
            else:
                normalized = min(1.5, value / scale)
            contribution = weight * normalized
            total += contribution
            contributions.append((label, contribution))

        contributions.sort(key=lambda item: item[1], reverse=True)
        score_0_100 = float(np.clip(total * 100, 0, 100))
        return score_0_100, contributions

    def score_row(self, row: pd.Series, ml_probabilities: pd.Series) -> RiskAssessment:
        tier_score_map = {"Low": 10, "Medium": 40, "High": 70, "Critical": 95}
        ml_component = float(sum(ml_probabilities.get(tier, 0.0) * score for tier, score in tier_score_map.items()))

        engineering_component, contributions = self._engineering_score(row)

        composite = self.ml_weight * ml_component + (1 - self.ml_weight) * engineering_component
        composite = float(np.clip(composite, 0, 100))

        if composite < 30:
            tier = "Low"
        elif composite < 50:
            tier = "Medium"
        elif composite < 72:
            tier = "High"
        else:
            tier = "Critical"

        return RiskAssessment(
            composite_score=round(composite, 1),
            tier=tier,
            ml_component=round(ml_component, 1),
            engineering_component=round(engineering_component, 1),
            top_factors=contributions[:3],
        )

    def score_batch(self, df: pd.DataFrame, probabilities: pd.DataFrame) -> pd.DataFrame:
        results = []
        for idx in df.index:
            assessment = self.score_row(df.loc[idx], probabilities.loc[idx])
            results.append(
                {
                    "composite_risk_score": assessment.composite_score,
                    "risk_tier": assessment.tier,
                    "ml_component": assessment.ml_component,
                    "engineering_component": assessment.engineering_component,
                    "top_factors": "; ".join(f"{name} ({value * 100:.1f})" for name, value in assessment.top_factors),
                }
            )
        result_df = pd.DataFrame(results, index=df.index)
        return pd.concat([df, result_df], axis=1)


def save_metrics(metrics: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def load_metrics(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)
