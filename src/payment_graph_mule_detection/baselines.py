"""
Non-GNN Tabular & Graph-Feature Baselines.

Demonstrates the exact value-add of network topology by comparing:
1. Local Tabular Baseline (Balance and local volume only, no graph metrics).
2. Topological Graph Baseline (LightGBM on Degree, PageRank, Pass-Through ratio, etc.).
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from payment_graph_mule_detection.metrics import compute_mule_metrics, MuleEvaluationResult


class NonGNNBaselineModel:
    """
    Gradient boosted baseline using hand-crafted graph topological metrics.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=31,
            class_weight="balanced",
            random_state=random_state,
            verbose=-1,
        )
        self.feature_names = []

    def fit_and_evaluate(
        self,
        features_df: pd.DataFrame,
        feature_subset: str = "all", # "local_only" or "all"
        test_size: float = 0.25,
    ) -> Tuple[MuleEvaluationResult, np.ndarray, np.ndarray]:
        """
        Splits data, trains LightGBM, and returns evaluation metrics.
        """
        exclude_cols = ["account_id", "is_mule"]
        
        if feature_subset == "local_only":
            candidate_cols = ["initial_balance", "total_volume", "in_amount_sum", "out_amount_sum"]
            cols = [c for c in candidate_cols if c in features_df.columns]
        else:
            cols = [c for c in features_df.columns if c not in exclude_cols]

        self.feature_names = cols
        X = features_df[cols].values
        y = features_df["is_mule"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        self.model.fit(X_train, y_train)
        y_prob = self.model.predict_proba(X_test)[:, 1]

        metrics = compute_mule_metrics(y_test, y_prob)
        return metrics, y_test, y_prob

    def get_feature_importances(self) -> pd.DataFrame:
        """Returns sorted feature importances from the fitted LightGBM tree."""
        if not hasattr(self.model, "feature_importances_"):
            return pd.DataFrame()
        return pd.DataFrame({
            "feature": self.feature_names,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
