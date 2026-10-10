"""
Unit tests for Graph Feature Extraction.
"""

import pytest
import pandas as pd
from payment_graph_mule_detection.synthetic_mule_graph import SyntheticMuleGraphGenerator
from payment_graph_mule_detection.graph_features import GraphFeatureExtractor


def test_graph_feature_extractor():
    generator = SyntheticMuleGraphGenerator(
        num_normal_accounts=50,
        num_smurfing_rings=1,
        num_cycle_rings=1,
        num_pass_through_mules=2,
        seed=42,
    )
    dataset = generator.generate(total_steps=50)

    extractor = GraphFeatureExtractor()
    features_df = extractor.extract_features(dataset.nodes_df, dataset.edges_df, dataset.graph)

    expected_cols = [
        "account_id",
        "in_degree",
        "out_degree",
        "total_degree",
        "degree_ratio",
        "pass_through_ratio",
        "pagerank",
        "clustering_coefficient",
        "is_mule",
    ]
    for col in expected_cols:
        assert col in features_df.columns, f"Missing feature: {col}"

    assert len(features_df) == len(dataset.nodes_df)
    assert not features_df["pass_through_ratio"].isna().any()
