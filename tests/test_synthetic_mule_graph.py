"""
Unit tests for Synthetic Mule Graph Generator.
"""

import pytest
from payment_graph_mule_detection.synthetic_mule_graph import SyntheticMuleGraphGenerator


def test_synthetic_mule_generator():
    generator = SyntheticMuleGraphGenerator(
        num_normal_accounts=100,
        num_smurfing_rings=2,
        num_cycle_rings=2,
        num_pass_through_mules=4,
        seed=123,
    )
    dataset = generator.generate(total_steps=100)

    # Assert non-empty and consistent node/edge structure
    assert len(dataset.nodes_df) > 100
    assert len(dataset.edges_df) > 0
    assert "account_id" in dataset.nodes_df.columns
    assert "is_mule" in dataset.nodes_df.columns
    assert "source" in dataset.edges_df.columns
    assert "target" in dataset.edges_df.columns
    assert "amount" in dataset.edges_df.columns

    # Verify both normal and mule accounts exist
    mule_count = dataset.nodes_df["is_mule"].sum()
    assert mule_count > 0
    assert mule_count < len(dataset.nodes_df)

    # Verify NetworkX graph consistency (parallel transactions coalesce into directed edge attributes on DiGraph)
    assert dataset.graph.number_of_nodes() == len(dataset.nodes_df)
    assert 0 < dataset.graph.number_of_edges() <= len(dataset.edges_df)
