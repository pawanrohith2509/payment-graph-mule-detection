"""
Unit tests for Non-GNN Baseline and PyG GNN Architectures.
"""

import pytest
import torch
from payment_graph_mule_detection.synthetic_mule_graph import SyntheticMuleGraphGenerator
from payment_graph_mule_detection.graph_features import GraphFeatureExtractor
from payment_graph_mule_detection.baselines import NonGNNBaselineModel
from payment_graph_mule_detection.data_converter import convert_to_pyg_data
from payment_graph_mule_detection.models import (
    GCNClassifier,
    GraphSAGEClassifier,
    GATClassifier,
    train_and_evaluate_gnn,
)


@pytest.fixture
def small_graph_data():
    generator = SyntheticMuleGraphGenerator(
        num_normal_accounts=60,
        num_smurfing_rings=2,
        num_cycle_rings=1,
        num_pass_through_mules=2,
        seed=99,
    )
    dataset = generator.generate(total_steps=50)
    extractor = GraphFeatureExtractor()
    features_df = extractor.extract_features(dataset.nodes_df, dataset.edges_df, dataset.graph)
    pyg_data, _, _ = convert_to_pyg_data(dataset.nodes_df, dataset.edges_df, temporal_split_step=40, val_split_step=30)
    return features_df, pyg_data


def test_non_gnn_baseline(small_graph_data):
    features_df, _ = small_graph_data
    baseline = NonGNNBaselineModel(random_state=42)
    metrics, y_test, y_prob = baseline.fit_and_evaluate(features_df, feature_subset="all", test_size=0.3)

    assert 0.0 <= metrics.pr_auc <= 1.0
    assert 0.0 <= metrics.roc_auc <= 1.0
    assert len(y_test) == len(y_prob)


def test_gnn_forward_passes(small_graph_data):
    _, pyg_data = small_graph_data

    # GCN
    gcn = GCNClassifier(in_channels=pyg_data.num_node_features, hidden_channels=16, out_channels=2)
    out_gcn = gcn(pyg_data.x, pyg_data.edge_index)
    assert out_gcn.shape == (pyg_data.num_nodes, 2)

    # GraphSAGE
    sage = GraphSAGEClassifier(in_channels=pyg_data.num_node_features, hidden_channels=16, out_channels=2)
    out_sage = sage(pyg_data.x, pyg_data.edge_index)
    assert out_sage.shape == (pyg_data.num_nodes, 2)

    # GAT
    gat = GATClassifier(in_channels=pyg_data.num_node_features, hidden_channels=8, heads=2, out_channels=2)
    out_gat = gat(pyg_data.x, pyg_data.edge_index)
    assert out_gat.shape == (pyg_data.num_nodes, 2)


def test_gnn_training_loop(small_graph_data):
    _, pyg_data = small_graph_data
    gcn = GCNClassifier(in_channels=pyg_data.num_node_features, hidden_channels=16, out_channels=2)
    results, history = train_and_evaluate_gnn(gcn, pyg_data, epochs=5, lr=0.01)

    assert len(history["train_loss"]) == 5
    assert 0.0 <= results.pr_auc <= 1.0
