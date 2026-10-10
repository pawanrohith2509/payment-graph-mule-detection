"""
End-to-End Benchmark Execution Pipeline.
"""

from pathlib import Path
import pandas as pd
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
from payment_graph_mule_detection.visualizer import MuleSubgraphVisualizer


def run_benchmark():
    """Executes the complete graph mule detection experiment."""
    print("=" * 70)
    print("       PAYMENT GRAPH MULE & AML DETECTION BENCHMARK SUITE       ")
    print("=" * 70)

    output_dir = Path("data/processed")
    reports_dir = Path("reports/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate Graph
    print("\n[Step 1/6] Generating Synthetic Banking Transaction Network...")
    generator = SyntheticMuleGraphGenerator(
        num_normal_accounts=1200,
        num_smurfing_rings=15,
        num_cycle_rings=10,
        num_pass_through_mules=25,
        seed=42,
    )
    dataset = generator.generate(total_steps=720)
    summary = dataset.summary()
    print(f"  * Total Accounts: {summary['total_accounts']}")
    print(f"  * Total Transactions: {summary['total_transactions']}")
    print(f"  * Mule Accounts: {summary['mule_accounts']} ({summary['mule_prevalence_pct']}% prevalence)")

    # 2. Extract Graph Features
    print("\n[Step 2/6] Extracting Topological Network Features...")
    extractor = GraphFeatureExtractor()
    features_df = extractor.extract_features(dataset.nodes_df, dataset.edges_df, dataset.graph)
    print(f"  * Feature Matrix Shape: {features_df.shape}")

    # 3. Non-GNN Baselines
    print("\n[Step 3/6] Training Non-GNN Tabular & Graph Baselines...")
    baseline = NonGNNBaselineModel(random_state=42)

    # Baseline 1: Local Tabular Only (No Graph Features)
    local_metrics, _, _ = baseline.fit_and_evaluate(features_df, feature_subset="local_only")
    print("  [OK] Model 1: Local Tabular Baseline (No Graph Topology)")
    print(f"    - PR-AUC: {local_metrics.pr_auc:.4f} | ROC-AUC: {local_metrics.roc_auc:.4f} | F1: {local_metrics.best_f1:.4f}")

    # Baseline 2: Topological Graph Features (LightGBM on Degree, PageRank, Pass-Through)
    topo_metrics, _, _ = baseline.fit_and_evaluate(features_df, feature_subset="all")
    print("  [OK] Model 2: Topological Graph Baseline (LightGBM + Network Features)")
    print(f"    - PR-AUC: {topo_metrics.pr_auc:.4f} | ROC-AUC: {topo_metrics.roc_auc:.4f} | F1: {topo_metrics.best_f1:.4f}")

    # 4. Convert to PyTorch Geometric Data
    print("\n[Step 4/6] Preparing Graph for PyTorch Geometric...")
    pyg_data, scaler, node_to_idx = convert_to_pyg_data(
        dataset.nodes_df,
        dataset.edges_df,
        temporal_split_step=500,
        val_split_step=400,
    )
    print(f"  * PyG Nodes: {pyg_data.num_nodes}, Edges: {pyg_data.num_edges}")
    print(f"  * Train Nodes: {int(pyg_data.train_mask.sum())}, Test Nodes: {int(pyg_data.test_mask.sum())}")

    # 5. Deep Graph Neural Networks
    print("\n[Step 5/6] Training Deep GNN Architectures (PyG)...")
    torch.manual_seed(42)

    # Model 3: GCN
    gcn_model = GCNClassifier(in_channels=pyg_data.num_node_features, hidden_channels=32, out_channels=2)
    gcn_metrics, _ = train_and_evaluate_gnn(gcn_model, pyg_data, epochs=80, lr=0.01)
    print("  [OK] Model 3: Graph Convolutional Network (GCN)")
    print(f"    - PR-AUC: {gcn_metrics.pr_auc:.4f} | ROC-AUC: {gcn_metrics.roc_auc:.4f} | F1: {gcn_metrics.best_f1:.4f}")

    # Model 4: GraphSAGE
    sage_model = GraphSAGEClassifier(in_channels=pyg_data.num_node_features, hidden_channels=32, out_channels=2)
    sage_metrics, _ = train_and_evaluate_gnn(sage_model, pyg_data, epochs=80, lr=0.01)
    print("  [OK] Model 4: GraphSAGE (Mean Aggregator)")
    print(f"    - PR-AUC: {sage_metrics.pr_auc:.4f} | ROC-AUC: {sage_metrics.roc_auc:.4f} | F1: {sage_metrics.best_f1:.4f}")

    # Model 5: GAT
    gat_model = GATClassifier(in_channels=pyg_data.num_node_features, hidden_channels=16, heads=4, out_channels=2)
    gat_metrics, _ = train_and_evaluate_gnn(gat_model, pyg_data, epochs=80, lr=0.01)
    print("  [OK] Model 5: Graph Attention Network (GAT)")
    print(f"    - PR-AUC: {gat_metrics.pr_auc:.4f} | ROC-AUC: {gat_metrics.roc_auc:.4f} | F1: {gat_metrics.best_f1:.4f}")

    # 6. Benchmark Comparison Table
    print("\n[Step 6/6] Generating Visualizations and Final Comparison...")
    results = [
        {"Model": "Local Tabular Only (LightGBM)", **local_metrics.to_dict()},
        {"Model": "Topological Graph Features (LightGBM)", **topo_metrics.to_dict()},
        {"Model": "Graph Convolutional Network (GCN)", **gcn_metrics.to_dict()},
        {"Model": "GraphSAGE (Inductive)", **sage_metrics.to_dict()},
        {"Model": "Graph Attention Network (GAT)", **gat_metrics.to_dict()},
    ]
    results_df = pd.DataFrame(results)
    results_csv_path = output_dir / "benchmark_results.csv"
    results_df.to_csv(results_csv_path, index=False)
    print(f"  * Saved benchmark table to: {results_csv_path}")

    # Visualizing a Smurfing Ring
    visualizer = MuleSubgraphVisualizer(dataset.graph)
    smurf_boss = "ACC_SMURF_SRC_000"
    if smurf_boss in dataset.graph:
        subgraph = visualizer.extract_ego_subgraph(smurf_boss, radius=2)
        fig_path = reports_dir / "detected_smurfing_ring.png"
        visualizer.plot_subgraph(
            subgraph,
            title="Detected Laundering Pattern: Smurfing Ring (Fan-Out -> Fan-In)",
            save_path=str(fig_path),
        )
        print(f"  * Saved detected smurfing ring visualization to: {fig_path}")

    print("\n" + "=" * 70)
    print("                      BENCHMARK COMPARISON TABLE                      ")
    print("=" * 70)
    print(results_df[["Model", "pr_auc", "roc_auc", "best_f1", "recall_at_90_precision"]].to_string(index=False))
    print("=" * 70)
    return results_df
