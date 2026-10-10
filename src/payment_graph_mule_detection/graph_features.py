"""
Graph Feature Engineering for Banking & Mule Detection.

Extracts domain-informed network topology features from directed transaction graphs
for training tabular machine learning baselines (e.g. LightGBM).
"""

from typing import Dict, List, Optional
import networkx as nx
import numpy as np
import pandas as pd


class GraphFeatureExtractor:
    """
    Extracts topological and flow metrics from a directed transaction graph.
    """

    def __init__(self, damping_factor: float = 0.85):
        self.damping_factor = damping_factor

    def extract_features(
        self,
        nodes_df: pd.DataFrame,
        edges_df: pd.DataFrame,
        G: Optional[nx.DiGraph] = None,
    ) -> pd.DataFrame:
        """
        Computes node-level graph features combining edge flows and network topology.
        """
        if G is None:
            G = nx.DiGraph()
            for _, row in nodes_df.iterrows():
                G.add_node(row["account_id"])
            for _, row in edges_df.iterrows():
                G.add_edge(row["source"], row["target"], amount=float(row["amount"]), step=int(row["step"]))

        # 1. Degree Metrics
        in_degrees = dict(G.in_degree())
        out_degrees = dict(G.out_degree())

        # 2. PageRank
        try:
            pagerank = nx.pagerank(G, alpha=self.damping_factor, max_iter=200)
        except Exception:
            pagerank = {node: 1.0 / len(G) for node in G.nodes()}

        # 3. In/Out Degree Centrality
        in_centrality = nx.in_degree_centrality(G)
        out_centrality = nx.out_degree_centrality(G)

        # 4. Clustering coefficient (computed on undirected projection)
        G_undirected = G.to_undirected()
        clustering = nx.clustering(G_undirected)

        # 5. Volume and Flow calculations from edges_df
        in_flow = edges_df.groupby("target")["amount"].agg(["sum", "mean", "count"]).rename(
            columns={"sum": "in_amount_sum", "mean": "in_amount_mean", "count": "in_tx_count"}
        )
        out_flow = edges_df.groupby("source")["amount"].agg(["sum", "mean", "count"]).rename(
            columns={"sum": "out_amount_sum", "mean": "out_amount_mean", "count": "out_tx_count"}
        )

        # First and last transaction timestamps per account
        first_in = edges_df.groupby("target")["step"].min()
        last_in = edges_df.groupby("target")["step"].max()
        first_out = edges_df.groupby("source")["step"].min()
        last_out = edges_df.groupby("source")["step"].max()

        features: List[Dict[str, float]] = []

        for _, row in nodes_df.iterrows():
            acc_id = row["account_id"]

            in_deg = in_degrees.get(acc_id, 0)
            out_deg = out_degrees.get(acc_id, 0)
            tot_deg = in_deg + out_deg

            # Inbound monetary flow
            in_amt_sum = float(in_flow.loc[acc_id, "in_amount_sum"]) if acc_id in in_flow.index else 0.0
            in_amt_mean = float(in_flow.loc[acc_id, "in_amount_mean"]) if acc_id in in_flow.index else 0.0

            # Outbound monetary flow
            out_amt_sum = float(out_flow.loc[acc_id, "out_amount_sum"]) if acc_id in out_flow.index else 0.0
            out_amt_mean = float(out_flow.loc[acc_id, "out_amount_mean"]) if acc_id in out_flow.index else 0.0

            tot_vol = in_amt_sum + out_amt_sum
            net_flow = in_amt_sum - out_amt_sum

            # Pass-through ratio: min(in, out) / max(in, out)
            max_amt = max(in_amt_sum, out_amt_sum)
            min_amt = min(in_amt_sum, out_amt_sum)
            pass_through_ratio = (min_amt / (max_amt + 1e-5)) if max_amt > 0 else 0.0

            # Degree ratio: (in + 1) / (out + 1)
            degree_ratio = (in_deg + 1.0) / (out_deg + 1.0)

            # Active transaction time span
            steps_seen = []
            if acc_id in first_in:
                steps_seen.extend([first_in[acc_id], last_in[acc_id]])
            if acc_id in first_out:
                steps_seen.extend([first_out[acc_id], last_out[acc_id]])
            
            if steps_seen:
                tx_span = max(steps_seen) - min(steps_seen)
                activity_velocity = tot_deg / max(tx_span, 1.0)
            else:
                tx_span = 0
                activity_velocity = 0.0

            # Initial balance
            init_bal = float(row.get("initial_balance", 0.0))

            features.append({
                "account_id": acc_id,
                "initial_balance": init_bal,
                "in_degree": in_deg,
                "out_degree": out_deg,
                "total_degree": tot_deg,
                "degree_ratio": degree_ratio,
                "in_amount_sum": in_amt_sum,
                "in_amount_mean": in_amt_mean,
                "out_amount_sum": out_amt_sum,
                "out_amount_mean": out_amt_mean,
                "total_volume": tot_vol,
                "net_flow": net_flow,
                "pass_through_ratio": pass_through_ratio,
                "pagerank": pagerank.get(acc_id, 0.0),
                "in_degree_centrality": in_centrality.get(acc_id, 0.0),
                "out_degree_centrality": out_centrality.get(acc_id, 0.0),
                "clustering_coefficient": clustering.get(acc_id, 0.0),
                "tx_span_hours": tx_span,
                "activity_velocity": activity_velocity,
                "is_mule": int(row.get("is_mule", 0)),
            })

        return pd.DataFrame(features)
