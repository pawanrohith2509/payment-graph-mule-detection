"""
Converter from Tabular / NetworkX Graph to PyTorch Geometric Data Object.
"""

from typing import Tuple, List, Dict
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler


def convert_to_pyg_data(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    temporal_split_step: int = 500,
    val_split_step: int = 400,
) -> Tuple[Data, StandardScaler, Dict[str, int]]:
    """
    Constructs a PyG Data object with strict temporal masks.
    """
    # 1. Map account_ids to contiguous 0-indexed integers
    acc_list = nodes_df["account_id"].tolist()
    node_to_idx = {acc: i for i, acc in enumerate(acc_list)}
    num_nodes = len(acc_list)

    # 2. Extract Node Features (excluding IDs and targets)
    # If initial_balance is available
    features_list = []
    for _, row in nodes_df.iterrows():
        init_bal = float(row.get("initial_balance", 0.0))
        features_list.append([init_bal, np.log1p(max(init_bal, 0.0))])

    scaler = StandardScaler()
    X = scaler.fit_transform(np.array(features_list, dtype=np.float32))
    x_tensor = torch.tensor(X, dtype=torch.float)

    # 3. Construct Edge Index & Edge Attributes
    src_indices = [node_to_idx[s] for s in edges_df["source"] if s in node_to_idx and edges_df.loc[edges_df["source"] == s, "target"].iloc[0] in node_to_idx]
    # More robust vectorized edge mapping:
    valid_mask = edges_df["source"].isin(node_to_idx) & edges_df["target"].isin(node_to_idx)
    valid_edges = edges_df[valid_mask].copy()

    sources = valid_edges["source"].map(node_to_idx).values
    targets = valid_edges["target"].map(node_to_idx).values
    edge_index = torch.tensor(np.vstack([sources, targets]), dtype=torch.long)

    # Edge features: [log_amount, normalized_step]
    amounts = np.log1p(valid_edges["amount"].values.astype(np.float32))
    steps = valid_edges["step"].values.astype(np.float32) / 720.0
    edge_attr = torch.tensor(np.column_stack([amounts, steps]), dtype=torch.float)

    # 4. Target Labels
    y_tensor = torch.tensor(nodes_df["is_mule"].values, dtype=torch.long)

    # 5. Strict Temporal Splits
    # A node belongs to Train if all its transactions occurred before val_split_step
    # Belongs to Val if its transactions occurred between val_split_step and temporal_split_step
    # Belongs to Test if its transactions occurred after temporal_split_step
    # Let's derive node last active step
    in_steps = edges_df.groupby("target")["step"].max()
    out_steps = edges_df.groupby("source")["step"].max()
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    for acc, idx in node_to_idx.items():
        s_in = in_steps.get(acc, 0)
        s_out = out_steps.get(acc, 0)
        max_step = max(s_in, s_out)

        if max_step <= val_split_step:
            train_mask[idx] = True
        elif max_step <= temporal_split_step:
            val_mask[idx] = True
        else:
            test_mask[idx] = True

    # Fallback in case some nodes have no transactions
    unassigned = ~(train_mask | val_mask | test_mask)
    train_mask[unassigned] = True

    data = Data(
        x=x_tensor,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y_tensor,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )
    return data, scaler, node_to_idx
