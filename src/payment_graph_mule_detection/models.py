"""
Graph Neural Network Architectures for Mule & Financial Fraud Detection.

Implements GCN, GraphSAGE, and GAT models with class-weighted loss handling.
"""

from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from payment_graph_mule_detection.metrics import compute_mule_metrics, MuleEvaluationResult


class GCNClassifier(nn.Module):
    """2-Layer Graph Convolutional Network."""

    def __init__(self, in_channels: int, hidden_channels: int = 32, out_channels: int = 2, dropout: float = 0.2):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class GraphSAGEClassifier(nn.Module):
    """2-Layer Inductive GraphSAGE Network."""

    def __init__(self, in_channels: int, hidden_channels: int = 32, out_channels: int = 2, dropout: float = 0.2):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr="mean")
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels, aggr="mean")
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class GATClassifier(nn.Module):
    """Graph Attention Network with Multi-Head Attention."""

    def __init__(self, in_channels: int, hidden_channels: int = 16, heads: int = 4, out_channels: int = 2, dropout: float = 0.2):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.bn1 = nn.BatchNorm1d(hidden_channels * heads)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def train_and_evaluate_gnn(
    model: nn.Module,
    data: Data,
    epochs: int = 100,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
) -> Tuple[MuleEvaluationResult, Dict[str, list]]:
    """
    Trains GNN model on data.train_mask with class-balanced cross-entropy loss
    and evaluates on data.test_mask.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Class imbalance weighting
    y_train = data.y[data.train_mask]
    n_pos = float((y_train == 1).sum())
    n_neg = float((y_train == 0).sum())
    pos_weight = (n_neg / (n_pos + 1e-5)) if n_pos > 0 else 1.0

    class_weights = torch.tensor([1.0, pos_weight], dtype=torch.float)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    history = {"train_loss": [], "val_f1": []}

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        history["train_loss"].append(float(loss.item()))

    # Final Evaluation on Test Mask
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
        test_y = data.y[data.test_mask].cpu().numpy()
        test_probs = probs[data.test_mask.cpu().numpy()]

    results = compute_mule_metrics(test_y, test_probs)
    return results, history
