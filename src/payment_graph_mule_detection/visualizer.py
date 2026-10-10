"""
Mule Subgraph Visualizer & Ring Detection Plots.

Extracts multi-hop ego-networks around flagged accounts and renders visual graphs
highlighting money-laundering typologies (Smurfing, Circular Washing, Pass-Through).
"""

from typing import List, Optional
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


class MuleSubgraphVisualizer:
    """
    Renders visual diagrams of suspicious account clusters and laundering rings.
    """

    def __init__(self, G: nx.DiGraph):
        self.G = G

    def extract_ego_subgraph(self, target_account: str, radius: int = 2) -> nx.DiGraph:
        """
        Extracts directed k-hop ego subgraph around the given account.
        """
        nodes = set([target_account])
        current_layer = set([target_account])

        for _ in range(radius):
            next_layer = set()
            for node in current_layer:
                if node in self.G:
                    next_layer.update(self.G.predecessors(node))
                    next_layer.update(self.G.successors(node))
            nodes.update(next_layer)
            current_layer = next_layer

        return self.G.subgraph(nodes).copy()

    def plot_subgraph(
        self,
        subgraph: nx.DiGraph,
        title: str,
        save_path: Optional[str] = None,
        figsize: tuple = (10, 8),
    ):
        """
        Plots the subgraph with color coding for mules vs legitimate accounts.
        """
        plt.figure(figsize=figsize)
        pos = nx.spring_layout(subgraph, seed=42, k=0.5)

        # Node colors
        node_colors = []
        for n in subgraph.nodes():
            is_mule = subgraph.nodes[n].get("is_mule", 0)
            node_colors.append("#e74c3c" if is_mule == 1 else "#3498db") # Red for mule, blue for normal

        # Draw nodes & labels
        nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors, node_size=800, alpha=0.9)
        labels = {n: n.replace("ACC_", "") for n in subgraph.nodes()}
        nx.draw_networkx_labels(subgraph, pos, labels=labels, font_size=8, font_family="sans-serif")

        # Draw directed edges
        nx.draw_networkx_edges(
            subgraph,
            pos,
            edge_color="#7f8c8d",
            arrows=True,
            arrowsize=15,
            arrowstyle="-|>",
            connectionstyle="arc3,rad=0.1",
            alpha=0.7,
        )

        plt.title(title, fontsize=12, fontweight="bold", pad=15)
        plt.axis("off")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()
