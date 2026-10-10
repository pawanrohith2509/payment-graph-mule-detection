"""
Synthetic Banking & Payment Mule Network Graph Generator.

Generates realistic account-to-account payment transaction networks containing:
1. Legitimate peer-to-peer and merchant commerce.
2. Smurfing rings (Fan-Out -> Fan-In layering).
3. Circular washing rings (cycles to obscure fund origins).
4. High-velocity pass-through mule accounts (instant balance evacuation).
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import networkx as nx


@dataclass
class MuleGraphDataset:
    """Container for graph transactions and ground-truth node labels."""
    nodes_df: pd.DataFrame
    edges_df: pd.DataFrame
    graph: nx.DiGraph

    def summary(self) -> Dict[str, float]:
        """Returns key statistical properties of the generated graph."""
        total_nodes = len(self.nodes_df)
        mule_nodes = int(self.nodes_df["is_mule"].sum())
        mule_ratio = mule_nodes / total_nodes if total_nodes > 0 else 0.0
        return {
            "total_accounts": total_nodes,
            "total_transactions": len(self.edges_df),
            "mule_accounts": mule_nodes,
            "mule_prevalence_pct": round(mule_ratio * 100, 2),
            "density": nx.density(self.graph),
            "time_steps_range": (int(self.edges_df["step"].min()), int(self.edges_df["step"].max())),
        }


class SyntheticMuleGraphGenerator:
    """
    Simulates high-fidelity financial transaction networks with explicit mule structures.
    """

    def __init__(
        self,
        num_normal_accounts: int = 1000,
        num_smurfing_rings: int = 15,
        num_cycle_rings: int = 10,
        num_pass_through_mules: int = 25,
        seed: int = 42,
    ):
        self.num_normal = num_normal_accounts
        self.num_smurfing_rings = num_smurfing_rings
        self.num_cycle_rings = num_cycle_rings
        self.num_pass_through_mules = num_pass_through_mules
        self.rng = np.random.default_rng(seed)

    def generate(self, total_steps: int = 720) -> MuleGraphDataset:
        """
        Generates full transaction network over `total_steps` (hours).
        """
        nodes = []
        edges = []

        # 1. Generate Normal Accounts
        for i in range(self.num_normal):
            acc_id = f"ACC_NORM_{i:05d}"
            init_bal = float(self.rng.lognormal(mean=7.5, sigma=1.0))
            nodes.append({
                "account_id": acc_id,
                "account_type": "INDIVIDUAL",
                "initial_balance": init_bal,
                "is_mule": 0,
                "mule_typology": "NONE",
            })

        # Generate normal transactions (scale-free power-law activity)
        num_normal_txs = self.num_normal * 12
        normal_acc_ids = [n["account_id"] for n in nodes]
        
        # Power-law popularity for merchants/recipients
        weights = 1.0 / (np.arange(1, self.num_normal + 1) ** 0.8)
        weights /= weights.sum()

        for _ in range(num_normal_txs):
            src, dst = self.rng.choice(normal_acc_ids, size=2, replace=False, p=weights)
            step = int(self.rng.integers(1, total_steps + 1))
            amount = float(self.rng.exponential(scale=120.0) + 5.0)
            edges.append({
                "source": src,
                "target": dst,
                "amount": round(amount, 2),
                "step": step,
                "tx_type": "P2P_OR_MERCHANT",
                "is_fraud_edge": 0,
            })

        # 2. Generate Smurfing Rings (Fan-Out -> Fan-In)
        # Structure: Source Boss -> M mules (small smurf amounts) -> Aggregator Boss
        for ring_idx in range(self.num_smurfing_rings):
            boss_src = f"ACC_SMURF_SRC_{ring_idx:03d}"
            boss_dst = f"ACC_SMURF_DST_{ring_idx:03d}"
            nodes.append({
                "account_id": boss_src,
                "account_type": "SMURF_ORGANIZER",
                "initial_balance": 500000.0,
                "is_mule": 1,
                "mule_typology": "SMURF_ORGANIZER",
            })
            nodes.append({
                "account_id": boss_dst,
                "account_type": "SMURF_AGGREGATOR",
                "initial_balance": 1000.0,
                "is_mule": 1,
                "mule_typology": "SMURF_AGGREGATOR",
            })

            ring_mule_count = self.rng.integers(4, 9)
            s_min = max(1, int(total_steps * 0.05))
            s_max = max(s_min + 1, int(total_steps * 0.8))
            base_step = int(self.rng.integers(s_min, s_max))
            stolen_sum = float(self.rng.uniform(40000.0, 150000.0))
            smurf_slice = stolen_sum / ring_mule_count

            for m in range(ring_mule_count):
                mule_id = f"ACC_MULE_SMURF_{ring_idx:03d}_{m:02d}"
                nodes.append({
                    "account_id": mule_id,
                    "account_type": "MULE_LAYER",
                    "initial_balance": 50.0,
                    "is_mule": 1,
                    "mule_typology": "SMURFING_MULE",
                })
                # Fan-out: Boss -> Mule
                fan_out_step = base_step + int(self.rng.integers(0, 4))
                edges.append({
                    "source": boss_src,
                    "target": mule_id,
                    "amount": round(smurf_slice * float(self.rng.uniform(0.95, 1.05)), 2),
                    "step": fan_out_step,
                    "tx_type": "SMURF_DISBURSEMENT",
                    "is_fraud_edge": 1,
                })
                # Fan-in: Mule -> Aggregator (rapidly within 1-6 hours)
                fan_in_step = fan_out_step + int(self.rng.integers(1, 6))
                edges.append({
                    "source": mule_id,
                    "target": boss_dst,
                    "amount": round(smurf_slice * float(self.rng.uniform(0.90, 0.98)), 2), # mule cut subtracted
                    "step": fan_in_step,
                    "tx_type": "SMURF_CONSOLIDATION",
                    "is_fraud_edge": 1,
                })

        # 3. Generate Circular Washing Rings (Cycles)
        # Structure: Acc_0 -> Acc_1 -> Acc_2 -> Acc_3 -> Acc_0 (layered to simulate trading/consulting)
        for cycle_idx in range(self.num_cycle_rings):
            cycle_len = self.rng.integers(3, 6)
            cycle_nodes = []
            for c in range(cycle_len):
                cid = f"ACC_CYCLE_{cycle_idx:03d}_{c:02d}"
                nodes.append({
                    "account_id": cid,
                    "account_type": "MULE_CYCLE",
                    "initial_balance": 2000.0,
                    "is_mule": 1,
                    "mule_typology": "CYCLE_WASHING",
                })
                cycle_nodes.append(cid)

            wash_sum = float(self.rng.uniform(25000.0, 90000.0))
            c_min = max(1, int(total_steps * 0.05))
            c_max = max(c_min + 1, int(total_steps * 0.7))
            start_step = int(self.rng.integers(c_min, c_max))

            for c in range(cycle_len):
                src_node = cycle_nodes[c]
                dst_node = cycle_nodes[(c + 1) % cycle_len]
                tx_step = start_step + (c * int(self.rng.integers(2, 6)))
                # slight friction/decay per hop
                wash_sum *= float(self.rng.uniform(0.97, 0.99))
                edges.append({
                    "source": src_node,
                    "target": dst_node,
                    "amount": round(wash_sum, 2),
                    "step": tx_step,
                    "tx_type": "CYCLE_TRANSFER",
                    "is_fraud_edge": 1,
                })

        # 4. Generate Pass-Through Mules (Instant Drain Transit Nodes)
        # Victim -> Pass-Through Mule -> External Cash Out
        for pt_idx in range(self.num_pass_through_mules):
            mule_id = f"ACC_MULE_PASSTHROUGH_{pt_idx:03d}"
            nodes.append({
                "account_id": mule_id,
                "account_type": "MULE_PASSTHROUGH",
                "initial_balance": 10.0,
                "is_mule": 1,
                "mule_typology": "PASS_THROUGH",
            })
            victim = self.rng.choice(normal_acc_ids)
            sink = f"ACC_CASH_OUT_{pt_idx:03d}"
            nodes.append({
                "account_id": sink,
                "account_type": "CASH_OUT_SINK",
                "initial_balance": 0.0,
                "is_mule": 1,
                "mule_typology": "CASH_OUT_SINK",
            })

            drain_amount = float(self.rng.uniform(15000.0, 80000.0))
            p_min = max(1, int(total_steps * 0.05))
            p_max = max(p_min + 1, max(total_steps - 5, p_min + 2))
            drain_step = int(self.rng.integers(p_min, p_max))

            # Inbound from victim
            edges.append({
                "source": victim,
                "target": mule_id,
                "amount": round(drain_amount, 2),
                "step": drain_step,
                "tx_type": "COMPROMISED_DRAIN",
                "is_fraud_edge": 1,
            })
            # Immediate outbound to sink (within 1 hour)
            edges.append({
                "source": mule_id,
                "target": sink,
                "amount": round(drain_amount * 0.96, 2),
                "step": drain_step + 1,
                "tx_type": "RAPID_EVACUATION",
                "is_fraud_edge": 1,
            })

        nodes_df = pd.DataFrame(nodes).drop_duplicates(subset=["account_id"]).reset_index(drop=True)
        edges_df = pd.DataFrame(edges).sort_values("step").reset_index(drop=True)

        # Build NetworkX DiGraph
        G = nx.DiGraph()
        for _, row in nodes_df.iterrows():
            G.add_node(
                row["account_id"],
                account_type=row["account_type"],
                initial_balance=row["initial_balance"],
                is_mule=int(row["is_mule"]),
                mule_typology=row["mule_typology"],
            )

        for _, row in edges_df.iterrows():
            G.add_edge(
                row["source"],
                row["target"],
                amount=row["amount"],
                step=row["step"],
                tx_type=row["tx_type"],
                is_fraud_edge=int(row["is_fraud_edge"]),
            )

        return MuleGraphDataset(nodes_df=nodes_df, edges_df=edges_df, graph=G)
