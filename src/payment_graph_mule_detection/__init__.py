"""
Payment Graph Mule & AML Detection Package.
"""

def main() -> None:
    """Entry point for the payment-graph-mule-detection CLI."""
    from payment_graph_mule_detection.pipeline import run_benchmark
    run_benchmark()
