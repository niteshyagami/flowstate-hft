"""FlowState HFT — Avellaneda-Stoikov market-making research stack.

Package layout
--------------
``ingestion``  Async Hyperliquid L2 WebSocket client and normalized book model.
``features``   Micro-price, order flow imbalance, realized volatility.
``strategy``   Avellaneda-Stoikov reservation price and optimal spread.
``execution``  Paper-fill simulator with queue-position modelling and analytics.

This package is simulation-only. No module authenticates against, signs for, or
transmits orders to any venue.
"""

__version__ = "0.1.0"
__all__ = ["ingestion", "features", "strategy", "execution"]
