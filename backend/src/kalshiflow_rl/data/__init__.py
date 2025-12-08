"""
Data ingestion and storage components.

Provides non-blocking orderbook ingestion from Kalshi WebSocket streams,
async write queues for batched database persistence, and data replay
utilities for training environments.
"""