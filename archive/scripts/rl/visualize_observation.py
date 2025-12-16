#!/usr/bin/env python3
"""
Visualize the observation vector from session data.

This script loads a session and builds the actual 47-feature observation vector
that would be fed to the RL model, showing exactly what features are extracted.

Usage:
    # Visualize observation from most recent session
    python visualize_observation.py
    
    # Visualize observation from specific session
    python visualize_observation.py 6
    
    # Show specific timestep
    python visualize_observation.py 6 --step 10
"""

import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime
import argparse
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kalshiflow_rl.environments.session_data_loader import SessionDataLoader
from kalshiflow_rl.environments.feature_extractors import (
    build_observation_from_session_data,
    extract_market_agnostic_features,
    extract_temporal_features,
    extract_portfolio_features
)


def format_feature_value(value: float, feature_name: str) -> str:
    """Format a feature value for display with appropriate precision."""
    if 'norm' in feature_name or 'ratio' in feature_name or 'score' in feature_name:
        # Normalized values [0,1]
        return f"{value:7.4f}"
    elif 'bias' in feature_name or 'imbalance' in feature_name or 'momentum' in feature_name:
        # Values in [-1,1]
        return f"{value:+7.4f}"
    elif 'indicator' in feature_name:
        # Binary values
        return f"{value:7.1f}"
    else:
        return f"{value:7.4f}"


async def visualize_observation(session_id: int = None, timestep: int = 0):
    """Load session and visualize observation vector."""
    
    print("=" * 80)
    print("OBSERVATION VECTOR VISUALIZATION")
    print("=" * 80)
    
    # Get database URL
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("❌ DATABASE_URL not set in environment")
        return None
    
    # Load session
    loader = SessionDataLoader(database_url=database_url)
    
    if session_id is None:
        # Find most recent session
        import asyncpg
        conn = await asyncpg.connect(database_url)
        try:
            row = await conn.fetchrow("""
                SELECT session_id FROM rl_orderbook_sessions 
                WHERE status = 'closed' 
                ORDER BY session_id DESC LIMIT 1
            """)
            if row:
                session_id = row['session_id']
            else:
                print("❌ No closed sessions found!")
                return None
        finally:
            await conn.close()
    
    print(f"\nLoading session {session_id}...")
    session_data = await loader.load_session(session_id)
    
    if not session_data:
        print(f"❌ Failed to load session {session_id}")
        return None
    
    print(f"✅ Session loaded: {session_data.get_episode_length()} timesteps, {len(session_data.markets_involved)} markets")
    
    # Get data point for specified timestep
    if timestep >= session_data.get_episode_length():
        timestep = session_data.get_episode_length() - 1
        print(f"⚠️  Timestep adjusted to {timestep} (last available)")
    
    current_point = session_data.get_timestep_data(timestep)
    if not current_point:
        print(f"❌ No data at timestep {timestep}")
        return None
    
    # Get historical context (last 10 points)
    historical = []
    for i in range(max(0, timestep - 10), timestep):
        point = session_data.get_timestep_data(i)
        if point:
            historical.append(point)
    
    print(f"\n📊 Building observation for timestep {timestep}:")
    print(f"   Timestamp: {current_point.timestamp}")
    print(f"   Markets: {list(current_point.markets_data.keys())[:3]}{'...' if len(current_point.markets_data) > 3 else ''}")
    
    # Simulate portfolio state (empty for demonstration)
    portfolio_state = {}
    portfolio_value = 10000.0
    cash_balance = 10000.0
    
    # Build the actual observation vector
    observation = build_observation_from_session_data(
        current_point, 
        historical, 
        portfolio_state,
        portfolio_value,
        cash_balance,
        max_markets=1  # Single-market architecture
    )
    
    print(f"\n✅ Observation vector built: shape={observation.shape}")
    
    # Display features in organized groups
    print("\n" + "=" * 80)
    print("FEATURE BREAKDOWN (47 total features)")
    print("=" * 80)
    
    feature_idx = 0
    
    # Extract the most active market for display
    markets_by_volume = sorted(
        current_point.markets_data.items(),
        key=lambda x: x[1].get('total_volume', 0),
        reverse=True
    )
    
    if markets_by_volume:
        market_ticker, market_data = markets_by_volume[0]
        
        print(f"\n1️⃣  MARKET FEATURES (21) - {market_ticker}")
        print("-" * 50)
        
        market_features = extract_market_agnostic_features(market_data)
        feature_names = [
            "best_yes_bid_norm", "best_yes_ask_norm", "best_no_bid_norm", "best_no_ask_norm",
            "yes_spread_norm", "no_spread_norm", "yes_mid_price_norm", "no_mid_price_norm",
            "yes_volume_norm", "no_volume_norm", "total_volume_norm",
            "volume_imbalance", "yes_side_imbalance", "no_side_imbalance",
            "yes_book_depth_norm", "no_book_depth_norm",
            "yes_liquidity_concentration", "no_liquidity_concentration",
            "arbitrage_opportunity", "market_efficiency", "cross_side_efficiency"
        ]
        
        for i, name in enumerate(feature_names):
            if i < len(observation):
                value = observation[feature_idx + i]
                formatted = format_feature_value(value, name)
                print(f"   [{feature_idx + i:2d}] {name:30s} = {formatted}")
        
        feature_idx += 21
    
    print(f"\n2️⃣  TEMPORAL FEATURES (14)")
    print("-" * 50)
    
    temporal_features = extract_temporal_features(current_point, historical)
    temporal_names = list(temporal_features.keys())
    
    for i, name in enumerate(temporal_names):
        if feature_idx + i < len(observation):
            value = observation[feature_idx + i]
            formatted = format_feature_value(value, name)
            print(f"   [{feature_idx + i:2d}] {name:30s} = {formatted}")
    
    feature_idx += 14
    
    print(f"\n3️⃣  PORTFOLIO FEATURES (12)")
    print("-" * 50)
    
    # Extract current prices for portfolio features
    current_prices = {}
    for ticker, mkt_data in current_point.markets_data.items():
        yes_bids = mkt_data.get('yes_bids', {})
        yes_asks = mkt_data.get('yes_asks', {})
        if yes_bids and yes_asks:
            best_bid = max(map(int, yes_bids.keys()))
            best_ask = min(map(int, yes_asks.keys()))
            mid_price = (best_bid + best_ask) / 2.0 / 100.0
            current_prices[ticker] = mid_price
    
    portfolio_features = extract_portfolio_features(
        portfolio_state, portfolio_value, cash_balance, current_prices
    )
    portfolio_names = list(portfolio_features.keys())
    
    for i, name in enumerate(portfolio_names):
        if feature_idx + i < len(observation):
            value = observation[feature_idx + i]
            formatted = format_feature_value(value, name)
            print(f"   [{feature_idx + i:2d}] {name:30s} = {formatted}")
    
    print("\n" + "=" * 80)
    print("OBSERVATION STATISTICS")
    print("=" * 80)
    
    print(f"\n📈 Value Distribution:")
    print(f"   Min:     {observation.min():+.4f}")
    print(f"   Max:     {observation.max():+.4f}")
    print(f"   Mean:    {observation.mean():+.4f}")
    print(f"   Std:     {observation.std():.4f}")
    
    # Check for any unusual values
    nan_count = np.isnan(observation).sum()
    inf_count = np.isinf(observation).sum()
    zero_count = (observation == 0).sum()
    
    print(f"\n🔍 Data Quality:")
    print(f"   NaN values:  {nan_count}")
    print(f"   Inf values:  {inf_count}")
    print(f"   Zero values: {zero_count} ({zero_count/len(observation)*100:.1f}%)")
    
    # Show histogram of values
    print(f"\n📊 Value Histogram:")
    hist, bins = np.histogram(observation, bins=10)
    for i in range(len(hist)):
        bar_width = int(hist[i] * 30 / max(hist))
        bar = '█' * bar_width
        print(f"   [{bins[i]:+.2f}, {bins[i+1]:+.2f}]: {bar} ({hist[i]} features)")
    
    print("\n" + "=" * 80)
    print("✅ This is the exact 47-feature vector the RL model would receive!")
    print("=" * 80)
    
    return observation


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Visualize observation vector from session data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize_observation.py          # Use most recent session, step 0
  python visualize_observation.py 6        # Use session 6, step 0
  python visualize_observation.py 6 --step 50  # Use session 6, step 50
        """
    )
    
    parser.add_argument('session_id', type=int, nargs='?',
                       help='Session ID to load (uses latest if not provided)')
    parser.add_argument('--step', '-s', type=int, default=0,
                       help='Timestep to visualize (default: 0)')
    
    args = parser.parse_args()
    
    try:
        observation = await visualize_observation(args.session_id, args.step)
        
        if observation is not None:
            print(f"\n✨ Successfully visualized {len(observation)}-dimensional observation vector!")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())