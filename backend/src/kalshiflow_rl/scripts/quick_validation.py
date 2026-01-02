#!/usr/bin/env python3
"""
Rapid Independent Strategy Validation
=====================================

This script provides rapid validation of the key trading strategy claims
using simplified analysis to quickly assess their validity.
"""

import asyncio
import os
import sys
import numpy as np
from typing import List

# Add the src directory to path for imports
sys.path.append('/Users/samuelclark/Desktop/kalshiflow/backend/src')

from kalshiflow_rl.environments.session_data_loader import SessionDataLoader

async def quick_validate():
    """Quickly validate the main claims using session statistics"""
    print("RAPID INDEPENDENT STRATEGY VALIDATION")
    print("====================================")
    
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("❌ DATABASE_URL not set")
        return
        
    # Load just session 9 and 32 for quick analysis
    test_sessions = [9, 32]  # High quality sessions
    
    loader = SessionDataLoader(database_url=database_url)
    
    total_data_points = 0
    total_markets = 0
    spreads_data = []
    
    for session_id in test_sessions:
        print(f"\nLoading session {session_id}...")
        try:
            session_data = await loader.load_session(session_id)
            if session_data:
                total_data_points += len(session_data.data_points)
                total_markets += len(session_data.markets_involved)
                spreads_data.append(session_data.avg_spread)
                print(f"  ✅ Session {session_id}: {len(session_data.data_points)} data points, {len(session_data.markets_involved)} markets")
                print(f"     Avg spread: {session_data.avg_spread:.1f}, Volatility: {session_data.volatility_score:.3f}")
                
                # Sample the first few data points to understand structure
                sample_size = min(5, len(session_data.data_points))
                print(f"     Sampling {sample_size} data points for anchor analysis...")
                
                anchor_samples = analyze_anchor_samples(session_data.data_points[:sample_size])
                print(f"     Found {anchor_samples['near_25']} points near 25¢, {anchor_samples['near_50']} near 50¢, {anchor_samples['near_75']} near 75¢")
                
            else:
                print(f"  ❌ Failed to load session {session_id}")
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print(f"\n" + "="*60)
    print("RAPID VALIDATION RESULTS")
    print("="*60)
    
    print(f"\nData Quality Assessment:")
    print(f"  Total data points analyzed: {total_data_points:,}")
    print(f"  Total unique markets: {total_markets}")
    print(f"  Average spread across sessions: {np.mean(spreads_data):.1f} cents")
    
    # Assess key claims
    print(f"\nClaim Assessment:")
    
    # Claim 1: Anchor behavior exists
    if total_data_points > 10000:
        print("  ✅ SUFFICIENT DATA for anchor analysis")
    else:
        print("  ❌ INSUFFICIENT DATA for reliable analysis")
    
    # Claim 2: Spread patterns 
    avg_spread = np.mean(spreads_data) if spreads_data else 0
    if 10 <= avg_spread <= 40:
        print(f"  ✅ SPREADS IN EXPECTED RANGE ({avg_spread:.1f}¢) for algo trading")
    else:
        print(f"  ⚠️  SPREADS UNUSUAL ({avg_spread:.1f}¢) - may indicate different market conditions")
    
    # Overall assessment
    print(f"\nOverall Assessment:")
    if total_data_points > 10000 and 10 <= avg_spread <= 40:
        print("  ✅ DATA SUPPORTS POSSIBILITY of anchor strategies")
        print("  📋 RECOMMENDATION: Proceed with detailed validation")
    else:
        print("  ❌ DATA DOES NOT SUPPORT anchor strategy claims")
        print("  📋 RECOMMENDATION: Claims likely invalid or data quality issues")

def analyze_anchor_samples(data_points) -> dict:
    """Quick analysis of a few data points for anchor proximity"""
    results = {'near_25': 0, 'near_50': 0, 'near_75': 0}
    
    for data_point in data_points:
        for market_ticker, market_data in data_point.markets_data.items():
            # Try to extract any price information
            for key, value in market_data.items():
                if 'price' in key.lower() and isinstance(value, (int, float)):
                    price = float(value)
                    # Check proximity to anchors (within 2 cents)
                    if 23 <= price <= 27:
                        results['near_25'] += 1
                    elif 48 <= price <= 52:
                        results['near_50'] += 1
                    elif 73 <= price <= 77:
                        results['near_75'] += 1
    
    return results

async def main():
    await quick_validate()

if __name__ == "__main__":
    asyncio.run(main())