#!/usr/bin/env python3
"""Test portfolio value calculation logic."""

import asyncio
import json
import sys
sys.path.append('src')

from kalshiflow_rl.trading.services.position_tracker import PositionTracker, Position

async def test_portfolio_calculation():
    print('🧪 Testing Portfolio Value Calculation...')
    
    # Create position tracker
    tracker = PositionTracker(initial_cash_balance=5000.0)
    
    # Add some test positions
    tracker.positions['TEST-MARKET-1'] = Position(
        ticker='TEST-MARKET-1',
        contracts=100,  # 100 YES contracts
        cost_basis=65.0,  # Bought at 65 cents = $65 total
        realized_pnl=0.0
    )
    
    tracker.positions['TEST-MARKET-2'] = Position(
        ticker='TEST-MARKET-2',
        contracts=-50,  # 50 NO contracts
        cost_basis=20.0,  # Bought at 40 cents (NO = 1 - 0.60 YES) = $20 total
        realized_pnl=5.0  # Made $5 on previous trades
    )
    
    # Calculate portfolio with market prices
    market_prices = {
        'TEST-MARKET-1': {'bid': 70, 'ask': 72},  # YES at 70/72 cents
        'TEST-MARKET-2': {'bid': 55, 'ask': 58}   # YES at 55/58 cents (NO = 45/42 cents)
    }
    
    summary = tracker.get_portfolio_summary(market_prices)
    
    print('\n📊 Portfolio Summary:')
    print(f'  Cash Balance: ${summary["cash_balance"]:.2f}')
    print(f'  Portfolio Value: ${summary["portfolio_value"]:.2f} (market value of positions)')
    print(f'  Total Value: ${summary["total_value"]:.2f} (cash + positions)')
    print(f'  Realized P&L: ${summary["realized_pnl"]:.2f}')
    print(f'  Unrealized P&L: ${summary["unrealized_pnl"]:.2f}')
    print(f'  Total P&L: ${summary["total_pnl"]:.2f}')
    
    # Verify calculations
    print('\n✅ Verification:')
    # Position 1: 100 YES contracts at 70 cents bid = $70 market value
    # Position 2: 50 NO contracts at (1 - 0.58) = 42 cents NO bid = $21 market value
    expected_portfolio = 70.0 + 21.0  # $91 total
    print(f'  Expected portfolio value: ${expected_portfolio:.2f}')
    print(f'  Actual portfolio value: ${summary["portfolio_value"]:.2f}')
    print(f'  Match: {abs(summary["portfolio_value"] - expected_portfolio) < 0.01}')
    
    # Total value should be cash + portfolio
    expected_total = 5000.0 + expected_portfolio
    print(f'\n  Expected total value: ${expected_total:.2f}')
    print(f'  Actual total value: ${summary["total_value"]:.2f}')
    print(f'  Match: {abs(summary["total_value"] - expected_total) < 0.01}')
    
    print('\n✅ Portfolio calculation test complete!')
    return summary["portfolio_value"] == expected_portfolio and summary["total_value"] == expected_total

if __name__ == '__main__':
    success = asyncio.run(test_portfolio_calculation())
    sys.exit(0 if success else 1)