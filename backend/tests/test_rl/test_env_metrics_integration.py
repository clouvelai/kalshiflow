"""
Simple integration tests to verify KalshiTradingEnv and TradingMetricsCalculator work together.

These tests verify the essential functionality without complex scenarios.
"""

import pytest
from kalshiflow_rl.environments.kalshi_env import KalshiTradingEnv
from kalshiflow_rl.trading.trading_metrics import TradingMetricsCalculator


def test_env_uses_metrics_calculator():
    """Test that KalshiTradingEnv properly integrates with TradingMetricsCalculator."""
    env = KalshiTradingEnv(market_tickers=['TEST'])
    
    # Verify env has a metrics calculator
    assert hasattr(env, 'metrics_calculator')
    assert isinstance(env.metrics_calculator, TradingMetricsCalculator)
    
    # Verify the positions property delegates to metrics calculator
    assert hasattr(env, 'positions')
    env_positions = env.positions
    calc_positions = env.metrics_calculator.get_positions_dict()
    assert env_positions == calc_positions
    
    # Verify cash balance property delegates to metrics calculator
    assert env.cash_balance == env.metrics_calculator.cash_balance


def test_fee_calculation_consistency():
    """Test that fee calculations are consistent between env config and metrics calculator."""
    # Create env with specific fee rate
    custom_fee_rate = 0.005  # 0.5%
    env = KalshiTradingEnv(
        market_tickers=['TEST'],
        reward_config={'trading_fee_rate': custom_fee_rate}
    )
    
    # Verify the fee rate was passed to metrics calculator
    assert env.metrics_calculator.reward_config['trading_fee_rate'] == custom_fee_rate
    
    # Test fee calculation
    trade_value = 100.0
    expected_fee = trade_value * custom_fee_rate
    actual_fee = env.metrics_calculator.calculate_trade_fee(trade_value)
    assert abs(actual_fee - expected_fee) < 0.001


def test_metrics_calculator_tracks_trades():
    """Test that trades executed through metrics calculator are properly tracked."""
    env = KalshiTradingEnv(market_tickers=['TEST'])
    
    # Reset to clear any initial state
    env.reset()
    
    # Execute a trade directly through metrics calculator (simulating what env.step() does)
    trade_result = env.metrics_calculator.execute_trade(
        market_ticker='TEST',
        side='yes',
        direction='buy',
        quantity=10,
        price_cents=50
    )
    
    # Verify trade was tracked
    assert env.metrics_calculator.total_trades == 1
    assert trade_result is not None
    assert 'fee' in trade_result
    assert 'trade_value' in trade_result
    
    # Verify position was updated
    positions = env.metrics_calculator.get_positions_dict()
    assert 'TEST' in positions
    assert positions['TEST']['position_yes'] == 10
    assert positions['TEST']['avg_cost_yes'] == 0.50
    
    # Verify cash was reduced
    initial_cash = env.metrics_calculator.initial_cash
    expected_cash = initial_cash - trade_result['trade_value'] - trade_result['fee']
    assert abs(env.metrics_calculator.cash_balance - expected_cash) < 0.01