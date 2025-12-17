#!/usr/bin/env python3
"""
Verification script for RL action space configuration.

Tests the current configuration and identifies inconsistencies between
expected 5-action setup and actual implementation.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_position_config():
    """Test PositionConfig settings."""
    from kalshiflow_rl.environments.limit_order_action_space import PositionConfig
    
    print("=== Testing PositionConfig ===")
    config = PositionConfig()
    print(f"Default position sizes: {config.sizes}")
    print(f"Expected action count: {1 + 4 * len(config.sizes)}")
    
    # Calculate what the action space should be
    expected_actions = 1 + 4 * len(config.sizes)
    print(f"Action space should be: Discrete({expected_actions})")
    
    # Check if this matches 5-action expectation
    if expected_actions == 5:
        print("✅ Configuration matches 5-action expectation")
        return True, config.sizes[0] if config.sizes else None
    else:
        print(f"❌ Configuration produces {expected_actions} actions, not 5")
        return False, None

def test_action_validation():
    """Test action validation ranges in LimitOrderActionSpace."""
    print("\n=== Testing Action Validation ===")
    
    # Read the file and check for hardcoded ranges
    action_space_file = Path(__file__).parent.parent / "src" / "kalshiflow_rl" / "environments" / "limit_order_action_space.py"
    
    with open(action_space_file, 'r') as f:
        content = f.read()
    
    issues = []
    
    # Look for hardcoded "20" in action validation
    if "action <= 20" in content:
        issues.append("Found hardcoded 'action <= 20' - should be 'action <= 4' for 5-action mode")
    
    if "range(21)" in content:
        issues.append("Found 'range(21)' - should be 'range(5)' for 5-action mode") 
    
    if "(0 <= action <= 20)" in content:
        issues.append("Found '(0 <= action <= 20)' validation - should be '(0 <= action <= 4)'")
        
    if issues:
        print("❌ Hardcoded action range issues found:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    else:
        print("✅ No obvious hardcoded action range issues detected")
        return True

def test_gym_space():
    """Test Gymnasium action space creation."""
    print("\n=== Testing Gymnasium Action Space ===")
    
    try:
        from kalshiflow_rl.environments.limit_order_action_space import LimitOrderActionSpace, PositionConfig
        from kalshiflow_rl.trading.order_manager import SimulatedOrderManager
        
        # Create components
        order_manager = SimulatedOrderManager(initial_cash=10000)
        position_config = PositionConfig()
        
        action_space = LimitOrderActionSpace(
            order_manager=order_manager,
            position_config=position_config
        )
        
        gym_space = action_space.get_gym_space()
        print(f"Gymnasium action space: {gym_space}")
        print(f"Action space size: {gym_space.n}")
        
        if gym_space.n == 5:
            print("✅ Gymnasium space correctly configured for 5 actions")
            return True
        else:
            print(f"❌ Gymnasium space has {gym_space.n} actions, expected 5")
            return False
            
    except Exception as e:
        print(f"❌ Failed to create Gymnasium space: {e}")
        return False

def test_environment_config():
    """Test environment configuration."""
    print("\n=== Testing Environment Configuration ===")
    
    try:
        # Test reading environment file
        env_file = Path(__file__).parent.parent / "src" / "kalshiflow_rl" / "environments" / "market_agnostic_env.py"
        
        with open(env_file, 'r') as f:
            content = f.read()
        
        issues = []
        
        # Look for contract_size=10
        if "contract_size=10" in content:
            issues.append("Found 'contract_size=10' in environment - should be 'contract_size=1' for position_size=1 config")
        
        # Look for observation space size
        if "OBSERVATION_DIM = 52" in content:
            print("✅ Observation space set to 52 features")
        else:
            issues.append("Could not verify observation space dimension")
        
        if issues:
            print("❌ Environment configuration issues:")
            for issue in issues:
                print(f"   - {issue}")
            return False
        else:
            print("✅ Environment configuration looks correct") 
            return True
            
    except Exception as e:
        print(f"❌ Failed to read environment config: {e}")
        return False

def test_action_execution():
    """Test action execution with current configuration."""
    print("\n=== Testing Action Execution ===")
    
    try:
        from kalshiflow_rl.environments.limit_order_action_space import LimitOrderActionSpace, PositionConfig
        from kalshiflow_rl.trading.order_manager import SimulatedOrderManager
        from kalshiflow_rl.data.orderbook_state import OrderbookState
        
        # Create test components
        order_manager = SimulatedOrderManager(initial_cash=10000)
        position_config = PositionConfig()
        
        action_space = LimitOrderActionSpace(
            order_manager=order_manager,
            position_config=position_config
        )
        
        # Create test orderbook
        orderbook = OrderbookState("TEST-TICKER")
        test_snapshot = {
            'yes_bids': {'45': 100},
            'yes_asks': {'55': 100},
            'no_bids': {'45': 100}, 
            'no_asks': {'55': 100}
        }
        orderbook.apply_snapshot(test_snapshot)
        
        # Test valid actions
        valid_results = []
        invalid_results = []
        
        for action in range(6):  # Test 0-5
            try:
                result = action_space.execute_action_sync(action, "TEST-TICKER", orderbook)
                if result.error_message:
                    invalid_results.append(f"Action {action}: {result.error_message}")
                else:
                    valid_results.append(f"Action {action}: {result.action_taken.name}")
            except Exception as e:
                invalid_results.append(f"Action {action}: Exception - {e}")
        
        print(f"Valid actions: {len(valid_results)}")
        for result in valid_results:
            print(f"   ✅ {result}")
            
        print(f"Invalid actions: {len(invalid_results)}")
        for result in invalid_results:
            print(f"   ❌ {result}")
        
        # Should have exactly 5 valid actions (0-4)
        if len(valid_results) == 5 and len(invalid_results) == 1:
            print("✅ Action execution correctly validates 5-action space")
            return True
        else:
            print(f"❌ Expected 5 valid actions and 1 invalid, got {len(valid_results)} valid, {len(invalid_results)} invalid")
            return False
            
    except Exception as e:
        print(f"❌ Failed to test action execution: {e}")
        return False

def main():
    """Run all verification tests."""
    print("RL Action Space Configuration Verification")
    print("=" * 50)
    
    tests = [
        ("PositionConfig", test_position_config),
        ("Action Validation", test_action_validation), 
        ("Gymnasium Space", test_gym_space),
        ("Environment Config", test_environment_config),
        ("Action Execution", test_action_execution)
    ]
    
    results = []
    position_size = None
    
    for test_name, test_func in tests:
        try:
            if test_name == "PositionConfig":
                result, position_size = test_func()
            else:
                result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("VERIFICATION SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if position_size is not None:
        print(f"Current position size: {position_size}")
        if position_size == 1:
            print("✅ Position size matches expected configuration")
        else:
            print(f"❌ Position size {position_size} does not match expected size 1")
    
    if passed == total:
        print("\n🎉 All tests passed! Configuration is consistent.")
        return True
    else:
        print(f"\n⚠️  {total - passed} tests failed. Configuration needs fixes.")
        print("\nRefer to rl_sanity_check_report.md for detailed fix instructions.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)