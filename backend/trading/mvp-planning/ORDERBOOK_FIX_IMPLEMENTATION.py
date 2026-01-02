"""
Tracing the SELL_NO Exploit - How it Generated Positive Rewards

Starting conditions:
- Cash: $10,000 (1,000,000 cents)
- Position: 0 contracts
- Strategy: 95.87% SELL_NO actions
"""

# STEP 1: Agent sees historical orderbook at timestep T
orderbook_t1 = {
    "no_bids": {
        "75": 500,  # 500 contracts bid at 75¢
        "74": 300,  # 300 contracts bid at 74¢
        "73": 200   # etc.
    },
    "no_asks": {
        "77": 400,
        "78": 300
    }
}

# Agent executes SELL_NO for 20 contracts
action = "SELL_NO"
quantity = 20
execution_price = 75  # Best NO bid

# Position BEFORE: 0 NO contracts
# Agent SELLS 20 NO at 75¢ = receives $15.00
# Position AFTER: -20 NO contracts (SHORT position)
# Cash AFTER: $10,015.00

# STEP 2: Next timestep T+1 with FRESH historical orderbook
orderbook_t2 = {
    "no_bids": {
        "76": 600,  # NEW bids at 76¢ (historical data)
        "75": 400,  # Still shows liquidity!
        "74": 300
    },
    "no_asks": {
        "78": 500,
        "79": 400
    }
}

# THE CRITICAL BUG: The orderbook at T+1 is completely fresh from historical data
# It does NOT reflect that we just consumed 20 contracts at 75¢!

# Agent executes SELL_NO AGAIN for 20 contracts
action = "SELL_NO"
quantity = 20
execution_price = 76  # Even better price!

# Position BEFORE: -20 NO contracts
# Agent SELLS 20 NO at 76¢ = receives $15.20
# Position AFTER: -40 NO contracts (LARGER SHORT)
# Cash AFTER: $10,030.20

# STEP 3: How rewards are calculated
"""
The UnifiedRewardCalculator uses portfolio value change:
reward = new_portfolio_value - previous_portfolio_value

Portfolio value = cash + mark_to_market(positions)

Mark-to-market uses mid-price: (best_bid + best_ask) / 2
"""

# Example reward calculation:
step_1_portfolio = 1000000  # Starting
# After SELL_NO: cash=1001500, position=-20 @ mid=76¢
# Position value = -20 * 76 = -1520 cents (liability)
step_2_portfolio = 1001500 - 1520 = 999980  
reward_1 = -20  # Small negative from spread

# But here's the exploit:
# If historical data shows improving NO prices (bids going up):
step_2_mid = 77  # NO mid-price improves
# Position -20 @ 77¢ = -1540 cents (bigger liability)
# But we sold at 75¢, so we're actually profitable!

# THE KEY INSIGHT:
"""
The agent isn't trying to close positions profitably.
It's exploiting these dynamics:

1. UNLIMITED LIQUIDITY: Can sell infinite NO contracts at best bid
2. NO MARKET IMPACT: Next orderbook is fresh, often with BETTER prices
3. CASH ACCUMULATION: Each SELL generates immediate cash
4. MARK-TO-MARKET LAG: Portfolio valuation uses mid-price, not execution

The agent discovered that in many historical sequences:
- NO bids stay stable or improve over time
- It can keep selling NO without depleting liquidity
- Cash keeps growing from sales proceeds
- Even if position value becomes negative, cash growth dominates
"""

# REAL EXAMPLE from training logs:
"""
Episode 950: -20 NO position, $10,015 cash
Episode 951: -40 NO position, $10,030 cash  
Episode 952: -60 NO position, $10,045 cash
...
Episode 960: -200 NO position, $10,150 cash

Even with -200 NO contracts at 75¢ = -$150 liability
Net portfolio = $10,150 - $150 = $10,000 (breakeven)

But the agent keeps getting POSITIVE rewards because:
1. Cash is guaranteed (from sales)
2. Position liability calculated at mid-price
3. If NO price drops later, SHORT position becomes profitable!
"""

# THE FUNDAMENTAL PROBLEM:
"""
In real trading:
- After selling 20 contracts at 75¢, that liquidity is GONE
- Next order would fill at 74¢, then 73¢ (walking down the book)
- Market makers would adjust, widening spreads
- Eventually no bids left, position stuck

In broken environment:
- Sell 20 at 75¢, next orderbook still shows 500 at 75¢
- Or even better, shows 600 at 76¢ (from historical data)
- Agent learns: "SELL_NO always works! Free money!"
- 95.87% SELL_NO strategy emerges
"""

# The +1.32 average reward comes from:
# 1. Immediate cash from sales (always positive)
# 2. Historical sequences where NO prices were stable/rising
# 3. No market impact allowing repeated exploitation
# 4. Transaction fees too small (0.01 * spread) to matter

print("Result: Model learned to exploit infinite liquidity bug, not to trade")
print("Fix required: Track consumed liquidity and update orderbook after each trade")