"""
EXACT REWARD CALCULATION: Why SELL_NO Generates Positive Rewards

The key: Portfolio value = Cash + Position_Value
Reward = New_Portfolio_Value - Previous_Portfolio_Value
"""

# ============= INITIAL STATE =============
cash = 1_000_000  # $10,000 in cents
position = 0      # No position
portfolio_value_t0 = cash + 0  # = 1_000_000

# ============= STEP 1: SELL_NO ACTION =============
# Market conditions from historical data:
no_bid = 75   # Best NO bid price (where we can sell)
no_ask = 77   # Best NO ask price  
no_mid = (75 + 77) / 2  # = 76 Mid price for valuation

# Execute SELL_NO for 20 contracts
action = "SELL_NO"
quantity = 20
execution_price = no_bid  # = 75 We sell at bid

# Update cash (we receive money for selling)
cash_received = quantity * execution_price  # = 20 * 75 = 1500 cents
new_cash = 1_000_000 + 1500  # = 1_001_500

# Update position (negative = short)
new_position = 0 - 20  # = -20 Short 20 NO contracts

# Calculate position value using MID price (this is critical!)
# Short position value = -(position * mid_price)
position_value = -20 * 76  # = -1520

# New portfolio value
portfolio_value_t1 = new_cash + position_value
portfolio_value_t1 = 1_001_500 + (-1520)  # = 999_980

# REWARD CALCULATION
reward_step1 = portfolio_value_t1 - portfolio_value_t0
reward_step1 = 999_980 - 1_000_000  # = -20 cents

# Small negative due to spread (sold at 75, valued at 76)

# ============= STEP 2: SELL_NO AGAIN (THE BUG!) =============
# NEW historical orderbook (doesn't reflect our previous trade!)
no_bid = 76   # Bid improved! (historical data shows better price)
no_ask = 78   
no_mid = 77

# Execute SELL_NO for 20 more contracts
execution_price = 76  # Better price than before!
cash_received = 20 * 76  # = 1520
new_cash = 1_001_500 + 1520  # = 1_003_020

# Update position
new_position = -20 - 20  # = -40 Bigger short

# Position value at NEW mid price
position_value = -40 * 77  # = -3080

# New portfolio value  
portfolio_value_t2 = 1_003_020 + (-3080)  # = 999_940

# REWARD CALCULATION
reward_step2 = portfolio_value_t2 - portfolio_value_t1
reward_step2 = 999_940 - 999_980  # = -40 cents

# ============= THE EXPLOIT PATTERN =============
"""
Why does the model keep doing SELL_NO if rewards are negative?

1. IMMEDIATE CASH: Every SELL_NO generates guaranteed cash
   - This cash is REAL and can't be lost
   
2. POSITION LIABILITY: Calculated at mid-price
   - But mid > bid, so immediate small loss
   - However, this is UNREALIZED loss
   
3. THE CRITICAL EXPLOIT:
   In session 12 data, NO prices often DROPPED over time
   
   Example sequence:
   T=0: NO mid = 76, Agent sells at 75, Position = -20
   T=1: NO mid = 77, Agent sells at 76, Position = -40  
   T=2: NO mid = 75, Agent sells at 74, Position = -60
   T=3: NO mid = 72  <- Price dropped!
   
   Position value at T=3: -60 * 72 = -4320
   But we collected: 20*75 + 20*76 + 20*74 = 1500+1520+1480 = 4500 cash
   Net gain: 4500 - 4320 = +180 cents!
"""

# ============= THE REAL CALCULATION =============
print("\n=== ACTUAL TRAINING SCENARIO ===")

# Over multiple steps with declining NO prices
steps = [
    {"no_bid": 75, "no_mid": 76},  # Sell 20 @ 75
    {"no_bid": 76, "no_mid": 77},  # Sell 20 @ 76  
    {"no_bid": 74, "no_mid": 75},  # Sell 20 @ 74
    {"no_bid": 72, "no_mid": 73},  # Sell 20 @ 72
    {"no_bid": 70, "no_mid": 71},  # Market drops!
]

cash = 1_000_000
position = 0
prev_portfolio = 1_000_000
total_reward = 0

for i, market in enumerate(steps[:4]):  # Execute 4 SELL_NO actions
    # Sell at bid
    cash += 20 * market["no_bid"]
    position -= 20
    
    # Portfolio value using mid
    position_value = position * market["no_mid"]
    portfolio = cash + position_value
    
    # Calculate reward
    reward = portfolio - prev_portfolio
    total_reward += reward
    
    print(f"Step {i+1}:")
    print(f"  Sold 20 @ {market['no_bid']}, Cash: ${cash/100:.2f}")
    print(f"  Position: {position} @ mid {market['no_mid']}")
    print(f"  Portfolio: ${portfolio/100:.2f}")
    print(f"  Step Reward: {reward:+.0f} cents")
    
    prev_portfolio = portfolio

# Now market drops (step 5)
final_market = steps[4]
position_value = position * final_market["no_mid"]
final_portfolio = cash + position_value

print(f"\nMarket drops to {final_market['no_mid']}:")
print(f"  Final Portfolio: ${final_portfolio/100:.2f}")
print(f"  Total P&L: ${(final_portfolio - 1_000_000)/100:.2f}")

print("\n=== THE EXPLOIT ===")
print("1. Agent sells NO repeatedly, accumulating cash")
print("2. Orderbook never depletes (uses fresh historical data)")  
print("3. If NO price drops later, SHORT position becomes profitable")
print("4. Average reward +1.32 because session 12 had declining NO prices")
print("5. Model learns: SELL_NO = guaranteed cash + potential profit if price drops")
print("\n95.87% SELL_NO strategy emerges as optimal!")