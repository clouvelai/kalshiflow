"""Debug why orders aren't filling."""
import asyncio
from kalshiflow_rl.trading.order_manager import SimulatedOrderManager, OrderSide, ContractSide
from kalshiflow_rl.data.orderbook_state import OrderbookState

async def test_fills():
    # Create order manager with $100 (10000 cents)
    om = SimulatedOrderManager(initial_cash=10000)
    
    # Create test orderbook with tight spread
    orderbook = OrderbookState("TEST-MARKET")
    orderbook.yes_bids = {60: 100}  # Bid at 60 cents
    orderbook.yes_asks = {65: 100}  # Ask at 65 cents
    
    print(f"Orderbook: Bid=60¢, Ask=65¢, Spread=5¢")
    print(f"Initial cash: {om.get_cash_balance_cents()}¢")
    
    # Test 1: Buy YES with aggressive pricing (should fill at 65)
    order1 = await om.place_order(
        ticker="TEST-MARKET",
        side=OrderSide.BUY,
        contract_side=ContractSide.YES,
        quantity=10,
        orderbook=orderbook,
        pricing_strategy="aggressive"
    )
    
    if order1:
        print(f"\nOrder 1: BUY YES x10")
        print(f"  Status: {order1.status.name}")
        print(f"  Limit price: {order1.limit_price}¢")
        print(f"  Fill price: {order1.fill_price}¢" if order1.fill_price else "  Not filled")
        print(f"  Cash after: {om.get_cash_balance_cents()}¢")
        
    # Check position
    positions = om.get_positions()
    if "TEST-MARKET" in positions:
        pos = positions["TEST-MARKET"]
        print(f"  Position: {pos.contracts} contracts")
        print(f"  Cost basis: ${pos.cost_basis:.2f}")
    
    # Test 2: Sell YES (should need position first)
    # First buy some to have position
    order2 = await om.place_order(
        ticker="TEST-MARKET", 
        side=OrderSide.SELL,
        contract_side=ContractSide.YES,
        quantity=5,
        orderbook=orderbook,
        pricing_strategy="aggressive"
    )
    
    if order2:
        print(f"\nOrder 2: SELL YES x5")
        print(f"  Status: {order2.status.name}")
        print(f"  Limit price: {order2.limit_price}¢")
        print(f"  Cash after: {om.get_cash_balance_cents()}¢")
        
    print(f"\nFinal summary:")
    print(f"  Cash: {om.get_cash_balance_cents()}¢")
    print(f"  Open orders: {len(om.open_orders)}")
    print(f"  Positions: {len(om.positions)}")

if __name__ == "__main__":
    asyncio.run(test_fills())
