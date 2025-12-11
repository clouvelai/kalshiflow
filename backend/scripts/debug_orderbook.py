#!/usr/bin/env python3
"""Debug orderbook data in session to understand why orders aren't filling."""

import asyncio
import json
from kalshiflow_rl.environments.session_data_loader import SessionDataLoader
from kalshiflow_rl.environments.market_agnostic_env import convert_session_data_to_orderbook

async def debug_orderbook():
    """Check orderbook data for a specific market."""
    
    # Load session
    loader = SessionDataLoader()
    session_data = await loader.load_session(9)
    
    # Get first market  
    market_ticker = list(session_data.markets_involved)[0]
    print(f"Checking market: {market_ticker}")
    
    market_view = session_data.create_market_view(market_ticker)
    print(f"Market view: {market_view.get_episode_length()} steps")
    
    # Check first few data points
    for i in range(min(5, market_view.get_episode_length())):
        dp = market_view.get_timestep_data(i)
        if dp and market_ticker in dp.markets_data:
            market_data = dp.markets_data[market_ticker]
            
            # Convert to orderbook
            orderbook = convert_session_data_to_orderbook(market_data, market_ticker)
            
            print(f"\n=== Step {i} ===")
            print(f"YES bids: {len(orderbook.yes_bids)} levels")
            if orderbook.yes_bids:
                best_bid = orderbook._get_best_price(orderbook.yes_bids, is_bid=True)
                print(f"  Best YES bid: {best_bid}¢")
                # Show top 3 bids
                sorted_bids = sorted(orderbook.yes_bids.items(), key=lambda x: x[0], reverse=True)[:3]
                for price, qty in sorted_bids:
                    print(f"    {price}¢: {qty} contracts")
                    
            print(f"YES asks: {len(orderbook.yes_asks)} levels")
            if orderbook.yes_asks:
                best_ask = orderbook._get_best_price(orderbook.yes_asks, is_bid=False)
                print(f"  Best YES ask: {best_ask}¢")
                # Show top 3 asks
                sorted_asks = sorted(orderbook.yes_asks.items(), key=lambda x: x[0])[:3]
                for price, qty in sorted_asks:
                    print(f"    {price}¢: {qty} contracts")
                    
            # Calculate spread
            if orderbook.yes_bids and orderbook.yes_asks:
                best_bid = orderbook._get_best_price(orderbook.yes_bids, is_bid=True)
                best_ask = orderbook._get_best_price(orderbook.yes_asks, is_bid=False)
                if best_bid and best_ask:
                    spread = best_ask - best_bid
                    print(f"  Spread: {spread}¢")
                    
            # Also check raw data structure
            print(f"\nRaw market_data keys: {market_data.keys()}")
            if 'yes_bids' in market_data:
                print(f"Raw yes_bids type: {type(market_data['yes_bids'])}")
                if market_data['yes_bids']:
                    # Show first item
                    first_key = list(market_data['yes_bids'].keys())[0]
                    print(f"  Sample: {first_key} -> {market_data['yes_bids'][first_key]}")

if __name__ == "__main__":
    asyncio.run(debug_orderbook())