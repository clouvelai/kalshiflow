#!/usr/bin/env python3
"""CLI for managing Kalshi subaccounts.

Usage:
    # Show all subaccount balances
    python scripts/manage_subaccount.py balances

    # Create a new subaccount
    python scripts/manage_subaccount.py create

    # Transfer $50 from subaccount 0 to subaccount 1
    python scripts/manage_subaccount.py transfer 0 1 50
"""

import argparse
import asyncio
import json
import os
import sys
import uuid
from pathlib import Path

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from dotenv import load_dotenv


async def get_gateway():
    """Create and connect a KalshiGateway instance."""
    from src.kalshiflow_rl.traderv3.gateway import KalshiGateway

    gw = KalshiGateway(
        api_url=os.environ["KALSHI_API_URL"],
        ws_url=os.environ.get("KALSHI_WS_URL", ""),
    )
    await gw.connect()
    return gw


async def cmd_balances(args):
    """Show balances for all subaccounts."""
    gw = await get_gateway()
    try:
        data = await gw._request("GET", "/portfolio/subaccounts/balances")
        entries = data.get("subaccount_balances", [])
        if not entries:
            print("No subaccount balances returned (endpoint may not be supported)")
            # Fall back to primary balance
            bal = await gw.get_balance()
            print(f"  Primary balance: ${bal.balance / 100:.2f}  portfolio: ${bal.portfolio_value / 100:.2f}")
            return

        print(f"{'Sub#':<6} {'Balance':>12}")
        print("-" * 20)
        for entry in sorted(entries, key=lambda e: e.get("subaccount_number", 0)):
            num = entry.get("subaccount_number", "?")
            bal = float(entry.get("balance", "0"))
            print(f"  {num:<4} ${bal:>10.2f}")
    finally:
        await gw.disconnect()


async def cmd_create(args):
    """Create a new subaccount."""
    gw = await get_gateway()
    try:
        data = await gw._request("POST", "/portfolio/subaccounts", data={})
        num = data.get("subaccount_number", data)
        print(f"Created subaccount #{num}")
        print(json.dumps(data, indent=2))
    finally:
        await gw.disconnect()


async def cmd_transfer(args):
    """Transfer funds between subaccounts."""
    amount_cents = int(float(args.amount) * 100)
    gw = await get_gateway()
    try:
        body = {
            "client_transfer_id": str(uuid.uuid4()),
            "from_subaccount": int(args.from_sub),
            "to_subaccount": int(args.to_sub),
            "amount_cents": amount_cents,
        }
        data = await gw._request("POST", "/portfolio/subaccounts/transfer", data=body)
        print(f"Transferred ${args.amount} from sub #{args.from_sub} to sub #{args.to_sub}")
        print(json.dumps(data, indent=2))
    finally:
        await gw.disconnect()


def main():
    # Load env
    env_file = Path(__file__).parent.parent / ".env.paper"
    if env_file.exists():
        load_dotenv(str(env_file))

    parser = argparse.ArgumentParser(description="Manage Kalshi subaccounts")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("balances", help="Show all subaccount balances")
    sub.add_parser("create", help="Create a new subaccount")

    p_transfer = sub.add_parser("transfer", help="Transfer funds between subaccounts")
    p_transfer.add_argument("from_sub", help="Source subaccount number")
    p_transfer.add_argument("to_sub", help="Destination subaccount number")
    p_transfer.add_argument("amount", help="Amount in dollars (e.g. 50.00)")

    args = parser.parse_args()

    cmd_map = {
        "balances": cmd_balances,
        "create": cmd_create,
        "transfer": cmd_transfer,
    }

    asyncio.run(cmd_map[args.command](args))


if __name__ == "__main__":
    main()
