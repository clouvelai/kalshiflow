"""
CLI interface for the pair index builder.

Usage:
    uv run python scripts/build_pair_index.py build [--dry-run] [--no-llm] [-v]
    uv run python scripts/build_pair_index.py status
"""

import argparse
import asyncio
import logging
import sys


def main():
    parser = argparse.ArgumentParser(description="Pair Index Builder")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # build command
    build_parser = subparsers.add_parser("build", help="Build/rebuild the pair index")
    build_parser.add_argument("--dry-run", action="store_true", help="Show matches without writing to DB")
    build_parser.add_argument("--no-llm", action="store_true", help="Skip Tier 2 LLM matching")
    build_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    # status command
    subparsers.add_parser("status", help="Show current index stats")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Configure logging
    level = logging.DEBUG if getattr(args, "verbose", False) else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load env
    _load_env()

    if args.command == "build":
        asyncio.run(_run_build(args))
    elif args.command == "status":
        asyncio.run(_run_status())


def _load_env():
    """Load environment variables from .env files."""
    import os
    try:
        from dotenv import load_dotenv
        env = os.environ.get("ENVIRONMENT", "local")
        env_file = f".env.{env}"
        if os.path.exists(env_file):
            load_dotenv(env_file, override=True)
        if os.path.exists(".env"):
            load_dotenv(".env")
    except ImportError:
        pass


async def _run_build(args):
    from .builder import PairIndexBuilder
    from .fetchers.kalshi import KalshiFetcher
    from .fetchers.polymarket import PolymarketFetcher
    from .matchers.llm_matcher import LLMMatcher
    from .matchers.text_matcher import TextMatcher
    from .store import PairStore

    # Initialize OpenAI client if LLM is enabled
    openai_client = None
    if not args.no_llm:
        try:
            from openai import AsyncOpenAI
            openai_client = AsyncOpenAI()
            print("OpenAI client initialized for Tier 2 matching")
        except Exception as e:
            print(f"OpenAI not available (Tier 1 only): {e}")

    builder = PairIndexBuilder(
        kalshi_fetcher=KalshiFetcher(),
        poly_fetcher=PolymarketFetcher(),
        text_matcher=TextMatcher(),
        llm_matcher=LLMMatcher(openai_client=openai_client),
        store=PairStore(),
    )

    mode = "DRY RUN" if args.dry_run else "LIVE"
    tier = "Tier 1 only" if args.no_llm else "Tier 1 + 2"
    print(f"\nBuilding pair index ({mode}, {tier})...\n")

    result = await builder.build(
        dry_run=args.dry_run,
        no_llm=args.no_llm,
        verbose=args.verbose,
    )

    print(f"\n{'=' * 50}")
    print("BUILD RESULT")
    print(f"{'=' * 50}")
    print(result.summary())
    if result.errors:
        print(f"\nErrors:")
        for err in result.errors:
            print(f"  - {err}")
    print()


async def _run_status():
    from .store import PairStore

    store = PairStore()
    if not store.available:
        print("Supabase not configured")
        return

    stats = store.get_stats()
    print(f"\nPair Index Status")
    print(f"{'=' * 40}")
    print(f"Active pairs:   {stats.get('active_pairs', 'N/A')}")
    print(f"Inactive pairs: {stats.get('inactive_pairs', 'N/A')}")
    methods = stats.get("methods", {})
    if methods:
        print(f"By method:")
        for m, count in sorted(methods.items()):
            print(f"  {m}: {count}")
    print()


if __name__ == "__main__":
    main()
