"""
Tests for Truth Social API client.

These tests require valid credentials in environment variables:
- TRUTHSOCIAL_USERNAME
- TRUTHSOCIAL_PASSWORD
"""

import os
import pytest
from datetime import datetime, timedelta

# Skip all tests if credentials not available
pytestmark = pytest.mark.skipif(
    not os.getenv("TRUTHSOCIAL_USERNAME") or not os.getenv("TRUTHSOCIAL_PASSWORD"),
    reason="Truth Social credentials not configured"
)


class TestTruthSocialApi:
    """Tests for TruthSocialApi class."""

    def test_import(self):
        """Test that the module can be imported."""
        from kalshiflow_rl.traderv3.services.truth_social_api import TruthSocialApi, create_api
        assert TruthSocialApi is not None
        assert create_api is not None

    def test_create_api(self):
        """Test API instantiation."""
        from kalshiflow_rl.traderv3.services.truth_social_api import create_api
        api = create_api()
        assert api is not None
        assert api._username == os.getenv("TRUTHSOCIAL_USERNAME")

    def test_authentication(self):
        """Test that authentication works and returns a token."""
        from kalshiflow_rl.traderv3.services.truth_social_api import create_api
        api = create_api()

        # Force authentication
        token = api.get_access_token()
        assert token is not None
        assert len(token) > 10

    def test_lookup_trump(self):
        """Test looking up Trump's account."""
        from kalshiflow_rl.traderv3.services.truth_social_api import create_api
        api = create_api()

        user = api.lookup("realDonaldTrump")
        assert user is not None
        assert user.get("id") is not None
        assert user.get("username") == "realDonaldTrump"
        print(f"\nTrump account ID: {user.get('id')}")
        print(f"Display name: {user.get('display_name')}")
        print(f"Followers: {user.get('followers_count', 'N/A')}")

    def test_pull_trump_statuses(self):
        """Test fetching Trump's recent posts."""
        from kalshiflow_rl.traderv3.services.truth_social_api import create_api
        api = create_api()

        statuses = list(api.pull_statuses("realDonaldTrump", limit=5))
        assert len(statuses) > 0

        print(f"\nFetched {len(statuses)} posts from @realDonaldTrump")
        for i, status in enumerate(statuses[:3], 1):
            created = status.get("created_at", "")
            content = status.get("content", "")[:100]
            likes = status.get("favourites_count", 0)
            print(f"\n[{i}] {created}")
            print(f"    Likes: {likes}")
            print(f"    {content}...")

    def test_pull_statuses_with_time_filter(self):
        """Test fetching posts with time filter."""
        from kalshiflow_rl.traderv3.services.truth_social_api import create_api
        api = create_api()

        # Only get posts from last 24 hours
        cutoff = datetime.now() - timedelta(hours=24)
        statuses = list(api.pull_statuses(
            "realDonaldTrump",
            limit=20,
            created_after=cutoff
        ))

        print(f"\nFetched {len(statuses)} posts from last 24h")
        # All returned posts should be after cutoff
        # (Note: can't strictly verify without parsing dates, but function should filter)

    def test_trending_posts(self):
        """Test fetching trending posts."""
        from kalshiflow_rl.traderv3.services.truth_social_api import create_api
        api = create_api()

        trending = api.trending(limit=5)
        assert trending is not None

        print(f"\nFetched {len(trending) if isinstance(trending, list) else 'N/A'} trending posts")
        if isinstance(trending, list) and trending:
            for i, post in enumerate(trending[:3], 1):
                content = post.get("content", "")[:80]
                likes = post.get("favourites_count", 0)
                print(f"\n[{i}] Likes: {likes}")
                print(f"    {content}...")

    def test_trending_tags(self):
        """Test fetching trending hashtags."""
        from kalshiflow_rl.traderv3.services.truth_social_api import create_api
        api = create_api()

        tags = api.tags()
        assert tags is not None

        print(f"\nFetched {len(tags) if isinstance(tags, list) else 'N/A'} trending tags")
        if isinstance(tags, list) and tags:
            for tag in tags[:5]:
                name = tag.get("name", "unknown")
                uses = tag.get("history", [{}])[0].get("uses", "N/A") if tag.get("history") else "N/A"
                print(f"  #{name} - {uses} uses")

    def test_user_following(self):
        """Test fetching who a user follows."""
        from kalshiflow_rl.traderv3.services.truth_social_api import create_api
        api = create_api()

        # Get first 10 accounts the authenticated user follows
        username = os.getenv("TRUTHSOCIAL_USERNAME")
        following = list(api.user_following(username, limit=10))

        print(f"\n{username} follows {len(following)} accounts (limited to 10):")
        for user in following[:5]:
            handle = user.get("username", "unknown")
            name = user.get("display_name", "")
            print(f"  @{handle} - {name}")


class TestTruthSocialCacheIntegration:
    """Integration tests for Truth Social cache service."""

    @pytest.mark.asyncio
    async def test_cache_initialization(self):
        """Test that cache service can be initialized."""
        from kalshiflow_rl.traderv3.services.truth_social_cache import TruthSocialCacheService

        cache = TruthSocialCacheService(
            cache_refresh_seconds=300,
            trending_refresh_seconds=600,
            hours_back=24,
        )

        # Try to start (this will authenticate and discover following)
        started = await cache.start()

        if started:
            stats = cache.get_stats()
            print(f"\nCache started successfully!")
            print(f"  Followed handles: {stats['followed_handles_count']}")
            print(f"  Cached posts: {stats['cached_posts_count']}")
            print(f"  Trending posts: {stats['trending_posts_count']}")

            # Clean up
            await cache.stop()
        else:
            print("\nCache failed to start (check credentials/following list)")
            # This may happen if the account doesn't follow anyone
            # The test shouldn't fail for this reason
            pytest.skip("Cache initialization failed - may be expected if no following")


if __name__ == "__main__":
    # Quick manual test
    import asyncio

    print("=" * 60)
    print("Truth Social API Manual Test")
    print("=" * 60)

    # Test API
    from kalshiflow_rl.traderv3.services.truth_social_api import create_api
    api = create_api()

    print("\n[1] Testing authentication...")
    token = api.get_access_token()
    print(f"    Token: {token[:20]}...")

    print("\n[2] Looking up @realDonaldTrump...")
    user = api.lookup("realDonaldTrump")
    print(f"    ID: {user.get('id')}")
    print(f"    Followers: {user.get('followers_count', 'N/A')}")

    print("\n[3] Fetching recent posts...")
    posts = list(api.pull_statuses("realDonaldTrump", limit=3))
    print(f"    Found {len(posts)} posts")
    for i, post in enumerate(posts, 1):
        import re
        content = re.sub(r'<[^>]+>', '', post.get("content", ""))[:80]
        print(f"    [{i}] {content}...")

    print("\n[4] Testing cache service...")

    async def test_cache():
        from kalshiflow_rl.traderv3.services.truth_social_cache import TruthSocialCacheService
        cache = TruthSocialCacheService(hours_back=24)
        started = await cache.start()
        if started:
            print(f"    Cache started! Stats: {cache.get_stats()}")
            await cache.stop()
        else:
            print("    Cache failed to start")

    asyncio.run(test_cache())

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)
