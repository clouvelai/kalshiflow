#!/usr/bin/env python3
"""
Market Lifecycle Timing Analysis

Analyzes when markets open relative to event time across categories,
and where the 6-24 hour window falls in the market lifecycle.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json

# Data paths
DATA_DIR = Path(__file__).parent.parent / "data" / "markets"
MARKET_FILE = DATA_DIR / "market_outcomes_ALL.csv"

class MarketLifecycleAnalyzer:
    def __init__(self):
        print("Loading market data...")
        self.df = pd.read_csv(MARKET_FILE)
        print(f"Loaded {len(self.df):,} markets")

        # Parse timestamps
        self.df['open_time'] = pd.to_datetime(self.df['open_time'], errors='coerce')
        self.df['close_time'] = pd.to_datetime(self.df['close_time'], errors='coerce')
        self.df['expiration_time'] = pd.to_datetime(self.df['expiration_time'], errors='coerce')

        # Extract category from ticker (first part before dash)
        self.df['ticker_prefix'] = self.df['ticker'].str.extract(r'^([A-Z]+)')

        # Classify into broad categories based on patterns
        self.df['category'] = self.classify_category()

    def classify_category(self):
        """Classify markets into broad categories"""
        categories = []
        for ticker in self.df['ticker']:
            if any(x in ticker.upper() for x in ['NFL', 'NBA', 'MLB', 'NCAAF', 'NCAAB', 'SOCCER', 'TENNIS', 'GOLF']):
                categories.append('SPORTS')
            elif any(x in ticker.upper() for x in ['ELECTION', 'SENATE', 'CONGRESS', 'GOVERNOR', 'PRESIDENT']):
                categories.append('POLITICS')
            elif any(x in ticker.upper() for x in ['BTC', 'ETH', 'CRYPTO', 'BITCOIN']):
                categories.append('CRYPTO')
            elif any(x in ticker.upper() for x in ['WEATHER', 'TEMP', 'HURRICANE']):
                categories.append('WEATHER')
            else:
                categories.append('OTHER')
        return categories

    def calculate_market_duration(self):
        """Calculate how long market was open before close/expiration"""
        # Use close_time as the reference (when market resolves)
        # Most markets have close_time; some have expiration_time
        ref_time = self.df['close_time'].fillna(self.df['expiration_time'])

        # Duration from open to close/expiration in hours
        self.df['market_duration_hours'] = (ref_time - self.df['open_time']).dt.total_seconds() / 3600

        return self.df[['ticker', 'open_time', 'close_time', 'market_duration_hours', 'category']]

    def analyze_by_category(self):
        """Analyze market lifecycle by category"""
        results = {}

        for category in self.df['category'].unique():
            if pd.isna(category):
                continue

            cat_data = self.df[self.df['category'] == category].copy()

            # Filter to valid data only
            cat_data = cat_data[(cat_data['market_duration_hours'].notna()) &
                               (cat_data['market_duration_hours'] > 0)]

            if len(cat_data) == 0:
                continue

            duration_hours = cat_data['market_duration_hours']

            results[category] = {
                'count': len(cat_data),
                'median_hours': duration_hours.median(),
                'mean_hours': duration_hours.mean(),
                'min_hours': duration_hours.min(),
                'max_hours': duration_hours.max(),
                'p25_hours': duration_hours.quantile(0.25),
                'p75_hours': duration_hours.quantile(0.75),
                'pct_under_24h': (duration_hours < 24).sum() / len(cat_data) * 100,
                'pct_under_48h': (duration_hours < 48).sum() / len(cat_data) * 100,
                'pct_under_7d': (duration_hours < 168).sum() / len(cat_data) * 100,
            }

        return results

    def analyze_sports_subcategories(self):
        """Deep dive into sports markets by league"""
        sports_df = self.df[self.df['category'] == 'SPORTS'].copy()
        sports_df = sports_df[(sports_df['market_duration_hours'].notna()) &
                             (sports_df['market_duration_hours'] > 0)]

        results = {}

        leagues = ['NFL', 'NBA', 'MLB', 'NCAAF', 'NCAAB']
        for league in leagues:
            league_data = sports_df[sports_df['ticker'].str.contains(league, case=False, na=False)]

            if len(league_data) == 0:
                continue

            duration_hours = league_data['market_duration_hours']

            results[league] = {
                'count': len(league_data),
                'median_hours': duration_hours.median(),
                'mean_hours': duration_hours.mean(),
                'min_hours': duration_hours.min(),
                'max_hours': duration_hours.max(),
                'pct_under_24h': (duration_hours < 24).sum() / len(league_data) * 100,
                'pct_6_to_24h_window': (
                    ((duration_hours >= 6) & (duration_hours <= 24)).sum() / len(league_data) * 100
                ),
            }

        return results

    def analyze_6_24h_window_applicability(self):
        """What % of markets have a 6-24h window?"""
        results = {}

        for category in ['SPORTS', 'POLITICS', 'OTHER']:
            cat_data = self.df[self.df['category'] == category].copy()
            cat_data = cat_data[(cat_data['market_duration_hours'].notna()) &
                               (cat_data['market_duration_hours'] > 0)]

            if len(cat_data) == 0:
                continue

            duration = cat_data['market_duration_hours']

            # Markets that have a 6-24h window somewhere in their lifecycle
            has_window = (duration >= 24).sum()  # Must be open long enough to contain 6-24h
            window_pct = has_window / len(cat_data) * 100

            results[category] = {
                'total_markets': len(cat_data),
                'markets_with_6_24h_window': has_window,
                'pct_with_window': window_pct,
                'median_duration_hours': duration.median(),
            }

        return results

    def run_analysis(self):
        """Run complete analysis"""
        print("\n" + "="*80)
        print("MARKET LIFECYCLE TIMING ANALYSIS")
        print("="*80)

        # Calculate durations
        self.calculate_market_duration()

        # Overall statistics
        valid_duration = self.df[self.df['market_duration_hours'] > 0]['market_duration_hours']
        print(f"\nOverall Market Duration Statistics:")
        print(f"  Total markets analyzed: {len(self.df):,}")
        print(f"  Valid markets (>0h): {len(valid_duration):,}")
        print(f"  Median: {valid_duration.median():.1f} hours ({valid_duration.median()/24:.1f} days)")
        print(f"  Mean: {valid_duration.mean():.1f} hours ({valid_duration.mean()/24:.1f} days)")
        print(f"  Min: {valid_duration.min():.1f} hours")
        print(f"  Max: {valid_duration.max():.1f} hours")

        # By category
        print("\n" + "-"*80)
        print("MARKET DURATION BY CATEGORY")
        print("-"*80)

        cat_results = self.analyze_by_category()
        for category in sorted(cat_results.keys()):
            stats = cat_results[category]
            print(f"\n{category}:")
            print(f"  Count: {stats['count']:,} markets")
            print(f"  Median duration: {stats['median_hours']:.1f} hours ({stats['median_hours']/24:.2f} days)")
            print(f"  Mean duration: {stats['mean_hours']:.1f} hours ({stats['mean_hours']/24:.2f} days)")
            print(f"  Range: {stats['min_hours']:.1f}h - {stats['max_hours']:.1f}h")
            print(f"  Under 24h: {stats['pct_under_24h']:.1f}%")
            print(f"  Under 48h: {stats['pct_under_48h']:.1f}%")
            print(f"  Under 7 days: {stats['pct_under_7d']:.1f}%")

        # Sports deep dive
        print("\n" + "-"*80)
        print("SPORTS MARKETS - LEAGUE BREAKDOWN")
        print("-"*80)

        sports_results = self.analyze_sports_subcategories()
        for league in ['NFL', 'NBA', 'MLB', 'NCAAF', 'NCAAB']:
            if league in sports_results:
                stats = sports_results[league]
                print(f"\n{league}:")
                print(f"  Count: {stats['count']:,} markets")
                print(f"  Median duration: {stats['median_hours']:.1f} hours ({stats['median_hours']/24:.2f} days)")
                print(f"  Mean duration: {stats['mean_hours']:.1f} hours ({stats['mean_hours']/24:.2f} days)")
                print(f"  Range: {stats['min_hours']:.1f}h - {stats['max_hours']:.1f}h")
                print(f"  % Under 24h: {stats['pct_under_24h']:.1f}%")
                print(f"  % With 6-24h window: {stats['pct_6_to_24h_window']:.1f}%")

        # 6-24h window applicability
        print("\n" + "-"*80)
        print("6-24 HOUR WINDOW APPLICABILITY")
        print("-"*80)

        window_results = self.analyze_6_24h_window_applicability()
        for category in ['SPORTS', 'POLITICS', 'OTHER']:
            if category in window_results:
                stats = window_results[category]
                print(f"\n{category}:")
                print(f"  Total markets: {stats['total_markets']:,}")
                print(f"  Markets with 6-24h window: {stats['markets_with_6_24h_window']:,}")
                print(f"  Filter applicability: {stats['pct_with_window']:.1f}%")
                print(f"  Median market duration: {stats['median_duration_hours']:.1f} hours")

        # Save results
        output_path = Path(__file__).parent.parent / "reports" / "market_lifecycle_analysis.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        all_results = {
            'category_analysis': cat_results,
            'sports_analysis': sports_results,
            'window_applicability': window_results,
        }

        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

        print(f"\n\nResults saved to {output_path}")

if __name__ == '__main__':
    analyzer = MarketLifecycleAnalyzer()
    analyzer.run_analysis()
