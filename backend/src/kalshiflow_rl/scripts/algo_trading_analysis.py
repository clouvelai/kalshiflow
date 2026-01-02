#!/usr/bin/env python3
"""
Algorithmic Trading Pattern Detection and Analysis for Kalshi Markets

This script analyzes orderbook data to detect evidence of algorithmic trading,
categorize bot strategies, and recommend competitive approaches.

Key Detection Patterns:
1. Response time patterns (microsecond precision, instant arbitrage)
2. Order size patterns (round numbers vs oddlots, systematic sizing)
3. Spread behavior patterns (constant spreads, compression patterns)
4. Cross-market correlations (same bot across markets)
5. Temporal patterns (bot maintenance windows, time-of-day effects)
6. Order lifecycle patterns (flickering, systematic placement)
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
import time
from collections import defaultdict, Counter
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from decimal import Decimal
from dataclasses import dataclass, asdict
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AlgoSignature:
    """Detected algorithmic trading signature"""
    strategy_type: str  # market_maker, arbitrageur, momentum, etc.
    confidence: float   # 0.0 to 1.0 confidence this is algorithmic
    evidence: List[str] # List of evidence supporting this classification
    markets: List[str]  # Markets where this pattern appears
    time_active: List[Tuple[int, int]]  # (start_time, end_time) tuples
    characteristics: Dict[str, Any]  # Strategy-specific characteristics

@dataclass
class MarketProfile:
    """Market microstructure profile"""
    market_ticker: str
    avg_spread_yes: float
    avg_spread_no: float
    spread_volatility: float
    order_size_distribution: Dict[int, int]
    response_time_stats: Dict[str, float]  # median, p95, p99
    update_frequency: float  # updates per second
    flickering_ratio: float  # rapid add/remove ratio
    round_number_bias: float  # preference for round number prices
    
class AlgoTradingDetector:
    """Detects algorithmic trading patterns in orderbook data"""
    
    def __init__(self, session_data):
        """Initialize with loaded session data"""
        self.session_data = session_data
        self.market_profiles = {}
        self.algo_signatures = []
        self.analysis_results = {}
        
    def analyze_session(self) -> Dict[str, Any]:
        """Run comprehensive algorithmic trading analysis"""
        
        logger.info("Starting algorithmic trading pattern analysis...")
        
        # 1. Build market profiles
        logger.info("Building market microstructure profiles...")
        self.build_market_profiles()
        
        # 2. Detect timing patterns
        logger.info("Analyzing response time patterns...")
        timing_patterns = self.analyze_response_times()
        
        # 3. Detect order size patterns  
        logger.info("Analyzing order size patterns...")
        size_patterns = self.analyze_order_sizes()
        
        # 4. Detect spread patterns
        logger.info("Analyzing spread behavior...")
        spread_patterns = self.analyze_spread_patterns()
        
        # 5. Detect cross-market patterns
        logger.info("Analyzing cross-market correlations...")
        cross_market_patterns = self.analyze_cross_market_patterns()
        
        # 6. Detect temporal patterns
        logger.info("Analyzing temporal patterns...")
        temporal_patterns = self.analyze_temporal_patterns()
        
        # 7. Classify algorithms
        logger.info("Classifying algorithmic strategies...")
        self.classify_algorithms()
        
        # 8. Generate competitive assessment
        logger.info("Assessing competitive landscape...")
        competitive_assessment = self.assess_competition()
        
        # 9. Generate strategy recommendations
        logger.info("Generating strategy recommendations...")
        strategy_recommendations = self.generate_strategy_recommendations()
        
        return {
            "session_id": getattr(self.session_data, 'session_id', 'unknown'),
            "analysis_timestamp": datetime.now().isoformat(),
            "market_profiles": {k: asdict(v) for k, v in self.market_profiles.items()},
            "timing_patterns": timing_patterns,
            "size_patterns": size_patterns,
            "spread_patterns": spread_patterns,
            "cross_market_patterns": cross_market_patterns,
            "temporal_patterns": temporal_patterns,
            "detected_algorithms": [asdict(sig) for sig in self.algo_signatures],
            "competitive_assessment": competitive_assessment,
            "strategy_recommendations": strategy_recommendations,
            "summary_stats": self.generate_summary_stats()
        }
    
    def build_market_profiles(self) -> None:
        """Build detailed profiles for each market"""
        
        # Group data by market
        market_data = defaultdict(list)
        
        # Process session data to extract market-specific information
        for idx, data_point in enumerate(self.session_data):
            # data_point is a SessionDataPoint object
            timestamp_ms = data_point.timestamp_ms
            markets_data_dict = data_point.markets_data
            
            # Extract market data from each market in this data point
            for market_ticker, market_orderbook_data in markets_data_dict.items():
                market_data[market_ticker].append({
                    'timestamp': timestamp_ms,
                    'yes_spread': market_orderbook_data.get('yes_spread'),
                    'no_spread': market_orderbook_data.get('no_spread'),
                    'yes_bids': market_orderbook_data.get('yes_bids', {}),
                    'yes_asks': market_orderbook_data.get('yes_asks', {}),
                    'no_bids': market_orderbook_data.get('no_bids', {}),
                    'no_asks': market_orderbook_data.get('no_asks', {}),
                    'sequence': market_orderbook_data.get('last_sequence', 0)
                })
        
        logger.info(f"Building profiles for {len(market_data)} markets")
        
        # Build profile for each market
        for market_ticker, data_points in market_data.items():
            if len(data_points) < 10:  # Skip markets with too little data
                continue
                
            profile = self.build_single_market_profile(market_ticker, data_points)
            self.market_profiles[market_ticker] = profile
            
        logger.info(f"Built {len(self.market_profiles)} market profiles")
    
    def build_single_market_profile(self, market_ticker: str, data_points: List[Dict]) -> MarketProfile:
        """Build detailed profile for a single market"""
        
        # Extract spreads
        yes_spreads = [dp['yes_spread'] for dp in data_points if dp['yes_spread'] is not None]
        no_spreads = [dp['no_spread'] for dp in data_points if dp['no_spread'] is not None]
        
        # Calculate spread statistics
        avg_spread_yes = np.mean(yes_spreads) if yes_spreads else 0
        avg_spread_no = np.mean(no_spreads) if no_spreads else 0
        spread_volatility = np.std(yes_spreads + no_spreads) if (yes_spreads or no_spreads) else 0
        
        # Analyze order sizes
        order_sizes = []
        for dp in data_points:
            for book in [dp['yes_bids'], dp['yes_asks'], dp['no_bids'], dp['no_asks']]:
                if book:
                    order_sizes.extend(book.values())
        
        size_distribution = Counter(order_sizes)
        
        # Calculate response time statistics (time between sequence numbers)
        response_times = []
        timestamps = [dp['timestamp'] for dp in data_points if dp['timestamp']]
        
        if len(timestamps) > 1:
            response_times = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
            response_times = [rt for rt in response_times if rt > 0 and rt < 300000]  # Filter outliers (5 min)
        
        response_stats = {}
        if response_times:
            response_stats = {
                'median_ms': np.median(response_times),
                'p95_ms': np.percentile(response_times, 95),
                'p99_ms': np.percentile(response_times, 99),
                'min_ms': np.min(response_times),
                'max_ms': np.max(response_times)
            }
        
        # Calculate update frequency
        total_time_ms = max(timestamps) - min(timestamps) if len(timestamps) > 1 else 1
        update_frequency = len(data_points) / (total_time_ms / 1000.0) if total_time_ms > 0 else 0
        
        # Calculate flickering ratio (orders that appear and disappear quickly)
        flickering_events = self.detect_flickering_orders(data_points)
        flickering_ratio = len(flickering_events) / len(data_points) if data_points else 0
        
        # Calculate round number bias
        round_number_bias = self.calculate_round_number_bias(data_points)
        
        return MarketProfile(
            market_ticker=market_ticker,
            avg_spread_yes=avg_spread_yes,
            avg_spread_no=avg_spread_no,
            spread_volatility=spread_volatility,
            order_size_distribution=dict(size_distribution),
            response_time_stats=response_stats,
            update_frequency=update_frequency,
            flickering_ratio=flickering_ratio,
            round_number_bias=round_number_bias
        )
    
    def detect_flickering_orders(self, data_points: List[Dict]) -> List[Dict]:
        """Detect orders that appear and disappear rapidly (bot signature)"""
        flickering_events = []
        
        # Track price levels over time
        price_level_history = defaultdict(list)
        
        for i, dp in enumerate(data_points):
            timestamp = dp.get('timestamp', 0)
            
            # Track all price levels
            for side in ['yes_bids', 'yes_asks', 'no_bids', 'no_asks']:
                book = dp.get(side, {})
                for price, size in book.items():
                    price_level_history[(side, price)].append({
                        'timestamp': timestamp,
                        'size': size,
                        'index': i
                    })
        
        # Look for rapid add/remove patterns
        for (side, price), history in price_level_history.items():
            if len(history) < 3:
                continue
                
            # Look for patterns: size > 0, size = 0, size > 0 within short timeframe
            for i in range(len(history) - 2):
                event1, event2, event3 = history[i:i+3]
                
                # Pattern: add -> remove -> add within 5 seconds
                if (event1['size'] > 0 and event2['size'] == 0 and event3['size'] > 0 and
                    event3['timestamp'] - event1['timestamp'] < 5000):
                    
                    flickering_events.append({
                        'side': side,
                        'price': price,
                        'start_time': event1['timestamp'],
                        'duration_ms': event3['timestamp'] - event1['timestamp'],
                        'pattern': 'add_remove_add'
                    })
        
        return flickering_events
    
    def calculate_round_number_bias(self, data_points: List[Dict]) -> float:
        """Calculate preference for round number prices (human vs algo indicator)"""
        all_prices = []
        
        for dp in data_points:
            for book in [dp['yes_bids'], dp['yes_asks'], dp['no_bids'], dp['no_asks']]:
                if book:
                    all_prices.extend(book.keys())
        
        if not all_prices:
            return 0.0
        
        # Count round numbers (multiples of 5 or 10)
        round_5 = sum(1 for p in all_prices if p % 5 == 0)
        round_10 = sum(1 for p in all_prices if p % 10 == 0)
        
        # Expected frequency if random: 20% for mod 5, 10% for mod 10
        expected_round_5 = len(all_prices) * 0.2
        expected_round_10 = len(all_prices) * 0.1
        
        # Calculate bias (higher means more human-like, lower means more algorithmic)
        bias_5 = round_5 / expected_round_5 if expected_round_5 > 0 else 1.0
        bias_10 = round_10 / expected_round_10 if expected_round_10 > 0 else 1.0
        
        return (bias_5 + bias_10) / 2
    
    def analyze_response_times(self) -> Dict[str, Any]:
        """Analyze response time patterns across markets"""
        
        all_response_times = []
        ultra_fast_responses = []  # < 100ms
        synchronized_responses = []  # Multiple markets updating simultaneously
        
        # Collect response times across all markets
        for market_ticker, profile in self.market_profiles.items():
            response_stats = profile.response_time_stats
            if 'median_ms' in response_stats:
                all_response_times.append(response_stats['median_ms'])
                
                # Flag ultra-fast responses (likely algorithmic)
                if response_stats['min_ms'] < 100:  # < 100ms
                    ultra_fast_responses.append({
                        'market': market_ticker,
                        'min_response_ms': response_stats['min_ms'],
                        'median_response_ms': response_stats['median_ms']
                    })
        
        # Analyze distribution
        timing_analysis = {
            'overall_median_response_ms': np.median(all_response_times) if all_response_times else 0,
            'ultra_fast_markets': len(ultra_fast_responses),
            'ultra_fast_details': ultra_fast_responses,
            'response_time_distribution': {
                'p50': np.percentile(all_response_times, 50) if all_response_times else 0,
                'p95': np.percentile(all_response_times, 95) if all_response_times else 0,
                'p99': np.percentile(all_response_times, 99) if all_response_times else 0
            }
        }
        
        # Flag markets with suspiciously fast response times (likely bots)
        algo_suspected_markets = [
            market for market in ultra_fast_responses 
            if market['median_response_ms'] < 1000  # < 1 second median
        ]
        
        timing_analysis['algo_suspected_markets'] = algo_suspected_markets
        timing_analysis['algo_penetration_rate'] = len(algo_suspected_markets) / len(self.market_profiles) if self.market_profiles else 0
        
        return timing_analysis
    
    def analyze_order_sizes(self) -> Dict[str, Any]:
        """Analyze order size patterns for algorithmic signatures"""
        
        # Aggregate order sizes across all markets
        all_order_sizes = []
        market_size_patterns = {}
        
        for market_ticker, profile in self.market_profiles.items():
            sizes = list(profile.order_size_distribution.keys())
            counts = list(profile.order_size_distribution.values())
            
            # Weight by frequency
            weighted_sizes = []
            for size, count in zip(sizes, counts):
                weighted_sizes.extend([size] * count)
            
            all_order_sizes.extend(weighted_sizes)
            
            # Analyze this market's size patterns
            if sizes:
                # Check for systematic sizing (multiples, powers of 2, etc.)
                systematic_patterns = self.detect_systematic_sizing(sizes, counts)
                market_size_patterns[market_ticker] = systematic_patterns
        
        # Overall analysis
        size_analysis = {
            'total_orders_analyzed': len(all_order_sizes),
            'unique_order_sizes': len(set(all_order_sizes)),
            'most_common_sizes': Counter(all_order_sizes).most_common(10),
            'size_statistics': {
                'median': np.median(all_order_sizes) if all_order_sizes else 0,
                'mean': np.mean(all_order_sizes) if all_order_sizes else 0,
                'std': np.std(all_order_sizes) if all_order_sizes else 0
            },
            'market_patterns': market_size_patterns
        }
        
        # Detect algorithmic sizing patterns
        algo_size_indicators = self.detect_algo_size_patterns(all_order_sizes)
        size_analysis['algorithmic_indicators'] = algo_size_indicators
        
        return size_analysis
    
    def detect_systematic_sizing(self, sizes: List[int], counts: List[int]) -> Dict[str, Any]:
        """Detect systematic order sizing patterns (algo signature)"""
        
        if not sizes:
            return {}
        
        total_orders = sum(counts)
        
        # Check for power-of-2 bias (algorithmic)
        powers_of_2 = [s for s in sizes if s > 0 and (s & (s-1)) == 0]  # Powers of 2
        power_of_2_ratio = len(powers_of_2) / len(sizes) if sizes else 0
        
        # Check for round number bias (100s, 1000s)
        round_hundreds = [s for s in sizes if s % 100 == 0]
        round_thousands = [s for s in sizes if s % 1000 == 0]
        
        # Check for Fibonacci-like sequences (some algos use these)
        fib_like = self.check_fibonacci_pattern(sizes)
        
        # Check for very precise sizes (unlikely to be human)
        precise_sizes = [s for s in sizes if s > 1000 and s % 10 != 0 and s % 5 != 0]
        precision_ratio = len(precise_sizes) / len(sizes) if sizes else 0
        
        return {
            'power_of_2_ratio': power_of_2_ratio,
            'round_number_bias': len(round_hundreds) / len(sizes) if sizes else 0,
            'thousand_multiple_bias': len(round_thousands) / len(sizes) if sizes else 0,
            'fibonacci_pattern_detected': fib_like,
            'precision_ratio': precision_ratio,
            'size_entropy': self.calculate_entropy([c/total_orders for c in counts]),
            'algorithmic_score': power_of_2_ratio + precision_ratio + (1 - fib_like)
        }
    
    def check_fibonacci_pattern(self, sizes: List[int]) -> bool:
        """Check if sizes follow Fibonacci-like progression"""
        if len(sizes) < 3:
            return False
        
        sorted_sizes = sorted(set(sizes))
        
        # Check if consecutive sizes approximate Fibonacci ratios
        fibonacci_like_count = 0
        for i in range(len(sorted_sizes) - 1):
            if sorted_sizes[i] > 0:
                ratio = sorted_sizes[i+1] / sorted_sizes[i]
                # Golden ratio ~1.618, allow 10% tolerance
                if 1.45 <= ratio <= 1.78:
                    fibonacci_like_count += 1
        
        # If more than 30% of ratios are Fibonacci-like, flag it
        return fibonacci_like_count / max(len(sorted_sizes) - 1, 1) > 0.3
    
    def calculate_entropy(self, probabilities: List[float]) -> float:
        """Calculate Shannon entropy of a distribution"""
        entropy = 0
        for p in probabilities:
            if p > 0:
                entropy -= p * np.log2(p)
        return entropy
    
    def detect_algo_size_patterns(self, all_order_sizes: List[int]) -> Dict[str, Any]:
        """Detect algorithmic patterns in order sizes"""
        
        if not all_order_sizes:
            return {}
        
        size_counter = Counter(all_order_sizes)
        
        # Look for suspiciously uniform distributions (algo signature)
        size_frequencies = list(size_counter.values())
        uniformity_score = 1 - (np.std(size_frequencies) / np.mean(size_frequencies)) if size_frequencies else 0
        
        # Look for concentration in specific size ranges
        small_orders = sum(1 for s in all_order_sizes if s <= 100)
        medium_orders = sum(1 for s in all_order_sizes if 100 < s <= 1000) 
        large_orders = sum(1 for s in all_order_sizes if s > 1000)
        
        total = len(all_order_sizes)
        
        return {
            'uniformity_score': uniformity_score,  # Higher = more algorithmic
            'size_concentration': {
                'small_orders_pct': small_orders / total * 100,
                'medium_orders_pct': medium_orders / total * 100,
                'large_orders_pct': large_orders / total * 100
            },
            'dominant_size': size_counter.most_common(1)[0] if size_counter else None,
            'size_diversity': len(set(all_order_sizes)),
            'algorithmic_likelihood': uniformity_score * 0.6 + (1 - len(set(all_order_sizes)) / len(all_order_sizes)) * 0.4
        }
    
    def analyze_spread_patterns(self) -> Dict[str, Any]:
        """Analyze spread behavior for algorithmic signatures"""
        
        spread_patterns = {
            'constant_spread_markets': [],
            'spread_compression_events': [],
            'abnormal_spread_markets': [],
            'spread_statistics': {}
        }
        
        all_spreads_yes = []
        all_spreads_no = []
        
        for market_ticker, profile in self.market_profiles.items():
            all_spreads_yes.append(profile.avg_spread_yes)
            all_spreads_no.append(profile.avg_spread_no)
            
            # Detect constant spreads (market maker bots)
            if profile.spread_volatility < 0.5:  # Very low volatility
                spread_patterns['constant_spread_markets'].append({
                    'market': market_ticker,
                    'avg_spread_yes': profile.avg_spread_yes,
                    'avg_spread_no': profile.avg_spread_no,
                    'volatility': profile.spread_volatility
                })
            
            # Detect abnormally wide/narrow spreads
            if profile.avg_spread_yes < 1 or profile.avg_spread_no < 1:
                spread_patterns['abnormal_spread_markets'].append({
                    'market': market_ticker,
                    'type': 'abnormally_narrow',
                    'yes_spread': profile.avg_spread_yes,
                    'no_spread': profile.avg_spread_no
                })
            elif profile.avg_spread_yes > 20 or profile.avg_spread_no > 20:
                spread_patterns['abnormal_spread_markets'].append({
                    'market': market_ticker,
                    'type': 'abnormally_wide',
                    'yes_spread': profile.avg_spread_yes,
                    'no_spread': profile.avg_spread_no
                })
        
        # Overall spread statistics
        spread_patterns['spread_statistics'] = {
            'median_yes_spread': np.median([s for s in all_spreads_yes if s > 0]),
            'median_no_spread': np.median([s for s in all_spreads_no if s > 0]),
            'spread_range_yes': {
                'min': min(s for s in all_spreads_yes if s > 0) if all_spreads_yes else 0,
                'max': max(all_spreads_yes) if all_spreads_yes else 0
            },
            'spread_range_no': {
                'min': min(s for s in all_spreads_no if s > 0) if all_spreads_no else 0,
                'max': max(all_spreads_no) if all_spreads_no else 0
            }
        }
        
        # Market maker detection
        market_maker_candidates = [
            market for market in spread_patterns['constant_spread_markets']
            if market['volatility'] < 0.2 and 1 <= market['avg_spread_yes'] <= 3
        ]
        
        spread_patterns['market_maker_candidates'] = market_maker_candidates
        spread_patterns['market_maker_penetration'] = len(market_maker_candidates) / len(self.market_profiles) if self.market_profiles else 0
        
        return spread_patterns
    
    def analyze_cross_market_patterns(self) -> Dict[str, Any]:
        """Analyze patterns that suggest the same algorithm operates across multiple markets"""
        
        cross_market = {
            'correlation_clusters': [],
            'synchronized_updates': [],
            'common_characteristics': {}
        }
        
        # Group markets by similar characteristics
        characteristic_groups = defaultdict(list)
        
        for market_ticker, profile in self.market_profiles.items():
            # Create signature based on key characteristics
            signature = (
                round(profile.update_frequency, 1),
                round(profile.flickering_ratio, 2),
                round(profile.round_number_bias, 2)
            )
            characteristic_groups[signature].append(market_ticker)
        
        # Find groups with multiple markets (potential same-bot operation)
        for signature, markets in characteristic_groups.items():
            if len(markets) > 1:
                cross_market['correlation_clusters'].append({
                    'signature': signature,
                    'markets': markets,
                    'market_count': len(markets),
                    'characteristics': {
                        'update_frequency': signature[0],
                        'flickering_ratio': signature[1],
                        'round_number_bias': signature[2]
                    }
                })
        
        # Sort by market count to find largest bot operations
        cross_market['correlation_clusters'].sort(key=lambda x: x['market_count'], reverse=True)
        
        return cross_market
    
    def analyze_temporal_patterns(self) -> Dict[str, Any]:
        """Analyze time-based patterns in trading activity"""
        
        # This is a simplified version - would need timestamp analysis for full implementation
        temporal = {
            'activity_patterns': {},
            'maintenance_windows': [],
            'synchronized_activity': []
        }
        
        # Analyze update frequency patterns across markets
        update_frequencies = [profile.update_frequency for profile in self.market_profiles.values()]
        
        temporal['activity_patterns'] = {
            'avg_update_frequency': np.mean(update_frequencies) if update_frequencies else 0,
            'update_frequency_std': np.std(update_frequencies) if update_frequencies else 0,
            'high_activity_markets': [
                market for market, profile in self.market_profiles.items()
                if profile.update_frequency > np.mean(update_frequencies) + 2 * np.std(update_frequencies)
            ] if update_frequencies and np.std(update_frequencies) > 0 else []
        }
        
        return temporal
    
    def classify_algorithms(self) -> None:
        """Classify detected algorithms into strategy types"""
        
        self.algo_signatures = []
        
        # Market Maker Detection
        market_makers = []
        for market_ticker, profile in self.market_profiles.items():
            # Market maker signals: low spread volatility, frequent updates, round number bias
            if (profile.spread_volatility < 1.0 and 
                profile.update_frequency > 0.1 and 
                profile.round_number_bias > 1.0 and
                1 <= profile.avg_spread_yes <= 5):
                
                market_makers.append(market_ticker)
        
        if market_makers:
            self.algo_signatures.append(AlgoSignature(
                strategy_type="market_maker",
                confidence=0.8,
                evidence=[
                    "Low spread volatility",
                    "Consistent spread maintenance", 
                    "High update frequency",
                    "Round number price preference"
                ],
                markets=market_makers,
                time_active=[],  # Would need temporal analysis
                characteristics={
                    "avg_spread_maintained": np.mean([self.market_profiles[m].avg_spread_yes for m in market_makers]),
                    "update_frequency": np.mean([self.market_profiles[m].update_frequency for m in market_makers])
                }
            ))
        
        # High Frequency Trading Detection
        hft_markets = []
        for market_ticker, profile in self.market_profiles.items():
            # HFT signals: very fast responses, high update frequency, low round number bias
            response_stats = profile.response_time_stats
            if (response_stats.get('median_ms', float('inf')) < 500 and
                profile.update_frequency > 1.0 and
                profile.round_number_bias < 0.8):
                
                hft_markets.append(market_ticker)
        
        if hft_markets:
            self.algo_signatures.append(AlgoSignature(
                strategy_type="high_frequency_trading",
                confidence=0.9,
                evidence=[
                    "Ultra-fast response times (<500ms median)",
                    "Very high update frequency",
                    "Non-human order sizing",
                    "Low round number bias"
                ],
                markets=hft_markets,
                time_active=[],
                characteristics={
                    "median_response_time_ms": np.mean([
                        self.market_profiles[m].response_time_stats.get('median_ms', 0) 
                        for m in hft_markets
                    ]),
                    "avg_update_frequency": np.mean([self.market_profiles[m].update_frequency for m in hft_markets])
                }
            ))
        
        # Arbitrage Bot Detection
        arbitrage_markets = []
        for market_ticker, profile in self.market_profiles.items():
            # Arbitrage signals: very fast responses, flickering orders, precise sizing
            response_stats = profile.response_time_stats
            if (response_stats.get('min_ms', float('inf')) < 100 and
                profile.flickering_ratio > 0.1 and
                profile.round_number_bias < 0.5):
                
                arbitrage_markets.append(market_ticker)
        
        if arbitrage_markets:
            self.algo_signatures.append(AlgoSignature(
                strategy_type="arbitrageur",
                confidence=0.75,
                evidence=[
                    "Ultra-fast minimum response times (<100ms)",
                    "High flickering ratio (rapid order changes)",
                    "Precise, non-round order sizes",
                    "Low round number bias"
                ],
                markets=arbitrage_markets,
                time_active=[],
                characteristics={
                    "min_response_time_ms": np.mean([
                        self.market_profiles[m].response_time_stats.get('min_ms', 0) 
                        for m in arbitrage_markets
                    ]),
                    "avg_flickering_ratio": np.mean([self.market_profiles[m].flickering_ratio for m in arbitrage_markets])
                }
            ))
    
    def assess_competition(self) -> Dict[str, Any]:
        """Assess the competitive algorithmic trading landscape"""
        
        total_markets = len(self.market_profiles)
        if total_markets == 0:
            return {"error": "No markets to analyze"}
        
        # Count markets with algorithmic activity
        markets_with_algos = set()
        for signature in self.algo_signatures:
            markets_with_algos.update(signature.markets)
        
        algo_penetration = len(markets_with_algos) / total_markets
        
        # Categorize markets by competition level
        high_competition = []
        medium_competition = []
        low_competition = []
        
        for market_ticker, profile in self.market_profiles.items():
            # Count how many different algo types detected in this market
            algo_count = sum(1 for sig in self.algo_signatures if market_ticker in sig.markets)
            
            if algo_count >= 2:
                high_competition.append({
                    'market': market_ticker,
                    'algo_count': algo_count,
                    'update_frequency': profile.update_frequency
                })
            elif algo_count == 1:
                medium_competition.append({
                    'market': market_ticker,
                    'algo_count': algo_count,
                    'update_frequency': profile.update_frequency
                })
            else:
                low_competition.append({
                    'market': market_ticker,
                    'algo_count': algo_count,
                    'update_frequency': profile.update_frequency
                })
        
        return {
            "algorithmic_penetration_rate": algo_penetration,
            "total_markets_analyzed": total_markets,
            "markets_with_algos": len(markets_with_algos),
            "competition_levels": {
                "high_competition": {
                    "count": len(high_competition),
                    "markets": high_competition[:10]  # Top 10
                },
                "medium_competition": {
                    "count": len(medium_competition),
                    "markets": medium_competition[:10]
                },
                "low_competition": {
                    "count": len(low_competition),
                    "markets": low_competition[:10]
                }
            },
            "algorithm_dominance": {
                sig.strategy_type: len(sig.markets) for sig in self.algo_signatures
            }
        }
    
    def generate_strategy_recommendations(self) -> Dict[str, Any]:
        """Generate concrete strategy recommendations based on analysis"""
        
        competition_assessment = self.assess_competition()
        
        recommendations = {
            "optimal_strategy": {},
            "market_selection": {},
            "execution_tactics": {},
            "risk_considerations": {}
        }
        
        # Determine optimal strategy based on competitive landscape
        algo_penetration = competition_assessment["algorithmic_penetration_rate"]
        low_competition_markets = competition_assessment["competition_levels"]["low_competition"]["count"]
        
        if algo_penetration > 0.7:  # Highly saturated
            recommendations["optimal_strategy"] = {
                "primary_approach": "niche_exploitation",
                "description": "Market is highly algorithmic. Focus on niches bots avoid.",
                "tactics": [
                    "Target low-competition markets exclusively",
                    "Use longer-term strategies (>1 hour holds)", 
                    "Focus on markets with low update frequency",
                    "Implement position-based strategies bots can't easily replicate"
                ]
            }
        elif algo_penetration > 0.3:  # Moderately saturated
            recommendations["optimal_strategy"] = {
                "primary_approach": "selective_competition",
                "description": "Mixed environment. Compete where possible, avoid where dominated.",
                "tactics": [
                    "Compete in medium-competition markets with better execution",
                    "Use adaptive spread strategies",
                    "Implement bot detection and response logic",
                    "Focus on markets with human-like characteristics"
                ]
            }
        else:  # Low saturation
            recommendations["optimal_strategy"] = {
                "primary_approach": "aggressive_expansion", 
                "description": "Low algorithmic penetration. Opportunity for broad market making.",
                "tactics": [
                    "Implement broad market making across many markets",
                    "Use conservative spreads to establish market share",
                    "Build fast execution to compete with emerging bots",
                    "Focus on volume capture while competition is low"
                ]
            }
        
        # Market selection recommendations
        low_comp_markets = competition_assessment["competition_levels"]["low_competition"]["markets"]
        medium_comp_markets = competition_assessment["competition_levels"]["medium_competition"]["markets"]
        
        recommendations["market_selection"] = {
            "primary_targets": [m["market"] for m in low_comp_markets[:5]],
            "secondary_targets": [m["market"] for m in medium_comp_markets[:3]],
            "avoid_markets": [
                m["market"] for m in competition_assessment["competition_levels"]["high_competition"]["markets"][:5]
            ],
            "selection_criteria": [
                "Update frequency < 0.5 (less bot activity)",
                "Round number bias > 1.0 (more human trading)",
                "Spread volatility > 1.0 (less systematic market making)"
            ]
        }
        
        # Execution tactics based on detected bot behaviors
        execution_tactics = []
        
        # Counter market maker bots
        if any(sig.strategy_type == "market_maker" for sig in self.algo_signatures):
            execution_tactics.append({
                "target": "market_maker_bots",
                "tactic": "penny_jumping",
                "description": "Place orders 1¢ better than bot-maintained spreads",
                "implementation": "Monitor for constant spreads, improve by 1¢ when detected"
            })
        
        # Counter HFT bots
        if any(sig.strategy_type == "high_frequency_trading" for sig in self.algo_signatures):
            execution_tactics.append({
                "target": "hft_bots", 
                "tactic": "time_delay_exploitation",
                "description": "Use longer time horizons where HFT can't compete",
                "implementation": "Hold positions >5 minutes, focus on trend following"
            })
        
        # Counter arbitrage bots
        if any(sig.strategy_type == "arbitrageur" for sig in self.algo_signatures):
            execution_tactics.append({
                "target": "arbitrage_bots",
                "tactic": "false_signal_generation",
                "description": "Create apparent arbitrage opportunities that disappear",
                "implementation": "Brief spread widening to trigger bot responses, then capture reversion"
            })
        
        recommendations["execution_tactics"] = execution_tactics
        
        # Risk considerations
        recommendations["risk_considerations"] = {
            "bot_evolution_risk": "Algorithms may adapt to our strategies over time",
            "latency_requirements": "Need <1 second execution to compete with detected HFT",
            "capital_efficiency": f"Focus on {low_competition_markets} markets to maximize capital efficiency",
            "monitoring_requirements": [
                "Continuous bot detection and classification",
                "Response time monitoring",
                "Spread pattern analysis", 
                "Cross-market correlation tracking"
            ]
        }
        
        return recommendations
    
    def generate_summary_stats(self) -> Dict[str, Any]:
        """Generate high-level summary statistics"""
        
        if not self.market_profiles:
            return {"error": "No market profiles available"}
        
        # Overall statistics
        all_update_freq = [p.update_frequency for p in self.market_profiles.values()]
        all_spread_vol = [p.spread_volatility for p in self.market_profiles.values()]
        all_flickering = [p.flickering_ratio for p in self.market_profiles.values()]
        
        return {
            "total_markets_analyzed": len(self.market_profiles),
            "algorithms_detected": len(self.algo_signatures),
            "algorithm_types": [sig.strategy_type for sig in self.algo_signatures],
            "market_characteristics": {
                "avg_update_frequency": np.mean(all_update_freq),
                "avg_spread_volatility": np.mean(all_spread_vol),
                "avg_flickering_ratio": np.mean(all_flickering),
                "most_active_market": max(self.market_profiles.items(), key=lambda x: x[1].update_frequency)[0] if self.market_profiles else None
            },
            "competitive_intensity": len([
                market for market, profile in self.market_profiles.items()
                if sum(1 for sig in self.algo_signatures if market in sig.markets) >= 2
            ]) / len(self.market_profiles) if self.market_profiles else 0
        }


def main():
    """Main analysis function"""
    import sys
    import os
    
    # Add the src directory to the Python path
    current_dir = Path(__file__).parent
    src_dir = current_dir.parent
    sys.path.insert(0, str(src_dir))
    
    from kalshiflow_rl.environments.session_data_loader import SessionDataLoader
    
    if len(sys.argv) < 2:
        print("Usage: python algo_trading_analysis.py <session_id>")
        print("Example: python algo_trading_analysis.py 72")
        sys.exit(1)
    
    session_id = int(sys.argv[1])
    
    # Load session data
    print(f"Loading session {session_id}...")
    
    try:
        # Get database URL from environment
        import os
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            print("❌ DATABASE_URL not set in environment")
            sys.exit(1)
        
        loader = SessionDataLoader(database_url=database_url)
        session_data = asyncio.run(loader.load_session(session_id))
        if not session_data:
            print(f"No data found for session {session_id}")
            sys.exit(1)
        
        print(f"Loaded {len(session_data.data_points)} data points from session {session_data.session_id}")
        
        # Run analysis
        detector = AlgoTradingDetector(session_data.data_points)
        results = detector.analyze_session()
        
        # Save results
        output_file = f"/Users/samuelclark/Desktop/kalshiflow/backend/algo_analysis_session_{session_id}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Analysis complete. Results saved to {output_file}")
        
        # Print summary
        summary = results.get('summary_stats', {})
        print(f"\n=== SUMMARY ===")
        print(f"Markets analyzed: {summary.get('total_markets_analyzed', 0)}")
        print(f"Algorithms detected: {summary.get('algorithms_detected', 0)}")
        print(f"Algorithm types: {', '.join(summary.get('algorithm_types', []))}")
        print(f"Competitive intensity: {summary.get('competitive_intensity', 0):.1%}")
        
        # Print key recommendations  
        strategy = results.get('strategy_recommendations', {}).get('optimal_strategy', {})
        print(f"\nRecommended strategy: {strategy.get('primary_approach', 'Unknown')}")
        print(f"Description: {strategy.get('description', 'No description')}")
        
    except Exception as e:
        print(f"Error analyzing session {session_id}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()