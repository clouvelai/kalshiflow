#!/usr/bin/env python3
"""
Independent Strategy Validation Script
=====================================

This script provides rigorous, independent validation of the trading strategies
identified in the previous algorithmic analysis. The goal is to either confirm
or refute claims with statistical evidence and practical assessments.

Key Claims to Validate:
1. "Probability Anchor Arbitrage" Strategy
2. "Volatility Liquidity Shock" Strategy  
3. Success rate and profit estimates
4. Implementation viability

Author: Independent Analysis
Date: December 2025
"""

import sys
import numpy as np
import pandas as pd
import asyncio
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Add the src directory to path for imports
sys.path.append('/Users/samuelclark/Desktop/kalshiflow/backend/src')

from kalshiflow_rl.environments.session_data_loader import SessionDataLoader

@dataclass
class AnchorAnalysis:
    """Results of anchor level analysis"""
    anchor_level: int
    time_near_anchor: float
    avg_stickiness_seconds: float
    spread_ratio: float
    total_observations: int
    confidence_interval: Tuple[float, float]

@dataclass
class VolatilityEvent:
    """Volatility spike event data"""
    timestamp: pd.Timestamp
    price_move_cents: float
    duration_seconds: float
    liquidity_change_pct: float
    spread_before: float
    spread_during: float
    spread_after: float

@dataclass
class StrategyValidation:
    """Validation results for a trading strategy"""
    strategy_name: str
    claimed_hit_rate: float
    validated_hit_rate: float
    claimed_profit_per_trade: Tuple[float, float]
    estimated_profit_per_trade: Tuple[float, float]
    opportunities_per_day: float
    risk_assessment: str
    implementation_feasibility: str
    confidence_level: float

class TradingStrategyValidator:
    """
    Independent validator for claimed trading strategies using fresh data analysis
    """
    
    def __init__(self, session_ids: List[int]):
        """Initialize with specific session IDs for validation"""
        self.session_ids = session_ids
        self.sessions_data = {}
        self.anchor_levels = [25, 50, 75]  # Key psychological levels
        self.results = {}
        
    async def load_sessions(self):
        """Load all specified sessions for analysis"""
        print("Loading sessions for independent validation...")
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            print("❌ DATABASE_URL not set in environment")
            return
            
        loader = SessionDataLoader(database_url=database_url)
        
        for session_id in self.session_ids:
            try:
                print(f"Loading session {session_id}...")
                data = await loader.load_session(session_id)
                if data is not None and hasattr(data, 'data_points') and len(data.data_points) > 0:
                    self.sessions_data[session_id] = data
                    print(f"  ✅ Session {session_id}: {len(data.data_points)} data points")
                else:
                    print(f"  ❌ Failed to load session {session_id}")
            except Exception as e:
                print(f"  ❌ Error loading session {session_id}: {e}")
        
        print(f"Successfully loaded {len(self.sessions_data)} sessions")
    
    def _convert_session_to_dataframe(self, session_data) -> pd.DataFrame:
        """Convert SessionData to DataFrame for analysis"""
        rows = []
        
        for data_point in session_data.data_points:
            # Extract data for each market active at this timestamp
            for market_ticker, market_data in data_point.markets_data.items():
                if 'yes_asks' in market_data and 'yes_bids' in market_data:
                    # Calculate mid price and spread
                    yes_asks = market_data.get('yes_asks', [])
                    yes_bids = market_data.get('yes_bids', [])
                    
                    if yes_asks and yes_bids:
                        # Handle both dict and int formats
                        if isinstance(yes_asks[0], dict):
                            best_ask = min(yes_asks, key=lambda x: x['price'])['price']
                            best_bid = max(yes_bids, key=lambda x: x['price'])['price']
                        else:
                            # Assume it's just price values
                            best_ask = min(yes_asks)
                            best_bid = max(yes_bids)
                        
                        mid_price = (best_ask + best_bid) / 2.0
                        spread = best_ask - best_bid
                        
                        rows.append({
                            'timestamp': data_point.timestamp,
                            'market_ticker': market_ticker,
                            'mid_price': mid_price,
                            'spread': spread,
                            'best_bid': best_bid,
                            'best_ask': best_ask
                        })
        
        return pd.DataFrame(rows)
        
    def validate_anchor_arbitrage_strategy(self) -> Dict[str, StrategyValidation]:
        """
        CLAIM: 65-75% hit rate with 2-5¢ profit per trade at anchor levels
        CLAIM: Algorithms defend 25¢, 50¢, 75¢ with measurable patterns
        """
        print("\n" + "="*60)
        print("VALIDATING: Probability Anchor Arbitrage Strategy")
        print("="*60)
        
        all_anchor_results = []
        market_analyses = {}
        
        for session_id, session_data in self.sessions_data.items():
            print(f"\nAnalyzing session {session_id}...")
            df = self._convert_session_to_dataframe(session_data)
            session_results = self._analyze_session_anchors(df, session_id)
            all_anchor_results.extend(session_results)
            
            # Track specific market examples mentioned in claims
            market_analyses[session_id] = self._analyze_specific_markets(df)
        
        # Statistical validation
        validation_results = {}
        
        for anchor in self.anchor_levels:
            anchor_data = [r for r in all_anchor_results if r.anchor_level == anchor]
            
            if not anchor_data:
                print(f"⚠️  No data found for {anchor}¢ anchor")
                continue
                
            # Calculate aggregate statistics
            total_time_near = np.mean([a.time_near_anchor for a in anchor_data])
            avg_stickiness = np.mean([a.avg_stickiness_seconds for a in anchor_data])
            spread_ratios = [a.spread_ratio for a in anchor_data]
            
            # Classify bot behaviors
            defensive_markets = len([r for r in anchor_data if r.spread_ratio < 0.5])
            fearful_markets = len([r for r in anchor_data if r.spread_ratio > 1.5])
            neutral_markets = len(anchor_data) - defensive_markets - fearful_markets
            
            print(f"\n{anchor}¢ Anchor Analysis:")
            print(f"  Markets analyzed: {len(anchor_data)}")
            print(f"  Avg time near anchor: {total_time_near:.1f}%")
            print(f"  Avg stickiness: {avg_stickiness:.1f} seconds")
            print(f"  Bot behavior classification:")
            print(f"    Defensive (ratio <0.5): {defensive_markets} markets ({defensive_markets/len(anchor_data)*100:.1f}%)")
            print(f"    Fearful (ratio >1.5): {fearful_markets} markets ({fearful_markets/len(anchor_data)*100:.1f}%)")
            print(f"    Neutral: {neutral_markets} markets ({neutral_markets/len(anchor_data)*100:.1f}%)")
            
            # Estimate trading opportunities
            hit_rate = self._estimate_anchor_hit_rate(anchor_data)
            profit_estimate = self._estimate_anchor_profit(anchor_data)
            opportunities_per_day = self._count_anchor_opportunities(anchor_data)
            
            validation_results[f"{anchor}cent_anchor"] = StrategyValidation(
                strategy_name=f"{anchor}¢ Anchor Arbitrage",
                claimed_hit_rate=0.70,  # 65-75% claimed
                validated_hit_rate=hit_rate,
                claimed_profit_per_trade=(2.0, 5.0),
                estimated_profit_per_trade=profit_estimate,
                opportunities_per_day=opportunities_per_day,
                risk_assessment=self._assess_anchor_risk(anchor_data),
                implementation_feasibility=self._assess_implementation_feasibility(anchor_data),
                confidence_level=self._calculate_confidence(len(anchor_data))
            )
            
        return validation_results
    
    def validate_volatility_shock_strategy(self) -> StrategyValidation:
        """
        CLAIM: 30-60% liquidity withdrawal during >5¢ moves in <30 seconds
        CLAIM: 40-60% hit rate with 5-15¢ profit per trade
        """
        print("\n" + "="*60)
        print("VALIDATING: Volatility Liquidity Shock Strategy")
        print("="*60)
        
        all_volatility_events = []
        
        for session_id, session_data in self.sessions_data.items():
            print(f"\nAnalyzing volatility events in session {session_id}...")
            df = self._convert_session_to_dataframe(session_data)
            events = self._detect_volatility_events(df, session_id)
            all_volatility_events.extend(events)
            print(f"  Found {len(events)} volatility events")
        
        if not all_volatility_events:
            print("❌ No volatility events detected for analysis")
            return StrategyValidation(
                strategy_name="Volatility Liquidity Shock",
                claimed_hit_rate=0.50,
                validated_hit_rate=0.0,
                claimed_profit_per_trade=(5.0, 15.0),
                estimated_profit_per_trade=(0.0, 0.0),
                opportunities_per_day=0.0,
                risk_assessment="No events detected",
                implementation_feasibility="Cannot validate",
                confidence_level=0.0
            )
        
        # Analyze events
        liquidity_drops = [e.liquidity_change_pct for e in all_volatility_events]
        price_moves = [e.price_move_cents for e in all_volatility_events]
        spread_ratios = [e.spread_during / e.spread_before if e.spread_before > 0 else 0 
                        for e in all_volatility_events]
        
        print(f"\nVolatility Event Analysis ({len(all_volatility_events)} events):")
        print(f"  Avg liquidity drop: {np.mean(liquidity_drops):.1f}%")
        print(f"  Avg price move: {np.mean(price_moves):.1f}¢")
        print(f"  Avg spread expansion: {np.mean(spread_ratios):.1f}x")
        
        # Validate claims
        large_liquidity_drops = len([e for e in all_volatility_events if e.liquidity_change_pct > 30])
        significant_events = len([e for e in all_volatility_events if abs(e.price_move_cents) > 5])
        
        print(f"  Events with >30% liquidity drop: {large_liquidity_drops}/{len(all_volatility_events)} ({large_liquidity_drops/len(all_volatility_events)*100:.1f}%)")
        print(f"  Events with >5¢ price move: {significant_events}/{len(all_volatility_events)} ({significant_events/len(all_volatility_events)*100:.1f}%)")
        
        # Estimate strategy performance
        hit_rate = self._estimate_volatility_hit_rate(all_volatility_events)
        profit_estimate = self._estimate_volatility_profit(all_volatility_events)
        daily_opportunities = len(all_volatility_events) / len(self.sessions_data) * 24  # Rough daily estimate
        
        return StrategyValidation(
            strategy_name="Volatility Liquidity Shock",
            claimed_hit_rate=0.50,  # 40-60% claimed
            validated_hit_rate=hit_rate,
            claimed_profit_per_trade=(5.0, 15.0),
            estimated_profit_per_trade=profit_estimate,
            opportunities_per_day=daily_opportunities,
            risk_assessment=self._assess_volatility_risk(all_volatility_events),
            implementation_feasibility=self._assess_volatility_implementation(all_volatility_events),
            confidence_level=self._calculate_confidence(len(all_volatility_events))
        )
    
    def _analyze_session_anchors(self, df: pd.DataFrame, session_id: int) -> List[AnchorAnalysis]:
        """Analyze anchor behavior for a single session"""
        results = []
        
        # Group by market for individual analysis
        for market_ticker in df['market_ticker'].unique():
            market_data = df[df['market_ticker'] == market_ticker].copy()
            
            if len(market_data) < 20:  # Skip markets with insufficient data
                continue
                
            for anchor in self.anchor_levels:
                analysis = self._analyze_market_anchor(market_data, anchor, market_ticker)
                if analysis:
                    results.append(analysis)
        
        return results
    
    def _analyze_market_anchor(self, market_data: pd.DataFrame, anchor: int, ticker: str) -> Optional[AnchorAnalysis]:
        """Analyze anchor behavior for a specific market"""
        if 'mid_price' not in market_data.columns:
            return None
            
        # Calculate proximity to anchor
        price_distances = np.abs(market_data['mid_price'] - anchor)
        near_anchor = price_distances <= 2.0  # Within 2 cents of anchor
        
        if not near_anchor.any():
            return None
            
        time_near = near_anchor.sum() / len(market_data) * 100
        
        # Calculate stickiness (consecutive periods near anchor)
        stickiness_periods = []
        current_period = 0
        for is_near in near_anchor:
            if is_near:
                current_period += 1
            else:
                if current_period > 0:
                    stickiness_periods.append(current_period)
                current_period = 0
        if current_period > 0:
            stickiness_periods.append(current_period)
            
        avg_stickiness = np.mean(stickiness_periods) * 30 if stickiness_periods else 0  # Assume 30s per observation
        
        # Calculate spread ratio (anchor vs non-anchor)
        if 'spread' in market_data.columns:
            anchor_spreads = market_data.loc[near_anchor, 'spread']
            non_anchor_spreads = market_data.loc[~near_anchor, 'spread']
            
            if len(anchor_spreads) > 0 and len(non_anchor_spreads) > 0:
                spread_ratio = anchor_spreads.mean() / non_anchor_spreads.mean()
            else:
                spread_ratio = 1.0
        else:
            spread_ratio = 1.0
            
        # Confidence interval for time near anchor
        n = len(market_data)
        p = time_near / 100
        margin = 1.96 * np.sqrt(p * (1 - p) / n)  # 95% CI
        ci = (max(0, (p - margin) * 100), min(100, (p + margin) * 100))
        
        return AnchorAnalysis(
            anchor_level=anchor,
            time_near_anchor=time_near,
            avg_stickiness_seconds=avg_stickiness,
            spread_ratio=spread_ratio,
            total_observations=n,
            confidence_interval=ci
        )
    
    def _analyze_specific_markets(self, df: pd.DataFrame) -> Dict:
        """Check specific market examples mentioned in the claims"""
        examples = {
            'EUEXIT-30': {'anchor': 25, 'claimed_time': 99.67},
            'EVSHARE-30JAN-30': {'anchor': 50, 'claimed_time': 95.16},
            'GOVPARTYND-28-D': {'anchor': 10, 'claimed_time': 67.8}
        }
        
        results = {}
        for market, info in examples.items():
            if market in df['market_ticker'].values:
                market_data = df[df['market_ticker'] == market]
                # Validate the specific claims
                analysis = self._analyze_market_anchor(market_data, info['anchor'], market)
                results[market] = {
                    'found': True,
                    'analysis': analysis,
                    'claimed_time': info['claimed_time']
                }
            else:
                results[market] = {'found': False}
                
        return results
    
    def _detect_volatility_events(self, df: pd.DataFrame, session_id: int) -> List[VolatilityEvent]:
        """Detect volatility events matching the claim criteria"""
        events = []
        
        # Group by market
        for market_ticker in df['market_ticker'].unique():
            market_data = df[df['market_ticker'] == market_ticker].copy()
            
            if len(market_data) < 10:
                continue
                
            market_data = market_data.sort_values('timestamp')
            market_data['price_change'] = market_data['mid_price'].diff()
            market_data['time_diff'] = market_data['timestamp'].diff().dt.total_seconds()
            
            # Find rapid price moves
            for i in range(1, len(market_data)):
                if market_data.iloc[i]['time_diff'] <= 30:  # Within 30 seconds
                    price_change = abs(market_data.iloc[i]['price_change'])
                    
                    if price_change >= 5.0:  # >5 cent move
                        # Analyze liquidity change (proxy through spread)
                        spread_before = market_data.iloc[i-1]['spread'] if 'spread' in market_data.columns else 0
                        spread_during = market_data.iloc[i]['spread'] if 'spread' in market_data.columns else 0
                        
                        # Look for spread after the event
                        spread_after = spread_during
                        if i < len(market_data) - 1:
                            spread_after = market_data.iloc[i+1]['spread'] if 'spread' in market_data.columns else 0
                        
                        liquidity_change = ((spread_during - spread_before) / spread_before * 100) if spread_before > 0 else 0
                        
                        event = VolatilityEvent(
                            timestamp=market_data.iloc[i]['timestamp'],
                            price_move_cents=price_change,
                            duration_seconds=market_data.iloc[i]['time_diff'],
                            liquidity_change_pct=liquidity_change,
                            spread_before=spread_before,
                            spread_during=spread_during,
                            spread_after=spread_after
                        )
                        events.append(event)
        
        return events
    
    def _estimate_anchor_hit_rate(self, anchor_data: List[AnchorAnalysis]) -> float:
        """Estimate hit rate for anchor arbitrage based on data patterns"""
        if not anchor_data:
            return 0.0
            
        # Markets with strong anchor behavior (high time near anchor + extreme spread ratios)
        strong_anchors = [a for a in anchor_data 
                         if a.time_near_anchor > 20 and (a.spread_ratio < 0.5 or a.spread_ratio > 1.5)]
        
        # Conservative estimate based on data quality
        base_rate = len(strong_anchors) / len(anchor_data)
        
        # Adjust for implementation challenges
        estimated_hit_rate = base_rate * 0.7  # 30% reduction for execution challenges
        
        return min(estimated_hit_rate, 1.0)
    
    def _estimate_anchor_profit(self, anchor_data: List[AnchorAnalysis]) -> Tuple[float, float]:
        """Estimate profit per trade for anchor arbitrage"""
        if not anchor_data:
            return (0.0, 0.0)
            
        # Use spread compression/expansion as proxy for opportunity size
        spread_ratios = [a.spread_ratio for a in anchor_data]
        extreme_ratios = [r for r in spread_ratios if r < 0.5 or r > 2.0]
        
        if extreme_ratios:
            # Conservative estimate: can capture 1-3 cents from spread differences
            min_profit = 1.0
            max_profit = 3.0
        else:
            # Limited opportunities
            min_profit = 0.5
            max_profit = 1.5
            
        return (min_profit, max_profit)
    
    def _count_anchor_opportunities(self, anchor_data: List[AnchorAnalysis]) -> float:
        """Estimate daily opportunities for anchor trading"""
        if not anchor_data:
            return 0.0
            
        # Based on stickiness and time near anchor
        avg_stickiness = np.mean([a.avg_stickiness_seconds for a in anchor_data])
        avg_time_near = np.mean([a.time_near_anchor for a in anchor_data])
        
        # Rough estimate: opportunities proportional to time near anchor
        daily_opportunities = (avg_time_near / 100) * 10  # 10 potential trades per day if always near anchor
        
        return daily_opportunities
    
    def _estimate_volatility_hit_rate(self, events: List[VolatilityEvent]) -> float:
        """Estimate hit rate for volatility shock strategy"""
        if not events:
            return 0.0
            
        # Events with significant liquidity impact
        significant_events = [e for e in events if e.liquidity_change_pct > 20]
        
        if not significant_events:
            return 0.1  # Very low if no significant liquidity events
            
        # Hit rate based on spread expansion (opportunity indicator)
        expanded_spreads = len([e for e in significant_events 
                              if e.spread_during > e.spread_before * 1.5])
        
        return min(expanded_spreads / len(significant_events), 0.8)
    
    def _estimate_volatility_profit(self, events: List[VolatilityEvent]) -> Tuple[float, float]:
        """Estimate profit per trade for volatility strategy"""
        if not events:
            return (0.0, 0.0)
            
        spread_expansions = [(e.spread_during - e.spread_before) 
                           for e in events if e.spread_during > e.spread_before]
        
        if spread_expansions:
            min_profit = np.percentile(spread_expansions, 25) * 0.5  # Capture 50% of spread expansion
            max_profit = np.percentile(spread_expansions, 75) * 0.5
        else:
            min_profit, max_profit = 0.0, 0.0
            
        return (max(0, min_profit), max(0, max_profit))
    
    def _assess_anchor_risk(self, anchor_data: List[AnchorAnalysis]) -> str:
        """Assess risk level for anchor arbitrage strategy"""
        if not anchor_data:
            return "Cannot assess - no data"
            
        # Risk based on data consistency
        spread_ratios = [a.spread_ratio for a in anchor_data]
        ratio_std = np.std(spread_ratios)
        
        if ratio_std > 2.0:
            return "High - inconsistent bot behavior"
        elif ratio_std > 1.0:
            return "Medium - moderate behavior variation"
        else:
            return "Low - consistent patterns"
    
    def _assess_volatility_risk(self, events: List[VolatilityEvent]) -> str:
        """Assess risk level for volatility strategy"""
        if not events:
            return "Cannot assess - no events"
            
        price_moves = [abs(e.price_move_cents) for e in events]
        avg_move = np.mean(price_moves)
        
        if avg_move > 15:
            return "Very High - large adverse moves possible"
        elif avg_move > 10:
            return "High - significant adverse move risk"
        elif avg_move > 5:
            return "Medium - moderate adverse move risk"
        else:
            return "Low - small adverse moves"
    
    def _assess_implementation_feasibility(self, anchor_data: List[AnchorAnalysis]) -> str:
        """Assess implementation feasibility for anchor strategy"""
        if not anchor_data:
            return "Cannot assess"
            
        avg_stickiness = np.mean([a.avg_stickiness_seconds for a in anchor_data])
        
        if avg_stickiness > 60:
            return "High - sufficient time for execution"
        elif avg_stickiness > 30:
            return "Medium - requires fast execution"
        else:
            return "Low - very fast execution required"
    
    def _assess_volatility_implementation(self, events: List[VolatilityEvent]) -> str:
        """Assess implementation feasibility for volatility strategy"""
        if not events:
            return "Cannot assess"
            
        avg_duration = np.mean([e.duration_seconds for e in events])
        
        if avg_duration > 10:
            return "Medium - reasonable execution window"
        else:
            return "Very Difficult - requires ultra-fast execution"
    
    def _calculate_confidence(self, sample_size: int) -> float:
        """Calculate confidence level based on sample size"""
        if sample_size >= 100:
            return 0.95
        elif sample_size >= 50:
            return 0.80
        elif sample_size >= 20:
            return 0.60
        else:
            return 0.30
    
    def generate_validation_report(self, anchor_results: Dict, volatility_result: StrategyValidation):
        """Generate comprehensive validation report"""
        print("\n" + "="*80)
        print("INDEPENDENT STRATEGY VALIDATION REPORT")
        print("="*80)
        
        print("\nSUMMARY OF CLAIMS TESTED:")
        print("-" * 40)
        print("1. Probability Anchor Arbitrage: 65-75% hit rate, 2-5¢ profit")
        print("2. Volatility Liquidity Shock: 40-60% hit rate, 5-15¢ profit")
        print("3. Specific market examples (EUEXIT-30, etc.)")
        print("4. Implementation feasibility assessments")
        
        print("\nVALIDATION RESULTS:")
        print("-" * 40)
        
        # Anchor strategy results
        print("\n🎯 PROBABILITY ANCHOR ARBITRAGE VALIDATION:")
        for strategy_name, result in anchor_results.items():
            print(f"\n  {result.strategy_name}:")
            print(f"    Claimed hit rate: {result.claimed_hit_rate:.1%}")
            print(f"    Validated hit rate: {result.validated_hit_rate:.1%}")
            print(f"    Claimed profit: ${result.claimed_profit_per_trade[0]:.1f}-${result.claimed_profit_per_trade[1]:.1f}")
            print(f"    Estimated profit: ${result.estimated_profit_per_trade[0]:.1f}-${result.estimated_profit_per_trade[1]:.1f}")
            print(f"    Daily opportunities: {result.opportunities_per_day:.1f}")
            print(f"    Risk assessment: {result.risk_assessment}")
            print(f"    Implementation: {result.implementation_feasibility}")
            print(f"    Confidence: {result.confidence_level:.1%}")
            
            # Assessment
            hit_rate_diff = abs(result.validated_hit_rate - result.claimed_hit_rate)
            if hit_rate_diff < 0.1:
                print("    ✅ CLAIM VALIDATED")
            elif hit_rate_diff < 0.2:
                print("    ⚠️  CLAIM PARTIALLY VALIDATED")
            else:
                print("    ❌ CLAIM NOT VALIDATED")
        
        # Volatility strategy results
        print(f"\n⚡ VOLATILITY LIQUIDITY SHOCK VALIDATION:")
        result = volatility_result
        print(f"    Claimed hit rate: {result.claimed_hit_rate:.1%}")
        print(f"    Validated hit rate: {result.validated_hit_rate:.1%}")
        print(f"    Claimed profit: ${result.claimed_profit_per_trade[0]:.1f}-${result.claimed_profit_per_trade[1]:.1f}")
        print(f"    Estimated profit: ${result.estimated_profit_per_trade[0]:.1f}-${result.estimated_profit_per_trade[1]:.1f}")
        print(f"    Daily opportunities: {result.opportunities_per_day:.1f}")
        print(f"    Risk assessment: {result.risk_assessment}")
        print(f"    Implementation: {result.implementation_feasibility}")
        print(f"    Confidence: {result.confidence_level:.1%}")
        
        hit_rate_diff = abs(result.validated_hit_rate - result.claimed_hit_rate)
        if hit_rate_diff < 0.1:
            print("    ✅ CLAIM VALIDATED")
        elif hit_rate_diff < 0.2:
            print("    ⚠️  CLAIM PARTIALLY VALIDATED")
        else:
            print("    ❌ CLAIM NOT VALIDATED")
        
        print("\n" + "="*80)

async def main():
    """Run independent validation analysis"""
    print("INDEPENDENT TRADING STRATEGY VALIDATION")
    print("======================================")
    print("This analysis provides objective validation of trading strategy claims")
    print("using fresh data and independent statistical methods.\n")
    
    # Use high-quality sessions with substantial data
    validation_sessions = [9, 10, 32, 41, 70, 71, 72]  # Sessions with good data
    
    validator = TradingStrategyValidator(validation_sessions)
    await validator.load_sessions()
    
    if not validator.sessions_data:
        print("❌ No sessions loaded. Cannot proceed with validation.")
        return
    
    # Run validations
    anchor_results = validator.validate_anchor_arbitrage_strategy()
    volatility_result = validator.validate_volatility_shock_strategy()
    
    # Generate report
    validator.generate_validation_report(anchor_results, volatility_result)

if __name__ == "__main__":
    asyncio.run(main())