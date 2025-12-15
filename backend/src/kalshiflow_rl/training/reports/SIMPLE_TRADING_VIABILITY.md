# Simple Trading Strategy Viability Analysis

Generated: 2025-12-15T12:57:46.491068
Sessions Analyzed: 14
Total Timesteps: 445,132
Total Observations: 407,074

## Executive Summary

Based on analysis of real orderbook data from Kalshi markets:

**Market Making**: ✅ VIABLE
  - Opportunity Rate: 87.87%
  - Suitable Markets: 980

**Spread Capture**: ✅ VIABLE
  - Opportunity Rate: 63.23%
  - Suitable Markets: 969

**Arbitrage**: ⚠️ RARE
  - Opportunity Rate: 0.00%
  - Average Profit: 2.00 cents

## Spread Distribution Analysis

### Overall Spread Statistics
```
tight (<2¢)            49,392 ( 12.13%)
market_making (2-5¢)  100,303 ( 24.64%)
wide (5-10¢)           63,210 ( 15.53%)
very_wide (≥10¢)      194,169 ( 47.70%)
```

### Spread Percentiles (cents)
```
p10  :   1.00¢
p25  :   3.00¢
p50  :   8.00¢
p75  :  47.00¢
p90  :  79.00¢
p95  :  89.00¢
p99  :  96.00¢
```

## Market Making Opportunities

Minimum Spread Threshold: 2 cents
Opportunity Rate: 87.87%

### Top Markets for Market Making
| Market | Avg Spread | Opportunity % | Volume | Sessions |
|--------|------------|---------------|---------|----------|
| BEYONCEGENRE-30-AFA | 31.00¢ | 100.0% | 12619 | 14 |
| BEYONCEGENRE-30-DEA | 42.00¢ | 100.0% | 10634 | 14 |
| BEYONCEGENRE-30-RA | 39.00¢ | 100.0% | 9758 | 14 |
| BEYONCEGENRE-30-TAA | 40.51¢ | 100.0% | 10313 | 14 |
| BEYONCEGENRE-30-TCA | 42.00¢ | 100.0% | 10740 | 14 |
| BEYONCEGENRE-30-TGA | 32.00¢ | 100.0% | 11871 | 14 |
| BEYONCEGENRE-30-THA | 46.39¢ | 100.0% | 9727 | 14 |
| BEYONCEGENRE-30-TRA | 22.00¢ | 100.0% | 13194 | 14 |
| BEYONCEGENRE-30-TRHA | 22.00¢ | 100.0% | 13215 | 14 |
| EUCLIMATE-2030 | 8.00¢ | 100.0% | 913965 | 14 |

## Spread Capture Opportunities

Minimum Spread Threshold: 5 cents
Opportunity Rate: 63.23%

### Top Markets for Spread Capture
| Market | Avg Spread | Wide Spread % | Volume | Sessions |
|--------|------------|---------------|---------|----------|
| BEYONCEGENRE-30-AFA | 31.00¢ | 100.0% | 12619 | 14 |
| BEYONCEGENRE-30-DEA | 42.00¢ | 100.0% | 10634 | 14 |
| BEYONCEGENRE-30-RA | 39.00¢ | 100.0% | 9758 | 14 |
| BEYONCEGENRE-30-TAA | 40.51¢ | 100.0% | 10313 | 14 |
| BEYONCEGENRE-30-TCA | 42.00¢ | 100.0% | 10740 | 14 |
| BEYONCEGENRE-30-TGA | 32.00¢ | 100.0% | 11871 | 14 |
| BEYONCEGENRE-30-THA | 46.39¢ | 100.0% | 9727 | 14 |
| BEYONCEGENRE-30-TRA | 22.00¢ | 100.0% | 13194 | 14 |
| BEYONCEGENRE-30-TRHA | 22.00¢ | 100.0% | 13215 | 14 |
| EUCLIMATE-2030 | 8.00¢ | 100.0% | 913965 | 14 |

## Arbitrage Analysis

Minimum Edge Threshold: 2 cents
Total Opportunities Found: 5
Rate per 1000 Timesteps: 0.01

### Top Arbitrage Opportunities
| Market | YES Ask | NO Ask | Profit | Timestep |
|--------|---------|--------|--------|----------|
| KXGDPUSMAX-28-5 | 61¢ | 37¢ | 2¢ | 30699 |
| KXGDPUSMAX-28-5 | 61¢ | 37¢ | 2¢ | 30894 |
| KXGDPUSMAX-28-5 | 61¢ | 37¢ | 2¢ | 31478 |
| KXGDPUSMAX-28-5 | 61¢ | 37¢ | 2¢ | 31658 |
| KXGDPUSMAX-28-5 | 61¢ | 37¢ | 2¢ | 31659 |

## Key Findings

### Positive Indicators
- **Frequent Market Making Opportunities**: 87.9% of observations have spreads ≥2¢
- **Regular Wide Spreads**: 63.2% of observations have spreads ≥5¢
- **Favorable Spread Distribution**: 75th percentile spread is 47.00¢
- **Liquid Markets Available**: 665 markets with average volume >10,000

### Risk Factors
- **Rare Arbitrage**: Only 5 arbitrage opportunities found

## Implementation Recommendations

### Priority Order

1. **Market Making**
   - Opportunity Rate: 87.87%
   - Implementation Complexity: Low
   - Expected Profitability: High

2. **Spread Capture**
   - Opportunity Rate: 63.23%
   - Implementation Complexity: Medium
   - Expected Profitability: High

### Suggested Parameters

**Market Making:**
- Minimum Spread: 3 cents
- Target Spread: 8 cents
- Order Size: Start with 5-10 contracts
- Max Position: 100 contracts

**Spread Capture:**
- Minimum Spread: 47 cents
- Target Spread: 79 cents
- Order Size: 5 contracts (conservative)
- Frequency: Act on 63.2% of timesteps

## Conclusion

✅ **Simple trading strategies ARE VIABLE** based on the collected orderbook data.

The data shows sufficient spread opportunities to support profitable trading, particularly
through market making strategies. Implementation should begin with the highest opportunity
rate strategies and conservative position sizing.