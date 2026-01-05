# H123 vs Current RLM Parameters: Full Bucket Comparison

Generated: 2026-01-01T13:16:05.947114

## Executive Summary

### H123 (5 trades, any drop, 70% YES)
- Total Markets: 1986
- Overall Win Rate: 90.2%
- Weighted Baseline: 76.8%
- Overall Improvement: +13.4%

### Current (25 trades, 2c drop, 70% YES)
- Total Markets: 788
- Overall Win Rate: 94.3%
- Weighted Baseline: 76.3%
- Overall Improvement: +18.0%

## Table 1: H123 Parameters (5 trades, any drop)
**Parameters**: yes_threshold=70%, min_trades=5, min_price_drop=0c

| NO Price Bucket | N Markets | RLM Win Rate | Baseline | Improvement | P-value | Significant |
|-----------------|-----------|--------------|----------|-------------|---------|-------------|
| 0-5c         |         1 |        0.0% |    3.9% |      -3.9% |  1.0000 | no          |
| 5-10c        |         1 |        0.0% |    2.4% |      -2.4% |  1.0000 | no          |
| 10-15c       |         4 |        0.0% |    5.6% |      -5.6% |  1.0000 | no          |
| 15-20c       |         5 |        0.0% |    8.2% |      -8.2% |  1.0000 | no          |
| 20-25c       |        18 |       38.9% |   11.6% |     +27.3% |  0.0001 | YES         |
| 25-30c       |        10 |       50.0% |   15.1% |     +34.9% |  0.0010 | YES         |
| 30-35c       |        21 |       52.4% |   22.3% |     +30.1% |  0.0005 | YES         |
| 35-40c       |        39 |       59.0% |   32.3% |     +26.7% |  0.0002 | YES         |
| 40-45c       |        56 |       67.9% |   38.2% |     +29.6% |  0.0000 | YES         |
| 45-50c       |        91 |       72.5% |   46.9% |     +25.7% |  0.0000 | YES         |
| 50-55c       |       127 |       73.2% |   55.7% |     +17.6% |  0.0000 | YES         |
| 55-60c       |       114 |       86.8% |   63.8% |     +23.1% |  0.0000 | YES         |
| 60-65c       |       126 |       91.3% |   70.3% |     +20.9% |  0.0000 | YES         |
| 65-70c       |       142 |       95.1% |   76.5% |     +18.6% |  0.0000 | YES         |
| 70-75c       |       196 |       94.9% |   78.0% |     +16.9% |  0.0000 | YES         |
| 75-80c       |       212 |       96.7% |   84.0% |     +12.7% |  0.0000 | YES         |
| 80-85c       |       208 |       96.6% |   88.6% |      +8.1% |  0.0001 | YES         |
| 85-90c       |       238 |       97.9% |   91.9% |      +6.0% |  0.0003 | YES         |
| 90-95c       |       232 |       99.1% |   95.7% |      +3.5% |  0.0048 | YES         |
| 95-100c      |       145 |       99.3% |   98.8% |      +0.6% |  0.2739 | no          |
|-----------------|-----------|--------------|----------|-------------|---------|-------------|
| **TOTAL**       |      1986 |       90.2% |   76.8% |     +13.4% |         |             |

## Table 2: Current Parameters (25 trades, 2c drop)
**Parameters**: yes_threshold=70%, min_trades=25, min_price_drop=2c

| NO Price Bucket | N Markets | RLM Win Rate | Baseline | Improvement | P-value | Significant |
|-----------------|-----------|--------------|----------|-------------|---------|-------------|
| 10-15c       |         1 |        0.0% |    5.6% |      -5.6% |  1.0000 | no          |
| 15-20c       |         1 |        0.0% |    8.2% |      -8.2% |  1.0000 | no          |
| 20-25c       |         6 |       50.0% |   11.6% |     +38.4% |  0.0016 | YES         |
| 25-30c       |         3 |       66.7% |   15.1% |     +51.6% |  0.0063 | YES         |
| 30-35c       |         4 |       75.0% |   22.3% |     +52.7% |  0.0057 | YES         |
| 35-40c       |        16 |       62.5% |   32.3% |     +30.2% |  0.0048 | YES         |
| 40-45c       |        16 |       62.5% |   38.2% |     +24.3% |  0.0228 | YES         |
| 45-50c       |        38 |       84.2% |   46.9% |     +37.3% |  0.0000 | YES         |
| 50-55c       |        52 |       80.8% |   55.7% |     +25.1% |  0.0001 | YES         |
| 55-60c       |        50 |       94.0% |   63.8% |     +30.2% |  0.0000 | YES         |
| 60-65c       |        63 |       95.2% |   70.3% |     +24.9% |  0.0000 | YES         |
| 65-70c       |        72 |       97.2% |   76.5% |     +20.7% |  0.0000 | YES         |
| 70-75c       |        84 |       97.6% |   78.0% |     +19.6% |  0.0000 | YES         |
| 75-80c       |       101 |      100.0% |   84.0% |     +16.0% |  0.0000 | YES         |
| 80-85c       |        99 |      100.0% |   88.6% |     +11.4% |  0.0002 | YES         |
| 85-90c       |        89 |      100.0% |   91.9% |      +8.1% |  0.0025 | YES         |
| 90-95c       |        61 |      100.0% |   95.7% |      +4.3% |  0.0485 | YES         |
| 95-100c      |        32 |      100.0% |   98.8% |      +1.2% |  0.2629 | no          |
|-----------------|-----------|--------------|----------|-------------|---------|-------------|
| **TOTAL**       |       788 |       94.3% |   76.3% |     +18.0% |         |             |

## Table 3: Side-by-Side Comparison

| Bucket | H123 N | H123 Win | H123 Improve | H123 Sig | Current N | Current Win | Current Improve | Current Sig |
|--------|--------|----------|--------------|----------|-----------|-------------|-----------------|-------------|
| 0-5c   | 1      | 0.0%     | -3.9%        | no       | -         | -           | -               | -           |
| 5-10c  | 1      | 0.0%     | -2.4%        | no       | -         | -           | -               | -           |
| 10-15c | 4      | 0.0%     | -5.6%        | no       | 1         | 0.0%        | -5.6%           | no          |
| 15-20c | 5      | 0.0%     | -8.2%        | no       | 1         | 0.0%        | -8.2%           | no          |
| 20-25c | 18     | 38.9%    | +27.3%       | YES      | 6         | 50.0%       | +38.4%          | YES         |
| 25-30c | 10     | 50.0%    | +34.9%       | YES      | 3         | 66.7%       | +51.6%          | YES         |
| 30-35c | 21     | 52.4%    | +30.1%       | YES      | 4         | 75.0%       | +52.7%          | YES         |
| 35-40c | 39     | 59.0%    | +26.7%       | YES      | 16        | 62.5%       | +30.2%          | YES         |
| 40-45c | 56     | 67.9%    | +29.6%       | YES      | 16        | 62.5%       | +24.3%          | YES         |
| 45-50c | 91     | 72.5%    | +25.7%       | YES      | 38        | 84.2%       | +37.3%          | YES         |
| 50-55c | 127    | 73.2%    | +17.6%       | YES      | 52        | 80.8%       | +25.1%          | YES         |
| 55-60c | 114    | 86.8%    | +23.1%       | YES      | 50        | 94.0%       | +30.2%          | YES         |
| 60-65c | 126    | 91.3%    | +20.9%       | YES      | 63        | 95.2%       | +24.9%          | YES         |
| 65-70c | 142    | 95.1%    | +18.6%       | YES      | 72        | 97.2%       | +20.7%          | YES         |
| 70-75c | 196    | 94.9%    | +16.9%       | YES      | 84        | 97.6%       | +19.6%          | YES         |
| 75-80c | 212    | 96.7%    | +12.7%       | YES      | 101       | 100.0%      | +16.0%          | YES         |
| 80-85c | 208    | 96.6%    | +8.1%        | YES      | 99        | 100.0%      | +11.4%          | YES         |
| 85-90c | 238    | 97.9%    | +6.0%        | YES      | 89        | 100.0%      | +8.1%           | YES         |
| 90-95c | 232    | 99.1%    | +3.5%        | YES      | 61        | 100.0%      | +4.3%           | YES         |
| 95-100c | 145    | 99.3%    | +0.6%        | no       | 32        | 100.0%      | +1.2%           | no          |

## Bucket Performance Analysis

### H123 Bucket Analysis

**Buckets that beat baseline**: 16/20
  - Buckets: 20-25c, 25-30c, 30-35c, 35-40c, 40-45c, 45-50c, 50-55c, 55-60c, 60-65c, 65-70c, 70-75c, 75-80c, 80-85c, 85-90c, 90-95c, 95-100c

**Statistically significant improvements (p<0.05)**: 15/20
  - Buckets: 20-25c, 25-30c, 30-35c, 35-40c, 40-45c, 45-50c, 50-55c, 55-60c, 60-65c, 65-70c, 70-75c, 75-80c, 80-85c, 85-90c, 90-95c

**Buckets that lose to baseline**: 4/20
  - 0-5c: -3.9% (N=1)
  - 5-10c: -2.4% (N=1)
  - 10-15c: -5.6% (N=4)
  - 15-20c: -8.2% (N=5)

**Sweet spot (N>=20)**: 30-35c with +30.1% improvement (N=21)

### Current Bucket Analysis

**Buckets that beat baseline**: 16/18
  - Buckets: 20-25c, 25-30c, 30-35c, 35-40c, 40-45c, 45-50c, 50-55c, 55-60c, 60-65c, 65-70c, 70-75c, 75-80c, 80-85c, 85-90c, 90-95c, 95-100c

**Statistically significant improvements (p<0.05)**: 15/18
  - Buckets: 20-25c, 25-30c, 30-35c, 35-40c, 40-45c, 45-50c, 50-55c, 55-60c, 60-65c, 65-70c, 70-75c, 75-80c, 80-85c, 85-90c, 90-95c

**Buckets that lose to baseline**: 2/18
  - 10-15c: -5.6% (N=1)
  - 15-20c: -8.2% (N=1)

**Sweet spot (N>=20)**: 45-50c with +37.3% improvement (N=38)

## Key Insights

### 1. Trade-off: Volume vs Purity
- H123 generates 1986 signals vs Current's 788 (2.5x more)
- Current has higher win rate (94.3% vs 90.2%)
- But H123 has higher improvement over baseline (+13.4% vs +18.0%)

### 2. Price Range Comparison
- H123 average NO price: 72.8c
- Current average NO price: 71.8c
- H123 trades at lower prices, capturing more alpha from mispriced markets

### 3. Statistical Significance by Bucket
- H123: 15/20 buckets significant
- Current: 15/18 buckets significant