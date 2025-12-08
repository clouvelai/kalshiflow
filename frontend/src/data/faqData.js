// FAQ data structure for Kalshi Flowboard application
export const faqData = [
  {
    id: 'data-source',
    question: 'What data is this application using?',
    answer: `This application displays live public trades data from the Kalshi API. All trade information, market prices, and trading volumes are sourced directly from Kalshi's real-time public trades WebSocket stream. This includes:
    
• Live trade executions (YES/NO positions, prices, volumes)
• Market metadata (titles, categories, expiration dates)
• Trading activity analytics (volume patterns, flow direction)

The application does not provide investment advice and is purely for informational purposes to visualize market activity.`
  },
  {
    id: 'markets',
    question: 'What markets are displayed?',
    answer: `The application displays the most actively traded markets based on recent trading volume. Markets are ranked by their 10-minute trading volume and include various categories such as:

• Elections and political events
• Economic indicators
• Sports and entertainment
• Current events and news

Only markets with recent trading activity appear in the "Hot Markets" grid to focus on the most liquid and actively traded opportunities.`
  },
  {
    id: 'net-flow',
    question: 'What is Net Flow and how should I interpret it?',
    answer: `Net Flow measures the directional bias of trading activity by calculating YES volume minus NO volume over the time window. It shows which side (YES or NO) aggressive traders are taking, revealing market sentiment through actual trades. Remember that Kalshi is a zero-sum prediction market - every trade requires both a buyer and seller - so flow imbalances indicate which side traders are actively seeking.

Common Net Flow Scenarios:

• Positive net flow on complementary markets (e.g., both Trump-YES and Harris-YES showing +$50k): Traders are buying YES on both outcomes, likely hedging or market makers are selling overpriced YES contracts on both sides since only one can win.

• Large positive net flow (+$100k): Heavy YES buying pressure - traders are aggressively taking YES positions, pushing prices up as sellers demand higher prices to part with contracts.

• Large negative net flow (-$100k): Heavy NO buying pressure - traders are aggressively taking NO positions (betting against the outcome), pushing YES prices down.

• Zero net flow with high volume: Equal YES and NO buying - balanced market with good liquidity, prices stable as buyers and sellers are matched efficiently.

• Sudden flow reversal (from +$50k to -$50k): Sharp sentiment change - often triggered by breaking news, new polls, or whale traders entering/exiting positions.`
  }
];

export default faqData;