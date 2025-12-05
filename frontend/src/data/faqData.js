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
  }
];

export default faqData;