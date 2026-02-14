import React, { memo, useState, useMemo } from 'react';
import OrderbookView from './OrderbookView';
import MarketSelector from './MarketSelector';
import QuoteStatusBar from './QuoteStatusBar';
import AdmiralAgentPanel from '../panels/AdmiralAgentPanel';
import QuotePerformancePanel from '../panels/QuotePerformancePanel';

/**
 * MMCenterContent - Tab container: Orderbook | Agent | Performance.
 */
const TABS = [
  { id: 'orderbook', label: 'Orderbook' },
  { id: 'agent', label: 'Admiral' },
  { id: 'performance', label: 'Performance' },
];

const MMCenterContent = ({
  events,
  quoteState,
  // Agent props
  isRunning, cycleCount, cycleMode, thinking, toolCalls,
  // Performance
  performance,
}) => {
  const [activeTab, setActiveTab] = useState('orderbook');
  const [selectedMarketTicker, setSelectedMarketTicker] = useState(null);

  // Get markets from events map
  const markets = useMemo(() => {
    if (!events || events.size === 0) return [];
    const allMarkets = [];
    events.forEach((event) => {
      if (event.markets) {
        Object.values(event.markets).forEach(m => {
          allMarkets.push({
            ...m,
            event_ticker: event.event_ticker,
            subtitle: m.subtitle || m.title || m.ticker,
          });
        });
      }
    });
    return allMarkets;
  }, [events]);

  // Selected market data
  const selectedMarket = useMemo(() => {
    if (selectedMarketTicker) {
      return markets.find(m => m.ticker === selectedMarketTicker) || markets[0];
    }
    return markets[0] || null;
  }, [markets, selectedMarketTicker]);

  // Our quotes for selected market (from quoteState)
  const ourQuotes = useMemo(() => {
    if (!selectedMarket || !quoteState?.market_quotes) return null;
    return quoteState.market_quotes[selectedMarket.ticker] || null;
  }, [selectedMarket, quoteState]);

  const fairValue = selectedMarket?.fair_value ?? null;

  return (
    <div className="flex-1 flex flex-col min-h-0 min-w-0 bg-gray-950/30">
      {/* Tab bar */}
      <div className="flex items-center gap-1 px-3 py-1.5 border-b border-gray-800/30 shrink-0">
        {TABS.map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`px-3 py-1 rounded-md text-[10px] font-medium transition-colors ${
              activeTab === tab.id
                ? 'bg-gray-800/60 text-white border border-gray-700/40'
                : 'text-gray-500 hover:text-gray-300 hover:bg-gray-800/20 border border-transparent'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      {activeTab === 'orderbook' && (
        <div className="flex-1 flex flex-col min-h-0">
          <MarketSelector
            markets={markets}
            selectedTicker={selectedMarket?.ticker}
            onSelect={setSelectedMarketTicker}
          />
          <QuoteStatusBar quoteState={quoteState} />
          <OrderbookView
            market={selectedMarket}
            ourQuotes={ourQuotes}
            fairValue={fairValue}
          />
        </div>
      )}

      {activeTab === 'agent' && (
        <AdmiralAgentPanel
          isRunning={isRunning}
          cycleCount={cycleCount}
          cycleMode={cycleMode}
          thinking={thinking}
          toolCalls={toolCalls}
        />
      )}

      {activeTab === 'performance' && (
        <QuotePerformancePanel performance={performance} />
      )}
    </div>
  );
};

export default memo(MMCenterContent);
