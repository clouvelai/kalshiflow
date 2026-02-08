import React, { memo, useMemo } from 'react';
import { BarChart2, List } from 'lucide-react';
import MarketOrderbook from '../../ui/MarketOrderbook';
import EventTradeFeed from '../../ui/EventTradeFeed';

const TabButton = memo(({ active, onClick, icon: Icon, label }) => (
  <button
    onClick={onClick}
    className={`flex items-center gap-1.5 px-3 py-1 text-[11px] font-medium rounded-md transition-colors ${
      active
        ? 'bg-cyan-500/12 text-cyan-400 border border-cyan-500/20'
        : 'text-gray-500 hover:text-gray-300 hover:bg-gray-800/30'
    }`}
  >
    <Icon className="w-3.5 h-3.5" />
    {label}
  </button>
));
TabButton.displayName = 'TabButton';

const BookTradesTab = memo(({ event, eventTrades = [], arbTrades = [], selectedMarket, onSelectMarket }) => {
  const markets = event.markets || {};
  const selectedMarketData = selectedMarket ? markets[selectedMarket] : null;

  const mergedTrades = useMemo(() => {
    const eventMarkets = new Set(Object.keys(markets));
    const publicTrades = (eventTrades || [])
      .filter(t => t.event_ticker === event.event_ticker || eventMarkets.has(t.market_ticker))
      .map(t => ({ ...t, source: 'public' }));
    const captainTrades = (arbTrades || [])
      .filter(t => eventMarkets.has(t.market_ticker))
      .map(t => ({ ...t, source: 'captain' }));
    return [...publicTrades, ...captainTrades].sort((a, b) => (b.ts || 0) - (a.ts || 0));
  }, [event.event_ticker, markets, eventTrades, arbTrades]);

  const [activeView, setActiveView] = React.useState('orderbook');

  return (
    <div className="space-y-3">
      {/* Tab bar */}
      <div className="flex items-center gap-2">
        <TabButton active={activeView === 'orderbook'} onClick={() => setActiveView('orderbook')} icon={BarChart2} label="Orderbook" />
        <TabButton active={activeView === 'trades'} onClick={() => setActiveView('trades')} icon={List} label={`Trades (${mergedTrades.length})`} />
        {selectedMarket && (
          <span className="ml-auto text-[10px] text-gray-500 font-mono">{selectedMarket}</span>
        )}
      </div>

      {/* Content */}
      <div className="bg-gray-800/15 rounded-lg border border-gray-800/20 p-3">
        {activeView === 'orderbook' ? (
          selectedMarketData ? (
            <MarketOrderbook market={selectedMarketData} />
          ) : (
            <div className="text-center py-8 text-gray-600">
              <BarChart2 className="w-5 h-5 mx-auto mb-1.5 opacity-40" />
              <p className="text-[11px]">Select a market from the Markets tab</p>
            </div>
          )
        ) : (
          <EventTradeFeed trades={mergedTrades} arbTrades={arbTrades} showSource={true} />
        )}
      </div>
    </div>
  );
});

BookTradesTab.displayName = 'BookTradesTab';

export default BookTradesTab;
