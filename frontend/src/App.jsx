import React, { useState } from 'react';
import { TradeProvider, useTradeContext } from './context/TradeContext';
import Layout from './components/Layout';
import UnifiedAnalytics from './components/UnifiedAnalytics';
import MarketGrid from './components/MarketGrid';
import TradeTape from './components/TradeTape';
import TickerDetailDrawer from './components/TickerDetailDrawer';

const AppContent = () => {
  const {
    recentTrades,
    hotMarkets,
    selectedTicker,
    globalStats,
    analyticsData,
    analyticsSummary,
    connectionStatus,
    error,
    selectTicker,
    getTickerData
  } = useTradeContext();

  const [isDrawerOpen, setIsDrawerOpen] = useState(false);

  const handleTickerSelect = (ticker) => {
    selectTicker(ticker);
    setIsDrawerOpen(true);
  };

  const handleCloseDrawer = () => {
    setIsDrawerOpen(false);
  };

  const tickerData = selectedTicker ? getTickerData(selectedTicker) : null;

  return (
    <Layout 
      connectionStatus={connectionStatus}
    >
      {/* Unified Analytics Section - replaces HeroStats and AnalyticsChart */}
      <UnifiedAnalytics 
        analyticsData={analyticsData || { 
          hour_minute_mode: { time_series: [], summary_stats: {} },
          day_hour_mode: { time_series: [], summary_stats: {} }
        }}
      />

      {/* Market Grid Section */}
      <MarketGrid
        markets={hotMarkets}
        selectedTicker={selectedTicker}
        onTickerSelect={handleTickerSelect}
      />

      {/* Live Trade Feed Section - Bottom with reduced prominence */}
      <div className="max-w-4xl">
        <TradeTape
          trades={recentTrades}
          selectedTicker={selectedTicker}
          onTickerSelect={handleTickerSelect}
        />
      </div>

      {/* Ticker Detail Drawer */}
      <TickerDetailDrawer
        ticker={selectedTicker}
        tickerData={tickerData}
        isOpen={isDrawerOpen}
        onClose={handleCloseDrawer}
      />

      {/* Error Display */}
      {error && (
        <div className="fixed bottom-4 right-4 bg-red-500 text-white px-4 py-2 rounded-lg shadow-lg">
          <p className="text-sm">{error}</p>
        </div>
      )}
    </Layout>
  );
};

function App() {
  return (
    <TradeProvider>
      <AppContent />
    </TradeProvider>
  );
}

export default App
