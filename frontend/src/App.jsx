import React, { useState } from 'react';
import { TradeProvider, useTradeContext } from './context/TradeContext';
import Layout from './components/Layout';
import UnifiedAnalytics from './components/UnifiedAnalytics';
import MarketGrid from './components/MarketGrid';
import TradeTape from './components/TradeTape';
import TickerDetailDrawer from './components/TickerDetailDrawer';
import FAQ from './components/FAQ';

const AppContent = () => {
  const {
    recentTrades,
    hotMarkets,
    selectedTicker,
    globalStats,
    hourAnalyticsData,
    dayAnalyticsData,
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
      data-testid="main-layout"
    >
      {/* Unified Analytics Section - simplified with single analytics_update message */}
      <UnifiedAnalytics 
        hourAnalyticsData={hourAnalyticsData}
        dayAnalyticsData={dayAnalyticsData}
        data-testid="unified-analytics"
      />

      {/* Market Grid Section */}
      <MarketGrid
        markets={hotMarkets}
        selectedTicker={selectedTicker}
        onTickerSelect={handleTickerSelect}
        data-testid="market-grid"
      />

      {/* FAQ and Live Trade Feed Section - Side by side */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 lg:gap-8 max-w-6xl mx-auto mt-12 px-4 sm:px-0" data-testid="faq-tradetape-section">
        <div className="order-1" data-testid="faq-component">
          <FAQ />
        </div>
        <div className="order-2" data-testid="trade-tape-section">
          <TradeTape
            trades={recentTrades}
            selectedTicker={selectedTicker}
            onTickerSelect={handleTickerSelect}
            data-testid="trade-tape"
          />
        </div>
      </div>

      {/* Ticker Detail Drawer */}
      <TickerDetailDrawer
        ticker={selectedTicker}
        tickerData={tickerData}
        isOpen={isDrawerOpen}
        onClose={handleCloseDrawer}
        data-testid="ticker-detail-drawer"
      />

      {/* Error Display */}
      {error && (
        <div className="fixed bottom-4 right-4 bg-red-500 text-white px-4 py-2 rounded-lg shadow-lg" data-testid="error-display">
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
