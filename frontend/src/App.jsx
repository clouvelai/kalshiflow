import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { TradeProvider, useTradeContext } from './context/TradeContext';
import Layout from './components/Layout';
import UnifiedAnalytics from './components/UnifiedAnalytics';
import MarketGrid from './components/MarketGrid';
import TradeFlowRiver from './components/TradeFlowRiver';
import TopTradesList from './components/TopTradesList';
import FAQ from './components/FAQ';
import RLTraderDashboard from './components/RLTraderDashboard';
import V3TraderConsole from './components/V3TraderConsole';
import AgentPage from './components/v3-trader/pages/AgentPage';
import { LifecycleDiscovery } from './components/lifecycle';
import { ArbDashboard } from './components/arb';

const MainDashboard = () => {
  const {
    recentTrades,
    hotMarkets,
    topTrades,
    globalStats,
    hourAnalyticsData,
    dayAnalyticsData,
    connectionStatus,
    error
  } = useTradeContext();

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

      {/* Trade Flow River - Visual representation of trades */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-8">
        <TradeFlowRiver trades={recentTrades} />
      </div>

      {/* Top Trades List - Biggest trades by volume */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-8">
        <TopTradesList trades={topTrades} />
      </div>

      {/* Market Grid Section */}
      <MarketGrid
        markets={hotMarkets}
        data-testid="market-grid"
      />

      {/* FAQ Section */}
      <div className="max-w-3xl mx-auto mt-12 px-4 sm:px-0" data-testid="faq-section">
        <FAQ />
      </div>

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
    <Router>
      <Routes>
        <Route 
          path="/" 
          element={
            <TradeProvider>
              <MainDashboard />
            </TradeProvider>
          } 
        />
        <Route 
          path="/rl-trader" 
          element={<RLTraderDashboard />} 
        />
        <Route 
          path="/trader" 
          element={<RLTraderDashboard />} 
        />
        <Route
          path="/v3-trader"
          element={<ArbDashboard />}
        />
        <Route
          path="/v3"
          element={<ArbDashboard />}
        />
        <Route
          path="/arb"
          element={<ArbDashboard />}
        />
        <Route
          path="/v3-trader/legacy"
          element={<V3TraderConsole />}
        />
        <Route
          path="/v3-trader/agent"
          element={<AgentPage />}
        />
        <Route
          path="/lifecycle"
          element={<LifecycleDiscovery />}
        />
      </Routes>
    </Router>
  );
}

export default App
