import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { TradeProvider, useTradeContext } from './context/TradeContext';
import Layout from './components/Layout';
import UnifiedAnalytics from './components/UnifiedAnalytics';
import TopTradesList from './components/TopTradesList';
import FAQ from './components/FAQ';
import { ArbDashboard } from './components/arb';

const MainDashboard = () => {
  const {
    topTrades,
    hourAnalyticsData,
    connectionStatus,
    error
  } = useTradeContext();

  return (
    <Layout 
      connectionStatus={connectionStatus}
      data-testid="main-layout"
    >
      <UnifiedAnalytics 
        hourAnalyticsData={hourAnalyticsData}
        data-testid="unified-analytics"
      />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-8">
        <TopTradesList trades={topTrades} />
      </div>

      <div className="max-w-3xl mx-auto mt-12 px-4 sm:px-0" data-testid="faq-section">
        <FAQ />
      </div>

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
      </Routes>
    </Router>
  );
}

export default App
