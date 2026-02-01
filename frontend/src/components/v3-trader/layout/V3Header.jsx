import React, { memo } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Activity, Wifi, WifiOff, Brain } from 'lucide-react';
import { StateBadge } from '../ui/StateBadge';

/**
 * NavTabs - Navigation between Trader and Agent views
 */
const NavTabs = () => {
  const location = useLocation();
  const isTrader = location.pathname === '/v3' || location.pathname === '/v3-trader';
  const isAgent = location.pathname === '/v3-trader/agent';

  return (
    <div className="flex items-center bg-gray-800/50 rounded-lg p-1" data-testid="v3-nav-tabs">
      <Link
        to="/v3"
        data-testid="v3-nav-trader"
        className={`px-3 py-1.5 text-sm font-medium rounded-md transition-colors ${
          isTrader
            ? 'bg-cyan-500/30 text-cyan-300'
            : 'text-gray-400 hover:text-white hover:bg-gray-700/50'
        }`}
      >
        Trader
      </Link>
      <Link
        to="/v3-trader/agent"
        data-testid="v3-nav-agent"
        className={`px-3 py-1.5 text-sm font-medium rounded-md transition-colors flex items-center gap-1.5 ${
          isAgent
            ? 'bg-violet-500/30 text-violet-300'
            : 'text-gray-400 hover:text-white hover:bg-gray-700/50'
        }`}
      >
        <Brain className="w-3.5 h-3.5" />
        Agent
      </Link>
    </div>
  );
};

/**
 * V3Header - Header bar for V3 Trader Console
 */
const V3Header = ({ wsStatus, currentState, balance = 0, minTraderCash = 0 }) => {
  const isLowBalance = minTraderCash > 0 && balance < minTraderCash;

  return (
    <div className="border-b border-gray-800 bg-black/30 backdrop-blur-sm sticky top-0 z-10" data-testid="v3-header">
      <div className="max-w-7xl mx-auto px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <Activity className="w-6 h-6 text-cyan-400" />
              <h1 className="text-xl font-semibold text-white">TRADER V3</h1>
              <span className="px-2 py-0.5 text-xs font-mono bg-cyan-500/20 text-cyan-300 rounded-full border border-cyan-500/30">
                CONSOLE
              </span>
            </div>

            {/* Navigation Tabs */}
            <NavTabs />

            {/* Low Balance Indicator */}
            {isLowBalance && (
              <span
                className="text-xs text-amber-300/70 px-2 py-0.5 bg-amber-900/20 rounded"
                title={`Balance $${(balance/100).toFixed(2)} below $${(minTraderCash/100).toFixed(2)} minimum`}
              >
                Low Cash
              </span>
            )}
          </div>

          <div className="flex items-center space-x-6">
            {/* Connection Status */}
            <div className="flex items-center space-x-2" data-testid="v3-connection-status">
              {wsStatus === 'connected' ? (
                <>
                  <Wifi className="w-5 h-5 text-green-400" />
                  <span className="text-sm text-green-400 font-medium" data-testid="v3-connection-text">Connected</span>
                </>
              ) : (
                <>
                  <WifiOff className="w-5 h-5 text-red-400" />
                  <span className="text-sm text-red-400 font-medium" data-testid="v3-connection-text">Disconnected</span>
                </>
              )}
            </div>

            {/* Current State Badge */}
            <span data-testid="v3-trader-state"><StateBadge state={currentState} /></span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default memo(V3Header);
