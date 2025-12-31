import React from 'react';
import { Link, useLocation } from 'react-router-dom';

/**
 * NavTabs - Navigation between Trader and Discovery views
 */
const NavTabs = () => {
  const location = useLocation();
  const isTrader = location.pathname.includes('v3');
  const isDiscovery = location.pathname.includes('lifecycle');

  return (
    <div className="flex items-center bg-gray-800/50 rounded-lg p-1">
      <Link
        to="/v3"
        className={`px-3 py-1.5 text-sm font-medium rounded-md transition-colors ${
          isTrader
            ? 'bg-cyan-500/30 text-cyan-300'
            : 'text-gray-400 hover:text-white hover:bg-gray-700/50'
        }`}
      >
        Trader
      </Link>
      <Link
        to="/lifecycle"
        className={`px-3 py-1.5 text-sm font-medium rounded-md transition-colors ${
          isDiscovery
            ? 'bg-cyan-500/30 text-cyan-300'
            : 'text-gray-400 hover:text-white hover:bg-gray-700/50'
        }`}
      >
        Discovery
      </Link>
    </div>
  );
};

/**
 * LifecycleHeader - Title, capacity bar, and connection status
 */
const LifecycleHeader = ({ wsStatus, stats, isAtCapacity, balance = 0, minTraderCash = 0 }) => {
  const tracked = stats?.tracked || 0;
  const capacity = stats?.capacity || 1000;
  const percentage = (tracked / capacity) * 100;
  const isLowBalance = minTraderCash > 0 && balance < minTraderCash;

  // Color coding for capacity
  const getCapacityColor = () => {
    if (percentage >= 95) return 'bg-red-500';
    if (percentage >= 80) return 'bg-amber-500';
    return 'bg-emerald-500';
  };

  const getCapacityTextColor = () => {
    if (percentage >= 95) return 'text-red-400';
    if (percentage >= 80) return 'text-amber-400';
    return 'text-emerald-400';
  };

  return (
    <header className="bg-gray-900/80 border-b border-gray-800 sticky top-0 z-10 backdrop-blur-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 py-4">
        <div className="flex items-center justify-between">
          {/* Title and Navigation */}
          <div className="flex items-center gap-4">
            <h1 className="text-xl font-bold text-white tracking-tight">
              LIFECYCLE DISCOVERY
            </h1>

            {/* Navigation Tabs */}
            <NavTabs />

            {/* Capacity indicator */}
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2">
                <span className={`text-sm font-mono ${getCapacityTextColor()}`}>
                  {tracked}/{capacity}
                </span>
                <span className="text-gray-500 text-sm">tracked</span>
              </div>

              {/* Capacity bar */}
              <div className="w-24 h-2 bg-gray-800 rounded-full overflow-hidden">
                <div
                  className={`h-full ${getCapacityColor()} transition-all duration-500`}
                  style={{ width: `${Math.min(percentage, 100)}%` }}
                />
              </div>

              {isAtCapacity && (
                <span className="text-xs text-red-400 font-medium px-2 py-0.5 bg-red-900/30 rounded">
                  AT CAPACITY
                </span>
              )}

              {isLowBalance && (
                <span
                  className="text-xs text-amber-300/70 px-2 py-0.5 bg-amber-900/20 rounded"
                  title={`Balance $${(balance/100).toFixed(2)} below $${(minTraderCash/100).toFixed(2)} minimum`}
                >
                  Low Cash
                </span>
              )}
            </div>
          </div>

          {/* Connection status */}
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${
                wsStatus === 'connected' ? 'bg-emerald-500 animate-pulse' :
                wsStatus === 'connecting' ? 'bg-amber-500 animate-pulse' :
                'bg-red-500'
              }`} />
              <span className={`text-sm ${
                wsStatus === 'connected' ? 'text-emerald-400' :
                wsStatus === 'connecting' ? 'text-amber-400' :
                'text-red-400'
              }`}>
                {wsStatus === 'connected' ? 'Live' :
                 wsStatus === 'connecting' ? 'Connecting...' :
                 'Disconnected'}
              </span>
            </div>

            {/* Stats summary */}
            {stats && (
              <div className="flex items-center gap-4 ml-4 pl-4 border-l border-gray-700">
                <div className="text-sm">
                  <span className="text-gray-500">Active:</span>
                  <span className="text-white ml-1 font-mono">{stats.by_status?.active || 0}</span>
                </div>
                <div className="text-sm">
                  <span className="text-gray-500">Determined:</span>
                  <span className="text-amber-400 ml-1 font-mono">{stats.by_status?.determined || 0}</span>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </header>
  );
};

export default LifecycleHeader;
