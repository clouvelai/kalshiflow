import React, { memo } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Activity, Wifi, WifiOff } from 'lucide-react';
import { StateBadge } from '../ui/StateBadge';

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
 * V3Header - Header bar for V3 Trader Console
 */
const V3Header = ({ wsStatus, currentState }) => {
  return (
    <div className="border-b border-gray-800 bg-black/30 backdrop-blur-sm sticky top-0 z-10">
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
          </div>

          <div className="flex items-center space-x-6">
            {/* Connection Status */}
            <div className="flex items-center space-x-2">
              {wsStatus === 'connected' ? (
                <>
                  <Wifi className="w-5 h-5 text-green-400" />
                  <span className="text-sm text-green-400 font-medium">Connected</span>
                </>
              ) : (
                <>
                  <WifiOff className="w-5 h-5 text-red-400" />
                  <span className="text-sm text-red-400 font-medium">Disconnected</span>
                </>
              )}
            </div>

            {/* Current State Badge */}
            <StateBadge state={currentState} />
          </div>
        </div>
      </div>
    </div>
  );
};

export default memo(V3Header);
