import React, { memo } from 'react';
import { Activity, Wifi, WifiOff } from 'lucide-react';

/**
 * ArbHeader - Top bar for the arbitrage dashboard
 *
 * Shows title, connection dot + text, and system state badge.
 */
const ArbHeader = ({ connectionStatus, systemState }) => {
  const isConnected = connectionStatus === 'connected';

  const stateLabel = (systemState || 'initializing').replace(/_/g, ' ').toUpperCase();

  const STATE_BADGE_COLORS = {
    ready: 'bg-emerald-900/30 text-emerald-400 border-emerald-600/30',
    trading: 'bg-emerald-900/30 text-emerald-400 border-emerald-600/30',
    acting: 'bg-emerald-900/30 text-emerald-400 border-emerald-600/30',
    error: 'bg-red-900/30 text-red-400 border-red-600/30',
    calibrating: 'bg-amber-900/30 text-amber-400 border-amber-600/30',
  };
  const stateBadgeColor = STATE_BADGE_COLORS[systemState] || 'bg-gray-800/50 text-gray-400 border-gray-700/40';

  return (
    <div className="border-b border-gray-800 bg-black/30 backdrop-blur-sm sticky top-0 z-10">
      <div className="max-w-[1600px] mx-auto px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <Activity className="w-6 h-6 text-cyan-400" />
            <h1 className="text-xl font-semibold text-white">KalshiFlow Arb</h1>
            <span className="px-2 py-0.5 text-xs font-mono bg-violet-500/20 text-violet-300 rounded-full border border-violet-500/30">
              DASHBOARD
            </span>
          </div>

          <div className="flex items-center space-x-6">
            <div className="flex items-center space-x-2">
              {isConnected ? (
                <>
                  <span className="relative flex h-2.5 w-2.5">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75" />
                    <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-green-500" />
                  </span>
                  <Wifi className="w-4 h-4 text-green-400" />
                  <span className="text-sm text-green-400 font-medium">Live</span>
                </>
              ) : (
                <>
                  <span className="inline-flex rounded-full h-2.5 w-2.5 bg-red-500" />
                  <WifiOff className="w-4 h-4 text-red-400" />
                  <span className="text-sm text-red-400 font-medium">Disconnected</span>
                </>
              )}
            </div>

            <span className={`px-3 py-1 rounded-full text-[10px] font-bold border uppercase tracking-wider ${stateBadgeColor}`}>
              {stateLabel}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default memo(ArbHeader);
