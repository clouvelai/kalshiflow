import React, { memo } from 'react';
import { Activity, Wifi, WifiOff } from 'lucide-react';
import { StateBadge } from '../ui/StateBadge';

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
              <Activity className="w-6 h-6 text-purple-400" />
              <h1 className="text-xl font-semibold text-white">TRADER V3</h1>
              <span className="px-2 py-0.5 text-xs font-mono bg-purple-500/20 text-purple-300 rounded-full border border-purple-500/30">
                CONSOLE
              </span>
            </div>
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
