import React from 'react';

/**
 * LifecycleHeader - Title, capacity bar, and connection status
 */
const LifecycleHeader = ({ wsStatus, stats, isAtCapacity }) => {
  const tracked = stats?.tracked || 0;
  const capacity = stats?.capacity || 1000;
  const percentage = (tracked / capacity) * 100;

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
          {/* Title */}
          <div className="flex items-center gap-4">
            <h1 className="text-xl font-bold text-white tracking-tight">
              LIFECYCLE DISCOVERY
            </h1>

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
