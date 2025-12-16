import React from 'react';

const CollectionStatus = ({ status }) => {
  // Helper function to format numbers
  const formatNumber = (num) => {
    if (num === null || num === undefined) return '0';
    if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`;
    if (num >= 1000) return `${(num / 1000).toFixed(1)}K`;
    return num.toString();
  };

  // Helper function to format time ago
  const formatTimeAgo = (timestamp) => {
    if (!timestamp) return 'Never';
    const now = new Date();
    const then = new Date(timestamp);
    const seconds = Math.floor((now - then) / 1000);
    
    if (seconds < 5) return 'Just now';
    if (seconds < 60) return `${seconds}s ago`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
    if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
    return `${Math.floor(seconds / 86400)}d ago`;
  };

  // Default status if none provided
  const defaultStatus = {
    status: 'inactive',
    markets: [],
    markets_active: 0,
    snapshots_processed: 0,
    deltas_processed: 0,
    messages_per_second: 0,
    total_messages: 0,
    uptime_seconds: 0,
    lastUpdate: null
  };

  const displayStatus = status || defaultStatus;

  // Calculate uptime
  const formatUptime = (seconds) => {
    if (!seconds) return '00:00:00';
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  // Determine status color
  const getStatusColor = () => {
    switch (displayStatus.status) {
      case 'active':
        return 'text-green-400';
      case 'connected':
        return 'text-blue-400';
      case 'inactive':
        return 'text-yellow-400';
      case 'error':
        return 'text-red-400';
      default:
        return 'text-gray-400';
    }
  };

  return (
    <div className="space-y-4">
      {/* Service Status */}
      <div className="flex items-center justify-between">
        <span className="text-sm text-gray-400">Service</span>
        <div className="flex items-center space-x-2">
          <div className={`w-2 h-2 rounded-full ${
            displayStatus.status === 'active' ? 'bg-green-400 animate-pulse' : 'bg-gray-500'
          }`} />
          <span className={`text-sm font-medium ${getStatusColor()}`}>
            {displayStatus.status?.toUpperCase() || 'UNKNOWN'}
          </span>
        </div>
      </div>

      {/* Uptime */}
      <div className="flex items-center justify-between">
        <span className="text-sm text-gray-400">Uptime</span>
        <span className="text-sm font-mono text-gray-300">
          {formatUptime(displayStatus.uptime_seconds)}
        </span>
      </div>

      {/* Markets Being Monitored */}
      {displayStatus.markets && displayStatus.markets.length > 0 && (
        <div className="bg-gray-700/30 rounded p-3 space-y-2">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-semibold text-gray-400 uppercase">Active Markets</span>
            <span className="text-xs text-gray-300 font-mono">
              {displayStatus.markets_active || displayStatus.markets.length}
            </span>
          </div>
          
          <div className="space-y-1 max-h-32 overflow-y-auto">
            {displayStatus.markets.map((market, index) => (
              <div 
                key={market} 
                className="text-xs font-mono text-gray-400 hover:text-gray-300 transition-colors flex items-center space-x-1"
              >
                <span className="text-blue-400">›</span>
                <span className="truncate">{market}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Collection Metrics */}
      <div className="space-y-3">
        <h4 className="text-xs font-semibold text-gray-400 uppercase">Collection Metrics</h4>
        
        <div className="grid grid-cols-2 gap-2">
          <MetricCard
            label="Snapshots"
            value={formatNumber(displayStatus.snapshots_processed || 0)}
          />
          <MetricCard
            label="Deltas"
            value={formatNumber(displayStatus.deltas_processed || 0)}
          />
        </div>
        
        <div className="grid grid-cols-2 gap-2">
          <MetricCard
            label="Total Msgs"
            value={formatNumber(displayStatus.total_messages || 0)}
          />
          <MetricCard
            label="Msgs/sec"
            value={displayStatus.messages_per_second?.toFixed(1) || '0.0'}
          />
        </div>
      </div>

      {/* Database Stats (if available) */}
      {displayStatus.database_stats && (
        <div className="space-y-2">
          <h4 className="text-xs font-semibold text-gray-400 uppercase">Database</h4>
          
          <div className="bg-gray-700/30 rounded p-2 space-y-1">
            <div className="flex justify-between items-center">
              <span className="text-xs text-gray-400">Queue Size</span>
              <span className="text-xs font-mono text-gray-300">
                {displayStatus.database_stats.queue_size || 0}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-xs text-gray-400">Write Rate</span>
              <span className="text-xs font-mono text-gray-300">
                {displayStatus.database_stats.write_rate?.toFixed(1) || '0.0'}/s
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Last Update */}
      <div className="pt-2 border-t border-gray-700">
        <div className="flex items-center justify-between">
          <span className="text-xs text-gray-500">Last Update</span>
          <span className="text-xs text-gray-400">
            {formatTimeAgo(displayStatus.lastUpdate)}
          </span>
        </div>
      </div>

      {/* Connection Quality Indicator */}
      {displayStatus.messages_per_second !== undefined && (
        <div className="bg-gray-700/30 rounded p-2">
          <div className="flex items-center justify-between mb-1">
            <span className="text-xs text-gray-400">Connection Quality</span>
            <ConnectionQuality messagesPerSecond={displayStatus.messages_per_second} />
          </div>
          <div className="w-full h-1 bg-gray-600 rounded overflow-hidden">
            <div 
              className={`h-full transition-all duration-500 ${
                displayStatus.messages_per_second > 10 ? 'bg-green-400' :
                displayStatus.messages_per_second > 1 ? 'bg-yellow-400' :
                displayStatus.messages_per_second > 0 ? 'bg-orange-400' : 'bg-red-400'
              }`}
              style={{ 
                width: `${Math.min(100, (displayStatus.messages_per_second / 20) * 100)}%` 
              }}
            />
          </div>
        </div>
      )}
    </div>
  );
};

// Metric Card Component
const MetricCard = ({ label, value }) => {
  return (
    <div className="bg-gray-700/30 rounded p-2">
      <p className="text-xs text-gray-400">{label}</p>
      <p className="text-sm font-mono text-gray-300">{value}</p>
    </div>
  );
};

// Connection Quality Component
const ConnectionQuality = ({ messagesPerSecond }) => {
  let quality = 'Poor';
  let color = 'text-red-400';
  
  if (messagesPerSecond > 10) {
    quality = 'Excellent';
    color = 'text-green-400';
  } else if (messagesPerSecond > 5) {
    quality = 'Good';
    color = 'text-green-400';
  } else if (messagesPerSecond > 1) {
    quality = 'Fair';
    color = 'text-yellow-400';
  } else if (messagesPerSecond > 0) {
    quality = 'Poor';
    color = 'text-orange-400';
  }
  
  return <span className={`text-xs font-medium ${color}`}>{quality}</span>;
};

export default CollectionStatus;