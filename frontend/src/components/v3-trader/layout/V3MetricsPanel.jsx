import React, { memo } from 'react';
import { getPingHealthColor, getPingAgeColor } from '../../../utils/v3-trader';

/**
 * V3MetricsPanel - System metrics sidebar panel
 *
 * Shows V3-specific metrics:
 * - Tracked Markets: Markets tracked via lifecycle discovery
 * - OB Subscriptions: Active orderbook WebSocket subscriptions
 * - Signal Buckets: Markets with active 10-sec signal aggregation
 * - Signals Flushed: Count of signal buckets written to DB
 */
const V3MetricsPanel = ({ metrics }) => {
  // Extract signal aggregator stats
  const signalStats = metrics.signal_aggregator || {};

  return (
    <div className="bg-gray-900/50 backdrop-blur-sm rounded-xl border border-gray-800 p-6 space-y-5">
      <h3 className="text-sm font-bold text-gray-300 uppercase tracking-wider">System Metrics</h3>

      <div className="space-y-4">
        {/* Core Market Metrics */}
        <div className="space-y-3">
          <div className="flex justify-between items-center">
            <span className="text-sm text-gray-500">Tracked Markets</span>
            <span className="text-lg font-mono font-bold text-white">
              {metrics.tracked_markets ?? metrics.markets_connected ?? 0}
            </span>
          </div>

          <div className="flex justify-between items-center">
            <span className="text-sm text-gray-500">OB Subscriptions</span>
            <span className="text-lg font-mono font-bold text-blue-400">
              {metrics.subscribed_markets ?? 0}
            </span>
          </div>

          <div className="flex justify-between items-center">
            <span className="text-sm text-gray-500">Signal Buckets</span>
            <span className="text-lg font-mono font-bold text-purple-400">
              {signalStats.active_buckets ?? 0}
            </span>
          </div>

          <div className="flex justify-between items-center">
            <span className="text-sm text-gray-500">Signals Flushed</span>
            <span className="text-lg font-mono font-bold text-emerald-400">
              {signalStats.signals_flushed ?? 0}
            </span>
          </div>
        </div>

        {/* Connection Health */}
        <div className="pt-4 border-t border-gray-700 space-y-3">
          <div className="flex justify-between items-center">
            <span className="text-sm text-gray-500">Ping Health</span>
            <div className={`px-3 py-1 rounded-full text-xs font-bold ${getPingHealthColor(metrics.ping_health)}`}>
              {metrics.ping_health?.toUpperCase() || 'UNKNOWN'}
            </div>
          </div>

          <div className="flex justify-between items-center">
            <span className="text-sm text-gray-500">Last Message</span>
            <span className={`text-lg font-mono font-bold ${getPingAgeColor(metrics.last_ping_age)}`}>
              {metrics.last_ping_age !== null ? `${Math.floor(metrics.last_ping_age)}s ago` : 'N/A'}
            </span>
          </div>

          <div className="flex justify-between items-center">
            <span className="text-sm text-gray-500">Uptime</span>
            <span className="text-lg font-mono font-bold text-green-400">
              {metrics.uptime ? `${Math.floor(metrics.uptime)}s` : '0s'}
            </span>
          </div>
        </div>

        {/* Health Status */}
        <div className="pt-4 border-t border-gray-700">
          <div className="flex justify-between items-center">
            <span className="text-sm text-gray-500">Health Status</span>
            <div className={`px-3 py-1 rounded-full text-xs font-bold ${
              metrics.health === 'healthy' ? 'bg-green-900/30 text-green-400' :
              metrics.health === 'unhealthy' ? 'bg-red-900/30 text-red-400' :
              'bg-gray-900/30 text-gray-400'
            }`}>
              {metrics.health?.toUpperCase() || 'UNKNOWN'}
            </div>
          </div>
        </div>
      </div>

      {/* API Status */}
      <div className="pt-4 border-t border-gray-700">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs text-gray-500 uppercase tracking-wider">API Status</span>
          <div className={`px-3 py-1 rounded-full text-xs font-bold ${
            metrics.api_connected ? 'bg-green-900/30 text-green-400' : 'bg-red-900/30 text-red-400'
          }`}>
            {metrics.api_connected ? 'CONNECTED' : 'DISCONNECTED'}
          </div>
        </div>
        {metrics.api_url && (
          <div className="mt-2 space-y-2">
            <div>
              <span className="text-xs text-gray-500">API:</span>
              <div className="text-xs text-gray-300 font-mono mt-1 truncate" title={metrics.api_url}>
                {metrics.api_url.replace('https://', '').replace('/trade-api/v2', '')}
              </div>
            </div>
            {metrics.ws_url && (
              <div>
                <span className="text-xs text-gray-500">WebSocket:</span>
                <div className="text-xs text-gray-300 font-mono mt-1 truncate" title={metrics.ws_url}>
                  {metrics.ws_url.replace('wss://', '').replace('/trade-api/ws/v2', '')}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default memo(V3MetricsPanel);
