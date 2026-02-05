import React, { memo } from 'react';
import { Activity, Wifi, WifiOff, Radio, Clock, Pause, Play } from 'lucide-react';

/**
 * FeedStatusBadge - Single feed indicator with count and age
 */
const FeedStatusBadge = memo(({ label, count, ageSeconds, isActive }) => {
  const ageDisplay = ageSeconds != null
    ? ageSeconds < 60
      ? `${Math.round(ageSeconds)}s`
      : `${Math.floor(ageSeconds / 60)}m`
    : '--';

  const isStale = ageSeconds != null && ageSeconds > 30;
  const isVeryStale = ageSeconds != null && ageSeconds > 60;

  const colorClass = !isActive
    ? 'text-gray-600'
    : isVeryStale
      ? 'text-red-400'
      : isStale
        ? 'text-amber-400'
        : 'text-emerald-400';

  const dotColor = !isActive
    ? 'bg-gray-600'
    : isVeryStale
      ? 'bg-red-500'
      : isStale
        ? 'bg-amber-500'
        : 'bg-emerald-500';

  return (
    <div className="flex items-center gap-1.5" title={`${label}: ${count} updates, last ${ageDisplay} ago`}>
      <span className={`w-1.5 h-1.5 rounded-full ${dotColor}`} />
      <span className={`text-[10px] font-mono ${colorClass}`}>
        {label}
      </span>
      <span className={`text-[10px] font-mono font-bold ${colorClass}`}>
        {count}
      </span>
      {isActive && ageSeconds != null && (
        <span className={`text-[10px] font-mono opacity-60 ${colorClass}`}>
          {ageDisplay}
        </span>
      )}
    </div>
  );
});
FeedStatusBadge.displayName = 'FeedStatusBadge';

/**
 * FeedStatusBar - Row of feed status indicators
 */
const FeedStatusBar = memo(({ feedStats }) => {
  if (!feedStats?.feeds) return null;

  const { orderbook, ticker, trade, poll } = feedStats.feeds;

  return (
    <div
      id="feed-status-bar"
      data-testid="feed-status-bar"
      className="flex items-center gap-4 px-3 py-1.5 bg-gray-900/50 rounded-lg border border-gray-800/50"
    >
      <Radio className="w-3.5 h-3.5 text-gray-500" />
      <FeedStatusBadge
        label="OB"
        count={orderbook?.count ?? 0}
        ageSeconds={orderbook?.age_seconds}
        isActive={orderbook?.count > 0}
      />
      <FeedStatusBadge
        label="Tick"
        count={ticker?.count ?? 0}
        ageSeconds={ticker?.age_seconds}
        isActive={ticker?.count > 0}
      />
      <FeedStatusBadge
        label="Trade"
        count={trade?.count ?? 0}
        ageSeconds={trade?.age_seconds}
        isActive={trade?.count > 0}
      />
      <FeedStatusBadge
        label="REST"
        count={poll?.count ?? 0}
        ageSeconds={poll?.age_seconds}
        isActive={poll?.count > 0}
      />
    </div>
  );
});
FeedStatusBar.displayName = 'FeedStatusBar';

/**
 * ArbHeader - Top bar for the arbitrage dashboard
 *
 * Shows title, connection dot + text, system state badge, feed status, and captain pause toggle.
 */
const ArbHeader = ({ connectionStatus, systemState, feedStats, captainPaused, onCaptainPauseToggle }) => {
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
    <div id="arb-header" data-testid="arb-header" className="border-b border-gray-800 bg-black/30 backdrop-blur-sm sticky top-0 z-10">
      <div className="max-w-[1600px] mx-auto px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <Activity className="w-6 h-6 text-cyan-400" />
            <h1 className="text-xl font-semibold text-white">KalshiFlow Arb</h1>
            <span className="px-2 py-0.5 text-xs font-mono bg-violet-500/20 text-violet-300 rounded-full border border-violet-500/30">
              DASHBOARD
            </span>
          </div>

          <div className="flex items-center space-x-4">
            {/* Captain Pause Toggle */}
            <button
              onClick={onCaptainPauseToggle}
              disabled={!isConnected}
              title={captainPaused ? 'Resume Captain' : 'Pause Captain'}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg border transition-colors ${
                !isConnected
                  ? 'bg-gray-800/30 text-gray-600 border-gray-700/30 cursor-not-allowed'
                  : captainPaused
                    ? 'bg-amber-900/30 text-amber-400 border-amber-600/30 hover:bg-amber-900/50'
                    : 'bg-gray-800/50 text-gray-300 border-gray-700/40 hover:bg-gray-700/50'
              }`}
            >
              {captainPaused ? (
                <>
                  <Play className="w-3.5 h-3.5" />
                  <span className="text-xs font-medium">Resume</span>
                </>
              ) : (
                <>
                  <Pause className="w-3.5 h-3.5" />
                  <span className="text-xs font-medium">Pause</span>
                </>
              )}
            </button>

            {/* Feed Status */}
            <FeedStatusBar feedStats={feedStats} />

            <div
              id="connection-status"
              data-testid="connection-status"
              data-connected={isConnected}
              className="flex items-center space-x-2"
            >
              {isConnected ? (
                <>
                  <span className="relative flex h-2.5 w-2.5">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75" />
                    <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-green-500" />
                  </span>
                  <Wifi className="w-4 h-4 text-green-400" />
                  <span data-testid="connection-label" className="text-sm text-green-400 font-medium">Live</span>
                </>
              ) : (
                <>
                  <span className="inline-flex rounded-full h-2.5 w-2.5 bg-red-500" />
                  <WifiOff className="w-4 h-4 text-red-400" />
                  <span data-testid="connection-label" className="text-sm text-red-400 font-medium">Disconnected</span>
                </>
              )}
            </div>

            <span
              id="system-state"
              data-testid="system-state"
              data-state={systemState}
              className={`px-3 py-1 rounded-full text-[10px] font-bold border uppercase tracking-wider ${stateBadgeColor}`}
            >
              {stateLabel}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default memo(ArbHeader);
