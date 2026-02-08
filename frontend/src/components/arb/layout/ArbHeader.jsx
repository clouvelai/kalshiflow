import React, { memo } from 'react';
import { Activity, Wifi, WifiOff, Radio, Pause, Play, AlertTriangle, CheckCircle } from 'lucide-react';
import { HEADER_HEIGHT } from '../utils/styleConstants';

const FeedStatusBadge = memo(({ label, count, ageSeconds, isActive }) => {
  const ageDisplay = ageSeconds != null
    ? ageSeconds < 60 ? `${Math.round(ageSeconds)}s` : `${Math.floor(ageSeconds / 60)}m`
    : '--';

  const isStale = ageSeconds != null && ageSeconds > 30;
  const isVeryStale = ageSeconds != null && ageSeconds > 60;

  const colorClass = !isActive ? 'text-gray-600'
    : isVeryStale ? 'text-red-400'
    : isStale ? 'text-amber-400'
    : 'text-emerald-400';

  const dotColor = !isActive ? 'bg-gray-600'
    : isVeryStale ? 'bg-red-500'
    : isStale ? 'bg-amber-500'
    : 'bg-emerald-500';

  return (
    <div className="flex items-center gap-1" title={`${label}: ${count} updates, last ${ageDisplay} ago`}>
      <span className={`w-1 h-1 rounded-full ${dotColor}`} />
      <span className={`text-[9px] font-mono ${colorClass}`}>{label}</span>
      <span className={`text-[9px] font-mono font-semibold tabular-nums ${colorClass}`}>{count}</span>
    </div>
  );
});
FeedStatusBadge.displayName = 'FeedStatusBadge';

const FeedStatusBar = memo(({ feedStats }) => {
  if (!feedStats?.feeds) return null;
  const { orderbook, ticker, trade, poll } = feedStats.feeds;

  return (
    <div
      data-testid="feed-status-bar"
      className="flex items-center gap-2.5 px-2 py-0.5 bg-gray-900/40 rounded-md border border-gray-800/30"
    >
      <Radio className="w-2.5 h-2.5 text-gray-500" />
      <FeedStatusBadge label="OB" count={orderbook?.count ?? 0} ageSeconds={orderbook?.age_seconds} isActive={orderbook?.count > 0} />
      <FeedStatusBadge label="Tick" count={ticker?.count ?? 0} ageSeconds={ticker?.age_seconds} isActive={ticker?.count > 0} />
      <FeedStatusBadge label="Trade" count={trade?.count ?? 0} ageSeconds={trade?.age_seconds} isActive={trade?.count > 0} />
      <FeedStatusBadge label="REST" count={poll?.count ?? 0} ageSeconds={poll?.age_seconds} isActive={poll?.count > 0} />
    </div>
  );
});
FeedStatusBar.displayName = 'FeedStatusBar';

const ArbHeader = ({ connectionStatus, systemState, feedStats, captainPaused, onCaptainPauseToggle, exchangeStatus }) => {
  const isConnected = connectionStatus === 'connected';
  const stateLabel = (systemState || 'initializing').replace(/_/g, ' ').toUpperCase();
  const isActive = exchangeStatus?.active ?? true;

  const STATE_BADGE_COLORS = {
    ready: 'bg-emerald-500/10 text-emerald-400/80 border-emerald-500/15',
    trading: 'bg-emerald-500/10 text-emerald-400/80 border-emerald-500/15',
    acting: 'bg-emerald-500/10 text-emerald-400/80 border-emerald-500/15',
    error: 'bg-red-500/10 text-red-400/80 border-red-500/15',
    calibrating: 'bg-amber-500/10 text-amber-400/80 border-amber-500/15',
  };
  const stateBadgeColor = STATE_BADGE_COLORS[systemState] || 'bg-gray-800/40 text-gray-400 border-gray-700/20';

  return (
    <div
      id="arb-header"
      data-testid="arb-header"
      className="border-b border-gray-800/60 bg-black/60 backdrop-blur-sm shrink-0 z-10"
      style={{ height: HEADER_HEIGHT }}
    >
      <div className="h-full px-4 flex items-center justify-between">
        {/* Left: branding */}
        <div className="flex items-center gap-2">
          <Activity className="w-4 h-4 text-cyan-400" />
          <h1 className="text-[13px] font-semibold text-white tracking-tight">KalshiFlow</h1>
          <span className="px-1.5 py-0.5 text-[8px] font-mono font-semibold bg-violet-500/12 text-violet-300/80 rounded-full border border-violet-500/15 uppercase tracking-wider">
            Arb
          </span>
        </div>

        {/* Right: controls */}
        <div className="flex items-center gap-2.5">
          {/* Exchange */}
          {isActive ? (
            <div className="flex items-center gap-1 px-1.5 py-0.5 rounded-md bg-emerald-900/20 border border-emerald-500/15" title="Exchange active">
              <CheckCircle className="w-2.5 h-2.5 text-emerald-400/70" />
              <span className="text-[8px] font-medium text-emerald-400/80 uppercase tracking-wider">Exch</span>
            </div>
          ) : (
            <div className="flex items-center gap-1 px-1.5 py-0.5 rounded-md bg-red-900/25 border border-red-500/20 animate-pulse" title={exchangeStatus?.error || 'Exchange unavailable'}>
              <AlertTriangle className="w-2.5 h-2.5 text-red-400" />
              <span className="text-[8px] font-medium text-red-400 uppercase tracking-wider">Down</span>
            </div>
          )}

          {/* Pause/Resume */}
          <button
            onClick={onCaptainPauseToggle}
            disabled={!isConnected}
            title={captainPaused ? 'Resume Captain' : 'Pause Captain'}
            className={`flex items-center gap-1 px-2 py-0.5 rounded-md border text-[10px] font-medium transition-colors ${
              !isConnected
                ? 'bg-gray-800/20 text-gray-600 border-gray-700/20 cursor-not-allowed'
                : captainPaused
                  ? 'bg-amber-900/20 text-amber-400 border-amber-500/20 hover:bg-amber-900/30'
                  : 'bg-gray-800/30 text-gray-300 border-gray-700/25 hover:bg-gray-700/40'
            }`}
          >
            {captainPaused
              ? <><Play className="w-2.5 h-2.5" /> Resume</>
              : <><Pause className="w-2.5 h-2.5" /> Pause</>
            }
          </button>

          <FeedStatusBar feedStats={feedStats} />

          {/* Connection */}
          <div
            id="connection-status"
            data-testid="connection-status"
            data-connected={isConnected}
            className="flex items-center gap-1"
          >
            {isConnected ? (
              <>
                <span className="relative flex h-1.5 w-1.5">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-50" />
                  <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-green-500" />
                </span>
                <Wifi className="w-3 h-3 text-green-400" />
                <span data-testid="connection-label" className="text-[10px] text-green-400 font-medium">Live</span>
              </>
            ) : (
              <>
                <span className="inline-flex rounded-full h-1.5 w-1.5 bg-red-500" />
                <WifiOff className="w-3 h-3 text-red-400" />
                <span data-testid="connection-label" className="text-[10px] text-red-400 font-medium">Off</span>
              </>
            )}
          </div>

          {/* State badge */}
          <span
            id="system-state"
            data-testid="system-state"
            data-state={systemState}
            className={`px-2 py-0.5 rounded-full text-[8px] font-semibold border uppercase tracking-wider ${stateBadgeColor}`}
          >
            {stateLabel}
          </span>
        </div>
      </div>
    </div>
  );
};

export default memo(ArbHeader);
