import React, { memo } from 'react';
import { Activity, Wifi, WifiOff, Pause, Play } from 'lucide-react';

const HEADER_HEIGHT = 44;

const MMHeader = ({ connectionStatus, quoteState, performance, balanceInfo, onPauseToggle, isPaused }) => {
  const isConnected = connectionStatus === 'connected';
  const activeQuotes = quoteState?.active_quotes ?? 0;
  const realizedPnL = balanceInfo?.total_realized_pnl_cents ?? (performance?.realized_pnl_cents ?? 0);
  const unrealizedPnL = balanceInfo?.total_unrealized_pnl_cents ?? 0;
  const totalPnL = realizedPnL + unrealizedPnL;
  const pnlDollars = (totalPnL / 100).toFixed(2);
  const pnlColor = totalPnL > 0 ? 'text-emerald-400' : totalPnL < 0 ? 'text-red-400' : 'text-gray-500';
  const balanceDollars = balanceInfo?.balance_cents != null ? (balanceInfo.balance_cents / 100).toFixed(2) : null;

  return (
    <div
      className="border-b border-gray-800/60 bg-black/60 backdrop-blur-sm shrink-0 z-10"
      style={{ height: HEADER_HEIGHT }}
    >
      <div className="h-full px-4 flex items-center justify-between">
        {/* Left: branding */}
        <div className="flex items-center gap-2">
          <Activity className="w-4 h-4 text-cyan-400" />
          <h1 className="text-[13px] font-semibold text-white tracking-tight">KalshiFlow</h1>
          <span className="px-1.5 py-0.5 text-[8px] font-mono font-semibold bg-amber-500/12 text-amber-300/80 rounded-full border border-amber-500/15 uppercase tracking-wider">
            Admiral
          </span>
        </div>

        {/* Center: key metrics */}
        <div className="flex items-center gap-4">
          {balanceDollars != null && (
            <div className="flex items-center gap-1.5">
              <span className="text-[9px] text-gray-500 uppercase tracking-wider">Balance</span>
              <span className="text-[11px] font-mono font-semibold tabular-nums text-gray-300">
                ${balanceDollars}
              </span>
            </div>
          )}
          <div className="flex items-center gap-1.5">
            <span className="text-[9px] text-gray-500 uppercase tracking-wider">Quotes</span>
            <span className={`text-[11px] font-mono font-semibold tabular-nums ${activeQuotes > 0 ? 'text-emerald-400' : 'text-gray-600'}`}>
              {activeQuotes}
            </span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="text-[9px] text-gray-500 uppercase tracking-wider">P&L</span>
            <span className={`text-[11px] font-mono font-semibold tabular-nums ${pnlColor}`}>
              {totalPnL >= 0 ? '+' : ''}${pnlDollars}
            </span>
          </div>
          {quoteState?.spread_multiplier != null && quoteState.spread_multiplier !== 1 && (
            <div className="flex items-center gap-1">
              <span className="text-[9px] text-amber-400 uppercase tracking-wider">Spread</span>
              <span className="text-[10px] font-mono text-amber-300 tabular-nums">
                {quoteState.spread_multiplier.toFixed(1)}x
              </span>
            </div>
          )}
        </div>

        {/* Right: controls */}
        <div className="flex items-center gap-2.5">
          {/* Pause/Resume */}
          <button
            onClick={onPauseToggle}
            disabled={!isConnected}
            title={isPaused ? 'Resume Quotes' : 'Pause Quotes'}
            className={`flex items-center gap-1 px-2 py-0.5 rounded-md border text-[10px] font-medium transition-colors ${
              !isConnected
                ? 'bg-gray-800/20 text-gray-600 border-gray-700/20 cursor-not-allowed'
                : isPaused
                  ? 'bg-amber-900/20 text-amber-400 border-amber-500/20 hover:bg-amber-900/30'
                  : 'bg-gray-800/30 text-gray-300 border-gray-700/25 hover:bg-gray-700/40'
            }`}
          >
            {isPaused
              ? <><Play className="w-2.5 h-2.5" /> Resume</>
              : <><Pause className="w-2.5 h-2.5" /> Pause</>
            }
          </button>

          {/* Connection */}
          <div className="flex items-center gap-1">
            {isConnected ? (
              <>
                <span className="relative flex h-1.5 w-1.5">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-50" />
                  <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-green-500" />
                </span>
                <Wifi className="w-3 h-3 text-green-400" />
                <span className="text-[10px] text-green-400 font-medium">Live</span>
              </>
            ) : (
              <>
                <span className="inline-flex rounded-full h-1.5 w-1.5 bg-red-500" />
                <WifiOff className="w-3 h-3 text-red-400" />
                <span className="text-[10px] text-red-400 font-medium">Off</span>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default memo(MMHeader);
