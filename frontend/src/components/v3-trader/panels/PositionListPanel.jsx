import React, { useState, useEffect, useRef, memo } from 'react';
import { ChevronRight, ChevronDown, Briefcase } from 'lucide-react';
import { formatPnLCurrency, formatRelativeTime, getSideClasses, getPnLColor } from '../../../utils/v3-trader';

/**
 * PositionRow - Memoized row component for position table
 */
const PositionRow = memo(({ pos, index, isRecentlyChanged }) => {
  const qty = Math.abs(pos.position);
  const costPerContract = qty > 0 ? Math.round((pos.total_traded || 0) / qty) : 0;

  // Value fallback hierarchy
  const kalshiValuePerContract = qty > 0 ? Math.round((pos.market_exposure || 0) / qty) : 0;
  let valuePerContract;
  if (pos.market_bid > 0) {
    valuePerContract = pos.market_bid;
  } else if (pos.market_last > 0) {
    valuePerContract = pos.side === 'no' ? (100 - pos.market_last) : pos.market_last;
  } else {
    valuePerContract = kalshiValuePerContract;
  }

  const unrealizedPerContract = valuePerContract - costPerContract;
  const unrealizedTotal = unrealizedPerContract * qty;
  const pnlPercent = costPerContract > 0
    ? ((unrealizedPerContract / costPerContract) * 100).toFixed(1)
    : 0;

  return (
    <tr
      className={`border-b border-gray-700/30 hover:bg-gray-800/50 transition-all duration-500
        ${isRecentlyChanged
          ? 'border-l-2 border-l-emerald-400/70 bg-emerald-900/10'
          : 'border-l-2 border-l-transparent'
        }`}
    >
      <td className="px-3 py-2 font-mono text-gray-300 text-xs">
        <div className="flex items-center">
          {isRecentlyChanged ? (
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 mr-2 animate-pulse" />
          ) : pos.session_updated ? (
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-600/50 mr-2" title="Updated this session" />
          ) : null}
          {pos.ticker}
        </div>
      </td>
      <td className="px-3 py-2 text-center">
        <span className={`px-2 py-0.5 rounded text-xs font-bold uppercase ${
          pos.side === 'yes'
            ? 'bg-green-900/30 text-green-400 border border-green-700/50'
            : 'bg-red-900/30 text-red-400 border border-red-700/50'
        }`}>
          {pos.side}
        </span>
      </td>
      <td className="px-3 py-2 text-right font-mono text-gray-300">{qty}</td>
      <td className="px-3 py-2 text-right font-mono text-gray-400">{costPerContract}c</td>
      <td className="px-3 py-2 text-right font-mono text-gray-300">{valuePerContract}c</td>
      <td className={`px-3 py-2 text-right font-mono ${getPnLColor(unrealizedPerContract)}`}>
        {unrealizedPerContract >= 0 ? '+' : ''}{unrealizedPerContract}c
      </td>
      <td className={`px-3 py-2 text-right font-mono font-bold ${getPnLColor(unrealizedTotal)}`}>
        {formatPnLCurrency(unrealizedTotal)}
        <span className="text-xs ml-1 opacity-70">({pnlPercent}%)</span>
      </td>
      <td className="px-3 py-2 text-center font-mono text-xs">
        {pos.market_close_time ? (
          <span className="text-gray-400" title={new Date(pos.market_close_time).toLocaleString()}>
            {formatRelativeTime(pos.market_close_time)}
          </span>
        ) : (
          <span className="text-gray-600">-</span>
        )}
      </td>
    </tr>
  );
});

PositionRow.displayName = 'PositionRow';

/**
 * Calculate real P&L from market prices with fallback hierarchy
 */
const calcRealPnL = (pos) => {
  const qty = Math.abs(pos.position || 0);
  if (qty === 0) return 0;
  const costPerContract = Math.round((pos.total_traded || 0) / qty);
  const kalshiValue = Math.round((pos.market_exposure || 0) / qty);

  let valuePerContract;
  if (pos.market_bid > 0) {
    valuePerContract = pos.market_bid;
  } else if (pos.market_last > 0) {
    valuePerContract = pos.side === 'no' ? (100 - pos.market_last) : pos.market_last;
  } else {
    valuePerContract = kalshiValue;
  }
  return (valuePerContract - costPerContract) * qty;
};

/**
 * PositionListPanel - Detailed position breakdown with per-position P&L
 */
const PositionListPanel = ({ positions, positionListener, sessionUpdates }) => {
  const [isExpanded, setIsExpanded] = useState(true);
  const [changedTickers, setChangedTickers] = useState(new Set());
  const prevPositionsRef = useRef({});
  const isFirstRender = useRef(true);

  // Detect position changes and trigger subtle highlight animation
  useEffect(() => {
    if (!positions || positions.length === 0) return;

    if (isFirstRender.current) {
      isFirstRender.current = false;
      const newRef = {};
      positions.forEach(pos => { newRef[pos.ticker] = { ...pos }; });
      prevPositionsRef.current = newRef;
      return;
    }

    const changed = new Set();
    positions.forEach(pos => {
      const prev = prevPositionsRef.current[pos.ticker];
      if (prev) {
        if (prev.position !== pos.position ||
            prev.total_traded !== pos.total_traded ||
            prev.unrealized_pnl !== pos.unrealized_pnl ||
            prev.market_bid !== pos.market_bid ||
            prev.market_last !== pos.market_last) {
          changed.add(pos.ticker);
        }
      } else {
        changed.add(pos.ticker);
      }
    });

    const newRef = {};
    positions.forEach(pos => {
      newRef[pos.ticker] = { ...pos };
    });
    prevPositionsRef.current = newRef;

    if (changed.size > 0) {
      setChangedTickers(changed);
      const timeout = setTimeout(() => setChangedTickers(new Set()), 2000);
      return () => clearTimeout(timeout);
    }
  }, [positions]);

  if (!positions || positions.length === 0) {
    return null;
  }

  // Calculate YES/NO summary totals
  const yesPositions = positions.filter(p => p.side === 'yes');
  const noPositions = positions.filter(p => p.side === 'no');
  const yesTotalQty = yesPositions.reduce((sum, p) => sum + Math.abs(p.position || 0), 0);
  const noTotalQty = noPositions.reduce((sum, p) => sum + Math.abs(p.position || 0), 0);
  const yesTotalPnL = yesPositions.reduce((sum, p) => sum + calcRealPnL(p), 0);
  const noTotalPnL = noPositions.reduce((sum, p) => sum + calcRealPnL(p), 0);

  return (
    <div className="bg-gray-900/50 backdrop-blur-sm rounded-xl border border-gray-800 p-4 mb-4">
      <div
        className="flex items-center justify-between cursor-pointer"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center space-x-2">
          <Briefcase className="w-4 h-4 text-purple-400" />
          <h3 className="text-sm font-bold text-gray-300 uppercase tracking-wider">Open Positions</h3>
          <span className="text-xs text-gray-500">({positions.length})</span>
          {/* Position listener status indicator */}
          {positionListener && (
            <div className="flex items-center gap-1.5 ml-3 px-2 py-0.5 bg-gray-800/50 rounded-full">
              <span className={`w-1.5 h-1.5 rounded-full ${
                positionListener.connected
                  ? 'bg-emerald-400 animate-pulse'
                  : 'bg-yellow-400'
              }`} />
              <span className="text-xs text-gray-500">
                {positionListener.connected ? 'Live' : 'Polling'}
              </span>
              {positionListener.positions_received > 0 && (
                <span className="text-xs text-gray-600">
                  ({positionListener.positions_received} updates)
                </span>
              )}
            </div>
          )}
          {/* Session updates count */}
          {sessionUpdates && sessionUpdates.count > 0 && (
            <div className="flex items-center gap-1.5 ml-2 px-2 py-0.5 bg-emerald-900/30 rounded-full border border-emerald-800/30">
              <span className="text-xs text-emerald-400 font-medium">
                {sessionUpdates.count} updated this session
              </span>
            </div>
          )}
        </div>
        {isExpanded ? (
          <ChevronDown className="w-4 h-4 text-gray-500" />
        ) : (
          <ChevronRight className="w-4 h-4 text-gray-500" />
        )}
      </div>

      {isExpanded && (
        <div className="mt-4 bg-gray-800/30 rounded-lg border border-gray-700/50 overflow-hidden max-h-[280px] overflow-y-auto">
          <table className="w-full text-sm">
            <thead className="sticky top-0 z-10">
              <tr className="bg-gray-900 border-b border-gray-700/50">
                <th className="px-3 py-2 text-left text-xs text-gray-500 uppercase font-medium">Ticker</th>
                <th className="px-3 py-2 text-center text-xs text-gray-500 uppercase font-medium">Side</th>
                <th className="px-3 py-2 text-right text-xs text-gray-500 uppercase font-medium">Qty</th>
                <th className="px-3 py-2 text-right text-xs text-gray-500 uppercase font-medium" title="Entry cost per contract">Cost/C</th>
                <th className="px-3 py-2 text-right text-xs text-gray-500 uppercase font-medium" title="Current market bid">Value/C</th>
                <th className="px-3 py-2 text-right text-xs text-gray-500 uppercase font-medium" title="Unrealized P&L per contract">Unreal/C</th>
                <th className="px-3 py-2 text-right text-xs text-gray-500 uppercase font-medium">P&L</th>
                <th className="px-3 py-2 text-center text-xs text-gray-500 uppercase font-medium" title="When market closes">Closes</th>
              </tr>
              <tr className="bg-emerald-900/50 text-emerald-400 text-sm font-medium border-b border-emerald-800/30">
                <td className="py-1.5 px-3" colSpan="2">YES ({yesPositions.length})</td>
                <td className="py-1.5 px-3 text-right font-mono">{yesTotalQty}</td>
                <td className="py-1.5 px-3 text-right font-mono text-gray-500">-</td>
                <td className="py-1.5 px-3 text-right font-mono text-gray-500">-</td>
                <td className="py-1.5 px-3 text-right font-mono text-gray-500">-</td>
                <td className={`py-1.5 px-3 text-right font-mono ${getPnLColor(yesTotalPnL)}`}>{formatPnLCurrency(yesTotalPnL)}</td>
                <td className="py-1.5 px-3 text-center text-gray-500">-</td>
              </tr>
              <tr className="bg-red-900/50 text-red-400 text-sm font-medium border-b border-gray-700/50">
                <td className="py-1.5 px-3" colSpan="2">NO ({noPositions.length})</td>
                <td className="py-1.5 px-3 text-right font-mono">{noTotalQty}</td>
                <td className="py-1.5 px-3 text-right font-mono text-gray-500">-</td>
                <td className="py-1.5 px-3 text-right font-mono text-gray-500">-</td>
                <td className="py-1.5 px-3 text-right font-mono text-gray-500">-</td>
                <td className={`py-1.5 px-3 text-right font-mono ${getPnLColor(noTotalPnL)}`}>{formatPnLCurrency(noTotalPnL)}</td>
                <td className="py-1.5 px-3 text-center text-gray-500">-</td>
              </tr>
            </thead>
            <tbody>
              {positions.map((pos, index) => (
                <PositionRow
                  key={pos.ticker || index}
                  pos={pos}
                  index={index}
                  isRecentlyChanged={changedTickers.has(pos.ticker)}
                />
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

export default memo(PositionListPanel);
