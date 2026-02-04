import React, { memo } from 'react';
import SpreadBadge from '../ui/SpreadBadge';

/**
 * PairCard - Two-tier spread pair card
 *
 * Primary row: API prices (Kalshi + Poly) + Spread - big, prominent.
 * Secondary row: WS/orderbook BBO + age - small, muted, informational.
 * Untradeable pairs dimmed with reason badge.
 */
const PairCard = ({ pair }) => {
  const {
    question,
    kalshi_ticker,
    kalshi_yes_bid,
    kalshi_yes_ask,
    kalshi_yes_mid,
    poly_yes_cents,
    spread_cents,
    tradeable,
    tradeable_reason,
    // Kalshi WS
    kalshi_ws_yes_mid,
    kalshi_ws_yes_bid,
    kalshi_ws_yes_ask,
    kalshi_ws_age_ms,
    // Kalshi API
    kalshi_api_yes_mid,
    kalshi_api_yes_bid,
    kalshi_api_yes_ask,
    kalshi_api_age_ms,
    // Poly WS
    poly_ws_yes_cents,
    poly_ws_age_ms,
    // Poly API
    poly_api_yes_cents,
    poly_api_age_ms,
  } = pair;

  // Format age: <1s -> "0s", 1-59s -> "Xs", 60+ -> "Xm"
  const fmtAge = (ms) => {
    if (ms == null) return '--';
    const s = Math.round(ms / 1000);
    return s < 60 ? `${s}s` : `${Math.floor(s / 60)}m`;
  };

  // Color: green <5s, amber 5-30s, red >30s
  const ageColor = (ms) => {
    if (ms == null) return 'text-gray-600';
    const s = ms / 1000;
    if (s < 5) return 'text-emerald-500';
    if (s < 30) return 'text-amber-500';
    return 'text-red-400';
  };

  const fmtPrice = (cents) => cents != null ? `${cents}c` : '--';

  const isUntradeable = tradeable === false;

  return (
    <div className={`
      bg-gradient-to-br from-gray-900/70 via-gray-900/50 to-gray-950/60
      rounded-xl border border-gray-800/50
      p-3 transition-all duration-200
      hover:border-gray-700/60 hover:bg-gray-900/60
      ${isUntradeable ? 'opacity-45' : ''}
    `}>
      {/* Header: question + spread badge */}
      <div className="flex items-start justify-between gap-2 mb-2">
        <p className="text-sm text-gray-300 leading-snug line-clamp-2 flex-1">
          {question || kalshi_ticker}
        </p>
        <SpreadBadge spreadCents={spread_cents} />
      </div>

      {/* Untradeable reason badge */}
      {isUntradeable && tradeable_reason && (
        <div className="mb-2">
          <span className="inline-block text-[9px] font-medium text-amber-400/80 bg-amber-400/10 border border-amber-400/20 rounded px-1.5 py-0.5">
            {tradeable_reason}
          </span>
        </div>
      )}

      {/* === PRIMARY ROW: API prices (prominent) === */}
      <div className="grid grid-cols-3 gap-2 mb-2">
        {/* Kalshi API */}
        <div>
          <span className="text-[9px] text-gray-500 uppercase tracking-wider font-semibold block mb-0.5">Kalshi</span>
          <span className="font-mono text-base text-cyan-400 font-bold">
            {fmtPrice(kalshi_yes_mid)}
          </span>
          {kalshi_yes_bid != null && (
            <div className="text-[9px] text-gray-500 font-mono">
              {kalshi_yes_bid}b/{kalshi_yes_ask ?? '--'}a
            </div>
          )}
          {kalshi_api_age_ms != null && (
            <span className={`text-[9px] font-mono ${ageColor(kalshi_api_age_ms)}`}>
              {fmtAge(kalshi_api_age_ms)}
            </span>
          )}
        </div>

        {/* Poly API */}
        <div>
          <span className="text-[9px] text-gray-500 uppercase tracking-wider font-semibold block mb-0.5">Poly</span>
          <span className="font-mono text-base text-violet-400 font-bold">
            {fmtPrice(poly_yes_cents)}
          </span>
          {poly_api_age_ms != null && (
            <span className={`text-[9px] font-mono block ${ageColor(poly_api_age_ms)}`}>
              {fmtAge(poly_api_age_ms)}
            </span>
          )}
        </div>

        {/* Spread */}
        <div className="flex flex-col items-center justify-center">
          <span className="text-[9px] text-gray-500 uppercase tracking-wider font-semibold block mb-0.5">Spread</span>
          <span className={`font-mono text-lg font-bold ${
            spread_cents != null
              ? Math.abs(spread_cents) >= 5
                ? 'text-emerald-400'
                : Math.abs(spread_cents) >= 3
                  ? 'text-amber-400'
                  : 'text-gray-400'
              : 'text-gray-600'
          }`}>
            {spread_cents != null ? `${spread_cents >= 0 ? '+' : ''}${spread_cents}c` : '--'}
          </span>
        </div>
      </div>

      {/* === SECONDARY ROW: WS/orderbook (muted, informational) === */}
      <div className="flex items-center gap-3 text-[10px] text-gray-600 font-mono border-t border-gray-800/40 pt-1.5 mt-1">
        {/* Kalshi WS */}
        <div className="flex items-center gap-1">
          <span className="text-gray-700">WS:</span>
          <span className="text-gray-500">
            {kalshi_ws_yes_mid != null ? `${kalshi_ws_yes_mid}c` : '--'}
          </span>
          {kalshi_ws_yes_bid != null && (
            <span className="text-gray-700">
              ({kalshi_ws_yes_bid}b/{kalshi_ws_yes_ask ?? '--'}a)
            </span>
          )}
          <span className={ageColor(kalshi_ws_age_ms)}>
            {fmtAge(kalshi_ws_age_ms)}
          </span>
        </div>

        <span className="text-gray-800">|</span>

        {/* Poly WS */}
        <div className="flex items-center gap-1">
          <span className="text-gray-700">WS:</span>
          <span className="text-gray-500">
            {poly_ws_yes_cents != null ? `${poly_ws_yes_cents}c` : '--'}
          </span>
          <span className={ageColor(poly_ws_age_ms)}>
            {fmtAge(poly_ws_age_ms)}
          </span>
        </div>

        {/* Ticker (pushed right) */}
        <span className="ml-auto text-gray-600">{kalshi_ticker}</span>
      </div>

    </div>
  );
};

export default memo(PairCard);
