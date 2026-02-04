import React, { memo, useMemo } from 'react';
import { BookOpen, Check, X } from 'lucide-react';
import SpreadRibbon from '../charts/SpreadRibbon';

/**
 * EventDetailsPanel - Detailed view of a selected event.
 *
 * Shows: Kalshi event metadata (title, subtitle, ticker, category, # markets)
 * and Polymarket metadata (slug, volume, liquidity).
 * Spread history charts are shown per-market when candle data is available.
 */
/** Filter candle array to those with at least 2 non-null close prices */
const hasValidCandles = (candles) => {
  if (!Array.isArray(candles)) return false;
  const valid = candles.filter(c => (c.close ?? c.open ?? c.high ?? c.low) != null);
  return valid.length >= 2;
};

const SpreadHistorySection = memo(({ entry }) => {
  const marketsWithCandles = useMemo(() => {
    if (!entry?.market_candles?.length) return [];
    return entry.market_candles.filter(
      mc => hasValidCandles(mc.kalshi) || hasValidCandles(mc.poly)
    );
  }, [entry?.market_candles]);

  if (marketsWithCandles.length === 0) {
    return (
      <div className="bg-gray-800/20 rounded-lg p-3 border border-gray-800/30">
        <div className="text-[10px] font-semibold text-gray-500 uppercase tracking-wider mb-2">
          Spread History
        </div>
        <p className="text-[11px] text-gray-600 text-center py-2">
          No price history &mdash; markets may be illiquid
        </p>
      </div>
    );
  }

  return (
    <div className="bg-gray-800/20 rounded-lg p-3 border border-gray-800/30">
      <div className="text-[10px] font-semibold text-gray-500 uppercase tracking-wider mb-2">
        Spread History ({marketsWithCandles.length})
      </div>
      <div className="space-y-3">
        {marketsWithCandles.map(mc => (
          <div key={mc.kalshi_ticker}>
            <div className="text-[10px] text-gray-400 truncate mb-1">
              {mc.question || mc.kalshi_ticker}
            </div>
            <SpreadRibbon
              kalshiCandles={mc.kalshi || []}
              polyCandles={mc.poly || []}
              height={60}
            />
          </div>
        ))}
      </div>
    </div>
  );
});

const EventDetailsPanel = ({ selectedEventTicker, eventCodex, pairIndex }) => {
  const entry = useMemo(() => {
    if (!selectedEventTicker || !eventCodex?.entries) return null;
    return eventCodex.entries[selectedEventTicker] || null;
  }, [selectedEventTicker, eventCodex]);

  // Get event info from pair index for additional context
  const eventInfo = useMemo(() => {
    if (!selectedEventTicker || !pairIndex?.events) return null;
    return pairIndex.events.find(e => e.event_ticker === selectedEventTicker);
  }, [selectedEventTicker, pairIndex]);

  if (!selectedEventTicker) {
    return (
      <div className="
        bg-gradient-to-br from-gray-900/90 via-gray-950/80 to-black/90
        rounded-2xl border border-cyan-500/10 shadow-lg shadow-cyan-500/5
        h-full flex flex-col items-center justify-center min-h-[420px]
      ">
        <BookOpen className="w-8 h-8 text-gray-700 mb-3" />
        <p className="text-sm text-gray-500">Select an event from the index</p>
        <p className="text-[10px] text-gray-600 mt-1">Click any event row to view details</p>
      </div>
    );
  }

  if (!entry) {
    return (
      <div className="
        bg-gradient-to-br from-gray-900/90 via-gray-950/80 to-black/90
        rounded-2xl border border-cyan-500/10 shadow-lg shadow-cyan-500/5
        h-full flex flex-col items-center justify-center min-h-[420px]
      ">
        <div className="animate-pulse space-y-3 w-3/4">
          <div className="h-3 bg-gray-800 rounded w-1/3" />
          <div className="h-5 bg-gray-800 rounded w-2/3" />
          <div className="h-20 bg-gray-800/50 rounded" />
          <div className="h-20 bg-gray-800/50 rounded" />
          <div className="h-32 bg-gray-800/30 rounded" />
        </div>
        <p className="text-[10px] text-gray-600 mt-4">
          Waiting for codex sync...
        </p>
      </div>
    );
  }

  const {
    kalshi_event_ticker,
    series_ticker,
    title,
    category,
    kalshi_subtitle,
    kalshi_mutually_exclusive,
    kalshi_strike_date,
    kalshi_markets = [],
    kalshi_product_metadata,
    poly_event_id,
    poly_title,
    poly_slug,
    poly_volume,
    poly_volume_24h,
    poly_liquidity,
    poly_live_volume,
  } = entry;

  // Aggregate Kalshi volume
  const kalshiVolume = kalshi_markets.reduce(
    (sum, m) => sum + (m.volume || 0) + (m.volume_24h || 0), 0
  );

  const formatVol = (v) => {
    if (v == null || v <= 0) return '--';
    if (v >= 1000000) return `$${(v / 1000000).toFixed(1)}M`;
    if (v >= 1000) return `$${(v / 1000).toFixed(0)}k`;
    return `$${Math.round(v)}`;
  };

  return (
    <div className="
      bg-gradient-to-br from-gray-900/90 via-gray-950/80 to-black/90
      rounded-2xl border border-cyan-500/10 shadow-lg shadow-cyan-500/5
      overflow-hidden h-full flex flex-col
    ">
      {/* Header */}
      <div className="px-5 py-3 border-b border-gray-800/50 flex-shrink-0">
        <div className="flex items-center gap-2 mb-1">
          <span className="text-[10px] font-mono text-gray-500">{kalshi_event_ticker}</span>
          {category && (
            <span className="text-[10px] font-mono bg-gray-800/60 text-gray-400 rounded-full px-2 py-0.5">
              {category}
            </span>
          )}
        </div>
        <h3 className="text-sm font-semibold text-gray-200 leading-tight">
          {title || kalshi_event_ticker}
        </h3>
        {kalshi_subtitle && (
          <p className="text-xs text-gray-400 mt-0.5 leading-tight">{kalshi_subtitle}</p>
        )}
      </div>

      {/* Scrollable content */}
      <div className="flex-1 overflow-y-auto px-5 py-3 space-y-3">
        {/* Kalshi section */}
        <div className="bg-gray-800/30 rounded-lg p-3 border border-gray-700/30">
          <div className="text-[10px] font-semibold text-cyan-400/80 uppercase tracking-wider mb-2">
            Kalshi
          </div>
          <div className="space-y-1.5 text-xs text-gray-300">
            <div className="flex justify-between">
              <span className="text-gray-500">Event Ticker</span>
              <span className="font-mono text-cyan-400/70">{kalshi_event_ticker}</span>
            </div>
            {series_ticker && (
              <div className="flex justify-between">
                <span className="text-gray-500">Series</span>
                <span className="font-mono">{series_ticker}</span>
              </div>
            )}
            <div className="flex justify-between">
              <span className="text-gray-500">Markets</span>
              <span className="font-mono">
                {kalshi_markets.length}
                {kalshi_mutually_exclusive && (
                  <span className="text-gray-600 ml-1">(mut. excl.)</span>
                )}
              </span>
            </div>
            {kalshi_strike_date && (
              <div className="flex justify-between">
                <span className="text-gray-500">Strike</span>
                <span className="font-mono">{kalshi_strike_date}</span>
              </div>
            )}
            <div className="flex justify-between">
              <span className="text-gray-500">Volume</span>
              <span className="font-mono text-cyan-400/80">{formatVol(kalshiVolume)}</span>
            </div>
            {kalshi_product_metadata && Object.keys(kalshi_product_metadata).length > 0 && (
              <div className="pt-1 border-t border-gray-700/20">
                <span className="text-[10px] text-gray-600 block mb-1">product_metadata</span>
                {Object.entries(kalshi_product_metadata).map(([k, v]) => (
                  <div key={k} className="flex justify-between text-[11px]">
                    <span className="text-gray-500 truncate mr-2">{k}</span>
                    <span className="font-mono text-gray-400 truncate max-w-[60%] text-right">
                      {typeof v === 'object' ? JSON.stringify(v) : String(v)}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Polymarket section */}
        <div className="bg-gray-800/30 rounded-lg p-3 border border-gray-700/30">
          <div className="flex items-center justify-between mb-2">
            <span className="text-[10px] font-semibold text-violet-400/80 uppercase tracking-wider">
              Polymarket
            </span>
            {poly_event_id ? (
              <span className="flex items-center gap-1 text-[10px] text-emerald-500/70">
                <Check className="w-3 h-3" /> matched
              </span>
            ) : (
              <span className="flex items-center gap-1 text-[10px] text-gray-600">
                <X className="w-3 h-3" /> no match
              </span>
            )}
          </div>
          {poly_event_id ? (
            <div className="space-y-1.5 text-xs text-gray-300">
              {poly_title && (
                <div className="text-gray-400 truncate">{poly_title}</div>
              )}
              {poly_slug && (
                <div className="flex justify-between">
                  <span className="text-gray-500">Slug</span>
                  <span className="font-mono text-violet-400/70 truncate max-w-[60%] text-right">
                    {poly_slug}
                  </span>
                </div>
              )}
              <div className="flex justify-between">
                <span className="text-gray-500">Live vol</span>
                <span className="font-mono text-violet-400/80">{formatVol(poly_live_volume)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">Total vol</span>
                <span className="font-mono">{formatVol(poly_volume)}</span>
              </div>
              {poly_liquidity != null && poly_liquidity > 0 && (
                <div className="flex justify-between">
                  <span className="text-gray-500">Liquidity</span>
                  <span className="font-mono">{formatVol(poly_liquidity)}</span>
                </div>
              )}
            </div>
          ) : (
            <div className="text-xs text-gray-600">No Polymarket event matched</div>
          )}
        </div>

        {/* Market list */}
        {kalshi_markets.length > 0 && (
          <div className="bg-gray-800/20 rounded-lg p-3 border border-gray-800/30">
            <div className="text-[10px] font-semibold text-gray-500 uppercase tracking-wider mb-2">
              Markets ({kalshi_markets.length})
            </div>
            <div className="space-y-1">
              {kalshi_markets.map((m) => (
                <div key={m.ticker} className="flex items-center justify-between text-[11px]">
                  <span className="text-gray-400 truncate flex-1 min-w-0 mr-2">
                    {m.yes_sub_title || m.ticker}
                  </span>
                  <div className="flex items-center gap-2 font-mono flex-shrink-0">
                    {m.last_price != null && (
                      <span className="text-gray-500">{m.last_price}c</span>
                    )}
                    <span className="text-gray-600">{formatVol((m.volume || 0) + (m.volume_24h || 0))}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Spread History */}
        <SpreadHistorySection entry={entry} />
      </div>
    </div>
  );
};

export default memo(EventDetailsPanel);
