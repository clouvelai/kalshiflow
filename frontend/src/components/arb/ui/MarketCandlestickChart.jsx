import React, { memo, useMemo, useState } from 'react';
import {
  ComposedChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';

const COLORS = [
  '#22d3ee', // cyan-400
  '#34d399', // emerald-400
  '#a78bfa', // violet-400
  '#fb923c', // orange-400
  '#f472b6', // pink-400
  '#facc15', // yellow-400
  '#60a5fa', // blue-400
  '#4ade80', // green-400
];

const shortLabel = (ticker) => {
  // Strip common prefixes to get short outcome name
  const parts = ticker.split('-');
  return parts[parts.length - 1]?.toUpperCase() || ticker;
};

const formatTs = (ts) => {
  const d = new Date(ts * 1000);
  return `${d.getMonth() + 1}/${d.getDate()} ${d.getHours()}:${String(d.getMinutes()).padStart(2, '0')}`;
};

const formatTsShort = (ts) => {
  const d = new Date(ts * 1000);
  return `${d.getMonth() + 1}/${d.getDate()}`;
};

const CustomTooltip = ({ active, payload, label, marketNames }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-gray-900/95 border border-gray-700/40 rounded-lg px-3 py-2 text-[10px] shadow-xl">
      <div className="text-gray-500 mb-1">{formatTs(label)}</div>
      {payload.map((entry, i) => (
        <div key={i} className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ backgroundColor: entry.color }} />
          <span className="text-gray-400 truncate max-w-[140px]">{marketNames[entry.dataKey] || entry.dataKey}</span>
          <span className="font-mono text-gray-200 ml-auto">{entry.value}c</span>
        </div>
      ))}
    </div>
  );
};

const MarketCandlestickChart = memo(({ candlestickSeries, markets }) => {
  const [selectedMarket, setSelectedMarket] = useState(null); // null = show all

  const { tickers, chartData, marketNames, colorMap } = useMemo(() => {
    if (!candlestickSeries || typeof candlestickSeries !== 'object') {
      return { tickers: [], chartData: [], marketNames: {}, colorMap: {} };
    }

    const tickers = Object.keys(candlestickSeries);
    if (tickers.length === 0) {
      return { tickers: [], chartData: [], marketNames: {}, colorMap: {} };
    }

    // Build color map and name map
    const colorMap = {};
    const marketNames = {};
    tickers.forEach((t, i) => {
      colorMap[t] = COLORS[i % COLORS.length];
      const mkt = markets?.[t];
      marketNames[t] = mkt?.title || shortLabel(t);
    });

    // Merge all series into unified time-series keyed by timestamp
    const byTs = {};
    for (const ticker of tickers) {
      const points = candlestickSeries[ticker] || [];
      for (const p of points) {
        if (!byTs[p.ts]) byTs[p.ts] = { ts: p.ts };
        byTs[p.ts][ticker] = p.c;
      }
    }

    const chartData = Object.values(byTs).sort((a, b) => a.ts - b.ts);
    return { tickers, chartData, marketNames, colorMap };
  }, [candlestickSeries, markets]);

  if (tickers.length === 0 || chartData.length === 0) return null;

  const visibleTickers = selectedMarket ? [selectedMarket] : tickers;

  return (
    <div className="bg-gray-800/20 rounded-lg p-3 border border-gray-700/15">
      <div className="flex items-center justify-between mb-2">
        <div className="text-[9px] font-semibold text-cyan-400/70 uppercase tracking-wider">
          Price History (7d hourly)
        </div>
      </div>

      {/* Market pills */}
      {tickers.length > 1 && (
        <div className="flex flex-wrap gap-1 mb-2">
          <button
            onClick={() => setSelectedMarket(null)}
            className={`text-[9px] font-medium px-2 py-0.5 rounded-full transition-colors ${
              selectedMarket === null
                ? 'bg-cyan-500/15 text-cyan-400 border border-cyan-500/25'
                : 'text-gray-500 hover:text-gray-300 bg-gray-800/30 border border-gray-700/20'
            }`}
          >
            All
          </button>
          {tickers.map((t) => (
            <button
              key={t}
              onClick={() => setSelectedMarket(selectedMarket === t ? null : t)}
              className={`text-[9px] font-medium px-2 py-0.5 rounded-full transition-colors flex items-center gap-1 ${
                selectedMarket === t
                  ? 'bg-gray-700/30 border border-gray-600/30'
                  : 'text-gray-500 hover:text-gray-300 bg-gray-800/30 border border-gray-700/20'
              }`}
            >
              <span className="w-1.5 h-1.5 rounded-full flex-shrink-0" style={{ backgroundColor: colorMap[t] }} />
              <span className="truncate max-w-[80px]" style={selectedMarket === t ? { color: colorMap[t] } : undefined}>
                {marketNames[t]}
              </span>
            </button>
          ))}
        </div>
      )}

      <ResponsiveContainer width="100%" height={160}>
        <ComposedChart data={chartData} margin={{ top: 4, right: 4, bottom: 0, left: -20 }}>
          <XAxis
            dataKey="ts"
            tickFormatter={formatTsShort}
            tick={{ fill: '#6b7280', fontSize: 9 }}
            axisLine={{ stroke: '#374151', strokeWidth: 0.5 }}
            tickLine={false}
            minTickGap={40}
          />
          <YAxis
            tick={{ fill: '#6b7280', fontSize: 9 }}
            axisLine={false}
            tickLine={false}
            domain={['auto', 'auto']}
            tickFormatter={(v) => `${v}`}
          />
          <Tooltip
            content={<CustomTooltip marketNames={marketNames} />}
            cursor={{ stroke: '#4b5563', strokeWidth: 0.5, strokeDasharray: '3 3' }}
          />
          {visibleTickers.map((t) => (
            <Area
              key={t}
              type="monotone"
              dataKey={t}
              stroke={colorMap[t]}
              strokeWidth={1.5}
              fill={colorMap[t]}
              fillOpacity={0.08}
              dot={false}
              connectNulls
              isAnimationActive={false}
            />
          ))}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
});

MarketCandlestickChart.displayName = 'MarketCandlestickChart';

export default MarketCandlestickChart;
