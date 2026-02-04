import React, { memo, useMemo } from 'react';

/**
 * SpreadRibbon - Inline SVG spread visualization for Kalshi vs Polymarket prices.
 *
 * Renders:
 * 1. Two overlaid price lines (Kalshi = cyan, Poly = violet)
 * 2. Filled area between lines colored by spread direction
 * 3. Current price labels at right edge
 */
const SpreadRibbon = ({ kalshiCandles = [], polyCandles = [], width = 320, height = 52 }) => {
  const { kalshiPath, polyPath, segments, yDomain, kalshiLast, polyLast, timeLabels } = useMemo(() => {
    if (!kalshiCandles.length && !polyCandles.length) {
      return { kalshiPath: '', polyPath: '', segments: [], yDomain: [0, 100], kalshiLast: null, polyLast: null, timeLabels: null };
    }

    // Extract close prices, align by timestamp
    const kMap = new Map();
    for (const c of kalshiCandles) {
      const price = c.close ?? c.open ?? c.high ?? c.low;
      if (price != null && c.ts) kMap.set(c.ts, price);
    }

    const pMap = new Map();
    for (const c of polyCandles) {
      const price = c.close ?? c.open ?? c.high ?? c.low;
      if (price != null && c.ts) pMap.set(c.ts, price);
    }

    // Union of timestamps, sorted
    const allTs = [...new Set([...kMap.keys(), ...pMap.keys()])].sort((a, b) => a - b);
    if (allTs.length < 2) {
      return { kalshiPath: '', polyPath: '', segments: [], yDomain: [0, 100], kalshiLast: null, polyLast: null, timeLabels: null };
    }

    // Interpolate missing values (carry forward)
    const points = [];
    let lastK = null;
    let lastP = null;
    for (const ts of allTs) {
      const k = kMap.get(ts) ?? lastK;
      const p = pMap.get(ts) ?? lastP;
      if (k != null) lastK = k;
      if (p != null) lastP = p;
      if (k != null && p != null) {
        points.push({ ts, k, p });
      }
    }

    if (points.length < 2) {
      return { kalshiPath: '', polyPath: '', segments: [], yDomain: [0, 100], kalshiLast: null, polyLast: null, timeLabels: null };
    }

    // Compute domain
    const allPrices = points.flatMap(pt => [pt.k, pt.p]);
    const minP = Math.min(...allPrices);
    const maxP = Math.max(...allPrices);
    const padding = Math.max((maxP - minP) * 0.15, 2);
    const yMin = minP - padding;
    const yMax = maxP + padding;
    const yRange = yMax - yMin || 1;

    // Map to SVG coords
    const pad = { left: 2, right: 2, top: 4, bottom: 4 };
    const plotW = width - pad.left - pad.right;
    const plotH = height - pad.top - pad.bottom;
    const tsMin = points[0].ts;
    const tsMax = points[points.length - 1].ts;
    const tsRange = tsMax - tsMin || 1;

    const toX = (ts) => pad.left + ((ts - tsMin) / tsRange) * plotW;
    const toY = (price) => pad.top + (1 - (price - yMin) / yRange) * plotH;

    // Build paths
    const kPts = points.map(pt => `${toX(pt.ts).toFixed(1)},${toY(pt.k).toFixed(1)}`);
    const pPts = points.map(pt => `${toX(pt.ts).toFixed(1)},${toY(pt.p).toFixed(1)}`);

    const kalshiPathStr = `M ${kPts.join(' L ')}`;
    const polyPathStr = `M ${pPts.join(' L ')}`;

    // Build area segments (colored per spread direction)
    const segs = [];
    for (let i = 0; i < points.length - 1; i++) {
      const p1 = points[i];
      const p2 = points[i + 1];
      const x1 = toX(p1.ts);
      const x2 = toX(p2.ts);
      const ky1 = toY(p1.k);
      const ky2 = toY(p2.k);
      const py1 = toY(p1.p);
      const py2 = toY(p2.p);
      const kalshiAbove = (p1.k + p2.k) / 2 >= (p1.p + p2.p) / 2;
      const spread = Math.abs((p1.k - p1.p + p2.k - p2.p) / 2);
      const opacity = Math.min(0.4, spread / 20 * 0.4);

      segs.push({
        key: i,
        d: `M ${x1.toFixed(1)},${ky1.toFixed(1)} L ${x2.toFixed(1)},${ky2.toFixed(1)} L ${x2.toFixed(1)},${py2.toFixed(1)} L ${x1.toFixed(1)},${py1.toFixed(1)} Z`,
        fill: kalshiAbove ? '#22d3ee' : '#a78bfa',
        opacity,
      });
    }

    const lastPt = points[points.length - 1];

    // Time labels
    const formatTime = (ts) => {
      const d = new Date(ts * 1000);
      return `${d.getHours().toString().padStart(2, '0')}:${d.getMinutes().toString().padStart(2, '0')}`;
    };

    return {
      kalshiPath: kalshiPathStr,
      polyPath: polyPathStr,
      segments: segs,
      yDomain: [yMin, yMax],
      kalshiLast: lastPt?.k,
      polyLast: lastPt?.p,
      timeLabels: { start: formatTime(tsMin), end: formatTime(tsMax) },
    };
  }, [kalshiCandles, polyCandles, width, height]);

  if (!kalshiPath && !polyPath) {
    return (
      <div className="flex items-center justify-center text-[10px] text-gray-600 font-mono" style={{ height }}>
        No candle data
      </div>
    );
  }

  const avgSpread = kalshiLast != null && polyLast != null
    ? (kalshiLast - polyLast).toFixed(1)
    : null;

  return (
    <div className="relative">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="w-full"
        preserveAspectRatio="none"
        style={{ height }}
      >
        {/* Spread area fill */}
        {segments.map(seg => (
          <path
            key={seg.key}
            d={seg.d}
            fill={seg.fill}
            opacity={seg.opacity}
          />
        ))}
        {/* Kalshi line */}
        {kalshiPath && (
          <path d={kalshiPath} stroke="#22d3ee" strokeWidth={1.5} fill="none" />
        )}
        {/* Poly line */}
        {polyPath && (
          <path d={polyPath} stroke="#a78bfa" strokeWidth={1.5} fill="none" />
        )}
      </svg>
      {/* Labels */}
      <div className="flex items-center justify-between mt-0.5">
        {timeLabels && (
          <span className="text-[9px] text-gray-600 font-mono">{timeLabels.start}</span>
        )}
        <div className="flex items-center gap-2 text-[9px] font-mono">
          {avgSpread != null && (
            <span className={Number(avgSpread) >= 0 ? 'text-cyan-400/70' : 'text-violet-400/70'}>
              {Number(avgSpread) >= 0 ? '+' : ''}{avgSpread}c spread
            </span>
          )}
        </div>
        {timeLabels && (
          <span className="text-[9px] text-gray-600 font-mono">{timeLabels.end}</span>
        )}
      </div>
    </div>
  );
};

export default memo(SpreadRibbon);
