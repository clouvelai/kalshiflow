import React, { memo, useMemo, useRef, useEffect } from 'react';

/**
 * HeatStripMatrix - A heatmap where each row is a market and each column
 * is an hourly time bucket. Color intensity = price level (0-100c).
 * Markets sorted by current price descending. Rendered on canvas for performance.
 */

const ROW_HEIGHT = 7;
const ROW_GAP = 1;
const LABEL_WIDTH = 100;
const TIME_AXIS_HEIGHT = 16;

// Interpolate between two colors based on t (0-1)
// Per-row normalized: 0 = row's min price, 1 = row's max price
function lerpColor(t) {
  t = Math.max(0, Math.min(1, t));

  // Dark charcoal → teal → bright cyan
  // Spread across full range for maximum contrast
  const r = Math.round(15 + t * 20);
  const g = Math.round(25 + t * 190);
  const b = Math.round(40 + t * 198);
  return `rgb(${r},${g},${b})`;
}

function shortName(title, ticker) {
  if (title && title.length <= 16) return title;
  if (title) return title.slice(0, 14) + '..';
  const parts = ticker.split('-');
  return parts[parts.length - 1] || ticker;
}

const HeatStripMatrix = memo(({ candlestickSeries, markets }) => {
  const canvasRef = useRef(null);

  const { rows, allTimestamps, dateLabels } = useMemo(() => {
    if (!candlestickSeries || typeof candlestickSeries !== 'object') {
      return { rows: [], allTimestamps: [], dateLabels: [] };
    }

    const tickers = Object.keys(candlestickSeries);
    if (tickers.length === 0) {
      return { rows: [], allTimestamps: [], dateLabels: [] };
    }

    // Collect all timestamps across all markets
    const tsSet = new Set();
    for (const ticker of tickers) {
      for (const p of (candlestickSeries[ticker] || [])) {
        tsSet.add(p.ts);
      }
    }
    const allTimestamps = [...tsSet].sort((a, b) => a - b);

    if (allTimestamps.length === 0) {
      return { rows: [], allTimestamps: [], dateLabels: [] };
    }

    // Build ts→index map for fast lookup
    const tsIndex = {};
    allTimestamps.forEach((ts, i) => { tsIndex[ts] = i; });

    // Build rows: one per market with price array aligned to allTimestamps
    const rows = tickers.map(ticker => {
      const points = candlestickSeries[ticker] || [];
      const prices = new Array(allTimestamps.length).fill(null);
      for (const p of points) {
        const idx = tsIndex[p.ts];
        if (idx !== undefined) prices[idx] = p.c;
      }

      // Forward-fill nulls for cleaner visualization
      for (let i = 1; i < prices.length; i++) {
        if (prices[i] === null) prices[i] = prices[i - 1];
      }

      const mkt = markets?.[ticker];
      const currentPrice = prices[prices.length - 1] ?? 0;
      const title = mkt?.title || shortName(null, ticker);

      // Per-row min/max for normalized coloring
      const validPrices = prices.filter(p => p !== null);
      const rowMin = validPrices.length ? Math.min(...validPrices) : 0;
      const rowMax = validPrices.length ? Math.max(...validPrices) : 100;

      return { ticker, title: shortName(title, ticker), prices, currentPrice, rowMin, rowMax };
    });

    // Sort by current price descending
    rows.sort((a, b) => b.currentPrice - a.currentPrice);

    // Generate date labels for the time axis
    const dateLabels = [];
    let lastDate = '';
    for (let i = 0; i < allTimestamps.length; i++) {
      const d = new Date(allTimestamps[i] * 1000);
      const dateStr = `${d.getMonth() + 1}/${d.getDate()}`;
      if (dateStr !== lastDate) {
        dateLabels.push({ index: i, label: dateStr });
        lastDate = dateStr;
      }
    }

    return { rows, allTimestamps, dateLabels };
  }, [candlestickSeries, markets]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || rows.length === 0 || allTimestamps.length === 0) return;

    const dpr = window.devicePixelRatio || 1;
    const containerWidth = canvas.parentElement?.clientWidth || 600;
    const stripWidth = containerWidth - LABEL_WIDTH;
    const totalHeight = rows.length * (ROW_HEIGHT + ROW_GAP) + TIME_AXIS_HEIGHT;

    canvas.width = containerWidth * dpr;
    canvas.height = totalHeight * dpr;
    canvas.style.width = `${containerWidth}px`;
    canvas.style.height = `${totalHeight}px`;

    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, containerWidth, totalHeight);

    const colWidth = stripWidth / allTimestamps.length;

    // Draw rows
    rows.forEach((row, rowIdx) => {
      const y = rowIdx * (ROW_HEIGHT + ROW_GAP);

      // Label
      ctx.font = '9px ui-monospace, monospace';
      ctx.fillStyle = '#9ca3af'; // gray-400
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      ctx.fillText(row.title, LABEL_WIDTH - 8, y + ROW_HEIGHT / 2 + 0.5);

      // Heat cells - per-row normalized for visible contrast
      const range = row.rowMax - row.rowMin || 1;
      for (let col = 0; col < row.prices.length; col++) {
        const price = row.prices[col];
        if (price === null) continue;

        const x = LABEL_WIDTH + col * colWidth;
        const t = (price - row.rowMin) / range;
        ctx.fillStyle = lerpColor(t);
        ctx.fillRect(x, y, Math.ceil(colWidth) + 0.5, ROW_HEIGHT);
      }
    });

    // Time axis
    const axisY = rows.length * (ROW_HEIGHT + ROW_GAP) + 2;
    ctx.font = '8px ui-monospace, monospace';
    ctx.fillStyle = '#4b5563'; // gray-600
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';

    dateLabels.forEach(({ index, label }) => {
      const x = LABEL_WIDTH + index * colWidth;
      // Only draw if there's enough space (skip if too close to previous)
      ctx.fillText(label, x, axisY);
    });

    // Subtle grid lines at date boundaries
    ctx.strokeStyle = 'rgba(75, 85, 99, 0.15)';
    ctx.lineWidth = 0.5;
    dateLabels.forEach(({ index }) => {
      const x = LABEL_WIDTH + index * colWidth;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, axisY - 2);
      ctx.stroke();
    });

  }, [rows, allTimestamps, dateLabels]);

  if (rows.length === 0) return null;

  const totalHeight = rows.length * (ROW_HEIGHT + ROW_GAP) + TIME_AXIS_HEIGHT;

  return (
    <div className="bg-gray-800/20 rounded-lg p-3 border border-gray-700/15">
      <div className="flex items-center justify-between mb-2">
        <div className="text-[9px] font-semibold text-cyan-400/70 uppercase tracking-wider">
          Price History (7d hourly)
        </div>
        <div className="flex items-center gap-2 text-[8px] text-gray-600">
          <span className="flex items-center gap-1">
            <span className="inline-block w-3 h-2 rounded-sm" style={{ background: lerpColor(0) }} />
            Low
          </span>
          <span className="inline-block w-8 h-2 rounded-sm" style={{ background: 'linear-gradient(to right, ' + lerpColor(0) + ', ' + lerpColor(0.5) + ', ' + lerpColor(1) + ')' }} />
          <span className="flex items-center gap-1">
            <span className="inline-block w-3 h-2 rounded-sm" style={{ background: lerpColor(1) }} />
            High
          </span>
          <span className="text-gray-700 ml-1">(per market)</span>
        </div>
      </div>
      <canvas
        ref={canvasRef}
        style={{ height: `${totalHeight}px` }}
        className="w-full"
      />
    </div>
  );
});

HeatStripMatrix.displayName = 'HeatStripMatrix';

export default HeatStripMatrix;
