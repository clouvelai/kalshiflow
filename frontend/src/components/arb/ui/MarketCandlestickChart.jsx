import React, { memo, useMemo, useRef, useEffect } from 'react';

/**
 * HeatStripMatrix - A heatmap where each row is a market and each column
 * is an hourly time bucket. Color = absolute price on 0-100c scale.
 * Smooth multi-stop gradient with good differentiation at all price levels.
 * Markets sorted by current price descending. Canvas-rendered for performance.
 */

const ROW_HEIGHT = 8;
const ROW_GAP = 1;
const LABEL_WIDTH = 100;
const PRICE_WIDTH = 30;
const TIME_AXIS_HEIGHT = 16;

// Multi-stop color ramp: 0c → 100c
// Designed for smooth perceptual progression on dark backgrounds
const COLOR_STOPS = [
  { at: 0,    r: 12,  g: 16,  b: 28  },  // near-black
  { at: 0.05, r: 20,  g: 24,  b: 50  },  // very dark indigo
  { at: 0.15, r: 30,  g: 40,  b: 80  },  // dark blue
  { at: 0.30, r: 25,  g: 75,  b: 110 },  // steel blue
  { at: 0.50, r: 18,  g: 130, b: 148 },  // teal
  { at: 0.70, r: 22,  g: 188, b: 200 },  // cyan
  { at: 0.85, r: 80,  g: 220, b: 230 },  // light cyan
  { at: 1.0,  r: 180, g: 248, b: 252 },  // near-white cyan
];

function priceToColor(cents) {
  const t = Math.max(0, Math.min(1, cents / 100));

  // Find the two stops we're between
  let lo = COLOR_STOPS[0], hi = COLOR_STOPS[COLOR_STOPS.length - 1];
  for (let i = 0; i < COLOR_STOPS.length - 1; i++) {
    if (t >= COLOR_STOPS[i].at && t <= COLOR_STOPS[i + 1].at) {
      lo = COLOR_STOPS[i];
      hi = COLOR_STOPS[i + 1];
      break;
    }
  }

  const range = hi.at - lo.at || 1;
  const s = (t - lo.at) / range;
  const r = Math.round(lo.r + s * (hi.r - lo.r));
  const g = Math.round(lo.g + s * (hi.g - lo.g));
  const b = Math.round(lo.b + s * (hi.b - lo.b));
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

    // Collect all timestamps
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

    const tsIndex = {};
    allTimestamps.forEach((ts, i) => { tsIndex[ts] = i; });

    const rows = tickers.map(ticker => {
      const points = candlestickSeries[ticker] || [];
      const prices = new Array(allTimestamps.length).fill(null);
      for (const p of points) {
        const idx = tsIndex[p.ts];
        if (idx !== undefined) prices[idx] = p.c;
      }

      // Forward-fill nulls
      for (let i = 1; i < prices.length; i++) {
        if (prices[i] === null) prices[i] = prices[i - 1];
      }

      const mkt = markets?.[ticker];
      const currentPrice = prices[prices.length - 1] ?? 0;
      const title = mkt?.title || shortName(null, ticker);

      return { ticker, title: shortName(title, ticker), prices, currentPrice };
    });

    rows.sort((a, b) => b.currentPrice - a.currentPrice);

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
    const stripWidth = containerWidth - LABEL_WIDTH - PRICE_WIDTH;
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

      // Market name label (left)
      ctx.font = '9px ui-monospace, monospace';
      ctx.fillStyle = '#9ca3af';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      ctx.fillText(row.title, LABEL_WIDTH - 6, y + ROW_HEIGHT / 2 + 0.5);

      // Heat cells - absolute 0-100c scale
      for (let col = 0; col < row.prices.length; col++) {
        const price = row.prices[col];
        if (price === null) continue;

        const x = LABEL_WIDTH + col * colWidth;
        ctx.fillStyle = priceToColor(price);
        ctx.fillRect(x, y, Math.ceil(colWidth) + 0.5, ROW_HEIGHT);
      }

      // Current price label (right)
      const priceX = LABEL_WIDTH + stripWidth + 4;
      ctx.font = '8px ui-monospace, monospace';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.fillStyle = row.currentPrice >= 50 ? '#67e8f9' : row.currentPrice >= 10 ? '#5eead4' : '#6b7280';
      ctx.fillText(`${row.currentPrice}c`, priceX, y + ROW_HEIGHT / 2 + 0.5);
    });

    // Time axis
    const axisY = rows.length * (ROW_HEIGHT + ROW_GAP) + 2;
    ctx.font = '8px ui-monospace, monospace';
    ctx.fillStyle = '#4b5563';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';

    dateLabels.forEach(({ index, label }) => {
      const x = LABEL_WIDTH + index * colWidth;
      ctx.fillText(label, x, axisY);
    });

    // Date grid lines
    ctx.strokeStyle = 'rgba(75, 85, 99, 0.12)';
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
        <div className="flex items-center gap-1.5 text-[8px] text-gray-500">
          <span>0c</span>
          <span className="inline-block w-20 h-2 rounded-sm" style={{
            background: `linear-gradient(to right, ${priceToColor(0)}, ${priceToColor(5)}, ${priceToColor(15)}, ${priceToColor(30)}, ${priceToColor(50)}, ${priceToColor(75)}, ${priceToColor(100)})`
          }} />
          <span>100c</span>
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
