import React, { memo, useMemo } from 'react';

/**
 * Sparkline - Tiny inline SVG sparkline for a single market's price history.
 * No axes, no labels - just the trend shape. Fits inside a table cell.
 */
const Sparkline = memo(({ points, width = 64, height = 20, color = '#22d3ee' }) => {
  const path = useMemo(() => {
    if (!points || points.length < 2) return null;

    const values = points.map(p => p.c);
    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min || 1;

    const coords = values.map((v, i) => {
      const x = (i / (values.length - 1)) * width;
      const y = height - ((v - min) / range) * (height - 2) - 1; // 1px padding top/bottom
      return `${x},${y}`;
    });

    return `M${coords.join(' L')}`;
  }, [points, width, height]);

  if (!path) return <span className="text-gray-700 text-[9px]">--</span>;

  // Determine trend color from first vs last value
  const first = points[0]?.c;
  const last = points[points.length - 1]?.c;
  const trendColor = last > first + 2 ? '#34d399' : last < first - 2 ? '#f87171' : '#6b7280';

  return (
    <svg width={width} height={height} className="inline-block">
      <path
        d={path}
        fill="none"
        stroke={trendColor}
        strokeWidth={1.2}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
});

Sparkline.displayName = 'Sparkline';

export default Sparkline;
