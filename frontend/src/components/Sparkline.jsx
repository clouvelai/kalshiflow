import React from 'react';

const Sparkline = ({ data = [], width = 60, height = 20, className = "" }) => {
  if (!data || data.length === 0) {
    return (
      <div className={`${className} flex items-center justify-center`} style={{ width, height }}>
        <span className="text-gray-400 text-xs">No data</span>
      </div>
    );
  }

  // If only one data point, show a flat line
  if (data.length === 1) {
    const y = height / 2;
    return (
      <svg width={width} height={height} className={className}>
        <line 
          x1={0} 
          y1={y} 
          x2={width} 
          y2={y} 
          stroke="currentColor" 
          strokeWidth="1.5"
          opacity="0.6"
        />
      </svg>
    );
  }

  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1; // Prevent division by zero

  // Generate path points
  const points = data.map((value, index) => {
    const x = (index / (data.length - 1)) * width;
    const y = height - ((value - min) / range) * height;
    return `${x},${y}`;
  });

  const pathData = `M ${points.join(' L ')}`;
  
  // Determine trend color
  const firstValue = data[0];
  const lastValue = data[data.length - 1];
  const trend = lastValue > firstValue ? 'up' : lastValue < firstValue ? 'down' : 'flat';
  
  const strokeColor = trend === 'up' ? '#10b981' : trend === 'down' ? '#ef4444' : '#6b7280';

  return (
    <svg width={width} height={height} className={className}>
      <path
        d={pathData}
        fill="none"
        stroke={strokeColor}
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      {/* Optional: Add a subtle glow effect */}
      <path
        d={pathData}
        fill="none"
        stroke={strokeColor}
        strokeWidth="3"
        opacity="0.1"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
};

export default Sparkline;