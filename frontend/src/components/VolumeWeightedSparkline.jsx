import React, { useMemo, useEffect, useRef, useState } from 'react';

/**
 * VolumeWeightedSparkline - An advanced price visualization component that shows:
 * - Volume-weighted line thickness
 * - Momentum-based color intensity 
 * - Real-time pulse effects
 * - Gradient backgrounds showing trend strength
 * - Subtle volume bars underneath
 */
const VolumeWeightedSparkline = ({ 
  data = [], 
  width = 180, 
  height = 32, 
  className = "",
  showVolumeIndicators = true,
  animationDuration = 300
}) => {
  const svgRef = useRef(null);
  const [animationKey, setAnimationKey] = useState(0);
  const previousDataLength = useRef(0);

  // Trigger animation when new data arrives
  useEffect(() => {
    if (data.length > previousDataLength.current) {
      setAnimationKey(prev => prev + 1);
    }
    previousDataLength.current = data.length;
  }, [data.length]);

  // Process data and calculate enhanced metrics
  const processedData = useMemo(() => {
    if (!data || data.length === 0) {
      return { points: [], volumes: [], maxVolume: 0, priceRange: { min: 0, max: 1 } };
    }

    // Handle both old format (numbers) and new format (objects)
    const normalizedData = data.map((item, index) => {
      if (typeof item === 'number') {
        return {
          price: item,
          volume: 1, // Default volume for backward compatibility
          timestamp: Date.now() - (data.length - index) * 1000,
          side: 'yes'
        };
      }
      return {
        price: item.price || 0,
        volume: item.volume || 1,
        timestamp: item.timestamp || Date.now() - (data.length - index) * 1000,
        side: item.side || 'yes'
      };
    });

    const prices = normalizedData.map(d => d.price);
    const volumes = normalizedData.map(d => d.volume);
    
    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);
    const priceRange = maxPrice - minPrice || 0.01; // Prevent division by zero
    
    const maxVolume = Math.max(...volumes) || 1;
    
    // Calculate momentum for each point
    const points = normalizedData.map((item, index) => {
      const x = (index / Math.max(normalizedData.length - 1, 1)) * width;
      const y = height * 0.8 - ((item.price - minPrice) / priceRange) * height * 0.6; // Leave room for volume bars
      
      // Calculate momentum (price velocity)
      let momentum = 0;
      let momentumIntensity = 0;
      
      if (index > 0 && index < normalizedData.length - 1) {
        const prevPrice = normalizedData[index - 1].price;
        const nextPrice = normalizedData[index + 1] ? normalizedData[index + 1].price : item.price;
        const priceChange = nextPrice - prevPrice;
        momentum = priceChange / priceRange; // Normalize momentum
        momentumIntensity = Math.abs(momentum) * item.volume / maxVolume; // Weight by volume
      }
      
      // Calculate line thickness based on volume
      const baseThickness = 1.5;
      const maxThickness = 4;
      const volumeRatio = item.volume / maxVolume;
      const thickness = baseThickness + (volumeRatio * (maxThickness - baseThickness));
      
      return {
        x,
        y,
        price: item.price,
        volume: item.volume,
        momentum,
        momentumIntensity,
        thickness,
        side: item.side,
        timestamp: item.timestamp,
        volumeRatio
      };
    });
    
    return {
      points,
      volumes,
      maxVolume,
      priceRange: { min: minPrice, max: maxPrice, range: priceRange }
    };
  }, [data, width, height]);

  // Generate color based on momentum and intensity
  const getMomentumColor = (momentum, intensity) => {
    const baseIntensity = Math.min(intensity * 2, 1); // Amplify intensity
    
    if (momentum > 0.001) {
      // Bullish momentum - Green shades
      const alpha = 0.4 + (baseIntensity * 0.6);
      return `rgba(16, 185, 129, ${alpha})`; // Green with varying intensity
    } else if (momentum < -0.001) {
      // Bearish momentum - Red shades
      const alpha = 0.4 + (baseIntensity * 0.6);
      return `rgba(239, 68, 68, ${alpha})`; // Red with varying intensity
    } else {
      // Neutral momentum - Gray
      return `rgba(107, 114, 128, 0.6)`;
    }
  };

  // Generate gradient for trend strength
  const generateTrendGradient = () => {
    if (processedData.points.length < 2) return null;
    
    const firstPoint = processedData.points[0];
    const lastPoint = processedData.points[processedData.points.length - 1];
    const overallTrend = lastPoint.price - firstPoint.price;
    
    const trendStrength = Math.abs(overallTrend) / processedData.priceRange.range;
    const alpha = Math.min(trendStrength * 0.3, 0.2);
    
    if (overallTrend > 0) {
      return `linear-gradient(180deg, rgba(16, 185, 129, ${alpha}) 0%, rgba(16, 185, 129, 0) 100%)`;
    } else if (overallTrend < 0) {
      return `linear-gradient(180deg, rgba(239, 68, 68, ${alpha}) 0%, rgba(239, 68, 68, 0) 100%)`;
    }
    
    return null;
  };

  if (!data || data.length === 0) {
    return (
      <div className={`${className} flex items-center justify-center`} style={{ width, height }}>
        <span className="text-gray-400 text-xs">No data</span>
      </div>
    );
  }

  if (data.length === 1) {
    const y = height / 2;
    return (
      <svg width={width} height={height} className={className} ref={svgRef}>
        <defs>
          <linearGradient id="flatLineGlow" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="rgba(107, 114, 128, 0.3)" />
            <stop offset="100%" stopColor="rgba(107, 114, 128, 0.6)" />
          </linearGradient>
        </defs>
        <line 
          x1={0} 
          y1={y} 
          x2={width} 
          y2={y} 
          stroke="url(#flatLineGlow)" 
          strokeWidth="2"
          strokeLinecap="round"
        />
      </svg>
    );
  }

  const { points } = processedData;
  const mostRecentPoint = points[points.length - 1];
  const trendGradient = generateTrendGradient();

  return (
    <div className={`relative ${className}`} style={{ width, height }}>
      {/* Trend background gradient */}
      {trendGradient && (
        <div 
          className="absolute inset-0 rounded"
          style={{ 
            background: trendGradient,
            opacity: 0.8
          }}
        />
      )}
      
      <svg 
        width={width} 
        height={height} 
        className="relative z-10"
        ref={svgRef}
        key={animationKey}
      >
        <defs>
          {/* Dynamic gradients for momentum colors */}
          <linearGradient id={`bullishGradient-${animationKey}`} x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="rgba(16, 185, 129, 0.3)" />
            <stop offset="50%" stopColor="rgba(16, 185, 129, 0.8)" />
            <stop offset="100%" stopColor="rgba(16, 185, 129, 0.3)" />
          </linearGradient>
          
          <linearGradient id={`bearishGradient-${animationKey}`} x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="rgba(239, 68, 68, 0.3)" />
            <stop offset="50%" stopColor="rgba(239, 68, 68, 0.8)" />
            <stop offset="100%" stopColor="rgba(239, 68, 68, 0.3)" />
          </linearGradient>

          <linearGradient id={`neutralGradient-${animationKey}`} x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="rgba(107, 114, 128, 0.4)" />
            <stop offset="50%" stopColor="rgba(107, 114, 128, 0.7)" />
            <stop offset="100%" stopColor="rgba(107, 114, 128, 0.4)" />
          </linearGradient>

          {/* Pulse effect filter for most recent point */}
          <filter id={`pulseGlow-${animationKey}`} x="-50%" y="-50%" width="200%" height="200%">
            <feMorphology operator="dilate" radius="2" result="dilated"/>
            <feGaussianBlur in="dilated" stdDeviation="3" result="blurred"/>
            <feColorMatrix in="blurred" type="matrix" values="0 0 0 0 0.2 0 0 0 0 0.7 0 0 0 0 1 0 0 0 0.6 0"/>
            <feComposite in="SourceGraphic" in2="blurred" operator="over"/>
          </filter>
        </defs>

        {/* Volume bars underneath (if enabled) */}
        {showVolumeIndicators && points.map((point, index) => (
          <rect
            key={`volume-${index}`}
            x={point.x - 1}
            y={height * 0.85}
            width={2}
            height={point.volumeRatio * height * 0.12}
            fill={getMomentumColor(point.momentum, point.momentumIntensity)}
            opacity={0.4}
            rx={1}
          >
            <animate
              attributeName="height"
              values={`0;${point.volumeRatio * height * 0.12}`}
              dur={`${animationDuration}ms`}
              begin={`${index * 20}ms`}
              fill="freeze"
            />
          </rect>
        ))}

        {/* Main price line with variable thickness */}
        <path
          d={`M ${points.map(point => `${point.x},${point.y}`).join(' L ')}`}
          fill="none"
          stroke={getMomentumColor(mostRecentPoint.momentum, mostRecentPoint.momentumIntensity)}
          strokeWidth={2}
          strokeLinecap="round"
          strokeLinejoin="round"
          opacity={0.9}
        >
          <animate
            attributeName="stroke-dasharray"
            values={`0 ${width * 2};${width * 2} 0`}
            dur={`${animationDuration * 2}ms`}
            fill="freeze"
          />
        </path>

        {/* Enhanced line segments with variable thickness */}
        {points.slice(0, -1).map((point, index) => {
          const nextPoint = points[index + 1];
          const segmentLength = Math.sqrt(
            Math.pow(nextPoint.x - point.x, 2) + Math.pow(nextPoint.y - point.y, 2)
          );
          
          const avgMomentum = (point.momentum + nextPoint.momentum) / 2;
          const avgIntensity = (point.momentumIntensity + nextPoint.momentumIntensity) / 2;
          const avgThickness = (point.thickness + nextPoint.thickness) / 2;
          
          return (
            <line
              key={`segment-${index}`}
              x1={point.x}
              y1={point.y}
              x2={nextPoint.x}
              y2={nextPoint.y}
              stroke={getMomentumColor(avgMomentum, avgIntensity)}
              strokeWidth={avgThickness}
              strokeLinecap="round"
              opacity={0.8}
            >
              <animate
                attributeName="stroke-width"
                values={`1;${avgThickness}`}
                dur={`${animationDuration}ms`}
                begin={`${index * 30}ms`}
                fill="freeze"
              />
            </line>
          );
        })}

        {/* Glow effect for high-momentum segments */}
        {points.slice(0, -1).map((point, index) => {
          const nextPoint = points[index + 1];
          const avgIntensity = (point.momentumIntensity + nextPoint.momentumIntensity) / 2;
          
          if (avgIntensity > 0.3) {
            return (
              <line
                key={`glow-${index}`}
                x1={point.x}
                y1={point.y}
                x2={nextPoint.x}
                y2={nextPoint.y}
                stroke={getMomentumColor(point.momentum, point.momentumIntensity)}
                strokeWidth={point.thickness * 2}
                strokeLinecap="round"
                opacity={0.2}
              >
                <animate
                  attributeName="opacity"
                  values="0.1;0.3;0.1"
                  dur="1500ms"
                  repeatCount="indefinite"
                />
              </line>
            );
          }
          return null;
        })}

        {/* Pulse effect on most recent price point */}
        <circle
          cx={mostRecentPoint.x}
          cy={mostRecentPoint.y}
          r={3}
          fill={getMomentumColor(mostRecentPoint.momentum, mostRecentPoint.momentumIntensity)}
          filter={`url(#pulseGlow-${animationKey})`}
        >
          <animate
            attributeName="r"
            values="2;4;2"
            dur="2000ms"
            repeatCount="indefinite"
          />
          <animate
            attributeName="opacity"
            values="0.7;1;0.7"
            dur="2000ms"
            repeatCount="indefinite"
          />
        </circle>

        {/* Additional glow effect for very recent data */}
        <circle
          cx={mostRecentPoint.x}
          cy={mostRecentPoint.y}
          r={6}
          fill="none"
          stroke={getMomentumColor(mostRecentPoint.momentum, mostRecentPoint.momentumIntensity)}
          strokeWidth="1"
          opacity={0.3}
        >
          <animate
            attributeName="r"
            values="3;8;3"
            dur="3000ms"
            repeatCount="indefinite"
          />
          <animate
            attributeName="opacity"
            values="0.5;0.1;0.5"
            dur="3000ms"
            repeatCount="indefinite"
          />
        </circle>
      </svg>
    </div>
  );
};

export default VolumeWeightedSparkline;