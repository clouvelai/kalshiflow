import React, { useState, useEffect, useRef } from 'react';

// Individual trade drop component
const TradeDrop = ({ trade, onComplete }) => {
  const [position, setPosition] = useState({ x: -5, y: 0 });
  const [opacity, setOpacity] = useState(0);
  
  // Calculate size based on trade amount (price * count)
  const amount = trade.taker_side?.toLowerCase() === 'yes' 
    ? (trade.yes_price * trade.count) / 100
    : (trade.no_price * trade.count) / 100;
  
  // Sleek horizontal pills for flowing motion
  const width = Math.min(32, Math.max(12, Math.log10(amount + 1) * 12));
  const height = Math.min(12, Math.max(4, width * 0.35)); // Horizontal pill shape
  
  // Color based on trade side
  const isYes = trade.taker_side?.toLowerCase() === 'yes';
  const color = isYes ? '#10B981' : '#EF4444';
  
  useEffect(() => {
    // Two clean lanes with proper separation from meter
    const lanes = isYes ? [20, 25, 30] : [40, 45, 50];
    const startY = lanes[Math.floor(Math.random() * lanes.length)];
    const startX = -5;
    const endX = 95;
    
    // Start position
    let currentX = startX;
    let currentY = startY;
    let currentOpacity = 0;
    
    // Animation timing
    const animationDuration = 4000;
    const startTime = performance.now();
    let rafId;
    
    let frameCount = 0;
    const animate = (currentTime) => {
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / animationDuration, 1);
      
      if (progress >= 1) {
        onComplete(trade);
        return;
      }
      
      // Linear motion for smooth flow
      currentX = startX + (endX - startX) * progress;
      
      // Very subtle wave
      const waveOffset = Math.sin(progress * Math.PI * 3) * 1;
      currentY = startY + waveOffset;
      
      // Smooth fade in/out
      if (progress < 0.05) {
        currentOpacity = (progress / 0.05) * 0.85;
      } else if (progress < 0.95) {
        currentOpacity = 0.85;
      } else {
        currentOpacity = (1 - ((progress - 0.95) / 0.05)) * 0.85;
      }
      
      // Update state every 2 frames to reduce re-renders
      if (frameCount % 2 === 0) {
        setPosition({ x: currentX, y: currentY });
        setOpacity(currentOpacity);
      }
      frameCount++;
      
      rafId = requestAnimationFrame(animate);
    };
    
    rafId = requestAnimationFrame(animate);
    
    return () => {
      if (rafId) cancelAnimationFrame(rafId);
    };
  }, [isYes]);
  
  return (
    <div
      className="absolute pointer-events-none"
      style={{
        left: `${position.x}%`,
        top: `${position.y}%`,
        transform: `translate3d(-50%, -50%, 0)`,
        opacity: opacity,
        willChange: 'transform, opacity',
      }}
    >
      <div
        style={{
          width: `${width}px`,
          height: `${height}px`,
          borderRadius: `${height/2}px`,
          backgroundColor: color,
          transform: 'translateZ(0)',
        }}
      />
    </div>
  );
};

// Animated amount popup when trade completes
const AmountPopup = ({ trade, onComplete }) => {
  const [opacity, setOpacity] = useState(0);
  const [yOffset, setYOffset] = useState(0);
  
  const amount = trade.taker_side?.toLowerCase() === 'yes' 
    ? (trade.yes_price * trade.count) / 100
    : (trade.no_price * trade.count) / 100;
  
  const isYes = trade.taker_side?.toLowerCase() === 'yes';
  const formattedAmount = amount >= 1000 ? `+${(amount/1000).toFixed(1)}k` : `+${Math.round(amount)}`;
  
  // Make text smaller for trades under $100
  const textSize = amount < 100 ? 'text-[10px]' : 'text-xs';
  
  useEffect(() => {
    // Animate in and up
    const startTime = Date.now();
    const duration = 1500;
    
    const animate = () => {
      const elapsed = Date.now() - startTime;
      const progress = elapsed / duration;
      
      if (progress >= 1) {
        onComplete();
        return;
      }
      
      // Fade in then out
      let newOpacity;
      if (progress < 0.2) {
        newOpacity = progress / 0.2;
      } else if (progress < 0.7) {
        newOpacity = 1;
      } else {
        newOpacity = 1 - ((progress - 0.7) / 0.3);
      }
      
      // Float upward
      const newY = -progress * 30;
      
      setOpacity(newOpacity);
      setYOffset(newY);
      
      requestAnimationFrame(animate);
    };
    
    requestAnimationFrame(animate);
  }, []);
  
  return (
    <div
      className="absolute pointer-events-none"
      style={{
        right: '20px',
        top: isYes ? '25%' : '45%',  // NO popup moved up to align with NO lane
        transform: `translate(0, ${yOffset}px)`,
        opacity: opacity,
      }}
    >
      <div className={`px-2 py-0.5 rounded-full ${textSize} font-bold ${
        isYes ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
      }`}>
        {formattedAmount}
      </div>
    </div>
  );
};

// Horizontal meter at the bottom
const FlowMeter = ({ yesVolume, noVolume }) => {
  const total = yesVolume + noVolume;
  const yesPercent = total > 0 ? (yesVolume / total) * 100 : 50;
  const noPercent = total > 0 ? (noVolume / total) * 100 : 50;
  
  const formatVolume = (vol) => {
    if (vol >= 1000000) return `${(vol/1000000).toFixed(1)}M`;
    if (vol >= 1000) return `${(vol/1000).toFixed(1)}k`;
    return Math.round(vol).toString();
  };
  
  return (
    <div className="absolute bottom-0 left-0 right-0 bg-white/90 backdrop-blur-sm border-t border-gray-200 p-2">
      <div className="flex items-center space-x-3">
        {/* NO label and volume */}
        <div className="flex items-center space-x-1.5">
          <span className="text-xs font-medium text-red-600">NO</span>
          <span className="text-xs text-red-500">{formatVolume(noVolume)}</span>
        </div>
        
        {/* Meter bar */}
        <div className="flex-1 h-3 bg-gray-100 rounded-full overflow-hidden flex">
          <div 
            className="bg-gradient-to-r from-red-400 to-red-500 transition-all duration-500 ease-out"
            style={{ width: `${noPercent}%` }}
          />
          <div 
            className="bg-gradient-to-l from-green-400 to-green-500 transition-all duration-500 ease-out"
            style={{ width: `${yesPercent}%` }}
          />
        </div>
        
        {/* YES label and volume */}
        <div className="flex items-center space-x-1.5">
          <span className="text-xs text-green-500">{formatVolume(yesVolume)}</span>
          <span className="text-xs font-medium text-green-600">YES</span>
        </div>
      </div>
    </div>
  );
};

// Finish line indicator (just a subtle visual marker)
const FinishLine = () => {
  return (
    <div className="absolute right-8 top-0 bottom-0 w-8">
      {/* Subtle gradient fade */}
      <div className="absolute inset-0 bg-gradient-to-l from-gray-100/50 to-transparent"></div>
      {/* Dotted line */}
      <div className="absolute left-0 top-2 bottom-2 w-px border-l border-dashed border-gray-300"></div>
    </div>
  );
};

const TradeFlowRiver = ({ trades = [] }) => {
  const [drops, setDrops] = useState([]);
  const [tradeQueue, setTradeQueue] = useState([]);
  const [volumes, setVolumes] = useState({ yes: 0, no: 0 });
  const [amountPopups, setAmountPopups] = useState([]);
  const dropIdRef = useRef(0);
  const popupIdRef = useRef(0);
  const lastProcessedRef = useRef(null);
  
  // Process new trades
  useEffect(() => {
    if (trades.length === 0) return;
    
    const latestTrade = trades[0];
    
    if (lastProcessedRef.current && 
        lastProcessedRef.current.ts === latestTrade.ts) {
      return;
    }
    
    setTradeQueue(prev => [...prev, latestTrade]);
    lastProcessedRef.current = latestTrade;
  }, [trades]);
  
  // Process trade queue
  useEffect(() => {
    if (tradeQueue.length === 0) return;
    
    const processNext = () => {
      setTradeQueue(prev => {
        if (prev.length === 0) return prev;
        
        const [trade, ...rest] = prev;
        
        const newDrop = {
          id: dropIdRef.current++,
          trade: trade,
        };
        
        setDrops(current => [...current, newDrop]);
        
        return rest;
      });
    };
    
    const timer = setTimeout(processNext, 200); // Stagger drops
    return () => clearTimeout(timer);
  }, [tradeQueue]);
  
  // Handle drop completion and show amount popup
  const handleDropComplete = (id, trade) => {
    setDrops(prev => prev.filter(d => d.id !== id));
    
    // Calculate trade amount
    const isYes = trade.taker_side?.toLowerCase() === 'yes';
    const amount = isYes 
      ? (trade.yes_price * trade.count) / 100
      : (trade.no_price * trade.count) / 100;
    
    // Update volumes
    setVolumes(prev => ({
      yes: prev.yes + (isYes ? amount : 0),
      no: prev.no + (!isYes ? amount : 0)
    }));
    
    // Create amount popup
    const newPopup = {
      id: popupIdRef.current++,
      trade: trade
    };
    setAmountPopups(prev => [...prev, newPopup]);
  };
  
  // Handle popup completion
  const handlePopupComplete = (id) => {
    setAmountPopups(prev => prev.filter(p => p.id !== id));
  };
  
  // Limit drops to prevent performance issues
  // Can handle many drops but need some reasonable limit
  useEffect(() => {
    if (drops.length > 50) {
      setDrops(prev => prev.slice(-50));
    }
  }, [drops.length]);
  
  return (
    <div className="relative w-full bg-gradient-to-br from-gray-50 via-white to-gray-50 rounded-2xl overflow-hidden shadow-lg border border-gray-200">
      {/* Subtle gradient overlays */}
      <div className="absolute inset-0 bg-gradient-to-r from-blue-50/30 via-transparent to-purple-50/30"></div>
      
      {/* River lanes - subtle guides */}
      <div className="absolute inset-0">
        <div className="absolute inset-x-0 top-1/3 h-px bg-gradient-to-r from-transparent via-gray-200/30 to-transparent"></div>
        <div className="absolute inset-x-0 top-2/3 h-px bg-gradient-to-r from-transparent via-gray-200/30 to-transparent"></div>
      </div>
      
      {/* Container - proper height accounting for meter */}
      <div className="relative h-24 sm:h-28 lg:h-32" data-testid="trade-flow-river-container">
        
        {/* Render drops */}
        {drops.map(drop => (
          <TradeDrop
            key={drop.id}
            trade={drop.trade}
            onComplete={(trade) => handleDropComplete(drop.id, trade)}
          />
        ))}
        
        {/* Render amount popups */}
        {amountPopups.map(popup => (
          <AmountPopup
            key={popup.id}
            trade={popup.trade}
            onComplete={() => handlePopupComplete(popup.id)}
          />
        ))}
        
        {/* Finish line indicator */}
        <FinishLine />
        
        {/* Empty state */}
        {trades.length === 0 && drops.length === 0 && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-gray-400 text-sm">
              Waiting for trades...
            </div>
          </div>
        )}
      </div>
      
      {/* Horizontal flow meter at the bottom */}
      <FlowMeter yesVolume={volumes.yes} noVolume={volumes.no} />
    </div>
  );
};

export default React.memo(TradeFlowRiver, (prevProps, nextProps) => {
  // Only re-render if trades array reference changed or length changed
  // This prevents re-renders when parent updates for other reasons
  return (
    prevProps.trades === nextProps.trades ||
    (prevProps.trades?.length === nextProps.trades?.length &&
     prevProps.trades?.[0]?.trade_id === nextProps.trades?.[0]?.trade_id)
  );
});