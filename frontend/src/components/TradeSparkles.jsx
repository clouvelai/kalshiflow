import React, { useState, useEffect, useRef } from 'react';

// Individual particle component
const Particle = ({ trade, onComplete }) => {
  const [position, setPosition] = useState({ x: 50, y: 50 });
  const [opacity, setOpacity] = useState(1);
  
  // Calculate size based on trade amount (price * count)
  const amount = trade.taker_side?.toLowerCase() === 'yes' 
    ? (trade.yes_price * trade.count) / 100
    : (trade.no_price * trade.count) / 100;
  
  // Size ranges from 4px to 20px based on amount
  const size = Math.min(20, Math.max(4, Math.log10(amount + 1) * 6));
  
  // Color based on trade side
  const isYes = trade.taker_side?.toLowerCase() === 'yes';
  const color = isYes ? '#10B981' : '#EF4444'; // green for YES, red for NO
  
  useEffect(() => {
    // Random starting position across width
    const startX = Math.random() * 100;
    const startY = 50;
    setPosition({ x: startX, y: startY });
    
    // Animate the particle
    const animationDuration = 3000 + Math.random() * 2000; // 3-5 seconds
    const startTime = Date.now();
    
    const animate = () => {
      const elapsed = Date.now() - startTime;
      const progress = elapsed / animationDuration;
      
      if (progress >= 1) {
        onComplete();
        return;
      }
      
      // Calculate new position
      const newY = isYes 
        ? startY - (progress * 60) // YES goes up
        : startY + (progress * 60); // NO goes down
      
      // Add slight horizontal drift
      const drift = Math.sin(progress * Math.PI * 2) * 10;
      const newX = startX + drift;
      
      // Fade out in last 30% of animation
      const newOpacity = progress > 0.7 ? 1 - ((progress - 0.7) / 0.3) : 1;
      
      setPosition({ x: newX, y: newY });
      setOpacity(newOpacity);
      
      requestAnimationFrame(animate);
    };
    
    requestAnimationFrame(animate);
  }, []);
  
  return (
    <div
      className="absolute pointer-events-none"
      style={{
        left: `${position.x}%`,
        top: `${position.y}%`,
        transform: 'translate(-50%, -50%)',
        opacity: opacity,
        transition: 'none',
      }}
    >
      <div
        className="rounded-full animate-pulse"
        style={{
          width: `${size}px`,
          height: `${size}px`,
          backgroundColor: color,
          boxShadow: `0 0 ${size * 2}px ${color}40, 0 0 ${size}px ${color}60`,
          filter: 'blur(0.5px)',
        }}
      />
    </div>
  );
};

const TradeSparkles = ({ trades = [] }) => {
  const [particles, setParticles] = useState([]);
  const [tradeQueue, setTradeQueue] = useState([]);
  const particleIdRef = useRef(0);
  const lastProcessedRef = useRef(null);
  
  // Process new trades
  useEffect(() => {
    if (trades.length === 0) return;
    
    // Get latest trade
    const latestTrade = trades[0];
    
    // Check if this is a new trade (different from last processed)
    if (lastProcessedRef.current && 
        latestTrade.ts === lastProcessedRef.current.ts &&
        latestTrade.market_ticker === lastProcessedRef.current.market_ticker) {
      return; // Already processed this trade
    }
    
    // Add to queue for staggered release
    setTradeQueue(prev => [...prev, latestTrade]);
    lastProcessedRef.current = latestTrade;
  }, [trades]);
  
  // Process trade queue with staggered timing
  useEffect(() => {
    if (tradeQueue.length === 0) return;
    
    const processNext = () => {
      setTradeQueue(prev => {
        if (prev.length === 0) return prev;
        
        const [trade, ...rest] = prev;
        
        // Create new particle
        const newParticle = {
          id: particleIdRef.current++,
          trade: trade,
        };
        
        setParticles(current => [...current, newParticle]);
        
        return rest;
      });
    };
    
    const timer = setTimeout(processNext, 100); // Release particles every 100ms
    return () => clearTimeout(timer);
  }, [tradeQueue]);
  
  // Clean up completed particles
  const handleParticleComplete = (id) => {
    setParticles(prev => prev.filter(p => p.id !== id));
  };
  
  // Limit max particles to prevent performance issues
  useEffect(() => {
    if (particles.length > 50) {
      setParticles(prev => prev.slice(-50));
    }
  }, [particles.length]);
  
  return (
    <div className="relative w-full bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 rounded-2xl overflow-hidden shadow-2xl border border-slate-800/50">
      {/* Background gradient effects */}
      <div className="absolute inset-0 bg-gradient-to-b from-green-500/5 via-transparent to-red-500/5"></div>
      <div className="absolute inset-0 bg-gradient-to-r from-transparent via-blue-500/5 to-transparent"></div>
      
      {/* Container for particles */}
      <div className="relative h-64 sm:h-72 lg:h-80" data-testid="trade-sparkles-container">
        {/* Center line indicator */}
        <div className="absolute inset-x-0 top-1/2 h-px bg-gradient-to-r from-transparent via-slate-700/50 to-transparent"></div>
        
        {/* Direction indicators */}
        <div className="absolute left-4 top-4 flex items-center space-x-2 text-xs text-slate-500">
          <div className="flex items-center space-x-1">
            <div className="w-2 h-2 bg-green-500 rounded-full"></div>
            <span>YES ↑</span>
          </div>
        </div>
        <div className="absolute left-4 bottom-4 flex items-center space-x-2 text-xs text-slate-500">
          <div className="flex items-center space-x-1">
            <div className="w-2 h-2 bg-red-500 rounded-full"></div>
            <span>NO ↓</span>
          </div>
        </div>
        
        {/* Trade counter */}
        <div className="absolute right-4 top-4 text-xs text-slate-500">
          <span className="font-mono">{particles.length} active</span>
        </div>
        
        {/* Render particles */}
        {particles.map(particle => (
          <Particle
            key={particle.id}
            trade={particle.trade}
            onComplete={() => handleParticleComplete(particle.id)}
          />
        ))}
        
        {/* Empty state */}
        {trades.length === 0 && particles.length === 0 && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-slate-600 text-sm animate-pulse">
              Waiting for trades...
            </div>
          </div>
        )}
      </div>
      
      {/* Bottom glow effect */}
      <div className="absolute bottom-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-blue-500/50 to-transparent"></div>
    </div>
  );
};

export default TradeSparkles;