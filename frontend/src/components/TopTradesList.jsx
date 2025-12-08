import React, { useMemo, useEffect, useRef, useState } from 'react';
import { TrendingUp, TrendingDown, Sparkles, ArrowUp, ArrowDown } from 'lucide-react';

const TopTradesList = ({ trades = [], windowMinutes = 10 }) => {
  const [previousTrades, setPreviousTrades] = useState([]);
  const [tradeTransitions, setTradeTransitions] = useState({});
  const prevTradesRef = useRef([]);
  
  // Format volume in thousands or millions
  const formatVolume = (volume) => {
    if (volume >= 1000000) {
      return `$${(volume / 1000000).toFixed(2)}M`;
    }
    if (volume >= 1000) {
      return `$${(volume / 1000).toFixed(1)}k`;
    }
    return `$${volume.toFixed(0)}`;
  };

  // Format price with cents symbol
  const formatPrice = (price) => {
    return `${price}¢`;
  };

  // Track trade transitions and positions
  useEffect(() => {
    const currentTradeIds = trades.slice(0, 10).map(t => 
      `${t.market_ticker}-${t.ts}-${t.volume_dollars}`
    );
    const prevTradeIds = prevTradesRef.current.slice(0, 10).map(t => 
      `${t.market_ticker}-${t.ts}-${t.volume_dollars}`
    );
    
    const transitions = {};
    
    // Identify new entries, exits, and position changes
    currentTradeIds.forEach((id, currentIndex) => {
      const prevIndex = prevTradeIds.indexOf(id);
      
      if (prevIndex === -1) {
        // New entry
        transitions[id] = {
          type: 'new',
          from: -1,
          to: currentIndex,
          animation: 'slide-in-glow'
        };
      } else if (prevIndex !== currentIndex) {
        // Position changed
        transitions[id] = {
          type: prevIndex > currentIndex ? 'rise' : 'fall',
          from: prevIndex,
          to: currentIndex,
          animation: 'position-change'
        };
      }
    });
    
    // Mark trades that are leaving
    prevTradeIds.forEach((id, prevIndex) => {
      if (!currentTradeIds.includes(id)) {
        transitions[id] = {
          type: 'exit',
          from: prevIndex,
          to: -1,
          animation: 'slide-out'
        };
      }
    });
    
    setTradeTransitions(transitions);
    prevTradesRef.current = trades;
    
    // Clear transitions after animation
    const timer = setTimeout(() => {
      setTradeTransitions({});
    }, 1000);
    
    return () => clearTimeout(timer);
  }, [trades]);

  // Split trades into two columns of 5 each (max 10 total)
  const { leftColumn, rightColumn } = useMemo(() => {
    const topTen = trades.slice(0, 10);
    return {
      leftColumn: topTen.slice(0, 5),
      rightColumn: topTen.slice(5, 10)
    };
  }, [trades]);

  const TradeCard = ({ trade, index }) => {
    const isYes = trade.taker_side === 'yes';
    const tradeId = `${trade.market_ticker}-${trade.ts}-${trade.volume_dollars}`;
    const transition = tradeTransitions[tradeId];
    const [isAnimating, setIsAnimating] = useState(false);
    
    useEffect(() => {
      if (transition) {
        setIsAnimating(true);
        const timer = setTimeout(() => setIsAnimating(false), 1000);
        return () => clearTimeout(timer);
      }
    }, [transition]);
    
    // Determine animation classes based on transition type
    const getAnimationClass = () => {
      if (!transition) return '';
      
      switch (transition.type) {
        case 'new':
          return 'animate-new-entry';
        case 'rise':
          return 'animate-position-rise';
        case 'fall':
          return 'animate-position-fall';
        case 'exit':
          return 'animate-exit';
        default:
          return '';
      }
    };
    
    const animationClass = getAnimationClass();
    const isNew = transition?.type === 'new';
    const isRising = transition?.type === 'rise';
    const isFalling = transition?.type === 'fall';
    
    return (
      <div 
        className={`
          relative bg-gray-900 rounded-lg p-2.5 border shadow-sm
          transition-all duration-700 transform
          ${isNew ? 'border-yellow-500/50 shadow-yellow-500/20' : 'border-gray-700'}
          ${isRising ? 'border-green-500/30' : ''}
          ${isFalling ? 'border-red-500/30' : ''}
          ${animationClass}
          hover:border-gray-600 hover:shadow-lg hover:bg-gray-850
          ${isAnimating ? 'z-10' : 'z-0'}
        `}
        style={{
          animationDelay: isNew ? `${index * 50}ms` : '0ms',
        }}
      >
        {/* New entry indicator */}
        {isNew && (
          <div className="absolute -top-1 -right-1 animate-pulse">
            <Sparkles className="w-4 h-4 text-yellow-500" />
          </div>
        )}
        
        {/* Position change indicator */}
        {isRising && (
          <div className="absolute -left-1 top-1/2 -translate-y-1/2 animate-bounce">
            <ArrowUp className="w-3 h-3 text-green-500" />
          </div>
        )}
        {isFalling && (
          <div className="absolute -left-1 top-1/2 -translate-y-1/2 animate-bounce">
            <ArrowDown className="w-3 h-3 text-red-500" />
          </div>
        )}
        
        {/* Glow effect for new entries */}
        {isNew && (
          <div className="absolute inset-0 rounded-lg bg-yellow-500/10 animate-glow-fade pointer-events-none" />
        )}
        
        <div className="flex justify-between items-start mb-1.5 relative">
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-0.5">
              <span className={`text-xs font-mono transition-colors duration-500 ${
                isNew ? 'text-yellow-400' : 'text-gray-400'
              }`}>
                #{index + 1}
              </span>
              <span className="text-xs font-semibold text-gray-100 truncate">
                {trade.market_ticker}
              </span>
            </div>
            {trade.title && (
              <p className="text-xs text-gray-400 line-clamp-1">
                {trade.title}
              </p>
            )}
          </div>
          <span className="text-xs text-gray-500 ml-2 whitespace-nowrap">
            {trade.time_ago}
          </span>
        </div>
        
        <div className="flex items-center justify-between mt-1.5 relative">
          <div className="flex items-center gap-1.5">
            {isYes ? (
              <TrendingUp className="w-3.5 h-3.5 text-green-500" />
            ) : (
              <TrendingDown className="w-3.5 h-3.5 text-red-500" />
            )}
            <span className={`text-sm font-medium ${
              isYes ? 'text-green-500' : 'text-red-500'
            }`}>
              {isYes ? 'YES' : 'NO'}
            </span>
            <span className="text-xs text-gray-400">
              @ {formatPrice(isYes ? trade.yes_price : trade.no_price)}
            </span>
          </div>
          
          <div className="text-right">
            <div className={`text-sm font-bold transition-all duration-500 ${
              isNew ? 'text-yellow-400 scale-110' : 'text-gray-100 scale-100'
            }`}>
              {formatVolume(trade.volume_dollars)}
            </div>
            <div className="text-xs text-gray-400">
              {trade.count.toLocaleString()} shares
            </div>
          </div>
        </div>
      </div>
    );
  };

  if (!trades || trades.length === 0) {
    return (
      <div className="bg-white rounded-xl p-6 border border-gray-200 shadow-sm">
        <h2 className="text-lg font-bold text-gray-900 mb-4">
          Top Trades by Volume
          <span className="ml-2 text-xs text-gray-500">
            (Last {windowMinutes} minutes)
          </span>
        </h2>
        <div className="text-center py-8 text-gray-500">
          Waiting for trade data...
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-xl p-6 border border-gray-200 shadow-sm overflow-hidden">
      <h2 className="text-lg font-bold text-gray-900 mb-4">
        Top Trades by Volume
        <span className="ml-2 text-xs text-gray-500">
          (Last {windowMinutes} minutes)
        </span>
      </h2>
      
      <div className="grid grid-cols-2 gap-3 relative">
        <div className="space-y-2">
          {leftColumn.map((trade, idx) => (
            <TradeCard
              key={`${trade.market_ticker}-${trade.ts}-${trade.volume_dollars}`}
              trade={trade}
              index={idx}
            />
          ))}
        </div>
        
        <div className="space-y-2">
          {rightColumn.map((trade, idx) => (
            <TradeCard
              key={`${trade.market_ticker}-${trade.ts}-${trade.volume_dollars}`}
              trade={trade}
              index={idx + leftColumn.length}
            />
          ))}
        </div>
      </div>
    </div>
  );
};

export default TopTradesList;