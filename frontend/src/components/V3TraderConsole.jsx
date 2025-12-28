import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Activity, Wifi, WifiOff, Circle, ChevronRight, ChevronDown, Zap, Database, TrendingUp, AlertCircle, Copy, Check, Info, CheckCircle, XCircle, ArrowRight, DollarSign, Briefcase, ShoppingCart, FileText, TrendingDown, Clock, Shield, Fish } from 'lucide-react';

// TradingData Component - Displays real-time trading state
const TradingData = ({ tradingState, lastUpdateTime }) => {
  if (!tradingState || !tradingState.has_state) {
    return (
      <div className="bg-gray-900/50 backdrop-blur-sm rounded-xl border border-gray-800 p-4">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-bold text-gray-300 uppercase tracking-wider">Trading Data</h3>
          <span className="text-xs text-gray-500 font-mono">No data available</span>
        </div>
      </div>
    );
  }

  const formatCurrency = (cents) => {
    const dollars = cents / 100;
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(dollars);
  };

  const formatTime = (timestamp) => {
    if (!timestamp) return 'N/A';
    const date = new Date(timestamp * 1000);
    return date.toLocaleTimeString('en-US', { 
      hour12: false, 
      hour: '2-digit', 
      minute: '2-digit', 
      second: '2-digit' 
    });
  };

  const getChangeIndicator = (value, isPositive = true) => {
    if (value === 0 || value === null || value === undefined) return null;
    const color = (isPositive && value > 0) || (!isPositive && value < 0) ? 'text-green-400' : 'text-red-400';
    const icon = value > 0 ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />;
    return (
      <span className={`flex items-center space-x-1 ${color} text-xs font-medium`}>
        {icon}
        <span>{value > 0 ? '+' : ''}{isPositive ? formatCurrency(value) : value}</span>
      </span>
    );
  };

  const changes = tradingState.changes || {};

  return (
    <div className="bg-gray-900/50 backdrop-blur-sm rounded-xl border border-gray-800 p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-bold text-gray-300 uppercase tracking-wider">Trading Data</h3>
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <Clock className="w-3.5 h-3.5 text-blue-400" />
            <span className="text-xs text-gray-400 font-mono">
              <span className="text-gray-500">Sync:</span> {formatTime(tradingState.sync_timestamp)}
            </span>
          </div>
          <div className="flex items-center space-x-2">
            <Activity className="w-3.5 h-3.5 text-green-400" />
            <span className="text-xs text-gray-400 font-mono">
              <span className="text-gray-500">Update:</span> {formatTime(lastUpdateTime)}
            </span>
          </div>
        </div>
      </div>
      
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {/* Balance */}
        <div className="bg-gray-800/30 rounded-lg p-3 border border-gray-700/50">
          <div className="flex items-center space-x-2 mb-1">
            <DollarSign className="w-4 h-4 text-green-400" />
            <span className="text-xs text-gray-500 uppercase">Balance</span>
          </div>
          <div className="text-lg font-mono font-bold text-white">
            {formatCurrency(tradingState.balance)}
          </div>
          {changes.balance_change && getChangeIndicator(changes.balance_change)}
        </div>

        {/* Portfolio Value */}
        <div className="bg-gray-800/30 rounded-lg p-3 border border-gray-700/50">
          <div className="flex items-center space-x-2 mb-1">
            <Briefcase className="w-4 h-4 text-blue-400" />
            <span className="text-xs text-gray-500 uppercase">Portfolio</span>
          </div>
          <div className="text-lg font-mono font-bold text-white">
            {formatCurrency(tradingState.portfolio_value)}
          </div>
          {changes.portfolio_change && getChangeIndicator(changes.portfolio_change)}
        </div>

        {/* Positions */}
        <div className="bg-gray-800/30 rounded-lg p-3 border border-gray-700/50">
          <div className="flex items-center space-x-2 mb-1">
            <ShoppingCart className="w-4 h-4 text-purple-400" />
            <span className="text-xs text-gray-500 uppercase">Positions</span>
          </div>
          <div className="text-lg font-mono font-bold text-white">
            {tradingState.position_count || 0}
          </div>
          {changes.position_count_change !== undefined && changes.position_count_change !== 0 && (
            <span className={`text-xs font-medium ${
              changes.position_count_change > 0 ? 'text-green-400' : 'text-red-400'
            }`}>
              {changes.position_count_change > 0 ? '+' : ''}{changes.position_count_change}
            </span>
          )}
        </div>

        {/* Orders */}
        <div className="bg-gray-800/30 rounded-lg p-3 border border-gray-700/50">
          <div className="flex items-center space-x-2 mb-1">
            <FileText className="w-4 h-4 text-yellow-400" />
            <span className="text-xs text-gray-500 uppercase">Orders</span>
          </div>
          <div className="text-lg font-mono font-bold text-white">
            {tradingState.order_count || 0}
          </div>
          {changes.order_count_change !== undefined && changes.order_count_change !== 0 && (
            <span className={`text-xs font-medium ${
              changes.order_count_change > 0 ? 'text-yellow-400' : 'text-gray-400'
            }`}>
              {changes.order_count_change > 0 ? '+' : ''}{changes.order_count_change}
            </span>
          )}
        </div>
      </div>

      {/* Order Group Status - Clean minimal display */}
      {tradingState.order_group && tradingState.order_group.id && (
        <div className="mt-4 bg-gray-800/30 rounded-lg p-3 border border-gray-700/50">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Shield className="w-4 h-4 text-indigo-400" />
              <span className="text-xs text-gray-500 uppercase font-medium">Order Group</span>
              <span className="text-sm font-mono text-gray-300">
                {tradingState.order_group.id}
              </span>
            </div>
            <div className="flex items-center space-x-3">
              <span className={`px-2 py-0.5 text-xs font-bold rounded-full ${
                tradingState.order_group.status === 'active'
                  ? 'bg-green-900/50 text-green-400 border border-green-700/50'
                  : tradingState.order_group.status === 'inactive'
                  ? 'bg-gray-900/50 text-gray-400 border border-gray-700/50'
                  : 'bg-yellow-900/50 text-yellow-400 border border-yellow-700/50'
              }`}>
                {(tradingState.order_group.status || 'unknown').toUpperCase()}
              </span>
              <span className="text-sm text-gray-400">
                <span className="font-mono font-bold text-white">{tradingState.order_group.order_count || 0}</span>
                <span className="text-gray-500 ml-1">{(tradingState.order_group.order_count || 0) === 1 ? 'order' : 'orders'}</span>
              </span>
            </div>
          </div>

          {/* Show order IDs if available and there are orders */}
          {tradingState.order_group.order_ids && tradingState.order_group.order_ids.length > 0 && (
            <div className="mt-2 pt-2 border-t border-gray-700/30">
              <div className="flex items-center flex-wrap gap-1.5">
                {tradingState.order_group.order_ids.slice(0, 8).map((orderId, index) => (
                  <span
                    key={orderId}
                    className="text-xs font-mono text-gray-500 bg-gray-800/50 px-2 py-0.5 rounded"
                  >
                    {orderId.substring(0, 8)}
                  </span>
                ))}
                {tradingState.order_group.order_ids.length > 8 && (
                  <span className="text-xs text-gray-600">
                    +{tradingState.order_group.order_ids.length - 8} more
                  </span>
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

// WhaleQueuePanel Component - Displays detected whale bets with smooth animations
const WhaleQueuePanel = ({ whaleQueue, processingWhaleId }) => {
  const [currentTime, setCurrentTime] = useState(Date.now());
  // Local state for smooth display - keeps whales visible for minimum time
  const [displayQueue, setDisplayQueue] = useState([]);
  // Track which whales are fading out
  const [fadingWhales, setFadingWhales] = useState(new Set());
  // Track when each whale was first seen (for minimum display time)
  const whaleFirstSeenRef = useRef(new Map());
  // Track which whales have pending removal timeouts to avoid duplicates
  const pendingRemovalRef = useRef(new Set());
  // Minimum display time in ms (3 seconds)
  const MIN_DISPLAY_TIME = 3000;
  // Fade out duration in ms
  const FADE_OUT_DURATION = 500;

  // Debug logging for whale queue data (reduced frequency)
  useEffect(() => {
    if (whaleQueue?.queue?.length > 0) {
      console.log('[WhaleQueuePanel] Queue update:', whaleQueue?.queue?.length, 'whales');
    }
  }, [whaleQueue?.queue?.length]);

  // Merge incoming queue with display queue, ensuring minimum display time
  useEffect(() => {
    const incomingQueue = whaleQueue?.queue || [];
    const incomingIds = new Set(incomingQueue.map(w => w.whale_id));
    const now = Date.now();

    // Add new whales and update existing ones
    incomingQueue.forEach(whale => {
      if (!whaleFirstSeenRef.current.has(whale.whale_id)) {
        whaleFirstSeenRef.current.set(whale.whale_id, now);
      }
    });

    // Merge: keep whales that are still in incoming OR haven't met minimum display time
    setDisplayQueue(prevQueue => {
      const mergedMap = new Map();

      // First, add all incoming whales (with latest data)
      incomingQueue.forEach(whale => {
        mergedMap.set(whale.whale_id, { ...whale, isRemoving: false });
        // If whale reappears in queue, cancel any pending removal
        pendingRemovalRef.current.delete(whale.whale_id);
      });

      // Then, keep any whales from prev that haven't met minimum display time
      prevQueue.forEach(whale => {
        if (!incomingIds.has(whale.whale_id)) {
          const firstSeen = whaleFirstSeenRef.current.get(whale.whale_id);
          const timeVisible = now - (firstSeen || now);

          // Only process if not already scheduled for removal and not already fading
          if (timeVisible < MIN_DISPLAY_TIME &&
              !fadingWhales.has(whale.whale_id) &&
              !pendingRemovalRef.current.has(whale.whale_id)) {
            // Keep in display queue but mark as processing/removing
            mergedMap.set(whale.whale_id, { ...whale, isRemoving: true });

            // Mark as pending removal to prevent duplicate timeouts
            pendingRemovalRef.current.add(whale.whale_id);

            // Schedule fade out after remaining time
            const remainingTime = Math.max(0, MIN_DISPLAY_TIME - timeVisible);
            setTimeout(() => {
              setFadingWhales(prev => new Set([...prev, whale.whale_id]));
              // Remove from display after fade animation
              setTimeout(() => {
                setDisplayQueue(q => q.filter(w => w.whale_id !== whale.whale_id));
                setFadingWhales(prev => {
                  const next = new Set(prev);
                  next.delete(whale.whale_id);
                  return next;
                });
                whaleFirstSeenRef.current.delete(whale.whale_id);
                pendingRemovalRef.current.delete(whale.whale_id);
              }, FADE_OUT_DURATION);
            }, remainingTime);
          } else if (fadingWhales.has(whale.whale_id) || pendingRemovalRef.current.has(whale.whale_id)) {
            // Keep whale visible while it's fading or pending removal
            mergedMap.set(whale.whale_id, { ...whale, isRemoving: true });
          }
        }
      });

      return Array.from(mergedMap.values());
    });
  }, [whaleQueue?.queue, fadingWhales]);

  // Update current time every second for age display
  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentTime(Date.now());
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  const formatCurrency = (dollars) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(dollars);
  };

  const formatAge = (ageSeconds) => {
    if (ageSeconds < 60) {
      return `${Math.floor(ageSeconds)}s ago`;
    } else if (ageSeconds < 3600) {
      return `${Math.floor(ageSeconds / 60)}m ago`;
    } else {
      return `${Math.floor(ageSeconds / 3600)}h ago`;
    }
  };

  const queue = displayQueue;
  const stats = whaleQueue?.stats || { trades_seen: 0, trades_discarded: 0, discard_rate_percent: 0 };
  // P2: Track followed whale IDs to show indicator
  const followedWhaleIds = new Set(whaleQueue?.followed_whale_ids || []);
  const followedCount = followedWhaleIds.size;

  return (
    <div className="bg-gray-900/50 backdrop-blur-sm rounded-xl border border-gray-800 p-4">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-2">
          <Fish className="w-4 h-4 text-cyan-400" />
          <h3 className="text-sm font-bold text-gray-300 uppercase tracking-wider">Whale Queue</h3>
        </div>
        <div className="flex items-center space-x-4 text-xs text-gray-400">
          <span className="font-mono">
            <span className="text-cyan-400">{queue.length}</span> tracked
          </span>
          <span className="text-gray-600">|</span>
          <span className="font-mono">
            <span className="text-green-400">{followedCount}</span> followed
          </span>
          <span className="text-gray-600">|</span>
          <span className="font-mono">
            <span className="text-gray-500">{stats.trades_discarded.toLocaleString()}</span> discarded
          </span>
          <span className="text-gray-600">|</span>
          <span className="font-mono">
            <span className="text-gray-500">{stats.discard_rate_percent.toFixed(1)}%</span> filtered
          </span>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-5 gap-4 mb-4">
        <div className="bg-gray-800/30 rounded-lg p-3 border border-gray-700/50">
          <div className="text-xs text-gray-500 uppercase mb-1">Trades Seen</div>
          <div className="text-lg font-mono font-bold text-white">{stats.trades_seen.toLocaleString()}</div>
        </div>
        <div className="bg-gray-800/30 rounded-lg p-3 border border-gray-700/50">
          <div className="text-xs text-gray-500 uppercase mb-1">Whales Found</div>
          <div className="text-lg font-mono font-bold text-cyan-400">{queue.length}</div>
        </div>
        <div className="bg-gray-800/30 rounded-lg p-3 border border-green-700/30">
          <div className="text-xs text-gray-500 uppercase mb-1">Followed</div>
          <div className="text-lg font-mono font-bold text-green-400">{followedCount}</div>
        </div>
        <div className="bg-gray-800/30 rounded-lg p-3 border border-gray-700/50">
          <div className="text-xs text-gray-500 uppercase mb-1">Discarded</div>
          <div className="text-lg font-mono font-bold text-gray-400">{stats.trades_discarded.toLocaleString()}</div>
        </div>
        <div className="bg-gray-800/30 rounded-lg p-3 border border-gray-700/50">
          <div className="text-xs text-gray-500 uppercase mb-1">Filter Rate</div>
          <div className="text-lg font-mono font-bold text-gray-400">{stats.discard_rate_percent.toFixed(1)}%</div>
        </div>
      </div>

      {/* Whale Queue Table */}
      {queue.length === 0 ? (
        <div className="bg-gray-800/30 rounded-lg p-8 border border-gray-700/50 text-center">
          <Fish className="w-8 h-8 text-gray-600 mx-auto mb-3" />
          <div className="text-gray-500 text-sm">No whales detected yet</div>
          <div className="text-gray-600 text-xs mt-1">Waiting for large trades (min ${stats.min_size_dollars || 10})</div>
        </div>
      ) : (
        <div className="bg-gray-800/30 rounded-lg border border-gray-700/50 overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-gray-900/50 border-b border-gray-700/50">
                <th className="px-3 py-2 text-center text-xs text-gray-500 uppercase font-medium">Status</th>
                <th className="px-3 py-2 text-left text-xs text-gray-500 uppercase font-medium">Market</th>
                <th className="px-3 py-2 text-center text-xs text-gray-500 uppercase font-medium">Side</th>
                <th className="px-3 py-2 text-right text-xs text-gray-500 uppercase font-medium">Price</th>
                <th className="px-3 py-2 text-right text-xs text-gray-500 uppercase font-medium">Count</th>
                <th className="px-3 py-2 text-right text-xs text-gray-500 uppercase font-medium">Cost</th>
                <th className="px-3 py-2 text-right text-xs text-gray-500 uppercase font-medium">Payout</th>
                <th className="px-3 py-2 text-right text-xs text-gray-500 uppercase font-medium">Profit</th>
                <th className="px-3 py-2 text-right text-xs text-gray-500 uppercase font-medium">Age</th>
              </tr>
            </thead>
            <tbody>
              {queue.map((whale, index) => {
                const isFollowed = followedWhaleIds.has(whale.whale_id);
                const isProcessing = processingWhaleId === whale.whale_id;
                const isFading = fadingWhales.has(whale.whale_id);
                const isRemoving = whale.isRemoving;

                // Determine status for display
                let statusDisplay;
                if (isFollowed) {
                  statusDisplay = (
                    <span className="inline-flex items-center space-x-1 px-2 py-0.5 rounded text-xs font-bold bg-green-900/30 text-green-400 border border-green-700/50 animate-pulse">
                      <CheckCircle className="w-3 h-3" />
                      <span>FOLLOWED</span>
                    </span>
                  );
                } else if (isProcessing) {
                  statusDisplay = (
                    <span className="inline-flex items-center space-x-1 px-2 py-0.5 rounded text-xs font-bold bg-blue-900/30 text-blue-400 border border-blue-700/50">
                      <Activity className="w-3 h-3 animate-spin" />
                      <span>PROCESSING</span>
                    </span>
                  );
                } else if (isRemoving) {
                  statusDisplay = (
                    <span className="inline-flex items-center space-x-1 px-2 py-0.5 rounded text-xs font-bold bg-yellow-900/30 text-yellow-400 border border-yellow-700/50">
                      <Clock className="w-3 h-3" />
                      <span>PROCESSED</span>
                    </span>
                  );
                } else {
                  statusDisplay = (
                    <span className="inline-flex items-center space-x-1 px-2 py-0.5 rounded text-xs font-bold bg-cyan-900/30 text-cyan-400 border border-cyan-700/50">
                      <Fish className="w-3 h-3" />
                      <span>QUEUED</span>
                    </span>
                  );
                }

                return (
                  <tr
                    key={whale.whale_id || `${whale.market_ticker}-${whale.price_cents}-${index}`}
                    className={`border-b border-gray-700/30 transition-all duration-500 ease-in-out ${
                      isFading
                        ? 'opacity-0 transform translate-x-4'
                        : 'opacity-100 transform translate-x-0'
                    } ${
                      isProcessing
                        ? 'bg-blue-500/20 ring-1 ring-blue-400/50 ring-inset animate-pulse'
                        : isFollowed
                        ? 'bg-green-900/20 ring-1 ring-green-700/30 ring-inset'
                        : isRemoving
                        ? 'bg-yellow-900/10'
                        : 'hover:bg-gray-800/50'
                    }`}
                    style={{
                      animation: !isFading && !isRemoving ? 'slideInFromLeft 0.3s ease-out' : undefined
                    }}
                  >
                    <td className="px-3 py-2 text-center">
                      {statusDisplay}
                    </td>
                    <td className="px-3 py-2 font-mono text-gray-300 text-xs">{whale.market_ticker}</td>
                    <td className="px-3 py-2 text-center">
                      <span className={`px-2 py-0.5 rounded text-xs font-bold uppercase ${
                        whale.side === 'yes'
                          ? 'bg-green-900/30 text-green-400 border border-green-700/50'
                          : 'bg-red-900/30 text-red-400 border border-red-700/50'
                      }`}>
                        {whale.side}
                      </span>
                    </td>
                    <td className="px-3 py-2 text-right font-mono text-gray-300">{whale.price_cents}c</td>
                    <td className="px-3 py-2 text-right font-mono text-gray-300">{whale.count.toLocaleString()}</td>
                    <td className="px-3 py-2 text-right font-mono text-gray-400">{formatCurrency(whale.cost_dollars)}</td>
                    <td className="px-3 py-2 text-right font-mono text-gray-400">{formatCurrency(whale.payout_dollars)}</td>
                    {/* TODO: Subtract fees in a later milestone */}
                    <td className="px-3 py-2 text-right font-mono font-bold text-cyan-400">{formatCurrency(whale.payout_dollars - whale.cost_dollars)}</td>
                    <td className="px-3 py-2 text-right font-mono text-gray-500 text-xs">{formatAge(whale.age_seconds)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

// FollowedTradesPanel Component - Shows trades we've followed (persists beyond whale queue window)
const FollowedTradesPanel = ({ followedWhales }) => {
  // Don't render if no followed trades
  if (!followedWhales || followedWhales.length === 0) {
    return null;
  }

  const formatAge = (ageSeconds) => {
    if (ageSeconds < 60) {
      return `${Math.floor(ageSeconds)}s ago`;
    } else if (ageSeconds < 3600) {
      return `${Math.floor(ageSeconds / 60)}m ago`;
    } else {
      return `${Math.floor(ageSeconds / 3600)}h ago`;
    }
  };

  return (
    <div className="bg-gray-900/50 backdrop-blur-sm rounded-xl border border-green-800/50 p-4 mt-4">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-2">
          <CheckCircle className="w-4 h-4 text-green-400" />
          <h3 className="text-sm font-bold text-green-400 uppercase tracking-wider">Followed Trades</h3>
        </div>
        <span className="text-xs text-gray-400 font-mono">
          {followedWhales.length} trade{followedWhales.length !== 1 ? 's' : ''}
        </span>
      </div>

      <div className="bg-gray-800/30 rounded-lg border border-gray-700/50 overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-gray-900/50 border-b border-gray-700/50">
              <th className="px-3 py-2 text-left text-xs text-gray-500 uppercase font-medium">Market</th>
              <th className="px-3 py-2 text-center text-xs text-gray-500 uppercase font-medium">Side</th>
              <th className="px-3 py-2 text-right text-xs text-gray-500 uppercase font-medium">Price</th>
              <th className="px-3 py-2 text-right text-xs text-gray-500 uppercase font-medium">Count</th>
              <th className="px-3 py-2 text-right text-xs text-gray-500 uppercase font-medium">Cost</th>
              <th className="px-3 py-2 text-right text-xs text-gray-500 uppercase font-medium">Payout</th>
              <th className="px-3 py-2 text-right text-xs text-gray-500 uppercase font-medium">Size</th>
              <th className="px-3 py-2 text-right text-xs text-gray-500 uppercase font-medium">Age</th>
            </tr>
          </thead>
          <tbody>
            {followedWhales.map((trade, index) => (
              <tr
                key={trade.whale_id || index}
                className="border-b border-gray-700/30 hover:bg-gray-800/50 transition-colors bg-green-900/5"
              >
                <td className="px-3 py-2 font-mono text-gray-300 text-xs">{trade.market_ticker}</td>
                <td className="px-3 py-2 text-center">
                  <span className={`px-2 py-0.5 rounded text-xs font-bold uppercase ${
                    trade.side === 'yes'
                      ? 'bg-green-900/30 text-green-400 border border-green-700/50'
                      : 'bg-red-900/30 text-red-400 border border-red-700/50'
                  }`}>
                    {trade.side}
                  </span>
                </td>
                <td className="px-3 py-2 text-right font-mono text-gray-300">{trade.price_cents}c</td>
                <td className="px-3 py-2 text-right font-mono text-gray-300">{trade.our_count}</td>
                <td className="px-3 py-2 text-right font-mono text-gray-400">${trade.cost_dollars?.toFixed(2)}</td>
                <td className="px-3 py-2 text-right font-mono text-gray-400">${trade.payout_dollars?.toFixed(2)}</td>
                {/* TODO: Subtract fees in a later milestone */}
                <td className="px-3 py-2 text-right font-mono font-bold text-green-400">${((trade.payout_dollars || 0) - (trade.cost_dollars || 0)).toFixed(2)}</td>
                <td className="px-3 py-2 text-right font-mono text-gray-500 text-xs">{formatAge(trade.age_seconds)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

// DecisionAuditPanel Component - Shows why whales were followed/skipped
const DecisionAuditPanel = ({ decisionHistory, decisionStats }) => {
  const [isExpanded, setIsExpanded] = useState(true);

  const formatAge = (ageSeconds) => {
    if (ageSeconds < 60) {
      return `${Math.floor(ageSeconds)}s ago`;
    } else if (ageSeconds < 3600) {
      return `${Math.floor(ageSeconds / 60)}m ago`;
    } else {
      return `${Math.floor(ageSeconds / 3600)}h ago`;
    }
  };

  const getActionStyle = (action) => {
    switch (action) {
      case 'followed':
        return 'bg-green-900/30 text-green-400 border-green-700/50';
      case 'skipped_age':
        return 'bg-yellow-900/30 text-yellow-400 border-yellow-700/50';
      case 'skipped_position':
        return 'bg-purple-900/30 text-purple-400 border-purple-700/50';
      case 'skipped_orders':
        return 'bg-blue-900/30 text-blue-400 border-blue-700/50';
      case 'already_followed':
        return 'bg-gray-900/30 text-gray-400 border-gray-700/50';
      case 'rate_limited':
        return 'bg-red-900/30 text-red-400 border-red-700/50';
      default:
        return 'bg-gray-900/30 text-gray-400 border-gray-700/50';
    }
  };

  const getActionLabel = (action) => {
    switch (action) {
      case 'followed': return 'FOLLOWED';
      case 'skipped_age': return 'TOO OLD';
      case 'skipped_position': return 'HAS POSITION';
      case 'skipped_orders': return 'HAS ORDERS';
      case 'already_followed': return 'ALREADY DONE';
      case 'rate_limited': return 'RATE LIMITED';
      default: return action.toUpperCase();
    }
  };

  // Calculate stats summary
  const stats = decisionStats || {};
  const detected = stats.whales_detected || 0;
  const followed = stats.whales_followed || 0;
  const skipped = stats.whales_skipped || 0;
  const followRate = detected > 0 ? ((followed / detected) * 100).toFixed(1) : 0;

  return (
    <div className="bg-gray-900/50 backdrop-blur-sm rounded-xl border border-gray-800 p-4 mt-4">
      {/* Header with stats summary */}
      <div
        className="flex items-center justify-between cursor-pointer"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center space-x-2">
          <FileText className="w-4 h-4 text-orange-400" />
          <h3 className="text-sm font-bold text-gray-300 uppercase tracking-wider">Decision Audit</h3>
        </div>
        <div className="flex items-center space-x-4">
          {/* Stats summary row */}
          <div className="flex items-center space-x-3 text-xs">
            <span className="font-mono">
              <span className="text-gray-500">Detected:</span> <span className="text-white font-bold">{detected}</span>
            </span>
            <span className="text-gray-600">|</span>
            <span className="font-mono">
              <span className="text-gray-500">Followed:</span> <span className="text-green-400 font-bold">{followed}</span>
            </span>
            <span className="text-gray-600">|</span>
            <span className="font-mono">
              <span className="text-gray-500">Skipped:</span> <span className="text-yellow-400 font-bold">{skipped}</span>
            </span>
            <span className="text-gray-600">|</span>
            <span className="font-mono">
              <span className="text-gray-500">Rate:</span> <span className="text-cyan-400 font-bold">{followRate}%</span>
            </span>
          </div>
          {isExpanded ? (
            <ChevronDown className="w-4 h-4 text-gray-500" />
          ) : (
            <ChevronRight className="w-4 h-4 text-gray-500" />
          )}
        </div>
      </div>

      {/* Skip reason breakdown */}
      {isExpanded && stats && (
        <div className="grid grid-cols-6 gap-2 mt-4 mb-4">
          <div className="bg-gray-800/30 rounded-lg p-2 border border-gray-700/50 text-center">
            <div className="text-xs text-gray-500 uppercase">Age</div>
            <div className="text-sm font-mono font-bold text-yellow-400">{stats.skipped_age || 0}</div>
          </div>
          <div className="bg-gray-800/30 rounded-lg p-2 border border-gray-700/50 text-center">
            <div className="text-xs text-gray-500 uppercase">Position</div>
            <div className="text-sm font-mono font-bold text-purple-400">{stats.skipped_position || 0}</div>
          </div>
          <div className="bg-gray-800/30 rounded-lg p-2 border border-gray-700/50 text-center">
            <div className="text-xs text-gray-500 uppercase">Orders</div>
            <div className="text-sm font-mono font-bold text-blue-400">{stats.skipped_orders || 0}</div>
          </div>
          <div className="bg-gray-800/30 rounded-lg p-2 border border-gray-700/50 text-center">
            <div className="text-xs text-gray-500 uppercase">Already</div>
            <div className="text-sm font-mono font-bold text-gray-400">{stats.already_followed || 0}</div>
          </div>
          <div className="bg-gray-800/30 rounded-lg p-2 border border-gray-700/50 text-center">
            <div className="text-xs text-gray-500 uppercase">Rate Limit</div>
            <div className="text-sm font-mono font-bold text-red-400">{stats.rate_limited || 0}</div>
          </div>
          <div className="bg-green-900/20 rounded-lg p-2 border border-green-700/30 text-center">
            <div className="text-xs text-gray-500 uppercase">Followed</div>
            <div className="text-sm font-mono font-bold text-green-400">{stats.whales_followed || 0}</div>
          </div>
        </div>
      )}

      {/* Decision history table */}
      {isExpanded && decisionHistory && decisionHistory.length > 0 && (
        <div className="bg-gray-800/30 rounded-lg border border-gray-700/50 overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-gray-900/50 border-b border-gray-700/50">
                <th className="px-3 py-2 text-center text-xs text-gray-500 uppercase font-medium">Action</th>
                <th className="px-3 py-2 text-left text-xs text-gray-500 uppercase font-medium">Market</th>
                <th className="px-3 py-2 text-center text-xs text-gray-500 uppercase font-medium">Side</th>
                <th className="px-3 py-2 text-right text-xs text-gray-500 uppercase font-medium">Size</th>
                <th className="px-3 py-2 text-left text-xs text-gray-500 uppercase font-medium">Reason</th>
                <th className="px-3 py-2 text-right text-xs text-gray-500 uppercase font-medium">Age</th>
              </tr>
            </thead>
            <tbody>
              {decisionHistory.slice(0, 10).map((decision, index) => (
                <tr
                  key={`${decision.whale_id}-${index}`}
                  className={`border-b border-gray-700/30 hover:bg-gray-800/50 transition-colors ${
                    decision.action === 'followed' ? 'bg-green-900/10' : ''
                  }`}
                >
                  <td className="px-3 py-2 text-center">
                    <span className={`px-2 py-0.5 rounded text-xs font-bold uppercase border ${getActionStyle(decision.action)}`}>
                      {getActionLabel(decision.action)}
                    </span>
                  </td>
                  <td className="px-3 py-2 font-mono text-gray-300 text-xs">{decision.market_ticker}</td>
                  <td className="px-3 py-2 text-center">
                    <span className={`px-2 py-0.5 rounded text-xs font-bold uppercase ${
                      decision.side === 'yes'
                        ? 'bg-green-900/30 text-green-400 border border-green-700/50'
                        : 'bg-red-900/30 text-red-400 border border-red-700/50'
                    }`}>
                      {decision.side}
                    </span>
                  </td>
                  <td className="px-3 py-2 text-right font-mono text-gray-400 text-xs">
                    ${decision.whale_size_dollars?.toFixed(0) || 0}
                  </td>
                  <td className="px-3 py-2 text-gray-400 text-xs truncate max-w-xs" title={decision.reason}>
                    {decision.reason}
                  </td>
                  <td className="px-3 py-2 text-right font-mono text-gray-500 text-xs">
                    {formatAge(decision.age_seconds)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Empty state */}
      {isExpanded && (!decisionHistory || decisionHistory.length === 0) && (
        <div className="bg-gray-800/30 rounded-lg p-6 border border-gray-700/50 text-center mt-4">
          <FileText className="w-6 h-6 text-gray-600 mx-auto mb-2" />
          <div className="text-gray-500 text-sm">No decisions yet</div>
          <div className="text-gray-600 text-xs mt-1">Waiting for whale queue evaluations</div>
        </div>
      )}
    </div>
  );
};

// SessionSummaryPanel Component - Clear P&L display with cash flow visibility
const SessionSummaryPanel = ({ tradingState }) => {
  if (!tradingState || !tradingState.has_state) {
    return null;
  }

  const formatCurrency = (cents) => {
    const dollars = cents / 100;
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(dollars);
  };

  const formatPnLCurrency = (cents) => {
    const dollars = cents / 100;
    const prefix = cents >= 0 ? '+' : '';
    return prefix + new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(dollars);
  };

  const balance = tradingState.balance || 0;
  const portfolioValue = tradingState.portfolio_value || 0;
  const totalValue = balance + portfolioValue;
  const pnl = tradingState.pnl || null;

  return (
    <div className="bg-gray-900/50 backdrop-blur-sm rounded-xl border border-gray-800 p-4 mb-4">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-2">
          <TrendingUp className="w-4 h-4 text-cyan-400" />
          <h3 className="text-sm font-bold text-gray-300 uppercase tracking-wider">Session Summary</h3>
        </div>
        {pnl && pnl.session_start_time && (
          <span className="text-xs text-gray-500 font-mono">
            Session started {new Date(pnl.session_start_time * 1000).toLocaleTimeString('en-US', { hour12: false })}
          </span>
        )}
      </div>

      <div className="grid grid-cols-4 gap-4">
        {/* Cash Available */}
        <div className="bg-gray-800/30 rounded-lg p-4 border border-gray-700/50">
          <div className="flex items-center space-x-2 mb-2">
            <DollarSign className="w-4 h-4 text-green-400" />
            <span className="text-xs text-gray-500 uppercase">Cash Available</span>
          </div>
          <div className="text-2xl font-mono font-bold text-green-400">
            {formatCurrency(balance)}
          </div>
        </div>

        {/* In Positions */}
        <div className="bg-gray-800/30 rounded-lg p-4 border border-gray-700/50">
          <div className="flex items-center space-x-2 mb-2">
            <Briefcase className="w-4 h-4 text-blue-400" />
            <span className="text-xs text-gray-500 uppercase">In Positions</span>
          </div>
          <div className="text-2xl font-mono font-bold text-blue-400">
            {formatCurrency(portfolioValue)}
          </div>
        </div>

        {/* Total Value */}
        <div className="bg-gray-800/30 rounded-lg p-4 border border-gray-700/50">
          <div className="flex items-center space-x-2 mb-2">
            <Database className="w-4 h-4 text-white" />
            <span className="text-xs text-gray-500 uppercase">Total Value</span>
          </div>
          <div className="text-2xl font-mono font-bold text-white">
            {formatCurrency(totalValue)}
          </div>
        </div>

        {/* Session P&L (Total = Realized + Unrealized) */}
        <div className={`bg-gray-800/30 rounded-lg p-4 border ${
          pnl && pnl.session_pnl >= 0 ? 'border-green-700/30' : 'border-red-700/30'
        }`}>
          <div className="flex items-center space-x-2 mb-2">
            {pnl && pnl.session_pnl >= 0 ? (
              <TrendingUp className="w-4 h-4 text-green-400" />
            ) : (
              <TrendingDown className="w-4 h-4 text-red-400" />
            )}
            <span className="text-xs text-gray-500 uppercase" title="Total P&L = Realized + Unrealized">Session P&L</span>
          </div>
          {pnl ? (
            <>
              <div className={`text-2xl font-mono font-bold ${
                pnl.session_pnl >= 0 ? 'text-green-400' : 'text-red-400'
              }`}>
                {formatPnLCurrency(pnl.session_pnl)}
              </div>
              <div className={`text-xs font-mono ${
                pnl.session_pnl_percent >= 0 ? 'text-green-400' : 'text-red-400'
              }`}>
                {pnl.session_pnl_percent >= 0 ? '+' : ''}{pnl.session_pnl_percent}%
              </div>
            </>
          ) : (
            <div className="text-2xl font-mono font-bold text-gray-500">--</div>
          )}
        </div>
      </div>
    </div>
  );
};

// PositionListPanel Component - Detailed position breakdown with per-position P&L
const PositionListPanel = ({ positions }) => {
  const [isExpanded, setIsExpanded] = useState(true);
  const [changedTickers, setChangedTickers] = useState(new Set());
  const prevPositionsRef = useRef({});

  // Detect position changes and trigger subtle highlight animation
  useEffect(() => {
    if (!positions || positions.length === 0) return;

    const changed = new Set();
    positions.forEach(pos => {
      const prev = prevPositionsRef.current[pos.ticker];
      if (prev) {
        // Detect changes in key fields
        if (prev.position !== pos.position ||
            prev.total_traded !== pos.total_traded ||
            prev.unrealized_pnl !== pos.unrealized_pnl) {
          changed.add(pos.ticker);
        }
      } else {
        // New position - highlight it
        changed.add(pos.ticker);
      }
    });

    // Update ref for next comparison (always, so we track all positions)
    const newRef = {};
    positions.forEach(pos => {
      newRef[pos.ticker] = { ...pos };
    });
    prevPositionsRef.current = newRef;

    // Trigger highlight animation if anything changed
    if (changed.size > 0) {
      setChangedTickers(changed);
      // Clear highlight after 2 seconds (subtle, not rushed)
      const timeout = setTimeout(() => setChangedTickers(new Set()), 2000);
      return () => clearTimeout(timeout);
    }
  }, [positions]);

  const formatCurrency = (cents) => {
    const dollars = cents / 100;
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(dollars);
  };

  const formatPnLCurrency = (cents) => {
    const dollars = cents / 100;
    const prefix = cents >= 0 ? '+' : '';
    return prefix + new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(dollars);
  };

  // Don't render if no positions
  if (!positions || positions.length === 0) {
    return null;
  }

  return (
    <div className="bg-gray-900/50 backdrop-blur-sm rounded-xl border border-gray-800 p-4 mb-4">
      <div
        className="flex items-center justify-between cursor-pointer"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center space-x-2">
          <Briefcase className="w-4 h-4 text-purple-400" />
          <h3 className="text-sm font-bold text-gray-300 uppercase tracking-wider">Open Positions</h3>
          <span className="text-xs text-gray-500">({positions.length})</span>
        </div>
        {isExpanded ? (
          <ChevronDown className="w-4 h-4 text-gray-500" />
        ) : (
          <ChevronRight className="w-4 h-4 text-gray-500" />
        )}
      </div>

      {isExpanded && (
        <div className="mt-4 bg-gray-800/30 rounded-lg border border-gray-700/50 overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-gray-900/50 border-b border-gray-700/50">
                <th className="px-3 py-2 text-left text-xs text-gray-500 uppercase font-medium">Market</th>
                <th className="px-3 py-2 text-center text-xs text-gray-500 uppercase font-medium">Side</th>
                <th className="px-3 py-2 text-right text-xs text-gray-500 uppercase font-medium">Qty</th>
                <th className="px-3 py-2 text-right text-xs text-gray-500 uppercase font-medium">Cost</th>
                <th className="px-3 py-2 text-right text-xs text-gray-500 uppercase font-medium">Value</th>
                <th className="px-3 py-2 text-right text-xs text-gray-500 uppercase font-medium">P&L</th>
              </tr>
            </thead>
            <tbody>
              {positions.map((pos, index) => {
                const pnlPercent = pos.total_traded > 0
                  ? ((pos.unrealized_pnl / pos.total_traded) * 100).toFixed(1)
                  : 0;

                const isRecentlyChanged = changedTickers.has(pos.ticker);

                return (
                  <tr
                    key={pos.ticker || index}
                    className={`border-b border-gray-700/30 hover:bg-gray-800/50 transition-all duration-500
                      ${isRecentlyChanged
                        ? 'border-l-2 border-l-emerald-400/70 bg-emerald-900/10'
                        : 'border-l-2 border-l-transparent'
                      }`}
                  >
                    <td className="px-3 py-2 font-mono text-gray-300 text-xs">
                      <div className="flex items-center">
                        {isRecentlyChanged && (
                          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 mr-2 animate-pulse" />
                        )}
                        {pos.ticker}
                      </div>
                    </td>
                    <td className="px-3 py-2 text-center">
                      <span className={`px-2 py-0.5 rounded text-xs font-bold uppercase ${
                        pos.side === 'yes'
                          ? 'bg-green-900/30 text-green-400 border border-green-700/50'
                          : 'bg-red-900/30 text-red-400 border border-red-700/50'
                      }`}>
                        {pos.side}
                      </span>
                    </td>
                    <td className="px-3 py-2 text-right font-mono text-gray-300">
                      {Math.abs(pos.position)}
                    </td>
                    <td className="px-3 py-2 text-right font-mono text-gray-400">
                      {formatCurrency(pos.total_traded)}
                    </td>
                    <td className="px-3 py-2 text-right font-mono text-gray-300">
                      {formatCurrency(pos.market_exposure)}
                    </td>
                    <td className={`px-3 py-2 text-right font-mono font-bold ${
                      pos.unrealized_pnl >= 0 ? 'text-green-400' : 'text-red-400'
                    }`}>
                      {formatPnLCurrency(pos.unrealized_pnl)}
                      <span className="text-xs ml-1 opacity-70">({pnlPercent}%)</span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

const V3TraderConsole = () => {
  const [messages, setMessages] = useState([]);
  // Start with all messages expanded by default for visibility
  const [expandedMessages, setExpandedMessages] = useState(new Set());
  const [wsStatus, setWsStatus] = useState('disconnected');
  const [currentState, setCurrentState] = useState('UNKNOWN');
  const [tradingState, setTradingState] = useState(null);
  const [lastUpdateTime, setLastUpdateTime] = useState(null);
  const [whaleQueue, setWhaleQueue] = useState({
    queue: [],
    stats: { trades_seen: 0, trades_discarded: 0, discard_rate_percent: 0 },
    followed_whale_ids: [],
    followed_whales: [],
    decision_history: [],
    decision_stats: {}
  });
  // Track which whale is currently being processed for animation
  const [processingWhaleId, setProcessingWhaleId] = useState(null);
  const [metrics, setMetrics] = useState({
    markets_connected: 0,
    snapshots_received: 0,
    deltas_received: 0,
    uptime: 0,
    health: 'unknown',
    ping_health: 'unknown',
    last_ping_age: null,
    api_connected: false,
    api_url: null
  });
  const [copied, setCopied] = useState(false);
  // P4: Disabled auto-scroll - was causing page to jump on every update
  const [autoScroll, setAutoScroll] = useState(false);
  const wsRef = useRef(null);
  const messagesEndRef = useRef(null);
  const messagesContainerRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);
  const lastMessageRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  // P4: Handle scroll events to detect if user is near bottom
  const handleScroll = useCallback(() => {
    if (!messagesContainerRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = messagesContainerRef.current;
    // User is "near bottom" if within 100px of the bottom
    const isNearBottom = scrollHeight - scrollTop - clientHeight < 100;
    setAutoScroll(isNearBottom);
  }, []);

  // P4: Only auto-scroll if user is near bottom
  useEffect(() => {
    if (autoScroll) {
      scrollToBottom();
    }
  }, [messages, autoScroll]);

  const addMessage = useCallback((type, content, metadata = {}) => {
    // Skip messages with UNKNOWN or undefined state
    if (metadata.state === 'UNKNOWN' || metadata.state === 'unknown' || metadata.state === 'undefined') {
      console.log('Skipping message with UNKNOWN state:', content, metadata);
      return;
    }
    
    // Deduplicate rapid repeated messages
    const messageKey = `${type}-${content}-${metadata.state || ''}`;
    const now = Date.now();
    
    if (lastMessageRef.current) {
      const { key: lastKey, time: lastTime, content: lastContent } = lastMessageRef.current;
      // If same message within 1 second, skip it
      // Also skip if same content with different states (e.g., UNKNOWN vs ready)
      if ((lastKey === messageKey && (now - lastTime) < 1000) ||
          (lastContent === content && (now - lastTime) < 1000)) {
        return;
      }
    }
    
    lastMessageRef.current = { key: messageKey, time: now, content: content };
    
    const timestamp = new Date().toLocaleTimeString('en-US', { 
      hour12: false, 
      hour: '2-digit', 
      minute: '2-digit', 
      second: '2-digit' 
    });
    
    // Parse state transition info
    let fromState = null;
    let toState = null;
    let isTransition = false;
    
    // Detect state transitions from the message content
    if (metadata.to_state || content.includes('→')) {
      isTransition = true;
      if (metadata.from_state && metadata.to_state) {
        fromState = metadata.from_state;
        toState = metadata.to_state;
      } else if (content.includes('→')) {
        // Parse from the message content
        const match = content.match(/(\w+)\s*→\s*(\w+)/);
        if (match) {
          fromState = match[1];
          toState = match[2];
        }
      }
    }
    
    // Extract status from the message (SUCCESS, FAILED, etc.)
    let status = null;
    if (content.includes('SUCCESS')) status = 'SUCCESS';
    else if (content.includes('FAILED')) status = 'FAILED';
    else if (content.includes('ERROR')) status = 'ERROR';
    else if (content.includes('READY')) status = 'READY';
    else if (content.includes('INITIALIZING')) status = 'INITIALIZING';
    else if (content.includes('CONNECTING')) status = 'CONNECTING';
    else if (content.includes('CALIBRATING')) status = 'CALIBRATING';
    
    // Clean up the content for display
    let cleanContent = content;
    if (isTransition && fromState && toState) {
      // For state transitions, remove all the transition-related text
      // We'll display the transition as nice badges instead
      
      // Remove various patterns that might appear
      cleanContent = content
        .replace(new RegExp(`${fromState}\\s*→\\s*${toState}:?\\s*`, 'gi'), '')
        .replace(/→\s*State:\s*/gi, '')
        .replace(/State:\s*/gi, '')
        .replace(/→\s*\w+/gi, '')
        .trim();
      
      // If what's left is just a state name, remove it
      if (cleanContent.toLowerCase() === toState.toLowerCase() || 
          cleanContent.toLowerCase() === fromState.toLowerCase() ||
          cleanContent === 'State' ||
          cleanContent === '→') {
        cleanContent = '';
      }
    }
    
    // Format metadata for display
    const formattedMetadata = {};
    if (metadata.metadata && typeof metadata.metadata === 'object') {
      for (const [key, value] of Object.entries(metadata.metadata)) {
        if (key === 'markets' && Array.isArray(value)) {
          formattedMetadata[key] = value.join(', ');
        } else if (typeof value === 'object') {
          formattedMetadata[key] = JSON.stringify(value, null, 2);
        } else {
          formattedMetadata[key] = value;
        }
      }
    }
    
    // Add any additional metadata fields
    if (metadata.context && metadata.context !== 'State transition') {
      formattedMetadata.context = metadata.context;
    }
    
    const newMessage = {
      id: Date.now() + Math.random(),
      type,
      content: cleanContent,
      originalContent: content,
      timestamp,
      metadata: {
        ...metadata,
        formattedMetadata: Object.keys(formattedMetadata).length > 0 ? formattedMetadata : null,
        isTransition,
        fromState,
        toState,
        status,
        state: metadata.state || metadata.to_state || toState
      }
    };
    
    setMessages(prev => [...prev.slice(-100), newMessage]);
    
    // Auto-expand messages with metadata
    if (Object.keys(formattedMetadata).length > 0) {
      setExpandedMessages(prev => {
        const newExpanded = new Set(prev);
        newExpanded.add(newMessage.id);
        return newExpanded;
      });
    }
  }, []);

  const connectWebSocket = useCallback(() => {
    // Prevent duplicate connections
    if (wsRef.current?.readyState === WebSocket.OPEN || 
        wsRef.current?.readyState === WebSocket.CONNECTING) {
      return;
    }

    try {
      // V3 trader always runs on port 8005
      const backendPort = import.meta.env.VITE_V3_BACKEND_PORT || import.meta.env.VITE_BACKEND_PORT || '8005';
      const ws = new WebSocket(`ws://localhost:${backendPort}/v3/ws`);
      
      ws.onopen = () => {
        setWsStatus('connected');
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          // Debug: Log ALL messages to see what's coming through
          console.log('WebSocket message received:', { type: data.type, hasMetrics: !!data.data?.metrics });
          
          switch(data.type) {
            case 'system_activity':
              // Handle unified system activity messages
              if (data.data) {
                const { activity_type, message, metadata, timestamp } = data.data;
                
                // Handle state transitions specially
                if (activity_type === 'state_transition' && metadata) {
                  if (metadata.to_state) {
                    setCurrentState(metadata.to_state);
                  }
                  
                  // Update API connection status based on state
                  if (metadata.to_state === 'trading_client_connect' && metadata.api_url) {
                    setMetrics(prev => ({
                      ...prev,
                      api_url: metadata.api_url,
                      api_connected: true
                    }));
                  }
                }
                
                // Determine message type based on severity (if present) or activity type
                let messageType = 'activity';
                if (metadata?.severity) {
                  // Use severity from backend if available
                  messageType = metadata.severity; // 'info', 'warning', 'error'
                } else if (activity_type === 'sync' && currentState === 'ERROR') {
                  // Override for sync messages in ERROR state - they're not errors!
                  messageType = 'info';
                }
                
                // Add message to console
                addMessage(messageType, message, {
                  activity_type,
                  timestamp,
                  metadata,
                  state: metadata?.to_state || currentState
                });
              }
              break;
              
            case 'trading_state':
              // Update trading state from WebSocket
              if (data.data) {
                // Update the last update time whenever we receive a trading_state message
                setLastUpdateTime(Math.floor(Date.now() / 1000));

                setTradingState({
                  has_state: true,
                  version: data.data.version,
                  balance: data.data.balance,
                  portfolio_value: data.data.portfolio_value,
                  position_count: data.data.position_count,
                  order_count: data.data.order_count,
                  positions: data.data.positions,
                  open_orders: data.data.open_orders,
                  sync_timestamp: data.data.sync_timestamp,
                  changes: data.data.changes,
                  order_group: data.data.order_group,  // Include order group data
                  // Session P&L and position details
                  pnl: data.data.pnl,
                  positions_details: data.data.positions_details || []
                });
              }
              break;
              
            case 'connection':
              // Initial connection acknowledgment - client registered with server
              // data.data.client_id contains the unique client identifier
              break;
              
            case 'history_replay':
              // Process batched historical transitions
              if (data.data.transitions) {
                data.data.transitions.forEach(transition => {
                  const fromState = transition.from_state || 'unknown';
                  const toState = transition.to_state || transition.state;
                  setCurrentState(toState);
                  addMessage('state', transition.message, {
                    state: toState,
                    from_state: fromState,
                    to_state: toState,
                    context: transition.context,
                    timestamp: transition.timestamp,
                    metadata: transition.metadata,
                    is_history: true
                  });
                });
              }
              break;
              
            case 'state_transition':
              const fromState = data.data.from_state || currentState;
              const toState = data.data.to_state || data.data.state;
              setCurrentState(toState);
              
              // Update API connection status based on state
              if (toState === 'trading_client_connect' && data.data.metadata?.api_url) {
                // Store API URL when connecting and set connected status
                setMetrics(prev => ({
                  ...prev,
                  api_url: data.data.metadata.api_url,
                  api_connected: true
                }));
              }
              
              // Update API connection status based on state
              if (toState === 'ready' || toState === 'calibrating' || toState === 'acting' || toState === 'trading_client_connect') {
                setMetrics(prev => ({ ...prev, api_connected: true }));
              } else if (toState === 'error' || toState === 'idle') {
                setMetrics(prev => ({ ...prev, api_connected: false }));
              }
              
              // Only add message if it's not just a current state update
              if (!data.data.is_current || data.data.message !== `Current state: ${toState}`) {
                addMessage('state', data.data.message, {
                  state: toState,
                  from_state: fromState,
                  to_state: toState,
                  context: data.data.context,
                  timestamp: data.data.timestamp,
                  metadata: data.data.metadata
                });
              }
              break;
              
            case 'trader_status':
              // Update metrics from single source of truth
              if (data.data.metrics) {
                // Debug: Log the full metrics object
                console.log('Full metrics object:', data.data.metrics);
                console.log('Received trader_status metrics:', {
                  ping_health: data.data.metrics.ping_health,
                  last_ping_age: data.data.metrics.last_ping_age
                });
                
                // Extract all metrics including Kalshi API ping health
                setMetrics(prev => ({
                  markets_connected: data.data.metrics.markets_connected || 0,
                  snapshots_received: data.data.metrics.snapshots_received || 0,
                  deltas_received: data.data.metrics.deltas_received || 0,
                  uptime: data.data.metrics.uptime || 0,
                  health: data.data.metrics.health || 'unknown',
                  // Use ping health from Kalshi API connection
                  ping_health: data.data.metrics.ping_health || 'unknown',
                  last_ping_age: data.data.metrics.last_ping_age || null,
                  // API connection info from backend
                  api_connected: data.data.metrics.api_connected || prev.api_connected,
                  api_url: data.data.metrics.api_url || prev.api_url,
                  ws_url: data.data.metrics.ws_url || prev.ws_url
                }));
              }
              if (data.data.state) {
                setCurrentState(data.data.state);
              }
              // Also update api_connected from metrics
              if (data.data.metrics && typeof data.data.metrics.api_connected === 'boolean') {
                setMetrics(prev => ({
                  ...prev,
                  api_connected: data.data.metrics.api_connected
                }));
              }
              break;
              
            case 'whale_queue':
              // Update whale queue state
              console.log('[V3TraderConsole] Received whale_queue message:', data.data);
              if (data.data) {
                console.log('[V3TraderConsole] Setting whale queue - queue length:', data.data.queue?.length || 0);
                console.log('[V3TraderConsole] Followed whale IDs:', data.data.followed_whale_ids);
                console.log('[V3TraderConsole] Followed whales:', data.data.followed_whales);
                console.log('[V3TraderConsole] Decision history:', data.data.decision_history);
                console.log('[V3TraderConsole] Decision stats:', data.data.decision_stats);
                setWhaleQueue({
                  queue: data.data.queue || [],
                  stats: data.data.stats || { trades_seen: 0, trades_discarded: 0, discard_rate_percent: 0 },
                  version: data.data.version,
                  // Include followed whale IDs for status indicator
                  followed_whale_ids: data.data.followed_whale_ids || [],
                  // Full followed whales data for Followed Trades section
                  followed_whales: data.data.followed_whales || [],
                  // Decision audit trail
                  decision_history: data.data.decision_history || [],
                  decision_stats: data.data.decision_stats || {}
                });
              }
              break;

            case 'whale_processing':
              // Handle whale processing animation events
              if (data.data) {
                const { whale_id, status } = data.data;
                if (status === 'processing') {
                  // Show animation on this whale row
                  setProcessingWhaleId(whale_id);
                } else if (status === 'complete') {
                  // Clear animation after a brief delay so user sees the result
                  setTimeout(() => {
                    setProcessingWhaleId(prev => prev === whale_id ? null : prev);
                  }, 300);
                }
              }
              break;

            case 'ping':
              // Respond to ping if needed
              if (ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'pong', timestamp: Date.now() }));
              }
              break;

            // system_metrics case removed - all metrics now come from trader_status
            // This case is no longer needed as trader_status is the single source of truth

            default:
              // Ignore all other message types
              break;
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      ws.onerror = (error) => {
        addMessage('error', 'WebSocket error occurred', { error: error.message });
        setWsStatus('error');
      };

      ws.onclose = (event) => {
        setWsStatus('disconnected');
        
        // Only log disconnect message if it wasn't a normal close
        if (event.code !== 1000) {
          addMessage('warning', 'Disconnected from TRADER V3', { 
            icon: 'disconnect',
            code: event.code,
            reason: event.reason || 'Connection lost'
          });
        }
        
        wsRef.current = null;
        
        // Auto-reconnect after 3 seconds
        reconnectTimeoutRef.current = setTimeout(() => {
          connectWebSocket();
        }, 3000);
      };

      wsRef.current = ws;
    } catch (error) {
      addMessage('error', `Failed to connect: ${error.message}`);
      setWsStatus('error');
    }
  }, [addMessage, currentState]);

  useEffect(() => {
    connectWebSocket();
    
    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [connectWebSocket]);

  const toggleMessageExpansion = (messageId) => {
    setExpandedMessages(prev => {
      const newSet = new Set(prev);
      if (newSet.has(messageId)) {
        newSet.delete(messageId);
      } else {
        newSet.add(messageId);
      }
      return newSet;
    });
  };

  const getStateColor = (state) => {
    const colors = {
      'startup': 'text-gray-400 bg-gray-900/50',
      'initializing': 'text-yellow-400 bg-yellow-900/30',
      'orderbook_connect': 'text-blue-400 bg-blue-900/30',
      'trading_client_connect': 'text-purple-400 bg-purple-900/30',
      'kalshi_data_sync': 'text-cyan-400 bg-cyan-900/30',
      'ready': 'text-green-400 bg-green-900/30',
      'error': 'text-red-400 bg-red-900/30',
      'shutdown': 'text-gray-500 bg-gray-900/50'
    };
    const stateKey = state?.toLowerCase();
    return colors[stateKey] || 'text-gray-400 bg-gray-900/50';
  };

  const getStatusColor = (status) => {
    const colors = {
      'SUCCESS': 'text-green-400 bg-green-900/30',
      'FAILED': 'text-red-400 bg-red-900/30',
      'ERROR': 'text-red-400 bg-red-900/30',
      'READY': 'text-green-400 bg-green-900/30',
      'INITIALIZING': 'text-yellow-400 bg-yellow-900/30',
      'CONNECTING': 'text-blue-400 bg-blue-900/30',
      'CALIBRATING': 'text-purple-400 bg-purple-900/30'
    };
    return colors[status] || 'text-gray-400 bg-gray-900/50';
  };

  const getMessageIcon = (type, metadata) => {
    if (metadata?.isTransition) {
      return <ArrowRight className="w-4 h-4 text-purple-400" />;
    }
    
    // Handle activity messages based on activity_type
    if (type === 'activity' && metadata?.activity_type) {
      switch(metadata.activity_type) {
        case 'state_transition': return <ArrowRight className="w-4 h-4 text-purple-400" />;
        case 'sync': return <Activity className="w-4 h-4 text-blue-400" />;
        case 'health_check': return <Activity className="w-4 h-4 text-green-400" />;
        case 'operation': return <ChevronRight className="w-4 h-4 text-gray-400" />;
        default: return <ChevronRight className="w-4 h-4 text-gray-500" />;
      }
    }
    
    switch(type) {
      case 'state': return <Zap className="w-4 h-4 text-purple-400" />;
      case 'data': return <Database className="w-4 h-4 text-blue-400" />;
      case 'success': return <CheckCircle className="w-4 h-4 text-green-400" />;
      case 'warning': return <AlertCircle className="w-4 h-4 text-yellow-400" />;
      case 'error': return <XCircle className="w-4 h-4 text-red-400" />;
      case 'info': return <Info className="w-4 h-4 text-blue-400" />;
      default: return <ChevronRight className="w-4 h-4 text-gray-500" />;
    }
  };

  const getMessageColor = (type, metadata) => {
    // Handle activity messages based on activity_type
    if (type === 'activity' && metadata?.activity_type) {
      switch(metadata.activity_type) {
        case 'state_transition': return 'text-purple-200';
        case 'sync': return 'text-blue-200';
        case 'health_check': return 'text-green-200';
        case 'operation': return 'text-gray-200';
        default: return 'text-gray-200';
      }
    }
    
    switch(type) {
      case 'state': return 'text-purple-200';
      case 'data': return 'text-blue-200';
      case 'success': return 'text-green-200';
      case 'warning': return 'text-yellow-200';
      case 'error': return 'text-red-200';
      case 'info': return 'text-blue-200';
      case 'activity': return 'text-gray-200';
      default: return 'text-gray-200';
    }
  };

  const copyConsoleOutput = () => {
    const output = messages.map(msg => {
      const state = msg.metadata?.state ? `[${msg.metadata.state}] ` : '';
      const status = msg.metadata?.status ? `[${msg.metadata.status}] ` : '';
      const transition = msg.metadata?.isTransition 
        ? `[${msg.metadata.fromState} → ${msg.metadata.toState}] ` 
        : '';
      return `${msg.timestamp} ${transition || state}${status}${msg.content}`;
    }).join('\n');
    
    navigator.clipboard.writeText(output).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-950 via-gray-900 to-gray-950">
      {/* Header */}
      <div className="border-b border-gray-800 bg-black/30 backdrop-blur-sm sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <Activity className="w-6 h-6 text-purple-400" />
                <h1 className="text-xl font-semibold text-white">TRADER V3</h1>
                <span className="px-2 py-0.5 text-xs font-mono bg-purple-500/20 text-purple-300 rounded-full border border-purple-500/30">
                  CONSOLE
                </span>
              </div>
            </div>
            
            <div className="flex items-center space-x-6">
              {/* Connection Status */}
              <div className="flex items-center space-x-2">
                {wsStatus === 'connected' ? (
                  <>
                    <Wifi className="w-5 h-5 text-green-400" />
                    <span className="text-sm text-green-400 font-medium">Connected</span>
                  </>
                ) : (
                  <>
                    <WifiOff className="w-5 h-5 text-red-400" />
                    <span className="text-sm text-red-400 font-medium">Disconnected</span>
                  </>
                )}
              </div>
              
              {/* Current State Badge */}
              <div className={`px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider ${getStateColor(currentState)}`}>
                {currentState}
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-6 py-6">
        {/* Session Summary Panel - Clear P&L at the top */}
        <SessionSummaryPanel tradingState={tradingState} />

        {/* Trading Data Panel - Balance, Portfolio, Positions, Orders */}
        <div className="mb-4">
          <TradingData tradingState={tradingState} lastUpdateTime={lastUpdateTime} />
        </div>

        {/* Position List Panel - Detailed per-position P&L */}
        <PositionListPanel positions={tradingState?.positions_details} />

        {/* Whale Queue Panel - Full width below Trading Data */}
        <div className="mb-6">
          <WhaleQueuePanel whaleQueue={whaleQueue} processingWhaleId={processingWhaleId} />
          <FollowedTradesPanel followedWhales={whaleQueue.followed_whales} />
          <DecisionAuditPanel
            decisionHistory={whaleQueue.decision_history}
            decisionStats={whaleQueue.decision_stats}
          />
        </div>

        <div className="grid grid-cols-12 gap-6">
          {/* Metrics Panel */}
          <div className="col-span-3">
            <div className="bg-gray-900/50 backdrop-blur-sm rounded-xl border border-gray-800 p-6 space-y-5">
              <h3 className="text-sm font-bold text-gray-300 uppercase tracking-wider">System Metrics</h3>
              
              <div className="space-y-4">
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-500">Markets</span>
                    <span className="text-lg font-mono font-bold text-white">{metrics.markets_connected || 0}</span>
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-500">Snapshots</span>
                    <span className="text-lg font-mono font-bold text-blue-400">{metrics.snapshots_received || 0}</span>
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-500">Deltas</span>
                    <span className="text-lg font-mono font-bold text-purple-400">{metrics.deltas_received || 0}</span>
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-500">Uptime</span>
                    <span className="text-lg font-mono font-bold text-green-400">
                      {metrics.uptime ? `${Math.floor(metrics.uptime)}s` : '0s'}
                    </span>
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-500">Ping Health</span>
                    <div className={`px-3 py-1 rounded-full text-xs font-bold ${
                      metrics.ping_health === 'healthy' ? 'bg-green-900/30 text-green-400' : 
                      metrics.ping_health === 'degraded' ? 'bg-yellow-900/30 text-yellow-400' : 
                      metrics.ping_health === 'unhealthy' ? 'bg-red-900/30 text-red-400' : 
                      'bg-gray-900/30 text-gray-400'
                    }`}>
                      {metrics.ping_health?.toUpperCase() || 'UNKNOWN'}
                    </div>
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-500">Last Message</span>
                    <span className={`text-lg font-mono font-bold ${
                      metrics.last_ping_age === null ? 'text-gray-400' :
                      metrics.last_ping_age < 10 ? 'text-green-400' :
                      metrics.last_ping_age < 30 ? 'text-yellow-400' : 'text-red-400'
                    }`}>
                      {metrics.last_ping_age !== null ? `${Math.floor(metrics.last_ping_age)}s ago` : 'N/A'}
                    </span>
                  </div>
                </div>
                
                <div className="pt-4 border-t border-gray-700">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-500">Health Status</span>
                    <div className={`px-3 py-1 rounded-full text-xs font-bold ${
                      metrics.health === 'healthy' ? 'bg-green-900/30 text-green-400' : 
                      metrics.health === 'unhealthy' ? 'bg-red-900/30 text-red-400' : 
                      'bg-gray-900/30 text-gray-400'
                    }`}>
                      {metrics.health?.toUpperCase() || 'UNKNOWN'}
                    </div>
                  </div>
                </div>
              </div>
              
              {/* API Status */}
              <div className="pt-4 border-t border-gray-700">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs text-gray-500 uppercase tracking-wider">API Status</span>
                  <div className={`px-3 py-1 rounded-full text-xs font-bold ${
                    metrics.api_connected ? 'bg-green-900/30 text-green-400' : 'bg-red-900/30 text-red-400'
                  }`}>
                    {metrics.api_connected ? 'CONNECTED' : 'DISCONNECTED'}
                  </div>
                </div>
                {metrics.api_url && (
                  <div className="mt-2 space-y-2">
                    <div>
                      <span className="text-xs text-gray-500">API:</span>
                      <div className="text-xs text-gray-300 font-mono mt-1 truncate" title={metrics.api_url}>
                        {metrics.api_url.replace('https://', '').replace('/trade-api/v2', '')}
                      </div>
                    </div>
                    {metrics.ws_url && (
                      <div>
                        <span className="text-xs text-gray-500">WebSocket:</span>
                        <div className="text-xs text-gray-300 font-mono mt-1 truncate" title={metrics.ws_url}>
                          {metrics.ws_url.replace('wss://', '').replace('/trade-api/ws/v2', '')}
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
              
              {/* Activity Indicator */}
              <div className="pt-4 border-t border-gray-700">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs text-gray-500 uppercase tracking-wider">Activity</span>
                </div>
                <div className="flex space-x-1">
                  {[...Array(10)].map((_, i) => (
                    <div
                      key={i}
                      className={`flex-1 h-2 rounded-full transition-all duration-300 ${
                        i < (metrics.markets_connected || 0) 
                          ? 'bg-gradient-to-r from-purple-500 to-blue-500 animate-pulse' 
                          : 'bg-gray-800'
                      }`}
                    />
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Console */}
          <div className="col-span-9">
            <div className="bg-gray-900/50 backdrop-blur-sm rounded-xl border border-gray-800 overflow-hidden">
              {/* Console Header */}
              <div className="bg-black/50 px-6 py-3 border-b border-gray-800 flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <div className="flex space-x-1.5">
                    <div className="w-3 h-3 rounded-full bg-red-500/80 hover:bg-red-500 transition-colors cursor-pointer" />
                    <div className="w-3 h-3 rounded-full bg-yellow-500/80 hover:bg-yellow-500 transition-colors cursor-pointer" />
                    <div className="w-3 h-3 rounded-full bg-green-500/80 hover:bg-green-500 transition-colors cursor-pointer" />
                  </div>
                  <span className="text-xs text-gray-400 font-mono uppercase tracking-wider">System Console</span>
                </div>
                <div className="flex items-center space-x-4">
                  <button
                    onClick={copyConsoleOutput}
                    className="flex items-center space-x-2 px-3 py-1.5 text-xs text-gray-400 hover:text-gray-200 hover:bg-gray-800 rounded-lg transition-all"
                    title="Copy console output"
                  >
                    {copied ? (
                      <>
                        <Check className="w-3.5 h-3.5" />
                        <span className="font-medium">Copied!</span>
                      </>
                    ) : (
                      <>
                        <Copy className="w-3.5 h-3.5" />
                        <span className="font-medium">Copy</span>
                      </>
                    )}
                  </button>
                  <span className="text-xs text-gray-500 font-mono">
                    {messages.length} messages
                  </span>
                </div>
              </div>
              
              {/* Messages */}
              <div
                ref={messagesContainerRef}
                onScroll={handleScroll}
                className="h-[600px] overflow-y-auto p-4 font-mono text-sm bg-black/20"
              >
                {messages.length === 0 ? (
                  <div className="flex flex-col items-center justify-center h-full text-gray-600">
                    <Activity className="w-8 h-8 mb-3 animate-pulse" />
                    <div>Waiting for messages...</div>
                  </div>
                ) : (
                  <div className="space-y-2">
                    {messages.map((message) => {
                      const isExpanded = expandedMessages.has(message.id);
                      const hasMetadata = message.metadata?.formattedMetadata && 
                                         Object.keys(message.metadata.formattedMetadata).length > 0;
                      
                      return (
                        <div 
                          key={message.id} 
                          className="group hover:bg-gray-800/20 rounded-lg transition-all duration-200"
                        >
                          <div className="flex items-start space-x-3 p-2">
                            {/* Timestamp */}
                            <span className="text-gray-500 text-xs w-20 flex-shrink-0 font-mono">
                              [{message.timestamp}]
                            </span>
                            
                            {/* State Transition or State Badge */}
                            <div className="w-44 flex-shrink-0">
                              {message.metadata?.isTransition && message.metadata?.fromState && message.metadata?.toState ? (
                                <div className="flex items-center space-x-1 text-xs">
                                  <span className={`px-2 py-0.5 rounded font-bold ${getStateColor(message.metadata.fromState)}`}>
                                    {message.metadata.fromState}
                                  </span>
                                  <ArrowRight className="w-3 h-3 text-gray-400" />
                                  <span className={`px-2 py-0.5 rounded font-bold ${getStateColor(message.metadata.toState)}`}>
                                    {message.metadata.toState}
                                  </span>
                                </div>
                              ) : message.metadata?.state ? (
                                <span className={`px-2 py-0.5 rounded text-xs font-bold ${getStateColor(message.metadata.state)}`}>
                                  [{message.metadata.state}]
                                </span>
                              ) : null}
                            </div>
                            
                            {/* Status Badge */}
                            <div className="w-24 flex-shrink-0">
                              {message.metadata?.status && (
                                <span className={`px-2 py-0.5 rounded text-xs font-bold ${getStatusColor(message.metadata.status)}`}>
                                  {message.metadata.status}
                                </span>
                              )}
                            </div>
                            
                            {/* Icon */}
                            <div className="flex-shrink-0 mt-0.5">
                              {getMessageIcon(message.type, message.metadata)}
                            </div>
                            
                            {/* Message Content - only show if there's content */}
                            {message.content && (
                              <div className={`flex-1 ${getMessageColor(message.type, message.metadata)}`}>
                                <div className="leading-relaxed">{message.content}</div>
                              </div>
                            )}
                            {!message.content && (
                              <div className="flex-1"></div>
                            )}
                            
                            {/* Expand/Collapse Button */}
                            {hasMetadata && (
                              <button
                                onClick={() => toggleMessageExpansion(message.id)}
                                className="flex-shrink-0 p-1 text-gray-500 hover:text-gray-300 transition-colors"
                                title={isExpanded ? "Collapse metadata" : "Expand metadata"}
                              >
                                {isExpanded ? (
                                  <ChevronDown className="w-4 h-4" />
                                ) : (
                                  <ChevronRight className="w-4 h-4" />
                                )}
                              </button>
                            )}
                          </div>
                          
                          {/* Expandable Metadata Section */}
                          {hasMetadata && isExpanded && (
                            <div className="ml-24 mb-2 mr-4">
                              <div className="bg-gray-800/30 rounded-lg p-3 border border-gray-700/50">
                                <div className="text-xs text-gray-400 font-mono space-y-1">
                                  {Object.entries(message.metadata.formattedMetadata).map(([key, value]) => (
                                    <div key={key} className="flex">
                                      <span className="text-gray-500 mr-2">{key}:</span>
                                      <span className="text-gray-300 break-all whitespace-pre-wrap">{value}</span>
                                    </div>
                                  ))}
                                </div>
                              </div>
                            </div>
                          )}
                        </div>
                      );
                    })}
                    <div ref={messagesEndRef} />
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default V3TraderConsole;