import React, { useState, useEffect, useRef, memo } from 'react';
import { Fish, CheckCircle, Activity, Clock } from 'lucide-react';
import { formatCurrency, formatAge } from '../../../utils/v3-trader';

/**
 * WhaleRow - Memoized row component for whale queue table
 */
const WhaleRow = memo(({ whale, index, isFollowed, isProcessing, isFading, isRemoving, followedWhaleIds }) => {
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
      <td className="px-3 py-2 text-center">{statusDisplay}</td>
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
      <td className="px-3 py-2 text-right font-mono font-bold text-cyan-400">
        {formatCurrency(whale.payout_dollars - whale.cost_dollars)}
      </td>
      <td className="px-3 py-2 text-right font-mono text-gray-500 text-xs">{formatAge(whale.age_seconds)}</td>
    </tr>
  );
});

WhaleRow.displayName = 'WhaleRow';

/**
 * WhaleQueuePanel - Displays detected whale bets with smooth animations
 */
const WhaleQueuePanel = ({ whaleQueue, processingWhaleId }) => {
  const [currentTime, setCurrentTime] = useState(Date.now());
  const [displayQueue, setDisplayQueue] = useState([]);
  const [fadingWhales, setFadingWhales] = useState(new Set());
  const fadingWhalesRef = useRef(new Set());
  const whaleFirstSeenRef = useRef(new Map());
  const pendingRemovalRef = useRef(new Set());
  const lastQueueVersionRef = useRef(null);

  const MIN_DISPLAY_TIME = 3000;
  const FADE_OUT_DURATION = 500;

  // Keep fadingWhalesRef in sync with fadingWhales state
  useEffect(() => {
    fadingWhalesRef.current = fadingWhales;
  }, [fadingWhales]);

  // Merge incoming queue with display queue, ensuring minimum display time
  useEffect(() => {
    const incomingQueue = whaleQueue?.queue || [];
    const queueVersion = whaleQueue?.version;

    if (queueVersion !== undefined && queueVersion === lastQueueVersionRef.current) {
      return;
    }
    lastQueueVersionRef.current = queueVersion;

    const incomingIds = new Set(incomingQueue.map(w => w.whale_id));
    const now = Date.now();

    incomingQueue.forEach(whale => {
      if (!whaleFirstSeenRef.current.has(whale.whale_id)) {
        whaleFirstSeenRef.current.set(whale.whale_id, now);
      }
    });

    setDisplayQueue(prevQueue => {
      const mergedMap = new Map();

      incomingQueue.forEach(whale => {
        mergedMap.set(whale.whale_id, { ...whale, isRemoving: false });
        pendingRemovalRef.current.delete(whale.whale_id);
      });

      prevQueue.forEach(whale => {
        if (!incomingIds.has(whale.whale_id)) {
          const firstSeen = whaleFirstSeenRef.current.get(whale.whale_id);
          const timeVisible = now - (firstSeen || now);

          if (timeVisible < MIN_DISPLAY_TIME &&
              !fadingWhalesRef.current.has(whale.whale_id) &&
              !pendingRemovalRef.current.has(whale.whale_id)) {
            mergedMap.set(whale.whale_id, { ...whale, isRemoving: true });
            pendingRemovalRef.current.add(whale.whale_id);

            const remainingTime = Math.max(0, MIN_DISPLAY_TIME - timeVisible);
            setTimeout(() => {
              setFadingWhales(prev => new Set([...prev, whale.whale_id]));
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
          } else if (fadingWhalesRef.current.has(whale.whale_id) || pendingRemovalRef.current.has(whale.whale_id)) {
            mergedMap.set(whale.whale_id, { ...whale, isRemoving: true });
          }
        }
      });

      return Array.from(mergedMap.values());
    });
  }, [whaleQueue?.queue]);

  // Update current time every second for age display
  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentTime(Date.now());
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  const queue = displayQueue;
  const stats = whaleQueue?.stats || { trades_seen: 0, trades_discarded: 0, discard_rate_percent: 0 };
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
              {queue.map((whale, index) => (
                <WhaleRow
                  key={whale.whale_id || `${whale.market_ticker}-${whale.price_cents}-${index}`}
                  whale={whale}
                  index={index}
                  isFollowed={followedWhaleIds.has(whale.whale_id)}
                  isProcessing={processingWhaleId === whale.whale_id}
                  isFading={fadingWhales.has(whale.whale_id)}
                  isRemoving={whale.isRemoving}
                  followedWhaleIds={followedWhaleIds}
                />
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

export default memo(WhaleQueuePanel);
