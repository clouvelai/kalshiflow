import { useState, useEffect, useRef, useCallback } from 'react';

/**
 * Initial state for tracked markets
 */
const INITIAL_TRACKED_MARKETS = {
  markets: [],
  stats: {
    tracked: 0,
    capacity: 50,
    total: 0,
    by_category: {},
    by_status: { active: 0, determined: 0, settled: 0 },
    determined_today: 0,
    tracked_total: 0,
    rejected_capacity: 0,
    rejected_category: 0,
    version: 0
  },
  version: 0
};

/**
 * Initial state for lifecycle events feed
 */
const INITIAL_EVENTS = [];

/**
 * useLifecycleWebSocket - Hook for managing Lifecycle Discovery WebSocket connection
 *
 * Connects to the V3 backend WebSocket endpoint and handles lifecycle-specific
 * message types: tracked_markets, lifecycle_event, market_info_update
 *
 * Also handles RLM (Reverse Line Movement) state updates:
 * - rlm_states_snapshot: Initial snapshot of all RLM market states
 * - rlm_market_state: Real-time update for a single market's RLM state
 * - rlm_trade_arrived: Trade pulse for animation (green=YES, red=NO)
 *
 * @param {Object} options
 * @param {Function} options.onMessage - Callback for console/log messages
 * @returns {Object} - State and connection info
 */
export const useLifecycleWebSocket = ({ onMessage } = {}) => {
  const [wsStatus, setWsStatus] = useState('disconnected');
  const [trackedMarkets, setTrackedMarkets] = useState(INITIAL_TRACKED_MARKETS);
  const [recentEvents, setRecentEvents] = useState(INITIAL_EVENTS);
  const [lastUpdateTime, setLastUpdateTime] = useState(null);

  // RLM (Reverse Line Movement) state
  // rlmStates: Map of ticker -> RLM state { yes_trades, no_trades, yes_ratio, price_drop, etc. }
  const [rlmStates, setRlmStates] = useState({});
  // tradePulses: Map of ticker -> { side: 'yes'|'no', ts: timestamp } for pulse animation
  const [tradePulses, setTradePulses] = useState({});

  // Upcoming markets (unopened markets opening within 4 hours)
  const [upcomingMarkets, setUpcomingMarkets] = useState([]);

  // Trading state (balance and min_trader_cash for low cash indicator)
  const [tradingState, setTradingState] = useState({ balance: 0, min_trader_cash: 0 });

  // Event exposure data (correlated positions across related markets)
  const [eventExposure, setEventExposure] = useState(null);

  const wsRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);
  const lastPingRef = useRef(Date.now());
  const heartbeatIntervalRef = useRef(null);
  const initialSnapshotReceivedRef = useRef(false);

  // Maximum number of events to keep in the feed
  const MAX_EVENTS = 100;

  // Define handleMessage BEFORE connectWebSocket to avoid stale closure
  const handleMessage = useCallback((data, ws) => {
    // Update heartbeat on ANY message (not just pings) to prevent false disconnects
    lastPingRef.current = Date.now();

    switch (data.type) {
      case 'tracked_markets':
        // Full snapshot of tracked markets (sent on connect and on major changes)
        if (data.data) {
          const markets = data.data.markets || [];
          const stats = data.data.stats || INITIAL_TRACKED_MARKETS.stats;

          setTrackedMarkets({
            markets,
            stats,
            version: data.data.version || 0
          });
          setLastUpdateTime(Date.now());

          // Create synthetic "loaded" event for initial snapshot with markets
          if (!initialSnapshotReceivedRef.current && markets.length > 0) {
            initialSnapshotReceivedRef.current = true;

            // Create a startup event showing how many markets were loaded
            const loadedEvent = {
              id: `startup-${Date.now()}`,
              event_type: 'startup',
              market_ticker: '',
              action: 'loaded',
              reason: 'startup',
              metadata: {
                count: markets.length,
                by_category: stats.by_category || {}
              },
              timestamp: new Date().toLocaleTimeString()
            };

            setRecentEvents(prev => [loadedEvent, ...prev].slice(0, MAX_EVENTS));

            onMessage?.('info', `Loaded ${markets.length} tracked markets`, {
              count: markets.length
            });
          }
        }
        break;

      case 'lifecycle_event':
        // Real-time lifecycle event (market tracked, status change, etc.)
        if (data.data) {
          const event = {
            id: `${data.data.market_ticker}-${Date.now()}`,
            event_type: data.data.event_type,
            market_ticker: data.data.market_ticker,
            action: data.data.action,
            reason: data.data.reason,
            metadata: data.data.metadata || {},
            timestamp: data.data.timestamp || new Date().toLocaleTimeString()
          };

          setRecentEvents(prev => {
            // Deduplicate: skip if same ticker+event_type within last 5 seconds
            const isDuplicate = prev.some(e =>
              e.market_ticker === event.market_ticker &&
              e.event_type === event.event_type &&
              e.timestamp === event.timestamp
            );
            if (isDuplicate) return prev;

            const updated = [event, ...prev].slice(0, MAX_EVENTS);
            return updated;
          });

          // Notify parent component for potential toast
          onMessage?.('lifecycle', `${event.action}: ${event.market_ticker}`, {
            event_type: event.event_type,
            action: event.action,
            ticker: event.market_ticker
          });
        }
        break;

      case 'market_info_update':
        // Incremental update for a single market's price/volume
        if (data.data && data.data.ticker) {
          setTrackedMarkets(prev => {
            const marketIndex = prev.markets.findIndex(
              m => m.ticker === data.data.ticker
            );

            if (marketIndex === -1) {
              return prev; // Market not found, skip update
            }

            // Create new array with updated market
            const updatedMarkets = [...prev.markets];
            updatedMarkets[marketIndex] = {
              ...updatedMarkets[marketIndex],
              price: data.data.price,
              volume: data.data.volume,
              open_interest: data.data.open_interest,
              yes_bid: data.data.yes_bid,
              yes_ask: data.data.yes_ask
            };

            return {
              ...prev,
              markets: updatedMarkets
            };
          });
          setLastUpdateTime(Date.now());
        }
        break;

      case 'connection':
        // Initial connection acknowledgment
        break;

      case 'ping':
        // Server ping - respond with pong (heartbeat already tracked above)
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: 'pong', timestamp: Date.now() }));
        }
        break;

      // ========== RLM (Reverse Line Movement) Messages ==========

      case 'rlm_states_snapshot':
        // Initial snapshot of all RLM market states on connect
        if (data.data && data.data.markets) {
          const statesMap = {};
          data.data.markets.forEach(state => {
            statesMap[state.market_ticker] = state;
          });
          setRlmStates(statesMap);
        }
        break;

      case 'rlm_market_state':
        // Real-time update for a single market's RLM state
        if (data.data && data.data.market_ticker) {
          setRlmStates(prev => ({
            ...prev,
            [data.data.market_ticker]: data.data
          }));
        }
        break;

      case 'rlm_trade_arrived':
        // Trade pulse for animation - set pulse and auto-clear after 1.5s
        if (data.data && data.data.market_ticker) {
          const ticker = data.data.market_ticker;
          const side = data.data.side;
          const ts = Date.now();

          setTradePulses(prev => ({
            ...prev,
            [ticker]: { side, ts }
          }));

          // Auto-clear pulse after 1.5s
          setTimeout(() => {
            setTradePulses(prev => {
              // Only clear if this is still the same pulse (not overwritten by newer trade)
              if (prev[ticker] && prev[ticker].ts === ts) {
                const next = { ...prev };
                delete next[ticker];
                return next;
              }
              return prev;
            });
          }, 1500);
        }
        break;

      // ========== Upcoming Markets Messages ==========

      case 'upcoming_markets':
        // Upcoming markets schedule (markets opening within 4 hours)
        if (data.data && data.data.markets) {
          setUpcomingMarkets(data.data.markets);
          setLastUpdateTime(Date.now());
        }
        break;

      case 'trading_state':
        // Trading state update (balance, min_trader_cash for low cash indicator)
        if (data.data) {
          setTradingState({
            balance: data.data.balance || 0,
            min_trader_cash: data.data.min_trader_cash || 0,
            rlm_config: data.data.rlm_config || null
          });
          // Extract event exposure data for correlated position tracking
          if (data.data.event_exposure) {
            setEventExposure(data.data.event_exposure);
          }
        }
        break;

      case 'system_activity':
        // Handle trading activity events (RLM signals, order fills, etc.)
        if (data.data) {
          const activityData = data.data;
          const event = {
            id: `activity-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
            event_type: activityData.activity_type,  // 'rlm_signal', 'order_fill', etc.
            market_ticker: activityData.metadata?.market_ticker || activityData.metadata?.ticker || '',
            action: activityData.activity_type,
            reason: activityData.message,
            metadata: activityData.metadata || {},
            timestamp: activityData.timestamp || new Date().toLocaleTimeString()
          };

          setRecentEvents(prev => {
            // Deduplicate: skip if same activity_type+timestamp within recent events
            const isDuplicate = prev.some(e =>
              e.event_type === event.event_type &&
              e.timestamp === event.timestamp
            );
            if (isDuplicate) return prev;

            const updated = [event, ...prev].slice(0, MAX_EVENTS);
            return updated;
          });

          onMessage?.('activity', activityData.message, {
            activity_type: activityData.activity_type,
            ...activityData.metadata
          });
        }
        break;

      case 'activity_feed_history':
        // Replay historical activity events on connect
        // This restores the Activity Feed when switching between Trader/Discovery views
        if (data.data?.events) {
          const historicalEvents = data.data.events.map((item, idx) => {
            const eventData = item.data;
            const eventType = item.type;

            if (eventType === 'system_activity') {
              // Format system_activity (RLM signals, order fills, etc.)
              return {
                id: `history-activity-${idx}-${item.timestamp}`,
                event_type: eventData.activity_type,
                market_ticker: eventData.metadata?.market_ticker || eventData.metadata?.ticker || '',
                action: eventData.activity_type,
                reason: eventData.message,
                metadata: eventData.metadata || {},
                timestamp: eventData.timestamp || '--:--:--'
              };
            } else {
              // Format lifecycle_event
              return {
                id: eventData.id || `history-lifecycle-${eventData.market_ticker}-${idx}`,
                event_type: eventData.event_type,
                market_ticker: eventData.market_ticker,
                action: eventData.action,
                reason: eventData.reason,
                metadata: eventData.metadata || {},
                timestamp: eventData.timestamp || '--:--:--'
              };
            }
          });

          setRecentEvents(prev => {
            // Merge historical events, avoiding duplicates
            const existingIds = new Set(prev.map(e => e.id));
            const newEvents = historicalEvents.filter(e => !existingIds.has(e.id));
            // Historical events should appear in chronological order (oldest first, newest last)
            // But we want newest at top, so reverse them and prepend
            return [...newEvents.reverse(), ...prev].slice(0, MAX_EVENTS);
          });

          onMessage?.('info', `Restored ${data.data.count} activity events`, {});
        }
        break;

      default:
        // Ignore other message types (they're for other V3 features)
        break;
    }
  }, [onMessage]);

  // connectWebSocket is defined AFTER handleMessage to properly reference it
  const connectWebSocket = useCallback(() => {
    // Clear any pending reconnect timeout
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    if (wsRef.current?.readyState === WebSocket.OPEN ||
        wsRef.current?.readyState === WebSocket.CONNECTING) {
      return;
    }

    try {
      const backendPort = import.meta.env.VITE_V3_BACKEND_PORT || import.meta.env.VITE_BACKEND_PORT || '8005';
      const ws = new WebSocket(`ws://localhost:${backendPort}/v3/ws`);

      ws.onopen = () => {
        setWsStatus('connected');
        onMessage?.('info', 'Connected to Lifecycle Discovery backend');
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          handleMessage(data, ws);
        } catch (error) {
          console.error('[useLifecycleWebSocket] Error parsing message:', error);
        }
      };

      ws.onerror = (error) => {
        onMessage?.('error', 'WebSocket error occurred', { error: error.message });
        setWsStatus('error');
      };

      ws.onclose = (event) => {
        setWsStatus('disconnected');

        if (event.code !== 1000) {
          onMessage?.('warning', 'Disconnected from Lifecycle Discovery', {
            code: event.code,
            reason: event.reason || 'Connection lost'
          });
        }

        wsRef.current = null;

        // Reconnect after 3 seconds
        reconnectTimeoutRef.current = setTimeout(() => {
          connectWebSocket();
        }, 3000);
      };

      wsRef.current = ws;
    } catch (error) {
      onMessage?.('error', `Failed to connect: ${error.message}`);
      setWsStatus('error');
    }
  }, [onMessage, handleMessage]);

  // Connect on mount
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

  // Heartbeat monitoring - detect zombie connections
  useEffect(() => {
    const checkHeartbeat = () => {
      const timeSinceLastPing = Date.now() - lastPingRef.current;
      // Server sends pings every 30s, so 60s timeout gives buffer
      if (timeSinceLastPing > 60000 && wsStatus === 'connected') {
        console.warn('[useLifecycleWebSocket] No ping in 60s, forcing reconnect');
        wsRef.current?.close();
      }
    };

    heartbeatIntervalRef.current = setInterval(checkHeartbeat, 15000);

    return () => {
      if (heartbeatIntervalRef.current) {
        clearInterval(heartbeatIntervalRef.current);
      }
    };
  }, [wsStatus]);

  // Clear events utility
  const clearEvents = useCallback(() => {
    setRecentEvents([]);
  }, []);

  return {
    wsStatus,
    trackedMarkets,
    recentEvents,
    lastUpdateTime,
    clearEvents,
    // Derived convenience stats
    stats: trackedMarkets.stats,
    markets: trackedMarkets.markets,
    isConnected: wsStatus === 'connected',
    isAtCapacity: trackedMarkets.stats.tracked >= trackedMarkets.stats.capacity,
    // RLM (Reverse Line Movement) state
    rlmStates,       // Map of ticker -> RLM state
    tradePulses,     // Map of ticker -> { side, ts } for pulse animation
    // Upcoming markets (opening within 4 hours)
    upcomingMarkets, // List of upcoming market objects
    // Trading state (balance for low cash indicator)
    tradingState,    // { balance, min_trader_cash }
    // Event exposure data (correlated positions across related markets)
    eventExposure    // { event_groups: { event_ticker -> EventGroup }, stats }
  };
};

export default useLifecycleWebSocket;
