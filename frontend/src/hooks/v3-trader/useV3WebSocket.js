import { useState, useEffect, useRef, useCallback } from 'react';

/**
 * Initial state for metrics
 */
const INITIAL_METRICS = {
  markets_connected: 0,
  tracked_markets: 0,
  subscribed_markets: 0,
  snapshots_received: 0,
  deltas_received: 0,
  uptime: 0,
  health: 'unknown',
  ping_health: 'unknown',
  last_ping_age: null,
  api_connected: false,
  api_url: null,
  ws_url: null,
  signal_aggregator: null
};

/**
 * Initial state for trade processing
 */
const INITIAL_TRADE_PROCESSING = {
  recent_trades: [],
  stats: { trades_seen: 0, trades_filtered: 0, trades_tracked: 0, filter_rate_percent: 0 },
  decisions: { detected: 0, executed: 0, rate_limited: 0, skipped: 0, reentries: 0 },
  decision_history: [],
  last_updated: null,
  timestamp: null
};

/**
 * Initial state for strategy status (Trading Strategies Panel)
 */
const INITIAL_STRATEGY_STATUS = {
  coordinator: {
    running: false,
    uptime_seconds: 0,
    strategies_running: 0,
    rate_limiter: {
      tokens_available: 0,
      capacity: 0,
      refill_rate_per_minute: 0,
      utilization_percent: 0
    }
  },
  strategies: {},
  recent_decisions: [],
  last_updated: null,
  timestamp: null
};

/**
 * useV3WebSocket - Hook for managing V3 Trader WebSocket connection
 */
export const useV3WebSocket = ({ onMessage }) => {
  const [wsStatus, setWsStatus] = useState('disconnected');
  const [currentState, setCurrentState] = useState('UNKNOWN');
  const [tradingState, setTradingState] = useState(null);
  const [lastUpdateTime, setLastUpdateTime] = useState(null);
  const [tradeProcessing, setTradeProcessing] = useState(INITIAL_TRADE_PROCESSING);
  const [strategyStatus, setStrategyStatus] = useState(INITIAL_STRATEGY_STATUS);
  const [settlements, setSettlements] = useState([]);
  const [newSettlement, setNewSettlement] = useState(null);
  const [newOrderFill, setNewOrderFill] = useState(null);
  const [newTtlCancellation, setNewTtlCancellation] = useState(null);
  const [metrics, setMetrics] = useState(INITIAL_METRICS);

  const wsRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);
  const currentStateRef = useRef(currentState);
  const lastPingRef = useRef(Date.now());
  const heartbeatIntervalRef = useRef(null);

  // Toast timeout refs for proper cleanup on reconnect/unmount
  const orderFillTimeoutRef = useRef(null);
  const ttlCancellationTimeoutRef = useRef(null);
  const settlementTimeoutRef = useRef(null);

  // Keep currentStateRef in sync
  useEffect(() => {
    currentStateRef.current = currentState;
  }, [currentState]);

  const connectWebSocket = useCallback(() => {
    // Clear any pending reconnect timeout first to prevent timeout accumulation
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
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          handleMessage(data, ws);
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      ws.onerror = (error) => {
        onMessage?.('error', 'WebSocket error occurred', { error: error.message });
        setWsStatus('error');
      };

      ws.onclose = (event) => {
        setWsStatus('disconnected');

        if (event.code !== 1000) {
          onMessage?.('warning', 'Disconnected from TRADER V3', {
            icon: 'disconnect',
            code: event.code,
            reason: event.reason || 'Connection lost'
          });
        }

        wsRef.current = null;

        reconnectTimeoutRef.current = setTimeout(() => {
          connectWebSocket();
        }, 3000);
      };

      wsRef.current = ws;
    } catch (error) {
      onMessage?.('error', `Failed to connect: ${error.message}`);
      setWsStatus('error');
    }
  }, [onMessage]);

  const handleMessage = useCallback((data, ws) => {
    switch (data.type) {
      case 'system_activity':
        if (data.data) {
          const { activity_type, message, metadata, timestamp } = data.data;

          if (activity_type === 'state_transition' && metadata) {
            if (metadata.to_state) {
              setCurrentState(metadata.to_state);
            }

            if (metadata.to_state === 'trading_client_connect' && metadata.api_url) {
              setMetrics(prev => ({
                ...prev,
                api_url: metadata.api_url,
                api_connected: true
              }));
            }
          }

          // Handle order fill toast notification
          if (activity_type === 'order_fill' && metadata) {
            setNewOrderFill({
              ticker: metadata.ticker,
              action: metadata.action,
              side: metadata.side,
              count: metadata.count,
              price_cents: metadata.price_cents,
              total_cents: metadata.total_cents
            });
            // Auto-dismiss after 5 seconds (with proper cleanup)
            if (orderFillTimeoutRef.current) {
              clearTimeout(orderFillTimeoutRef.current);
            }
            orderFillTimeoutRef.current = setTimeout(() => setNewOrderFill(null), 5000);
          }

          // Handle TTL cancellation toast notification
          if (activity_type === 'orders_cancelled_ttl' && metadata) {
            setNewTtlCancellation({
              count: metadata.count,
              tickers: metadata.tickers || [],
              ttl_seconds: metadata.ttl_seconds
            });
            // Auto-dismiss after 5 seconds (with proper cleanup)
            if (ttlCancellationTimeoutRef.current) {
              clearTimeout(ttlCancellationTimeoutRef.current);
            }
            ttlCancellationTimeoutRef.current = setTimeout(() => setNewTtlCancellation(null), 5000);
          }

          let messageType = 'activity';
          if (metadata?.severity) {
            messageType = metadata.severity;
          } else if (activity_type === 'sync' && currentStateRef.current === 'ERROR') {
            messageType = 'info';
          }

          onMessage?.(messageType, message, {
            activity_type,
            timestamp,
            metadata,
            state: metadata?.to_state || currentStateRef.current
          });
        }
        break;

      case 'trading_state':
        if (data.data) {
          setLastUpdateTime(Math.floor(Date.now() / 1000));

          // Use functional setState with version comparison to skip redundant updates
          setTradingState(prev => {
            // Skip update if version hasn't changed (avoids unnecessary re-renders)
            if (prev?.version === data.data.version) {
              return prev;
            }
            return {
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
              order_group: data.data.order_group,
              pnl: data.data.pnl,
              positions_details: data.data.positions_details || []
            };
          });

          // Handle settlements with reference stability
          if (data.data.settlements !== undefined) {
            const newSettlements = data.data.settlements || [];
            setSettlements(prevSettlements => {
              // Only update if actually changed (length or first item ticker)
              if (prevSettlements.length === newSettlements.length &&
                  prevSettlements[0]?.ticker === newSettlements[0]?.ticker) {
                return prevSettlements;
              }
              if (newSettlements.length > prevSettlements.length && newSettlements[0]) {
                setNewSettlement(newSettlements[0]);
                // Auto-dismiss after 5 seconds (with proper cleanup)
                if (settlementTimeoutRef.current) {
                  clearTimeout(settlementTimeoutRef.current);
                }
                settlementTimeoutRef.current = setTimeout(() => setNewSettlement(null), 5000);
              }
              return newSettlements;
            });
          }
        }
        break;

      case 'connection':
        // Initial connection acknowledgment
        break;

      case 'history_replay':
        if (data.data.transitions) {
          data.data.transitions.forEach(transition => {
            const fromState = transition.from_state || 'unknown';
            const toState = transition.to_state || transition.state;
            setCurrentState(toState);
            onMessage?.('state', transition.message, {
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

      case 'state_transition': {
        const fromState = data.data.from_state || currentStateRef.current;
        const toState = data.data.to_state || data.data.state;
        setCurrentState(toState);

        if (toState === 'trading_client_connect' && data.data.metadata?.api_url) {
          setMetrics(prev => ({
            ...prev,
            api_url: data.data.metadata.api_url,
            api_connected: true
          }));
        }

        if (['ready', 'calibrating', 'acting', 'trading_client_connect'].includes(toState)) {
          setMetrics(prev => ({ ...prev, api_connected: true }));
        } else if (['error', 'idle'].includes(toState)) {
          setMetrics(prev => ({ ...prev, api_connected: false }));
        }

        if (!data.data.is_current || data.data.message !== `Current state: ${toState}`) {
          onMessage?.('state', data.data.message, {
            state: toState,
            from_state: fromState,
            to_state: toState,
            context: data.data.context,
            timestamp: data.data.timestamp,
            metadata: data.data.metadata
          });
        }
        break;
      }

      case 'trader_status':
        if (data.data.metrics) {
          setMetrics(prev => ({
            markets_connected: data.data.metrics.markets_connected || 0,
            tracked_markets: data.data.metrics.tracked_markets || 0,
            subscribed_markets: data.data.metrics.subscribed_markets || 0,
            snapshots_received: data.data.metrics.snapshots_received || 0,
            deltas_received: data.data.metrics.deltas_received || 0,
            uptime: data.data.metrics.uptime || 0,
            health: data.data.metrics.health || 'unknown',
            ping_health: data.data.metrics.ping_health || 'unknown',
            last_ping_age: data.data.metrics.last_ping_age || null,
            api_connected: data.data.metrics.api_connected || prev.api_connected,
            api_url: data.data.metrics.api_url || prev.api_url,
            ws_url: data.data.metrics.ws_url || prev.ws_url,
            signal_aggregator: data.data.metrics.signal_aggregator || null
          }));
        }
        if (data.data.state) {
          setCurrentState(data.data.state);
        }
        break;

      case 'trade_processing':
        if (data.data) {
          setTradeProcessing({
            recent_trades: data.data.recent_trades || [],
            stats: data.data.stats || { trades_seen: 0, trades_filtered: 0, trades_tracked: 0, filter_rate_percent: 0 },
            decisions: data.data.decisions || { detected: 0, executed: 0, rate_limited: 0, skipped: 0, reentries: 0 },
            decision_history: data.data.decision_history || [],
            last_updated: data.data.last_updated || null,
            timestamp: data.data.timestamp || null
          });
        }
        break;

      case 'trading_strategies':
        if (data.data) {
          setStrategyStatus({
            coordinator: data.data.coordinator || INITIAL_STRATEGY_STATUS.coordinator,
            strategies: data.data.strategies || {},
            recent_decisions: data.data.recent_decisions || [],
            last_updated: data.data.last_updated || null,
            timestamp: data.data.timestamp || null
          });
        }
        break;

      case 'ping':
        // Track last ping time for heartbeat monitoring
        lastPingRef.current = Date.now();
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: 'pong', timestamp: Date.now() }));
        }
        break;

      default:
        break;
    }
  }, [onMessage]);

  // Connect on mount
  useEffect(() => {
    connectWebSocket();

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      // Clean up all toast timeouts on unmount
      if (orderFillTimeoutRef.current) {
        clearTimeout(orderFillTimeoutRef.current);
      }
      if (ttlCancellationTimeoutRef.current) {
        clearTimeout(ttlCancellationTimeoutRef.current);
      }
      if (settlementTimeoutRef.current) {
        clearTimeout(settlementTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [connectWebSocket]);

  // Heartbeat monitoring - detect zombie connections
  // If we haven't received a ping from server in 60s, force reconnect
  useEffect(() => {
    const checkHeartbeat = () => {
      const timeSinceLastPing = Date.now() - lastPingRef.current;
      // Server sends pings every 30s, so 60s timeout gives buffer
      if (timeSinceLastPing > 60000 && wsStatus === 'connected') {
        console.warn('[useV3WebSocket] No ping in 60s, forcing reconnect');
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

  // Clear new settlement after timeout
  const dismissSettlement = useCallback(() => {
    setNewSettlement(null);
  }, []);

  // Clear new order fill notification
  const dismissOrderFill = useCallback(() => {
    setNewOrderFill(null);
  }, []);

  // Clear new TTL cancellation notification
  const dismissTtlCancellation = useCallback(() => {
    setNewTtlCancellation(null);
  }, []);

  return {
    wsStatus,
    currentState,
    tradingState,
    lastUpdateTime,
    tradeProcessing,
    strategyStatus,
    settlements,
    newSettlement,
    dismissSettlement,
    newOrderFill,
    dismissOrderFill,
    newTtlCancellation,
    dismissTtlCancellation,
    metrics
  };
};

export default useV3WebSocket;
