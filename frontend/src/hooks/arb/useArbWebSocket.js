import { useState, useEffect, useRef, useCallback } from 'react';

/**
 * useArbWebSocket - WebSocket hook for the Arbitrage Dashboard
 *
 * Handles message types:
 *   spread_update, arb_trade_executed, arb_pairs,
 *   trading_state, agent_message, system_activity,
 *   tracked_markets_state, connection, ping/pong
 */

const INITIAL_TRADING_STATE = {
  balance: 0,
  positions: [],
  open_orders: [],
  settlements: [],
  pnl: null,
  portfolio_value: 0,
  position_count: 0,
  order_count: 0,
};

const INITIAL_METRICS = {
  uptime: 0,
  health: 'unknown',
  api_connected: false,
};

export const useArbWebSocket = () => {
  const [connectionStatus, setConnectionStatus] = useState('connecting');
  const [systemState, setSystemState] = useState('initializing');
  const [tradingState, setTradingState] = useState(INITIAL_TRADING_STATE);
  const [spreads, setSpreads] = useState(new Map());
  const [arbTrades, setArbTrades] = useState([]);
  const [agentMessages, setAgentMessages] = useState([]);
  const [metrics, setMetrics] = useState(INITIAL_METRICS);
  const [trackedMarkets, setTrackedMarkets] = useState([]);
  const [pairIndex, setPairIndex] = useState(null);
  const [eventCodex, setEventCodex] = useState(null);

  const wsRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);
  const lastPingRef = useRef(Date.now());
  const heartbeatIntervalRef = useRef(null);

  const handleMessage = useCallback((data, ws) => {
    switch (data.type) {
      case 'spread_update': {
        if (data.data) {
          const d = data.data;
          setSpreads(prev => {
            const next = new Map(prev);
            next.set(d.pair_id, {
              pair_id: d.pair_id,
              kalshi_ticker: d.kalshi_ticker,
              kalshi_yes_bid: d.kalshi_yes_bid,
              kalshi_yes_ask: d.kalshi_yes_ask,
              kalshi_yes_mid: d.kalshi_yes_mid,
              poly_yes_cents: d.poly_yes_cents,
              spread_cents: d.spread_cents,
              question: d.question,
              // Kalshi WS source
              kalshi_ws_yes_bid: d.kalshi_ws_yes_bid,
              kalshi_ws_yes_ask: d.kalshi_ws_yes_ask,
              kalshi_ws_yes_mid: d.kalshi_ws_yes_mid,
              kalshi_ws_age_ms: d.kalshi_ws_age_ms,
              // Kalshi API source
              kalshi_api_yes_bid: d.kalshi_api_yes_bid,
              kalshi_api_yes_ask: d.kalshi_api_yes_ask,
              kalshi_api_yes_mid: d.kalshi_api_yes_mid,
              kalshi_api_age_ms: d.kalshi_api_age_ms,
              // Poly WS source
              poly_ws_yes_cents: d.poly_ws_yes_cents,
              poly_ws_age_ms: d.poly_ws_age_ms,
              // Poly API source
              poly_api_yes_cents: d.poly_api_yes_cents,
              poly_api_age_ms: d.poly_api_age_ms,
              // Tradeable status
              tradeable: d.tradeable,
              tradeable_reason: d.tradeable_reason,
              updated_at: Date.now(),
            });
            return next;
          });
        }
        break;
      }

      case 'arb_trade_executed': {
        if (data.data) {
          const trade = {
            id: data.data.order_id || `arb-${Date.now()}`,
            pair_id: data.data.pair_id,
            kalshi_ticker: data.data.kalshi_ticker,
            side: data.data.side,
            action: data.data.action,
            contracts: data.data.contracts,
            price_cents: data.data.price_cents,
            spread_at_entry: data.data.spread_at_entry,
            order_id: data.data.order_id,
            reasoning: data.data.reasoning,
            timestamp: data.data.timestamp || new Date().toISOString(),
          };
          setArbTrades(prev => [trade, ...prev].slice(0, 100));
        }
        break;
      }

      case 'arb_pairs': {
        if (data.data?.pairs) {
          setSpreads(() => {
            const next = new Map();
            data.data.pairs.forEach(p => {
              next.set(p.pair_id, { ...p, updated_at: Date.now() });
            });
            return next;
          });
        }
        break;
      }

      case 'trading_state': {
        if (data.data) {
          setTradingState(prev => {
            if (prev?.version === data.data.version) return prev;
            return {
              version: data.data.version,
              balance: data.data.balance ?? 0,
              portfolio_value: data.data.portfolio_value ?? 0,
              position_count: data.data.position_count ?? 0,
              order_count: data.data.order_count ?? 0,
              positions: data.data.positions_details || data.data.positions || [],
              open_orders: data.data.open_orders || [],
              settlements: data.data.settlements || [],
              pnl: data.data.pnl ?? null,
              sync_timestamp: data.data.sync_timestamp,
              market_prices: data.data.market_prices || prev?.market_prices || {},
            };
          });
        }
        break;
      }

      case 'agent_message': {
        if (data.data) {
          const msg = {
            id: `agent-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
            type: data.data.type,
            subtype: data.data.subtype,
            agent: data.data.agent,
            prompt: data.data.prompt,
            response_preview: data.data.response_preview,
            tool_name: data.data.tool_name,
            tool_input: data.data.tool_input,
            tool_output: data.data.tool_output,
            duration: data.data.duration,
            error: data.data.error,
            text: data.data.text,
            action: data.data.action,
            ticker: data.data.ticker,
            side: data.data.side,
            contracts: data.data.contracts,
            price: data.data.price,
            pair_id: data.data.pair_id,
            result: data.data.result,
            order_id: data.data.order_id,
            elapsed: data.data.elapsed,
            timestamp: data.data.timestamp || new Date().toISOString(),
          };
          setAgentMessages(prev => [msg, ...prev].slice(0, 200));
        }
        break;
      }

      case 'system_activity': {
        if (data.data) {
          const { activity_type, metadata } = data.data;
          if (activity_type === 'state_transition' && metadata?.to_state) {
            setSystemState(metadata.to_state);
          }
        }
        break;
      }

      case 'tracked_markets_state':
      case 'tracked_markets': {
        if (data.data?.markets) {
          setTrackedMarkets(data.data.markets);
        }
        break;
      }

      case 'pair_index_snapshot': {
        if (data.data) {
          setPairIndex(data.data);
        }
        break;
      }

      case 'event_codex_snapshot': {
        if (data.data) {
          setEventCodex(data.data);
        }
        break;
      }

      case 'connection':
        break;

      case 'ping': {
        lastPingRef.current = Date.now();
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: 'pong', timestamp: Date.now() }));
        }
        break;
      }

      case 'trader_status': {
        if (data.data?.metrics) {
          setMetrics(prev => ({
            ...prev,
            uptime: data.data.metrics.uptime || 0,
            health: data.data.metrics.health || 'unknown',
            api_connected: data.data.metrics.api_connected || prev.api_connected,
          }));
        }
        if (data.data?.state) {
          setSystemState(data.data.state);
        }
        break;
      }

      default:
        break;
    }
  }, []);

  const connectWebSocket = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    if (
      wsRef.current?.readyState === WebSocket.OPEN ||
      wsRef.current?.readyState === WebSocket.CONNECTING
    ) {
      return;
    }

    try {
      const backendPort =
        import.meta.env.VITE_V3_BACKEND_PORT ||
        import.meta.env.VITE_BACKEND_PORT ||
        '8005';
      const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsHost = import.meta.env.VITE_V3_BACKEND_URL || `${wsProtocol}//localhost:${backendPort}`;
      const wsUrl = wsHost.startsWith('ws') ? `${wsHost}/v3/ws` : `${wsProtocol}//${wsHost}/v3/ws`;
      const ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        setConnectionStatus('connected');
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          handleMessage(data, ws);
        } catch (err) {
          console.error('[useArbWebSocket] Parse error:', err);
        }
      };

      ws.onerror = () => {
        setConnectionStatus('error');
      };

      ws.onclose = (event) => {
        setConnectionStatus('disconnected');
        wsRef.current = null;
        if (event.code !== 1000) {
          reconnectTimeoutRef.current = setTimeout(connectWebSocket, 3000);
        }
      };

      wsRef.current = ws;
    } catch (err) {
      console.error('[useArbWebSocket] Connect failed:', err);
      setConnectionStatus('error');
    }
  }, [handleMessage]);

  useEffect(() => {
    connectWebSocket();
    return () => {
      if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
      if (wsRef.current) wsRef.current.close();
    };
  }, [connectWebSocket]);

  useEffect(() => {
    const check = () => {
      if (Date.now() - lastPingRef.current > 90000 && connectionStatus === 'connected') {
        console.warn('[useArbWebSocket] No ping in 90s, reconnecting');
        wsRef.current?.close();
      }
    };
    heartbeatIntervalRef.current = setInterval(check, 15000);
    return () => {
      if (heartbeatIntervalRef.current) clearInterval(heartbeatIntervalRef.current);
    };
  }, [connectionStatus]);

  return {
    connectionStatus,
    systemState,
    tradingState,
    spreads,
    arbTrades,
    agentMessages,
    metrics,
    trackedMarkets,
    pairIndex,
    eventCodex,
  };
};

export default useArbWebSocket;
