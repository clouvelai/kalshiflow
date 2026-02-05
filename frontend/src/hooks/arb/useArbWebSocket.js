import { useState, useEffect, useRef, useCallback } from 'react';

/**
 * useArbWebSocket - WebSocket hook for the Single-Event Arb Dashboard
 *
 * Handles message types:
 *   event_arb_snapshot, event_arb_update, event_arb_ticker, event_arb_trade,
 *   arb_opportunity, arb_trade_executed,
 *   trading_state, agent_message, system_activity,
 *   trader_status, connection, ping/pong
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

export const useArbWebSocket = () => {
  const [connectionStatus, setConnectionStatus] = useState('connecting');
  const [systemState, setSystemState] = useState('initializing');
  const [tradingState, setTradingState] = useState(INITIAL_TRADING_STATE);
  const [arbTrades, setArbTrades] = useState([]);
  const [agentMessages, setAgentMessages] = useState([]);
  // Single-event arb state: Map<event_ticker, EventArbState>
  const [events, setEvents] = useState(new Map());
  // Public trade feed: Array of recent trades (newest first)
  const [eventTrades, setEventTrades] = useState([]);

  const wsRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);
  const lastPingRef = useRef(Date.now());
  const heartbeatIntervalRef = useRef(null);

  const handleMessage = useCallback((data, ws) => {
    switch (data.type) {
      // --- Single-event arb messages ---
      case 'event_arb_snapshot': {
        if (data.data?.events) {
          setEvents(() => {
            const next = new Map();
            Object.entries(data.data.events).forEach(([ticker, state]) => {
              next.set(ticker, { ...state, updated_at: Date.now() });
            });
            return next;
          });
        }
        break;
      }

      case 'event_arb_update': {
        if (data.data?.event_ticker) {
          const d = data.data;
          setEvents(prev => {
            const next = new Map(prev);
            next.set(d.event_ticker, { ...d, updated_at: Date.now() });
            return next;
          });
        }
        break;
      }

      case 'event_arb_ticker': {
        // Ticker V2 update: price, volume, OI delta for a market
        if (data.data?.event_ticker && data.data?.market_ticker) {
          const d = data.data;
          setEvents(prev => {
            const next = new Map(prev);
            const event = next.get(d.event_ticker);
            if (event?.markets?.[d.market_ticker]) {
              const m = { ...event.markets[d.market_ticker] };
              if (d.price != null) m.last_price = d.price;
              m.volume_delta_total = (m.volume_delta_total || 0) + (d.volume_delta || 0);
              m.oi_delta_total = (m.oi_delta_total || 0) + (d.open_interest_delta || 0);
              next.set(d.event_ticker, {
                ...event,
                markets: { ...event.markets, [d.market_ticker]: m },
                updated_at: Date.now(),
              });
            }
            return next;
          });
        }
        break;
      }

      case 'event_arb_trade': {
        // Public trade from trade channel
        if (data.data) {
          const trade = {
            id: `trade-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
            ...data.data,
            received_at: Date.now(),
          };
          setEventTrades(prev => [trade, ...prev].slice(0, 200));
        }
        break;
      }

      case 'arb_opportunity': {
        // Opportunities show as agent messages for visibility
        if (data.data) {
          const msg = {
            id: `opp-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
            type: 'agent_message',
            subtype: 'arb_opportunity',
            agent: 'single_arb_monitor',
            text: `${data.data.direction.toUpperCase()} ARB: ${data.data.event_ticker} edge=${data.data.edge_cents}c (after fees: ${data.data.edge_after_fees}c)`,
            timestamp: new Date().toISOString(),
          };
          setAgentMessages(prev => [msg, ...prev].slice(0, 200));
        }
        break;
      }

      case 'arb_trade_executed': {
        if (data.data) {
          const trade = {
            id: data.data.order_id || `arb-${Date.now()}`,
            event_ticker: data.data.event_ticker,
            pair_id: data.data.pair_id,
            kalshi_ticker: data.data.kalshi_ticker,
            direction: data.data.direction,
            side: data.data.side,
            action: data.data.action,
            contracts: data.data.contracts,
            price_cents: data.data.price_cents,
            spread_at_entry: data.data.spread_at_entry,
            legs_executed: data.data.legs_executed,
            total_cost_cents: data.data.total_cost_cents,
            order_id: data.data.order_id,
            reasoning: data.data.reasoning,
            timestamp: data.data.timestamp || new Date().toISOString(),
          };
          setArbTrades(prev => [trade, ...prev].slice(0, 100));
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
            category: data.data.category,
            todos: data.data.todos,
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
    arbTrades,
    agentMessages,
    events,
    eventTrades,
  };
};

export default useArbWebSocket;
