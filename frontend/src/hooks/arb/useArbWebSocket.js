import { useState, useEffect, useRef, useCallback } from 'react';

/**
 * useArbWebSocket - WebSocket hook for the Single-Event Arb Dashboard
 *
 * Handles message types:
 *   event_arb_snapshot, event_arb_update,
 *   arb_opportunity, arb_trade_executed, order_status_update, order_tracker_snapshot,
 *   trading_state, agent_message, system_activity,
 *   feed_stats, captain_paused, exchange_status,
 *   account_health_update, sniper_status, sniper_execution,
 *   attention_snapshot, auto_action_fired,
 *   captain_cycle_start, captain_cycle_complete, captain_config,
 *   connection, ping/pong
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
  // Feed stats from backend monitor (orderbook, ticker, trade, poll counts + timestamps)
  const [feedStats, setFeedStats] = useState(null);
  // Captain pause state
  const [captainPaused, setCaptainPaused] = useState(false);
  // Exchange status (active/down)
  const [exchangeStatus, setExchangeStatus] = useState({ active: true, error: null, lastCheck: null });
  // Sniper execution state
  const [sniperState, setSniperState] = useState({ enabled: false, lastAction: null, capitalActive: 0, capitalLimit: 0, recentActions: [] });
  // Startup progress messages
  const [startupMessages, setStartupMessages] = useState([]);
  // Account health from background service
  const [accountHealth, setAccountHealth] = useState(null);
  // Attention router state
  const [attentionItems, setAttentionItems] = useState([]);
  const [attentionStats, setAttentionStats] = useState(null);
  // Auto-action events
  const [autoActions, setAutoActions] = useState([]);
  // Captain cycle mode
  const [captainMode, setCaptainMode] = useState(null);
  // Captain timing: last completion timestamps per mode + configured intervals
  const [captainTiming, setCaptainTiming] = useState({
    lastStrategic: null,
    lastDeepScan: null,
    strategicInterval: 300,   // default 5min
    deepScanInterval: 1800,   // default 30min
  });
  // Discovery state: top events by volume
  const [discoveryState, setDiscoveryState] = useState({
    events: [],
    stats: null,
    lastFetch: null,
    recentEvictions: [],
  });

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
            const existing = next.get(d.event_ticker) || {};
            next.set(d.event_ticker, { ...existing, ...d, updated_at: Date.now() });
            return next;
          });
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
            status: data.data.status || 'placed',
            reasoning: data.data.reasoning,
            timestamp: data.data.timestamp || new Date().toISOString(),
          };
          setArbTrades(prev => [trade, ...prev].slice(0, 100));
        }
        break;
      }

      case 'order_status_update': {
        if (data.data?.order_id) {
          const update = data.data;
          setArbTrades(prev =>
            prev.map(trade =>
              trade.order_id === update.order_id
                ? {
                    ...trade,
                    status: update.status,
                    fill_count: update.fill_count,
                    remaining_count: update.remaining_count,
                    status_updated_at: update.updated_at,
                  }
                : trade
            )
          );
        }
        break;
      }

      case 'order_tracker_snapshot': {
        if (data.data?.orders) {
          const orderMap = {};
          data.data.orders.forEach(o => { orderMap[o.order_id] = o; });
          setArbTrades(prev =>
            prev.map(trade => {
              const tracked = orderMap[trade.order_id];
              if (tracked) {
                return {
                  ...trade,
                  status: tracked.status,
                  fill_count: tracked.fill_count,
                  remaining_count: tracked.remaining_count,
                  status_updated_at: tracked.updated_at,
                };
              }
              return trade;
            })
          );
        }
        break;
      }

      case 'trading_state': {
        if (data.data) {
          setTradingState(prev => {
            if (prev?.version === data.data.version) return prev;
            return {
              version: data.data.version,
              subaccount_number: data.data.subaccount_number ?? 0,
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
              event_exposure: data.data.event_exposure || prev?.event_exposure || null,
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
          const { activity_type, metadata, message } = data.data;
          if (activity_type === 'state_transition' && metadata?.to_state) {
            setSystemState(metadata.to_state);
          }
          if (activity_type === 'startup_progress' || activity_type === 'state_transition') {
            setStartupMessages(prev => [...prev, {
              message: message || metadata?.to_state || '',
              step: data.data.step,
              totalSteps: data.data.total_steps,
              timestamp: Date.now(),
            }].slice(-20));
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

      case 'feed_stats': {
        if (data.data) {
          setFeedStats(data.data);
        }
        break;
      }

      case 'captain_paused': {
        if (data.data != null) {
          setCaptainPaused(data.data.paused);
        }
        break;
      }

      case 'exchange_status': {
        if (data.data != null) {
          setExchangeStatus({
            active: data.data.active,
            error: data.data.error,
            lastCheck: data.data.last_check,
          });
        }
        break;
      }

      case 'account_health_update': {
        if (data.data) {
          setAccountHealth(data.data);
        }
        break;
      }

      case 'sniper_status': {
        if (data.data) {
          const d = data.data;
          setSniperState(prev => ({
            ...prev,
            enabled: d.config?.enabled ?? d.enabled ?? prev.enabled,
            capitalActive: d.state?.capital_active ?? prev.capitalActive,
            capitalLimit: d.config?.max_capital ?? prev.capitalLimit,
            recentActions: (d.state?.recent_actions || []).concat(prev.recentActions || []).slice(0, 20),
          }));
        }
        break;
      }

      case 'attention_snapshot': {
        if (data.data) {
          setAttentionItems(data.data.items || []);
          setAttentionStats(data.data.stats || null);
        }
        break;
      }

      case 'auto_action_fired': {
        if (data.data) {
          setAutoActions(prev => [data.data, ...prev].slice(0, 20));
        }
        break;
      }

      case 'captain_cycle_start': {
        if (data.data) {
          setCaptainMode({
            mode: data.data.mode,
            cycle_num: data.data.cycle_num,
            trigger_items: data.data.trigger_items,
            timestamp: Date.now(),
          });
        }
        break;
      }

      case 'captain_cycle_complete': {
        if (data.data) {
          const mode = data.data.mode;
          const now = Date.now();
          setCaptainTiming(prev => ({
            ...prev,
            ...(mode === 'strategic' ? { lastStrategic: now } : {}),
            ...(mode === 'deep_scan' ? { lastDeepScan: now, lastStrategic: now } : {}),
          }));
        }
        break;
      }

      case 'captain_config': {
        if (data.data) {
          setCaptainTiming(prev => ({
            ...prev,
            strategicInterval: data.data.strategic_interval ?? prev.strategicInterval,
            deepScanInterval: data.data.deep_scan_interval ?? prev.deepScanInterval,
          }));
        }
        break;
      }

      case 'discovery_state': {
        if (data.data) {
          setDiscoveryState(prev => ({
            events: data.data.events || [],
            stats: data.data.stats || null,
            lastFetch: data.data.timestamp ? data.data.timestamp * 1000 : Date.now(),
            recentEvictions: prev.recentEvictions || [],
          }));
        }
        break;
      }

      case 'discovery_update': {
        if (data.data?.event_ticker) {
          setDiscoveryState(prev => {
            const exists = prev.events.some(e => e.event_ticker === data.data.event_ticker);
            if (exists) return prev;
            return {
              ...prev,
              events: [...prev.events, data.data],
              lastFetch: Date.now(),
            };
          });
        }
        break;
      }

      case 'discovery_eviction': {
        if (data.data?.evicted) {
          const now = Date.now();
          const newEvictions = data.data.evicted.map(e => ({
            ...e,
            evicted_at: now,
          }));
          setDiscoveryState(prev => ({
            ...prev,
            recentEvictions: [...newEvictions, ...(prev.recentEvictions || [])].slice(0, 10),
          }));
        }
        break;
      }

      case 'sniper_execution': {
        if (data.data) {
          const d = data.data;
          setSniperState(prev => ({
            ...prev,
            enabled: true,
            lastAction: d,
            recentActions: [d, ...(prev.recentActions || [])].slice(0, 20),
          }));
          // Also surface as agent message for activity feed
          const msg = {
            id: `sniper-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
            type: 'agent_message',
            subtype: 'sniper_execution',
            agent: 'sniper',
            text: d.error
              ? `SNIPER REJECTED ${d.event_ticker}: ${d.error}`
              : `SNIPER ${d.direction?.toUpperCase()} ${d.event_ticker} legs=${d.legs_filled}/${d.legs_attempted} cost=${d.total_cost_cents}c edge=${d.edge_cents}c`,
            timestamp: new Date().toISOString(),
          };
          setAgentMessages(prev => [msg, ...prev].slice(0, 200));
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

      let wsUrl;
      const envUrl = import.meta.env.VITE_V3_BACKEND_URL;
      if (envUrl) {
        // Strip any existing protocol (http://, https://, ws://, wss://) and add ws/wss
        const host = envUrl.replace(/^(https?|wss?):\/\//, '');
        wsUrl = `${wsProtocol}//${host}/v3/ws`;
      } else {
        wsUrl = `${wsProtocol}//localhost:${backendPort}/v3/ws`;
      }
      const ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        setConnectionStatus('connected');
        // Clear session-scoped state so stale data from a previous backend session disappears
        setAgentMessages([]);
        setArbTrades([]);
        setAttentionItems([]);
        setAutoActions([]);
        setCaptainMode(null);
        setStartupMessages([]);
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
    // Clear any existing interval first to prevent memory leak during reconnection cycles
    if (heartbeatIntervalRef.current) {
      clearInterval(heartbeatIntervalRef.current);
    }

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

  const sendCaptainPauseToggle = useCallback(() => {
    const ws = wsRef.current;
    if (ws?.readyState === WebSocket.OPEN) {
      const msgType = captainPaused ? 'captain_resume' : 'captain_pause';
      ws.send(JSON.stringify({ type: msgType }));
    }
  }, [captainPaused]);

  return {
    connectionStatus,
    systemState,
    tradingState,
    arbTrades,
    agentMessages,
    events,
    eventTrades,
    feedStats,
    captainPaused,
    sendCaptainPauseToggle,
    exchangeStatus,
    sniperState,
    startupMessages,
    accountHealth,
    attentionItems,
    attentionStats,
    autoActions,
    captainMode,
    captainTiming,
    discoveryState,
  };
};

export default useArbWebSocket;
