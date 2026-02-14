import { useState, useEffect, useRef, useCallback } from 'react';

const V3_PORT = import.meta.env.VITE_V3_PORT || 8005;
const V3_BACKEND_URL = import.meta.env.VITE_V3_BACKEND_URL || null;

function getWsUrl() {
  if (V3_BACKEND_URL) {
    const url = new URL(V3_BACKEND_URL);
    const protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${protocol}//${url.host}/v3/ws`;
  }
  return `ws://localhost:${V3_PORT}/v3/ws`;
}

export function useMMWebSocket() {
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const [events, setEvents] = useState(new Map());
  const [quoteState, setQuoteState] = useState(null);
  const [inventory, setInventory] = useState([]);
  const [agentMessages, setAgentMessages] = useState([]);
  const [tradingState, setTradingState] = useState(null);
  const [performance, setPerformance] = useState(null);
  const [tradeLog, setTradeLog] = useState([]);
  const [balanceInfo, setBalanceInfo] = useState(null);

  const wsRef = useRef(null);
  const reconnectRef = useRef(null);
  const lastPingRef = useRef(Date.now());

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const url = getWsUrl();
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnectionStatus('connected');
      lastPingRef.current = Date.now();
    };

    ws.onclose = () => {
      setConnectionStatus('disconnected');
      reconnectRef.current = setTimeout(connect, 3000);
    };

    ws.onerror = () => {
      setConnectionStatus('error');
    };

    ws.onmessage = (event) => {
      lastPingRef.current = Date.now();
      try {
        const msg = JSON.parse(event.data);
        handleMessage(msg);
      } catch (e) {
        // ignore parse errors
      }
    };
  }, []);

  const handleMessage = useCallback((msg) => {
    const { type, data } = msg;

    switch (type) {
      case 'mm_snapshot':
        if (data?.events) {
          setEvents(new Map(Object.entries(data.events)));
        }
        if (data?.quote_state) {
          setQuoteState(data.quote_state);
        }
        break;

      case 'mm_market_update':
        // Incremental market data update
        break;

      case 'mm_inventory_update':
        if (data?.markets) {
          setInventory(data.markets);
        }
        break;

      case 'mm_quote_placed':
      case 'mm_quote_cancelled':
      case 'mm_quote_filled':
        setTradeLog(prev => [{
          type: type.replace('mm_', ''),
          ...data,
          timestamp: data.timestamp || Date.now() / 1000,
        }, ...prev].slice(0, 200));
        break;

      case 'mm_quotes_pulled':
        setQuoteState(prev => ({
          ...prev,
          quotes_pulled: true,
          pull_reason: data?.reason || '',
        }));
        setTradeLog(prev => [{
          type: 'quotes_pulled',
          cancelled: data?.cancelled || 0,
          reason: data?.reason || '',
          timestamp: Date.now() / 1000,
        }, ...prev].slice(0, 200));
        break;

      case 'mm_requote_cycle':
        setQuoteState(prev => ({
          ...prev,
          active_quotes: data?.active_quotes || 0,
          total_requote_cycles: data?.cycle || 0,
          spread_multiplier: data?.spread_multiplier || 1,
        }));
        break;

      case 'mm_balance_update':
        setBalanceInfo(data);
        break;

      case 'mm_performance_snapshot':
        setPerformance(data);
        break;

      case 'agent_message':
        setAgentMessages(prev => [
          { ...data, timestamp: data.timestamp || Date.now() / 1000 },
          ...prev,
        ].slice(0, 200));
        break;

      case 'trading_state':
        setTradingState(data);
        break;

      case 'ping':
        break;

      default:
        break;
    }
  }, []);

  useEffect(() => {
    connect();

    // Heartbeat check
    const heartbeat = setInterval(() => {
      if (Date.now() - lastPingRef.current > 90000) {
        wsRef.current?.close();
      }
    }, 15000);

    return () => {
      clearInterval(heartbeat);
      clearTimeout(reconnectRef.current);
      wsRef.current?.close();
    };
  }, [connect]);

  return {
    connectionStatus,
    events,
    quoteState,
    inventory,
    agentMessages,
    tradingState,
    performance,
    tradeLog,
    balanceInfo,
  };
}
