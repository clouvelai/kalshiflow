import { useState, useEffect, useCallback, useMemo } from 'react';
import useWebSocket from './useWebSocket';

const useTradeData = () => {
  const [recentTrades, setRecentTrades] = useState([]);
  const [hotMarkets, setHotMarkets] = useState([]);
  
  // Simplified state structure for single analytics_update message
  const [hourAnalyticsData, setHourAnalyticsData] = useState({
    current_period: { timestamp: 0, volume_usd: 0, trade_count: 0 },
    summary_stats: { total_volume_usd: 0, total_trades: 0, peak_volume_usd: 0, peak_trades: 0 },
    time_series: []
  });
  
  const [dayAnalyticsData, setDayAnalyticsData] = useState({
    current_period: { timestamp: 0, volume_usd: 0, trade_count: 0 },
    summary_stats: { total_volume_usd: 0, total_trades: 0, peak_volume_usd: 0, peak_trades: 0 },
    time_series: []
  });
  
  const [globalStats, setGlobalStats] = useState({
    daily_trades_count: 0,
    total_volume: 0,
    total_net_flow: 0,
    session_start_time: null,
    active_markets_count: 0,
    total_window_volume: 0
  });
  
  // Memoize WebSocket URL to prevent recreation on every render
  const wsUrl = useMemo(() => 
    import.meta.env.VITE_WS_URL || `ws://localhost:8000/ws/stream`, 
    []
  );
  const { connectionStatus, lastMessage, error } = useWebSocket(wsUrl);

  // Process incoming WebSocket messages - Single analytics_update message
  useEffect(() => {
    if (!lastMessage) return;

    try {
      switch (lastMessage.type) {
        case 'snapshot':
          // Initial data load
          if (lastMessage.data?.recent_trades) {
            setRecentTrades(lastMessage.data.recent_trades);
          }
          if (lastMessage.data?.hot_markets) {
            setHotMarkets(lastMessage.data.hot_markets);
          }
          if (lastMessage.data?.global_stats) {
            setGlobalStats(lastMessage.data.global_stats);
          }
          // Initial analytics data will come via analytics_update messages
          break;

        case 'trade':
          // Real-time trade update
          if (lastMessage.data?.trade) {
            setRecentTrades(prev => [lastMessage.data.trade, ...prev.slice(0, 199)]);
          }
          
          // Update global stats if provided
          if (lastMessage.data?.global_stats) {
            setGlobalStats(lastMessage.data.global_stats);
          }
          
          // Update hot markets only if explicitly provided in trade updates
          if (lastMessage.data?.hot_markets) {
            setHotMarkets(lastMessage.data.hot_markets);
          }
          break;

        case 'trades':
          // Batched trades update
          if (lastMessage.data?.trades && Array.isArray(lastMessage.data.trades)) {
            setRecentTrades(prev => {
              const newTrades = [...lastMessage.data.trades, ...prev];
              return newTrades.slice(0, 200);
            });
          }
          break;

        case 'analytics_update':
          // NEW: Single unified analytics update for clean real-time updates
          // Complete analytics data for one mode with guaranteed consistency
          if (lastMessage.data) {
            const updateData = lastMessage.data;
            
            // Validate the structure
            if (updateData.mode && updateData.current_period && 
                updateData.summary_stats && updateData.time_series !== undefined) {
              
              const analyticsData = {
                current_period: updateData.current_period,
                summary_stats: updateData.summary_stats,
                time_series: updateData.time_series
              };
              
              if (updateData.mode === 'hour') {
                setHourAnalyticsData(analyticsData);
              } else if (updateData.mode === 'day') {
                setDayAnalyticsData(analyticsData);
              } else {
                console.warn('Unknown analytics mode:', updateData.mode);
              }
            } else {
              console.warn('Received incomplete analytics_update data', updateData);
            }
          }
          break;

        case 'hot_markets_update':
          // NEW: Periodic hot markets update (every 30 seconds)
          if (lastMessage.data?.hot_markets) {
            setHotMarkets(lastMessage.data.hot_markets);
          }
          break;

        case 'ping':
          // WebSocket keepalive ping from server - no action needed
          break;

        case 'pong':
          // WebSocket keepalive pong response - no action needed
          break;

        default:
          console.warn('Received unknown message type:', lastMessage.type);
      }
    } catch (err) {
      console.error('Error processing WebSocket message:', err);
    }
  }, [lastMessage]);


  return {
    recentTrades,
    hotMarkets,
    hourAnalyticsData,  // Complete hour mode analytics data
    dayAnalyticsData,   // Complete day mode analytics data
    globalStats,
    connectionStatus,
    error
  };
};

export default useTradeData;