import { useState, useEffect, useCallback, useMemo } from 'react';
import useWebSocket from './useWebSocket';

const useTradeData = () => {
  const [recentTrades, setRecentTrades] = useState([]);
  const [hotMarkets, setHotMarkets] = useState([]);
  const [selectedTicker, setSelectedTicker] = useState(null);
  const [tradeCount, setTradeCount] = useState(0);
  
  // Simplified state structure for 2-message architecture
  const [analyticsData, setAnalyticsData] = useState({
    hour_minute_mode: { time_series: [], summary_stats: {} },
    day_hour_mode: { time_series: [], summary_stats: {} }
  });
  
  // Real-time current period data from realtime_update messages
  const [realtimeData, setRealtimeData] = useState({
    current_minute: { timestamp: 0, volume_usd: 0, trade_count: 0 },
    current_hour: { timestamp: 0, volume_usd: 0, trade_count: 0 },
    mode_totals: {
      hour_mode_total_volume_usd: 0,
      hour_mode_total_trades: 0,
      day_mode_total_volume_usd: 0,
      day_mode_total_trades: 0
    },
    peaks: { peak_volume_usd: 0, peak_trades: 0 }
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

  // Process incoming WebSocket messages - Simplified 2-message architecture
  useEffect(() => {
    if (!lastMessage) return;

    try {
      switch (lastMessage.type) {
        case 'snapshot':
          // Initial data load
          if (lastMessage.data?.recent_trades) {
            setRecentTrades(lastMessage.data.recent_trades);
            setTradeCount(lastMessage.data.recent_trades.length);
          }
          if (lastMessage.data?.hot_markets) {
            setHotMarkets(lastMessage.data.hot_markets);
          }
          if (lastMessage.data?.global_stats) {
            setGlobalStats(lastMessage.data.global_stats);
          }
          if (lastMessage.data?.analytics_data) {
            // Handle chart data for initial load
            if (lastMessage.data.analytics_data.hour_minute_mode && lastMessage.data.analytics_data.day_hour_mode) {
              setAnalyticsData(lastMessage.data.analytics_data);
            } else {
              console.warn('Received analytics data in unexpected format during snapshot');
            }
          }
          break;

        case 'trade':
          // Real-time trade update
          if (lastMessage.data?.trade) {
            const newTrade = lastMessage.data.trade;
            
            setRecentTrades(prevTrades => {
              const updatedTrades = [newTrade, ...prevTrades];
              // Keep only the most recent 200 trades
              return updatedTrades.slice(0, 200);
            });
            
            setTradeCount(prev => prev + 1);
          }
          
          // Update global stats if provided
          if (lastMessage.data?.global_stats) {
            setGlobalStats(lastMessage.data.global_stats);
          }
          
          // Update hot markets with metadata if provided in trade updates
          if (lastMessage.data?.hot_markets) {
            setHotMarkets(lastMessage.data.hot_markets);
          } else if (lastMessage.data?.ticker_state) {
            // Fallback: update individual ticker state while preserving metadata
            const tickerState = lastMessage.data.ticker_state;
            
            setHotMarkets(prevMarkets => {
              const updatedMarkets = [...prevMarkets];
              const existingIndex = updatedMarkets.findIndex(
                market => market.ticker === tickerState.ticker
              );
              
              if (existingIndex >= 0) {
                // Preserve metadata fields from existing market data
                const existingMarket = updatedMarkets[existingIndex];
                updatedMarkets[existingIndex] = {
                  ...tickerState,
                  // Preserve metadata fields
                  title: existingMarket.title,
                  category: existingMarket.category,
                  liquidity_dollars: existingMarket.liquidity_dollars,
                  open_interest: existingMarket.open_interest,
                  latest_expiration_time: existingMarket.latest_expiration_time
                };
              } else {
                updatedMarkets.push(tickerState);
              }
              
              // Sort by volume window and take top 12
              return updatedMarkets
                .sort((a, b) => (b.volume_window || 0) - (a.volume_window || 0))
                .slice(0, 12);
            });
          }
          break;

        case 'realtime_update':
          // NEW: Real-time update for current period data everywhere (instant updates on every trade)
          // Powers: Current minute/hour boxes + current chart bar + totals + peaks
          if (lastMessage.data) {
            const updateData = lastMessage.data;
            
            // Validate the structure
            if (updateData.current_minute && updateData.current_hour && 
                updateData.mode_totals && updateData.peaks) {
              
              setRealtimeData({
                current_minute: updateData.current_minute,
                current_hour: updateData.current_hour,
                mode_totals: updateData.mode_totals,
                peaks: updateData.peaks
              });
            } else {
              console.warn('Received incomplete realtime_update data', updateData);
            }
          }
          break;

        case 'chart_data':
          // NEW: Historical chart data only (60s updates, excludes current period)
          // Powers: Historical chart bars only
          if (lastMessage.data?.hour_minute_mode && lastMessage.data?.day_hour_mode) {
            setAnalyticsData({
              hour_minute_mode: lastMessage.data.hour_minute_mode,
              day_hour_mode: lastMessage.data.day_hour_mode
            });
          } else {
            console.warn('Received incomplete chart_data', lastMessage.data);
          }
          break;

        default:
          console.warn('Received unknown message type:', lastMessage.type);
      }
    } catch (err) {
      console.error('Error processing WebSocket message:', err);
    }
  }, [lastMessage]);

  const selectTicker = useCallback((ticker) => {
    setSelectedTicker(ticker);
  }, []);

  const getTickerData = useCallback((ticker) => {
    if (!ticker) return null;
    
    const marketData = hotMarkets.find(market => market.ticker === ticker);
    const tickerTrades = recentTrades.filter(trade => trade.market_ticker === ticker);
    
    return {
      marketData,
      recentTrades: tickerTrades
    };
  }, [hotMarkets, recentTrades]);

  return {
    recentTrades,
    hotMarkets,
    selectedTicker,
    tradeCount,
    analyticsData,
    realtimeData,  // NEW: Real-time current period data
    globalStats,
    connectionStatus,
    error,
    selectTicker,
    getTickerData
  };
};

export default useTradeData;