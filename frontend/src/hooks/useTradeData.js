import { useState, useEffect, useCallback, useMemo } from 'react';
import useWebSocket from './useWebSocket';

const useTradeData = () => {
  const [recentTrades, setRecentTrades] = useState([]);
  const [hotMarkets, setHotMarkets] = useState([]);
  const [selectedTicker, setSelectedTicker] = useState(null);
  const [tradeCount, setTradeCount] = useState(0);
  
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
            setTradeCount(lastMessage.data.recent_trades.length);
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
    hourAnalyticsData,  // NEW: Complete hour mode analytics data
    dayAnalyticsData,   // NEW: Complete day mode analytics data
    globalStats,
    connectionStatus,
    error,
    selectTicker,
    getTickerData
  };
};

export default useTradeData;