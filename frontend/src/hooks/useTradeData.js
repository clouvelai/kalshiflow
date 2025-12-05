import { useState, useEffect, useCallback } from 'react';
import useWebSocket from './useWebSocket';

const useTradeData = () => {
  const [recentTrades, setRecentTrades] = useState([]);
  const [hotMarkets, setHotMarkets] = useState([]);
  const [selectedTicker, setSelectedTicker] = useState(null);
  const [tradeCount, setTradeCount] = useState(0);
  const [analyticsData, setAnalyticsData] = useState({
    hour_minute_mode: { time_series: [], summary_stats: {} },
    day_hour_mode: { time_series: [], summary_stats: {} }
  });
  const [analyticsSummary, setAnalyticsSummary] = useState({
    peak_volume_usd: 0,
    total_volume_usd: 0,
    peak_trades: 0,
    total_trades: 0,
    current_minute_volume_usd: 0,
    current_minute_trades: 0
  });
  const [globalStats, setGlobalStats] = useState({
    daily_trades_count: 0,
    total_volume: 0,
    total_net_flow: 0,
    session_start_time: null,
    active_markets_count: 0,
    total_window_volume: 0
  });
  
  const wsUrl = `ws://localhost:8000/ws/stream`;
  const { connectionStatus, lastMessage, error } = useWebSocket(wsUrl);

  // Process incoming WebSocket messages
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
            // Handle new dual-mode format
            if (lastMessage.data.analytics_data.hour_minute_mode && lastMessage.data.analytics_data.day_hour_mode) {
              setAnalyticsData(lastMessage.data.analytics_data);
              // Set legacy analyticsSummary for backward compatibility (use hour mode by default)
              setAnalyticsSummary(lastMessage.data.analytics_data.hour_minute_mode.summary_stats || {});
            } else {
              // Fallback for old format
              console.warn('Received analytics data in old format, this should not happen with new backend');
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

        case 'analytics_data':
          // Real-time analytics update with dual-mode format (full time series)
          if (lastMessage.data?.hour_minute_mode && lastMessage.data?.day_hour_mode) {
            setAnalyticsData(lastMessage.data);
            // Update legacy analyticsSummary for backward compatibility (use hour mode by default)
            setAnalyticsSummary(lastMessage.data.hour_minute_mode.summary_stats || {});
          } else {
            console.warn('Received analytics_data in old format, this should not happen with new backend');
          }
          break;

        case 'analytics_incremental':
          // Lightweight incremental analytics update (current period data + summary only)
          if (lastMessage.data?.current_minute_data && lastMessage.data?.current_hour_data) {
            const incrementalData = lastMessage.data;
            
            // Efficiently update existing time series data with current period information
            setAnalyticsData(prevAnalytics => {
              const now = Date.now();
              const currentMinuteTimestamp = incrementalData.current_minute_data.timestamp;
              const currentHourTimestamp = incrementalData.current_hour_data.timestamp;
              
              // Helper function to create complete sliding window with zero-filled gaps
              const createCompleteTimeWindow = (existingTimeSeries, currentData, windowSize, periodMs) => {
                const currentPeriodStart = Math.floor(currentData.timestamp / periodMs) * periodMs;
                const windowStart = currentPeriodStart - ((windowSize - 1) * periodMs);
                
                // Create a map of existing data points for quick lookup
                const existingDataMap = new Map();
                existingTimeSeries.forEach(point => {
                  const periodStart = Math.floor(point.timestamp / periodMs) * periodMs;
                  existingDataMap.set(periodStart, point);
                });
                
                // Add/update current period data
                existingDataMap.set(currentPeriodStart, currentData);
                
                // Generate complete timeline with zero-filled gaps
                const completeTimeSeries = [];
                for (let i = 0; i < windowSize; i++) {
                  const periodTimestamp = windowStart + (i * periodMs);
                  const existingData = existingDataMap.get(periodTimestamp);
                  
                  completeTimeSeries.push(existingData || {
                    timestamp: periodTimestamp,
                    volume_usd: 0,
                    trade_count: 0
                  });
                }
                
                return completeTimeSeries;
              };
              
              // Update hour/minute mode with complete 60-minute sliding window
              const updatedHourMinuteMode = {
                time_series: createCompleteTimeWindow(
                  prevAnalytics.hour_minute_mode?.time_series || [],
                  incrementalData.current_minute_data,
                  60, // 60 minutes
                  60 * 1000 // 1 minute in milliseconds
                ),
                summary_stats: incrementalData.hour_minute_mode_summary || {}
              };
              
              // Update day/hour mode with complete 24-hour sliding window  
              const updatedDayHourMode = {
                time_series: createCompleteTimeWindow(
                  prevAnalytics.day_hour_mode?.time_series || [],
                  incrementalData.current_hour_data,
                  24, // 24 hours
                  60 * 60 * 1000 // 1 hour in milliseconds
                ),
                summary_stats: incrementalData.day_hour_mode_summary || {}
              };
              
              return {
                hour_minute_mode: updatedHourMinuteMode,
                day_hour_mode: updatedDayHourMode
              };
            });
            
            // Update legacy analyticsSummary for backward compatibility
            setAnalyticsSummary(incrementalData.hour_minute_mode_summary || {});
          } else {
            console.warn('Received incomplete analytics_incremental data');
          }
          break;

        default:
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
    analyticsSummary,
    globalStats,
    connectionStatus,
    error,
    selectTicker,
    getTickerData
  };
};

export default useTradeData;