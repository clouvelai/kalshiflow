import { useState, useEffect, useCallback, useMemo } from 'react';
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
  
  // Memoize WebSocket URL to prevent recreation on every render
  const wsUrl = useMemo(() => 
    import.meta.env.VITE_WS_URL || `ws://localhost:8000/ws/stream`, 
    []
  );
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
              // Extract the most recent data points for current minute/hour display
              const hourMinuteTimeSeries = lastMessage.data.analytics_data.hour_minute_mode.time_series || [];
              const dayHourTimeSeries = lastMessage.data.analytics_data.day_hour_mode.time_series || [];
              
              // Get current minute and hour data from the latest time series entries
              const latestMinuteData = hourMinuteTimeSeries.length > 0 
                ? hourMinuteTimeSeries[hourMinuteTimeSeries.length - 1] 
                : { timestamp: Date.now(), volume_usd: 0, trade_count: 0 };
              const latestHourData = dayHourTimeSeries.length > 0 
                ? dayHourTimeSeries[dayHourTimeSeries.length - 1] 
                : { timestamp: Date.now(), volume_usd: 0, trade_count: 0 };
              
              setAnalyticsData({
                ...lastMessage.data.analytics_data,
                // Initialize current period data for immediate availability
                current_minute_data: latestMinuteData,
                current_hour_data: latestHourData,
                last_snapshot_update: Date.now()
              });
              
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
            // Extract the most recent data points for current minute/hour display
            const hourMinuteTimeSeries = lastMessage.data.hour_minute_mode.time_series || [];
            const dayHourTimeSeries = lastMessage.data.day_hour_mode.time_series || [];
            
            // Get current minute and hour data from the latest time series entries
            const latestMinuteData = hourMinuteTimeSeries.length > 0 
              ? hourMinuteTimeSeries[hourMinuteTimeSeries.length - 1] 
              : { timestamp: Date.now(), volume_usd: 0, trade_count: 0 };
            const latestHourData = dayHourTimeSeries.length > 0 
              ? dayHourTimeSeries[dayHourTimeSeries.length - 1] 
              : { timestamp: Date.now(), volume_usd: 0, trade_count: 0 };
            
            setAnalyticsData({
              ...lastMessage.data,
              // Always include current period data for component access
              current_minute_data: latestMinuteData,
              current_hour_data: latestHourData,
              last_full_update: Date.now()
            });
            
            // Update legacy analyticsSummary for backward compatibility (use hour mode by default)
            setAnalyticsSummary(lastMessage.data.hour_minute_mode.summary_stats || {});
          } else {
            console.warn('Received analytics_data in old format, this should not happen with new backend');
          }
          break;

        case 'current_minute_fast':
          // ULTRA-FAST: Instant current minute + mode-specific total updates on every trade
          // Provides immediate responsiveness matching trade ticker speed with mode differentiation
          if (lastMessage.data?.volume_usd !== undefined && lastMessage.data?.trade_count !== undefined) {
            const fastData = lastMessage.data;
            
            // Immediately update current minute data in analytics state
            setAnalyticsData(prevAnalytics => ({
              ...prevAnalytics,
              // Update current_minute_data with ultra-fast data
              current_minute_data: {
                timestamp: fastData.timestamp,
                volume_usd: fastData.volume_usd,
                trade_count: fastData.trade_count
              },
              // CRITICAL: Update current_hour_data with ultra-fast data for day mode
              current_hour_data: {
                timestamp: fastData.current_hour_timestamp,
                volume_usd: fastData.current_hour_volume_usd || 0,
                trade_count: fastData.current_hour_trades || 0
              },
              // Mark this as an ultra-fast update for debugging
              last_ultra_fast_update: Date.now(),
              last_trade_ts: fastData.last_trade_ts,
              // CRITICAL FIX: Store MODE-SPECIFIC totals for immediate UI updates
              ultra_fast_totals: {
                // Hour mode totals (60-minute window)
                hour_mode_total_volume_usd: fastData.hour_mode_total_volume_usd || 0,
                hour_mode_total_trades: fastData.hour_mode_total_trades || 0,
                // Day mode totals (24-hour window)
                day_mode_total_volume_usd: fastData.day_mode_total_volume_usd || 0,
                day_mode_total_trades: fastData.day_mode_total_trades || 0,
                last_update: Date.now()
              }
            }));
            
            // Also update the analyticsSummary for legacy components + ultra-fast mode-specific totals
            setAnalyticsSummary(prevSummary => ({
              ...prevSummary,
              current_minute_volume_usd: fastData.volume_usd,
              current_minute_trades: fastData.trade_count,
              // CRITICAL: Add ultra-fast current hour stats for day mode
              current_hour_volume_usd: fastData.current_hour_volume_usd || 0,
              current_hour_trades: fastData.current_hour_trades || 0,
              // CRITICAL FIX: Store mode-specific totals instead of global totals
              hour_mode_total_volume_usd: fastData.hour_mode_total_volume_usd || 0,
              hour_mode_total_trades: fastData.hour_mode_total_trades || 0,
              day_mode_total_volume_usd: fastData.day_mode_total_volume_usd || 0,
              day_mode_total_trades: fastData.day_mode_total_trades || 0,
              last_ultra_fast_update: Date.now()
            }));

            // Ultra-fast update processed successfully
          }
          break;

        case 'analytics_incremental':
          // CRITICAL FIX: analytics_incremental now contains complete time series data
          // This prevents gaps in charts by using the backend's complete time series
          if (lastMessage.data?.hour_minute_mode && lastMessage.data?.day_hour_mode) {
            const incrementalData = lastMessage.data;
            
            // Extract current period data from the complete time series (latest entries)
            const hourMinuteTimeSeries = incrementalData.hour_minute_mode.time_series || [];
            const dayHourTimeSeries = incrementalData.day_hour_mode.time_series || [];
            
            // Get current minute and hour data from the latest time series entries
            const latestMinuteData = hourMinuteTimeSeries.length > 0 
              ? hourMinuteTimeSeries[hourMinuteTimeSeries.length - 1] 
              : { timestamp: Date.now(), volume_usd: 0, trade_count: 0 };
            const latestHourData = dayHourTimeSeries.length > 0 
              ? dayHourTimeSeries[dayHourTimeSeries.length - 1] 
              : { timestamp: Date.now(), volume_usd: 0, trade_count: 0 };
            
            // Use the complete time series directly from backend - no gaps!
            setAnalyticsData(prevAnalytics => ({
              // Use complete time series data directly from backend (no reconstruction needed)
              hour_minute_mode: incrementalData.hour_minute_mode,
              day_hour_mode: incrementalData.day_hour_mode,
              // CLEAN SEPARATION: DO NOT override current period data from ultra-fast messages
              // These are managed by current_minute_fast for ultra-fast responsiveness
              current_minute_data: prevAnalytics.current_minute_data || latestMinuteData,
              current_hour_data: prevAnalytics.current_hour_data || latestHourData,
              // Preserve ultra-fast totals
              ultra_fast_totals: prevAnalytics.ultra_fast_totals,
              // Add update timestamp for debugging/monitoring
              last_incremental_update: Date.now()
            }));
            
            // CLEAN SEPARATION: Update only PEAK and HISTORICAL stats, preserve current values
            setAnalyticsSummary(prevSummary => ({
              ...prevSummary,
              // Update ONLY peak and total stats from analytics_incremental
              peak_volume_usd: incrementalData.hour_minute_mode.summary_stats?.peak_volume_usd || prevSummary.peak_volume_usd || 0,
              peak_trades: incrementalData.hour_minute_mode.summary_stats?.peak_trades || prevSummary.peak_trades || 0,
              // Update total stats from incremental (backend aggregates these from all buckets)
              total_volume_usd: incrementalData.hour_minute_mode.summary_stats?.total_volume_usd || prevSummary.total_volume_usd || 0,
              total_trades: incrementalData.hour_minute_mode.summary_stats?.total_trades || prevSummary.total_trades || 0,
              // DO NOT update current minute/hour values - these are managed by current_minute_fast
              // Preserve current values that are updated by ultra-fast messages
              current_minute_volume_usd: prevSummary.current_minute_volume_usd,
              current_minute_trades: prevSummary.current_minute_trades,
              current_hour_volume_usd: prevSummary.current_hour_volume_usd,
              current_hour_trades: prevSummary.current_hour_trades,
              // Mark this as an incremental update (peaks + totals only)
              last_incremental_update: Date.now()
            }));
          } else {
            console.warn('Received incomplete analytics_incremental data (expected complete time series)', lastMessage.data);
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