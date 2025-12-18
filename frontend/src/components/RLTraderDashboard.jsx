import React, { useState, useEffect, useRef } from 'react';
import TradesFeed from './TradesFeed';
import CollectionStatus from './CollectionStatus';
import TraderStatePanel from './TraderStatePanel';

const RLTraderDashboard = () => {
  const [connectionStatus, setConnectionStatus] = useState('connecting');
  const [collectionStatus, setCollectionStatus] = useState(null);
  const [traderState, setTraderState] = useState(null);
  const [recentFills, setRecentFills] = useState([]); // Recent trading actions from RL agent
  const [actualFills, setActualFills] = useState([]); // Actual executed fills from Kalshi
  const [executionStats, setExecutionStats] = useState(null);
  const [tradingMode, setTradingMode] = useState('paper'); // paper or production
  const [apiUrls, setApiUrls] = useState(null); // Store API URLs from connection
  
  const wsRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);
  const reconnectAttempts = useRef(0);
  const maxReconnectAttempts = 10;
  const reconnectDelay = 2000;

  const connectWebSocket = () => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    // Connect to RL trader WebSocket on port 8003 (paper trading)
    const ws = new WebSocket('ws://localhost:8003/rl/ws');
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('RL Trader WebSocket connected');
      setConnectionStatus('connected');
      reconnectAttempts.current = 0;
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        switch(data.type) {
          case 'connection':
            console.log('Connection established:', data.data);
            setConnectionStatus('connected');
            
            // Extract API URLs from connection message
            if (data.data?.kalshi_api_url || data.data?.kalshi_ws_url) {
              setApiUrls({
                kalshi_api_url: data.data.kalshi_api_url,
                kalshi_ws_url: data.data.kalshi_ws_url
              });
              
              // Determine trading mode from API URLs
              if (data.data.kalshi_api_url?.includes('demo-api.kalshi.co')) {
                setTradingMode('paper');
              } else if (data.data.kalshi_api_url?.includes('api.elections.kalshi.com')) {
                setTradingMode('production');
              }
            }
            
            if (data.data?.markets) {
              setCollectionStatus({
                markets: data.data.markets,
                status: 'active'
              });
            }
            break;
            
          case 'stats':
            // Collection service statistics
            console.log('Stats message received:', data.data);
            if (data.data) {
              setCollectionStatus(prev => ({
                ...prev,
                ...data.data,
                lastUpdate: new Date().toISOString()
              }));
            }
            break;
            
          case 'trader_state':
            // Trader state update from backend
            // Enhance the trader state with additional metadata
            if (data.data) {
              const enhancedState = {
                ...data.data,
                // Ensure positions have all required fields
                positions: data.data.positions ? 
                  Object.entries(data.data.positions).reduce((acc, [ticker, position]) => {
                    acc[ticker] = {
                      ...position,
                      ticker: ticker,
                      current_price: position.current_price || position.last_price || 0,
                      avg_price: position.avg_price || position.entry_price || 0,
                      unrealized_pnl: position.unrealized_pnl || 0,
                      contracts: position.contracts || position.quantity || 0,
                      side: position.side || 'YES'
                    };
                    return acc;
                  }, {}) : {},
                // Ensure open orders have all required fields  
                open_orders: data.data.open_orders ? 
                  data.data.open_orders.map(order => ({
                    ...order,
                    ticker: order.ticker || order.market_ticker || 'Unknown',
                    order_type: order.order_type || order.type || 'LIMIT',
                    created_at: order.created_at || new Date().toISOString(),
                    current_price: order.current_price || order.market_price,
                    quantity: order.quantity || order.size || 0,
                    price: order.price || order.limit_price || 0,
                    side: order.side || 'BUY'
                  })) : []
              };
              setTraderState(enhancedState);
            } else {
              setTraderState(data.data);
            }
            break;
            
          case 'trades':
            // New comprehensive trades message with observation space
            if (data.data) {
              const tradeData = data.data;
              
              // Update recent fills with enhanced metadata
              if (tradeData.recent_fills) {
                const enhancedFills = tradeData.recent_fills.map(fill => ({
                  ...fill,
                  timestamp: fill.timestamp || new Date().toISOString(),
                  market_ticker: fill.market_ticker || fill.ticker || 'Unknown',
                  action: fill.action || { action_name: 'UNKNOWN' },
                  execution_result: fill.execution_result || { executed: false, status: 'unknown' }
                }));
                setRecentFills(enhancedFills);
              }
              
              // Update execution statistics with additional calculations
              if (tradeData.execution_stats) {
                const enhancedStats = {
                  ...tradeData.execution_stats,
                  total_fills: tradeData.execution_stats.total_fills || 0,
                  maker_fills: tradeData.execution_stats.maker_fills || 0,
                  taker_fills: tradeData.execution_stats.taker_fills || 0,
                  win_rate: tradeData.execution_stats.win_rate || 0,
                  success_rate: tradeData.execution_stats.success_rate || 0,
                  total_pnl: tradeData.execution_stats.total_pnl || 0
                };
                setExecutionStats(enhancedStats);
              }
              
              // Also update trader state if it's included
              if (tradeData.trader_state) {
                const enhancedState = {
                  ...tradeData.trader_state,
                  positions: tradeData.trader_state.positions ? 
                    Object.entries(tradeData.trader_state.positions).reduce((acc, [ticker, position]) => {
                      acc[ticker] = {
                        ...position,
                        ticker: ticker,
                        current_price: position.current_price || position.last_price || 0,
                        avg_price: position.avg_price || position.entry_price || 0,
                        unrealized_pnl: position.unrealized_pnl || 0,
                        contracts: position.contracts || position.quantity || 0,
                        side: position.side || 'YES'
                      };
                      return acc;
                    }, {}) : {},
                  open_orders: tradeData.trader_state.open_orders ? 
                    tradeData.trader_state.open_orders.map(order => ({
                      ...order,
                      ticker: order.ticker || order.market_ticker || 'Unknown',
                      order_type: order.order_type || order.type || 'LIMIT',
                      created_at: order.created_at || new Date().toISOString(),
                      current_price: order.current_price || order.market_price,
                      quantity: order.quantity || order.size || 0,
                      price: order.price || order.limit_price || 0,
                      side: order.side || 'BUY'
                    })) : []
                };
                setTraderState(enhancedState);
              }
            }
            break;
            
          case 'trader_action':
            // Individual trader action (legacy support)
            if (data.data) {
              setRecentFills(prev => {
                const enhancedFill = {
                  timestamp: data.data.timestamp || new Date().toISOString(),
                  market_ticker: data.data.market_ticker || data.data.ticker || 'Unknown',
                  action: data.data.action || { action_name: 'UNKNOWN' },
                  execution_result: data.data.execution_result || { executed: false, status: 'unknown' },
                  ...data.data
                };
                const newFills = [enhancedFill, ...prev].slice(0, 20);
                return newFills;
              });
            }
            break;
            
          case 'order_fill':
            // Actual order fill event from Kalshi
            if (data.data) {
              setActualFills(prev => {
                const fillEvent = {
                  timestamp: data.data.timestamp || new Date().toISOString(),
                  market_ticker: data.data.ticker || 'Unknown',
                  side: data.data.side || 'BUY',
                  quantity: data.data.quantity || 0,
                  price: data.data.price || 0,
                  fill_id: data.data.fill_id || 'unknown',
                  order_id: data.data.order_id || 'unknown',
                  ...data.data
                };
                const newFills = [fillEvent, ...prev].slice(0, 15);
                return newFills;
              });
            }
            break;
            
          case 'orderbook_snapshot':
          case 'orderbook_delta':
            // These are for data collection, not needed for UI
            break;
            
          default:
            console.log('Unknown message type:', data.type);
        }
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setConnectionStatus('error');
    };

    ws.onclose = () => {
      console.log('RL Trader WebSocket disconnected');
      setConnectionStatus('disconnected');
      wsRef.current = null;
      
      // Attempt to reconnect
      if (reconnectAttempts.current < maxReconnectAttempts) {
        reconnectAttempts.current++;
        console.log(`Attempting to reconnect... (attempt ${reconnectAttempts.current}/${maxReconnectAttempts})`);
        reconnectTimeoutRef.current = setTimeout(() => {
          connectWebSocket();
        }, reconnectDelay * Math.min(reconnectAttempts.current, 3));
      }
    };
  };

  useEffect(() => {
    connectWebSocket();

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, []);

  // Connection status helper functions
  const getConnectionColor = () => {
    switch (connectionStatus) {
      case 'connected':
        return 'text-green-400';
      case 'connecting':
        return 'text-yellow-400';
      case 'disconnected':
      case 'error':
        return 'text-red-400';
      default:
        return 'text-gray-400';
    }
  };

  const getConnectionText = () => {
    switch (connectionStatus) {
      case 'connected':
        return 'Live';
      case 'connecting':
        return 'Connecting...';
      case 'disconnected':
        return 'Disconnected';
      case 'error':
        return 'Error';
      default:
        return 'Unknown';
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Header */}
      <header className="bg-gray-800 border-b border-gray-700">
        <div className="max-w-full px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-4">
              <h1 className="text-xl font-bold text-white flex items-center">
                <span className="text-2xl mr-2">🤖</span>
                RL Trader v1.0
              </h1>
              <div className="flex items-center space-x-2">
                <span className={`px-3 py-1 rounded text-xs font-medium ${
                  tradingMode === 'paper' 
                    ? 'bg-blue-500/20 text-blue-400 border border-blue-500/50' 
                    : 'bg-orange-500/20 text-orange-400 border border-orange-500/50'
                }`}>
                  {tradingMode === 'paper' ? 'Paper Trading' : 'Production'}
                </span>
                <div className="flex items-center space-x-2">
                  <div className={`w-2 h-2 rounded-full ${
                    connectionStatus === 'connected' ? 'bg-green-400' : 'bg-red-400'
                  } animate-pulse`} />
                  <span className={`text-sm font-medium ${getConnectionColor()}`}>
                    Status: {getConnectionText()}
                  </span>
                </div>
              </div>
            </div>
            <a 
              href="/" 
              className="text-sm text-gray-400 hover:text-white transition-colors"
            >
              ← Back to Flowboard
            </a>
          </div>
        </div>
      </header>

      {/* Three-Panel Layout */}
      <main className="max-w-full px-4 sm:px-6 lg:px-8 py-6">
        <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
          {/* Left Panel: Collection Status */}
          <div className="xl:col-span-1">
            <div className="bg-gray-800 rounded-lg p-6 shadow-lg h-full">
              <h2 className="text-lg font-semibold mb-4 text-gray-100">Collection Status</h2>
              <CollectionStatus status={collectionStatus} apiUrls={apiUrls} />
            </div>
          </div>

          {/* Middle Panel: Trader State (Without Execution Stats) */}
          <div className="xl:col-span-1">
            <div className="bg-gray-800 rounded-lg shadow-lg h-full">
              <div className="p-6">
                <h2 className="text-lg font-semibold mb-4 text-gray-100">Trader State</h2>
                <TraderStatePanel 
                  state={traderState} 
                  executionStats={executionStats}
                  showExecutionStats={false}
                />
              </div>
            </div>
          </div>

          {/* Right Panel: Recent Fills & Recent Trades */}
          <div className="xl:col-span-1 space-y-6">
            {/* Recent Fills (Actual Fills) */}
            <div className="bg-gray-800 rounded-lg p-6 shadow-lg">
              <h2 className="text-lg font-semibold mb-4 text-gray-100">Recent Fills</h2>
              {actualFills && actualFills.length > 0 ? (
                <TradesFeed fills={actualFills} maxItems={15} />
              ) : (
                <div className="text-center text-gray-500 py-8">
                  <div className="mb-2">💸</div>
                  <div>No recent fills</div>
                  <div className="text-xs mt-1 text-gray-600">Executed orders will appear here</div>
                </div>
              )}
            </div>
            
            {/* Recent Trades (Recent Actions) */}
            <div className="bg-gray-800 rounded-lg p-6 shadow-lg">
              <h2 className="text-lg font-semibold mb-4 text-gray-100">Recent Trades</h2>
              <TradesFeed fills={recentFills} maxItems={30} />
            </div>
          </div>
        </div>

        {/* Connection Status Message (if disconnected) */}
        {connectionStatus !== 'connected' && (
          <div className="fixed bottom-4 right-4 bg-gray-800 text-white px-4 py-3 rounded-lg shadow-lg border border-gray-700">
            <div className="flex items-center space-x-2">
              <div className={`w-2 h-2 rounded-full ${
                connectionStatus === 'connecting' ? 'bg-yellow-400' : 'bg-red-400'
              } animate-pulse`} />
              <p className="text-sm">
                {connectionStatus === 'connecting' 
                  ? 'Connecting to RL trader service...' 
                  : 'Connection lost. Attempting to reconnect...'}
              </p>
            </div>
            {reconnectAttempts.current > 0 && (
              <p className="text-xs text-gray-400 mt-1">
                Reconnection attempt {reconnectAttempts.current}/{maxReconnectAttempts}
              </p>
            )}
          </div>
        )}
      </main>
    </div>
  );
};

export default RLTraderDashboard;