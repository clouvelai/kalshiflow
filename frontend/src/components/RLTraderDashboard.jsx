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
  const [openOrders, setOpenOrders] = useState([]); // Track open orders
  const [positions, setPositions] = useState({}); // Track positions
  
  // Collapsible state for grid sections
  const [collapsedSections, setCollapsedSections] = useState({
    orders: false,
    fills: false,
    positions: false,
    trades: false
  });
  
  const wsRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);
  const reconnectAttempts = useRef(0);
  const maxReconnectAttempts = 10;
  const reconnectDelay = 2000;

  const toggleSection = (section) => {
    setCollapsedSections(prev => ({
      ...prev,
      [section]: !prev[section]
    }));
  };

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
            
            // Initialize collection status as active when connected
            if (!collectionStatus) {
              setCollectionStatus({
                session_active: true,
                snapshots_processed: 0,
                deltas_processed: 0,
                uptime_seconds: 0
              });
            }
            
            // Extract API URLs from connection message
            if (data.data?.kalshi_api_url || data.data?.kalshi_ws_url) {
              setApiUrls({
                kalshi_api_url: data.data.kalshi_api_url,
                kalshi_ws_url: data.data.kalshi_ws_url
              });
              
              // Determine trading mode from API URL
              if (data.data.kalshi_api_url?.includes('demo-api.kalshi.co')) {
                setTradingMode('paper');
              } else if (data.data.kalshi_api_url?.includes('api.elections.kalshi.com')) {
                setTradingMode('production');
              }
            }
            break;
            
          case 'stats':
            // Legacy stats format (pre-throttling)
            if (data.stats) {
              setCollectionStatus({
                ...data.stats,
                session_active: data.stats.session_active !== undefined ? data.stats.session_active : true
              });
            }
            break;
            
          case 'trader_state':
            // Combined trader state with all info
            if (data.data) {
              setTraderState(data.data);
              // Extract positions and orders from state
              if (data.data.positions) {
                setPositions(data.data.positions);
              }
              if (data.data.open_orders) {
                setOpenOrders(data.data.open_orders);
              }
            }
            break;
            
          case 'trades':
            // Trade broadcast - only update execution stats, not fills
            // (fills are handled by individual trader_action messages to avoid duplicates)
            if (data.data) {
              if (data.data.execution_stats) {
                setExecutionStats(data.data.execution_stats);
              }
              // DON'T set recent_fills here - handled by trader_action messages
            }
            break;
            
          case 'trader_action':
            // Individual trader action from the RL agent
            if (data.data) {
              // Add the new trader action to recent fills
              setRecentFills(prev => {
                // Create a properly formatted fill entry
                const actionEntry = {
                  timestamp: data.data.timestamp || new Date().toISOString(),
                  market_ticker: data.data.market_ticker,
                  action: data.data.action,
                  observation: data.data.observation,
                  execution_result: data.data.execution_result,
                  success: data.data.execution_result?.executed !== false,
                  ...data.data
                };
                // Add to beginning and limit to 100 entries
                return [actionEntry, ...prev].slice(0, 100);
              });
            }
            break;
            
          case 'sync_complete':
            // Sync complete notification with updated state
            if (data.data) {
              console.log('Order sync complete:', data.data);
              if (data.data.cash_balance !== undefined) {
                setTraderState(prev => ({
                  ...prev,
                  cash_balance: data.data.cash_balance
                }));
              }
              if (data.data.open_orders) {
                setOpenOrders(data.data.open_orders);
                setTraderState(prev => ({
                  ...prev,
                  open_orders: data.data.open_orders
                }));
              }
            }
            break;
            
          case 'position_update':
            // Position update from fills or sync
            if (data.data) {
              const { ticker, position, cost_basis, realized_pnl, market_exposure } = data.data;
              if (ticker) {
                setPositions(prev => ({
                  ...prev,
                  [ticker]: { position, cost_basis, realized_pnl, market_exposure }
                }));
                setTraderState(prev => ({
                  ...prev,
                  positions: {
                    ...prev.positions,
                    [ticker]: { position, cost_basis, realized_pnl, market_exposure }
                  }
                }));
              }
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
            
          // New standardized message types
          case 'stats_update':
            // Stats update (throttled to 10s)
            console.log('Stats update received:', data.data);
            if (data.data) {
              // Update collection status with the new stats format
              const statsData = data.data.stats || data.data;
              setCollectionStatus(prev => ({
                ...prev,
                ...statsData,
                session_active: statsData.session_active !== undefined ? statsData.session_active : (prev?.session_active ?? true),
                lastUpdate: new Date().toISOString()
              }));
            }
            break;
            
          case 'orders_update':
            // Orders update from API sync or WebSocket
            if (data.data && data.data.orders) {
              setOpenOrders(data.data.orders);
              setTraderState(prev => ({
                ...prev,
                open_orders: data.data.orders
              }));
            }
            break;
            
          case 'positions_update':
            // Positions update from API sync or WebSocket
            if (data.data && data.data.positions) {
              setPositions(data.data.positions);
              setTraderState(prev => ({
                ...prev,
                positions: data.data.positions,
                portfolio_value: data.data.total_value || prev.portfolio_value
              }));
            }
            break;
            
          case 'portfolio_update':
            // Portfolio/balance update
            if (data.data) {
              setTraderState(prev => ({
                ...prev,
                cash_balance: data.data.cash_balance || prev.cash_balance,
                portfolio_value: data.data.portfolio_value || prev.portfolio_value
              }));
            }
            break;
            
          case 'fill_event':
            // Fill notification with updated position
            if (data.data) {
              // Add to actual fills list
              if (data.data.fill) {
                setActualFills(prev => {
                  const fillEvent = {
                    timestamp: data.data.timestamp || new Date().toISOString(),
                    ...data.data.fill
                  };
                  const newFills = [fillEvent, ...prev].slice(0, 15);
                  return newFills;
                });
              }
              
              // Update position if provided
              if (data.data.updated_position) {
                setPositions(prev => ({
                  ...prev,
                  [data.data.updated_position.ticker]: data.data.updated_position
                }));
                setTraderState(prev => ({
                  ...prev,
                  positions: {
                    ...prev.positions,
                    [data.data.updated_position.ticker]: data.data.updated_position
                  }
                }));
              }
            }
            break;
            
          case 'ping':
            // Heartbeat/ping message - ignore silently
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

  // Format positions for display
  const formatPositions = () => {
    if (!positions || Object.keys(positions).length === 0) return [];
    
    return Object.entries(positions).map(([ticker, pos]) => ({
      ticker,
      contracts: pos.position || pos.contracts || 0,
      side: (pos.position || pos.contracts || 0) > 0 ? 'YES' : 'NO',
      costBasis: pos.cost_basis || 0,
      realizedPnl: pos.realized_pnl || 0,
      marketExposure: pos.market_exposure || Math.abs((pos.position || pos.contracts || 0) * (pos.cost_basis || 0) / 100),
      totalTraded: pos.total_traded,
      feesPaid: pos.fees_paid || 0,
      lastUpdated: pos.last_updated_ts
    })).filter(p => p.contracts !== 0);
  };

  // Helper to format relative time
  const formatRelativeTime = (timestamp) => {
    if (!timestamp) return '--';
    const now = Date.now();
    const time = new Date(timestamp).getTime();
    const diff = Math.floor((now - time) / 1000);
    
    if (diff < 60) return `${diff}s ago`;
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
    return `${Math.floor(diff / 86400)}d ago`;
  };

  // Helper to format uptime
  const formatUptime = (seconds) => {
    if (!seconds) return '--';
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    
    if (hours > 0) return `${hours}h ${minutes}m`;
    if (minutes > 0) return `${minutes}m ${secs}s`;
    return `${secs}s`;
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Header */}
      <header className="bg-gray-800 border-b border-gray-700">
        <div className="max-w-full px-4 sm:px-6 lg:px-8">
          {/* Main Header Row */}
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
                    {getConnectionText()}
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
          
          {/* Collection Status Bar */}
          {collectionStatus && (
            <div className="border-t border-gray-700 py-2">
              <div className="flex items-center justify-between flex-wrap gap-2">
                {/* Left side - Status indicators */}
                <div className="flex items-center space-x-4 text-xs">
                  {/* Status */}
                  <div className="flex items-center space-x-1">
                    <span className="text-gray-500">Status:</span>
                    <span className={`font-medium ${
                      collectionStatus.session_active ? 'text-green-400' : 'text-gray-400'
                    }`}>
                      {collectionStatus.session_active ? 'ACTIVE' : 'INACTIVE'}
                    </span>
                  </div>
                  
                  {/* Uptime */}
                  <div className="flex items-center space-x-1">
                    <span className="text-gray-500">Uptime:</span>
                    <span className="text-gray-300 font-mono">
                      {formatUptime(collectionStatus.uptime_seconds)}
                    </span>
                  </div>
                  
                  {/* Snapshots */}
                  <div className="flex items-center space-x-1">
                    <span className="text-gray-500">Snapshots:</span>
                    <span className="text-blue-400 font-mono">
                      {collectionStatus.snapshots_processed || 0}
                    </span>
                  </div>
                  
                  {/* Deltas */}
                  <div className="flex items-center space-x-1">
                    <span className="text-gray-500">Deltas:</span>
                    <span className="text-purple-400 font-mono">
                      {collectionStatus.deltas_processed || 0}
                    </span>
                  </div>
                </div>
                
                {/* Right side - API endpoint and last update */}
                <div className="flex items-center space-x-4 text-xs">
                  {/* API Endpoint */}
                  {apiUrls?.kalshi_api_url && (
                    <div className="flex items-center space-x-1">
                      <span className="text-gray-500">API:</span>
                      <span className="text-gray-400 font-mono truncate max-w-xs">
                        {apiUrls.kalshi_api_url.replace('https://', '').replace('/trade-api/v2', '')}
                      </span>
                    </div>
                  )}
                  
                  {/* Last Updated */}
                  <div className="flex items-center space-x-1">
                    <span className="text-gray-500">Updated:</span>
                    <span className="text-gray-400">
                      {formatRelativeTime(collectionStatus.lastUpdate || collectionStatus.last_update_time)}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </header>

      {/* Main Content - Full Width Stacked Layout */}
      <main className="max-w-full px-4 sm:px-6 lg:px-8 py-6 space-y-6">
        
        {/* Trader State - Full Width with Subsections */}
        <div className="bg-gray-800 rounded-lg shadow-lg">
          <div className="p-6">
            <h2 className="text-lg font-semibold mb-4 text-gray-100">Trader State</h2>
            
            {/* Portfolio Stats & Action Breakdown */}
            <TraderStatePanel 
              state={traderState} 
              executionStats={executionStats}
              showExecutionStats={false}
              showPositions={false}
              showOrders={false}
              showActionBreakdown={true}
            />
            
            {/* Grid View Components - Ordered Chronologically: Trade → Order → Fill → Position */}
            <div className="mt-6 space-y-4">
              
              {/* Recent Trades Section - FIRST: AI Decisions */}
              <div className="border border-gray-700 rounded-lg">
                <div 
                  className="flex items-center justify-between p-4 cursor-pointer hover:bg-gray-700/50 transition-colors"
                  onClick={() => toggleSection('trades')}
                >
                  <h3 className="text-md font-medium text-gray-200">
                    🤖 Recent Trades {recentFills && recentFills.length > 0 && `(${recentFills.length})`}
                  </h3>
                  <span className="text-gray-400">
                    {collapsedSections.trades ? '▶' : '▼'}
                  </span>
                </div>
                {!collapsedSections.trades && (
                  <div className="border-t border-gray-700 p-4 max-h-64 overflow-y-auto">
                    <TradesFeed fills={recentFills} maxItems={100} />
                  </div>
                )}
              </div>

              {/* Open Orders Section - SECOND: Pending Orders */}
              <div className="border border-gray-700 rounded-lg">
                <div 
                  className="flex items-center justify-between p-4 cursor-pointer hover:bg-gray-700/50 transition-colors"
                  onClick={() => toggleSection('orders')}
                >
                  <h3 className="text-md font-medium text-gray-200">
                    📋 Open Orders {openOrders && openOrders.length > 0 && `(${openOrders.length})`}
                  </h3>
                  <span className="text-gray-400">
                    {collapsedSections.orders ? '▶' : '▼'}
                  </span>
                </div>
                {!collapsedSections.orders && (
                  <div className="border-t border-gray-700 p-4 max-h-96 overflow-y-auto">
                    {openOrders && openOrders.length > 0 ? (
                      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
                        {openOrders.map((order, idx) => (
                          <div key={order.order_id || idx} className="bg-gray-700/30 hover:bg-gray-700/50 rounded-lg p-3 border border-gray-700 hover:border-gray-600 transition-all">
                            {/* Header with ticker and side */}
                            <div className="flex justify-between items-center mb-2 pb-2 border-b border-gray-700/50">
                              <span className="font-mono text-xs text-gray-300 truncate flex-1 mr-2" title={order.ticker}>
                                {order.ticker}
                              </span>
                              <span className={`px-2 py-1 rounded text-xs font-medium ${
                                order.side === 'BUY' && order.contract_side === 'YES' ? 'bg-green-500/20 text-green-400' : 
                                order.side === 'BUY' && order.contract_side === 'NO' ? 'bg-orange-500/20 text-orange-400' :
                                order.side === 'SELL' && order.contract_side === 'YES' ? 'bg-red-500/20 text-red-400' :
                                'bg-purple-500/20 text-purple-400'
                              }`}>
                                {order.side === 'BUY' ? 'BUY' : 'SELL'} {order.contract_side}
                              </span>
                            </div>
                            
                            {/* Order details grid */}
                            <div className="space-y-2 text-xs">
                              <div className="grid grid-cols-2 gap-2">
                                <div>
                                  <span className="text-gray-500 block">Contracts</span>
                                  <span className="text-gray-200 font-mono">{order.quantity}</span>
                                </div>
                                <div>
                                  <span className="text-gray-500 block">Price</span>
                                  <span className="text-gray-200 font-mono">{order.limit_price}¢</span>
                                </div>
                              </div>
                              
                              <div className="grid grid-cols-2 gap-2">
                                <div>
                                  <span className="text-gray-500 block">Value</span>
                                  <span className="text-gray-200 font-mono">${((order.quantity * order.limit_price) / 100).toFixed(2)}</span>
                                </div>
                                <div>
                                  <span className="text-gray-500 block">Status</span>
                                  <span className="text-gray-300 font-mono">{order.status || 'OPEN'}</span>
                                </div>
                              </div>
                              
                              {/* Time placed */}
                              <div className="pt-2 border-t border-gray-700/30">
                                <span className="text-gray-500 block">Placed</span>
                                <span className="text-gray-300 text-xs">
                                  {new Date(order.placed_at * 1000).toLocaleTimeString()}
                                </span>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className="text-center text-gray-500 py-8">
                        <div className="mb-2">📭</div>
                        <div>No open orders</div>
                        <div className="text-xs mt-1 text-gray-600">Orders will appear here when placed</div>
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* Recent Fills Section - THIRD: Executed Orders */}
              <div className="border border-gray-700 rounded-lg">
                <div 
                  className="flex items-center justify-between p-4 cursor-pointer hover:bg-gray-700/50 transition-colors"
                  onClick={() => toggleSection('fills')}
                >
                  <h3 className="text-md font-medium text-gray-200">
                    💸 Recent Fills {actualFills && actualFills.length > 0 && `(${actualFills.length})`}
                  </h3>
                  <span className="text-gray-400">
                    {collapsedSections.fills ? '▶' : '▼'}
                  </span>
                </div>
                {!collapsedSections.fills && (
                  <div className="border-t border-gray-700 p-4 max-h-96 overflow-y-auto">
                    {actualFills && actualFills.length > 0 ? (
                      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
                        {actualFills.slice(0, 50).map((fill, idx) => (
                          <div key={fill.fill_id || idx} className="bg-gray-700/30 hover:bg-gray-700/50 rounded-lg p-3 border border-gray-700 hover:border-gray-600 transition-all">
                            {/* Header with ticker and side */}
                            <div className="flex justify-between items-center mb-2 pb-2 border-b border-gray-700/50">
                              <span className="font-mono text-xs text-gray-300 truncate flex-1 mr-2" title={fill.market_ticker || fill.ticker}>
                                {fill.market_ticker || fill.ticker || 'Unknown'}
                              </span>
                              <span className={`px-2 py-1 rounded text-xs font-medium ${
                                fill.side === 'BUY' ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
                              }`}>
                                {fill.side || 'BUY'}
                              </span>
                            </div>
                            
                            {/* Fill details grid */}
                            <div className="space-y-2 text-xs">
                              <div className="grid grid-cols-2 gap-2">
                                <div>
                                  <span className="text-gray-500 block">Contracts</span>
                                  <span className="text-gray-200 font-mono">{fill.quantity || 0}</span>
                                </div>
                                <div>
                                  <span className="text-gray-500 block">Price</span>
                                  <span className="text-gray-200 font-mono">{fill.price || 0}¢</span>
                                </div>
                              </div>
                              
                              <div className="grid grid-cols-2 gap-2">
                                <div>
                                  <span className="text-gray-500 block">Value</span>
                                  <span className="text-gray-200 font-mono">${(((fill.quantity || 0) * (fill.price || 0)) / 100).toFixed(2)}</span>
                                </div>
                                <div>
                                  <span className="text-gray-500 block">Type</span>
                                  <span className="text-gray-300 font-mono">{fill.type || 'FILL'}</span>
                                </div>
                              </div>
                              
                              {/* Time executed */}
                              <div className="pt-2 border-t border-gray-700/30">
                                <span className="text-gray-500 block">Executed</span>
                                <span className="text-gray-300 text-xs">
                                  {new Date(fill.timestamp).toLocaleTimeString()}
                                </span>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className="text-center text-gray-500 py-8">
                        <div className="mb-2">💸</div>
                        <div>No recent fills</div>
                        <div className="text-xs mt-1 text-gray-600">Executed orders will appear here</div>
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* Open Positions Section - FOURTH: Accumulated Positions */}
              <div className="border border-gray-700 rounded-lg">
                <div 
                  className="flex items-center justify-between p-4 cursor-pointer hover:bg-gray-700/50 transition-colors"
                  onClick={() => toggleSection('positions')}
                >
                  <h3 className="text-md font-medium text-gray-200">
                    📊 Open Positions {formatPositions().length > 0 && `(${formatPositions().length})`}
                  </h3>
                  <span className="text-gray-400">
                    {collapsedSections.positions ? '▶' : '▼'}
                  </span>
                </div>
                {!collapsedSections.positions && (
                  <div className="border-t border-gray-700 p-4 max-h-96 overflow-y-auto">
                    {formatPositions().length > 0 ? (
                      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
                        {formatPositions().map((pos, idx) => (
                          <div key={`${pos.ticker}-${idx}`} className="bg-gray-700/30 hover:bg-gray-700/50 rounded-lg p-3 border border-gray-700 hover:border-gray-600 transition-all">
                            {/* Header with ticker and side */}
                            <div className="flex justify-between items-center mb-2 pb-2 border-b border-gray-700/50">
                              <span className="font-mono text-xs text-gray-300 truncate flex-1 mr-2" title={pos.ticker}>
                                {pos.ticker}
                              </span>
                              <span className={`px-2 py-1 rounded text-xs font-medium ${
                                pos.side === 'YES' ? 'bg-green-500/20 text-green-400' : 'bg-orange-500/20 text-orange-400'
                              }`}>
                                {pos.side}
                              </span>
                            </div>
                            
                            {/* Position details grid */}
                            <div className="space-y-2 text-xs">
                              <div className="grid grid-cols-2 gap-2">
                                <div>
                                  <span className="text-gray-500 block">Contracts</span>
                                  <span className="text-gray-200 font-mono">{Math.abs(pos.contracts)}</span>
                                </div>
                                <div>
                                  <span className="text-gray-500 block">Cost</span>
                                  <span className="text-gray-200 font-mono">${(pos.costBasis / 100).toFixed(2)}</span>
                                </div>
                              </div>
                              
                              <div className="grid grid-cols-2 gap-2">
                                <div>
                                  <span className="text-gray-500 block">Market Exp</span>
                                  <span className="text-gray-200 font-mono">${(pos.marketExposure).toFixed(2)}</span>
                                </div>
                                <div>
                                  <span className="text-gray-500 block">P&L</span>
                                  <span className={`font-mono font-medium ${
                                    pos.realizedPnl >= 0 ? 'text-green-400' : 'text-red-400'
                                  }`}>
                                    {pos.realizedPnl >= 0 ? '+' : ''}${(pos.realizedPnl / 100).toFixed(2)}
                                  </span>
                                </div>
                              </div>
                              
                              {/* Additional data if available */}
                              {pos.totalTraded !== undefined && (
                                <div className="pt-2 border-t border-gray-700/30">
                                  <div className="grid grid-cols-2 gap-2">
                                    <div>
                                      <span className="text-gray-500 block">Traded</span>
                                      <span className="text-gray-300 font-mono">${(pos.totalTraded / 100).toFixed(2)}</span>
                                    </div>
                                    <div>
                                      <span className="text-gray-500 block">Fees</span>
                                      <span className="text-gray-300 font-mono">${(pos.feesPaid / 100).toFixed(2)}</span>
                                    </div>
                                  </div>
                                </div>
                              )}
                            </div>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className="text-center text-gray-500 py-8">
                        <div className="mb-2">📊</div>
                        <div>No open positions</div>
                        <div className="text-xs mt-1 text-gray-600">Positions will appear here when opened</div>
                      </div>
                    )}
                  </div>
                )}
              </div>

              
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