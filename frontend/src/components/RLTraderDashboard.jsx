import React, { useState, useEffect, useRef } from 'react';
import TradesFeed from './TradesFeed';
import ExecutionStats from './ExecutionStats';
import CollectionStatus from './CollectionStatus';
import TraderStatePanel from './TraderStatePanel';

const RLTraderDashboard = () => {
  const [connectionStatus, setConnectionStatus] = useState('connecting');
  const [collectionStatus, setCollectionStatus] = useState(null);
  const [traderState, setTraderState] = useState(null);
  const [recentFills, setRecentFills] = useState([]);
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

    // Connect to RL trader WebSocket on port 8002
    const ws = new WebSocket('ws://localhost:8002/rl/ws');
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
            setTraderState(data.data);
            break;
            
          case 'trades':
            // New comprehensive trades message with observation space
            if (data.data) {
              const tradeData = data.data;
              
              // Update recent fills
              if (tradeData.recent_fills) {
                setRecentFills(tradeData.recent_fills);
              }
              
              // Update execution statistics
              if (tradeData.execution_stats) {
                setExecutionStats(tradeData.execution_stats);
              }
            }
            break;
            
          case 'trader_action':
            // Individual trader action (legacy support)
            if (data.data) {
              setRecentFills(prev => {
                const newFills = [{
                  timestamp: new Date().toISOString(),
                  ...data.data
                }, ...prev].slice(0, 20);
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

          {/* Middle Panel: Trader State */}
          <div className="xl:col-span-1">
            <div className="bg-gray-800 rounded-lg shadow-lg h-full">
              <div className="p-6">
                <h2 className="text-lg font-semibold mb-4 text-gray-100">Trader State</h2>
                <TraderStatePanel state={traderState} />
              </div>
            </div>
          </div>

          {/* Right Panel: Execution Stats + Recent Fills */}
          <div className="xl:col-span-1 space-y-6">
            {/* Execution Statistics */}
            <div className="bg-gray-800 rounded-lg p-6 shadow-lg">
              <h2 className="text-lg font-semibold mb-4 text-gray-100">Execution Stats</h2>
              <ExecutionStats stats={executionStats} />
            </div>

            {/* Recent Fills with integrated observation space */}
            <div className="bg-gray-800 rounded-lg p-6 shadow-lg">
              <h2 className="text-lg font-semibold mb-4 text-gray-100">Recent Fills</h2>
              <TradesFeed fills={recentFills} />
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