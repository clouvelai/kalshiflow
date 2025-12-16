import React, { useState, useEffect, useRef } from 'react';
import Layout from '../Layout';
import TraderStatePanel from './TraderStatePanel';
import ActionFeed from './ActionFeed';

const TraderDashboard = () => {
  const [traderState, setTraderState] = useState(null);
  const [actions, setActions] = useState([]);
  const [connectionStatus, setConnectionStatus] = useState('connecting');
  const wsRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);
  const reconnectAttempts = useRef(0);
  const maxReconnectAttempts = 10;
  const reconnectDelay = 2000; // 2 seconds

  const connectWebSocket = () => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    const ws = new WebSocket('ws://localhost:8002/rl/ws');
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('Trader WebSocket connected');
      setConnectionStatus('connected');
      reconnectAttempts.current = 0;
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        switch(data.type) {
          case 'trader_state':
            setTraderState(data.data);
            break;
          case 'trader_action':
            setActions(prev => {
              // Keep only last 100 actions for performance
              const newActions = [data.data, ...prev].slice(0, 100);
              return newActions;
            });
            break;
          case 'connection':
            console.log('Connection established:', data.data);
            setConnectionStatus('connected');
            break;
          case 'orderbook_snapshot':
          case 'orderbook_delta':
            // These are for other purposes, ignore in trader dashboard
            break;
          case 'stats':
            // Could use stats for monitoring, but not needed for MVP
            console.log('Stats update:', data.data);
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
      console.log('Trader WebSocket disconnected');
      setConnectionStatus('disconnected');
      wsRef.current = null;
      
      // Attempt to reconnect
      if (reconnectAttempts.current < maxReconnectAttempts) {
        reconnectAttempts.current++;
        console.log(`Attempting to reconnect... (attempt ${reconnectAttempts.current}/${maxReconnectAttempts})`);
        reconnectTimeoutRef.current = setTimeout(() => {
          connectWebSocket();
        }, reconnectDelay * Math.min(reconnectAttempts.current, 3)); // Exponential backoff up to 3x
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

  // Connection status indicator colors
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
      {/* Custom header for trader dashboard */}
      <header className="bg-gray-800 border-b border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center">
              <h1 className="text-xl font-bold text-white">RL Trader Dashboard</h1>
              <span className="ml-4 text-sm text-gray-400">Real-time Trading Decisions</span>
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <div className={`w-2 h-2 rounded-full ${connectionStatus === 'connected' ? 'bg-green-400' : 'bg-red-400'} animate-pulse`} />
                <span className={`text-sm font-medium ${getConnectionColor()}`}>
                  {getConnectionText()}
                </span>
              </div>
              <a 
                href="/" 
                className="text-sm text-gray-400 hover:text-white transition-colors"
              >
                ← Back to Flowboard
              </a>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {/* Quick Stats Bar */}
        {traderState && (
          <div className="mb-6 bg-gray-800 rounded-lg p-4 shadow-lg">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div>
                <p className="text-gray-400 text-xs">Portfolio Value</p>
                <p className="text-lg font-mono text-white">
                  ${(traderState.portfolio_value || 0).toFixed(2)}
                </p>
              </div>
              <div>
                <p className="text-gray-400 text-xs">Available Cash</p>
                <p className="text-lg font-mono text-white">
                  ${(traderState.cash_balance || 0).toFixed(2)}
                </p>
              </div>
              <div>
                <p className="text-gray-400 text-xs">Open Orders</p>
                <p className="text-lg font-mono text-white">
                  {(traderState.open_orders || []).length}
                </p>
              </div>
              <div>
                <p className="text-gray-400 text-xs">Fill Rate</p>
                <p className="text-lg font-mono text-white">
                  {((traderState.metrics?.fill_rate || 0) * 100).toFixed(1)}%
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left Column - Trader State */}
          <div>
            <TraderStatePanel state={traderState} />
          </div>

          {/* Right Column - Action Feed */}
          <div>
            <ActionFeed actions={actions} />
          </div>
        </div>

        {/* Connection Status Message */}
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

export default TraderDashboard;