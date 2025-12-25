import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Activity, Wifi, WifiOff, Circle, ChevronRight, ChevronDown, Zap, Database, TrendingUp, AlertCircle, Copy, Check, Info, CheckCircle, XCircle, ArrowRight, DollarSign, Briefcase, ShoppingCart, FileText, TrendingDown, Clock } from 'lucide-react';

// TradingData Component - Displays real-time trading state
const TradingData = ({ tradingState }) => {
  if (!tradingState || !tradingState.has_state) {
    return (
      <div className="bg-gray-900/50 backdrop-blur-sm rounded-xl border border-gray-800 p-4">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-bold text-gray-300 uppercase tracking-wider">Trading Data</h3>
          <span className="text-xs text-gray-500 font-mono">No data available</span>
        </div>
      </div>
    );
  }

  const formatCurrency = (cents) => {
    const dollars = cents / 100;
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(dollars);
  };

  const formatTime = (timestamp) => {
    if (!timestamp) return 'N/A';
    const date = new Date(timestamp * 1000);
    return date.toLocaleTimeString('en-US', { 
      hour12: false, 
      hour: '2-digit', 
      minute: '2-digit', 
      second: '2-digit' 
    });
  };

  const getChangeIndicator = (value, isPositive = true) => {
    if (value === 0 || value === null || value === undefined) return null;
    const color = (isPositive && value > 0) || (!isPositive && value < 0) ? 'text-green-400' : 'text-red-400';
    const icon = value > 0 ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />;
    return (
      <span className={`flex items-center space-x-1 ${color} text-xs font-medium`}>
        {icon}
        <span>{value > 0 ? '+' : ''}{isPositive ? formatCurrency(value) : value}</span>
      </span>
    );
  };

  const changes = tradingState.changes || {};

  return (
    <div className="bg-gray-900/50 backdrop-blur-sm rounded-xl border border-gray-800 p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-bold text-gray-300 uppercase tracking-wider">Trading Data</h3>
        <div className="flex items-center space-x-2">
          <Clock className="w-4 h-4 text-gray-500" />
          <span className="text-xs text-gray-400 font-mono">
            Last sync: {formatTime(tradingState.sync_timestamp)}
          </span>
        </div>
      </div>
      
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {/* Balance */}
        <div className="bg-gray-800/30 rounded-lg p-3 border border-gray-700/50">
          <div className="flex items-center space-x-2 mb-1">
            <DollarSign className="w-4 h-4 text-green-400" />
            <span className="text-xs text-gray-500 uppercase">Balance</span>
          </div>
          <div className="text-lg font-mono font-bold text-white">
            {formatCurrency(tradingState.balance)}
          </div>
          {changes.balance_change && getChangeIndicator(changes.balance_change)}
        </div>

        {/* Portfolio Value */}
        <div className="bg-gray-800/30 rounded-lg p-3 border border-gray-700/50">
          <div className="flex items-center space-x-2 mb-1">
            <Briefcase className="w-4 h-4 text-blue-400" />
            <span className="text-xs text-gray-500 uppercase">Portfolio</span>
          </div>
          <div className="text-lg font-mono font-bold text-white">
            {formatCurrency(tradingState.portfolio_value)}
          </div>
          {changes.portfolio_change && getChangeIndicator(changes.portfolio_change)}
        </div>

        {/* Positions */}
        <div className="bg-gray-800/30 rounded-lg p-3 border border-gray-700/50">
          <div className="flex items-center space-x-2 mb-1">
            <ShoppingCart className="w-4 h-4 text-purple-400" />
            <span className="text-xs text-gray-500 uppercase">Positions</span>
          </div>
          <div className="text-lg font-mono font-bold text-white">
            {tradingState.position_count || 0}
          </div>
          {changes.position_count_change !== undefined && changes.position_count_change !== 0 && (
            <span className={`text-xs font-medium ${
              changes.position_count_change > 0 ? 'text-green-400' : 'text-red-400'
            }`}>
              {changes.position_count_change > 0 ? '+' : ''}{changes.position_count_change}
            </span>
          )}
        </div>

        {/* Orders */}
        <div className="bg-gray-800/30 rounded-lg p-3 border border-gray-700/50">
          <div className="flex items-center space-x-2 mb-1">
            <FileText className="w-4 h-4 text-yellow-400" />
            <span className="text-xs text-gray-500 uppercase">Orders</span>
          </div>
          <div className="text-lg font-mono font-bold text-white">
            {tradingState.order_count || 0}
          </div>
          {changes.order_count_change !== undefined && changes.order_count_change !== 0 && (
            <span className={`text-xs font-medium ${
              changes.order_count_change > 0 ? 'text-yellow-400' : 'text-gray-400'
            }`}>
              {changes.order_count_change > 0 ? '+' : ''}{changes.order_count_change}
            </span>
          )}
        </div>
      </div>
    </div>
  );
};

const V3TraderConsole = () => {
  const [messages, setMessages] = useState([]);
  // Start with all messages expanded by default for visibility
  const [expandedMessages, setExpandedMessages] = useState(new Set());
  const [wsStatus, setWsStatus] = useState('disconnected');
  const [currentState, setCurrentState] = useState('UNKNOWN');
  const [tradingState, setTradingState] = useState(null);
  const [metrics, setMetrics] = useState({
    markets_connected: 0,
    snapshots_received: 0,
    deltas_received: 0,
    uptime: 0,
    health: 'unknown',
    ping_health: 'unknown',
    last_ping_age: null,
    api_connected: false,
    api_url: null
  });
  const [copied, setCopied] = useState(false);
  const wsRef = useRef(null);
  const messagesEndRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);
  const lastMessageRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const addMessage = useCallback((type, content, metadata = {}) => {
    // Deduplicate rapid repeated messages
    const messageKey = `${type}-${content}-${metadata.state || ''}`;
    const now = Date.now();
    
    if (lastMessageRef.current) {
      const { key: lastKey, time: lastTime } = lastMessageRef.current;
      // If same message within 1 second, skip it
      if (lastKey === messageKey && (now - lastTime) < 1000) {
        return;
      }
    }
    
    lastMessageRef.current = { key: messageKey, time: now };
    
    const timestamp = new Date().toLocaleTimeString('en-US', { 
      hour12: false, 
      hour: '2-digit', 
      minute: '2-digit', 
      second: '2-digit' 
    });
    
    // Parse state transition info
    let fromState = null;
    let toState = null;
    let isTransition = false;
    
    // Detect state transitions from the message content
    if (metadata.to_state || content.includes('→')) {
      isTransition = true;
      if (metadata.from_state && metadata.to_state) {
        fromState = metadata.from_state;
        toState = metadata.to_state;
      } else if (content.includes('→')) {
        // Parse from the message content
        const match = content.match(/(\w+)\s*→\s*(\w+)/);
        if (match) {
          fromState = match[1];
          toState = match[2];
        }
      }
    }
    
    // Extract status from the message (SUCCESS, FAILED, etc.)
    let status = null;
    if (content.includes('SUCCESS')) status = 'SUCCESS';
    else if (content.includes('FAILED')) status = 'FAILED';
    else if (content.includes('ERROR')) status = 'ERROR';
    else if (content.includes('READY')) status = 'READY';
    else if (content.includes('INITIALIZING')) status = 'INITIALIZING';
    else if (content.includes('CONNECTING')) status = 'CONNECTING';
    else if (content.includes('CALIBRATING')) status = 'CALIBRATING';
    
    // Clean up the content for display
    let cleanContent = content;
    if (isTransition && fromState && toState) {
      // For state transitions, remove all the transition-related text
      // We'll display the transition as nice badges instead
      
      // Remove various patterns that might appear
      cleanContent = content
        .replace(new RegExp(`${fromState}\\s*→\\s*${toState}:?\\s*`, 'gi'), '')
        .replace(/→\s*State:\s*/gi, '')
        .replace(/State:\s*/gi, '')
        .replace(/→\s*\w+/gi, '')
        .trim();
      
      // If what's left is just a state name, remove it
      if (cleanContent.toLowerCase() === toState.toLowerCase() || 
          cleanContent.toLowerCase() === fromState.toLowerCase() ||
          cleanContent === 'State' ||
          cleanContent === '→') {
        cleanContent = '';
      }
    }
    
    // Format metadata for display
    const formattedMetadata = {};
    if (metadata.metadata && typeof metadata.metadata === 'object') {
      for (const [key, value] of Object.entries(metadata.metadata)) {
        if (key === 'markets' && Array.isArray(value)) {
          formattedMetadata[key] = value.join(', ');
        } else if (typeof value === 'object') {
          formattedMetadata[key] = JSON.stringify(value, null, 2);
        } else {
          formattedMetadata[key] = value;
        }
      }
    }
    
    // Add any additional metadata fields
    if (metadata.context && metadata.context !== 'State transition') {
      formattedMetadata.context = metadata.context;
    }
    
    const newMessage = {
      id: Date.now() + Math.random(),
      type,
      content: cleanContent,
      originalContent: content,
      timestamp,
      metadata: {
        ...metadata,
        formattedMetadata: Object.keys(formattedMetadata).length > 0 ? formattedMetadata : null,
        isTransition,
        fromState,
        toState,
        status,
        state: metadata.state || metadata.to_state || toState
      }
    };
    
    setMessages(prev => [...prev.slice(-100), newMessage]);
    
    // Auto-expand messages with metadata
    if (Object.keys(formattedMetadata).length > 0) {
      setExpandedMessages(prev => {
        const newExpanded = new Set(prev);
        newExpanded.add(newMessage.id);
        return newExpanded;
      });
    }
  }, []);

  const connectWebSocket = useCallback(() => {
    // Prevent duplicate connections
    if (wsRef.current?.readyState === WebSocket.OPEN || 
        wsRef.current?.readyState === WebSocket.CONNECTING) {
      return;
    }

    try {
      const ws = new WebSocket('ws://localhost:8005/v3/ws');
      
      ws.onopen = () => {
        setWsStatus('connected');
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          // Debug: Log ALL messages to see what's coming through
          console.log('WebSocket message received:', { type: data.type, hasMetrics: !!data.data?.metrics });
          
          switch(data.type) {
            case 'trading_state':
              // Update trading state from WebSocket
              if (data.data) {
                setTradingState({
                  has_state: true,
                  version: data.data.version,
                  balance: data.data.balance,
                  portfolio_value: data.data.portfolio_value,
                  position_count: data.data.position_count,
                  order_count: data.data.order_count,
                  positions: data.data.positions,
                  open_orders: data.data.open_orders,
                  sync_timestamp: data.data.sync_timestamp,
                  changes: data.data.changes
                });
              }
              break;
              
            case 'connection':
              // Initial connection acknowledgment - client registered with server
              // data.data.client_id contains the unique client identifier
              break;
              
            case 'history_replay':
              // Process batched historical transitions
              if (data.data.transitions) {
                data.data.transitions.forEach(transition => {
                  const fromState = transition.from_state || 'unknown';
                  const toState = transition.to_state || transition.state;
                  setCurrentState(toState);
                  addMessage('state', transition.message, {
                    state: toState,
                    from_state: fromState,
                    to_state: toState,
                    context: transition.context,
                    timestamp: transition.timestamp,
                    metadata: transition.metadata,
                    is_history: true
                  });
                });
              }
              break;
              
            case 'state_transition':
              const fromState = data.data.from_state || currentState;
              const toState = data.data.to_state || data.data.state;
              setCurrentState(toState);
              
              // Update API connection status based on state
              if (toState === 'trading_client_connect' && data.data.metadata?.api_url) {
                // Store API URL when connecting and set connected status
                setMetrics(prev => ({
                  ...prev,
                  api_url: data.data.metadata.api_url,
                  api_connected: true
                }));
              }
              
              // Update API connection status based on state
              if (toState === 'ready' || toState === 'calibrating' || toState === 'acting' || toState === 'trading_client_connect') {
                setMetrics(prev => ({ ...prev, api_connected: true }));
              } else if (toState === 'error' || toState === 'idle') {
                setMetrics(prev => ({ ...prev, api_connected: false }));
              }
              
              // Only add message if it's not just a current state update
              if (!data.data.is_current || data.data.message !== `Current state: ${toState}`) {
                addMessage('state', data.data.message, {
                  state: toState,
                  from_state: fromState,
                  to_state: toState,
                  context: data.data.context,
                  timestamp: data.data.timestamp,
                  metadata: data.data.metadata
                });
              }
              break;
              
            case 'trader_status':
              // Update metrics from single source of truth
              if (data.data.metrics) {
                // Debug: Log the full metrics object
                console.log('Full metrics object:', data.data.metrics);
                console.log('Received trader_status metrics:', {
                  ping_health: data.data.metrics.ping_health,
                  last_ping_age: data.data.metrics.last_ping_age
                });
                
                // Extract all metrics including Kalshi API ping health
                setMetrics(prev => ({
                  markets_connected: data.data.metrics.markets_connected || 0,
                  snapshots_received: data.data.metrics.snapshots_received || 0,
                  deltas_received: data.data.metrics.deltas_received || 0,
                  uptime: data.data.metrics.uptime || 0,
                  health: data.data.metrics.health || 'unknown',
                  // Use ping health from Kalshi API connection
                  ping_health: data.data.metrics.ping_health || 'unknown',
                  last_ping_age: data.data.metrics.last_ping_age || null,
                  // Preserve API connection info
                  api_connected: prev.api_connected,
                  api_url: prev.api_url
                }));
              }
              if (data.data.state) {
                setCurrentState(data.data.state);
              }
              // Also update api_connected from metrics
              if (data.data.metrics && typeof data.data.metrics.api_connected === 'boolean') {
                setMetrics(prev => ({
                  ...prev,
                  api_connected: data.data.metrics.api_connected
                }));
              }
              break;
              
            case 'ping':
              // Respond to ping if needed
              if (ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'pong', timestamp: Date.now() }));
              }
              break;
              
            // system_metrics case removed - all metrics now come from trader_status
            // This case is no longer needed as trader_status is the single source of truth
              
            default:
              // Ignore all other message types
              break;
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      ws.onerror = (error) => {
        addMessage('error', 'WebSocket error occurred', { error: error.message });
        setWsStatus('error');
      };

      ws.onclose = (event) => {
        setWsStatus('disconnected');
        
        // Only log disconnect message if it wasn't a normal close
        if (event.code !== 1000) {
          addMessage('warning', 'Disconnected from TRADER V3', { 
            icon: 'disconnect',
            code: event.code,
            reason: event.reason || 'Connection lost'
          });
        }
        
        wsRef.current = null;
        
        // Auto-reconnect after 3 seconds
        reconnectTimeoutRef.current = setTimeout(() => {
          connectWebSocket();
        }, 3000);
      };

      wsRef.current = ws;
    } catch (error) {
      addMessage('error', `Failed to connect: ${error.message}`);
      setWsStatus('error');
    }
  }, [addMessage, currentState]);

  useEffect(() => {
    connectWebSocket();
    
    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [connectWebSocket]);

  const toggleMessageExpansion = (messageId) => {
    setExpandedMessages(prev => {
      const newSet = new Set(prev);
      if (newSet.has(messageId)) {
        newSet.delete(messageId);
      } else {
        newSet.add(messageId);
      }
      return newSet;
    });
  };

  const getStateColor = (state) => {
    const colors = {
      'startup': 'text-gray-400 bg-gray-900/50',
      'initializing': 'text-yellow-400 bg-yellow-900/30',
      'orderbook_connect': 'text-blue-400 bg-blue-900/30',
      'trading_client_connect': 'text-purple-400 bg-purple-900/30',
      'kalshi_data_sync': 'text-cyan-400 bg-cyan-900/30',
      'ready': 'text-green-400 bg-green-900/30',
      'error': 'text-red-400 bg-red-900/30',
      'shutdown': 'text-gray-500 bg-gray-900/50'
    };
    const stateKey = state?.toLowerCase();
    return colors[stateKey] || 'text-gray-400 bg-gray-900/50';
  };

  const getStatusColor = (status) => {
    const colors = {
      'SUCCESS': 'text-green-400 bg-green-900/30',
      'FAILED': 'text-red-400 bg-red-900/30',
      'ERROR': 'text-red-400 bg-red-900/30',
      'READY': 'text-green-400 bg-green-900/30',
      'INITIALIZING': 'text-yellow-400 bg-yellow-900/30',
      'CONNECTING': 'text-blue-400 bg-blue-900/30',
      'CALIBRATING': 'text-purple-400 bg-purple-900/30'
    };
    return colors[status] || 'text-gray-400 bg-gray-900/50';
  };

  const getMessageIcon = (type, metadata) => {
    if (metadata?.isTransition) {
      return <ArrowRight className="w-4 h-4 text-purple-400" />;
    }
    switch(type) {
      case 'state': return <Zap className="w-4 h-4 text-purple-400" />;
      case 'data': return <Database className="w-4 h-4 text-blue-400" />;
      case 'success': return <CheckCircle className="w-4 h-4 text-green-400" />;
      case 'warning': return <AlertCircle className="w-4 h-4 text-yellow-400" />;
      case 'error': return <XCircle className="w-4 h-4 text-red-400" />;
      case 'info': return <Info className="w-4 h-4 text-blue-400" />;
      default: return <ChevronRight className="w-4 h-4 text-gray-500" />;
    }
  };

  const getMessageColor = (type) => {
    switch(type) {
      case 'state': return 'text-purple-200';
      case 'data': return 'text-blue-200';
      case 'success': return 'text-green-200';
      case 'warning': return 'text-yellow-200';
      case 'error': return 'text-red-200';
      case 'info': return 'text-blue-200';
      default: return 'text-gray-200';
    }
  };

  const copyConsoleOutput = () => {
    const output = messages.map(msg => {
      const state = msg.metadata?.state ? `[${msg.metadata.state}] ` : '';
      const status = msg.metadata?.status ? `[${msg.metadata.status}] ` : '';
      const transition = msg.metadata?.isTransition 
        ? `[${msg.metadata.fromState} → ${msg.metadata.toState}] ` 
        : '';
      return `${msg.timestamp} ${transition || state}${status}${msg.content}`;
    }).join('\n');
    
    navigator.clipboard.writeText(output).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-950 via-gray-900 to-gray-950">
      {/* Header */}
      <div className="border-b border-gray-800 bg-black/30 backdrop-blur-sm sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <Activity className="w-6 h-6 text-purple-400" />
                <h1 className="text-xl font-semibold text-white">TRADER V3</h1>
                <span className="px-2 py-0.5 text-xs font-mono bg-purple-500/20 text-purple-300 rounded-full border border-purple-500/30">
                  CONSOLE
                </span>
              </div>
            </div>
            
            <div className="flex items-center space-x-6">
              {/* Connection Status */}
              <div className="flex items-center space-x-2">
                {wsStatus === 'connected' ? (
                  <>
                    <Wifi className="w-5 h-5 text-green-400" />
                    <span className="text-sm text-green-400 font-medium">Connected</span>
                  </>
                ) : (
                  <>
                    <WifiOff className="w-5 h-5 text-red-400" />
                    <span className="text-sm text-red-400 font-medium">Disconnected</span>
                  </>
                )}
              </div>
              
              {/* Current State Badge */}
              <div className={`px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider ${getStateColor(currentState)}`}>
                {currentState}
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-6 py-6">
        {/* Trading Data Panel - Full width above everything */}
        <div className="mb-6">
          <TradingData tradingState={tradingState} />
        </div>
        
        <div className="grid grid-cols-12 gap-6">
          {/* Metrics Panel */}
          <div className="col-span-3">
            <div className="bg-gray-900/50 backdrop-blur-sm rounded-xl border border-gray-800 p-6 space-y-5">
              <h3 className="text-sm font-bold text-gray-300 uppercase tracking-wider">System Metrics</h3>
              
              <div className="space-y-4">
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-500">Markets</span>
                    <span className="text-lg font-mono font-bold text-white">{metrics.markets_connected || 0}</span>
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-500">Snapshots</span>
                    <span className="text-lg font-mono font-bold text-blue-400">{metrics.snapshots_received || 0}</span>
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-500">Deltas</span>
                    <span className="text-lg font-mono font-bold text-purple-400">{metrics.deltas_received || 0}</span>
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-500">Uptime</span>
                    <span className="text-lg font-mono font-bold text-green-400">
                      {metrics.uptime ? `${Math.floor(metrics.uptime)}s` : '0s'}
                    </span>
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-500">Ping Health</span>
                    <div className={`px-3 py-1 rounded-full text-xs font-bold ${
                      metrics.ping_health === 'healthy' ? 'bg-green-900/30 text-green-400' : 
                      metrics.ping_health === 'degraded' ? 'bg-yellow-900/30 text-yellow-400' : 
                      metrics.ping_health === 'unhealthy' ? 'bg-red-900/30 text-red-400' : 
                      'bg-gray-900/30 text-gray-400'
                    }`}>
                      {metrics.ping_health?.toUpperCase() || 'UNKNOWN'}
                    </div>
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-500">Last Message</span>
                    <span className={`text-lg font-mono font-bold ${
                      metrics.last_ping_age === null ? 'text-gray-400' :
                      metrics.last_ping_age < 10 ? 'text-green-400' :
                      metrics.last_ping_age < 30 ? 'text-yellow-400' : 'text-red-400'
                    }`}>
                      {metrics.last_ping_age !== null ? `${Math.floor(metrics.last_ping_age)}s ago` : 'N/A'}
                    </span>
                  </div>
                </div>
                
                <div className="pt-4 border-t border-gray-700">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-500">Health Status</span>
                    <div className={`px-3 py-1 rounded-full text-xs font-bold ${
                      metrics.health === 'healthy' ? 'bg-green-900/30 text-green-400' : 
                      metrics.health === 'unhealthy' ? 'bg-red-900/30 text-red-400' : 
                      'bg-gray-900/30 text-gray-400'
                    }`}>
                      {metrics.health?.toUpperCase() || 'UNKNOWN'}
                    </div>
                  </div>
                </div>
              </div>
              
              {/* API Status */}
              <div className="pt-4 border-t border-gray-700">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs text-gray-500 uppercase tracking-wider">API Status</span>
                  <div className={`px-3 py-1 rounded-full text-xs font-bold ${
                    metrics.api_connected ? 'bg-green-900/30 text-green-400' : 'bg-red-900/30 text-red-400'
                  }`}>
                    {metrics.api_connected ? 'CONNECTED' : 'DISCONNECTED'}
                  </div>
                </div>
                {metrics.api_url && (
                  <div className="mt-2">
                    <span className="text-xs text-gray-500">Endpoint:</span>
                    <div className="text-xs text-gray-300 font-mono mt-1 truncate" title={metrics.api_url}>
                      {metrics.api_url.replace('https://', '').replace('/trade-api/v2', '')}
                    </div>
                  </div>
                )}
              </div>
              
              {/* Activity Indicator */}
              <div className="pt-4 border-t border-gray-700">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs text-gray-500 uppercase tracking-wider">Activity</span>
                </div>
                <div className="flex space-x-1">
                  {[...Array(10)].map((_, i) => (
                    <div
                      key={i}
                      className={`flex-1 h-2 rounded-full transition-all duration-300 ${
                        i < (metrics.markets_connected || 0) 
                          ? 'bg-gradient-to-r from-purple-500 to-blue-500 animate-pulse' 
                          : 'bg-gray-800'
                      }`}
                    />
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Console */}
          <div className="col-span-9">
            <div className="bg-gray-900/50 backdrop-blur-sm rounded-xl border border-gray-800 overflow-hidden">
              {/* Console Header */}
              <div className="bg-black/50 px-6 py-3 border-b border-gray-800 flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <div className="flex space-x-1.5">
                    <div className="w-3 h-3 rounded-full bg-red-500/80 hover:bg-red-500 transition-colors cursor-pointer" />
                    <div className="w-3 h-3 rounded-full bg-yellow-500/80 hover:bg-yellow-500 transition-colors cursor-pointer" />
                    <div className="w-3 h-3 rounded-full bg-green-500/80 hover:bg-green-500 transition-colors cursor-pointer" />
                  </div>
                  <span className="text-xs text-gray-400 font-mono uppercase tracking-wider">System Console</span>
                </div>
                <div className="flex items-center space-x-4">
                  <button
                    onClick={copyConsoleOutput}
                    className="flex items-center space-x-2 px-3 py-1.5 text-xs text-gray-400 hover:text-gray-200 hover:bg-gray-800 rounded-lg transition-all"
                    title="Copy console output"
                  >
                    {copied ? (
                      <>
                        <Check className="w-3.5 h-3.5" />
                        <span className="font-medium">Copied!</span>
                      </>
                    ) : (
                      <>
                        <Copy className="w-3.5 h-3.5" />
                        <span className="font-medium">Copy</span>
                      </>
                    )}
                  </button>
                  <span className="text-xs text-gray-500 font-mono">
                    {messages.length} messages
                  </span>
                </div>
              </div>
              
              {/* Messages */}
              <div className="h-[600px] overflow-y-auto p-4 font-mono text-sm bg-black/20">
                {messages.length === 0 ? (
                  <div className="flex flex-col items-center justify-center h-full text-gray-600">
                    <Activity className="w-8 h-8 mb-3 animate-pulse" />
                    <div>Waiting for messages...</div>
                  </div>
                ) : (
                  <div className="space-y-2">
                    {messages.map((message) => {
                      const isExpanded = expandedMessages.has(message.id);
                      const hasMetadata = message.metadata?.formattedMetadata && 
                                         Object.keys(message.metadata.formattedMetadata).length > 0;
                      
                      return (
                        <div 
                          key={message.id} 
                          className="group hover:bg-gray-800/20 rounded-lg transition-all duration-200"
                        >
                          <div className="flex items-start space-x-3 p-2">
                            {/* Timestamp */}
                            <span className="text-gray-500 text-xs w-20 flex-shrink-0 font-mono">
                              [{message.timestamp}]
                            </span>
                            
                            {/* State Transition or State Badge */}
                            <div className="w-44 flex-shrink-0">
                              {message.metadata?.isTransition && message.metadata?.fromState && message.metadata?.toState ? (
                                <div className="flex items-center space-x-1 text-xs">
                                  <span className={`px-2 py-0.5 rounded font-bold ${getStateColor(message.metadata.fromState)}`}>
                                    {message.metadata.fromState}
                                  </span>
                                  <ArrowRight className="w-3 h-3 text-gray-400" />
                                  <span className={`px-2 py-0.5 rounded font-bold ${getStateColor(message.metadata.toState)}`}>
                                    {message.metadata.toState}
                                  </span>
                                </div>
                              ) : message.metadata?.state ? (
                                <span className={`px-2 py-0.5 rounded text-xs font-bold ${getStateColor(message.metadata.state)}`}>
                                  [{message.metadata.state}]
                                </span>
                              ) : null}
                            </div>
                            
                            {/* Status Badge */}
                            <div className="w-24 flex-shrink-0">
                              {message.metadata?.status && (
                                <span className={`px-2 py-0.5 rounded text-xs font-bold ${getStatusColor(message.metadata.status)}`}>
                                  {message.metadata.status}
                                </span>
                              )}
                            </div>
                            
                            {/* Icon */}
                            <div className="flex-shrink-0 mt-0.5">
                              {getMessageIcon(message.type, message.metadata)}
                            </div>
                            
                            {/* Message Content - only show if there's content */}
                            {message.content && (
                              <div className={`flex-1 ${getMessageColor(message.type)}`}>
                                <div className="leading-relaxed">{message.content}</div>
                              </div>
                            )}
                            {!message.content && (
                              <div className="flex-1"></div>
                            )}
                            
                            {/* Expand/Collapse Button */}
                            {hasMetadata && (
                              <button
                                onClick={() => toggleMessageExpansion(message.id)}
                                className="flex-shrink-0 p-1 text-gray-500 hover:text-gray-300 transition-colors"
                                title={isExpanded ? "Collapse metadata" : "Expand metadata"}
                              >
                                {isExpanded ? (
                                  <ChevronDown className="w-4 h-4" />
                                ) : (
                                  <ChevronRight className="w-4 h-4" />
                                )}
                              </button>
                            )}
                          </div>
                          
                          {/* Expandable Metadata Section */}
                          {hasMetadata && isExpanded && (
                            <div className="ml-24 mb-2 mr-4">
                              <div className="bg-gray-800/30 rounded-lg p-3 border border-gray-700/50">
                                <div className="text-xs text-gray-400 font-mono space-y-1">
                                  {Object.entries(message.metadata.formattedMetadata).map(([key, value]) => (
                                    <div key={key} className="flex">
                                      <span className="text-gray-500 mr-2">{key}:</span>
                                      <span className="text-gray-300 break-all whitespace-pre-wrap">{value}</span>
                                    </div>
                                  ))}
                                </div>
                              </div>
                            </div>
                          )}
                        </div>
                      );
                    })}
                    <div ref={messagesEndRef} />
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default V3TraderConsole;