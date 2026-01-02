import React, { useState, useEffect, useRef } from 'react';
import { Terminal, Activity, AlertCircle, CheckCircle, XCircle } from 'lucide-react';

const V3Console = () => {
  const [connected, setConnected] = useState(false);
  const [status, setStatus] = useState(null);
  const [messages, setMessages] = useState([]);
  const [state, setState] = useState('IDLE');
  const ws = useRef(null);
  const messagesEndRef = useRef(null);

  // Auto-scroll to bottom
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(scrollToBottom, [messages]);

  useEffect(() => {
    // Connect to V3 WebSocket
    const connectWebSocket = () => {
      ws.current = new WebSocket('ws://localhost:8006/v3/ws');

      ws.current.onopen = () => {
        setConnected(true);
        addMessage('info', 'Connected to TRADER V3');
      };

      ws.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          handleMessage(data);
        } catch (error) {
          console.error('Failed to parse message:', error);
        }
      };

      ws.current.onerror = (error) => {
        addMessage('error', `WebSocket error: ${error.message || 'Unknown error'}`);
      };

      ws.current.onclose = () => {
        setConnected(false);
        addMessage('warning', 'Disconnected from TRADER V3');
        // Reconnect after 3 seconds
        setTimeout(connectWebSocket, 3000);
      };
    };

    connectWebSocket();

    // Fetch initial status
    fetchStatus();
    const statusInterval = setInterval(fetchStatus, 5000);

    return () => {
      clearInterval(statusInterval);
      if (ws.current) {
        ws.current.close();
      }
    };
  }, []);

  const fetchStatus = async () => {
    try {
      const response = await fetch('http://localhost:8006/v3/status');
      if (response.ok) {
        const data = await response.json();
        setStatus(data);
        setState(data.state || 'UNKNOWN');
      }
    } catch (error) {
      console.error('Failed to fetch status:', error);
    }
  };

  const handleMessage = (data) => {
    switch (data.type) {
      case 'console':
        addMessage(data.data.level, data.data.message, data.data.context);
        break;

      case 'state_transition':
        setState(data.data.to_state);
        addMessage('info', `STATE: ${data.data.from_state} → ${data.data.to_state}`, data.data.context);
        break;

      case 'trader_status':
        if (data.data.metrics_summary) {
          addMessage('info', `STATUS: ${data.data.metrics_summary}`);
        }
        break;

      case 'connection':
        addMessage('info', data.data.message);
        break;

      default:
        console.log('Unknown message type:', data.type);
    }
  };

  const addMessage = (level, text, context = null) => {
    const timestamp = new Date().toLocaleTimeString();
    setMessages(prev => [...prev, { timestamp, level, text, context }]);
  };

  const getLevelIcon = (level) => {
    switch (level) {
      case 'error':
        return <XCircle className="w-4 h-4 text-red-500" />;
      case 'warning':
        return <AlertCircle className="w-4 h-4 text-yellow-500" />;
      case 'info':
        return <CheckCircle className="w-4 h-4 text-green-500" />;
      default:
        return <Activity className="w-4 h-4 text-gray-500" />;
    }
  };

  const getLevelColor = (level) => {
    switch (level) {
      case 'error':
        return 'text-red-400';
      case 'warning':
        return 'text-yellow-400';
      case 'info':
        return 'text-green-400';
      default:
        return 'text-gray-400';
    }
  };

  const getStateColor = (state) => {
    switch (state) {
      case 'IDLE':
        return 'bg-gray-500';
      case 'CALIBRATING':
        return 'bg-yellow-500';
      case 'READY':
        return 'bg-green-500';
      case 'ERROR':
        return 'bg-red-500';
      case 'STOPPED':
        return 'bg-gray-700';
      default:
        return 'bg-gray-600';
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white p-4">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <Terminal className="w-8 h-8 text-cyan-400" />
            <h1 className="text-2xl font-bold">TRADER V3 Console</h1>
            <div className={`px-3 py-1 rounded-full text-sm font-medium ${getStateColor(state)}`}>
              {state}
            </div>
          </div>
          <div className="flex items-center gap-2">
            <div className={`w-3 h-3 rounded-full ${connected ? 'bg-green-500' : 'bg-red-500'} animate-pulse`} />
            <span className="text-sm">{connected ? 'Connected' : 'Disconnected'}</span>
          </div>
        </div>

        {/* Stats Bar */}
        {status && (
          <div className="grid grid-cols-4 gap-4 mb-6">
            <div className="bg-gray-800 rounded-lg p-4">
              <div className="text-sm text-gray-400 mb-1">Uptime</div>
              <div className="text-xl font-mono">{Math.floor(status.uptime || 0)}s</div>
            </div>
            <div className="bg-gray-800 rounded-lg p-4">
              <div className="text-sm text-gray-400 mb-1">Markets</div>
              <div className="text-xl font-mono">
                {status.components?.orderbook_integration?.markets_connected || 0}
              </div>
            </div>
            <div className="bg-gray-800 rounded-lg p-4">
              <div className="text-sm text-gray-400 mb-1">Snapshots</div>
              <div className="text-xl font-mono">
                {status.components?.orderbook_integration?.snapshots_received || 0}
              </div>
            </div>
            <div className="bg-gray-800 rounded-lg p-4">
              <div className="text-sm text-gray-400 mb-1">Deltas</div>
              <div className="text-xl font-mono">
                {status.components?.orderbook_integration?.deltas_received || 0}
              </div>
            </div>
          </div>
        )}

        {/* Console */}
        <div className="bg-black rounded-lg border border-gray-700">
          <div className="border-b border-gray-700 p-3">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-red-500 rounded-full" />
              <div className="w-3 h-3 bg-yellow-500 rounded-full" />
              <div className="w-3 h-3 bg-green-500 rounded-full" />
              <span className="ml-3 text-sm text-gray-400">System Console</span>
            </div>
          </div>
          
          <div className="p-4 h-96 overflow-y-auto font-mono text-sm">
            {messages.length === 0 ? (
              <div className="text-gray-600">Waiting for messages...</div>
            ) : (
              messages.map((msg, index) => (
                <div key={index} className="flex items-start gap-2 mb-2">
                  <span className="text-gray-600 text-xs">{msg.timestamp}</span>
                  {getLevelIcon(msg.level)}
                  <span className={getLevelColor(msg.level)}>{msg.text}</span>
                </div>
              ))
            )}
            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Component Health */}
        {status?.components && (
          <div className="mt-6 grid grid-cols-2 gap-4">
            {Object.entries(status.components).map(([name, component]) => (
              <div key={name} className="bg-gray-800 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium">{name.replace(/_/g, ' ').toUpperCase()}</span>
                  <div className={`w-2 h-2 rounded-full ${component.healthy ? 'bg-green-500' : 'bg-red-500'}`} />
                </div>
                <div className="text-xs text-gray-400">
                  {component.running ? 'Running' : 'Stopped'}
                  {component.uptime_seconds && ` • ${Math.floor(component.uptime_seconds)}s`}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default V3Console;