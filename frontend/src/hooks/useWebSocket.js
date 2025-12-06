import { useState, useEffect, useRef, useCallback } from 'react';

const useWebSocket = (url) => {
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const [lastMessage, setLastMessage] = useState(null);
  const [error, setError] = useState(null);
  const ws = useRef(null);
  const reconnectTimeoutRef = useRef(null);
  const reconnectAttemptsRef = useRef(0);
  const isConnectingRef = useRef(false); // Prevent multiple simultaneous connections
  const messageCounterRef = useRef(0); // Counter to make each message unique
  const lastUpdateRef = useRef(0); // Throttle updates to prevent infinite loops
  const pendingMessageRef = useRef(null); // Store pending message for throttled updates
  const throttleTimerRef = useRef(null); // Timer for processing pending messages
  const maxReconnectAttempts = 5;
  const baseReconnectDelay = 1000;

  const connect = useCallback(() => {
    try {
      // Prevent multiple simultaneous connection attempts
      if (isConnectingRef.current || (ws.current && ws.current.readyState === WebSocket.CONNECTING)) {
        return;
      }
      
      isConnectingRef.current = true;
      setConnectionStatus('connecting');
      setError(null);

      ws.current = new WebSocket(url);

      ws.current.onopen = () => {
        isConnectingRef.current = false;
        setConnectionStatus('connected');
        reconnectAttemptsRef.current = 0;
      };

      ws.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          // Throttle updates to prevent infinite loops (max 10 updates per second)
          const now = Date.now();
          const timeSinceLastUpdate = now - lastUpdateRef.current;
          
          if (timeSinceLastUpdate >= 100) { // Allow update if 100ms have passed
            lastUpdateRef.current = now;
            messageCounterRef.current += 1;
            setLastMessage({
              ...data,
              _messageId: messageCounterRef.current
            });
            pendingMessageRef.current = null; // Clear any pending message
          } else {
            // Store the message for later if we're throttling
            pendingMessageRef.current = data;
            
            // Set a timer to process the pending message if one doesn't exist
            if (!throttleTimerRef.current) {
              throttleTimerRef.current = setTimeout(() => {
                if (pendingMessageRef.current) {
                  messageCounterRef.current += 1;
                  setLastMessage({
                    ...pendingMessageRef.current,
                    _messageId: messageCounterRef.current
                  });
                  pendingMessageRef.current = null;
                }
                throttleTimerRef.current = null;
                lastUpdateRef.current = Date.now();
              }, 100);
            }
          }
        } catch (err) {
          console.error('Failed to parse WebSocket message:', err);
        }
      };

      ws.current.onclose = (event) => {
        isConnectingRef.current = false;
        setConnectionStatus('disconnected');
        
        // Only attempt to reconnect if it wasn't a manual close and we haven't exceeded max attempts
        if (event.code !== 1000 && reconnectAttemptsRef.current < maxReconnectAttempts) {
          const delay = Math.min(
            baseReconnectDelay * Math.pow(2, reconnectAttemptsRef.current),
            30000
          );
          
          setError(`Connection lost. Reconnecting in ${Math.ceil(delay/1000)}s...`);
          
          reconnectTimeoutRef.current = setTimeout(() => {
            if (reconnectAttemptsRef.current < maxReconnectAttempts) {
              reconnectAttemptsRef.current++;
              connect();
            }
          }, delay);
        } else if (reconnectAttemptsRef.current >= maxReconnectAttempts) {
          setError('Connection failed after maximum retry attempts');
        }
      };

      ws.current.onerror = (event) => {
        isConnectingRef.current = false;
        console.error('WebSocket error:', event);
        setError('Connection error occurred');
        setConnectionStatus('error');
      };

    } catch (err) {
      isConnectingRef.current = false;
      console.error('Failed to create WebSocket connection:', err);
      setError(err.message);
      setConnectionStatus('error');
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Remove url from dependencies to prevent infinite recreations

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    
    if (throttleTimerRef.current) {
      clearTimeout(throttleTimerRef.current);
      throttleTimerRef.current = null;
    }
    
    if (ws.current && ws.current.readyState !== WebSocket.CLOSED) {
      ws.current.close(1000, 'Manual disconnect');
    }
  }, []);

  const sendMessage = useCallback((message) => {
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify(message));
    } else {
      console.warn('WebSocket is not connected');
    }
  }, []);

  useEffect(() => {
    connect();

    return () => {
      // Clear any pending reconnection attempts
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = null;
      }
      
      // Clear throttle timer
      if (throttleTimerRef.current) {
        clearTimeout(throttleTimerRef.current);
        throttleTimerRef.current = null;
      }
      
      // Close connection if open
      if (ws.current && ws.current.readyState !== WebSocket.CLOSED) {
        ws.current.close(1000, 'Component unmounting');
      }
    };
  }, [url]); // Depend on url directly instead of connect function

  return {
    connectionStatus,
    lastMessage,
    error,
    sendMessage,
    reconnect: connect
  };
};

export default useWebSocket;