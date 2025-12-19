import React, { useState, useEffect, useRef, useMemo } from 'react';
import TradesFeed from './TradesFeed';
import CollectionStatus from './CollectionStatus';
import TraderStatePanel from './TraderStatePanel';
import SystemHealth from './SystemHealth';
import { ChevronDownIcon, ChevronRightIcon } from '@heroicons/react/24/outline';

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
  const [allOrders, setAllOrders] = useState(new Map()); // Track ALL orders by trade_sequence_id
  const [positions, setPositions] = useState({}); // Track positions
  const [priceMode, setPriceMode] = useState('dollar'); // Price display mode: 'dollar' or 'cent'
  const [initializationStatus, setInitializationStatus] = useState(null); // Initialization checklist status
  const [componentHealth, setComponentHealth] = useState({}); // Component health status
  
  // Position update tracking for animations
  const [previousPositions, setPreviousPositions] = useState({}); // Track previous position values
  const [positionUpdateHistory, setPositionUpdateHistory] = useState(new Map()); // Map<ticker, lastUpdate>
  const [positionAnimations, setPositionAnimations] = useState({}); // Map<ticker, {highlightField, changeDirection, isNew, isSettled}>
  const [settledPositions, setSettledPositions] = useState({}); // Track settled positions separately
  const [positionsTab, setPositionsTab] = useState('active'); // 'active' or 'settled'
  // Default to 'system' tab if initialization is not complete
  const [activeTab, setActiveTab] = useState('system'); // Tab selection: 'portfolio' or 'system'
  
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
                // Ensure open_orders is an array (it might be an object or other type)
                const ordersArray = Array.isArray(data.data.open_orders) 
                  ? data.data.open_orders 
                  : Object.values(data.data.open_orders || {});
                
                setOpenOrders(ordersArray);
                // Add to allOrders map to preserve history
                ordersArray.forEach(order => {
                  if (order.trade_sequence_id) {
                    setAllOrders(prev => {
                      const newMap = new Map(prev);
                      const existingOrder = newMap.get(order.trade_sequence_id);
                      // Only update if not already filled
                      if (!existingOrder || existingOrder.status !== 'filled') {
                        newMap.set(order.trade_sequence_id, { ...order, status: 'pending' });
                      }
                      return newMap;
                    });
                  }
                });
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
                  trade_sequence_id: data.data.trade_sequence_id || data.data.execution_result?.trade_sequence_id,
                  ...data.data
                };
                
                // If this action resulted in an order, add it to allOrders
                if (data.data.execution_result?.order && data.data.trade_sequence_id) {
                  setAllOrders(prev => {
                    const newMap = new Map(prev);
                    newMap.set(data.data.trade_sequence_id, {
                      ...data.data.execution_result.order,
                      trade_sequence_id: data.data.trade_sequence_id,
                      status: 'pending'
                    });
                    return newMap;
                  });
                }
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
                // Ensure open_orders is an array
                const ordersArray = Array.isArray(data.data.open_orders) 
                  ? data.data.open_orders 
                  : Object.values(data.data.open_orders || {});
                
                setOpenOrders(ordersArray);
                setTraderState(prev => ({
                  ...prev,
                  open_orders: ordersArray
                }));
              }
            }
            break;
            
          case 'position_update':
            // Position update from WebSocket with change metadata
            if (data.data) {
              const { 
                ticker, 
                position, 
                position_cost,
                cost_basis,
                realized_pnl, 
                fees_paid,
                volume,
                market_exposure,
                market_exposure_cents,  // In cents (preferred)
                changed_fields = [],
                previous_values = {},
                update_source = 'websocket',
                timestamp,
                was_settled = false
              } = data.data;
              
              if (ticker) {
                const prevPosition = previousPositions[ticker] || {};
                const isNew = !prevPosition.position && position !== 0;
                
                // Determine change direction for each changed field
                const changeDirection = {};
                changed_fields.forEach(field => {
                  const prevValue = previous_values[field] ?? prevPosition[field] ?? 0;
                  const newValue = data.data[field] ?? 0;
                  if (newValue > prevValue) {
                    changeDirection[field] = 'up';
                  } else if (newValue < prevValue) {
                    changeDirection[field] = 'down';
                  } else {
                    changeDirection[field] = 'neutral';
                  }
                });
                
                // Set animation flags
                const animationData = {
                  highlightField: changed_fields.length > 0 ? changed_fields[0] : null,
                  changeDirection: changeDirection,
                  isNew: isNew,
                  isSettled: was_settled,
                  timestamp: timestamp || Date.now()
                };
                
                setPositionAnimations(prev => ({
                  ...prev,
                  [ticker]: animationData
                }));
                
                // Update position update history
                setPositionUpdateHistory(prev => {
                  const newMap = new Map(prev);
                  newMap.set(ticker, {
                    timestamp: timestamp || Date.now(),
                    changed_fields,
                    previous_values,
                    update_source
                  });
                  return newMap;
                });
                
                // Update positions - all monetary values are in cents
                const positionData = {
                  position: position || 0,
                  cost_basis: position_cost || cost_basis || 0,  // In cents
                  realized_pnl: realized_pnl || 0,  // In cents
                  fees_paid: fees_paid || 0,  // In cents
                  volume: volume || 0,
                  market_exposure_cents: market_exposure_cents || (market_exposure ? (typeof market_exposure === 'number' ? market_exposure * 100 : parseFloat(market_exposure) * 100) : undefined),  // Prefer market_exposure_cents, fallback to converting market_exposure
                  market_exposure: market_exposure,  // Keep for backward compatibility
                  last_updated_ts: timestamp || Date.now()
                };
                
                setPositions(prev => ({
                  ...prev,
                  [ticker]: positionData
                }));
                
                setTraderState(prev => ({
                  ...prev,
                  positions: {
                    ...prev.positions,
                    [ticker]: positionData
                  }
                }));
                
                // Update previous positions tracking
                setPreviousPositions(prev => ({
                  ...prev,
                  [ticker]: positionData
                }));
                
                // Handle settlement - move to settled positions
                if (was_settled || (prevPosition.position !== 0 && position === 0)) {
                  setSettledPositions(prev => ({
                    ...prev,
                    [ticker]: {
                      ...positionData,
                      settled_at: timestamp || Date.now(),
                      final_pnl: realized_pnl || 0
                    }
                  }));
                  
                  // Remove from active positions after a delay to show animation
                  setTimeout(() => {
                    setPositions(prev => {
                      const newPos = { ...prev };
                      delete newPos[ticker];
                      return newPos;
                    });
                  }, 2000); // Keep in active for 2s to show settlement animation
                }
                
                // Clear animation flags after animation completes
                setTimeout(() => {
                  setPositionAnimations(prev => {
                    const newAnim = { ...prev };
                    if (newAnim[ticker]) {
                      delete newAnim[ticker].highlightField;
                      if (Object.keys(newAnim[ticker]).length === 1) { // Only timestamp left
                        delete newAnim[ticker];
                      }
                    }
                    return newAnim;
                  });
                }, 1500); // Clear after 1.5s
              }
            }
            break;
            
          case 'settlements_update':
            // Settlements update from API sync
            if (data.data && data.data.settlements) {
              const settlements = data.data.settlements;
              const updateTimestamp = data.data.timestamp || Date.now();
              
              setSettledPositions(prev => {
                const updated = { ...prev };
                
                // Merge new settlements, preferring newer settled_time
                Object.entries(settlements).forEach(([ticker, settlement]) => {
                  const existing = prev[ticker];
                  if (!existing || !existing.settled_time || 
                      (settlement.settled_time && settlement.settled_time > existing.settled_time)) {
                    updated[ticker] = {
                      ...settlement,
                      synced_at: updateTimestamp
                    };
                  }
                });
                
                return updated;
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
                  trade_sequence_id: data.data.trade_sequence_id || null,
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
              // Also check for open_orders in portfolio updates
              if (data.data.open_orders) {
                // Ensure open_orders is an array
                const ordersArray = Array.isArray(data.data.open_orders) 
                  ? data.data.open_orders 
                  : Object.values(data.data.open_orders || {});
                
                setOpenOrders(ordersArray);
                ordersArray.forEach(order => {
                  if (order.trade_sequence_id) {
                    setAllOrders(prev => {
                      const newMap = new Map(prev);
                      const existingOrder = newMap.get(order.trade_sequence_id);
                      if (!existingOrder || existingOrder.status !== 'filled') {
                        newMap.set(order.trade_sequence_id, { ...order, status: 'pending' });
                      }
                      return newMap;
                    });
                  }
                });
              }
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
            
          case 'initialization_start':
            // Initialization sequence started
            if (data.data) {
              setInitializationStatus({
                started_at: data.data.started_at,
                steps: {},
                is_complete: false,
                has_errors: false,
                summary: {}
              });
              setActiveTab('system'); // Ensure system tab is open during initialization
            }
            break;
            
          case 'initialization_step':
            // Individual initialization step progress
            if (data.data) {
              setInitializationStatus(prev => {
                const updated = { ...prev };
                if (!updated.steps) updated.steps = {};
                updated.steps[data.data.step_id] = data.data;
                return updated;
              });
            }
            break;
            
          case 'initialization_complete':
            // Initialization sequence completed
            if (data.data) {
              setInitializationStatus(prev => ({
                ...prev,
                ...data.data,
                started_at: data.data.started_at || prev?.started_at,
                completed_at: data.data.completed_at,
                duration_seconds: data.data.duration_seconds,
                warnings: data.data.warnings || [],
                steps: data.data.steps || prev?.steps || {},
                is_complete: true,
                has_errors: (data.data.warnings || []).length > 0,
                summary: {
                  total_steps: data.data.total_steps || Object.keys(data.data.steps || {}).length,
                  completed_steps: data.data.completed_steps || Object.values(data.data.steps || {}).filter(s => s.status === 'complete').length,
                }
              }));
              // Switch to portfolio tab after initialization completes
              setActiveTab('portfolio');
            }
            break;
            
          case 'component_health':
            // Component health update
            if (data.data) {
              setComponentHealth(prev => ({
                ...prev,
                [data.data.component]: {
                  status: data.data.status,
                  last_update: data.data.last_update,
                  details: data.data.details || {},
                },
              }));
            }
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

  // Parse settlement date from ticker (e.g., "INXD-25JAN03" -> Date object)
  const parseSettlementDate = (ticker) => {
    try {
      // Kalshi ticker format: SYMBOL-DDMMMYY (e.g., "INXD-25JAN03", "PRES-05NOV24")
      const match = ticker.match(/-(\d{2})([A-Z]{3})(\d{2})$/);
      if (match) {
        const [, day, monthStr, year] = match;
        const monthMap = {
          'JAN': 0, 'FEB': 1, 'MAR': 2, 'APR': 3, 'MAY': 4, 'JUN': 5,
          'JUL': 6, 'AUG': 7, 'SEP': 8, 'OCT': 9, 'NOV': 10, 'DEC': 11
        };
        const month = monthMap[monthStr];
        if (month !== undefined) {
          const yearNum = parseInt(year);
          // Handle 2-digit years: assume 2000-2099 range
          const fullYear = yearNum < 50 ? 2000 + yearNum : 1900 + yearNum;
          const date = new Date(fullYear, month, parseInt(day));
          return date;
        }
      }
    } catch (e) {
      console.warn(`Failed to parse settlement date from ticker ${ticker}:`, e);
    }
    return null;
  };

  // Calculate time to expiration
  const getTimeToExpiration = (settlementDate) => {
    if (!settlementDate) return null;
    const now = new Date();
    const diff = settlementDate.getTime() - now.getTime();
    const days = Math.floor(diff / (1000 * 60 * 60 * 24));
    const hours = Math.floor((diff % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
    const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
    
    if (diff < 0) return { expired: true, display: 'Expired' };
    if (days > 0) return { expired: false, display: `${days}d ${hours}h`, days };
    if (hours > 0) return { expired: false, display: `${hours}h ${minutes}m`, days: 0, hours };
    return { expired: false, display: `${minutes}m`, days: 0, hours: 0, minutes };
  };

  // Format positions for display
  const formatPositions = () => {
    if (!positions || Object.keys(positions).length === 0) return [];
    
    const formatted = Object.entries(positions).map(([ticker, pos]) => {
      const contracts = pos.position || pos.contracts || 0;
      const settlementDate = parseSettlementDate(ticker);
      const timeToExp = getTimeToExpiration(settlementDate);
      
      // Calculate market exposure - all values are in cents
      let marketExposureCents = 0;
      if (pos.market_exposure_cents !== undefined && pos.market_exposure_cents !== null) {
        marketExposureCents = typeof pos.market_exposure_cents === 'number' 
          ? pos.market_exposure_cents 
          : parseFloat(pos.market_exposure_cents) || 0;
      } else if (pos.market_exposure_dollars !== undefined && pos.market_exposure_dollars !== null) {
        // Convert dollars to cents
        const exposureDollars = typeof pos.market_exposure_dollars === 'number' 
          ? pos.market_exposure_dollars 
          : parseFloat(pos.market_exposure_dollars) || 0;
        marketExposureCents = exposureDollars * 100.0;
      } else if (pos.market_exposure !== undefined && pos.market_exposure !== null) {
        // If market_exposure is provided, assume it's in centi-cents, convert to cents
        const exposure = typeof pos.market_exposure === 'number' 
          ? pos.market_exposure 
          : parseFloat(pos.market_exposure) || 0;
        marketExposureCents = exposure / 100.0;  // Convert centi-cents to cents
      }
      
      return {
        ticker,
        contracts,
        side: contracts > 0 ? 'YES' : 'NO',
        costBasis: pos.cost_basis || 0,  // In cents
        realizedPnl: pos.realized_pnl || 0,  // In cents
        marketExposure: marketExposureCents,  // In cents
        totalTraded: pos.total_traded,
        feesPaid: pos.fees_paid || 0,  // In cents
        lastUpdated: pos.last_updated_ts,
        lastUpdatedTimestamp: pos.last_updated_ts ? new Date(pos.last_updated_ts).getTime() : 0,
        settlementDate,
        timeToExp,
        settlementTimestamp: settlementDate ? settlementDate.getTime() : Infinity
      };
    }).filter(p => p.contracts !== 0);
    
    // Sort by last updated date (most recent first), with stable sorting to prevent jumping
    // Use a combination of last_updated_ts and ticker for stable sort
    return formatted.sort((a, b) => {
      // First, sort by last updated timestamp (most recent first)
      const aTime = a.lastUpdatedTimestamp || 0;
      const bTime = b.lastUpdatedTimestamp || 0;
      if (aTime !== bTime) {
        return bTime - aTime;  // Descending (newest first)
      }
      // If timestamps are equal (or both 0), use ticker for stable sort
      return a.ticker.localeCompare(b.ticker);
    });
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

  // Helper function to format timestamp
  const formatTimestamp = (timestamp) => {
    if (!timestamp) return '--:--:--';
    const date = new Date(timestamp);
    return date.toLocaleTimeString('en-US', { 
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    });
  };

  // Helper function to get action style
  const getActionStyle = (action, success) => {
    const actionStr = typeof action === 'object' ? action.action_name : action;
    const isSuccess = success !== false;
    
    if (!isSuccess) {
      return {
        color: 'text-red-400',
        bg: 'bg-red-900/20',
        borderColor: 'border-red-600/30'
      };
    }
    
    const actionUpper = actionStr?.toUpperCase() || '';
    
    if (actionUpper.includes('HOLD')) {
      return {
        color: 'text-amber-400',
        bg: 'bg-amber-900/20',
        borderColor: 'border-amber-600/30',
        icon: '⏸'
      };
    } else if (actionUpper.includes('BUY_YES') || actionUpper === 'BUY') {
      return {
        color: 'text-green-400',
        bg: 'bg-green-900/20',
        borderColor: 'border-green-600/30',
        icon: '↑'
      };
    } else if (actionUpper.includes('SELL_YES') || actionUpper === 'SELL') {
      return {
        color: 'text-red-400',
        bg: 'bg-red-900/20',
        borderColor: 'border-red-600/30',
        icon: '↓'
      };
    } else if (actionUpper.includes('BUY_NO')) {
      return {
        color: 'text-purple-400',
        bg: 'bg-purple-900/20',
        borderColor: 'border-purple-600/30',
        icon: '↓'
      };
    } else if (actionUpper.includes('SELL_NO')) {
      return {
        color: 'text-blue-400',
        bg: 'bg-blue-900/20',
        borderColor: 'border-blue-600/30',
        icon: '↑'
      };
    }
    
    return {
      color: 'text-gray-400',
      bg: 'bg-gray-800/20',
      borderColor: 'border-gray-600/30'
    };
  };

  // Helper function to format action text
  const formatAction = (action) => {
    if (!action) return 'UNKNOWN';
    
    let actionStr = typeof action === 'object' ? 
      (action.action_name || action.name || JSON.stringify(action)) : 
      action.toString();
    
    actionStr = actionStr.toUpperCase();
    
    // Simplify the display names
    if (actionStr.includes('BUY_YES_LIMIT')) return 'BUY YES';
    if (actionStr.includes('SELL_YES_LIMIT')) return 'SELL YES';
    if (actionStr.includes('BUY_NO_LIMIT')) return 'BUY NO';
    if (actionStr.includes('SELL_NO_LIMIT')) return 'SELL NO';
    if (actionStr.includes('HOLD')) return 'HOLD';
    if (actionStr.includes('CANCEL')) return 'CANCEL';
    
    return actionStr.replace(/_/g, ' ');
  };

  // Helper function to format price from cents (prefers dollars field if available)
  const formatPriceFromCents = (cents, dollarsValue, mode) => {
    if (cents === null || cents === undefined) return '--';
    
    if (mode === 'dollar') {
      // Prefer dollars field if available
      if (dollarsValue !== null && dollarsValue !== undefined && dollarsValue !== '') {
        const dollars = typeof dollarsValue === 'string' ? parseFloat(dollarsValue) : dollarsValue;
        return `$${dollars.toFixed(2)}`;
      }
      // Fallback: convert from cents
      return `$${(cents / 100).toFixed(2)}`;
    } else {
      // Cent mode: use cents directly
      return `${cents}¢`;
    }
  };

  // Helper function to format price from normalized observation values (0-1 range)
  const formatPriceFromNormalized = (dollars, mode) => {
    if (dollars === null || dollars === undefined) return '--';
    
    if (mode === 'dollar') {
      return `$${dollars.toFixed(2)}`;
    } else {
      // Convert to cents
      return `${Math.round(dollars * 100)}¢`;
    }
  };

  // Helper function to format derived prices (spread, mid-price, etc.)
  const formatDerivedPrice = (value, mode, isCentsSource) => {
    if (value === null || value === undefined) return '--';
    
    if (mode === 'dollar') {
      if (isCentsSource) {
        // Value is in cents, convert to dollars
        return `$${(value / 100).toFixed(2)}`;
      } else {
        // Value is already in dollars
        return `$${value.toFixed(2)}`;
      }
    } else {
      if (isCentsSource) {
        // Value is in cents, use directly
        return `${Math.round(value)}¢`;
      } else {
        // Value is in dollars, convert to cents
        return `${Math.round(value * 100)}¢`;
      }
    }
  };

  // Group trades, orders, and fills by trade_sequence_id
  const tradeLifecycleRows = useMemo(() => {
    const rows = new Map();
    
    // Process trades (AI decisions)
    recentFills.forEach((trade) => {
      const tradeId = trade.trade_sequence_id || trade.execution_result?.trade_sequence_id;
      if (tradeId) {
        if (!rows.has(tradeId)) {
          rows.set(tradeId, {
            trade_sequence_id: tradeId,
            trade: null,
            order: null,
            fill: null,
            timestamp: null
          });
        }
        const row = rows.get(tradeId);
        row.trade = trade;
        row.timestamp = trade.timestamp;
      }
    });
    
    // Process ALL orders (both pending and filled) from allOrders map
    allOrders.forEach((order, tradeId) => {
      if (!rows.has(tradeId)) {
        rows.set(tradeId, {
          trade_sequence_id: tradeId,
          trade: null,
          order: null,
          fill: null,
          timestamp: order.placed_at ? new Date(order.placed_at * 1000).toISOString() : null
        });
      }
      const row = rows.get(tradeId);
      row.order = order; // This now includes the status field
      if (!row.timestamp) {
        row.timestamp = order.placed_at ? new Date(order.placed_at * 1000).toISOString() : null;
      }
    });
    
    // Process fills
    actualFills.forEach((fill) => {
      const tradeId = fill.trade_sequence_id;
      if (tradeId) {
        if (!rows.has(tradeId)) {
          rows.set(tradeId, {
            trade_sequence_id: tradeId,
            trade: null,
            order: null,
            fill: null,
            timestamp: fill.timestamp
          });
        }
        const row = rows.get(tradeId);
        row.fill = fill;
        if (!row.timestamp) {
          row.timestamp = fill.timestamp;
        }
      }
    });
    
    // Convert to array and sort by timestamp (newest first)
    return Array.from(rows.values())
      .sort((a, b) => {
        const timeA = new Date(a.timestamp || 0).getTime();
        const timeB = new Date(b.timestamp || 0).getTime();
        return timeB - timeA;
      })
      .slice(0, 50); // Limit to 50 most recent for performance
  }, [recentFills, allOrders, actualFills]);

  // Track expanded rows for trade details
  const [expandedTradeRows, setExpandedTradeRows] = useState(new Set());

  const toggleTradeRowExpansion = (tradeId) => {
    const newExpanded = new Set(expandedTradeRows);
    if (newExpanded.has(tradeId)) {
      newExpanded.delete(tradeId);
    } else {
      newExpanded.add(tradeId);
    }
    setExpandedTradeRows(newExpanded);
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

      {/* Main Content - Kanban Board Layout */}
      <main className="max-w-full px-4 sm:px-6 lg:px-8 py-6 space-y-6">
        
        {/* Trader System Section with Tabs */}
        <div className="bg-gray-800 rounded-lg shadow-lg">
          {/* Tab Navigation */}
          <div className="border-b border-gray-700">
            <div className="flex space-x-1 px-4 pt-4">
              <button
                onClick={() => setActiveTab('portfolio')}
                className={`px-4 py-2 text-sm font-medium rounded-t-lg transition-colors ${
                  activeTab === 'portfolio'
                    ? 'bg-gray-700 text-white border-b-2 border-blue-500'
                    : 'text-gray-400 hover:text-gray-300 hover:bg-gray-700/50'
                }`}
              >
                Portfolio
              </button>
              <button
                onClick={() => setActiveTab('system')}
                className={`px-4 py-2 text-sm font-medium rounded-t-lg transition-colors ${
                  activeTab === 'system'
                    ? 'bg-gray-700 text-white border-b-2 border-blue-500'
                    : 'text-gray-400 hover:text-gray-300 hover:bg-gray-700/50'
                }`}
              >
                System
              </button>
            </div>
          </div>

          {/* Tab Content */}
          <div className="p-4">
            {activeTab === 'portfolio' ? (
              <>
                <h2 className="text-lg font-semibold mb-3 text-gray-100">Trader State</h2>
                
                {/* Portfolio Stats & Action Breakdown */}
                <TraderStatePanel 
                  state={traderState} 
                  executionStats={executionStats}
                  showExecutionStats={false}
                  showPositions={false}
                  showOrders={false}
                  showActionBreakdown={true}
                />
              </>
            ) : (
              <SystemHealth 
                initializationStatus={initializationStatus}
                componentHealth={componentHealth}
              />
            )}
          </div>
        </div>

        {/* Trade Lifecycle Kanban Board */}
        <div className="bg-gray-800 rounded-lg shadow-lg">
          <div className="p-4">
            {/* Kanban Header */}
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold text-gray-100 flex items-center">
                <span className="mr-2">📊</span>
                Trade Lifecycle Board
                <span className="ml-2 text-sm text-gray-400 font-normal">
                  ({tradeLifecycleRows.length} active trades)
                </span>
              </h2>
              {/* Price Mode Toggle */}
              <div className="flex items-center space-x-2">
                <span className="text-sm text-gray-400">Price Display:</span>
                <div className="inline-flex rounded-lg border border-gray-600 bg-gray-700 p-1">
                  <button
                    onClick={() => setPriceMode('dollar')}
                    className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
                      priceMode === 'dollar'
                        ? 'bg-blue-600 text-white'
                        : 'text-gray-300 hover:text-white'
                    }`}
                  >
                    Dollar
                  </button>
                  <button
                    onClick={() => setPriceMode('cent')}
                    className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
                      priceMode === 'cent'
                        ? 'bg-blue-600 text-white'
                        : 'text-gray-300 hover:text-white'
                    }`}
                  >
                    Cent
                  </button>
                </div>
              </div>
            </div>

            {/* Column Headers */}
            <div className="grid grid-cols-12 gap-2 mb-3 pb-3 border-b border-gray-700">
              <div className="col-span-6 px-3">
                <h3 className="text-sm font-semibold text-gray-300 flex items-center">
                  <span className="text-green-400 mr-2">🤖</span>
                  Trade Decision
                </h3>
              </div>
              <div className="col-span-3 px-3">
                <h3 className="text-sm font-semibold text-gray-300 flex items-center">
                  <span className="text-yellow-400 mr-2">📋</span>
                  Order
                </h3>
              </div>
              <div className="col-span-3 px-3">
                <h3 className="text-sm font-semibold text-gray-300 flex items-center">
                  <span className="text-blue-400 mr-2">💸</span>
                  Fill
                </h3>
              </div>
            </div>

            {/* Kanban Rows */}
            <div className="space-y-2 max-h-[600px] overflow-y-auto">
              {tradeLifecycleRows.length > 0 ? (
                tradeLifecycleRows.map((row) => {
                  const isExpanded = expandedTradeRows.has(row.trade_sequence_id);
                  const trade = row.trade;
                  const order = row.order;
                  const fill = row.fill;
                  const hasObservation = trade?.observation && (trade.observation.raw_array?.length > 0 || trade.observation.features);
                  
                  return (
                    <div key={row.trade_sequence_id} className="grid grid-cols-12 gap-2 group">
                      {/* Trade Decision Column (50% width) */}
                      <div className="col-span-6">
                        {trade ? (
                          <div className={`h-full ${getActionStyle(trade.action?.action_name || trade.action, trade.success).bg} 
                                        border ${getActionStyle(trade.action?.action_name || trade.action, trade.success).borderColor} 
                                        rounded-lg transition-all duration-200 hover:shadow-md`}>
                            {/* Main Trade Card */}
                            <div 
                              className={`p-3 ${hasObservation ? 'cursor-pointer' : ''}`}
                              onClick={() => hasObservation && toggleTradeRowExpansion(row.trade_sequence_id)}
                            >
                              <div className="flex items-center justify-between mb-2">
                                <div className="flex items-center space-x-2">
                                  {hasObservation && (
                                    <div className="text-gray-500">
                                      {isExpanded ? 
                                        <ChevronDownIcon className="h-3 w-3" /> : 
                                        <ChevronRightIcon className="h-3 w-3" />
                                      }
                                    </div>
                                  )}
                                  <span className="px-2 py-0.5 bg-gray-700/70 text-gray-300 rounded text-xs font-bold border border-gray-600">
                                    #{row.trade_sequence_id}
                                  </span>
                                  <span className="text-gray-500 text-xs font-mono">
                                    {formatTimestamp(trade.timestamp)}
                                  </span>
                                </div>
                                <div className="flex items-center space-x-2">
                                  {trade.execution_result && (
                                    <span className={`text-xs px-2 py-0.5 rounded font-medium ${
                                      trade.execution_result.executed ? 'bg-green-900/30 text-green-400' : 
                                      trade.execution_result.status === 'hold' ? 'bg-amber-900/30 text-amber-400' :
                                      'bg-red-900/30 text-red-400'
                                    }`}>
                                      {trade.execution_result.status || (trade.execution_result.executed ? 'executed' : 'failed')}
                                    </span>
                                  )}
                                </div>
                              </div>
                              
                              <div className="flex items-center space-x-3">
                                {getActionStyle(trade.action?.action_name || trade.action, trade.success).icon && (
                                  <span className={`${getActionStyle(trade.action?.action_name || trade.action, trade.success).color} text-lg`}>
                                    {getActionStyle(trade.action?.action_name || trade.action, trade.success).icon}
                                  </span>
                                )}
                                <div className="flex-1">
                                  <div className="flex items-center space-x-2">
                                    <span className={`${getActionStyle(trade.action?.action_name || trade.action, trade.success).color} font-semibold text-sm`}>
                                      {formatAction(trade.action?.action_name || trade.action)}
                                    </span>
                                    {trade.action?.position_size > 0 && (
                                      <span className="text-gray-400 text-xs">
                                        {trade.action.position_size} contracts
                                      </span>
                                    )}
                                    {trade.action?.limit_price !== undefined && trade.action?.limit_price !== null && (
                                      <span className="text-gray-300 text-xs">
                                        @{formatPriceFromCents(
                                          trade.action.limit_price,
                                          trade.action?.limit_price_dollars || trade.execution_result?.limit_price_dollars,
                                          priceMode
                                        )}
                                      </span>
                                    )}
                                  </div>
                                  <div className="text-xs text-gray-400 mt-1">
                                    {trade.market_ticker}
                                  </div>
                                </div>
                              </div>
                            </div>
                            
                            {/* Expanded Observation Details */}
                            {isExpanded && hasObservation && (
                              <div className="border-t border-gray-700/50 px-3 pb-3">
                                <div className="mt-3 space-y-3">
                                  {/* Orderbook State - Full Details */}
                                  {trade.observation.features?.orderbook && (
                                    <div className="p-3 bg-gray-900/50 rounded-lg border border-gray-700/50">
                                      <h4 className="text-xs font-semibold text-gray-300 mb-2 flex items-center">
                                        <span className="mr-1">📊</span>
                                        Orderbook State
                                      </h4>
                                      <div className="space-y-2">
                                        {/* YES Side */}
                                        <div className="grid grid-cols-2 gap-3 text-xs">
                                          <div>
                                            <div className="flex justify-between mb-1">
                                              <span className="text-gray-500">YES Bid:</span>
                                              <span className="text-green-400 font-mono">
                                                {formatPriceFromNormalized(trade.observation.features.orderbook.yes_bid, priceMode)}
                                              </span>
                                            </div>
                                            <div className="flex justify-between text-xxs">
                                              <span className="text-gray-600">Size:</span>
                                              <span className="text-gray-400 font-mono">
                                                {trade.observation.features.orderbook.yes_bid_size || 0}
                                              </span>
                                            </div>
                                          </div>
                                          <div>
                                            <div className="flex justify-between mb-1">
                                              <span className="text-gray-500">YES Ask:</span>
                                              <span className="text-green-400 font-mono">
                                                {formatPriceFromNormalized(trade.observation.features.orderbook.yes_ask, priceMode)}
                                              </span>
                                            </div>
                                            <div className="flex justify-between text-xxs">
                                              <span className="text-gray-600">Size:</span>
                                              <span className="text-gray-400 font-mono">
                                                {trade.observation.features.orderbook.yes_ask_size || 0}
                                              </span>
                                            </div>
                                          </div>
                                        </div>
                                        
                                        {/* NO Side */}
                                        <div className="grid grid-cols-2 gap-3 text-xs">
                                          <div>
                                            <div className="flex justify-between mb-1">
                                              <span className="text-gray-500">NO Bid:</span>
                                              <span className="text-orange-400 font-mono">
                                                {formatPriceFromNormalized(trade.observation.features.orderbook.no_bid, priceMode)}
                                              </span>
                                            </div>
                                            <div className="flex justify-between text-xxs">
                                              <span className="text-gray-600">Size:</span>
                                              <span className="text-gray-400 font-mono">
                                                {trade.observation.features.orderbook.no_bid_size || 0}
                                              </span>
                                            </div>
                                          </div>
                                          <div>
                                            <div className="flex justify-between mb-1">
                                              <span className="text-gray-500">NO Ask:</span>
                                              <span className="text-orange-400 font-mono">
                                                {formatPriceFromNormalized(trade.observation.features.orderbook.no_ask, priceMode)}
                                              </span>
                                            </div>
                                            <div className="flex justify-between text-xxs">
                                              <span className="text-gray-600">Size:</span>
                                              <span className="text-gray-400 font-mono">
                                                {trade.observation.features.orderbook.no_ask_size || 0}
                                              </span>
                                            </div>
                                          </div>
                                        </div>
                                        
                                        {/* Spreads and Mid */}
                                        <div className="pt-2 border-t border-gray-700/50">
                                          <div className="grid grid-cols-3 gap-2 text-xs">
                                            <div>
                                              <span className="text-gray-600 block">Spread</span>
                                              <span className="text-yellow-400 font-mono">
                                                {formatDerivedPrice(
                                                  (trade.observation.features.orderbook.yes_ask - trade.observation.features.orderbook.yes_bid),
                                                  priceMode,
                                                  false  // Spread is in dollars (normalized range)
                                                )}
                                              </span>
                                            </div>
                                            <div>
                                              <span className="text-gray-600 block">Mid Price</span>
                                              <span className="text-purple-400 font-mono">
                                                {formatDerivedPrice(
                                                  (trade.observation.features.orderbook.yes_bid + trade.observation.features.orderbook.yes_ask) / 2,
                                                  priceMode,
                                                  false  // Mid price is in dollars (normalized range)
                                                )}
                                              </span>
                                            </div>
                                            <div>
                                              <span className="text-gray-600 block">Depth Ratio</span>
                                              <span className="text-blue-400 font-mono">
                                                {((trade.observation.features.orderbook.yes_bid_size || 0) / 
                                                  Math.max(1, (trade.observation.features.orderbook.yes_ask_size || 0))).toFixed(2)}
                                              </span>
                                            </div>
                                          </div>
                                        </div>
                                      </div>
                                    </div>
                                  )}
                                  
                                  {/* Market Dynamics - Enhanced */}
                                  {trade.observation.features?.market_dynamics && (
                                    <div className="p-3 bg-gray-900/50 rounded-lg border border-gray-700/50">
                                      <h4 className="text-xs font-semibold text-gray-300 mb-2 flex items-center">
                                        <span className="mr-1">📈</span>
                                        Market Dynamics
                                      </h4>
                                      <div className="grid grid-cols-2 gap-3 text-xs">
                                        <div>
                                          <div className="flex justify-between mb-1">
                                            <span className="text-gray-500">Imbalance:</span>
                                            <span className={`font-mono font-medium ${
                                              trade.observation.features.market_dynamics.imbalance > 0 ? 'text-green-400' : 
                                              trade.observation.features.market_dynamics.imbalance < 0 ? 'text-red-400' : 'text-gray-400'
                                            }`}>
                                              {trade.observation.features.market_dynamics.imbalance > 0 ? '+' : ''}
                                              {(trade.observation.features.market_dynamics.imbalance * 100).toFixed(1)}%
                                            </span>
                                          </div>
                                          <div className="flex justify-between">
                                            <span className="text-gray-500">Volume Ratio:</span>
                                            <span className="text-purple-400 font-mono">
                                              {trade.observation.features.market_dynamics.volume_ratio?.toFixed(3) || '0.000'}
                                            </span>
                                          </div>
                                        </div>
                                        <div>
                                          <div className="flex justify-between mb-1">
                                            <span className="text-gray-500">Price Momentum:</span>
                                            <span className={`font-mono font-medium ${
                                              trade.observation.features.market_dynamics.price_momentum > 0 ? 'text-green-400' : 
                                              trade.observation.features.market_dynamics.price_momentum < 0 ? 'text-red-400' : 'text-gray-400'
                                            }`}>
                                              {trade.observation.features.market_dynamics.price_momentum > 0 ? '+' : ''}
                                              {trade.observation.features.market_dynamics.price_momentum?.toFixed(4) || '0.0000'}
                                            </span>
                                          </div>
                                          <div className="flex justify-between">
                                            <span className="text-gray-500">Volatility:</span>
                                            <span className="text-amber-400 font-mono">
                                              {trade.observation.features.market_dynamics.volatility?.toFixed(3) || '0.000'}
                                            </span>
                                          </div>
                                        </div>
                                      </div>
                                    </div>
                                  )}
                                  
                                  {/* Portfolio State */}
                                  {trade.observation.features?.portfolio && (
                                    <div className="p-3 bg-gray-900/50 rounded-lg border border-gray-700/50">
                                      <h4 className="text-xs font-semibold text-gray-300 mb-2 flex items-center">
                                        <span className="mr-1">💼</span>
                                        Portfolio State
                                      </h4>
                                      <div className="grid grid-cols-2 gap-3 text-xs">
                                        <div>
                                          <div className="flex justify-between mb-1">
                                            <span className="text-gray-500">Position:</span>
                                            <span className={`font-mono font-medium ${
                                              trade.observation.features.portfolio.position > 0 ? 'text-green-400' : 
                                              trade.observation.features.portfolio.position < 0 ? 'text-red-400' : 'text-gray-400'
                                            }`}>
                                              {trade.observation.features.portfolio.position > 0 ? '+' : ''}
                                              {trade.observation.features.portfolio.position}
                                            </span>
                                          </div>
                                          <div className="flex justify-between">
                                            <span className="text-gray-500">Cash Available:</span>
                                            <span className="text-blue-400 font-mono">
                                              ${(trade.observation.features.portfolio.cash_available / 100).toFixed(2)}
                                            </span>
                                          </div>
                                        </div>
                                        <div>
                                          <div className="flex justify-between mb-1">
                                            <span className="text-gray-500">Unrealized P&L:</span>
                                            <span className={`font-mono font-medium ${
                                              trade.observation.features.portfolio.unrealized_pnl >= 0 ? 'text-green-400' : 'text-red-400'
                                            }`}>
                                              {trade.observation.features.portfolio.unrealized_pnl >= 0 ? '+' : ''}
                                              ${(trade.observation.features.portfolio.unrealized_pnl / 100).toFixed(2)}
                                            </span>
                                          </div>
                                          <div className="flex justify-between">
                                            <span className="text-gray-500">Avg Entry:</span>
                                            <span className="text-purple-400 font-mono">
                                              {trade.observation.features.portfolio.avg_entry_price ? 
                                                formatPriceFromNormalized(trade.observation.features.portfolio.avg_entry_price, priceMode) : '--'}
                                            </span>
                                          </div>
                                        </div>
                                      </div>
                                    </div>
                                  )}
                                  
                                  {/* Raw Observation Vector - Collapsible */}
                                  {trade.observation.raw_vector && (
                                    <details className="p-3 bg-gray-900/30 rounded-lg border border-gray-700/50">
                                      <summary className="text-xs font-semibold text-gray-400 cursor-pointer hover:text-gray-300 select-none">
                                        <span className="mr-1">🔢</span>
                                        Raw Observation Vector ({trade.observation.raw_vector.length} dimensions)
                                      </summary>
                                      <div className="mt-2 p-2 bg-black/30 rounded border border-gray-800">
                                        <pre className="text-xxs text-gray-500 font-mono overflow-x-auto">
                                          [{trade.observation.raw_vector.map(v => v.toFixed(3)).join(', ')}]
                                        </pre>
                                      </div>
                                    </details>
                                  )}
                                </div>
                              </div>
                            )}
                          </div>
                        ) : (
                          <div className="h-full min-h-[80px] border border-gray-700/30 rounded-lg bg-gray-900/20"></div>
                        )}
                      </div>

                      {/* Order Column (25% width) */}
                      <div className="col-span-3">
                        {order ? (
                          <div className={`h-full ${order.status === 'filled' ? 'bg-green-900/20 border-green-600/30' : 'bg-yellow-900/20 border-yellow-600/30'} border rounded-lg p-3 hover:shadow-md transition-all`}>
                            <div className="space-y-2">
                              <div className="flex items-center justify-between">
                                <span className={`font-semibold text-xs ${order.status === 'filled' ? 'text-green-400' : 'text-yellow-400'}`}>
                                  {order.status === 'filled' ? 'FILLED' : 'PENDING'}
                                </span>
                                <span className="text-gray-500 text-xs">
                                  {formatTimestamp(order.filled_at || (order.placed_at ? new Date(order.placed_at * 1000).toISOString() : null))}
                                </span>
                              </div>
                              <div className="text-xs space-y-1">
                                <div className="flex justify-between">
                                  <span className="text-gray-500">Side:</span>
                                  <span className={`font-medium ${
                                    order.side === 'BUY' ? 'text-green-400' : 'text-red-400'
                                  }`}>
                                    {order.side} {order.contract_side}
                                  </span>
                                </div>
                                <div className="flex justify-between">
                                  <span className="text-gray-500">Qty:</span>
                                  <span className="text-gray-300 font-mono">{order.quantity}</span>
                                </div>
                                <div className="flex justify-between">
                                  <span className="text-gray-500">Price:</span>
                                  <span className="text-gray-300 font-mono">
                                    {formatPriceFromCents(order.limit_price, order.limit_price_dollars, priceMode)}
                                  </span>
                                </div>
                                <div className="flex justify-between">
                                  <span className="text-gray-500">Value:</span>
                                  <span className="text-gray-300 font-mono">
                                    ${((order.quantity * order.limit_price) / 100).toFixed(2)}
                                  </span>
                                </div>
                              </div>
                              <div className="pt-2 border-t border-yellow-700/30">
                                <div className="text-xs text-gray-400 truncate" title={order.ticker}>
                                  {order.ticker}
                                </div>
                              </div>
                            </div>
                          </div>
                        ) : (
                          <div className="h-full min-h-[80px] border border-gray-700/30 rounded-lg bg-gray-900/20 flex items-center justify-center">
                            <span className="text-gray-600 text-xs">No order</span>
                          </div>
                        )}
                      </div>

                      {/* Fill Column (25% width) */}
                      <div className="col-span-3">
                        {fill ? (
                          <div className="h-full bg-blue-900/20 border border-blue-600/30 rounded-lg p-3 hover:shadow-md transition-all">
                            <div className="space-y-2">
                              <div className="flex items-center justify-between">
                                <span className="text-blue-400 font-semibold text-xs">FILLED</span>
                                <span className="text-gray-500 text-xs">
                                  {formatTimestamp(fill.timestamp)}
                                </span>
                              </div>
                              <div className="text-xs space-y-1">
                                <div className="flex justify-between">
                                  <span className="text-gray-500">Side:</span>
                                  <span className={`font-medium ${
                                    fill.side === 'BUY' ? 'text-green-400' : 'text-red-400'
                                  }`}>
                                    {fill.side}
                                  </span>
                                </div>
                                <div className="flex justify-between">
                                  <span className="text-gray-500">Qty:</span>
                                  <span className="text-gray-300 font-mono">{fill.quantity || 0}</span>
                                </div>
                                <div className="flex justify-between">
                                  <span className="text-gray-500">Price:</span>
                                  <span className="text-gray-300 font-mono">
                                    {formatPriceFromCents(
                                      fill.price || fill.fill_price || 0,
                                      fill.fill_price_dollars || fill.price_dollars || fill.yes_price_dollars,
                                      priceMode
                                    )}
                                  </span>
                                </div>
                                <div className="flex justify-between">
                                  <span className="text-gray-500">Value:</span>
                                  <span className="text-gray-300 font-mono">
                                    ${(((fill.quantity || 0) * (fill.price || 0)) / 100).toFixed(2)}
                                  </span>
                                </div>
                              </div>
                              <div className="pt-2 border-t border-blue-700/30">
                                <div className="text-xs text-gray-400 truncate" title={fill.market_ticker || fill.ticker}>
                                  {fill.market_ticker || fill.ticker || 'Unknown'}
                                </div>
                              </div>
                            </div>
                          </div>
                        ) : (
                          <div className="h-full min-h-[80px] border border-gray-700/30 rounded-lg bg-gray-900/20 flex items-center justify-center">
                            <span className="text-gray-600 text-xs">Not filled</span>
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })
              ) : (
                <div className="text-center text-gray-500 py-16">
                  <div className="mb-3">
                    <span className="text-4xl">📊</span>
                  </div>
                  <div className="text-lg font-medium">No trades yet</div>
                  <div className="text-sm mt-2 text-gray-600">
                    Trade lifecycles will appear here as the AI makes trading decisions
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Open Positions Section - Keep as separate card */}
        <div className="bg-gray-800 rounded-lg shadow-lg">
          <div className="p-4">
            <div 
              className="flex items-center justify-between cursor-pointer hover:bg-gray-700/30 p-2 rounded transition-colors"
              onClick={() => toggleSection('positions')}
            >
              <h3 className="text-lg font-semibold text-gray-100 flex items-center">
                <span className="mr-2">📈</span>
                Positions
                {positionsTab === 'active' && formatPositions().length > 0 && (
                  <span className="ml-2 text-sm text-gray-400 font-normal">
                    ({formatPositions().length} active)
                  </span>
                )}
                {positionsTab === 'settled' && Object.keys(settledPositions).length > 0 && (
                  <span className="ml-2 text-sm text-gray-400 font-normal">
                    ({Object.keys(settledPositions).length} settled)
                  </span>
                )}
              </h3>
              <span className="text-gray-400">
                {collapsedSections.positions ? '▶' : '▼'}
              </span>
            </div>
            
            {!collapsedSections.positions && (
              <div className="mt-4">
                {/* Tabs */}
                <div className="flex space-x-2 mb-4 border-b border-gray-700">
                  <button
                    onClick={() => setPositionsTab('active')}
                    className={`px-4 py-2 text-sm font-medium transition-colors ${
                      positionsTab === 'active'
                        ? 'text-blue-400 border-b-2 border-blue-400'
                        : 'text-gray-400 hover:text-gray-300'
                    }`}
                  >
                    Active ({formatPositions().length})
                  </button>
                  <button
                    onClick={() => setPositionsTab('settled')}
                    className={`px-4 py-2 text-sm font-medium transition-colors ${
                      positionsTab === 'settled'
                        ? 'text-yellow-400 border-b-2 border-yellow-400'
                        : 'text-gray-400 hover:text-gray-300'
                    }`}
                  >
                    Settled ({Object.keys(settledPositions).length})
                  </button>
                </div>
                
                {/* Active Positions Tab */}
                {positionsTab === 'active' && (
                  <>
                  {formatPositions().length > 0 ? (
                  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
                    {formatPositions().map((pos, idx) => {
                      const animData = positionAnimations[pos.ticker] || {};
                      const isNew = animData.isNew;
                      const isSettled = animData.isSettled;
                      const highlightField = animData.highlightField;
                      const changeDirection = animData.changeDirection || {};
                      
                      // Get position data with all fields
                      const posData = positions[pos.ticker] || {};
                      const positionCost = posData.cost_basis || pos.costBasis || 0;
                      const feesPaid = posData.fees_paid || pos.feesPaid || 0;
                      const volume = posData.volume || pos.totalTraded || 0;
                      
                      // Determine animation classes
                      const cardClasses = [
                        "bg-gray-700/30 hover:bg-gray-700/50 rounded-lg p-3 border transition-all",
                        isNew ? "animate-new-position" : "",
                        isSettled ? "animate-settlement border-yellow-500/50" : "border-gray-700 hover:border-gray-600",
                        highlightField ? "animate-position-update" : ""
                      ].filter(Boolean).join(" ");
                      
                      // Field-specific animation classes
                      const getFieldAnimationClass = (fieldName) => {
                        const direction = changeDirection[fieldName];
                        if (direction === 'up') return 'animate-value-increase';
                        if (direction === 'down') return 'animate-value-decrease';
                        if (highlightField === fieldName) return 'animate-field-highlight animate-counter';
                        return '';
                      };
                      
                      return (
                      <div key={`${pos.ticker}-${idx}`} className={cardClasses}>
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
                        
                        <div className="space-y-2 text-xs">
                          <div className="grid grid-cols-2 gap-2">
                            <div className={getFieldAnimationClass('position')}>
                              <span className="text-gray-500 block">Contracts</span>
                              <span className="text-gray-200 font-mono">{Math.abs(pos.contracts)}</span>
                            </div>
                            <div className={getFieldAnimationClass('realized_pnl')}>
                              <span className="text-gray-500 block">P&L</span>
                              <span className={`font-mono font-medium ${
                                pos.realizedPnl >= 0 ? 'text-green-400' : 'text-red-400'
                              }`}>
                                {priceMode === 'dollar' 
                                  ? `${pos.realizedPnl >= 0 ? '+' : ''}$${(pos.realizedPnl / 100).toFixed(2)}`
                                  : `${pos.realizedPnl >= 0 ? '+' : ''}${pos.realizedPnl.toFixed(0)}¢`
                                }
                              </span>
                            </div>
                          </div>
                          
                          {/* Position Cost */}
                          <div className={getFieldAnimationClass('position_cost')}>
                            <span className="text-gray-500 block">Cost Basis</span>
                            <span className="text-purple-400 font-mono font-medium">
                              {priceMode === 'dollar' 
                                ? `$${((pos.costBasis || 0) / 100).toFixed(2)}`
                                : `${(pos.costBasis || 0).toFixed(0)}¢`
                              }
                            </span>
                          </div>
                          
                          {/* Market Exposure */}
                          <div>
                            <span className="text-gray-500 block">Market Exposure</span>
                            <span className="text-blue-400 font-mono font-medium">
                              {priceMode === 'dollar' 
                                ? `$${((pos.marketExposure || 0) / 100).toFixed(2)}`
                                : `${(pos.marketExposure || 0).toFixed(0)}¢`
                              }
                            </span>
                          </div>
                          
                          {/* Fees Paid */}
                          {feesPaid > 0 && (
                            <div className={getFieldAnimationClass('fees_paid')}>
                              <span className="text-gray-500 block">Fees Paid</span>
                              <span className="text-orange-400 font-mono font-medium">
                                {priceMode === 'dollar' 
                                  ? `$${((typeof feesPaid === 'number' ? feesPaid : parseFloat(feesPaid) || 0) / 100).toFixed(2)}`
                                  : `${(typeof feesPaid === 'number' ? feesPaid : parseFloat(feesPaid) || 0).toFixed(0)}¢`
                                }
                              </span>
                            </div>
                          )}
                          
                          {/* Volume */}
                          {volume > 0 && (
                            <div className={getFieldAnimationClass('volume')}>
                              <span className="text-gray-500 block">Volume</span>
                              <span className="text-cyan-400 font-mono font-medium">
                                {volume}
                              </span>
                            </div>
                          )}
                          
                          {/* Last Updated */}
                          {pos.lastUpdated && (
                            <div className="pt-2 border-t border-gray-700/50">
                              <span className="text-gray-500 block text-xs">Last Updated</span>
                              <span className="text-gray-400 font-mono text-xs">
                                {new Date(pos.lastUpdated).toLocaleString()}
                              </span>
                            </div>
                          )}
                          
                          {/* Settlement Date & Time to Expiration */}
                          {pos.settlementDate && pos.timeToExp && (
                            <div className="pt-2 border-t border-gray-700/50">
                              <div className="flex justify-between items-center mb-1">
                                <span className="text-gray-500">Expires:</span>
                                <span className={`font-medium ${
                                  pos.timeToExp.expired 
                                    ? 'text-red-400' 
                                    : (pos.timeToExp.days !== undefined && pos.timeToExp.days < 1)
                                    ? 'text-orange-400'
                                    : 'text-yellow-400'
                                }`}>
                                  {pos.timeToExp.display}
                                </span>
                              </div>
                              <div className="text-gray-600 text-xs">
                                {pos.settlementDate.toLocaleDateString('en-US', { 
                                  month: 'short', 
                                  day: 'numeric', 
                                  year: 'numeric' 
                                })}
                              </div>
                            </div>
                          )}
                        </div>
                      </div>
                      );
                    })}
                  </div>
                ) : (
                  <div className="text-center text-gray-500 py-8">
                    <div className="mb-2">📊</div>
                    <div>No open positions</div>
                    <div className="text-xs mt-1 text-gray-600">Positions will appear here when opened</div>
                  </div>
                )}
                </>
                )}
                
                {/* Settled Positions Tab */}
                {positionsTab === 'settled' && (
                  <>
                  {Object.keys(settledPositions).length > 0 ? (
                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
                      {Object.entries(settledPositions).map(([ticker, settlement]) => {
                        // All monetary values from API are in cents except fee_cost which is a string in dollars
                        const finalPnl = settlement.final_pnl || 0;
                        const revenue = settlement.revenue || 0;
                        const value = settlement.value || 0;
                        const yesTotalCost = settlement.yes_total_cost || 0;
                        const noTotalCost = settlement.no_total_cost || 0;
                        const yesCount = settlement.yes_count || 0;
                        const noCount = settlement.no_count || 0;
                        const feeCostStr = settlement.fee_cost || "0.0";
                        const feeCostCents = settlement.fee_cost_cents || (parseFloat(feeCostStr) * 100);
                        const marketResult = settlement.market_result || "unknown";
                        const eventTicker = settlement.event_ticker || "";
                        const settledTime = settlement.settled_time || settlement.settled_at;
                        
                        return (
                          <div 
                            key={ticker} 
                            className="bg-yellow-900/20 border border-yellow-500/30 rounded-lg p-3 animate-settlement"
                          >
                            <div className="flex justify-between items-center mb-2 pb-2 border-b border-yellow-700/50">
                              <div className="flex-1 mr-2 min-w-0">
                                <span className="font-mono text-xs text-gray-300 truncate block" title={ticker}>
                                  {ticker}
                                </span>
                                {eventTicker && eventTicker !== ticker && (
                                  <span className="font-mono text-xs text-gray-500 truncate block" title={eventTicker}>
                                    {eventTicker}
                                  </span>
                                )}
                              </div>
                              <span className={`px-2 py-1 rounded text-xs font-medium ${
                                marketResult === 'yes' 
                                  ? 'bg-green-500/20 text-green-400' 
                                  : marketResult === 'no'
                                  ? 'bg-red-500/20 text-red-400'
                                  : 'bg-yellow-500/20 text-yellow-400'
                              }`}>
                                {marketResult.toUpperCase()}
                              </span>
                            </div>
                            
                            <div className="space-y-2 text-xs">
                              {/* Final P&L - Most prominent */}
                              <div>
                                <span className="text-gray-500 block">Final P&L</span>
                                <span className={`font-mono font-bold text-lg ${
                                  finalPnl >= 0 ? 'text-green-400' : 'text-red-400'
                                }`}>
                                  {priceMode === 'dollar' 
                                    ? `${finalPnl >= 0 ? '+' : ''}$${(finalPnl / 100).toFixed(2)}`
                                    : `${finalPnl >= 0 ? '+' : ''}${finalPnl.toFixed(0)}¢`
                                  }
                                </span>
                              </div>
                              
                              {/* Revenue and Value */}
                              <div className="grid grid-cols-2 gap-2">
                                <div>
                                  <span className="text-gray-500 block">Revenue</span>
                                  <span className="text-green-400 font-mono font-medium">
                                    {priceMode === 'dollar' 
                                      ? `$${(revenue / 100).toFixed(2)}`
                                      : `${revenue.toFixed(0)}¢`
                                    }
                                  </span>
                                </div>
                                <div>
                                  <span className="text-gray-500 block">Value</span>
                                  <span className="text-blue-400 font-mono font-medium">
                                    {priceMode === 'dollar' 
                                      ? `$${(value / 100).toFixed(2)}`
                                      : `${value.toFixed(0)}¢`
                                    }
                                  </span>
                                </div>
                              </div>
                              
                              {/* Contract Counts */}
                              {(yesCount > 0 || noCount > 0) && (
                                <div className="grid grid-cols-2 gap-2">
                                  {yesCount > 0 && (
                                    <div>
                                      <span className="text-gray-500 block">YES Count</span>
                                      <span className="text-green-400 font-mono">
                                        {yesCount}
                                      </span>
                                    </div>
                                  )}
                                  {noCount > 0 && (
                                    <div>
                                      <span className="text-gray-500 block">NO Count</span>
                                      <span className="text-red-400 font-mono">
                                        {noCount}
                                      </span>
                                    </div>
                                  )}
                                </div>
                              )}
                              
                              {/* Costs */}
                              {(yesTotalCost > 0 || noTotalCost > 0) && (
                                <div className="grid grid-cols-2 gap-2">
                                  {yesTotalCost > 0 && (
                                    <div>
                                      <span className="text-gray-500 block">YES Cost</span>
                                      <span className="text-gray-300 font-mono">
                                        {priceMode === 'dollar' 
                                          ? `$${(yesTotalCost / 100).toFixed(2)}`
                                          : `${yesTotalCost.toFixed(0)}¢`
                                        }
                                      </span>
                                    </div>
                                  )}
                                  {noTotalCost > 0 && (
                                    <div>
                                      <span className="text-gray-500 block">NO Cost</span>
                                      <span className="text-gray-300 font-mono">
                                        {priceMode === 'dollar' 
                                          ? `$${(noTotalCost / 100).toFixed(2)}`
                                          : `${noTotalCost.toFixed(0)}¢`
                                        }
                                      </span>
                                    </div>
                                  )}
                                </div>
                              )}
                              
                              {/* Fee Cost */}
                              {feeCostCents > 0 && (
                                <div>
                                  <span className="text-gray-500 block">Fee Cost</span>
                                  <span className="text-orange-400 font-mono">
                                    {priceMode === 'dollar' 
                                      ? `$${(feeCostCents / 100).toFixed(2)}`
                                      : `${feeCostCents.toFixed(0)}¢`
                                    }
                                  </span>
                                </div>
                              )}
                              
                              {/* Settled Time */}
                              {settledTime && (
                                <div className="pt-2 border-t border-yellow-700/50">
                                  <span className="text-gray-500 block">Settled At</span>
                                  <span className="text-gray-400 text-xs">
                                    {new Date(settledTime).toLocaleString()}
                                  </span>
                                </div>
                              )}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  ) : (
                    <div className="text-center text-gray-500 py-8">
                      <div className="mb-2">🏆</div>
                      <div>No settled positions</div>
                      <div className="text-xs mt-1 text-gray-600">Settled positions will appear here</div>
                    </div>
                  )}
                  </>
                )}
              </div>
            )}
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