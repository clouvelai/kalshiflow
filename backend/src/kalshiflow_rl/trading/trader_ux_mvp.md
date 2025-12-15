# Trader UX Frontend MVP Plan

## Executive Summary

This document outlines the MVP implementation plan for adding a trader UX frontend to the kalshiflow application. The goal is to create a real-time dashboard that displays the RL trader in action, showing order management state, trading decisions, and orderbook events through a WebSocket-only driven architecture.

## Architecture Overview

### Core Principles
1. **Event-Driven Architecture**: All data flows via WebSocket messages (no REST API calls except health endpoint)
2. **Real-time Updates**: State changes broadcast immediately when they occur
3. **Minimal Backend Changes**: Maximum 2-3 new WebSocket message types
4. **Reuse Existing Infrastructure**: Leverage existing WebSocket manager and event bus

### System Components

```
[ActorService] --> [EventBus] --> [WebSocketManager] --> [Frontend /trader]
       |                |                  |
       v                v                  v
[OrderManager]    [Orderbook]        [WebSocket]
       |           Updates             Messages
       v                                   |
[State Changes]                            v
                                    [React Components]
```

## Backend Implementation

### 1. New WebSocket Message Types (2 messages)

#### Message Type 1: `trader_state`
Broadcasts the complete state of the MultiMarketKalshiOrderManager whenever it changes.

```json
{
  "type": "trader_state",
  "data": {
    "timestamp": 1702334567.89,
    "cash_balance": 9850.50,
    "promised_cash": 150.00,
    "portfolio_value": 10025.75,
    "positions": {
      "INXD-25JAN03": {
        "contracts": 10,
        "cost_basis": 5.50,
        "realized_pnl": 2.30,
        "unrealized_pnl": -0.45
      }
    },
    "open_orders": [
      {
        "order_id": "order_123",
        "ticker": "INXD-25JAN03",
        "side": "BUY",
        "contract_side": "YES",
        "quantity": 10,
        "limit_price": 55,
        "placed_at": 1702334560.0,
        "status": "PENDING"
      }
    ],
    "metrics": {
      "orders_placed": 45,
      "orders_filled": 38,
      "orders_cancelled": 5,
      "fill_rate": 0.84,
      "total_volume_traded": 15234.50
    }
  }
}
```

#### Message Type 2: `trader_action`
Broadcasts every trading decision made by the actor, including the observation and action taken.

```json
{
  "type": "trader_action",
  "data": {
    "timestamp": 1702334567.89,
    "market_ticker": "INXD-25JAN03",
    "sequence_number": 12345,
    "observation": {
      "yes_bid": 54,
      "yes_ask": 56,
      "no_bid": 43,
      "no_ask": 45,
      "yes_bid_size": 100,
      "yes_ask_size": 150,
      "spread": 2,
      "mid_price": 55,
      "imbalance": 0.2,
      "position": 10,
      "cash_available": 9850.50
    },
    "action": {
      "action_id": 1,
      "action_name": "BUY_YES_LIMIT",
      "quantity": 10,
      "limit_price": 55,
      "reason": "model_prediction"
    },
    "execution_result": {
      "executed": true,
      "order_id": "order_124",
      "status": "placed",
      "error": null
    }
  }
}
```

### 2. Backend Changes

#### A. Extend WebSocketManager (`websocket_manager.py`)

Add two new message types to the existing WebSocketManager:

```python
# In websocket_manager.py

@dataclass
class TraderStateMessage:
    """Trader state update message."""
    type: str = "trader_state"
    data: Dict[str, Any] = None

@dataclass  
class TraderActionMessage:
    """Trader action decision message."""
    type: str = "trader_action"
    data: Dict[str, Any] = None

# Add broadcast methods
async def broadcast_trader_state(self, state_data: Dict[str, Any]):
    """Broadcast trader state to all connected clients."""
    message = TraderStateMessage(data=state_data)
    await self._broadcast(message)
    
async def broadcast_trader_action(self, action_data: Dict[str, Any]):
    """Broadcast trader action to all connected clients."""
    message = TraderActionMessage(data=action_data)
    await self._broadcast(message)
```

#### B. Hook into KalshiMultiMarketOrderManager

Add state change notifications to the order manager:

```python
# In kalshi_multi_market_order_manager.py

class KalshiMultiMarketOrderManager:
    def __init__(self, initial_cash: float = 10000.0):
        # ... existing init ...
        self._state_change_callbacks = []
    
    def add_state_change_callback(self, callback):
        """Register callback for state changes."""
        self._state_change_callbacks.append(callback)
    
    async def _notify_state_change(self):
        """Notify all callbacks of state change."""
        state = self.get_current_state()
        for callback in self._state_change_callbacks:
            await callback(state)
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get complete current state snapshot."""
        return {
            "timestamp": time.time(),
            "cash_balance": self.cash_balance,
            "promised_cash": self.promised_cash,
            "portfolio_value": self.get_portfolio_value(),
            "positions": self.get_positions(),
            "open_orders": self.get_open_orders(),
            "metrics": self.get_metrics()
        }
```

#### C. Hook into ActorService

Add action broadcasting to the actor service:

```python
# In actor_service.py

async def _process_market_event(self, event: ActorEvent) -> None:
    """Process a single market event."""
    # ... existing processing ...
    
    # Build observation
    observation = await self._build_observation(market_ticker)
    
    # Select action
    action = await self._select_action(observation, market_ticker)
    
    # Execute action
    result = await self._execute_action(action, market_ticker, orderbook_snapshot)
    
    # Broadcast action decision
    if self._websocket_manager:
        await self._websocket_manager.broadcast_trader_action({
            "timestamp": time.time(),
            "market_ticker": market_ticker,
            "sequence_number": event.sequence_number,
            "observation": observation,
            "action": {
                "action_id": action,
                "action_name": self._get_action_name(action),
                "quantity": 10,  # Fixed for MVP
                "limit_price": result.get("limit_price"),
                "reason": "model_prediction"
            },
            "execution_result": result
        })
```

## Frontend Implementation

**IMPORTANT**: This will be implemented in the main kalshiflow frontend (`/frontend/` directory). There is only ONE frontend service that serves both the main Kalshiflow app and the RL Trader dashboard. The styling will be consistent with the existing Kalshiflow app using Tailwind CSS and the same design patterns.

### 1. New Route: `/trader`

Add a new route to the existing React application in `/frontend/src/App.jsx` for the trader dashboard. This will be a separate full-page view accessible at `/trader`.

### 2. Component Architecture

```
/frontend/src/
  |
  +-- App.jsx (add route for /trader)
  |
  +-- components/
        |
        +-- trader/  (new directory)
              |
              +-- TraderDashboard.jsx (main container)
              |     |
              |     +-- TraderStatePanel (left side)
              |     |     |
              |     |     +-- PortfolioSummary
              |     |     +-- PositionsTable
              |     |     +-- OpenOrdersTable
              |     |     +-- PerformanceMetrics
              |     |
              |     +-- ActionFeed (right side)
              |           |
              |           +-- ActionList (scrollable)
              |           +-- ActionDetail (selected action)
              |
              +-- (uses existing Layout component for consistency)
```

### 3. React Components

All components will follow the existing Kalshiflow design patterns using:
- Tailwind CSS classes matching existing components
- Dark theme (bg-gray-900, text-white) consistent with main app
- Rounded corners, shadows, and spacing from existing design system
- Connection status indicator matching existing WebSocket status display

#### A. TraderDashboard Component

```jsx
// frontend/src/components/trader/TraderDashboard.jsx
// NOTE: This integrates with the existing kalshiflow frontend

import React, { useState, useEffect } from 'react';
import Layout from '../Layout'; // Use existing Layout component
import { useWebSocket } from '../../hooks/useWebSocket'; // Consider extending existing hook
import TraderStatePanel from './TraderStatePanel';
import ActionFeed from './ActionFeed';

const TraderDashboard = () => {
  const [traderState, setTraderState] = useState(null);
  const [actions, setActions] = useState([]);
  const [connectionStatus, setConnectionStatus] = useState('connecting');
  
  const ws = useWebSocket('ws://localhost:8002/rl/ws', {
    onMessage: (message) => {
      const data = JSON.parse(message.data);
      
      switch(data.type) {
        case 'trader_state':
          setTraderState(data.data);
          break;
        case 'trader_action':
          setActions(prev => [data.data, ...prev].slice(0, 100));
          break;
        case 'connection':
          setConnectionStatus('connected');
          break;
      }
    },
    onClose: () => setConnectionStatus('disconnected'),
    onError: () => setConnectionStatus('error')
  });
  
  return (
    <Layout connectionStatus={connectionStatus}>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <header className="mb-6">
          <h1 className="text-3xl font-bold text-white">RL Trader Dashboard</h1>
          <p className="text-gray-400 mt-2">Real-time view of trading decisions and portfolio state</p>
        </header>
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <TraderStatePanel state={traderState} />
          <ActionFeed actions={actions} />
        </div>
      </div>
    </Layout>
  );
};

export default TraderDashboard;
```

#### B. TraderStatePanel Component

```jsx
// frontend/src/components/trader/TraderStatePanel.jsx
// Styled consistently with existing kalshiflow components

const TraderStatePanel = ({ state }) => {
  if (!state) {
    return (
      <div className="bg-gray-800 rounded-lg p-6 text-gray-400">
        <p>Waiting for trader state...</p>
      </div>
    );
  }
  
  return (
    <div className="space-y-4">
      {/* Portfolio Summary - matches MarketGrid card styling */}
      <div className="bg-gray-800 rounded-lg p-4 shadow-lg">
        <h2 className="text-xl font-semibold mb-3">Portfolio</h2>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <p className="text-gray-400 text-sm">Cash Available</p>
            <p className="text-2xl font-mono">${state.cash_balance.toFixed(2)}</p>
          </div>
          <div>
            <p className="text-gray-400 text-sm">Portfolio Value</p>
            <p className="text-2xl font-mono">${state.portfolio_value.toFixed(2)}</p>
          </div>
        </div>
      </div>
      
      {/* Positions */}
      <div className="bg-gray-800 rounded-lg p-4">
        <h2 className="text-xl font-semibold mb-3">Positions</h2>
        {/* Position table */}
      </div>
      
      {/* Open Orders */}
      <div className="bg-gray-800 rounded-lg p-4">
        <h2 className="text-xl font-semibold mb-3">Open Orders</h2>
        {/* Orders table */}
      </div>
    </div>
  );
};
```

#### C. ActionFeed Component

```jsx
// frontend/src/components/trader/ActionFeed.jsx

const ActionFeed = ({ actions }) => {
  const [selectedAction, setSelectedAction] = useState(null);
  
  return (
    <div className="space-y-4">
      <div className="bg-gray-800 rounded-lg p-4 h-[600px] overflow-y-auto">
        <h2 className="text-xl font-semibold mb-3">Trading Actions</h2>
        <div className="space-y-2">
          {actions.map((action, idx) => (
            <ActionItem 
              key={`${action.timestamp}-${idx}`}
              action={action}
              onClick={() => setSelectedAction(action)}
              isSelected={selectedAction === action}
            />
          ))}
        </div>
      </div>
      
      {selectedAction && (
        <ActionDetail action={selectedAction} />
      )}
    </div>
  );
};

const ActionItem = ({ action, onClick, isSelected }) => {
  const actionColors = {
    'BUY_YES_LIMIT': 'text-green-400',
    'SELL_YES_LIMIT': 'text-red-400',
    'BUY_NO_LIMIT': 'text-blue-400',
    'SELL_NO_LIMIT': 'text-orange-400',
    'HOLD': 'text-gray-400'
  };
  
  return (
    <div 
      className={`p-3 rounded cursor-pointer transition-colors ${
        isSelected ? 'bg-gray-700' : 'bg-gray-900 hover:bg-gray-800'
      }`}
      onClick={onClick}
    >
      <div className="flex justify-between items-center">
        <div className="flex items-center gap-2">
          <span className={actionColors[action.action.action_name] || 'text-white'}>
            {action.action.action_name}
          </span>
          <span className="text-sm text-gray-500">
            {action.market_ticker}
          </span>
        </div>
        <span className="text-xs text-gray-500">
          {new Date(action.timestamp * 1000).toLocaleTimeString()}
        </span>
      </div>
      {action.execution_result.executed && (
        <div className="text-xs text-gray-400 mt-1">
          Order {action.execution_result.order_id}
        </div>
      )}
    </div>
  );
};
```

### 4. WebSocket Hook

Extend the existing WebSocket infrastructure or create a dedicated hook:

```jsx
// frontend/src/hooks/useTraderWebSocket.js

import { useEffect, useRef, useState } from 'react';

export const useTraderWebSocket = (url) => {
  const ws = useRef(null);
  const [traderState, setTraderState] = useState(null);
  const [actions, setActions] = useState([]);
  const [status, setStatus] = useState('connecting');
  
  useEffect(() => {
    ws.current = new WebSocket(url);
    
    ws.current.onopen = () => {
      setStatus('connected');
    };
    
    ws.current.onmessage = (event) => {
      const message = JSON.parse(event.data);
      
      switch(message.type) {
        case 'trader_state':
          setTraderState(message.data);
          break;
        case 'trader_action':
          setActions(prev => [message.data, ...prev].slice(0, 100));
          break;
      }
    };
    
    ws.current.onclose = () => {
      setStatus('disconnected');
    };
    
    return () => {
      ws.current?.close();
    };
  }, [url]);
  
  return { traderState, actions, status };
};
```

## Deployment Considerations

### Single Frontend Service
- The trader dashboard will be deployed as part of the main kalshiflow frontend service
- No separate deployment needed - uses the same Railway service
- Accessible at `https://kalshiflow.app/trader` in production
- Both WebSocket connections (main app at `/ws` and trader at `/rl/ws`) handled by the same frontend

### Environment Configuration
- Frontend connects to the appropriate backend WebSocket endpoint based on environment
- Development: `ws://localhost:8002/rl/ws`
- Production: `wss://kalshiflow-backend.railway.app/rl/ws`

## Implementation Timeline

### Phase 1: Backend (Day 1)
1. Extend WebSocketManager with new message types (2 hours)
2. Add state change callbacks to OrderManager (2 hours)
3. Hook ActorService to broadcast actions (2 hours)
4. Test WebSocket message flow (1 hour)

### Phase 2: Frontend (Day 2)
1. Create TraderDashboard route and component (2 hours)
2. Implement TraderStatePanel with portfolio display (3 hours)
3. Implement ActionFeed with action list (2 hours)
4. Style and polish UI (1 hour)

### Phase 3: Integration & Testing (Day 3)
1. End-to-end testing with live trader (2 hours)
2. Performance optimization (batching, throttling) (2 hours)
3. Error handling and edge cases (2 hours)
4. Documentation and deployment (2 hours)

## Success Metrics

1. **Real-time Updates**: < 100ms latency from state change to UI update
2. **Data Completeness**: All order manager state visible in UI
3. **Action Visibility**: Every trading decision logged and displayed
4. **Performance**: Handle 100+ actions/minute without UI lag
5. **Reliability**: Automatic reconnection on WebSocket disconnect

## Future Enhancements (Post-MVP)

1. **Action Replay**: Ability to replay historical trading sessions
2. **Performance Charts**: P&L over time, win rate, drawdown visualization
3. **Market Depth Visualization**: Show orderbook alongside actions
4. **Multi-Market View**: Show all markets being traded simultaneously
5. **Risk Metrics**: Real-time risk exposure and position limits
6. **Action Filtering**: Filter actions by type, market, success/failure
7. **Export Functionality**: Export trading logs and performance metrics

## Technical Considerations

### Performance
- Use React.memo() for expensive components
- Implement virtual scrolling for action list if >100 items
- Throttle state updates to max 10/second if needed
- Consider using Immer for immutable state updates

### State Management
- Keep action history limited to last 100-500 items
- Use context or Zustand for trader state if needed across components
- Consider persisting some state to localStorage for page refreshes

### Error Handling
- Graceful degradation when WebSocket disconnected
- Show stale data indicators
- Implement exponential backoff for reconnection
- Log errors to monitoring service

### Testing
- Unit tests for state transformations
- Integration tests for WebSocket message handling
- E2E tests for complete trader flow
- Performance tests for high-frequency updates

## Conclusion

This MVP provides a solid foundation for visualizing the RL trader in action with minimal backend changes and a clean, real-time frontend. The architecture maintains the event-driven approach while providing comprehensive visibility into the trader's decision-making process and portfolio state.