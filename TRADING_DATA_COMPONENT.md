# TradingData Component Implementation

## Overview
The TradingData component is a horizontal dashboard panel that displays real-time trading state information from the V3 trader WebSocket connection. It sits above the System Metrics panel in the V3TraderConsole and provides at-a-glance monitoring of trading performance.

## Component Features

### Visual Design
- **Dark Theme Integration**: Matches the existing gray-900 dark theme aesthetic
- **Horizontal Layout**: Full-width panel positioned above System Metrics
- **Responsive Grid**: 2-column mobile, 4-column desktop layout
- **Visual Indicators**: 
  - Icons for each metric (DollarSign, Briefcase, ShoppingCart, FileText)
  - Color-coded changes (green for positive, red for negative)
  - Trend arrows for value changes

### Data Display

#### Primary Metrics (Always Visible)
1. **Balance** 
   - Current account balance in USD (converted from cents)
   - Change indicator with +/- and trend arrow
   - Green dollar sign icon

2. **Portfolio Value**
   - Total value of all positions in USD
   - Change indicator with trend visualization
   - Blue briefcase icon

3. **Positions Count**
   - Number of active positions
   - Change indicator for position count changes
   - Purple shopping cart icon

4. **Orders Count**
   - Number of open orders
   - Change indicator for order count changes
   - Yellow file text icon

#### Secondary Information
- **Last Sync Time**: Timestamp of last Kalshi API sync (top right)
- **Active Positions List**: Up to 3 positions shown with:
  - Market ticker (truncated with tooltip)
  - Side (YES/NO) with color coding
  - Quantity
  - Market value in USD

### WebSocket Integration

The component receives `trading_state` messages with this structure:
```javascript
{
  type: "trading_state",
  data: {
    version: 1,
    balance: 100000,         // cents
    portfolio_value: 250000, // cents
    position_count: 5,
    order_count: 3,
    positions: [...],        // array of position objects
    open_orders: [...],      // array of order objects
    sync_timestamp: 1703520000,
    changes: {
      balance_change: 5000,      // cents
      portfolio_change: -2000,   // cents
      position_count_change: 1,
      order_count_change: -1
    }
  }
}
```

## Implementation Details

### Component Structure
```jsx
<TradingData tradingState={tradingState} />
```

### State Management
- Trading state is managed in V3TraderConsole parent component
- Updates via WebSocket message handler for `trading_state` type
- Component re-renders automatically on state updates

### Formatting Functions
- **formatCurrency**: Converts cents to USD with proper formatting
- **formatTime**: Converts Unix timestamp to HH:MM:SS format
- **getChangeIndicator**: Creates visual change indicators with arrows

### Styling Approach
- Tailwind CSS classes for consistent theming
- Background: `bg-gray-900/50` with backdrop blur
- Borders: `border-gray-800` for subtle separation
- Metric cards: `bg-gray-800/30` with `border-gray-700/50`
- Responsive padding and spacing

## Testing

### Manual Testing
1. Start V3 trader: `./scripts/run-rl-trader.sh`
2. Open frontend: `npm run dev`
3. Navigate to V3 Trader Console
4. Verify TradingData panel appears above System Metrics
5. Check that values update when trading state changes

### Test Script
```bash
# Test WebSocket broadcasts
uv run python backend/scripts/test_trading_state_broadcast.py
```

### Component Test
```bash
# Run component tests
cd frontend && npm test TradingData.test.jsx
```

## File Modifications

### Modified Files
1. **frontend/src/components/V3TraderConsole.jsx**
   - Added TradingData component definition
   - Added trading state to component state
   - Added WebSocket handler for `trading_state` messages
   - Integrated TradingData above System Metrics panel
   - Added new icon imports for the component

### New Files
1. **backend/scripts/test_trading_state_broadcast.py**
   - WebSocket test script for trading_state messages

2. **frontend/src/components/TradingData.test.jsx**
   - Unit tests for TradingData component

## Usage Example

The component automatically displays when trading state is available:

```jsx
// In V3TraderConsole
const [tradingState, setTradingState] = useState(null);

// WebSocket message handler
case 'trading_state':
  if (data.data) {
    setTradingState({
      has_state: true,
      ...data.data
    });
  }
  break;

// In render
<div className="mb-6">
  <TradingData tradingState={tradingState} />
</div>
```

## Visual Features

### Empty State
When no trading data is available, displays:
- "Trading Data" header
- "No data available" message
- Maintains consistent height to prevent layout shift

### Active State
When trading data is available, displays:
- 4 primary metric cards in a grid
- Real-time value updates with change indicators
- Color-coded trends (green up, red down)
- Last sync timestamp
- Optional positions list for active trades

### Responsive Design
- Mobile (2 columns): Balance/Portfolio on top, Positions/Orders below
- Desktop (4 columns): All metrics in single row
- Positions list adapts to available space

## Future Enhancements

Potential improvements for future iterations:
1. Add sparkline charts for balance/portfolio trends
2. Include P&L calculations and percentages
3. Add click-to-expand for full positions/orders lists
4. Implement real-time animation for value changes
5. Add export functionality for trading data
6. Include historical comparison (day/week/month)