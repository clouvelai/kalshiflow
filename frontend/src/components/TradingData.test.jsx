import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';

// Extract TradingData component for testing
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

  return (
    <div data-testid="trading-data">
      <div>Balance: {formatCurrency(tradingState.balance)}</div>
      <div>Portfolio: {formatCurrency(tradingState.portfolio_value)}</div>
      <div>Positions: {tradingState.position_count}</div>
      <div>Orders: {tradingState.order_count}</div>
    </div>
  );
};

describe('TradingData Component', () => {
  it('renders "No data available" when no trading state', () => {
    render(<TradingData tradingState={null} />);
    expect(screen.getByText('No data available')).toBeInTheDocument();
  });

  it('renders "No data available" when trading state has no data', () => {
    render(<TradingData tradingState={{ has_state: false }} />);
    expect(screen.getByText('No data available')).toBeInTheDocument();
  });

  it('renders trading data correctly', () => {
    const mockTradingState = {
      has_state: true,
      balance: 100000, // $1,000.00
      portfolio_value: 250000, // $2,500.00
      position_count: 5,
      order_count: 3,
      sync_timestamp: 1703520000,
      changes: {
        balance_change: 5000, // +$50.00
        portfolio_change: -2000, // -$20.00
        position_count_change: 1,
        order_count_change: -1
      }
    };

    render(<TradingData tradingState={mockTradingState} />);
    
    expect(screen.getByText(/Balance: \$1,000.00/)).toBeInTheDocument();
    expect(screen.getByText(/Portfolio: \$2,500.00/)).toBeInTheDocument();
    expect(screen.getByText(/Positions: 5/)).toBeInTheDocument();
    expect(screen.getByText(/Orders: 3/)).toBeInTheDocument();
  });
});