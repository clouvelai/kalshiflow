import React from 'react';

const ObservationSpaceVisualization = ({ observation }) => {
  // Helper function to generate intensity bars
  const getIntensityBar = (value, maxBars = 10) => {
    if (value === null || value === undefined) return '░░░░░░░░░░';
    
    const normalizedValue = Math.max(0, Math.min(1, value));
    const filledBars = Math.round(normalizedValue * maxBars);
    const emptyBars = maxBars - filledBars;
    
    let bar = '';
    // Use different characters for different intensity levels
    for (let i = 0; i < filledBars; i++) {
      if (i < maxBars * 0.3) bar += '█'; // High intensity
      else if (i < maxBars * 0.7) bar += '▓'; // Medium intensity
      else bar += '▒'; // Low intensity
    }
    for (let i = 0; i < emptyBars; i++) {
      bar += '░';
    }
    
    return bar;
  };

  // Helper function to get color based on intensity
  const getIntensityColor = (value) => {
    if (value === null || value === undefined) return 'text-gray-500';
    if (value >= 0.7) return 'text-red-400';
    if (value >= 0.4) return 'text-yellow-400';
    return 'text-green-400';
  };

  // Helper function to get intensity label
  const getIntensityLabel = (value) => {
    if (value === null || value === undefined) return 'N/A';
    if (value >= 0.7) return 'HIGH';
    if (value >= 0.4) return 'MED';
    return 'LOW';
  };

  // Default empty state
  if (!observation) {
    return (
      <div className="space-y-4">
        <div className="text-gray-500 text-sm">Waiting for observation data...</div>
      </div>
    );
  }

  const { orderbook_features, market_dynamics, portfolio_state, technical_indicators } = observation;

  return (
    <div className="space-y-4">
      {/* Orderbook Features */}
      {orderbook_features && (
        <div className="space-y-2">
          <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Orderbook</h3>
          
          <div className="grid grid-cols-1 gap-2">
            <FeatureCard
              label="Spread"
              value={orderbook_features.spread}
              bar={getIntensityBar(orderbook_features.spread)}
              color={getIntensityColor(orderbook_features.spread)}
              intensity={orderbook_features.intensity || getIntensityLabel(orderbook_features.spread)}
            />
            
            <FeatureCard
              label="Bid Depth"
              value={orderbook_features.bid_depth}
              bar={getIntensityBar(orderbook_features.bid_depth)}
              color={getIntensityColor(orderbook_features.bid_depth)}
              intensity={getIntensityLabel(orderbook_features.bid_depth)}
            />
            
            <FeatureCard
              label="Ask Depth"
              value={orderbook_features.ask_depth}
              bar={getIntensityBar(orderbook_features.ask_depth)}
              color={getIntensityColor(orderbook_features.ask_depth)}
              intensity={getIntensityLabel(orderbook_features.ask_depth)}
            />
            
            {orderbook_features.imbalance !== undefined && (
              <FeatureCard
                label="Imbalance"
                value={orderbook_features.imbalance}
                bar={getIntensityBar(Math.abs(orderbook_features.imbalance))}
                color={getIntensityColor(Math.abs(orderbook_features.imbalance))}
                intensity={getIntensityLabel(Math.abs(orderbook_features.imbalance))}
                showSign={true}
              />
            )}
          </div>
        </div>
      )}

      {/* Market Dynamics */}
      {market_dynamics && (
        <div className="space-y-2">
          <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Market Dynamics</h3>
          
          <div className="grid grid-cols-1 gap-2">
            <FeatureCard
              label="Momentum"
              value={market_dynamics.momentum}
              bar={getIntensityBar(Math.abs(market_dynamics.momentum || 0))}
              color={getIntensityColor(Math.abs(market_dynamics.momentum || 0))}
              intensity={getIntensityLabel(Math.abs(market_dynamics.momentum || 0))}
              showSign={true}
            />
            
            <FeatureCard
              label="Volatility"
              value={market_dynamics.volatility}
              bar={getIntensityBar(market_dynamics.volatility)}
              color={getIntensityColor(market_dynamics.volatility)}
              intensity={getIntensityLabel(market_dynamics.volatility)}
            />
            
            <FeatureCard
              label="Activity"
              value={market_dynamics.activity_level}
              bar={getIntensityBar(market_dynamics.activity_level)}
              color={getIntensityColor(market_dynamics.activity_level)}
              intensity={getIntensityLabel(market_dynamics.activity_level)}
            />
            
            {market_dynamics.trend_strength !== undefined && (
              <FeatureCard
                label="Trend"
                value={market_dynamics.trend_strength}
                bar={getIntensityBar(market_dynamics.trend_strength)}
                color={getIntensityColor(market_dynamics.trend_strength)}
                intensity={getIntensityLabel(market_dynamics.trend_strength)}
              />
            )}
          </div>
        </div>
      )}

      {/* Portfolio State */}
      {portfolio_state && (
        <div className="space-y-2">
          <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Portfolio</h3>
          
          <div className="grid grid-cols-1 gap-2">
            <FeatureCard
              label="Cash Ratio"
              value={portfolio_state.cash_ratio}
              bar={getIntensityBar(portfolio_state.cash_ratio)}
              color={portfolio_state.cash_ratio < 0.2 ? 'text-red-400' : 
                     portfolio_state.cash_ratio < 0.5 ? 'text-yellow-400' : 'text-green-400'}
              intensity={portfolio_state.cash_ratio < 0.2 ? 'LOW' : 
                        portfolio_state.cash_ratio < 0.5 ? 'MED' : 'HIGH'}
            />
            
            <FeatureCard
              label="Exposure"
              value={portfolio_state.exposure}
              bar={getIntensityBar(portfolio_state.exposure)}
              color={getIntensityColor(portfolio_state.exposure)}
              intensity={getIntensityLabel(portfolio_state.exposure)}
            />
            
            <FeatureCard
              label="Risk Level"
              value={portfolio_state.risk_level}
              bar={getIntensityBar(portfolio_state.risk_level)}
              color={getIntensityColor(portfolio_state.risk_level)}
              intensity={getIntensityLabel(portfolio_state.risk_level)}
            />
            
            {portfolio_state.unrealized_pnl !== undefined && (
              <FeatureCard
                label="Unreal. P&L"
                value={portfolio_state.unrealized_pnl}
                bar={getIntensityBar(Math.abs(portfolio_state.unrealized_pnl) / 100)}
                color={portfolio_state.unrealized_pnl >= 0 ? 'text-green-400' : 'text-red-400'}
                intensity={portfolio_state.unrealized_pnl >= 0 ? '+' : '-'}
                format="currency"
              />
            )}
          </div>
        </div>
      )}

      {/* Technical Indicators (if available) */}
      {technical_indicators && (
        <div className="space-y-2">
          <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Technical</h3>
          
          <div className="grid grid-cols-1 gap-2">
            {technical_indicators.rsi !== undefined && (
              <FeatureCard
                label="RSI"
                value={technical_indicators.rsi}
                bar={getIntensityBar(technical_indicators.rsi / 100)}
                color={technical_indicators.rsi > 70 ? 'text-red-400' : 
                       technical_indicators.rsi < 30 ? 'text-green-400' : 'text-yellow-400'}
                intensity={technical_indicators.rsi > 70 ? 'OVERBOUGHT' : 
                          technical_indicators.rsi < 30 ? 'OVERSOLD' : 'NEUTRAL'}
                format="rsi"
              />
            )}
            
            {technical_indicators.macd_signal !== undefined && (
              <FeatureCard
                label="MACD"
                value={technical_indicators.macd_signal}
                bar={getIntensityBar(Math.abs(technical_indicators.macd_signal))}
                color={technical_indicators.macd_signal > 0 ? 'text-green-400' : 'text-red-400'}
                intensity={technical_indicators.macd_signal > 0 ? 'BULLISH' : 'BEARISH'}
                showSign={true}
              />
            )}
          </div>
        </div>
      )}
    </div>
  );
};

// Feature Card Component
const FeatureCard = ({ label, value, bar, color, intensity, showSign = false, format = 'decimal' }) => {
  // Format the displayed value
  const formatValue = (val) => {
    if (val === null || val === undefined) return 'N/A';
    
    switch (format) {
      case 'currency':
        return `$${Math.abs(val).toFixed(2)}`;
      case 'rsi':
        return val.toFixed(0);
      case 'percentage':
        return `${(val * 100).toFixed(1)}%`;
      case 'decimal':
      default:
        if (showSign && val > 0) {
          return `+${val.toFixed(3)}`;
        }
        return val.toFixed(3);
    }
  };

  return (
    <div className="bg-gray-700/50 rounded px-3 py-2 flex items-center justify-between">
      <div className="flex items-center space-x-3 flex-1">
        <span className="text-xs text-gray-400 w-20">{label}</span>
        <span className={`font-mono text-xs ${color}`}>{bar}</span>
      </div>
      <div className="flex items-center space-x-2">
        <span className="text-xs font-mono text-gray-300">
          {formatValue(value)}
        </span>
        <span className={`text-xs font-semibold ${color} w-16 text-right`}>
          {intensity}
        </span>
      </div>
    </div>
  );
};

export default ObservationSpaceVisualization;