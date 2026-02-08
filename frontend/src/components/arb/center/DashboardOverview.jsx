import React, { memo } from 'react';
import ArbMetricsBar from '../layout/ArbMetricsBar';
import TradeLogPanel from '../panels/TradeLogPanel';
import PositionPanel from '../panels/PositionPanel';

const DashboardOverview = memo(({ tradingState, arbTrades, events }) => {
  return (
    <div className="p-4 space-y-4">
      <ArbMetricsBar
        tradingState={tradingState}
        arbTradeCount={arbTrades.length}
        events={events}
      />

      {arbTrades.length > 0 && (
        <TradeLogPanel arbTrades={arbTrades} />
      )}

      <PositionPanel tradingState={tradingState} events={events} />
    </div>
  );
});

DashboardOverview.displayName = 'DashboardOverview';

export default DashboardOverview;
