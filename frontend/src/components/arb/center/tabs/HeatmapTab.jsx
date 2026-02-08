import React, { memo } from 'react';
import HeatStripMatrix from '../../ui/MarketCandlestickChart';

const HeatmapTab = memo(({ event }) => {
  const hasCandlestick = event.candlestick_series && Object.keys(event.candlestick_series).length > 0;

  if (!hasCandlestick) {
    return (
      <div className="text-center py-12 text-gray-600">
        <p className="text-[11px]">No price history data available</p>
        <p className="text-[10px] text-gray-700 mt-1">Heatmap populates as orderbook data streams in</p>
      </div>
    );
  }

  return (
    <div className="w-full">
      <HeatStripMatrix candlestickSeries={event.candlestick_series} markets={event.markets || {}} />
    </div>
  );
});

HeatmapTab.displayName = 'HeatmapTab';

export default HeatmapTab;
