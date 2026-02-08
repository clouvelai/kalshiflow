import React, { memo } from 'react';
import EventUnderstandingCard from '../../panels/EventUnderstandingCard';

const OverviewTab = memo(({ event }) => {
  const hasUnderstanding = event.understanding && (
    event.understanding.trading_summary || event.understanding.event_summary ||
    (event.understanding.participants && event.understanding.participants.length > 0) ||
    (event.understanding.key_factors && event.understanding.key_factors.length > 0)
  );

  return (
    <div className="space-y-3">
      {hasUnderstanding && (
        <EventUnderstandingCard
          understanding={event.understanding}
          lifecycle={event.lifecycle || null}
          causalModel={event.causal_model || null}
          markets={event.markets || null}
        />
      )}

      {!hasUnderstanding && (
        <div className="text-center py-8 text-gray-600">
          <p className="text-[11px]">No event understanding data yet</p>
          <p className="text-[10px] text-gray-700 mt-1">Waiting for Captain analysis...</p>
        </div>
      )}
    </div>
  );
});

OverviewTab.displayName = 'OverviewTab';

export default OverviewTab;
