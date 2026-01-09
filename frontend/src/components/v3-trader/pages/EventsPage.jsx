import React, { useCallback, useMemo } from 'react';
import { useV3WebSocket } from '../../../hooks/v3-trader/useV3WebSocket';
import V3Header from '../layout/V3Header';
import EventsPanel from '../panels/EventsPanel';

/**
 * EventsPage - Dedicated page for viewing AI event research results
 *
 * Displays event research context, semantic frames, key drivers, evidence,
 * and per-market probability assessments from the agentic research system.
 */
const EventsPage = () => {
  // Handle WebSocket messages (can add toast notifications here if needed)
  const handleMessage = useCallback((type, message, context) => {
    // Future: Could add toast notifications for new research alerts
  }, []);

  const {
    wsStatus,
    currentState,
    tradingState,
    eventResearch,
    newResearchAlert,
    dismissResearchAlert,
  } = useV3WebSocket({ onMessage: handleMessage });

  // Convert eventResearch object to sorted array (newest first)
  // Filter out internal _marketIndex key
  const eventsArray = useMemo(() => {
    if (!eventResearch) return [];

    return Object.entries(eventResearch)
      .filter(([key]) => key !== '_marketIndex')
      .map(([eventTicker, data]) => ({
        eventTicker,
        ...data,
      }))
      .sort((a, b) => (b.researched_at || 0) - (a.researched_at || 0));
  }, [eventResearch]);

  // Extract balance for header
  const balance = tradingState?.balance || 0;

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-950 via-gray-900 to-gray-950">
      {/* Header with navigation */}
      <V3Header
        wsStatus={wsStatus}
        currentState={currentState}
        balance={balance}
      />

      {/* Main content */}
      <div className="max-w-7xl mx-auto px-6 py-6">
        <EventsPanel
          events={eventsArray}
          newResearchAlert={newResearchAlert}
          onDismissAlert={dismissResearchAlert}
        />
      </div>
    </div>
  );
};

export default EventsPage;
