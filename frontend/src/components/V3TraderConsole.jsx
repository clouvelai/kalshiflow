import React from 'react';
import TradingSessionPanel from './TradingSessionPanel';

// Panel components
import {
  WhaleQueuePanel,
  FollowedTradesPanel,
  DecisionAuditPanel,
  PositionListPanel,
  SettlementsPanel
} from './v3-trader/panels';

// UI components
import { SettlementToast } from './v3-trader/ui';

// Layout components
import { V3Header, V3MetricsPanel, V3ConsoleOutput } from './v3-trader/layout';

// Custom hooks
import { useV3WebSocket, useConsoleMessages } from '../hooks/v3-trader';

/**
 * V3TraderConsole - Main console component for V3 Trader
 *
 * This component orchestrates the V3 trading interface:
 * - WebSocket connection to V3 trader backend
 * - Trading state display (balance, positions, orders)
 * - Whale queue monitoring and decision audit
 * - System console output
 * - Real-time metrics display
 */
const V3TraderConsole = () => {
  // Console messages management
  const {
    messages,
    expandedMessages,
    addMessage,
    toggleMessageExpansion
  } = useConsoleMessages();

  // WebSocket connection and state management
  const {
    wsStatus,
    currentState,
    tradingState,
    lastUpdateTime,
    whaleQueue,
    processingWhaleId,
    settlements,
    newSettlement,
    dismissSettlement,
    metrics
  } = useV3WebSocket({ onMessage: addMessage });

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-950 via-gray-900 to-gray-950">
      {/* Settlement Toast - Fixed position notification */}
      <SettlementToast
        settlement={newSettlement}
        onDismiss={dismissSettlement}
      />

      {/* Header */}
      <V3Header wsStatus={wsStatus} currentState={currentState} />

      <div className="max-w-7xl mx-auto px-6 py-6">
        {/* Trading Session Panel - Unified session display with animations */}
        <TradingSessionPanel tradingState={tradingState} lastUpdateTime={lastUpdateTime} />

        {/* Whale Queue Panel - Above positions for visibility */}
        <div className="mb-6">
          <WhaleQueuePanel whaleQueue={whaleQueue} processingWhaleId={processingWhaleId} />
          <DecisionAuditPanel
            decisionHistory={whaleQueue.decision_history}
            decisionStats={whaleQueue.decision_stats}
          />
        </div>

        {/* Position List Panel - Detailed per-position P&L */}
        <PositionListPanel
          positions={tradingState?.positions_details}
          positionListener={tradingState?.position_listener}
          sessionUpdates={tradingState?.session_updates}
        />

        {/* Followed Trades - Below positions */}
        <div className="mb-6">
          <FollowedTradesPanel followedWhales={whaleQueue.followed_whales} />
        </div>

        {/* Settlements Panel - Recently closed positions */}
        <div className="mb-6">
          <SettlementsPanel settlements={settlements} />
        </div>

        <div className="grid grid-cols-12 gap-6">
          {/* Metrics Panel */}
          <div className="col-span-3">
            <V3MetricsPanel metrics={metrics} />
          </div>

          {/* Console */}
          <div className="col-span-9">
            <V3ConsoleOutput
              messages={messages}
              expandedMessages={expandedMessages}
              onToggleExpand={toggleMessageExpansion}
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default V3TraderConsole;
