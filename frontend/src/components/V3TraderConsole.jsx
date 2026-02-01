import React, { useCallback } from 'react';
import TradingSessionPanel from './TradingSessionPanel';

// Panel components
import {
  PositionListPanel,
  SettlementsPanel,
  TradeProcessingPanel,
  DeepAgentPanel,
} from './v3-trader/panels';

// UI components
import { SettlementToast, OrderFillToast, OrderCancelledToast, ResearchAlertToast } from './v3-trader/ui';

// Layout components
import { V3Header, V3MetricsPanel, V3ConsoleOutput } from './v3-trader/layout';

// Custom hooks
import { useV3WebSocket, useConsoleMessages, useDeepAgent } from '../hooks/v3-trader';

/**
 * V3TraderConsole - Main console component for V3 Trader
 *
 * This component orchestrates the V3 trading interface:
 * - WebSocket connection to V3 trader backend
 * - Trading state display (balance, positions, orders)
 * - Trade processing stats and recent trades
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

  // Deep agent state management (stats-only mode needs minimal state)
  const {
    agentState,
    trades: deepAgentTrades,
    settlements: deepAgentSettlements,
    processMessage: processDeepAgentMessage,
    isRunning: deepAgentIsRunning,
    isLearning: deepAgentIsLearning,
  } = useDeepAgent({ useV3WebSocketState: true });

  // Message handler that routes to both console AND deep agent processor
  const handleWebSocketMessage = useCallback((type, message, context) => {
    // Route deep_agent_* messages to the deep agent processor
    if (type.startsWith('deep_agent_')) {
      processDeepAgentMessage(type, message);
    }
    // Always add to console messages
    addMessage(type, message, context);
  }, [processDeepAgentMessage, addMessage]);

  // WebSocket connection and state management
  const {
    wsStatus,
    currentState,
    tradingState,
    lastUpdateTime,
    tradeProcessing,
    strategyStatus,
    settlements,
    newSettlement,
    dismissSettlement,
    newOrderFill,
    dismissOrderFill,
    newTtlCancellation,
    dismissTtlCancellation,
    newResearchAlert,
    dismissResearchAlert,
    metrics,
    // Extraction data
    extractions,
    marketSignals,
    eventConfigs,
    entityRedditPosts,
  } = useV3WebSocket({ onMessage: handleWebSocketMessage });

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-950 via-gray-900 to-gray-950">
      {/* Settlement Toast - Fixed position notification */}
      <SettlementToast
        settlement={newSettlement}
        onDismiss={dismissSettlement}
      />

      {/* Order Fill Toast - Shows when orders fill */}
      <OrderFillToast
        fill={newOrderFill}
        onDismiss={dismissOrderFill}
      />

      {/* Order Cancelled Toast - Shows when orders expire due to TTL */}
      <OrderCancelledToast
        cancellation={newTtlCancellation}
        onDismiss={dismissTtlCancellation}
      />

      {/* Research Alert Toast - Shows when AI finds high-confidence mispricing */}
      <ResearchAlertToast
        alert={newResearchAlert}
        onDismiss={dismissResearchAlert}
      />

      {/* Header */}
      <V3Header
        wsStatus={wsStatus}
        currentState={currentState}
        balance={tradingState?.balance || 0}
        minTraderCash={tradingState?.min_trader_cash || 0}
      />

      <div className="max-w-7xl mx-auto px-6 py-6">
        {/* Trading Session Panel - Unified session display with animations */}
        <TradingSessionPanel tradingState={tradingState} lastUpdateTime={lastUpdateTime} />

        {/* Trade Processing Panel - Stats and recent tracked trades */}
        <div className="mb-6">
          <TradeProcessingPanel tradeProcessing={tradeProcessing} />
        </div>

        {/* Deep Agent Panel - Stats-only compact view when deep_agent strategy is active */}
        {strategyStatus?.strategies?.deep_agent && (
          <DeepAgentPanel
            statsOnly={true}
            agentState={{
              ...agentState,
              postsProcessed: entityRedditPosts?.length || 0,
              extractionsTotal: extractions?.length || 0,
              extractionsMarketSignals: marketSignals?.length || 0,
              eventsTracked: eventConfigs?.length || 0,
            }}
            trades={deepAgentTrades}
            settlements={deepAgentSettlements}
            isRunning={deepAgentIsRunning || (strategyStatus?.strategies?.deep_agent?.running ?? false)}
            isLearning={deepAgentIsLearning}
            costData={agentState.costData}
          />
        )}

        {/* Position List Panel - Detailed per-position P&L */}
        <PositionListPanel
          positions={tradingState?.positions_details}
          positionListener={tradingState?.position_listener}
          sessionUpdates={tradingState?.session_updates}
        />

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
