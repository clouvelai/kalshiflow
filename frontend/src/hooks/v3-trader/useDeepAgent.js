import { useState, useEffect, useCallback, useRef } from 'react';

/**
 * Initial state for deep agent
 */
const INITIAL_DEEP_AGENT_STATE = {
  status: 'stopped',
  cycleCount: 0,
  tradesExecuted: 0,
  winRate: 0,
  pendingTrades: [],
  recentReflections: [],
  toolStats: {},
  targetEvents: [],
  redditEnabled: false,
  redditSignals: 0,
};

/**
 * Initial state for thinking stream
 */
const INITIAL_THINKING = {
  text: '',
  timestamp: null,
  cycle: 0,
};

/**
 * useDeepAgent - Hook for managing Deep Agent WebSocket events
 *
 * Listens for deep agent specific messages and maintains state:
 * - Agent thinking (real-time reasoning stream)
 * - Tool calls and results
 * - Trade executions and settlements
 * - Memory updates (learnings, strategy, mistakes)
 * - Reddit signals (if enabled)
 * - Price impacts (entity → market transformation)
 */
export const useDeepAgent = ({ useV3WebSocketState }) => {
  // Deep agent state
  const [agentState, setAgentState] = useState(INITIAL_DEEP_AGENT_STATE);
  const [thinking, setThinking] = useState(INITIAL_THINKING);
  const [toolCalls, setToolCalls] = useState([]);
  const [trades, setTrades] = useState([]);
  const [memoryUpdates, setMemoryUpdates] = useState([]);
  const [redditPosts, setRedditPosts] = useState([]);
  const [redditSignals, setRedditSignals] = useState([]);
  const [priceImpacts, setPriceImpacts] = useState([]);
  const [learnings, setLearnings] = useState([]);
  const [settlements, setSettlements] = useState([]);
  const [errors, setErrors] = useState([]);

  // Refs for managing max items
  const maxToolCalls = 50;
  const maxTrades = 50;
  const maxMemoryUpdates = 20;
  const maxRedditPosts = 30;
  const maxRedditSignals = 30;
  const maxPriceImpacts = 30;
  const maxLearnings = 50;
  const maxSettlements = 20;
  const maxErrors = 10;

  /**
   * Handle deep_agent_status message
   */
  const handleStatus = useCallback((data) => {
    setAgentState(prev => ({
      ...prev,
      status: data.status,
      config: data.config,
    }));
  }, []);

  /**
   * Handle deep_agent_cycle message
   */
  const handleCycle = useCallback((data) => {
    setAgentState(prev => ({
      ...prev,
      cycleCount: data.cycle,
      lastCyclePhase: data.phase,
    }));
  }, []);

  /**
   * Handle deep_agent_thinking message
   */
  const handleThinking = useCallback((data) => {
    setThinking({
      text: data.text,
      timestamp: data.timestamp,
      cycle: data.cycle,
    });
  }, []);

  /**
   * Handle deep_agent_tool_call message
   */
  const handleToolCall = useCallback((data) => {
    const toolCall = {
      id: `${data.timestamp}-${data.tool}`,
      tool: data.tool,
      input: data.input,
      outputPreview: data.output_preview,
      cycle: data.cycle,
      timestamp: data.timestamp,
    };

    setToolCalls(prev => [toolCall, ...prev].slice(0, maxToolCalls));
  }, []);

  /**
   * Handle deep_agent_trade message
   */
  const handleTrade = useCallback((data) => {
    const trade = {
      id: `${data.timestamp}-${data.ticker}`,
      ticker: data.ticker,
      side: data.side,
      contracts: data.contracts,
      priceCents: data.price_cents,
      reasoning: data.reasoning,
      timestamp: data.timestamp,
      status: 'executed',
    };

    setTrades(prev => [trade, ...prev].slice(0, maxTrades));
    setAgentState(prev => ({
      ...prev,
      tradesExecuted: prev.tradesExecuted + 1,
    }));
  }, []);

  /**
   * Handle deep_agent_settlement message
   */
  const handleSettlement = useCallback((data) => {
    const settlement = {
      id: `${data.timestamp}-${data.ticker}`,
      ticker: data.ticker,
      side: data.side,
      contracts: data.contracts,
      entryPrice: data.entry_price,
      exitPrice: data.exit_price,
      pnlCents: data.pnl_cents,
      result: data.result, // 'win', 'loss', 'break_even'
      reasoning: data.reasoning,
      timestamp: data.timestamp,
    };

    setSettlements(prev => [settlement, ...prev].slice(0, maxSettlements));

    // Update win rate
    setAgentState(prev => {
      const wins = prev.winRate * prev.tradesExecuted;
      const newWins = data.result === 'win' ? wins + 1 : wins;
      const newTotal = prev.tradesExecuted; // Already incremented in handleTrade
      return {
        ...prev,
        winRate: newTotal > 0 ? newWins / newTotal : 0,
      };
    });
  }, []);

  /**
   * Handle deep_agent_memory_update message
   */
  const handleMemoryUpdate = useCallback((data) => {
    const update = {
      id: `${data.timestamp}-${data.filename}`,
      filename: data.filename,
      contentPreview: data.content_preview,
      timestamp: data.timestamp,
    };

    setMemoryUpdates(prev => [update, ...prev].slice(0, maxMemoryUpdates));

    // Extract learnings from learnings.md updates
    if (data.filename === 'learnings.md') {
      const learning = {
        id: `learning-${data.timestamp}`,
        content: data.content_preview,
        timestamp: data.timestamp,
      };
      setLearnings(prev => [learning, ...prev].slice(0, maxLearnings));
    }
  }, []);

  /**
   * Handle reddit_submission message
   */
  const handleRedditSubmission = useCallback((data) => {
    const post = {
      id: data.id,
      subreddit: data.subreddit,
      title: data.title,
      url: data.url,
      score: data.score,
      timestamp: data.timestamp,
    };

    setRedditPosts(prev => [post, ...prev].slice(0, maxRedditPosts));
  }, []);

  /**
   * Handle reddit_signal message
   */
  const handleRedditSignal = useCallback((data) => {
    const signal = {
      id: data.post_id,
      subreddit: data.subreddit,
      title: data.title,
      url: data.url,
      direction: data.direction,
      strength: data.strength,
      reason: data.reason,
      relevantMarkets: data.relevant_markets,
      confidence: data.confidence,
      timestamp: data.extracted_at,
    };

    setRedditSignals(prev => [signal, ...prev].slice(0, maxRedditSignals));
    setAgentState(prev => ({
      ...prev,
      redditSignals: prev.redditSignals + 1,
    }));
  }, []);

  /**
   * Handle deep_agent_error message
   */
  const handleError = useCallback((data) => {
    const error = {
      id: `${data.timestamp}-error`,
      message: data.error,
      cycle: data.cycle,
      timestamp: data.timestamp,
    };

    setErrors(prev => [error, ...prev].slice(0, maxErrors));
  }, []);

  /**
   * Handle price_impact message - Entity sentiment transformed to market-specific impact
   */
  const handlePriceImpact = useCallback((data) => {
    const impact = {
      id: data.signal_id || `${Date.now()}-${data.market_ticker}`,
      marketTicker: data.market_ticker,
      eventTicker: data.event_ticker,
      entityId: data.entity_id,
      entityName: data.entity_name,
      sentimentScore: data.sentiment_score,
      priceImpactScore: data.price_impact_score,
      marketType: data.market_type,
      transformationLogic: data.transformation_logic,
      confidence: data.confidence,
      suggestedSide: data.suggested_side,
      subreddit: data.source_subreddit,
      postId: data.source_post_id,
      timestamp: data.created_at || Date.now(),
    };

    setPriceImpacts(prev => [impact, ...prev].slice(0, maxPriceImpacts));
    setAgentState(prev => ({
      ...prev,
      priceImpactsReceived: (prev.priceImpactsReceived || 0) + 1,
    }));
  }, []);

  /**
   * Process incoming WebSocket message
   */
  const processMessage = useCallback((type, data) => {
    switch (type) {
      case 'deep_agent_status':
        handleStatus(data);
        break;
      case 'deep_agent_cycle':
        handleCycle(data);
        break;
      case 'deep_agent_thinking':
        handleThinking(data);
        break;
      case 'deep_agent_tool_call':
        handleToolCall(data);
        break;
      case 'deep_agent_trade':
        handleTrade(data);
        break;
      case 'deep_agent_settlement':
        handleSettlement(data);
        break;
      case 'deep_agent_memory_update':
        handleMemoryUpdate(data);
        break;
      case 'reddit_submission':
        handleRedditSubmission(data);
        break;
      case 'reddit_signal':
        handleRedditSignal(data);
        break;
      case 'price_impact':
        handlePriceImpact(data);
        break;
      case 'deep_agent_error':
        handleError(data);
        break;
      default:
        // Ignore other message types
        break;
    }
  }, [
    handleStatus,
    handleCycle,
    handleThinking,
    handleToolCall,
    handleTrade,
    handleSettlement,
    handleMemoryUpdate,
    handleRedditSubmission,
    handleRedditSignal,
    handlePriceImpact,
    handleError,
  ]);

  /**
   * Clear thinking state after a delay
   */
  useEffect(() => {
    if (!thinking.text) return;

    // Clear thinking after 10 seconds of no updates
    const timeout = setTimeout(() => {
      setThinking(prev => ({
        ...prev,
        text: '',
      }));
    }, 10000);

    return () => clearTimeout(timeout);
  }, [thinking.text, thinking.timestamp]);

  /**
   * Reset state
   */
  const resetState = useCallback(() => {
    setAgentState(INITIAL_DEEP_AGENT_STATE);
    setThinking(INITIAL_THINKING);
    setToolCalls([]);
    setTrades([]);
    setMemoryUpdates([]);
    setRedditPosts([]);
    setRedditSignals([]);
    setPriceImpacts([]);
    setLearnings([]);
    setSettlements([]);
    setErrors([]);
  }, []);

  /**
   * Get stats summary
   */
  const getStats = useCallback(() => ({
    cycleCount: agentState.cycleCount,
    tradesExecuted: agentState.tradesExecuted,
    winRate: agentState.winRate,
    pendingTrades: trades.filter(t => t.status === 'pending').length,
    redditSignals: agentState.redditSignals,
    priceImpacts: agentState.priceImpactsReceived || 0,
    errors: errors.length,
  }), [agentState, trades, errors]);

  return {
    // State
    agentState,
    thinking,
    toolCalls,
    trades,
    memoryUpdates,
    redditPosts,
    redditSignals,
    priceImpacts,
    learnings,
    settlements,
    errors,
    // Actions
    processMessage,
    resetState,
    getStats,
    // Status helpers
    isRunning: agentState.status === 'active' || agentState.status === 'started',
    isLearning: thinking.text.length > 0,
    hasPriceImpacts: priceImpacts.length > 0,
  };
};

export default useDeepAgent;
