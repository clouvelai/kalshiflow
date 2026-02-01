import { useState, useEffect, useCallback, useRef } from 'react';
import { useSignalLifecycle } from './useSignalLifecycle';

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
  costData: null,
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
  // Signal lifecycle tracking
  const {
    handleLifecycleUpdate,
    handleLifecycleSnapshot,
    getLifecycle,
    summaryCounts: lifecycleSummary,
    getStatusSortPriority,
  } = useSignalLifecycle();

  // Deep agent state
  const [agentState, setAgentState] = useState(INITIAL_DEEP_AGENT_STATE);
  const [thinking, setThinking] = useState(INITIAL_THINKING);
  const [toolCalls, setToolCalls] = useState([]);
  const [trades, setTrades] = useState([]);
  const [memoryUpdates, setMemoryUpdates] = useState([]);
  const [redditPosts, setRedditPosts] = useState([]);
  const [redditSignals, setRedditSignals] = useState([]);
  const [learnings, setLearnings] = useState([]);
  const [settlements, setSettlements] = useState([]);
  const [errors, setErrors] = useState([]);
  const [costHistory, setCostHistory] = useState([]);
  const [gdeltResults, setGdeltResults] = useState([]);

  // Refs for managing max items
  const maxGdeltResults = 15;
  const maxCostHistory = 50;
  const maxToolCalls = 50;
  const maxTrades = 50;
  const maxMemoryUpdates = 20;
  const maxRedditPosts = 30;
  const maxRedditSignals = 30;
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
      durationMs: data.duration_ms || null,
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
      action: data.action || 'buy',
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
   * Handle deep_agent_cost message
   */
  const handleCost = useCallback((data) => {
    setAgentState(prev => ({
      ...prev,
      costData: {
        model: data.model,
        sessionCost: data.session_cost,
        sessionTokens: data.session_tokens,
        lastCycleCost: data.cycle_cost,
        lastCycleTokens: data.cycle_tokens,
        cycle: data.cycle,
        timestamp: data.timestamp,
      },
    }));

    setCostHistory(prev => [
      { cycle: data.cycle, cost: data.cycle_cost, tokens: data.cycle_tokens, timestamp: data.timestamp },
      ...prev,
    ].slice(0, maxCostHistory));
  }, []);

  /**
   * Handle deep_agent_gdelt_result message - GDELT news query result
   */
  const handleGdeltResult = useCallback((data) => {
    const idTerms = data.search_terms || data.actor_names || [];
    const result = {
      id: `gdelt-${data.timestamp}-${idTerms.join('-').slice(0, 30)}`,
      searchTerms: data.search_terms || [],
      actorNames: data.actor_names || [],
      windowHours: data.window_hours,
      timespan: data.timespan,
      toneFilter: data.tone_filter,
      articleCount: data.article_count || 0,
      sourceDiversity: data.source_diversity || 0,
      toneSummary: data.tone_summary || {},
      keyThemes: data.key_themes || [],
      keyPersons: data.key_persons || [],
      keyOrganizations: data.key_organizations || [],
      topArticles: data.top_articles || [],
      timeline: data.timeline || [],
      dataPoints: data.data_points || 0,
      cached: data.cached || false,
      source: data.source || 'gkg', // 'gkg', 'doc_api', 'volume_timeline', or 'events'
      cycle: data.cycle,
      timestamp: data.timestamp,
      durationMs: data.duration_ms || null,
      // Events-specific fields
      eventCount: data.event_count || 0,
      quadClassSummary: data.quad_class_summary || null,
      goldsteinSummary: data.goldstein_summary || null,
      topEventTriples: data.top_event_triples || [],
      topActors: data.top_actors || [],
      eventCodeDistribution: data.event_code_distribution || [],
      geoHotspots: data.geo_hotspots || [],
    };

    setGdeltResults(prev => [result, ...prev].slice(0, maxGdeltResults));
    setAgentState(prev => ({
      ...prev,
      gdeltQueries: (prev.gdeltQueries || 0) + 1,
    }));
  }, []);

  /**
   * Handle price_impacts_snapshot message - Restore price impacts after page refresh
   */
  /**
   * Handle deep_agent_snapshot message - Restore state after page refresh
   */
  const handleSnapshot = useCallback((data) => {
    // Restore cumulative state
    setAgentState(prev => ({
      ...prev,
      status: data.status,
      cycleCount: data.cycle_count || 0,
      tradesExecuted: data.trades_executed || 0,
      winRate: data.win_rate || 0,
      toolStats: data.tool_stats || {},
      targetEvents: data.config?.target_events || [],
    }));

    // Restore recent tool calls
    if (data.recent_tool_calls?.length > 0) {
      setToolCalls(data.recent_tool_calls.map(tc => ({
        id: `${tc.timestamp}-${tc.tool}`,
        tool: tc.tool,
        input: tc.input,
        outputPreview: tc.output_preview,
        cycle: tc.cycle,
        timestamp: tc.timestamp,
      })));
    }

    // Restore recent trades
    if (data.recent_trades?.length > 0) {
      setTrades(data.recent_trades.map(t => ({
        id: `${t.timestamp}-${t.ticker}`,
        ticker: t.ticker,
        side: t.side,
        action: t.action || 'buy',
        contracts: t.contracts,
        priceCents: t.price_cents,
        reasoning: t.reasoning,
        timestamp: t.timestamp,
        status: 'executed',
      })));
    }

    // Restore settlements from reflections
    if (data.settlements?.length > 0) {
      setSettlements(data.settlements.map(s => ({
        id: `${s.reflection_timestamp}-${s.ticker}`,
        ticker: s.ticker,
        side: s.side || null,
        contracts: s.contracts || null,
        entryPrice: s.entry_price || null,
        exitPrice: s.exit_price || null,
        pnlCents: s.pnl_cents,
        result: s.result,
        reasoning: s.reasoning || null,
        timestamp: s.reflection_timestamp,
      })));
    }

    // Restore thinking from snapshot (show most recent)
    if (data.recent_thinking?.length > 0) {
      const latestThinking = data.recent_thinking[data.recent_thinking.length - 1];
      setThinking({
        text: latestThinking.text || latestThinking,
        timestamp: latestThinking.timestamp || null,
        cycle: latestThinking.cycle || data.cycle_count || 0,
      });
    }

    // Restore learnings from snapshot
    if (data.recent_learnings?.length > 0) {
      setLearnings(data.recent_learnings.map((l, i) => ({
        id: l.id || `learning-restored-${i}`,
        content: l.content || l,
        timestamp: l.timestamp || new Date().toISOString(),
      })));
    }

    // Restore cost data from snapshot
    if (data.cost_data) {
      setAgentState(prev => ({
        ...prev,
        costData: {
          model: data.cost_data.model,
          sessionCost: data.cost_data.session_cost,
          sessionTokens: data.cost_data.session_tokens,
          lastCycleCost: null,
          lastCycleTokens: null,
          cycle: data.cost_data.cycle_count,
          timestamp: null,
        },
      }));
    }

    // Restore signal lifecycle state
    if (data.signal_lifecycle?.length > 0) {
      handleLifecycleSnapshot(data.signal_lifecycle);
    }

    // Restore GDELT results from snapshot
    if (data.gdelt_results?.length > 0) {
      setGdeltResults(data.gdelt_results.map((r, i) => ({
        id: `gdelt-restored-${i}-${(r.search_terms || []).join('-').slice(0, 30)}`,
        searchTerms: r.search_terms || [],
        windowHours: r.window_hours,
        toneFilter: r.tone_filter,
        articleCount: r.article_count || 0,
        sourceDiversity: r.source_diversity || 0,
        toneSummary: r.tone_summary || {},
        keyThemes: r.key_themes || [],
        keyPersons: r.key_persons || [],
        keyOrganizations: r.key_organizations || [],
        topArticles: r.top_articles || [],
        timeline: r.timeline || [],
        cached: r.cached || false,
        cycle: null,
        timestamp: r.timestamp,
        durationMs: null,
      })));
      // Update GDELT query count in agent state
      setAgentState(prev => ({
        ...prev,
        gdeltQueries: data.gdelt_queries?.length || data.gdelt_results.length,
      }));
    }

    console.log(`[useDeepAgent] Snapshot restored: cycle ${data.cycle_count}, ${data.trades_executed} trades, ${data.recent_thinking?.length || 0} thinking, ${data.recent_learnings?.length || 0} learnings, ${data.gdelt_results?.length || 0} GDELT results`);
  }, [handleLifecycleSnapshot]);

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
      case 'deep_agent_cost':
        handleCost(data);
        break;
      case 'deep_agent_gdelt_result':
        handleGdeltResult(data);
        break;
      case 'deep_agent_error':
        handleError(data);
        break;
      case 'deep_agent_snapshot':
        handleSnapshot(data);
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
    handleCost,
    handleGdeltResult,
    handleError,
    handleSnapshot,
  ]);

  /**
   * Track last thinking for display persistence
   * We DON'T clear thinking between cycles - user wants to see the last thought
   * The UI will show "Waiting for next cycle..." only on initial load (text === '')
   */
  // Removed auto-clear timeout - preserve last thinking for visibility

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
    setCostHistory([]);
    setGdeltResults([]);
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
    learnings,
    settlements,
    errors,
    costHistory,
    gdeltResults,
    // Signal lifecycle
    getSignalLifecycle: getLifecycle,
    lifecycleSummary,
    getStatusSortPriority,
    // Actions
    processMessage,
    resetState,
    getStats,
    // Status helpers
    isRunning: agentState.status === 'active' || agentState.status === 'started',
    isLearning: thinking.text.length > 0,
    hasGdeltResults: gdeltResults.length > 0,
  };
};

export default useDeepAgent;
