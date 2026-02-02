import { useState, useCallback, useRef, useEffect } from 'react';

/**
 * Initial state for deep agent
 */
const INITIAL_DEEP_AGENT_STATE = {
  status: 'stopped',
  cycleCount: 0,
  tradesExecuted: 0,
  winCount: 0,
  winRate: 0,
  pendingTrades: [],
  recentReflections: [],
  toolStats: {},
  targetEvents: [],
  costData: null,
  lastCycleAt: null,
  cycleInterval: 120,
};

/**
 * Initial watchdog state
 */
const INITIAL_WATCHDOG_STATE = {
  restartsThisHour: 0,
  maxRestartsPerHour: 5,
  permanentlyStopped: false,
};

/**
 * Initial state for thinking stream
 */
const INITIAL_THINKING = {
  text: '',
  timestamp: null,
  cycle: 0,
  streaming: false,
};

/**
 * useDeepAgent - Hook for managing Deep Agent WebSocket events
 *
 * Listens for deep agent specific messages and maintains state:
 * - Agent thinking (real-time reasoning stream)
 * - Tool calls and results
 * - Trade executions and settlements
 * - Memory updates (learnings, strategy, mistakes)
 */
export const useDeepAgent = () => {
  // Deep agent state
  const [agentState, setAgentState] = useState(INITIAL_DEEP_AGENT_STATE);
  const [thinking, setThinking] = useState(INITIAL_THINKING);
  const [toolCalls, setToolCalls] = useState([]);
  const [trades, setTrades] = useState([]);
  const [memoryUpdates, setMemoryUpdates] = useState([]);
  const [learnings, setLearnings] = useState([]);
  const [memoryByFile, setMemoryByFile] = useState({
    'learnings.md': [],
    'strategy.md': [],
    'mistakes.md': [],
    'patterns.md': [],
    'cycle_journal.md': [],
    'golden_rules.md': [],
  });
  const [settlements, setSettlements] = useState([]);
  const [errors, setErrors] = useState([]);
  const [costHistory, setCostHistory] = useState([]);
  const [gdeltResults, setGdeltResults] = useState([]);
  const [todos, setTodos] = useState([]);
  const [redditHistoricDigest, setRedditHistoricDigest] = useState(null);
  const [activeToolCall, setActiveToolCall] = useState(null);
  const [watchdog, setWatchdog] = useState(INITIAL_WATCHDOG_STATE);
  const [newWatchdogEvent, setNewWatchdogEvent] = useState(null);
  const [cycleCountdown, setCycleCountdown] = useState(null);
  const watchdogTimerRef = useRef(null);

  // Dismiss watchdog toast
  const dismissWatchdogEvent = useCallback(() => {
    setNewWatchdogEvent(null);
    if (watchdogTimerRef.current) {
      clearTimeout(watchdogTimerRef.current);
      watchdogTimerRef.current = null;
    }
  }, []);

  // Cleanup timer on unmount
  useEffect(() => {
    return () => {
      if (watchdogTimerRef.current) clearTimeout(watchdogTimerRef.current);
    };
  }, []);

  // Ref to guard against duplicate snapshot restores (Bug 1: excessive snapshot loops)
  const lastSnapshotFingerprintRef = useRef(null);

  // Refs for managing max items
  const maxGdeltResults = 15;
  const maxCostHistory = 50;
  const maxToolCalls = 50;
  const maxTrades = 50;
  const maxMemoryUpdates = 20;
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
      lastCycleAt: Date.now(),
      ...(data.cycle_interval ? { cycleInterval: data.cycle_interval } : {}),
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
      streaming: data.streaming ?? false,
    });
  }, []);

  /**
   * Handle deep_agent_tool_start message - tool is about to execute
   */
  const handleToolStart = useCallback((data) => {
    setActiveToolCall({
      tool: data.tool,
      id: data.id,
      cycle: data.cycle,
      timestamp: data.timestamp,
    });
  }, []);

  /**
   * Handle deep_agent_tool_call message - tool has completed
   */
  const handleToolCall = useCallback((data) => {
    // Clear the active tool indicator (tool finished)
    setActiveToolCall(null);

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
      id: data.order_id || `${data.timestamp}-${data.ticker}-${Date.now()}`,
      ticker: data.ticker,
      side: data.side,
      action: data.action || 'buy',
      contracts: data.contracts,
      priceCents: data.price_cents,
      limitPriceCents: data.limit_price_cents,
      orderId: data.order_id,
      orderStatus: data.order_status || 'executed',
      reasoning: data.reasoning,
      timestamp: data.timestamp,
    };

    const wasAdded = { current: false };
    setTrades(prev => {
      // Deduplicate by order_id if available
      if (trade.orderId) {
        const exists = prev.some(t => t.orderId === trade.orderId);
        if (exists) return prev;
      }
      wasAdded.current = true;
      return [trade, ...prev].slice(0, maxTrades);
    });
    if (wasAdded.current) {
      setAgentState(prev => ({
        ...prev,
        tradesExecuted: prev.tradesExecuted + 1,
      }));
    }
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

    // Update win rate using integer win counter (avoids floating-point drift)
    setAgentState(prev => {
      const newWinCount = data.result === 'win' ? (prev.winCount || 0) + 1 : (prev.winCount || 0);
      const total = prev.tradesExecuted || 1;
      return {
        ...prev,
        winCount: newWinCount,
        winRate: total > 0 ? newWinCount / total : 0,
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

    // Group into memoryByFile
    const filename = data.filename;
    setMemoryByFile(prev => {
      const fileEntries = prev[filename] || [];
      return {
        ...prev,
        [filename]: [update, ...fileEntries].slice(0, 20),
      };
    });

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
   * Handle deep_agent_error message
   */
  const handleError = useCallback((data) => {
    const error = {
      id: `${data.timestamp}-error-${Date.now()}`,
      message: data.error,
      severity: data.severity || 'warning',
      cycle: data.cycle,
      timestamp: data.timestamp,
    };

    setErrors(prev => [error, ...prev].slice(0, maxErrors));

    // Update watchdog state and fire toast when it's a watchdog event
    if (data.watchdog_event) {
      setWatchdog({
        restartsThisHour: data.restarts_this_hour ?? 0,
        maxRestartsPerHour: data.max_restarts_per_hour ?? 5,
        permanentlyStopped: data.permanently_stopped ?? false,
      });

      const toastData = {
        message: data.error,
        severity: data.severity || 'warning',
        restartsThisHour: data.restarts_this_hour ?? 0,
        maxRestartsPerHour: data.max_restarts_per_hour ?? 5,
        permanentlyStopped: data.permanently_stopped ?? false,
        timestamp: data.timestamp,
      };
      setNewWatchdogEvent(toastData);

      // Auto-dismiss: 8s for warning, 15s for critical
      if (watchdogTimerRef.current) clearTimeout(watchdogTimerRef.current);
      const dismissMs = data.severity === 'critical' ? 15000 : 8000;
      watchdogTimerRef.current = setTimeout(() => {
        setNewWatchdogEvent(null);
        watchdogTimerRef.current = null;
      }, dismissMs);
    }
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
   * Handle deep_agent_news_intelligence message - sub-agent analysis result
   * Maps intelligence data into the same gdeltResults state so the UI displays it.
   */
  const handleNewsIntelligence = useCallback((data) => {
    const intel = data.intelligence || {};
    const meta = data.metadata || {};
    const sentiment = intel.sentiment || {};
    const sourceAnalysis = intel.source_analysis || {};
    const result = {
      id: `news-intel-${data.timestamp}-${(data.search_terms || []).join('-').slice(0, 30)}-${Date.now()}`,
      searchTerms: data.search_terms || [],
      actorNames: [],
      windowHours: null,
      timespan: null,
      toneFilter: null,
      articleCount: meta.raw_article_count || sourceAnalysis.total_articles || 0,
      sourceDiversity: sourceAnalysis.unique_sources || 0,
      toneSummary: {
        avg_tone: sentiment.avg_tone || 0,
        positive_count: 0,
        negative_count: 0,
        neutral_count: 0,
      },
      keyThemes: [],
      keyPersons: [],
      keyOrganizations: [],
      topArticles: [],
      timeline: [],
      dataPoints: 0,
      cached: meta.cached || false,
      source: 'news_intelligence',
      cycle: data.cycle,
      timestamp: data.timestamp,
      durationMs: data.duration_ms || null,
      // News intelligence specific fields
      intelligence: intel,
      contextHint: data.context_hint || '',
      status: data.status || 'unknown',
    };

    setGdeltResults(prev => [result, ...prev].slice(0, maxGdeltResults));
    setAgentState(prev => ({
      ...prev,
      gdeltQueries: (prev.gdeltQueries || 0) + 1,
    }));
  }, []);

  /**
   * Handle deep_agent_snapshot message - Restore state after page refresh
   */
  const handleSnapshot = useCallback((data) => {
    // Guard against duplicate snapshot restores (Bug 1: excessive snapshot loops).
    // Build a fingerprint from the snapshot's key fields.  If it matches the
    // last-processed snapshot we skip entirely to avoid 10-15 redundant state
    // resets on reconnect.
    const fingerprint = `${data.cycle_count}:${data.trades_executed}:${data.recent_trades?.length || 0}:${data.recent_thinking?.length || 0}`;
    if (fingerprint === lastSnapshotFingerprintRef.current) {
      return; // Already applied this exact snapshot
    }
    lastSnapshotFingerprintRef.current = fingerprint;

    // Clear streaming state on snapshot restore
    setActiveToolCall(null);

    // Restore cumulative state
    const restoredTrades = data.trades_executed || 0;
    const restoredWinRate = data.win_rate || 0;
    setAgentState(prev => ({
      ...prev,
      status: data.status,
      cycleCount: data.cycle_count || 0,
      tradesExecuted: restoredTrades,
      winCount: data.win_count || Math.round(restoredWinRate * restoredTrades),
      winRate: restoredWinRate,
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

    // Restore recent trades (deduplicated by order_id)
    if (data.recent_trades?.length > 0) {
      const seenOrderIds = new Set();
      const dedupedTrades = [];
      for (let i = 0; i < data.recent_trades.length; i++) {
        const t = data.recent_trades[i];
        const key = t.order_id || `${t.timestamp}-${t.ticker}`;
        if (seenOrderIds.has(key)) continue;
        seenOrderIds.add(key);
        dedupedTrades.push({
          id: t.order_id || `${t.timestamp}-${t.ticker}-${i}`,
          ticker: t.ticker,
          side: t.side,
          action: t.action || 'buy',
          contracts: t.contracts,
          priceCents: t.price_cents,
          limitPriceCents: t.limit_price_cents,
          orderId: t.order_id,
          orderStatus: t.order_status || 'executed',
          reasoning: t.reasoning,
          timestamp: t.timestamp,
        });
      }
      setTrades(dedupedTrades);
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

    // Restore thinking from snapshot (show most recent).
    // Guard against stale thinking from a previous session: if the latest
    // thinking references a cycle higher than the current cycle_count,
    // the backend has restarted and this thinking is stale -- clear it.
    if (data.recent_thinking?.length > 0) {
      const latestThinking = data.recent_thinking[data.recent_thinking.length - 1];
      const thinkingCycle = latestThinking.cycle || 0;
      const currentCycle = data.cycle_count || 0;

      if (thinkingCycle <= currentCycle) {
        setThinking({
          text: latestThinking.text || latestThinking,
          timestamp: latestThinking.timestamp || null,
          cycle: thinkingCycle || currentCycle,
        });
      } else {
        // Stale thinking from a previous session -- reset to empty
        setThinking(INITIAL_THINKING);
      }
    } else {
      // No thinking data in snapshot -- ensure we start clean
      setThinking(INITIAL_THINKING);
    }

    // Restore learnings from snapshot
    if (data.recent_learnings?.length > 0) {
      const restoredLearnings = data.recent_learnings.map((l, i) => ({
        id: l.id || `learning-restored-${i}`,
        content: l.content || l,
        timestamp: l.timestamp || new Date().toISOString(),
      }));
      setLearnings(restoredLearnings);

      // Also restore into memoryByFile for backward compat
      setMemoryByFile(prev => ({
        ...prev,
        'learnings.md': restoredLearnings.map(l => ({
          id: l.id,
          filename: 'learnings.md',
          contentPreview: l.content,
          timestamp: l.timestamp,
        })),
      }));
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

    // Restore GDELT results from snapshot
    if (data.gdelt_results?.length > 0) {
      setGdeltResults(data.gdelt_results.map((r, i) => {
        const isNewsIntel = r.source === 'news_intelligence';
        const intelSentiment = isNewsIntel ? (r.intelligence?.sentiment || {}) : {};
        const intelSourceAnalysis = isNewsIntel ? (r.intelligence?.source_analysis || {}) : {};
        return {
          id: `gdelt-restored-${i}-${(r.search_terms || []).join('-').slice(0, 30)}`,
          searchTerms: r.search_terms || [],
          windowHours: r.window_hours,
          toneFilter: r.tone_filter,
          articleCount: r.article_count || 0,
          sourceDiversity: isNewsIntel
            ? (intelSourceAnalysis.unique_sources || 0)
            : (r.source_diversity || 0),
          toneSummary: isNewsIntel
            ? { avg_tone: intelSentiment.avg_tone || 0, positive_count: 0, negative_count: 0, neutral_count: 0 }
            : (r.tone_summary || {}),
          keyThemes: r.key_themes || [],
          keyPersons: r.key_persons || [],
          keyOrganizations: r.key_organizations || [],
          topArticles: r.top_articles || [],
          timeline: r.timeline || [],
          cached: r.cached || false,
          source: r.source || undefined,
          cycle: null,
          timestamp: r.timestamp,
          durationMs: null,
          // News intelligence specific fields (preserved from snapshot)
          ...(isNewsIntel ? {
            intelligence: r.intelligence || {},
            status: r.status || 'unknown',
            contextHint: r.context_hint || '',
          } : {}),
        };
      }));
      // Update GDELT query count in agent state
      setAgentState(prev => ({
        ...prev,
        gdeltQueries: data.gdelt_queries?.length || data.gdelt_results.length,
      }));
    }

    // Restore watchdog state from snapshot
    if (data.watchdog) {
      setWatchdog({
        restartsThisHour: data.watchdog.restarts_this_hour ?? 0,
        maxRestartsPerHour: data.watchdog.max_restarts_per_hour ?? 5,
        permanentlyStopped: data.watchdog.permanently_stopped ?? false,
      });
    }

    // Restore TODO list from snapshot
    if (data.todos?.length > 0) {
      setTodos(data.todos);
    }

    // Restore reddit historic digest from snapshot
    if (data.reddit_historic_digest) {
      setRedditHistoricDigest(data.reddit_historic_digest);
    }

    console.log(`[useDeepAgent] Snapshot restored: cycle ${data.cycle_count}, ${data.trades_executed} trades, ${data.recent_thinking?.length || 0} thinking, ${data.recent_learnings?.length || 0} learnings, ${data.gdelt_results?.length || 0} GDELT results`);
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
      case 'deep_agent_tool_start':
        handleToolStart(data);
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
      case 'deep_agent_cost':
        handleCost(data);
        break;
      case 'deep_agent_gdelt_result':
        handleGdeltResult(data);
        break;
      case 'deep_agent_news_intelligence':
        handleNewsIntelligence(data);
        break;
      case 'deep_agent_error':
        handleError(data);
        break;
      case 'deep_agent_todos':
        if (data.items) setTodos(data.items);
        break;
      case 'reddit_historic_digest':
        // Reddit historic agent completed a digest run
        if (data) setRedditHistoricDigest(data);
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
    handleToolStart,
    handleToolCall,
    handleTrade,
    handleSettlement,
    handleMemoryUpdate,
    handleCost,
    handleGdeltResult,
    handleNewsIntelligence,
    handleError,
    handleSnapshot,
  ]);

  /**
   * Reset state
   */
  const resetState = useCallback(() => {
    // Reset the snapshot fingerprint so the next snapshot is accepted
    lastSnapshotFingerprintRef.current = null;
    setAgentState(INITIAL_DEEP_AGENT_STATE);
    setThinking(INITIAL_THINKING);
    setActiveToolCall(null);
    setToolCalls([]);
    setTrades([]);
    setMemoryUpdates([]);
    setLearnings([]);
    setMemoryByFile({
      'learnings.md': [],
      'strategy.md': [],
      'mistakes.md': [],
      'patterns.md': [],
      'cycle_journal.md': [],
      'golden_rules.md': [],
    });
    setSettlements([]);
    setErrors([]);
    setCostHistory([]);
    setGdeltResults([]);
    setRedditHistoricDigest(null);
    setCycleCountdown(null);
    setWatchdog(INITIAL_WATCHDOG_STATE);
    setNewWatchdogEvent(null);
    if (watchdogTimerRef.current) {
      clearTimeout(watchdogTimerRef.current);
      watchdogTimerRef.current = null;
    }
  }, []);

  /**
   * Countdown to next cycle (ticks every second)
   */
  useEffect(() => {
    const { lastCycleAt, cycleInterval } = agentState;
    if (!lastCycleAt || agentState.status === 'stopped') {
      setCycleCountdown(null);
      return;
    }
    const tick = () => {
      const elapsed = (Date.now() - lastCycleAt) / 1000;
      const remaining = Math.max(0, Math.ceil(cycleInterval - elapsed));
      setCycleCountdown(remaining);
    };
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, [agentState.lastCycleAt, agentState.cycleInterval, agentState.status]);

  /**
   * Get stats summary
   */
  const getStats = useCallback(() => ({
    cycleCount: agentState.cycleCount,
    tradesExecuted: agentState.tradesExecuted,
    winRate: agentState.winRate,
    pendingTrades: trades.filter(t => t.status === 'pending').length,
    errors: errors.length,
  }), [agentState, trades, errors]);

  return {
    // State
    agentState,
    thinking,
    activeToolCall,
    toolCalls,
    trades,
    memoryUpdates,
    learnings,
    memoryByFile,
    settlements,
    errors,
    costHistory,
    gdeltResults,
    todos,
    redditHistoricDigest,
    watchdog,
    newWatchdogEvent,
    // Actions
    processMessage,
    resetState,
    getStats,
    dismissWatchdogEvent,
    // Status helpers
    isRunning: agentState.status === 'active' || agentState.status === 'started',
    isLearning: thinking.text.length > 0,
    hasGdeltResults: gdeltResults.length > 0,
    cycleCountdown,
  };
};

export default useDeepAgent;
