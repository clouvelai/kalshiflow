import { useState, useEffect, useRef, useCallback } from 'react';

/**
 * Initial state for metrics
 */
const INITIAL_METRICS = {
  markets_connected: 0,
  tracked_markets: 0,
  subscribed_markets: 0,
  snapshots_received: 0,
  deltas_received: 0,
  uptime: 0,
  health: 'unknown',
  ping_health: 'unknown',
  last_ping_age: null,
  api_connected: false,
  api_url: null,
  ws_url: null,
  signal_aggregator: null
};

/**
 * Initial state for trade processing
 */
const INITIAL_TRADE_PROCESSING = {
  recent_trades: [],
  stats: { trades_seen: 0, trades_filtered: 0, trades_tracked: 0, filter_rate_percent: 0 },
  decisions: { detected: 0, executed: 0, rate_limited: 0, skipped: 0, reentries: 0 },
  decision_history: [],
  last_updated: null,
  timestamp: null
};

/**
 * Initial state for strategy status (Trading Strategies Panel)
 */
const INITIAL_STRATEGY_STATUS = {
  coordinator: {
    running: false,
    uptime_seconds: 0,
    strategies_running: 0,
    rate_limiter: {
      tokens_available: 0,
      capacity: 0,
      refill_rate_per_minute: 0,
      utilization_percent: 0
    }
  },
  strategies: {},
  recent_decisions: [],
  last_updated: null,
  timestamp: null
};

/**
 * Initial state for event research (Agentic Research results)
 * Maps event_ticker -> research result
 */
const INITIAL_EVENT_RESEARCH = {};

/**
 * Initial state for event research feed (EventResearchAgent)
 */
const INITIAL_RESEARCH_FEED = {
  strategies: {},   // event_ticker -> ResearchStrategy
  feeds: {},        // event_ticker -> [ResearchItem, ...]
  signals: {},      // event_ticker -> [ResearchSignal, ...]
  batches: {},      // event_ticker -> last ResearchBatch
};

/**
 * Initial state for entity trading (Reddit Entity Pipeline)
 */
const INITIAL_ENTITY_STATE = {
  redditPosts: [],
  entities: [],
  priceImpacts: [],
  stats: {
    postsProcessed: 0,
    entitiesExtracted: 0,
    signalsGenerated: 0,
    indexSize: 0,
  },
  isActive: false,
};

/**
 * Initial state for entity index (Canonical entities with aliases)
 */
const INITIAL_ENTITY_INDEX = {
  totalEntities: 0,
  entities: [],
  entityLookup: {}, // entity_id -> entity for quick lookups
  timestamp: null,
};

/**
 * useV3WebSocket - Hook for managing V3 Trader WebSocket connection
 */
export const useV3WebSocket = ({ onMessage }) => {
  const [wsStatus, setWsStatus] = useState('disconnected');
  const [currentState, setCurrentState] = useState('UNKNOWN');
  const [tradingState, setTradingState] = useState(null);
  const [lastUpdateTime, setLastUpdateTime] = useState(null);
  const [tradeProcessing, setTradeProcessing] = useState(INITIAL_TRADE_PROCESSING);
  const [strategyStatus, setStrategyStatus] = useState(INITIAL_STRATEGY_STATUS);
  const [eventResearch, setEventResearch] = useState(INITIAL_EVENT_RESEARCH);
  const [newResearchAlert, setNewResearchAlert] = useState(null);

  // EventResearchAgent visible feed state
  const [researchStrategies, setResearchStrategies] = useState({});
  const [researchFeeds, setResearchFeeds] = useState({});
  const [researchSignals, setResearchSignals] = useState({});
  const [researchBatches, setResearchBatches] = useState({});
  const [settlements, setSettlements] = useState([]);

  // Entity Trading state (Reddit Entity Pipeline)
  const [entityRedditPosts, setEntityRedditPosts] = useState([]);
  const [entityExtractions, setEntityExtractions] = useState([]);
  const [entityPriceImpacts, setEntityPriceImpacts] = useState([]);
  const [entityStats, setEntityStats] = useState(INITIAL_ENTITY_STATE.stats);
  const [entitySystemActive, setEntitySystemActive] = useState(false);

  // Entity Index state (Canonical entities with aliases)
  const [entityIndex, setEntityIndex] = useState(INITIAL_ENTITY_INDEX);
  const [newSettlement, setNewSettlement] = useState(null);
  const [newOrderFill, setNewOrderFill] = useState(null);
  const [newTtlCancellation, setNewTtlCancellation] = useState(null);
  const [metrics, setMetrics] = useState(INITIAL_METRICS);

  // Trade flow state for Market Signal Table
  // tradeFlowStates: Map of ticker -> { yes_trades, no_trades, yes_ratio, price_drop, total_trades, etc. }
  const [tradeFlowStates, setTradeFlowStates] = useState({});
  // tradePulses: Map of ticker -> { side: 'yes'|'no', ts: timestamp } for pulse animation
  const [tradePulses, setTradePulses] = useState({});
  // eventTrades: Map of event_ticker -> array of recent trades (max 50)
  const [eventTrades, setEventTrades] = useState({});

  const wsRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);
  const currentStateRef = useRef(currentState);
  const lastPingRef = useRef(Date.now());
  const heartbeatIntervalRef = useRef(null);

  // Toast timeout refs for proper cleanup on reconnect/unmount
  const orderFillTimeoutRef = useRef(null);
  const ttlCancellationTimeoutRef = useRef(null);
  const settlementTimeoutRef = useRef(null);
  const researchAlertTimeoutRef = useRef(null);

  // Keep currentStateRef in sync
  useEffect(() => {
    currentStateRef.current = currentState;
  }, [currentState]);

  const connectWebSocket = useCallback(() => {
    // Clear any pending reconnect timeout first to prevent timeout accumulation
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    if (wsRef.current?.readyState === WebSocket.OPEN ||
        wsRef.current?.readyState === WebSocket.CONNECTING) {
      return;
    }

    try {
      const backendPort = import.meta.env.VITE_V3_BACKEND_PORT || import.meta.env.VITE_BACKEND_PORT || '8005';
      const ws = new WebSocket(`ws://localhost:${backendPort}/v3/ws`);

      ws.onopen = () => {
        setWsStatus('connected');
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          handleMessage(data, ws);
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      ws.onerror = (error) => {
        onMessage?.('error', 'WebSocket error occurred', { error: error.message });
        setWsStatus('error');
      };

      ws.onclose = (event) => {
        setWsStatus('disconnected');

        if (event.code !== 1000) {
          onMessage?.('warning', 'Disconnected from TRADER V3', {
            icon: 'disconnect',
            code: event.code,
            reason: event.reason || 'Connection lost'
          });
        }

        wsRef.current = null;

        reconnectTimeoutRef.current = setTimeout(() => {
          connectWebSocket();
        }, 3000);
      };

      wsRef.current = ws;
    } catch (error) {
      onMessage?.('error', `Failed to connect: ${error.message}`);
      setWsStatus('error');
    }
  }, [onMessage]);

  const handleMessage = useCallback((data, ws) => {
    switch (data.type) {
      case 'system_activity':
        if (data.data) {
          const { activity_type, message, metadata, timestamp } = data.data;

          if (activity_type === 'state_transition' && metadata) {
            if (metadata.to_state) {
              setCurrentState(metadata.to_state);
            }

            if (metadata.to_state === 'trading_client_connect' && metadata.api_url) {
              setMetrics(prev => ({
                ...prev,
                api_url: metadata.api_url,
                api_connected: true
              }));
            }
          }

          // Handle order fill toast notification
          if (activity_type === 'order_fill' && metadata) {
            setNewOrderFill({
              ticker: metadata.ticker,
              action: metadata.action,
              side: metadata.side,
              count: metadata.count,
              price_cents: metadata.price_cents,
              total_cents: metadata.total_cents
            });
            // Auto-dismiss after 5 seconds (with proper cleanup)
            if (orderFillTimeoutRef.current) {
              clearTimeout(orderFillTimeoutRef.current);
            }
            orderFillTimeoutRef.current = setTimeout(() => setNewOrderFill(null), 5000);
          }

          // Handle TTL cancellation toast notification
          if (activity_type === 'orders_cancelled_ttl' && metadata) {
            setNewTtlCancellation({
              count: metadata.count,
              tickers: metadata.tickers || [],
              ttl_seconds: metadata.ttl_seconds
            });
            // Auto-dismiss after 5 seconds (with proper cleanup)
            if (ttlCancellationTimeoutRef.current) {
              clearTimeout(ttlCancellationTimeoutRef.current);
            }
            ttlCancellationTimeoutRef.current = setTimeout(() => setNewTtlCancellation(null), 5000);
          }

          let messageType = 'activity';
          if (metadata?.severity) {
            messageType = metadata.severity;
          } else if (activity_type === 'sync' && currentStateRef.current === 'ERROR') {
            messageType = 'info';
          }

          onMessage?.(messageType, message, {
            activity_type,
            timestamp,
            metadata,
            state: metadata?.to_state || currentStateRef.current
          });
        }
        break;

      case 'trading_state':
        if (data.data) {
          setLastUpdateTime(Math.floor(Date.now() / 1000));

          // Use functional setState with version comparison to skip redundant updates
          setTradingState(prev => {
            // Skip update if version hasn't changed (avoids unnecessary re-renders)
            if (prev?.version === data.data.version) {
              return prev;
            }
            return {
              has_state: true,
              version: data.data.version,
              balance: data.data.balance,
              portfolio_value: data.data.portfolio_value,
              position_count: data.data.position_count,
              order_count: data.data.order_count,
              positions: data.data.positions,
              open_orders: data.data.open_orders,
              sync_timestamp: data.data.sync_timestamp,
              changes: data.data.changes,
              order_group: data.data.order_group,
              pnl: data.data.pnl,
              positions_details: data.data.positions_details || []
            };
          });

          // Handle settlements with reference stability
          if (data.data.settlements !== undefined) {
            const newSettlements = data.data.settlements || [];
            setSettlements(prevSettlements => {
              // Only update if actually changed (length or first item ticker)
              if (prevSettlements.length === newSettlements.length &&
                  prevSettlements[0]?.ticker === newSettlements[0]?.ticker) {
                return prevSettlements;
              }
              if (newSettlements.length > prevSettlements.length && newSettlements[0]) {
                setNewSettlement(newSettlements[0]);
                // Auto-dismiss after 5 seconds (with proper cleanup)
                if (settlementTimeoutRef.current) {
                  clearTimeout(settlementTimeoutRef.current);
                }
                settlementTimeoutRef.current = setTimeout(() => setNewSettlement(null), 5000);
              }
              return newSettlements;
            });
          }

          // Handle event_research from initial snapshot (Events tab persistence)
          // This merges cached research results so new clients see research
          // that was broadcast before they connected
          if (data.data.event_research) {
            setEventResearch(prev => {
              const marketIndex = { ...(prev._marketIndex || {}) };
              let hasNewData = false;

              // Merge each event's research into state
              Object.entries(data.data.event_research).forEach(([eventTicker, researchData]) => {
                // Skip if we already have this event (real-time takes precedence)
                if (prev[eventTicker]) {
                  return;
                }
                hasNewData = true;

                // Index each market assessment by ticker
                (researchData.markets || []).forEach(market => {
                  marketIndex[market.ticker] = {
                    eventTicker,
                    ...market,
                    eventTitle: researchData.event_title,
                    eventCategory: researchData.event_category,
                    primaryDriver: researchData.primary_driver,
                    evidenceSummary: researchData.evidence_summary,
                    researchedAt: researchData.researched_at,
                  };
                });
              });

              // Only update if we have new data to avoid unnecessary re-renders
              if (!hasNewData) {
                return prev;
              }

              return {
                ...prev,
                ...data.data.event_research,
                _marketIndex: marketIndex,
              };
            });

            console.log(
              `[useV3WebSocket] Loaded ${Object.keys(data.data.event_research).length} cached event research results from snapshot`
            );
          }
        }
        break;

      case 'connection':
        // Initial connection acknowledgment
        break;

      case 'history_replay':
        if (data.data.transitions) {
          data.data.transitions.forEach(transition => {
            const fromState = transition.from_state || 'unknown';
            const toState = transition.to_state || transition.state;
            setCurrentState(toState);
            onMessage?.('state', transition.message, {
              state: toState,
              from_state: fromState,
              to_state: toState,
              context: transition.context,
              timestamp: transition.timestamp,
              metadata: transition.metadata,
              is_history: true
            });
          });
        }
        break;

      case 'state_transition': {
        const fromState = data.data.from_state || currentStateRef.current;
        const toState = data.data.to_state || data.data.state;
        setCurrentState(toState);

        if (toState === 'trading_client_connect' && data.data.metadata?.api_url) {
          setMetrics(prev => ({
            ...prev,
            api_url: data.data.metadata.api_url,
            api_connected: true
          }));
        }

        if (['ready', 'calibrating', 'acting', 'trading_client_connect'].includes(toState)) {
          setMetrics(prev => ({ ...prev, api_connected: true }));
        } else if (['error', 'idle'].includes(toState)) {
          setMetrics(prev => ({ ...prev, api_connected: false }));
        }

        if (!data.data.is_current || data.data.message !== `Current state: ${toState}`) {
          onMessage?.('state', data.data.message, {
            state: toState,
            from_state: fromState,
            to_state: toState,
            context: data.data.context,
            timestamp: data.data.timestamp,
            metadata: data.data.metadata
          });
        }
        break;
      }

      case 'trader_status':
        if (data.data.metrics) {
          setMetrics(prev => ({
            markets_connected: data.data.metrics.markets_connected || 0,
            tracked_markets: data.data.metrics.tracked_markets || 0,
            subscribed_markets: data.data.metrics.subscribed_markets || 0,
            snapshots_received: data.data.metrics.snapshots_received || 0,
            deltas_received: data.data.metrics.deltas_received || 0,
            uptime: data.data.metrics.uptime || 0,
            health: data.data.metrics.health || 'unknown',
            ping_health: data.data.metrics.ping_health || 'unknown',
            last_ping_age: data.data.metrics.last_ping_age || null,
            api_connected: data.data.metrics.api_connected || prev.api_connected,
            api_url: data.data.metrics.api_url || prev.api_url,
            ws_url: data.data.metrics.ws_url || prev.ws_url,
            signal_aggregator: data.data.metrics.signal_aggregator || null
          }));
        }
        if (data.data.state) {
          setCurrentState(data.data.state);
        }
        break;

      case 'trade_processing':
        if (data.data) {
          setTradeProcessing({
            recent_trades: data.data.recent_trades || [],
            stats: data.data.stats || { trades_seen: 0, trades_filtered: 0, trades_tracked: 0, filter_rate_percent: 0 },
            decisions: data.data.decisions || { detected: 0, executed: 0, rate_limited: 0, skipped: 0, reentries: 0 },
            decision_history: data.data.decision_history || [],
            last_updated: data.data.last_updated || null,
            timestamp: data.data.timestamp || null
          });
        }
        break;

      case 'trading_strategies':
        if (data.data) {
          setStrategyStatus({
            coordinator: data.data.coordinator || INITIAL_STRATEGY_STATUS.coordinator,
            strategies: data.data.strategies || {},
            recent_decisions: data.data.recent_decisions || [],
            last_updated: data.data.last_updated || null,
            timestamp: data.data.timestamp || null
          });
        }
        break;

      case 'event_research':
        if (data.data) {
          const researchData = data.data;
          const eventTicker = researchData.event_ticker;

          // Store research results indexed by event_ticker
          // Also build a market-level index for quick lookup
          setEventResearch(prev => {
            const marketIndex = { ...(prev._marketIndex || {}) };

            // Index each market assessment by ticker
            (researchData.markets || []).forEach(market => {
              marketIndex[market.ticker] = {
                eventTicker,
                ...market,
                eventTitle: researchData.event_title,
                eventCategory: researchData.event_category,
                primaryDriver: researchData.primary_driver,
                evidenceSummary: researchData.evidence_summary,
                researchedAt: researchData.researched_at,
              };
            });

            return {
              ...prev,
              [eventTicker]: researchData,
              _marketIndex: marketIndex,
            };
          });

          // Check for high-confidence mispricing to show toast alert
          const highConfidenceMarkets = (researchData.markets || []).filter(
            m => m.confidence === 'high' && Math.abs(m.mispricing_magnitude) > 0.15
          );

          if (highConfidenceMarkets.length > 0) {
            const topMarket = highConfidenceMarkets[0];
            setNewResearchAlert({
              eventTicker,
              eventTitle: researchData.event_title,
              market: topMarket,
              marketsWithEdge: researchData.markets_with_edge,
            });

            // Auto-dismiss after 8 seconds
            if (researchAlertTimeoutRef.current) {
              clearTimeout(researchAlertTimeoutRef.current);
            }
            researchAlertTimeoutRef.current = setTimeout(() => setNewResearchAlert(null), 8000);
          }

          // Log for debugging
          console.log(
            `[useV3WebSocket] Event research received: ${eventTicker}`,
            `${researchData.markets_evaluated} markets, ${researchData.markets_with_edge} with edge`
          );
        }
        break;

      case 'ping':
        // Track last ping time for heartbeat monitoring
        lastPingRef.current = Date.now();
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: 'pong', timestamp: Date.now() }));
        }
        break;

      // Trade flow states for Market Signal Table
      case 'trade_flow_states_snapshot':
        // Initial snapshot of all trade flow market states
        if (data.data?.markets) {
          // Convert array to map by ticker
          const statesMap = {};
          data.data.markets.forEach(m => {
            statesMap[m.market_ticker || m.ticker] = m;
          });
          setTradeFlowStates(statesMap);
        }
        break;

      case 'trade_flow_market_state':
        // Real-time update for a single market's trade flow state
        // Backend sends: { ticker, market_ticker, yes_trades, no_trades, ... }
        if (data.data?.ticker) {
          const { ticker, market_ticker, timestamp, ...stateData } = data.data;
          setTradeFlowStates(prev => ({
            ...prev,
            [ticker]: stateData
          }));
        }
        break;

      case 'trade_flow_trade_arrived':
        // Trade pulse for animation + buffer for trade feed
        if (data.data) {
          const { ticker, side, event_ticker, count, yes_price, trade_id, timestamp } = data.data;

          // Set trade pulse for animation
          setTradePulses(prev => ({
            ...prev,
            [ticker]: { side, ts: Date.now() }
          }));

          // Buffer trade for EventTradeFeed
          if (event_ticker) {
            setEventTrades(prev => {
              const eventTradeList = prev[event_ticker] || [];
              const newTrade = { ticker, side, count, yes_price, trade_id, timestamp };
              return {
                ...prev,
                [event_ticker]: [newTrade, ...eventTradeList].slice(0, 50) // Keep max 50
              };
            });
          }
        }
        break;

      case 'activity_feed_history':
        // Replay historical activity events for cross-tab persistence
        // This allows activity to persist when switching between Trader and Discovery views
        if (data.data?.events) {
          data.data.events.forEach(item => {
            if (item.type === 'system_activity') {
              const activityData = item.data;
              onMessage?.('activity', activityData.message, {
                activity_type: activityData.activity_type,
                timestamp: activityData.timestamp,
                metadata: activityData.metadata,
                is_history: true
              });
            }
          });
        }
        break;

      // === EventResearchAgent Messages (Visible Research Feed) ===

      case 'research_strategy':
        // Calibration output - what we're looking for (once per event init)
        if (data.data?.event_ticker) {
          const eventTicker = data.data.event_ticker;
          setResearchStrategies(prev => ({
            ...prev,
            [eventTicker]: data.data,
          }));
          console.log(
            `[useV3WebSocket] Research strategy for ${eventTicker}:`,
            `${data.data.search_queries?.length || 0} queries`
          );
        }
        break;

      case 'research_item':
        // Every news item found (visible feed)
        if (data.data?.event_ticker) {
          const eventTicker = data.data.event_ticker;
          setResearchFeeds(prev => {
            const existingFeed = prev[eventTicker] || [];
            // Check for duplicates
            const isDuplicate = existingFeed.some(
              item => item.id === data.data.id || item.url === data.data.url
            );
            if (isDuplicate) return prev;

            // Add new item at the beginning, limit to 100 items
            const newFeed = [data.data, ...existingFeed].slice(0, 100);
            return {
              ...prev,
              [eventTicker]: newFeed,
            };
          });
        }
        break;

      case 'research_signal':
        // Actionable signal extracted from research
        if (data.data?.event_ticker) {
          const eventTicker = data.data.event_ticker;
          setResearchSignals(prev => {
            const existingSignals = prev[eventTicker] || [];
            // Check for duplicates
            const isDuplicate = existingSignals.some(sig => sig.id === data.data.id);
            if (isDuplicate) return prev;

            // Add new signal at the beginning, limit to 50 signals
            const newSignals = [data.data, ...existingSignals].slice(0, 50);
            return {
              ...prev,
              [eventTicker]: newSignals,
            };
          });
          console.log(
            `[useV3WebSocket] Research signal for ${eventTicker}:`,
            `${data.data.direction} on ${data.data.market_ticker}`
          );
        }
        break;

      case 'research_batch_complete':
        // Batch summary after each loop
        if (data.data?.event_ticker) {
          const eventTicker = data.data.event_ticker;
          setResearchBatches(prev => ({
            ...prev,
            [eventTicker]: data.data,
          }));
        }
        break;

      // === Entity Trading Messages (Reddit Entity Pipeline) ===

      case 'reddit_post':
        // New Reddit post from PRAW stream
        if (data.data?.post_id) {
          setEntityRedditPosts(prev => {
            // Check for duplicates
            if (prev.some(p => p.post_id === data.data.post_id)) return prev;
            return [data.data, ...prev].slice(0, 50); // Keep max 50
          });
          setEntityStats(prev => ({
            ...prev,
            postsProcessed: prev.postsProcessed + 1,
          }));
        }
        break;

      case 'entity_extracted':
        // Entity extracted from a post with sentiment
        if (data.data?.entity_id) {
          setEntityExtractions(prev => {
            // Check for duplicates
            const key = `${data.data.post_id}_${data.data.entity_id}`;
            if (prev.some(e => `${e.post_id}_${e.entity_id}` === key)) return prev;
            return [data.data, ...prev].slice(0, 100); // Keep max 100
          });
          setEntityStats(prev => ({
            ...prev,
            entitiesExtracted: prev.entitiesExtracted + 1,
          }));
        }
        break;

      case 'price_impact':
        // Transformed price impact signal for trading
        if (data.data?.signal_id) {
          setEntityPriceImpacts(prev => {
            // Check for duplicates
            if (prev.some(i => i.signal_id === data.data.signal_id)) return prev;
            return [data.data, ...prev].slice(0, 50); // Keep max 50
          });
          setEntityStats(prev => ({
            ...prev,
            signalsGenerated: prev.signalsGenerated + 1,
          }));
          console.log(
            `[useV3WebSocket] Price impact: ${data.data.entity_name} → ${data.data.market_ticker}`,
            `sentiment: ${data.data.sentiment_score} → impact: ${data.data.price_impact_score}`
          );
        }
        break;

      case 'entity_index_update':
        // Entity-market index stats update
        if (data.data?.entity_count !== undefined) {
          setEntityStats(prev => ({
            ...prev,
            indexSize: data.data.entity_count,
          }));
        }
        break;

      case 'entity_system_status':
        // Entity system status update
        setEntitySystemActive(data.data?.is_active || false);
        if (data.data?.stats) {
          setEntityStats(prev => ({
            ...prev,
            ...data.data.stats,
          }));
        }
        break;

      case 'entity_snapshot':
        // Initial snapshot with recent entity data
        if (data.data) {
          if (data.data.reddit_posts) {
            setEntityRedditPosts(data.data.reddit_posts);
          }
          if (data.data.entities) {
            setEntityExtractions(data.data.entities);
          }
          if (data.data.price_impacts) {
            setEntityPriceImpacts(data.data.price_impacts);
          }
          if (data.data.stats) {
            setEntityStats(data.data.stats);
          }
          setEntitySystemActive(data.data.is_active || false);
          console.log(
            `[useV3WebSocket] Entity snapshot loaded:`,
            `${data.data.reddit_posts?.length || 0} posts,`,
            `${data.data.entities?.length || 0} entities,`,
            `${data.data.price_impacts?.length || 0} impacts`
          );
        }
        break;

      // === Entity Index Messages (Canonical entities with aliases) ===

      case 'entity_index_snapshot':
        // Full entity index with canonical entities, aliases, and market mappings
        if (data.data?.entities) {
          const lookup = {};
          data.data.entities.forEach(entity => {
            lookup[entity.entity_id] = entity;
          });
          setEntityIndex({
            totalEntities: data.data.total_entities || data.data.entities.length,
            entities: data.data.entities,
            entityLookup: lookup,
            timestamp: data.data.timestamp,
          });
          console.log(
            `[useV3WebSocket] Entity index snapshot: ${data.data.entities.length} canonical entities`
          );
        }
        break;

      case 'entity_signal_update':
        // Reddit signal update for a specific entity
        if (data.data?.entity_id) {
          setEntityIndex(prev => {
            const entityId = data.data.entity_id;
            const existingEntity = prev.entityLookup[entityId];
            if (!existingEntity) return prev;

            // Update the entity's reddit signals
            const updatedEntity = {
              ...existingEntity,
              reddit_signals: data.data.reddit_signals,
            };

            const updatedLookup = {
              ...prev.entityLookup,
              [entityId]: updatedEntity,
            };

            // Update entities array
            const updatedEntities = prev.entities.map(e =>
              e.entity_id === entityId ? updatedEntity : e
            );

            return {
              ...prev,
              entities: updatedEntities,
              entityLookup: updatedLookup,
            };
          });
        }
        break;

      case 'entity_linked':
        // New entity linked from market discovery
        if (data.data?.entity_id) {
          setEntityIndex(prev => {
            // Check if entity already exists
            if (prev.entityLookup[data.data.entity_id]) return prev;

            const newEntity = data.data;
            return {
              ...prev,
              totalEntities: prev.totalEntities + 1,
              entities: [...prev.entities, newEntity],
              entityLookup: {
                ...prev.entityLookup,
                [newEntity.entity_id]: newEntity,
              },
            };
          });
          console.log(
            `[useV3WebSocket] New entity linked: ${data.data.canonical_name}`
          );
        }
        break;

      // === Deep Agent Messages (Self-Improving Agent) ===
      // These are passed through to onMessage for useDeepAgent to handle
      case 'deep_agent_status':
      case 'deep_agent_cycle':
      case 'deep_agent_thinking':
      case 'deep_agent_tool_call':
      case 'deep_agent_trade':
      case 'deep_agent_settlement':
      case 'deep_agent_memory_update':
      case 'deep_agent_error':
      case 'deep_agent_snapshot':
        // Pass deep agent messages to the onMessage callback
        // useDeepAgent.processMessage() handles these
        onMessage?.(data.type, data.data, { timestamp: data.data?.timestamp });
        break;

      default:
        // Log unhandled message types for debugging
        if (data.type && !['pong'].includes(data.type)) {
          console.debug(`[useV3WebSocket] Unhandled message type: ${data.type}`);
        }
        break;
    }
  }, [onMessage]);

  // Connect on mount
  useEffect(() => {
    connectWebSocket();

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      // Clean up all toast timeouts on unmount
      if (orderFillTimeoutRef.current) {
        clearTimeout(orderFillTimeoutRef.current);
      }
      if (ttlCancellationTimeoutRef.current) {
        clearTimeout(ttlCancellationTimeoutRef.current);
      }
      if (settlementTimeoutRef.current) {
        clearTimeout(settlementTimeoutRef.current);
      }
      if (researchAlertTimeoutRef.current) {
        clearTimeout(researchAlertTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [connectWebSocket]);

  // Heartbeat monitoring - detect zombie connections
  // If we haven't received a ping from server in 60s, force reconnect
  useEffect(() => {
    const checkHeartbeat = () => {
      const timeSinceLastPing = Date.now() - lastPingRef.current;
      // Server sends pings every 30s, so 60s timeout gives buffer
      if (timeSinceLastPing > 60000 && wsStatus === 'connected') {
        console.warn('[useV3WebSocket] No ping in 60s, forcing reconnect');
        wsRef.current?.close();
      }
    };

    heartbeatIntervalRef.current = setInterval(checkHeartbeat, 15000);

    return () => {
      if (heartbeatIntervalRef.current) {
        clearInterval(heartbeatIntervalRef.current);
      }
    };
  }, [wsStatus]);

  // Clear new settlement after timeout
  const dismissSettlement = useCallback(() => {
    setNewSettlement(null);
  }, []);

  // Clear new order fill notification
  const dismissOrderFill = useCallback(() => {
    setNewOrderFill(null);
  }, []);

  // Clear new TTL cancellation notification
  const dismissTtlCancellation = useCallback(() => {
    setNewTtlCancellation(null);
  }, []);

  // Clear new research alert notification
  const dismissResearchAlert = useCallback(() => {
    setNewResearchAlert(null);
  }, []);

  // Helper to get research assessment for a specific market
  const getMarketResearch = useCallback((marketTicker) => {
    return eventResearch._marketIndex?.[marketTicker] || null;
  }, [eventResearch]);

  return {
    wsStatus,
    currentState,
    tradingState,
    lastUpdateTime,
    tradeProcessing,
    strategyStatus,
    eventResearch,
    newResearchAlert,
    dismissResearchAlert,
    getMarketResearch,
    settlements,
    newSettlement,
    dismissSettlement,
    newOrderFill,
    dismissOrderFill,
    newTtlCancellation,
    dismissTtlCancellation,
    metrics,
    // Trade flow states for Market Signal Table
    tradeFlowStates,
    tradePulses,
    eventTrades,
    // EventResearchAgent visible feed state
    researchStrategies,
    researchFeeds,
    researchSignals,
    researchBatches,
    // Entity Trading state (Reddit Entity Pipeline)
    entityRedditPosts,
    entityExtractions,
    entityPriceImpacts,
    entityStats,
    entitySystemActive,
    // Entity Index (Canonical entities with aliases)
    entityIndex,
  };
};

export default useV3WebSocket;
