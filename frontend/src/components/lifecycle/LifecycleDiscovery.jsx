import React, { useState, useMemo } from 'react';
import { useLifecycleWebSocket } from '../../hooks/useLifecycleWebSocket';

// Lifecycle components
import LifecycleHeader from './LifecycleHeader';
import CategoryHealthBar from './CategoryHealthBar';
import FilterControls from './FilterControls';
import MarketGrid from './MarketGrid';
import ActivityFeed from './ActivityFeed';

/**
 * LifecycleDiscovery - Main page for lifecycle market discovery
 *
 * Displays markets discovered via Kalshi's market_lifecycle_v2 channel.
 * Emphasizes TIME and CHANGE since tracking began - that's the trader's edge.
 */
const LifecycleDiscovery = () => {
  // WebSocket connection and state
  const {
    wsStatus,
    markets,
    stats,
    recentEvents,
    isConnected,
    isAtCapacity,
    clearEvents,
    // RLM (Reverse Line Movement) state
    rlmStates,
    tradePulses,
    // Upcoming markets (opening within 4 hours)
    upcomingMarkets,
    // Trading state (balance for low cash indicator)
    tradingState,
    // Event exposure data (correlated positions across related markets)
    eventExposure,
    // Event research (AI-generated assessments)
    getMarketResearch
  } = useLifecycleWebSocket();

  // Filter state
  const [filters, setFilters] = useState({
    search: '',
    category: null, // null = all
    status: 'all', // all, active, determined
    spread: 'any', // any, lt2, lt5, lt10
    sort: 'newest', // newest, volume_delta, spread, price_move
    groupBy: 'category' // 'category' or 'event'
  });

  // Dormant market toggle (markets with 0 volume_24h)
  const [showDormant, setShowDormant] = useState(false);

  // Calculate dormant count (before other filters)
  const dormantCount = useMemo(
    () => markets.filter(m => (m.volume_24h || 0) === 0).length,
    [markets]
  );

  // Pre-filter dormant markets
  const activeMarkets = useMemo(
    () => showDormant ? markets : markets.filter(m => (m.volume_24h || 0) > 0),
    [markets, showDormant]
  );

  // Filter and sort markets (using activeMarkets which excludes dormant if toggle is off)
  const filteredMarkets = useMemo(() => {
    let result = [...activeMarkets];

    // Search filter
    if (filters.search) {
      const searchLower = filters.search.toLowerCase();
      result = result.filter(m =>
        m.ticker?.toLowerCase().includes(searchLower) ||
        m.title?.toLowerCase().includes(searchLower) ||
        m.event_title?.toLowerCase().includes(searchLower)
      );
    }

    // Category filter
    if (filters.category) {
      result = result.filter(m => m.category === filters.category);
    }

    // Status filter
    if (filters.status !== 'all') {
      if (filters.status === 'has_position') {
        // Include filled positions OR resting orders
        result = result.filter(m =>
          (m.trading?.position?.count > 0) ||
          (m.trading?.orders?.length > 0)
        );
      } else if (filters.status === 'signal_ready') {
        // Currently meeting threshold OR has triggered before
        result = result.filter(m => {
          const rlmState = rlmStates[m.ticker];
          return rlmState?.signalReady || (rlmState?.signal_trigger_count > 0);
        });
      } else {
        // Standard status filter (active, determined)
        result = result.filter(m => m.status === filters.status);
      }
    }

    // Spread filter
    if (filters.spread !== 'any') {
      result = result.filter(m => {
        const spread = (m.yes_ask || 0) - (m.yes_bid || 0);
        if (filters.spread === 'lt2') return spread <= 2;
        if (filters.spread === 'lt5') return spread <= 5;
        if (filters.spread === 'lt10') return spread <= 10;
        return true;
      });
    }

    // Sort
    switch (filters.sort) {
      case 'newest':
        result.sort((a, b) => (b.tracked_at || 0) - (a.tracked_at || 0));
        break;
      case 'volume_delta':
        result.sort((a, b) => Math.abs(b.volume_delta || 0) - Math.abs(a.volume_delta || 0));
        break;
      case 'spread':
        result.sort((a, b) => {
          const spreadA = (a.yes_ask || 0) - (a.yes_bid || 0);
          const spreadB = (b.yes_ask || 0) - (b.yes_bid || 0);
          return spreadA - spreadB;
        });
        break;
      case 'price_move':
        result.sort((a, b) => Math.abs(b.price_delta || 0) - Math.abs(a.price_delta || 0));
        break;
      default:
        break;
    }

    return result;
  }, [activeMarkets, filters, rlmStates]);

  // Group markets by category OR event for display
  const groupedMarkets = useMemo(() => {
    const groups = {};

    if (filters.groupBy === 'event') {
      // Group by event_ticker
      filteredMarkets.forEach(market => {
        const eventKey = market.event_ticker || 'no-event';
        if (!groups[eventKey]) {
          groups[eventKey] = [];
        }
        groups[eventKey].push(market);
      });
    } else {
      // Group by category (default)
      filteredMarkets.forEach(market => {
        const category = market.category || 'other';
        if (!groups[category]) {
          groups[category] = [];
        }
        groups[category].push(market);
      });
    }

    return groups;
  }, [filteredMarkets, filters.groupBy]);

  // Handle category click from CategoryHealthBar
  const handleCategoryClick = (category) => {
    setFilters(prev => ({
      ...prev,
      category: prev.category === category ? null : category
    }));
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-950 via-gray-900 to-gray-950">
      {/* Header with capacity and connection status */}
      <LifecycleHeader
        wsStatus={wsStatus}
        stats={stats}
        isAtCapacity={isAtCapacity}
        balance={tradingState?.balance || 0}
        minTraderCash={tradingState?.min_trader_cash || 0}
        showDormant={showDormant}
        onToggleDormant={() => setShowDormant(prev => !prev)}
        dormantCount={dormantCount}
      />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 py-4">
        {/* Category Health Bar */}
        <CategoryHealthBar
          stats={stats}
          activeCategory={filters.category}
          onCategoryClick={handleCategoryClick}
        />

        {/* Filter Controls */}
        <FilterControls
          filters={filters}
          onFilterChange={setFilters}
          marketCount={filteredMarkets.length}
          totalCount={activeMarkets.length}
        />

        {/* Main Content: 70/30 split */}
        <div className="flex gap-6 mt-4">
          {/* Market Grid - 70% */}
          <div className="flex-1 min-w-0">
            <MarketGrid
              marketsByCategory={groupedMarkets}
              showCategoryHeaders={!filters.category || filters.groupBy === 'event'}
              groupBy={filters.groupBy}
              rlmStates={rlmStates}
              tradePulses={tradePulses}
              rlmConfig={tradingState?.rlm_config}
              eventExposure={eventExposure}
              getMarketResearch={getMarketResearch}
            />
          </div>

          {/* Activity Feed - 30% */}
          <div className="w-80 flex-shrink-0">
            <ActivityFeed
              events={recentEvents}
              onClear={clearEvents}
              upcomingMarkets={upcomingMarkets}
              rlmStates={rlmStates}
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default LifecycleDiscovery;
