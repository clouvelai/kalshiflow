import React, { memo, useState, useMemo } from 'react';
import { Search, Brain, Clock, TrendingUp, Filter, SortDesc, RefreshCw } from 'lucide-react';
import EventCard from './EventCard';

/**
 * Format timestamp as relative time (e.g., "2m ago", "1h ago")
 */
const formatTimeAgo = (timestamp) => {
  if (!timestamp) return 'Unknown';
  const seconds = Math.floor(Date.now() / 1000 - timestamp);
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
};

/**
 * EmptyState - Beautiful empty state when no events exist
 */
const EmptyState = memo(() => (
  <div className="flex flex-col items-center justify-center py-20">
    <div className="relative mb-6">
      <div className="w-24 h-24 rounded-full bg-gradient-to-br from-emerald-900/30 to-gray-900/50 border border-emerald-700/20 flex items-center justify-center">
        <Brain className="w-10 h-10 text-emerald-600/50" />
      </div>
      <div className="absolute -bottom-1 -right-1 w-8 h-8 rounded-full bg-gray-800 border border-gray-700 flex items-center justify-center">
        <Search className="w-4 h-4 text-gray-500" />
      </div>
    </div>
    <h3 className="text-lg font-semibold text-gray-300 mb-2">No Events Researched Yet</h3>
    <p className="text-gray-500 text-sm text-center max-w-md leading-relaxed">
      The AI research system will analyze prediction market events and display
      probability assessments, key drivers, and trading recommendations here.
    </p>
    <div className="mt-6 flex items-center space-x-2 text-xs text-gray-600">
      <RefreshCw className="w-3.5 h-3.5 animate-spin" />
      <span>Waiting for research results...</span>
    </div>
  </div>
));

EmptyState.displayName = 'EmptyState';

/**
 * StatsBar - Summary statistics for all researched events
 */
const StatsBar = memo(({ events }) => {
  const stats = useMemo(() => {
    let totalMarkets = 0;
    let marketsWithEdge = 0;
    let buyYesCount = 0;
    let buyNoCount = 0;

    events.forEach(event => {
      totalMarkets += event.markets_evaluated || 0;
      marketsWithEdge += event.markets_with_edge || 0;
      (event.markets || []).forEach(m => {
        if (m.recommendation === 'BUY_YES') buyYesCount++;
        if (m.recommendation === 'BUY_NO') buyNoCount++;
      });
    });

    return {
      eventCount: events.length,
      totalMarkets,
      marketsWithEdge,
      buyYesCount,
      buyNoCount,
    };
  }, [events]);

  return (
    <div className="flex items-center space-x-6 text-sm">
      <div className="flex items-center space-x-2">
        <Brain className="w-4 h-4 text-emerald-400" />
        <span className="text-gray-400">Events:</span>
        <span className="font-mono font-semibold text-white">{stats.eventCount}</span>
      </div>
      <div className="flex items-center space-x-2">
        <TrendingUp className="w-4 h-4 text-cyan-400" />
        <span className="text-gray-400">Markets:</span>
        <span className="font-mono font-semibold text-white">{stats.totalMarkets}</span>
      </div>
      <div className="flex items-center space-x-2">
        <span className="text-gray-400">With Edge:</span>
        <span className="font-mono font-semibold text-emerald-400">{stats.marketsWithEdge}</span>
      </div>
      <div className="h-4 w-px bg-gray-700" />
      <div className="flex items-center space-x-3">
        <span className="px-2 py-0.5 rounded text-xs font-mono bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
          {stats.buyYesCount} YES
        </span>
        <span className="px-2 py-0.5 rounded text-xs font-mono bg-red-500/10 text-red-400 border border-red-500/20">
          {stats.buyNoCount} NO
        </span>
      </div>
    </div>
  );
});

StatsBar.displayName = 'StatsBar';

/**
 * SortDropdown - Dropdown for sort options
 */
const SortDropdown = memo(({ value, onChange }) => {
  return (
    <div className="relative">
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="appearance-none bg-slate-800/50 border border-slate-700/50 rounded-lg px-3 py-2 pr-8 text-sm text-gray-300 focus:outline-none focus:border-emerald-500/50 cursor-pointer"
      >
        <option value="newest">Newest First</option>
        <option value="oldest">Oldest First</option>
        <option value="most_edge">Most Edge</option>
        <option value="most_markets">Most Markets</option>
      </select>
      <SortDesc className="absolute right-2.5 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500 pointer-events-none" />
    </div>
  );
});

SortDropdown.displayName = 'SortDropdown';

/**
 * FilterDropdown - Dropdown for category filtering
 */
const FilterDropdown = memo(({ categories, value, onChange }) => {
  return (
    <div className="relative">
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="appearance-none bg-slate-800/50 border border-slate-700/50 rounded-lg px-3 py-2 pr-8 text-sm text-gray-300 focus:outline-none focus:border-emerald-500/50 cursor-pointer"
      >
        <option value="all">All Categories</option>
        {categories.map(cat => (
          <option key={cat} value={cat}>{cat}</option>
        ))}
      </select>
      <Filter className="absolute right-2.5 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500 pointer-events-none" />
    </div>
  );
});

FilterDropdown.displayName = 'FilterDropdown';

/**
 * EventsPanel - Container for event research cards
 *
 * Features:
 * - Sorting (newest, oldest, most edge, most markets)
 * - Category filtering
 * - Search functionality
 * - Beautiful empty state
 * - Stats summary bar
 */
const EventsPanel = ({ events, newResearchAlert, onDismissAlert }) => {
  const [sortBy, setSortBy] = useState('newest');
  const [filterCategory, setFilterCategory] = useState('all');
  const [searchQuery, setSearchQuery] = useState('');

  // Extract unique categories
  const categories = useMemo(() => {
    const cats = new Set();
    events.forEach(e => {
      if (e.event_category) cats.add(e.event_category);
    });
    return Array.from(cats).sort();
  }, [events]);

  // Filter and sort events
  const filteredEvents = useMemo(() => {
    let result = [...events];

    // Apply category filter
    if (filterCategory !== 'all') {
      result = result.filter(e => e.event_category === filterCategory);
    }

    // Apply search filter
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      result = result.filter(e =>
        e.event_title?.toLowerCase().includes(query) ||
        e.event_ticker?.toLowerCase().includes(query) ||
        e.primary_driver?.toLowerCase().includes(query)
      );
    }

    // Apply sorting
    switch (sortBy) {
      case 'oldest':
        result.sort((a, b) => (a.researched_at || 0) - (b.researched_at || 0));
        break;
      case 'most_edge':
        result.sort((a, b) => (b.markets_with_edge || 0) - (a.markets_with_edge || 0));
        break;
      case 'most_markets':
        result.sort((a, b) => (b.markets_evaluated || 0) - (a.markets_evaluated || 0));
        break;
      case 'newest':
      default:
        result.sort((a, b) => (b.researched_at || 0) - (a.researched_at || 0));
        break;
    }

    return result;
  }, [events, sortBy, filterCategory, searchQuery]);

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center space-x-3">
            <div className="p-2 rounded-lg bg-emerald-500/10 border border-emerald-500/20">
              <Brain className="w-6 h-6 text-emerald-400" />
            </div>
            <span>Event Research</span>
          </h1>
          <p className="text-gray-500 text-sm mt-1">
            AI-powered analysis of prediction market events
          </p>
        </div>

        {events.length > 0 && (
          <StatsBar events={events} />
        )}
      </div>

      {/* Controls Bar */}
      {events.length > 0 && (
        <div className="flex items-center justify-between bg-slate-800/30 backdrop-blur-sm border border-slate-700/30 rounded-xl px-4 py-3">
          {/* Search */}
          <div className="relative flex-1 max-w-md">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search events..."
              className="w-full bg-slate-800/50 border border-slate-700/50 rounded-lg pl-10 pr-4 py-2 text-sm text-gray-300 placeholder-gray-500 focus:outline-none focus:border-emerald-500/50"
            />
          </div>

          {/* Filters */}
          <div className="flex items-center space-x-3">
            <FilterDropdown
              categories={categories}
              value={filterCategory}
              onChange={setFilterCategory}
            />
            <SortDropdown
              value={sortBy}
              onChange={setSortBy}
            />
          </div>
        </div>
      )}

      {/* Results count */}
      {events.length > 0 && (
        <div className="text-sm text-gray-500">
          Showing {filteredEvents.length} of {events.length} events
        </div>
      )}

      {/* Event Cards */}
      {filteredEvents.length > 0 ? (
        <div className="space-y-4">
          {filteredEvents.map((event) => (
            <EventCard
              key={event.eventTicker || event.event_ticker}
              event={event}
              isNew={newResearchAlert?.eventTicker === event.event_ticker}
            />
          ))}
        </div>
      ) : events.length > 0 ? (
        <div className="text-center py-12 text-gray-500">
          <Filter className="w-8 h-8 mx-auto mb-3 text-gray-600" />
          <p>No events match your filters</p>
          <button
            onClick={() => {
              setSearchQuery('');
              setFilterCategory('all');
            }}
            className="mt-2 text-emerald-400 hover:text-emerald-300 text-sm"
          >
            Clear filters
          </button>
        </div>
      ) : (
        <EmptyState />
      )}
    </div>
  );
};

export default memo(EventsPanel);
