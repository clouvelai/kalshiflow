import React, { useState, useMemo, memo } from 'react';
import {
  Users,
  Tag,
  TrendingUp,
  TrendingDown,
  Search,
  Filter,
  ChevronRight,
  ChevronDown,
  Building,
  User,
  Award,
  MessageCircle
} from 'lucide-react';

/**
 * Entity type icon mapping
 */
const EntityTypeIcon = memo(({ type }) => {
  switch (type) {
    case 'person':
      return <User className="w-3.5 h-3.5 text-cyan-400" />;
    case 'organization':
      return <Building className="w-3.5 h-3.5 text-purple-400" />;
    case 'position':
      return <Award className="w-3.5 h-3.5 text-amber-400" />;
    default:
      return <User className="w-3.5 h-3.5 text-gray-400" />;
  }
});

EntityTypeIcon.displayName = 'EntityTypeIcon';

/**
 * Sentiment badge with color coding
 */
const SentimentBadge = memo(({ value }) => {
  if (value === null || value === undefined || value === 0) {
    return <span className="text-gray-500 text-xs">--</span>;
  }

  const isPositive = value > 0;
  const isNegative = value < 0;

  return (
    <span className={`
      inline-flex items-center gap-0.5 text-xs font-mono
      ${isPositive ? 'text-green-400' : isNegative ? 'text-red-400' : 'text-gray-400'}
    `}>
      {isPositive ? <TrendingUp className="w-3 h-3" /> : isNegative ? <TrendingDown className="w-3 h-3" /> : null}
      {value > 0 ? '+' : ''}{value.toFixed(1)}
    </span>
  );
});

SentimentBadge.displayName = 'SentimentBadge';

/**
 * Single entity row component
 */
const EntityRow = memo(({ entity, isExpanded, onToggle }) => {
  const aliasCount = entity.aliases?.length || 0;
  const marketCount = entity.markets?.length || 0;
  const mentions = entity.reddit_signals?.total_mentions || 0;
  const sentiment = entity.reddit_signals?.aggregate_sentiment || 0;
  const lastSignal = entity.reddit_signals?.last_signal_at;

  // Format relative time for last signal
  const formatLastSignal = (ts) => {
    if (!ts) return null;
    const now = Date.now() / 1000;
    const diff = now - ts;
    if (diff < 60) return 'now';
    if (diff < 3600) return `${Math.floor(diff / 60)}m`;
    if (diff < 86400) return `${Math.floor(diff / 3600)}h`;
    return `${Math.floor(diff / 86400)}d`;
  };
  const lastSignalDisplay = formatLastSignal(lastSignal);

  return (
    <div className="border-b border-gray-800/50 last:border-b-0">
      {/* Main row */}
      <div
        className="flex items-center gap-3 px-3 py-2 hover:bg-gray-800/30 cursor-pointer transition-colors"
        onClick={() => onToggle(entity.entity_id)}
      >
        {/* Expand indicator */}
        <button className="text-gray-500 hover:text-gray-300">
          {isExpanded ? (
            <ChevronDown className="w-4 h-4" />
          ) : (
            <ChevronRight className="w-4 h-4" />
          )}
        </button>

        {/* Entity type icon */}
        <EntityTypeIcon type={entity.entity_type} />

        {/* Name and type */}
        <div className="flex-1 min-w-0">
          <div className="text-sm font-medium text-white truncate">
            {entity.canonical_name}
          </div>
          <div className="text-xs text-gray-500 flex items-center gap-2">
            <span className="capitalize">{entity.entity_type}</span>
            <span className="text-gray-600">|</span>
            <span className="text-cyan-500/70">{aliasCount} aliases</span>
            <span className="text-gray-600">|</span>
            <span className="text-purple-500/70">{marketCount} markets</span>
          </div>
        </div>

        {/* Reddit signals */}
        <div className="flex items-center gap-4 text-right">
          {/* Mentions */}
          <div className="flex flex-col items-end">
            <span className="text-sm font-mono text-white">{mentions}</span>
            <span className="text-[10px] text-gray-500 uppercase">mentions</span>
          </div>

          {/* Sentiment */}
          <div className="flex flex-col items-end min-w-[60px]">
            <SentimentBadge value={sentiment} />
            <span className="text-[10px] text-gray-500 uppercase">sentiment</span>
          </div>

          {/* Last signal */}
          <div className="flex flex-col items-end min-w-[40px]">
            <span className="text-xs font-mono text-gray-400">
              {lastSignalDisplay || '--'}
            </span>
            <span className="text-[10px] text-gray-500 uppercase">last</span>
          </div>
        </div>
      </div>

      {/* Expanded content */}
      {isExpanded && (
        <div className="px-4 py-3 bg-gray-900/50 border-t border-gray-800/30">
          {/* Aliases */}
          <div className="mb-3">
            <div className="text-xs text-gray-500 uppercase mb-1.5 flex items-center gap-1.5">
              <Tag className="w-3 h-3" />
              Aliases ({aliasCount})
            </div>
            <div className="flex flex-wrap gap-1.5">
              {entity.aliases?.slice(0, 15).map((alias, i) => (
                <span
                  key={i}
                  className="px-2 py-0.5 text-xs bg-gray-800 text-gray-300 rounded border border-gray-700/50"
                >
                  {alias}
                </span>
              ))}
              {aliasCount > 15 && (
                <span className="px-2 py-0.5 text-xs text-gray-500">
                  +{aliasCount - 15} more
                </span>
              )}
            </div>
          </div>

          {/* Markets */}
          <div>
            <div className="text-xs text-gray-500 uppercase mb-1.5 flex items-center gap-1.5">
              <TrendingUp className="w-3 h-3" />
              Linked Markets ({marketCount})
            </div>
            <div className="grid grid-cols-2 gap-2">
              {entity.markets?.map((market, i) => (
                <div
                  key={i}
                  className="flex items-center justify-between px-2 py-1.5 bg-gray-800/50 rounded text-xs border border-gray-700/30"
                >
                  <span className="font-mono text-cyan-400 truncate max-w-[180px]">
                    {market.market_ticker}
                  </span>
                  <span className={`
                    px-1.5 py-0.5 rounded text-[10px] font-medium
                    ${market.market_type === 'OUT' ? 'bg-red-900/30 text-red-400' :
                      market.market_type === 'WIN' ? 'bg-green-900/30 text-green-400' :
                      market.market_type === 'CONFIRM' ? 'bg-blue-900/30 text-blue-400' :
                      'bg-gray-800 text-gray-400'}
                  `}>
                    {market.market_type}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
});

EntityRow.displayName = 'EntityRow';

/**
 * Stats summary component
 */
const StatsSummary = memo(({ entities }) => {
  const stats = useMemo(() => {
    let totalMentions = 0;
    let totalMarkets = 0;
    let totalAliases = 0;
    let withSignals = 0;

    entities.forEach(e => {
      totalMentions += e.reddit_signals?.total_mentions || 0;
      totalMarkets += e.markets?.length || 0;
      totalAliases += e.aliases?.length || 0;
      if (e.reddit_signals?.total_mentions > 0) withSignals++;
    });

    return {
      totalEntities: entities.length,
      totalMarkets,
      totalAliases,
      totalMentions,
      withSignals,
    };
  }, [entities]);

  return (
    <div className="grid grid-cols-5 gap-2 mb-4">
      <div className="bg-gradient-to-br from-cyan-950/30 to-gray-900/50 rounded-lg p-2 border border-cyan-500/20">
        <div className="text-lg font-mono font-bold text-cyan-400">{stats.totalEntities}</div>
        <div className="text-[10px] text-gray-500 uppercase">Entities</div>
      </div>
      <div className="bg-gradient-to-br from-purple-950/30 to-gray-900/50 rounded-lg p-2 border border-purple-500/20">
        <div className="text-lg font-mono font-bold text-purple-400">{stats.totalMarkets}</div>
        <div className="text-[10px] text-gray-500 uppercase">Markets</div>
      </div>
      <div className="bg-gradient-to-br from-amber-950/30 to-gray-900/50 rounded-lg p-2 border border-amber-500/20">
        <div className="text-lg font-mono font-bold text-amber-400">{stats.totalAliases}</div>
        <div className="text-[10px] text-gray-500 uppercase">Aliases</div>
      </div>
      <div className="bg-gradient-to-br from-green-950/30 to-gray-900/50 rounded-lg p-2 border border-green-500/20">
        <div className="text-lg font-mono font-bold text-green-400">{stats.totalMentions}</div>
        <div className="text-[10px] text-gray-500 uppercase">Mentions</div>
      </div>
      <div className="bg-gradient-to-br from-blue-950/30 to-gray-900/50 rounded-lg p-2 border border-blue-500/20">
        <div className="text-lg font-mono font-bold text-blue-400">{stats.withSignals}</div>
        <div className="text-[10px] text-gray-500 uppercase">Active</div>
      </div>
    </div>
  );
});

StatsSummary.displayName = 'StatsSummary';

/**
 * EntityIndexPanel - Display canonical entities with aliases and reddit signals
 */
const EntityIndexPanel = ({ entityIndex }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [typeFilter, setTypeFilter] = useState('all');
  const [sortBy, setSortBy] = useState('mentions');
  const [expandedEntities, setExpandedEntities] = useState(new Set());

  const entities = useMemo(() => entityIndex?.entities || [], [entityIndex?.entities]);

  // Filter and sort entities
  const filteredEntities = useMemo(() => {
    let filtered = entities;

    // Search filter
    if (searchTerm) {
      const term = searchTerm.toLowerCase();
      filtered = filtered.filter(e =>
        e.canonical_name?.toLowerCase().includes(term) ||
        e.entity_id?.toLowerCase().includes(term) ||
        e.aliases?.some(a => a.toLowerCase().includes(term))
      );
    }

    // Type filter
    if (typeFilter !== 'all') {
      filtered = filtered.filter(e => e.entity_type === typeFilter);
    }

    // Sort
    filtered = [...filtered].sort((a, b) => {
      switch (sortBy) {
        case 'mentions':
          return (b.reddit_signals?.total_mentions || 0) - (a.reddit_signals?.total_mentions || 0);
        case 'sentiment':
          return Math.abs(b.reddit_signals?.aggregate_sentiment || 0) - Math.abs(a.reddit_signals?.aggregate_sentiment || 0);
        case 'markets':
          return (b.markets?.length || 0) - (a.markets?.length || 0);
        case 'name':
        default:
          return (a.canonical_name || '').localeCompare(b.canonical_name || '');
      }
    });

    return filtered;
  }, [entities, searchTerm, typeFilter, sortBy]);

  const toggleExpanded = (entityId) => {
    setExpandedEntities(prev => {
      const next = new Set(prev);
      if (next.has(entityId)) {
        next.delete(entityId);
      } else {
        next.add(entityId);
      }
      return next;
    });
  };

  return (
    <div className="h-full flex flex-col bg-gray-950">
      {/* Header */}
      <div className="p-4 border-b border-gray-800">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-bold text-white flex items-center gap-2">
            <Users className="w-5 h-5 text-cyan-400" />
            Entity Index
          </h2>
          <span className="text-xs text-gray-500">
            {filteredEntities.length} of {entities.length} entities
          </span>
        </div>

        {/* Stats summary */}
        <StatsSummary entities={entities} />

        {/* Search and filters */}
        <div className="flex gap-2">
          {/* Search */}
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500" />
            <input
              type="text"
              placeholder="Search by name or alias..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-10 pr-3 py-2 bg-gray-900 border border-gray-700 rounded-lg text-sm text-white placeholder-gray-500 focus:border-cyan-500 focus:outline-none"
            />
          </div>

          {/* Type filter */}
          <select
            value={typeFilter}
            onChange={(e) => setTypeFilter(e.target.value)}
            className="px-3 py-2 bg-gray-900 border border-gray-700 rounded-lg text-sm text-white focus:border-cyan-500 focus:outline-none"
          >
            <option value="all">All Types</option>
            <option value="person">Person</option>
            <option value="organization">Organization</option>
            <option value="position">Position</option>
          </select>

          {/* Sort */}
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value)}
            className="px-3 py-2 bg-gray-900 border border-gray-700 rounded-lg text-sm text-white focus:border-cyan-500 focus:outline-none"
          >
            <option value="mentions">Most Mentions</option>
            <option value="sentiment">Strongest Sentiment</option>
            <option value="markets">Most Markets</option>
            <option value="name">Name A-Z</option>
          </select>
        </div>
      </div>

      {/* Entity list */}
      <div className="flex-1 overflow-y-auto">
        {filteredEntities.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-gray-500">
            <Users className="w-12 h-12 mb-3 opacity-30" />
            <p className="text-sm">
              {entities.length === 0
                ? 'No entities indexed yet'
                : 'No entities match your filters'}
            </p>
          </div>
        ) : (
          filteredEntities.map(entity => (
            <EntityRow
              key={entity.entity_id}
              entity={entity}
              isExpanded={expandedEntities.has(entity.entity_id)}
              onToggle={toggleExpanded}
            />
          ))
        )}
      </div>
    </div>
  );
};

export default memo(EntityIndexPanel);
