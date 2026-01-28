import React, { useState, useMemo, useCallback, memo } from 'react';
import {
  Database,
  User,
  Building2,
  Briefcase,
  Search,
  ChevronDown,
  ChevronRight,
  BarChart3,
  Zap,
  Filter,
  ArrowUpDown,
  Activity,
  Hash,
  ExternalLink,
} from 'lucide-react';
import { useEntityAnimations } from '../../../hooks/v3-trader/useEntityAnimations';

/**
 * Entity type configuration - colors and icons
 */
const ENTITY_TYPE_CONFIG = {
  person: {
    icon: User,
    gradient: 'from-cyan-950/40 via-cyan-900/20 to-gray-900/30',
    border: 'border-cyan-700/30',
    badge: 'bg-cyan-900/40 text-cyan-300 border-cyan-600/40',
    glow: 'rgba(6, 182, 212, 0.6)',
  },
  organization: {
    icon: Building2,
    gradient: 'from-purple-950/40 via-purple-900/20 to-gray-900/30',
    border: 'border-purple-700/30',
    badge: 'bg-purple-900/40 text-purple-300 border-purple-600/40',
    glow: 'rgba(147, 51, 234, 0.6)',
  },
  position: {
    icon: Briefcase,
    gradient: 'from-amber-950/40 via-amber-900/20 to-gray-900/30',
    border: 'border-amber-700/30',
    badge: 'bg-amber-900/40 text-amber-300 border-amber-600/40',
    glow: 'rgba(245, 158, 11, 0.6)',
  },
  default: {
    icon: User,
    gradient: 'from-gray-800/40 via-gray-800/20 to-gray-900/30',
    border: 'border-gray-700/30',
    badge: 'bg-gray-800/40 text-gray-300 border-gray-600/40',
    glow: 'rgba(107, 114, 128, 0.6)',
  },
};

/**
 * Market type badge colors
 */
const MARKET_TYPE_BADGES = {
  OUT: 'bg-amber-900/40 text-amber-300 border-amber-600/40',
  WIN: 'bg-emerald-900/40 text-emerald-300 border-emerald-600/40',
  CONFIRM: 'bg-blue-900/40 text-blue-300 border-blue-600/40',
  NOMINEE: 'bg-violet-900/40 text-violet-300 border-violet-600/40',
  PRESIDENT: 'bg-cyan-900/40 text-cyan-300 border-cyan-600/40',
  DEFAULT: 'bg-gray-800/40 text-gray-400 border-gray-600/40',
};

/**
 * Extract market type from ticker
 */
const getMarketType = (ticker) => {
  if (!ticker) return 'DEFAULT';
  const upper = ticker.toUpperCase();
  if (upper.includes('OUT')) return 'OUT';
  if (upper.includes('WIN')) return 'WIN';
  if (upper.includes('CONFIRM')) return 'CONFIRM';
  if (upper.includes('NOMINEE')) return 'NOMINEE';
  if (upper.includes('PRESIDENT')) return 'PRESIDENT';
  return 'DEFAULT';
};

/**
 * Sentiment Bar - Animated horizontal bar showing sentiment
 */
const SentimentBar = memo(({ sentiment, maxSentiment = 100 }) => {
  const normalizedSentiment = Math.min(Math.abs(sentiment), maxSentiment);
  const widthPercent = (normalizedSentiment / maxSentiment) * 100;
  const isPositive = sentiment >= 0;

  return (
    <div className="relative h-1.5 w-full bg-gray-800/60 rounded-full overflow-hidden">
      <div
        className={`
          h-full rounded-full transition-all duration-500 ease-out
          ${isPositive
            ? 'bg-gradient-to-r from-emerald-600 to-emerald-400'
            : 'bg-gradient-to-r from-rose-600 to-rose-400'}
        `}
        style={{ width: `${Math.max(widthPercent, 5)}%` }}
      />
      {/* Glow effect */}
      <div
        className={`
          absolute inset-0 rounded-full blur-sm opacity-50
          ${isPositive ? 'bg-emerald-500/30' : 'bg-rose-500/30'}
        `}
        style={{ width: `${Math.max(widthPercent, 5)}%` }}
      />
    </div>
  );
});

SentimentBar.displayName = 'SentimentBar';

/**
 * Stat Pill - Small stat display in header
 */
const StatPill = memo(({ icon: Icon, label, value, color = 'gray' }) => {
  const colorClasses = {
    cyan: 'text-cyan-400 bg-cyan-900/30 border-cyan-700/30',
    purple: 'text-purple-400 bg-purple-900/30 border-purple-700/30',
    emerald: 'text-emerald-400 bg-emerald-900/30 border-emerald-700/30',
    amber: 'text-amber-400 bg-amber-900/30 border-amber-700/30',
    gray: 'text-gray-400 bg-gray-800/30 border-gray-700/30',
  };

  return (
    <div className={`
      flex items-center gap-1.5 px-2.5 py-1 rounded-lg border text-xs
      ${colorClasses[color]}
    `}>
      <Icon className="w-3 h-3" />
      <span className="text-gray-500">{label}</span>
      <span className="font-mono font-bold">{value}</span>
    </div>
  );
});

StatPill.displayName = 'StatPill';

/**
 * Alias Chip - Small chip showing an alias
 */
const AliasChip = memo(({ alias }) => (
  <span className="px-2 py-0.5 text-[10px] bg-gray-800/60 text-gray-400 rounded-md border border-gray-700/40 truncate max-w-[120px]">
    {alias}
  </span>
));

AliasChip.displayName = 'AliasChip';

/**
 * Market Connection - Shows a linked market ticker
 */
const MarketConnection = memo(({ ticker }) => {
  const marketType = getMarketType(ticker);
  const badgeClass = MARKET_TYPE_BADGES[marketType] || MARKET_TYPE_BADGES.DEFAULT;

  return (
    <div className="flex items-center justify-between gap-2 px-2.5 py-1.5 bg-gray-800/40 rounded-lg border border-gray-700/30 hover:border-gray-600/40 transition-colors group">
      <div className="flex items-center gap-2 min-w-0">
        <BarChart3 className="w-3 h-3 text-gray-500 flex-shrink-0" />
        <span className="text-xs font-mono text-gray-300 truncate">{ticker}</span>
      </div>
      <div className="flex items-center gap-1.5">
        <span className={`px-1.5 py-0.5 text-[9px] font-bold rounded border ${badgeClass}`}>
          {marketType}
        </span>
        <ExternalLink className="w-3 h-3 text-gray-600 group-hover:text-gray-400 transition-colors" />
      </div>
    </div>
  );
});

MarketConnection.displayName = 'MarketConnection';

/**
 * Entity Card - Main card component for displaying an entity
 */
const EntityCard = memo(({
  entity,
  animationClasses,
  getEntitySentiment,
  isExpanded,
  onToggleExpand,
}) => {
  const entityType = (entity.entity_type || 'default').toLowerCase();
  const config = ENTITY_TYPE_CONFIG[entityType] || ENTITY_TYPE_CONFIG.default;
  const Icon = config.icon;

  const sentiment = getEntitySentiment(entity);
  const mentionCount = entity.reddit_signals?.mention_count || 0;
  const aliases = entity.aliases || [];
  const marketTickers = entity.market_tickers || [];

  const visibleAliases = aliases.slice(0, 3);
  const remainingAliasCount = Math.max(0, aliases.length - 3);

  return (
    <div
      className={`
        relative overflow-hidden rounded-xl border
        bg-gradient-to-br ${config.gradient} ${config.border}
        backdrop-blur-sm p-4
        transition-all duration-300 hover:border-opacity-60
        ${animationClasses}
      `}
      style={{
        '--entity-glow': config.glow,
      }}
    >
      {/* Header Row */}
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-2.5">
          <div className={`p-2 rounded-lg bg-gray-800/50 border ${config.border}`}>
            <Icon className="w-4 h-4 text-gray-300" />
          </div>
          <div>
            <div className="text-sm font-semibold text-white leading-tight">
              {entity.canonical_name}
            </div>
            {entity.entity_id && (
              <div className="flex items-center gap-1 mt-0.5">
                <Hash className="w-2.5 h-2.5 text-gray-600" />
                <span className="text-[9px] text-gray-600 font-mono">{entity.entity_id}</span>
              </div>
            )}
          </div>
        </div>
        <span className={`px-2 py-0.5 text-[10px] font-bold rounded-md border ${config.badge} uppercase`}>
          {entityType}
        </span>
      </div>

      {/* Sentiment Bar */}
      <div className="mb-3">
        <SentimentBar sentiment={sentiment} />
      </div>

      {/* Metrics Row */}
      <div className="flex items-center gap-4 mb-3">
        <div className="flex items-center gap-1.5">
          <Activity className="w-3 h-3 text-gray-500" />
          <span className="text-xs text-gray-500">Mentions:</span>
          <span className="text-xs font-mono font-bold text-white">{mentionCount}</span>
        </div>
        <div className="flex items-center gap-1.5">
          <Zap className="w-3 h-3 text-gray-500" />
          <span className="text-xs text-gray-500">Sentiment:</span>
          <span className={`text-xs font-mono font-bold ${sentiment >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
            {sentiment > 0 ? '+' : ''}{sentiment.toFixed(1)}
          </span>
        </div>
      </div>

      {/* Aliases */}
      {aliases.length > 0 && (
        <div className="flex items-center flex-wrap gap-1.5 mb-3">
          {visibleAliases.map((alias, idx) => (
            <AliasChip key={idx} alias={alias} />
          ))}
          {remainingAliasCount > 0 && (
            <span className="text-[10px] text-gray-500">+{remainingAliasCount} more</span>
          )}
        </div>
      )}

      {/* Expand/Collapse Button */}
      {marketTickers.length > 0 && (
        <button
          onClick={onToggleExpand}
          className="w-full flex items-center justify-between p-2 rounded-lg bg-gray-800/30 border border-gray-700/30 hover:bg-gray-800/50 transition-colors"
        >
          <div className="flex items-center gap-2">
            <BarChart3 className="w-3.5 h-3.5 text-gray-500" />
            <span className="text-xs text-gray-400">
              {marketTickers.length} market{marketTickers.length !== 1 ? 's' : ''} linked
            </span>
          </div>
          {isExpanded
            ? <ChevronDown className="w-4 h-4 text-gray-500" />
            : <ChevronRight className="w-4 h-4 text-gray-500" />
          }
        </button>
      )}

      {/* Expanded Content - Market Connections */}
      {isExpanded && marketTickers.length > 0 && (
        <div className="mt-3 space-y-1.5 animate-fade-in">
          {marketTickers.slice(0, 10).map((ticker, idx) => (
            <MarketConnection key={idx} ticker={ticker} />
          ))}
          {marketTickers.length > 10 && (
            <div className="text-center text-[10px] text-gray-500 py-1">
              +{marketTickers.length - 10} more markets
            </div>
          )}
        </div>
      )}
    </div>
  );
}, (prevProps, nextProps) => {
  // Custom comparison for memo
  return (
    prevProps.entity.entity_id === nextProps.entity.entity_id &&
    prevProps.entity.reddit_signals === nextProps.entity.reddit_signals &&
    prevProps.animationClasses === nextProps.animationClasses &&
    prevProps.isExpanded === nextProps.isExpanded
  );
});

EntityCard.displayName = 'EntityCard';

/**
 * Empty State - Shown when no entities
 */
const EmptyState = memo(({ entitySystemActive }) => (
  <div className="flex flex-col items-center justify-center py-12 px-4">
    <div className="p-4 rounded-2xl bg-gray-800/30 border border-gray-700/30 mb-4">
      <Database className="w-10 h-10 text-gray-600" />
    </div>
    <div className="text-gray-400 text-sm font-medium mb-1">
      No entities discovered yet
    </div>
    <div className="text-gray-600 text-xs text-center max-w-[280px]">
      {entitySystemActive
        ? 'Entities will appear as they are discovered from Reddit posts and market data'
        : 'Entity system is inactive. Start the V3 trader with Reddit entity extraction enabled.'}
    </div>
  </div>
));

EmptyState.displayName = 'EmptyState';

/**
 * EntityKnowledgeBasePanel - Main panel component
 *
 * Displays entity-to-market mappings with:
 * - Premium visual design
 * - Event-driven animations
 * - Search and filtering
 * - Progressive disclosure
 */
const EntityKnowledgeBasePanel = ({ entityIndex, entitySystemActive }) => {
  // State
  const [searchQuery, setSearchQuery] = useState('');
  const [filterType, setFilterType] = useState('all');
  const [sortBy, setSortBy] = useState('activity'); // activity, sentiment, markets
  const [expandedEntityIds, setExpandedEntityIds] = useState(new Set());

  // Animation hook
  const {
    getAnimationClasses,
    getEntitySentiment,
    recentActivityIds,
  } = useEntityAnimations({ entityIndex, entitySystemActive });

  // Filter and sort entities
  const filteredEntities = useMemo(() => {
    if (!entityIndex?.entities?.length) return [];

    let filtered = [...entityIndex.entities];

    // Apply search filter
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(entity => {
        const nameMatch = entity.canonical_name?.toLowerCase().includes(query);
        const aliasMatch = entity.aliases?.some(a => a.toLowerCase().includes(query));
        const tickerMatch = entity.market_tickers?.some(t => t.toLowerCase().includes(query));
        return nameMatch || aliasMatch || tickerMatch;
      });
    }

    // Apply type filter
    if (filterType !== 'all') {
      filtered = filtered.filter(entity =>
        (entity.entity_type || 'default').toLowerCase() === filterType
      );
    }

    // Apply sorting
    filtered.sort((a, b) => {
      switch (sortBy) {
        case 'activity': {
          const aRecent = recentActivityIds.has(a.entity_id) ? 1 : 0;
          const bRecent = recentActivityIds.has(b.entity_id) ? 1 : 0;
          if (aRecent !== bRecent) return bRecent - aRecent;
          const aMentions = a.reddit_signals?.mention_count || 0;
          const bMentions = b.reddit_signals?.mention_count || 0;
          return bMentions - aMentions;
        }
        case 'sentiment': {
          const aSentiment = Math.abs(getEntitySentiment(a));
          const bSentiment = Math.abs(getEntitySentiment(b));
          return bSentiment - aSentiment;
        }
        case 'markets': {
          const aMarkets = a.market_tickers?.length || 0;
          const bMarkets = b.market_tickers?.length || 0;
          return bMarkets - aMarkets;
        }
        default:
          return 0;
      }
    });

    return filtered;
  }, [entityIndex?.entities, searchQuery, filterType, sortBy, recentActivityIds, getEntitySentiment]);

  // Stats
  const stats = useMemo(() => {
    const entities = entityIndex?.entities || [];
    const totalMarkets = new Set(entities.flatMap(e => e.market_tickers || [])).size;
    const activeCount = recentActivityIds.size;
    const avgSentiment = entities.length > 0
      ? entities.reduce((sum, e) => sum + getEntitySentiment(e), 0) / entities.length
      : 0;

    return {
      totalEntities: entities.length,
      totalMarkets,
      activeCount,
      avgSentiment,
    };
  }, [entityIndex?.entities, recentActivityIds, getEntitySentiment]);

  // Toggle entity expansion
  const toggleEntityExpand = useCallback((entityId) => {
    setExpandedEntityIds(prev => {
      const next = new Set(prev);
      if (next.has(entityId)) {
        next.delete(entityId);
      } else {
        next.add(entityId);
      }
      return next;
    });
  }, []);

  // Debounced search
  const handleSearchChange = useCallback((e) => {
    setSearchQuery(e.target.value);
  }, []);

  return (
    <div className="bg-gray-900/50 rounded-2xl border border-gray-800/50 overflow-hidden">
      {/* Header */}
      <div className="p-4 border-b border-gray-800/50">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-gradient-to-br from-cyan-900/40 to-cyan-950/30 border border-cyan-700/30">
              <Database className="w-5 h-5 text-cyan-400" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-lg font-semibold text-white">Entity Knowledge Base</h2>
                {entitySystemActive && (
                  <span className="px-2 py-0.5 text-[10px] font-bold bg-emerald-900/40 text-emerald-400 border border-emerald-600/40 rounded animate-pulse">
                    LIVE
                  </span>
                )}
              </div>
              <p className="text-xs text-gray-500">Entity-to-market mappings with real-time signals</p>
            </div>
          </div>
        </div>

        {/* Stats Row */}
        <div className="flex items-center flex-wrap gap-2 mb-4">
          <StatPill icon={Database} label="Entities" value={stats.totalEntities} color="cyan" />
          <StatPill icon={BarChart3} label="Markets" value={stats.totalMarkets} color="purple" />
          <StatPill icon={Activity} label="Active" value={stats.activeCount} color="emerald" />
          <StatPill
            icon={Zap}
            label="Avg Sentiment"
            value={stats.avgSentiment >= 0 ? `+${stats.avgSentiment.toFixed(1)}` : stats.avgSentiment.toFixed(1)}
            color={stats.avgSentiment >= 0 ? 'emerald' : 'amber'}
          />
        </div>

        {/* Search and Filters */}
        <div className="flex items-center gap-3">
          {/* Search */}
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-500" />
            <input
              type="text"
              placeholder="Search entities, aliases, markets..."
              value={searchQuery}
              onChange={handleSearchChange}
              className="w-full pl-9 pr-3 py-2 bg-gray-800/50 border border-gray-700/40 rounded-lg text-sm text-gray-200 placeholder-gray-500 focus:outline-none focus:border-cyan-600/50"
            />
          </div>

          {/* Type Filter */}
          <div className="relative">
            <Filter className="absolute left-2.5 top-1/2 transform -translate-y-1/2 w-3.5 h-3.5 text-gray-500" />
            <select
              value={filterType}
              onChange={(e) => setFilterType(e.target.value)}
              className="pl-8 pr-8 py-2 bg-gray-800/50 border border-gray-700/40 rounded-lg text-sm text-gray-300 appearance-none cursor-pointer focus:outline-none focus:border-cyan-600/50"
            >
              <option value="all">All Types</option>
              <option value="person">Person</option>
              <option value="organization">Organization</option>
              <option value="position">Position</option>
            </select>
            <ChevronDown className="absolute right-2.5 top-1/2 transform -translate-y-1/2 w-3.5 h-3.5 text-gray-500 pointer-events-none" />
          </div>

          {/* Sort */}
          <div className="relative">
            <ArrowUpDown className="absolute left-2.5 top-1/2 transform -translate-y-1/2 w-3.5 h-3.5 text-gray-500" />
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
              className="pl-8 pr-8 py-2 bg-gray-800/50 border border-gray-700/40 rounded-lg text-sm text-gray-300 appearance-none cursor-pointer focus:outline-none focus:border-cyan-600/50"
            >
              <option value="activity">Recent Activity</option>
              <option value="sentiment">Sentiment</option>
              <option value="markets">Markets Linked</option>
            </select>
            <ChevronDown className="absolute right-2.5 top-1/2 transform -translate-y-1/2 w-3.5 h-3.5 text-gray-500 pointer-events-none" />
          </div>
        </div>
      </div>

      {/* Entity Grid */}
      <div className="p-4">
        {filteredEntities.length === 0 ? (
          <EmptyState entitySystemActive={entitySystemActive} />
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 max-h-[600px] overflow-y-auto custom-scrollbar pr-2">
            {filteredEntities.map(entity => (
              <EntityCard
                key={entity.entity_id}
                entity={entity}
                animationClasses={getAnimationClasses(entity.entity_id)}
                getEntitySentiment={getEntitySentiment}
                isExpanded={expandedEntityIds.has(entity.entity_id)}
                onToggleExpand={() => toggleEntityExpand(entity.entity_id)}
              />
            ))}
          </div>
        )}

        {/* Results count */}
        {filteredEntities.length > 0 && (
          <div className="mt-4 text-center text-xs text-gray-500">
            Showing {filteredEntities.length} of {entityIndex?.entities?.length || 0} entities
          </div>
        )}
      </div>
    </div>
  );
};

export default memo(EntityKnowledgeBasePanel);
