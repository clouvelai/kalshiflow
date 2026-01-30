import React, { useState, useCallback, useMemo, memo } from 'react';
import { formatRelativeTimestamp, formatLatency } from '../../../utils/v3-trader/formatters';
import useRelativeTime from '../../../hooks/v3-trader/useRelativeTime';
import useSignalFreshness from '../../../hooks/v3-trader/useSignalFreshness';
import {
  Brain,
  Sparkles,
  User,
  ArrowRight,
  BarChart3,
  Activity,
  TrendingUp,
  TrendingDown,
  Zap,
  RefreshCw,
  MessageSquare,
  Database,
  GitBranch,
  ChevronDown,
  ChevronRight,
  Wrench,
  BookOpen,
  AlertCircle,
  CheckCircle,
  XCircle,
  Clock,
  DollarSign,
  FileText,
} from 'lucide-react';
import V3Header from '../layout/V3Header';
import renderThinkingMarkdown from '../../../utils/renderThinkingMarkdown';
import { useV3WebSocket } from '../../../hooks/v3-trader/useV3WebSocket';
import { useDeepAgent } from '../../../hooks/v3-trader/useDeepAgent';
import { EntityIndexPanel } from '../panels';
import { CostPanel } from '../panels/DeepAgentPanel';

/**
 * Pipeline Stage Card - Visualizes a stage in the data pipeline
 */
const PipelineStage = memo(({
  icon: Icon,
  title,
  count,
  color = 'gray',
  isActive = false,
  description
}) => {
  const colorClasses = {
    orange: 'from-orange-900/30 to-orange-950/20 border-orange-700/30 text-orange-400',
    violet: 'from-violet-900/30 to-violet-950/20 border-violet-700/30 text-violet-400',
    cyan: 'from-cyan-900/30 to-cyan-950/20 border-cyan-700/30 text-cyan-400',
    emerald: 'from-emerald-900/30 to-emerald-950/20 border-emerald-700/30 text-emerald-400',
    gray: 'from-gray-800/30 to-gray-900/20 border-gray-700/30 text-gray-400',
  };

  return (
    <div className={`
      flex-1 min-w-[140px] p-4 rounded-xl border
      bg-gradient-to-br ${colorClasses[color]}
      ${isActive ? 'ring-1 ring-offset-1 ring-offset-gray-950' : ''}
      transition-all duration-300
    `}>
      <div className="flex items-center gap-2 mb-2">
        <Icon className={`w-4 h-4 ${isActive ? 'animate-pulse' : ''}`} />
        <span className="text-xs font-semibold uppercase tracking-wider">{title}</span>
      </div>
      <div className="text-2xl font-mono font-bold text-white mb-1">
        {count}
      </div>
      <div className="text-[10px] text-gray-500">
        {description}
      </div>
    </div>
  );
});

PipelineStage.displayName = 'PipelineStage';

/**
 * Price Impact Card - Full visualization of entity → market transformation.
 *
 * DATA CONTRACT: This component receives data from useV3WebSocket.entityPriceImpacts
 * which provides RAW snake_case fields from the backend (e.g. impact.sentiment_score,
 * impact.price_impact_score, impact.source_created_at).
 *
 * NOTE: DeepAgentPanel's PriceImpactRow uses DIFFERENT camelCase fields from
 * useDeepAgent (e.g. impact.sentimentScore). Do NOT mix these data sources.
 */
const PriceImpactCard = memo(({ impact, isNew = false, freshnessTier = 'normal', lifecycle = null }) => {
  const sentimentIsPositive = impact.sentiment_score > 0;
  const impactIsPositive = impact.price_impact_score > 0;
  const wasInverted = sentimentIsPositive !== impactIsPositive;

  // Lifecycle state
  const lcStatus = lifecycle?.status || 'new';
  const lcEvalCount = lifecycle?.evaluationCount || 0;
  const lcMaxEvals = lifecycle?.maxEvaluations || 3;
  const lcEvals = lifecycle?.evaluations || [];
  const isTerminal = ['traded', 'passed', 'expired', 'historical'].includes(lcStatus);

  const impactBg = impactIsPositive
    ? 'from-emerald-950/50 via-emerald-900/30 to-gray-900/50'
    : 'from-rose-950/50 via-rose-900/30 to-gray-900/50';

  const sideBadge = impact.suggested_side?.toUpperCase() === 'YES'
    ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40'
    : 'bg-rose-500/20 text-rose-300 border-rose-500/40';

  const marketTypeBadge = {
    OUT: 'bg-amber-900/40 text-amber-300 border-amber-600/40',
    CONFIRM: 'bg-blue-900/40 text-blue-300 border-blue-600/40',
    WIN: 'bg-emerald-900/40 text-emerald-300 border-emerald-600/40',
    NOMINEE: 'bg-violet-900/40 text-violet-300 border-violet-600/40',
    PRESIDENT: 'bg-cyan-900/40 text-cyan-300 border-cyan-600/40',
    UNKNOWN: 'bg-gray-800/40 text-gray-300 border-gray-600/40',
  }[impact.market_type] || 'bg-gray-800/40 text-gray-300 border-gray-600/40';

  // Source type config
  const sourceTypeConfig = {
    reddit_text: { style: 'bg-orange-900/30 text-orange-400 border-orange-600/30', label: 'Text', icon: FileText },
    video_transcript: { style: 'bg-rose-900/30 text-rose-400 border-rose-600/30', label: 'Video', icon: Activity },
    article_extract: { style: 'bg-blue-900/30 text-blue-400 border-blue-600/30', label: 'Article', icon: BookOpen },
  };
  const sourceType = sourceTypeConfig[impact.source_type] || sourceTypeConfig.reddit_text;
  const SourceIcon = sourceType.icon;

  // Live-updating timestamps (snake_case fields)
  const postedAgo = formatRelativeTimestamp(impact.source_created_at);
  const signalAgo = formatRelativeTimestamp(impact.timestamp || impact.created_at);
  const lag = formatLatency(impact.source_created_at, impact.timestamp || impact.created_at);

  // Contextual subtitles
  const sentimentSubtitle = impact.context_snippet
    ? (impact.context_snippet.length > 50 ? impact.context_snippet.slice(0, 50) + '...' : impact.context_snippet)
    : `${impact.source_type === 'reddit_text' ? 'Reddit' : 'Source'} discussion: ${sentimentIsPositive ? 'positive' : 'negative'} tone`;

  const impactSubtitle = wasInverted
    ? `${impact.market_type || 'OUT'}: negative news helps`
    : `${impact.market_type === 'WIN' ? 'WIN' : 'Direct'}: sentiment aligns`;

  const confidenceSubtitle = impact.confidence >= 0.9
    ? 'High: strong match'
    : impact.confidence >= 0.7
      ? 'Medium: likely match'
      : 'Low: uncertain match';

  // Build animation classes
  const animationClasses = [
    isNew ? 'animate-signal-slide-in' : '',
    freshnessTier === 'fresh' ? 'animate-signal-fresh' : '',
    freshnessTier === 'recent' ? 'animate-signal-recent' : '',
  ].filter(Boolean).join(' ');

  // Visual de-emphasis for terminal/historical signals
  const opacityClass = lcStatus === 'historical' ? 'opacity-40'
    : (lcStatus === 'expired' || lcStatus === 'passed') ? 'opacity-60'
    : '';
  const borderAccent = lcStatus === 'traded' ? 'border-l-2 border-l-emerald-400' : '';
  const hoverClass = isTerminal ? '' : 'hover:border-gray-600/60';
  const borderStyle = lcStatus === 'historical' ? 'border-gray-800/20' : 'border-gray-700/50';

  // Lifecycle status badge
  const lifecycleBadgeConfig = {
    new: { style: 'bg-blue-900/30 text-blue-400 border-blue-600/30', label: 'New' },
    evaluating: { style: 'bg-cyan-900/30 text-cyan-400 border-cyan-600/30', label: `Eval ${lcEvalCount}/${lcMaxEvals}` },
    traded: { style: 'bg-emerald-900/30 text-emerald-400 border-emerald-600/30', label: 'Traded' },
    passed: { style: 'bg-gray-700/40 text-gray-400 border-gray-600/30', label: 'Passed' },
    expired: { style: 'bg-gray-800/40 text-gray-500 border-gray-700/30', label: `Expired (${lcMaxEvals}/${lcMaxEvals})` },
    historical: { style: 'bg-gray-800/30 text-gray-600 border-gray-700/20', label: 'Historical' },
  };
  const lcBadge = lifecycleBadgeConfig[lcStatus] || lifecycleBadgeConfig.new;

  return (
    <div className={`
      relative overflow-hidden rounded-2xl border ${borderStyle}
      bg-gradient-to-br ${impactBg}
      backdrop-blur-sm p-5
      transition-all duration-300 ${hoverClass}
      group
      ${animationClasses}
      ${opacityClass}
      ${borderAccent}
    `}>
      {/* Glow effect */}
      {!isTerminal && (
        <div className={`
          absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500
          ${impactIsPositive ? 'bg-emerald-500/5' : 'bg-rose-500/5'}
        `} />
      )}

      {/* Header */}
      <div className="relative flex items-start justify-between mb-3">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-xl bg-gray-800/60 border border-gray-700/40">
            <User className="w-5 h-5 text-gray-300" />
          </div>
          <div>
            <div className="text-lg font-semibold text-white">
              {impact.entity_name}
            </div>
            {impact.source_subreddit && (
              <div className="flex items-center gap-1.5 mt-0.5">
                <MessageSquare className="w-3 h-3 text-orange-400" />
                <span className="text-xs text-orange-400">r/{impact.source_subreddit}</span>
              </div>
            )}
          </div>
        </div>

        <div className="flex items-center gap-2">
          {/* Lifecycle Badge */}
          <span className={`px-2 py-0.5 rounded-lg text-[10px] font-bold border ${lcBadge.style}`}>
            {lcBadge.label}
          </span>
          <span className={`px-2.5 py-1 rounded-lg text-[10px] font-bold border ${marketTypeBadge}`}>
            {impact.market_type}
          </span>
          {impact.suggested_side && (
            <span className={`px-3 py-1.5 rounded-lg text-xs font-bold border ${sideBadge}`}>
              {impact.suggested_side}
            </span>
          )}
        </div>
      </div>

      {/* Evaluation Progress Bar */}
      {lcMaxEvals > 0 && (lcEvalCount > 0 || lcStatus !== 'new') && (
        <div className="flex gap-0.5 h-1 mb-3">
          {Array.from({ length: lcMaxEvals }, (_, i) => (
            <div key={i} className={`flex-1 rounded-full ${
              i < lcEvalCount
                ? lcEvals[i]?.decision === 'TRADE' ? 'bg-emerald-400'
                  : lcEvals[i]?.decision === 'WAIT' ? 'bg-amber-400'
                  : 'bg-gray-500'
                : 'bg-gray-800'
            }`} />
          ))}
        </div>
      )}

      {/* Dual Timestamp Bar: Posted | Signal | Lag */}
      {(postedAgo || signalAgo) && (
        <div className="relative flex items-center gap-4 mb-4 px-3 py-2 bg-gray-900/50 rounded-lg border border-gray-800/30 text-xs">
          {postedAgo && (
            <span className="flex items-center gap-1.5">
              <Clock className="w-3.5 h-3.5 text-gray-500" />
              <span className="text-gray-500">Posted</span>
              <span className="text-orange-400 font-medium">{postedAgo}</span>
            </span>
          )}
          {postedAgo && signalAgo && (
            <span className="text-gray-700">|</span>
          )}
          {signalAgo && (
            <span className="flex items-center gap-1.5">
              <Zap className="w-3.5 h-3.5 text-gray-500" />
              <span className="text-gray-500">Signal</span>
              <span className="text-cyan-400 font-medium">{signalAgo}</span>
            </span>
          )}
          {lag && (
            <>
              <span className="text-gray-700">|</span>
              <span className="flex items-center gap-1.5">
                <Activity className="w-3.5 h-3.5 text-gray-500" />
                <span className="text-gray-500">{lag} lag</span>
              </span>
            </>
          )}
        </div>
      )}

      {/* Source Context: Reddit Post Title + Context Snippet */}
      {(impact.source_title || impact.context_snippet) && (
        <div className="relative mb-4 p-3 bg-gray-900/60 rounded-xl border border-gray-800/40">
          <div className="flex items-start gap-2">
            <div className="flex-shrink-0 mt-0.5">
              <SourceIcon className={`w-3.5 h-3.5 ${sourceType.style.includes('orange') ? 'text-orange-400/70' : sourceType.style.includes('rose') ? 'text-rose-400/70' : 'text-blue-400/70'}`} />
            </div>
            <div className="flex-1 min-w-0">
              {impact.source_title && (
                <div className="text-xs font-medium text-gray-200 line-clamp-2 leading-relaxed" title={impact.source_title}>
                  "{impact.source_title}"
                </div>
              )}
              {impact.context_snippet && (
                <div className="text-[10px] text-gray-400 mt-1 line-clamp-3" title={impact.context_snippet}>
                  {impact.context_snippet}
                </div>
              )}
              <div className="mt-1.5">
                <span className={`px-1.5 py-0.5 rounded text-[8px] font-medium border ${sourceType.style}`}>
                  {sourceType.label}
                </span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Transformation Pipeline */}
      <div className="relative flex items-center gap-4 p-4 bg-gray-900/50 rounded-xl border border-gray-800/60 mb-4">
        {/* Sentiment */}
        <div className="flex-1 text-center">
          <div className="text-[10px] text-gray-500 uppercase tracking-wider mb-2">Entity Sentiment</div>
          <div className={`text-3xl font-mono font-bold ${sentimentIsPositive ? 'text-emerald-400' : 'text-rose-400'}`}>
            {impact.sentiment_score > 0 ? '+' : ''}{impact.sentiment_score}
          </div>
          <div className="text-[10px] text-gray-500 mt-1">
            {sentimentSubtitle}
          </div>
        </div>

        {/* Arrow */}
        <div className="flex flex-col items-center px-4">
          <div className={`
            p-2 rounded-full
            ${wasInverted ? 'bg-amber-900/40 border border-amber-600/40' : 'bg-gray-800/50 border border-gray-700/40'}
          `}>
            <ArrowRight className={`w-5 h-5 ${wasInverted ? 'text-amber-400' : 'text-gray-500'}`} />
          </div>
          {wasInverted && (
            <span className="text-[9px] text-amber-400 font-semibold mt-1.5 uppercase tracking-wider">
              Inverted
            </span>
          )}
        </div>

        {/* Price Impact */}
        <div className="flex-1 text-center">
          <div className="text-[10px] text-gray-500 uppercase tracking-wider mb-2">Price Impact</div>
          <div className={`text-3xl font-mono font-bold ${impactIsPositive ? 'text-emerald-400' : 'text-rose-400'}`}>
            {impact.price_impact_score > 0 ? '+' : ''}{impact.price_impact_score}
          </div>
          <div className="text-[10px] text-gray-500 mt-1" title={impact.transformation_logic}>
            {impactSubtitle}
          </div>
        </div>

        {/* Confidence */}
        <div className="w-px h-16 bg-gray-700/50" />
        <div className="flex-1 text-center">
          <div className="text-[10px] text-gray-500 uppercase tracking-wider mb-2">Confidence</div>
          <div className="text-3xl font-mono font-bold text-cyan-400">
            {(impact.confidence * 100).toFixed(0)}%
          </div>
          <div className="text-[10px] text-gray-500 mt-1">{confidenceSubtitle}</div>
        </div>
      </div>

      {/* Market Info */}
      <div className="relative flex items-center justify-between border-t border-gray-800/30 pt-2">
        <div className="flex items-center gap-2">
          <BarChart3 className="w-4 h-4 text-gray-500" />
          <span className="font-mono text-sm text-gray-300">{impact.market_ticker}</span>
        </div>
        <span className="text-[10px] text-gray-400 max-w-[300px] truncate" title={impact.transformation_logic}>
          {impact.transformation_logic || (wasInverted ? 'Sentiment inverted for OUT market' : 'Direct sentiment correlation')}
        </span>
      </div>
    </div>
  );
});

PriceImpactCard.displayName = 'PriceImpactCard';

/**
 * Entity Extraction Card - Shows extracted entity with sentiment
 */
const EntityExtractionCard = memo(({ extraction }) => {
  const sentimentColor = extraction.sentiment_score > 0 ? 'text-emerald-400' : 'text-rose-400';
  const sentimentBg = extraction.sentiment_score > 0 ? 'bg-emerald-900/20' : 'bg-rose-900/20';

  return (
    <div className="p-3 bg-gray-800/30 rounded-lg border border-gray-700/30 hover:border-gray-600/40 transition-colors">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <User className="w-3.5 h-3.5 text-violet-400" />
          <span className="text-sm font-medium text-gray-200">{extraction.canonical_name}</span>
          <span className="px-1.5 py-0.5 text-[9px] bg-gray-700/50 text-gray-400 rounded">
            {extraction.entity_type}
          </span>
        </div>
        <div className={`px-2 py-0.5 rounded text-xs font-mono font-bold ${sentimentBg} ${sentimentColor}`}>
          {extraction.sentiment_score > 0 ? '+' : ''}{extraction.sentiment_score}
        </div>
      </div>
      {extraction.context_snippet && (
        <div className="text-[10px] text-gray-500 truncate" title={extraction.context_snippet}>
          "{extraction.context_snippet}"
        </div>
      )}
    </div>
  );
});

EntityExtractionCard.displayName = 'EntityExtractionCard';

/**
 * Related Entity Card - Shows entity detected by spaCy NER but NOT in Knowledge Base
 * These are PERSON, ORG, GPE, EVENT entities that don't map to any Kalshi market
 */
const RelatedEntityCard = memo(({ entity }) => {
  const sentimentColor = entity.sentiment_score > 0 ? 'text-emerald-400' : 'text-rose-400';
  const sentimentBg = entity.sentiment_score > 0 ? 'bg-emerald-900/20' : 'bg-rose-900/20';

  // Entity type colors
  const typeColors = {
    PERSON: 'bg-blue-900/30 text-blue-300 border-blue-700/30',
    ORG: 'bg-purple-900/30 text-purple-300 border-purple-700/30',
    GPE: 'bg-amber-900/30 text-amber-300 border-amber-700/30',
    EVENT: 'bg-pink-900/30 text-pink-300 border-pink-700/30',
  };
  const typeColor = typeColors[entity.entity_type] || 'bg-gray-700/50 text-gray-400';

  return (
    <div className="p-3 bg-gray-800/30 rounded-lg border border-gray-700/30 hover:border-blue-700/30 transition-colors">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <User className="w-3.5 h-3.5 text-blue-400" />
          <span className="text-sm font-medium text-gray-200">{entity.entity_text}</span>
          <span className={`px-1.5 py-0.5 text-[9px] rounded border ${typeColor}`}>
            {entity.entity_type}
          </span>
        </div>
        <div className={`px-2 py-0.5 rounded text-xs font-mono font-bold ${sentimentBg} ${sentimentColor}`}>
          {entity.sentiment_score > 0 ? '+' : ''}{entity.sentiment_score}
        </div>
      </div>
      {entity.source_subreddit && (
        <div className="flex items-center gap-1.5 mb-1">
          <MessageSquare className="w-3 h-3 text-orange-400" />
          <span className="text-[10px] text-orange-400">r/{entity.source_subreddit}</span>
        </div>
      )}
      {entity.context_snippet && (
        <div className="text-[10px] text-gray-500 truncate" title={entity.context_snippet}>
          "{entity.context_snippet}"
        </div>
      )}
      {entity.co_occurring_market_entities?.length > 0 && (
        <div className="mt-2 pt-2 border-t border-gray-700/30">
          <span className="text-[9px] text-gray-500">Co-occurring with: </span>
          {entity.co_occurring_market_entities.slice(0, 3).map((me, idx) => (
            <span key={idx} className="text-[9px] text-cyan-400 mr-1">{me}</span>
          ))}
        </div>
      )}
    </div>
  );
});

RelatedEntityCard.displayName = 'RelatedEntityCard';

/**
 * Reddit Post Card - Shows a Reddit post from the stream with timestamp and extracted entities
 */
const RedditPostCard = memo(({ post, entities = [] }) => {
  // Filter entities that belong to this post
  const postEntities = entities.filter(e => e.post_id === post.post_id);

  // Format timestamp
  const formatTime = (utc) => {
    if (!utc) return '';
    const date = new Date(utc * 1000);
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);

    if (diffMins < 1) return 'just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    const diffHours = Math.floor(diffMins / 60);
    if (diffHours < 24) return `${diffHours}h ago`;
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  };

  return (
    <div className="p-3 bg-gray-800/30 rounded-lg border border-gray-700/30 hover:border-orange-700/30 transition-colors">
      <div className="flex items-center justify-between mb-1.5">
        <div className="flex items-center gap-2">
          <span className="text-[10px] text-orange-400 font-medium">r/{post.subreddit}</span>
          {post.score > 0 && (
            <span className="text-[10px] text-gray-500">↑{post.score}</span>
          )}
        </div>
        {post.created_utc && (
          <span className="text-[9px] text-gray-600">{formatTime(post.created_utc)}</span>
        )}
      </div>
      <div className="text-xs text-gray-300 line-clamp-2 mb-1.5" title={post.title}>
        {post.title}
      </div>
      {/* Extracted entities from this post */}
      {postEntities.length > 0 && (
        <div className="flex flex-wrap gap-1 mt-2 pt-2 border-t border-gray-700/30">
          {postEntities.map((entity, idx) => {
            const sentimentColor = entity.sentiment_score > 0 ? 'text-emerald-400 bg-emerald-900/30 border-emerald-700/30' : 'text-rose-400 bg-rose-900/30 border-rose-700/30';
            return (
              <span
                key={`${entity.entity_id}_${idx}`}
                className={`inline-flex items-center gap-1 px-1.5 py-0.5 text-[9px] rounded border ${sentimentColor}`}
                title={`${entity.canonical_name}: ${entity.sentiment_score > 0 ? '+' : ''}${entity.sentiment_score}`}
              >
                <span className="font-medium truncate max-w-[80px]">{entity.canonical_name}</span>
                <span className="font-mono">{entity.sentiment_score > 0 ? '+' : ''}{entity.sentiment_score}</span>
              </span>
            );
          })}
        </div>
      )}
    </div>
  );
});

RedditPostCard.displayName = 'RedditPostCard';

/**
 * Agent Status Header - Shows running state, cycle count, P&L
 */
const AgentStatusHeader = memo(({ agentState, thinking, settlements, trades }) => {
  const isRunning = agentState.status === 'active' || agentState.status === 'started';

  // Calculate total P&L from settlements
  const totalPnL = useMemo(() => {
    return settlements.reduce((sum, s) => sum + (s.pnlCents || 0), 0) / 100;
  }, [settlements]);

  const pnlColor = totalPnL >= 0 ? 'text-emerald-400' : 'text-rose-400';
  const pnlBg = totalPnL >= 0 ? 'bg-emerald-900/20' : 'bg-rose-900/20';

  return (
    <div className="flex items-center justify-between p-3 bg-gray-800/50 rounded-xl border border-gray-700/40">
      <div className="flex items-center gap-4">
        <div className={`
          flex items-center gap-2 px-3 py-1.5 rounded-lg border
          ${isRunning
            ? 'bg-emerald-900/30 border-emerald-600/40 text-emerald-400'
            : 'bg-gray-800/50 border-gray-700/40 text-gray-400'}
        `}>
          {isRunning ? (
            <>
              <RefreshCw className="w-3.5 h-3.5 animate-spin" />
              <span className="text-xs font-bold">RUNNING</span>
            </>
          ) : (
            <>
              <Activity className="w-3.5 h-3.5" />
              <span className="text-xs font-bold">STOPPED</span>
            </>
          )}
        </div>

        <div className="flex items-center gap-2 text-xs text-gray-400">
          <Clock className="w-3.5 h-3.5" />
          <span>Cycle <span className="font-mono text-white">{agentState.cycleCount}</span></span>
        </div>

        <div className="flex items-center gap-2 text-xs text-gray-400">
          <BarChart3 className="w-3.5 h-3.5" />
          <span>Trades <span className="font-mono text-white">{trades.length}</span></span>
        </div>
      </div>

      <div className={`flex items-center gap-2 px-3 py-1.5 rounded-lg ${pnlBg}`}>
        <DollarSign className="w-3.5 h-3.5 text-gray-400" />
        <span className={`text-sm font-mono font-bold ${pnlColor}`}>
          {totalPnL >= 0 ? '+' : ''}{totalPnL.toFixed(2)}
        </span>
        <span className="text-[10px] text-gray-500">P&L</span>
      </div>
    </div>
  );
});

AgentStatusHeader.displayName = 'AgentStatusHeader';

/**
 * Thinking Stream - Real-time agent reasoning with pulse animation
 */
const ThinkingStream = memo(({ thinking, isRunning }) => {
  if (!thinking.text) {
    return (
      <div className="p-4 bg-gray-800/30 rounded-xl border border-gray-700/30">
        <div className="flex items-center gap-2 mb-2">
          <Brain className="w-4 h-4 text-violet-400" />
          <span className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Thinking</span>
        </div>
        <div className="text-sm text-gray-500 italic">
          Waiting for next cycle...
        </div>
      </div>
    );
  }

  return (
    <div className="p-4 bg-gradient-to-br from-violet-900/20 to-violet-950/10 rounded-xl border border-violet-700/30">
      <div className="flex items-center gap-2 mb-2">
        <Brain className="w-4 h-4 text-violet-400 animate-pulse" />
        <span className="text-xs font-semibold text-violet-300 uppercase tracking-wider">Thinking</span>
        <span className="ml-auto text-[10px] text-gray-500 font-mono">Cycle {thinking.cycle}</span>
      </div>
      <div className="text-sm text-gray-200 leading-relaxed">
        {renderThinkingMarkdown(thinking.text)}
      </div>
      {isRunning && (
        <div className="mt-2 flex items-center gap-2">
          <div className="w-2 h-2 bg-violet-400 rounded-full animate-pulse" />
          <span className="text-[10px] text-violet-400">Processing...</span>
        </div>
      )}
    </div>
  );
});

ThinkingStream.displayName = 'ThinkingStream';

/**
 * Tool Call Card - Shows a single tool invocation
 */
const ToolCallCard = memo(({ toolCall }) => {
  const toolIcons = {
    get_price_impacts: Sparkles,
    get_markets: BarChart3,
    trade: TrendingUp,
    read_memory: BookOpen,
    write_memory: BookOpen,
    get_event_context: Database,
    think: Brain,
    get_session_state: Activity,
  };

  const Icon = toolIcons[toolCall.tool] || Wrench;

  return (
    <div className="flex items-start gap-2 p-2 bg-gray-800/40 rounded-lg">
      <Icon className="w-3.5 h-3.5 text-cyan-400 mt-0.5 flex-shrink-0" />
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <span className="text-xs font-mono text-cyan-300">{toolCall.tool}()</span>
          {toolCall.outputPreview && (
            <span className="text-[10px] text-gray-500 truncate">
              • {toolCall.outputPreview}
            </span>
          )}
        </div>
      </div>
    </div>
  );
});

ToolCallCard.displayName = 'ToolCallCard';

/**
 * Tool Calls Panel - Collapsible list of tool invocations
 */
const ToolCallsPanel = memo(({ toolCalls, defaultCollapsed = true }) => {
  const [collapsed, setCollapsed] = useState(defaultCollapsed);

  return (
    <div className="bg-gray-900/50 rounded-xl border border-gray-800/50 overflow-hidden">
      <button
        onClick={() => setCollapsed(!collapsed)}
        className="w-full flex items-center justify-between p-3 hover:bg-gray-800/30 transition-colors"
      >
        <div className="flex items-center gap-2">
          <Wrench className="w-4 h-4 text-cyan-400" />
          <span className="text-xs font-semibold text-gray-300">Tool Calls</span>
          <span className="px-2 py-0.5 bg-cyan-900/30 text-cyan-400 text-[10px] font-bold rounded-full">
            {toolCalls.length}
          </span>
        </div>
        {collapsed ? <ChevronRight className="w-4 h-4 text-gray-500" /> : <ChevronDown className="w-4 h-4 text-gray-500" />}
      </button>
      {!collapsed && (
        <div className="px-3 pb-3 space-y-1.5 max-h-[200px] overflow-y-auto">
          {toolCalls.length === 0 ? (
            <div className="text-center py-4 text-gray-500 text-xs">
              No tool calls yet...
            </div>
          ) : (
            toolCalls.slice(0, 15).map((tc) => (
              <ToolCallCard key={tc.id} toolCall={tc} />
            ))
          )}
        </div>
      )}
    </div>
  );
});

ToolCallsPanel.displayName = 'ToolCallsPanel';

/**
 * Trade Card - Shows an executed trade
 */
const TradeCard = memo(({ trade }) => {
  const isYes = trade.side?.toLowerCase() === 'yes';
  const sideColor = isYes ? 'text-emerald-400' : 'text-rose-400';
  const sideBg = isYes ? 'bg-emerald-900/20' : 'bg-rose-900/20';

  return (
    <div className={`flex items-center justify-between p-2.5 rounded-lg border ${sideBg} border-gray-700/30`}>
      <div className="flex items-center gap-2">
        <div className={`text-xs font-bold ${sideColor}`}>
          {isYes ? 'YES' : 'NO'}
        </div>
        <span className="text-xs font-mono text-gray-300">{trade.ticker}</span>
        <span className="text-[10px] text-gray-500">
          {trade.contracts}x @ {trade.priceCents}c
        </span>
      </div>
      <span className="text-[10px] text-gray-500">
        ${((trade.contracts * trade.priceCents) / 100).toFixed(2)}
      </span>
    </div>
  );
});

TradeCard.displayName = 'TradeCard';

/**
 * Settlement Card - Shows a settled trade with P&L
 */
const SettlementCard = memo(({ settlement }) => {
  const isWin = settlement.result === 'win';
  const isLoss = settlement.result === 'loss';
  const pnlCents = settlement.pnlCents || 0;

  const ResultIcon = isWin ? CheckCircle : isLoss ? XCircle : AlertCircle;
  const resultColor = isWin ? 'text-emerald-400' : isLoss ? 'text-rose-400' : 'text-gray-400';
  const resultBg = isWin ? 'bg-emerald-900/20' : isLoss ? 'bg-rose-900/20' : 'bg-gray-800/30';

  return (
    <div className={`flex items-center justify-between p-2.5 rounded-lg border ${resultBg} border-gray-700/30`}>
      <div className="flex items-center gap-2">
        <ResultIcon className={`w-3.5 h-3.5 ${resultColor}`} />
        <span className="text-xs font-mono text-gray-300">{settlement.ticker}</span>
        <span className="text-[10px] text-gray-500 capitalize">{settlement.result}</span>
      </div>
      <span className={`text-xs font-mono font-bold ${resultColor}`}>
        {pnlCents >= 0 ? '+' : ''}${(pnlCents / 100).toFixed(2)}
      </span>
    </div>
  );
});

SettlementCard.displayName = 'SettlementCard';

/**
 * Trades & Settlements Panel - Shows recent trades and P&L
 */
const TradesPanel = memo(({ trades, settlements }) => {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <div className="bg-gray-900/50 rounded-xl border border-gray-800/50 overflow-hidden">
      <button
        onClick={() => setCollapsed(!collapsed)}
        className="w-full flex items-center justify-between p-3 hover:bg-gray-800/30 transition-colors"
      >
        <div className="flex items-center gap-2">
          <TrendingUp className="w-4 h-4 text-emerald-400" />
          <span className="text-xs font-semibold text-gray-300">Trades</span>
          <span className="px-2 py-0.5 bg-emerald-900/30 text-emerald-400 text-[10px] font-bold rounded-full">
            {trades.length}
          </span>
          {settlements.length > 0 && (
            <span className="px-2 py-0.5 bg-gray-700/30 text-gray-400 text-[10px] font-bold rounded-full">
              {settlements.length} settled
            </span>
          )}
        </div>
        {collapsed ? <ChevronRight className="w-4 h-4 text-gray-500" /> : <ChevronDown className="w-4 h-4 text-gray-500" />}
      </button>
      {!collapsed && (
        <div className="px-3 pb-3 space-y-1.5 max-h-[250px] overflow-y-auto">
          {trades.length === 0 && settlements.length === 0 ? (
            <div className="text-center py-4 text-gray-500 text-xs">
              No trades yet...
            </div>
          ) : (
            <>
              {settlements.slice(0, 5).map((s) => (
                <SettlementCard key={s.id} settlement={s} />
              ))}
              {trades.slice(0, 10).map((t) => (
                <TradeCard key={t.id} trade={t} />
              ))}
            </>
          )}
        </div>
      )}
    </div>
  );
});

TradesPanel.displayName = 'TradesPanel';

/**
 * Learnings Panel - Shows what the agent has learned
 */
const LearningsPanel = memo(({ learnings, memoryUpdates }) => {
  const [collapsed, setCollapsed] = useState(true);

  return (
    <div className="bg-gray-900/50 rounded-xl border border-gray-800/50 overflow-hidden">
      <button
        onClick={() => setCollapsed(!collapsed)}
        className="w-full flex items-center justify-between p-3 hover:bg-gray-800/30 transition-colors"
      >
        <div className="flex items-center gap-2">
          <BookOpen className="w-4 h-4 text-amber-400" />
          <span className="text-xs font-semibold text-gray-300">Learnings</span>
          <span className="px-2 py-0.5 bg-amber-900/30 text-amber-400 text-[10px] font-bold rounded-full">
            {learnings.length}
          </span>
        </div>
        {collapsed ? <ChevronRight className="w-4 h-4 text-gray-500" /> : <ChevronDown className="w-4 h-4 text-gray-500" />}
      </button>
      {!collapsed && (
        <div className="px-3 pb-3 space-y-2 max-h-[200px] overflow-y-auto">
          {learnings.length === 0 ? (
            <div className="text-center py-4 text-gray-500 text-xs">
              No learnings recorded yet...
            </div>
          ) : (
            learnings.slice(0, 10).map((learning) => (
              <div key={learning.id} className="p-2 bg-gray-800/40 rounded-lg">
                <div className="text-xs text-gray-300">{learning.content}</div>
              </div>
            ))
          )}
        </div>
      )}
    </div>
  );
});

LearningsPanel.displayName = 'LearningsPanel';

/**
 * AgentPage - Dedicated full-page view for the Deep Agent
 *
 * Displays the complete Reddit Entity → Kalshi Market pipeline:
 * 1. Reddit posts streaming in
 * 2. Entity extraction with sentiment
 * 3. Price impact transformation
 * 4. Agent activity and trades
 */
const AgentPage = () => {
  const [showPosts, setShowPosts] = useState(true);
  const [showEntities, setShowEntities] = useState(true);
  const [showRelatedEntities, setShowRelatedEntities] = useState(true);

  // Live-updating timestamps: forces re-render every 30s
  useRelativeTime(30000);

  // Initialize deep agent hook first to get processMessage
  const {
    agentState,
    thinking,
    toolCalls,
    trades,
    settlements,
    learnings,
    memoryUpdates,
    processMessage,
    isRunning: agentIsRunning,
    getSignalLifecycle,
    lifecycleSummary,
    getStatusSortPriority,
  } = useDeepAgent({ useV3WebSocketState: true });

  // Wire deep agent message processing to WebSocket
  const handleMessage = useCallback((type, message, context) => {
    // Process deep agent messages
    processMessage(type, message);
  }, [processMessage]);

  const {
    wsStatus,
    currentState,
    tradingState,
    strategyStatus,
    entityRedditPosts,
    entityExtractions,
    entityPriceImpacts,
    entityStats,
    entitySystemActive,
    entityIndex,
    redditAgentHealth,
    relatedEntities,
  } = useV3WebSocket({ onMessage: handleMessage });

  // Get deep agent strategy data
  const deepAgentStrategy = strategyStatus?.strategies?.deep_agent;
  const isAgentRunning = agentIsRunning || deepAgentStrategy?.running || entitySystemActive;

  // Track new signal arrivals + freshness tiers for price impact cards
  const { newSignalIds, getFreshnessTier } = useSignalFreshness(entityPriceImpacts);

  // Sort price impacts by lifecycle priority (evaluating > new > traded > passed/expired > historical)
  const sortedPriceImpacts = useMemo(() => {
    if (!entityPriceImpacts || entityPriceImpacts.length === 0) return [];
    return [...entityPriceImpacts].sort((a, b) => {
      const pa = getStatusSortPriority(a.signal_id);
      const pb = getStatusSortPriority(b.signal_id);
      if (pa !== pb) return pa - pb;
      // Within same priority, sort by timestamp (newest first)
      return (b.timestamp || 0) - (a.timestamp || 0);
    });
  }, [entityPriceImpacts, getStatusSortPriority]);

  // Pipeline stats
  const pipelineStats = useMemo(() => ({
    postsProcessed: entityStats?.postsProcessed || 0,
    entitiesExtracted: entityStats?.entitiesExtracted || 0,
    signalsGenerated: entityStats?.signalsGenerated || 0,
    indexSize: entityStats?.indexSize || 0,
  }), [entityStats]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-950 via-gray-900 to-gray-950">
      <V3Header
        wsStatus={wsStatus}
        currentState={currentState}
        balance={tradingState?.balance}
      />

      <div className="max-w-7xl mx-auto px-6 py-6">
        {/* Page Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="p-3 rounded-2xl bg-gradient-to-br from-violet-900/50 to-violet-950/30 border border-violet-700/30">
                <Brain className={`w-8 h-8 text-violet-400 ${isAgentRunning ? 'animate-pulse' : ''}`} />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-white">Deep Agent</h1>
                <p className="text-sm text-gray-400 mt-0.5">
                  Reddit Entity → Kalshi Market Trading Pipeline
                </p>
              </div>
            </div>

            <div className={`
              flex items-center gap-2 px-4 py-2 rounded-lg border
              ${isAgentRunning
                ? 'bg-emerald-900/30 border-emerald-600/40 text-emerald-400'
                : 'bg-gray-800/50 border-gray-700/40 text-gray-400'}
            `}>
              {isAgentRunning ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  <span className="font-semibold">ACTIVE</span>
                </>
              ) : (
                <>
                  <Activity className="w-4 h-4" />
                  <span className="font-semibold">INACTIVE</span>
                </>
              )}
            </div>
          </div>
        </div>

        {/* Pipeline Visualization */}
        <div className="mb-8">
          <div className="flex items-center gap-2 mb-4">
            <GitBranch className="w-4 h-4 text-gray-500" />
            <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider">
              Data Pipeline
            </h2>
          </div>

          <div className="flex items-center gap-3">
            <PipelineStage
              icon={MessageSquare}
              title="Reddit"
              count={pipelineStats.postsProcessed}
              color="orange"
              isActive={entityRedditPosts.length > 0}
              description="Posts processed"
            />
            <ArrowRight className="w-5 h-5 text-gray-600 flex-shrink-0" />
            <PipelineStage
              icon={User}
              title="Entities"
              count={pipelineStats.entitiesExtracted}
              color="violet"
              isActive={entityExtractions.length > 0}
              description="Entities extracted"
            />
            <ArrowRight className="w-5 h-5 text-gray-600 flex-shrink-0" />
            <PipelineStage
              icon={Sparkles}
              title="Signals"
              count={pipelineStats.signalsGenerated}
              color="cyan"
              isActive={entityPriceImpacts.length > 0}
              description="Price impacts"
            />
            <ArrowRight className="w-5 h-5 text-gray-600 flex-shrink-0" />
            <PipelineStage
              icon={Database}
              title="Index"
              count={pipelineStats.indexSize}
              color="emerald"
              description="Market mappings"
            />
          </div>
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-12 gap-6">
          {/* Left Column - Data Sources */}
          <div className="col-span-4 space-y-6">
            {/* Reddit Posts */}
            <div className="bg-gray-900/50 rounded-2xl border border-gray-800/50 overflow-hidden">
              <button
                onClick={() => setShowPosts(!showPosts)}
                className="w-full flex items-center justify-between p-4 hover:bg-gray-800/30 transition-colors"
              >
                <div className="flex items-center gap-2">
                  <MessageSquare className="w-4 h-4 text-orange-400" />
                  <span className="text-sm font-semibold text-gray-300">Reddit Stream</span>
                  <span className="px-2 py-0.5 bg-orange-900/30 text-orange-400 text-[10px] font-bold rounded-full">
                    {entityRedditPosts.length}
                  </span>
                  {/* Health Status Indicator */}
                  {redditAgentHealth.health === 'healthy' ? (
                    <span className="flex items-center gap-1 px-1.5 py-0.5 bg-emerald-900/30 text-emerald-400 text-[9px] rounded-full border border-emerald-700/30" title={`Connected to r/${redditAgentHealth.subreddits?.join(', r/') || 'politics, news'}`}>
                      <CheckCircle className="w-3 h-3" />
                      <span className="font-medium">Live</span>
                    </span>
                  ) : redditAgentHealth.health === 'degraded' ? (
                    <span className="flex items-center gap-1 px-1.5 py-0.5 bg-amber-900/30 text-amber-400 text-[9px] rounded-full border border-amber-700/30" title="Reddit connected but NLP/KB not ready">
                      <AlertCircle className="w-3 h-3" />
                      <span className="font-medium">Partial</span>
                    </span>
                  ) : redditAgentHealth.health === 'unhealthy' ? (
                    <span className="flex items-center gap-1 px-1.5 py-0.5 bg-red-900/30 text-red-400 text-[9px] rounded-full border border-red-700/30" title={redditAgentHealth.lastError || 'Reddit API not connected'}>
                      <XCircle className="w-3 h-3" />
                      <span className="font-medium">Offline</span>
                    </span>
                  ) : (
                    <span className="flex items-center gap-1 px-1.5 py-0.5 bg-gray-800/50 text-gray-500 text-[9px] rounded-full border border-gray-700/30" title="Waiting for status...">
                      <Clock className="w-3 h-3" />
                      <span className="font-medium">...</span>
                    </span>
                  )}
                </div>
                {showPosts ? <ChevronDown className="w-4 h-4 text-gray-500" /> : <ChevronRight className="w-4 h-4 text-gray-500" />}
              </button>
              {showPosts && (
                <div className="px-4 pb-4 space-y-2 max-h-[300px] overflow-y-auto">
                  {entityRedditPosts.length === 0 ? (
                    <div className="text-center py-6 text-gray-500 text-sm">
                      {redditAgentHealth.health === 'unhealthy' ? (
                        <div className="space-y-2">
                          <XCircle className="w-8 h-8 text-red-500 mx-auto" />
                          <div className="text-red-400 font-medium">Reddit API not connected</div>
                          <div className="text-[11px] text-gray-600">
                            {redditAgentHealth.lastError || 'Check REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET env vars'}
                          </div>
                        </div>
                      ) : redditAgentHealth.health === 'degraded' ? (
                        <div className="space-y-2">
                          <AlertCircle className="w-8 h-8 text-amber-500 mx-auto" />
                          <div className="text-amber-400 font-medium">Partially connected</div>
                          <div className="text-[11px] text-gray-600">
                            Reddit: ✓ | NLP: {redditAgentHealth.nlpAvailable ? '✓' : '✗'} | KB: {redditAgentHealth.kbAvailable ? '✓' : '✗'}
                          </div>
                        </div>
                      ) : redditAgentHealth.health === 'healthy' ? (
                        <div className="space-y-2">
                          <RefreshCw className="w-6 h-6 text-orange-400 mx-auto animate-spin" />
                          <div>Listening to r/{redditAgentHealth.subreddits?.join(', r/') || 'politics, news'}...</div>
                        </div>
                      ) : (
                        <div className="space-y-2">
                          <Clock className="w-6 h-6 text-gray-500 mx-auto" />
                          <div>Waiting for Reddit posts...</div>
                        </div>
                      )}
                    </div>
                  ) : (
                    entityRedditPosts.slice(0, 10).map((post) => (
                      <RedditPostCard key={post.post_id} post={post} entities={entityExtractions} />
                    ))
                  )}
                </div>
              )}
            </div>

            {/* Entity Extractions (Market Entities - in KB) */}
            <div className="bg-gray-900/50 rounded-2xl border border-gray-800/50 overflow-hidden">
              <button
                onClick={() => setShowEntities(!showEntities)}
                className="w-full flex items-center justify-between p-4 hover:bg-gray-800/30 transition-colors"
              >
                <div className="flex items-center gap-2">
                  <User className="w-4 h-4 text-violet-400" />
                  <span className="text-sm font-semibold text-gray-300">Entity Extractions</span>
                  <span className="px-2 py-0.5 bg-violet-900/30 text-violet-400 text-[10px] font-bold rounded-full">
                    {entityExtractions.length}
                  </span>
                  <span className="text-[9px] text-gray-500 italic">Market-linked</span>
                </div>
                {showEntities ? <ChevronDown className="w-4 h-4 text-gray-500" /> : <ChevronRight className="w-4 h-4 text-gray-500" />}
              </button>
              {showEntities && (
                <div className="px-4 pb-4 space-y-2 max-h-[300px] overflow-y-auto">
                  {entityExtractions.length === 0 ? (
                    <div className="text-center py-6 text-gray-500 text-sm">
                      No entities extracted yet...
                    </div>
                  ) : (
                    entityExtractions.slice(0, 10).map((extraction, idx) => (
                      <EntityExtractionCard key={`${extraction.post_id}_${extraction.entity_id}_${idx}`} extraction={extraction} />
                    ))
                  )}
                </div>
              )}
            </div>

            {/* Related Entities (PERSON, ORG, GPE, EVENT - NOT in KB) */}
            <div className="bg-gray-900/50 rounded-2xl border border-gray-800/50 overflow-hidden">
              <button
                onClick={() => setShowRelatedEntities(!showRelatedEntities)}
                className="w-full flex items-center justify-between p-4 hover:bg-gray-800/30 transition-colors"
              >
                <div className="flex items-center gap-2">
                  <User className="w-4 h-4 text-blue-400" />
                  <span className="text-sm font-semibold text-gray-300">Related Entities</span>
                  <span className="px-2 py-0.5 bg-blue-900/30 text-blue-400 text-[10px] font-bold rounded-full">
                    {relatedEntities.length}
                  </span>
                  <span className="text-[9px] text-gray-500 italic">Not in KB</span>
                </div>
                {showRelatedEntities ? <ChevronDown className="w-4 h-4 text-gray-500" /> : <ChevronRight className="w-4 h-4 text-gray-500" />}
              </button>
              {showRelatedEntities && (
                <div className="px-4 pb-4 space-y-2 max-h-[300px] overflow-y-auto">
                  {relatedEntities.length === 0 ? (
                    <div className="text-center py-6 text-gray-500 text-sm">
                      <div className="mb-2">No related entities detected yet...</div>
                      <div className="text-[10px] text-gray-600">
                        Shows PERSON, ORG, GPE, EVENT entities from spaCy NER
                        <br />that don't match any Kalshi market in the Knowledge Base
                      </div>
                    </div>
                  ) : (
                    relatedEntities.slice(0, 15).map((entity, idx) => (
                      <RelatedEntityCard key={`${entity.source_post_id}_${entity.normalized_id}_${idx}`} entity={entity} />
                    ))
                  )}
                </div>
              )}
            </div>
          </div>

          {/* Right Column - Agent Execution + Price Impacts */}
          <div className="col-span-8 space-y-4">
            {/* Agent Status Header - Sticky */}
            <div className="sticky top-0 z-10">
              <AgentStatusHeader
                agentState={agentState}
                thinking={thinking}
                settlements={settlements}
                trades={trades}
              />
            </div>

            {/* LLM Cost Panel */}
            <CostPanel costData={agentState.costData} />

            {/* Thinking Stream - Always visible */}
            <ThinkingStream thinking={thinking} isRunning={isAgentRunning} />

            {/* Tool Calls - Collapsible, default collapsed */}
            <ToolCallsPanel toolCalls={toolCalls} defaultCollapsed={true} />

            {/* Price Impacts - Main focus */}
            <div className="bg-gray-900/50 rounded-2xl border border-gray-800/50 p-6">
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-3">
                  <div className="p-2 rounded-lg bg-gradient-to-br from-orange-900/40 to-amber-900/30 border border-orange-700/30">
                    <Sparkles className="w-5 h-5 text-orange-400" />
                  </div>
                  <div>
                    <h2 className="text-lg font-semibold text-white">Price Impact Signals</h2>
                    <p className="text-xs text-gray-500">Entity sentiment → Market-specific trading signals</p>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  {(lifecycleSummary.evaluating > 0 || lifecycleSummary.traded > 0 || lifecycleSummary.historical > 0) && (
                    <span className="text-[10px] text-gray-500">
                      {[
                        lifecycleSummary.evaluating > 0 && `${lifecycleSummary.evaluating} evaluating`,
                        lifecycleSummary.traded > 0 && `${lifecycleSummary.traded} traded`,
                        lifecycleSummary.historical > 0 && `${lifecycleSummary.historical} historical`,
                      ].filter(Boolean).join(' \u00b7 ')}
                    </span>
                  )}
                  <span className="px-3 py-1 bg-orange-900/30 text-orange-400 text-sm font-bold rounded-lg border border-orange-700/30">
                    {entityPriceImpacts.length} signals
                  </span>
                </div>
              </div>

              <div className="space-y-4 max-h-[400px] overflow-y-auto pr-2">
                {sortedPriceImpacts.length === 0 ? (
                  <div className="text-center py-12">
                    <Sparkles className="w-10 h-10 text-gray-700 mx-auto mb-3" />
                    <div className="text-gray-500 text-sm font-medium mb-1">No price impacts yet</div>
                    <div className="text-gray-600 text-xs">
                      Signals appear when entities are detected and mapped to markets
                    </div>
                  </div>
                ) : (
                  sortedPriceImpacts.slice(0, 20).map((impact) => (
                    <PriceImpactCard
                      key={impact.signal_id}
                      impact={impact}
                      isNew={newSignalIds.has(impact.signal_id)}
                      freshnessTier={getFreshnessTier(impact)}
                      lifecycle={getSignalLifecycle(impact.signal_id)}
                    />
                  ))
                )}
              </div>
            </div>

            {/* Trades & Settlements */}
            <TradesPanel trades={trades} settlements={settlements} />

            {/* Learnings - Collapsed by default */}
            <LearningsPanel learnings={learnings} memoryUpdates={memoryUpdates} />

            {/* Entity Index - What the agent knows */}
            <div className="mt-6">
              <EntityIndexPanel
                entityIndex={entityIndex}
                entitySystemActive={entitySystemActive}
                redditAgentHealth={redditAgentHealth}
                showContentExtraction={true}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default memo(AgentPage);
