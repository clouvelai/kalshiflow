import React, { memo, useState } from 'react';
import {
  ChevronDown,
  ChevronRight,
  Clock,
  Target,
  Layers,
  Users,
  Briefcase,
  Trophy,
  Sparkles,
  AlertTriangle,
} from 'lucide-react';
import EventContextSection from './sections/EventContextSection';
import KeyDriverSection from './sections/KeyDriverSection';
import SemanticFrameSection from './sections/SemanticFrameSection';
import EvidenceSection from './sections/EvidenceSection';
import MarketAssessmentsSection from './sections/MarketAssessmentsSection';

/**
 * Frame type color mapping
 */
const FRAME_TYPE_COLORS = {
  nomination: {
    bg: 'bg-emerald-500/10',
    border: 'border-emerald-500/30',
    text: 'text-emerald-400',
    icon: Users,
  },
  competition: {
    bg: 'bg-violet-500/10',
    border: 'border-violet-500/30',
    text: 'text-violet-400',
    icon: Trophy,
  },
  achievement: {
    bg: 'bg-amber-500/10',
    border: 'border-amber-500/30',
    text: 'text-amber-400',
    icon: Target,
  },
  occurrence: {
    bg: 'bg-cyan-500/10',
    border: 'border-cyan-500/30',
    text: 'text-cyan-400',
    icon: Sparkles,
  },
  measurement: {
    bg: 'bg-blue-500/10',
    border: 'border-blue-500/30',
    text: 'text-blue-400',
    icon: Layers,
  },
  mention: {
    bg: 'bg-pink-500/10',
    border: 'border-pink-500/30',
    text: 'text-pink-400',
    icon: Briefcase,
  },
  unknown: {
    bg: 'bg-gray-500/10',
    border: 'border-gray-500/30',
    text: 'text-gray-400',
    icon: AlertTriangle,
  },
};

const getFrameTypeConfig = (frameType) => {
  const type = (frameType || 'unknown').toLowerCase();
  return FRAME_TYPE_COLORS[type] || FRAME_TYPE_COLORS.unknown;
};

/**
 * Format timestamp as relative time
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
 * FrameTypeBadge - Shows semantic frame type with icon and color
 */
const FrameTypeBadge = memo(({ frameType }) => {
  const config = getFrameTypeConfig(frameType);
  const Icon = config.icon;

  return (
    <span className={`
      inline-flex items-center space-x-1.5 px-2.5 py-1 rounded-lg text-xs font-medium
      ${config.bg} ${config.border} ${config.text} border
    `}>
      <Icon className="w-3.5 h-3.5" />
      <span className="uppercase tracking-wider">{frameType || 'Unknown'}</span>
    </span>
  );
});

FrameTypeBadge.displayName = 'FrameTypeBadge';

/**
 * CategoryBadge - Shows event category
 */
const CategoryBadge = memo(({ category }) => {
  if (!category) return null;

  return (
    <span className="px-2.5 py-1 rounded-lg text-xs font-medium bg-slate-700/50 text-slate-300 border border-slate-600/30">
      {category}
    </span>
  );
});

CategoryBadge.displayName = 'CategoryBadge';

/**
 * EdgeSummaryBadge - Quick summary of markets with edge
 */
const EdgeSummaryBadge = memo(({ marketsWithEdge, marketsEvaluated }) => {
  const hasEdge = marketsWithEdge > 0;

  return (
    <span className={`
      px-2.5 py-1 rounded-lg text-xs font-mono font-semibold
      ${hasEdge
        ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
        : 'bg-slate-700/30 text-slate-500 border border-slate-600/20'
      }
    `}>
      {marketsWithEdge}/{marketsEvaluated} edge
    </span>
  );
});

EdgeSummaryBadge.displayName = 'EdgeSummaryBadge';

/**
 * EventCard - Collapsible card displaying event research
 *
 * Header (always visible):
 * - Event title
 * - Category and frame type badges
 * - Time ago
 * - Markets count
 *
 * Body (expandable):
 * - Event context section
 * - Key driver section
 * - Semantic frame section
 * - Evidence section
 * - Market assessments section
 */
const EventCard = ({ event, isNew = false }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  const {
    event_ticker,
    event_title,
    event_category,
    event_description,
    primary_driver,
    primary_driver_reasoning,
    base_rate,
    evidence_summary,
    evidence_reliability,
    semantic_frame,
    resolution_criteria,
    time_horizon,
    secondary_factors,
    tail_risks,
    causal_chain,
    key_evidence,
    markets,
    researched_at,
    research_duration_seconds,
    markets_evaluated,
    markets_with_edge,
  } = event;

  const frameType = semantic_frame?.frame_type;

  return (
    <div className={`
      bg-slate-800/50 backdrop-blur-sm border rounded-xl overflow-hidden
      transition-all duration-200 ease-out
      ${isNew
        ? 'border-emerald-500/50 shadow-lg shadow-emerald-500/10 animate-pulse-once'
        : 'border-slate-700/50 hover:border-slate-600/50'
      }
    `}>
      {/* Header - Always Visible */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full px-5 py-4 flex items-center justify-between hover:bg-slate-700/20 transition-colors"
      >
        <div className="flex items-center space-x-4 flex-1 min-w-0">
          {/* Expand/Collapse Icon */}
          <div className="flex-shrink-0 text-slate-500">
            {isExpanded ? (
              <ChevronDown className="w-5 h-5" />
            ) : (
              <ChevronRight className="w-5 h-5" />
            )}
          </div>

          {/* Title and Ticker */}
          <div className="flex-1 min-w-0 text-left">
            <h3 className="text-base font-semibold text-white truncate">
              {event_title || event_ticker}
            </h3>
            <div className="flex items-center space-x-2 mt-1">
              <span className="text-xs font-mono text-slate-500">
                {event_ticker}
              </span>
            </div>
          </div>
        </div>

        {/* Right side: Badges and metadata */}
        <div className="flex items-center space-x-3 flex-shrink-0">
          {/* Category Badge */}
          <CategoryBadge category={event_category} />

          {/* Frame Type Badge */}
          {frameType && (
            <FrameTypeBadge frameType={frameType} />
          )}

          {/* Edge Summary */}
          <EdgeSummaryBadge
            marketsWithEdge={markets_with_edge || 0}
            marketsEvaluated={markets_evaluated || 0}
          />

          {/* Time ago */}
          <div className="flex items-center space-x-1.5 text-xs text-slate-500">
            <Clock className="w-3.5 h-3.5" />
            <span>{formatTimeAgo(researched_at)}</span>
          </div>
        </div>
      </button>

      {/* Body - Expandable */}
      {isExpanded && (
        <div className="px-5 pb-5 border-t border-slate-700/30">
          {/* Grid layout for sections */}
          <div className="pt-4 space-y-5">
            {/* Top Row: Context + Key Driver */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <EventContextSection
                description={event_description}
                resolutionCriteria={resolution_criteria}
                timeHorizon={time_horizon}
              />
              <KeyDriverSection
                primaryDriver={primary_driver}
                reasoning={primary_driver_reasoning}
                baseRate={base_rate}
                secondaryFactors={secondary_factors}
                tailRisks={tail_risks}
                causalChain={causal_chain}
              />
            </div>

            {/* Semantic Frame (if available) */}
            {semantic_frame && (
              <SemanticFrameSection
                semanticFrame={semantic_frame}
              />
            )}

            {/* Evidence Section */}
            <EvidenceSection
              evidenceSummary={evidence_summary}
              reliability={evidence_reliability}
              keyEvidence={key_evidence}
            />

            {/* Market Assessments */}
            <MarketAssessmentsSection
              markets={markets || []}
            />

            {/* Footer: Research metadata */}
            <div className="flex items-center justify-end space-x-4 pt-3 border-t border-slate-700/30 text-xs text-slate-500">
              <span>
                Research duration: {(research_duration_seconds || 0).toFixed(1)}s
              </span>
              <span>
                {markets_evaluated || 0} markets evaluated
              </span>
            </div>
          </div>
        </div>
      )}

      {/* CSS for pulse animation on new events */}
      <style>{`
        @keyframes pulse-once {
          0%, 100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.4); }
          50% { box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
        }
        .animate-pulse-once {
          animation: pulse-once 2s ease-out;
        }
      `}</style>
    </div>
  );
};

export default memo(EventCard);
