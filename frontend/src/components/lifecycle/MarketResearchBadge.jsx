import React, { memo, useState } from 'react';

/**
 * MarketResearchBadge - Display AI research assessment for a market
 *
 * Shows:
 * - AI probability estimate vs market probability
 * - Edge magnitude and direction
 * - Recommendation (BUY_YES, BUY_NO, HOLD)
 * - Confidence level with color coding
 *
 * Expands on click to show more details like edge explanation.
 *
 * Props:
 *   - research: Research assessment object from eventResearch._marketIndex
 *     {
 *       ticker, title, evidence_probability, market_probability,
 *       mispricing_magnitude, recommendation, confidence, edge_explanation,
 *       eventTitle, primaryDriver, evidenceSummary, researchedAt
 *     }
 */
const MarketResearchBadge = ({ research }) => {
  const [expanded, setExpanded] = useState(false);

  if (!research) return null;

  const {
    evidence_probability,
    market_probability,
    mispricing_magnitude,
    recommendation,
    confidence,
    edge_explanation,
    primaryDriver,
  } = research;

  // Calculate edge percentage (evidence - market)
  const edgePercent = Math.round(mispricing_magnitude * 100);
  const absEdge = Math.abs(edgePercent);

  // Determine recommendation styling
  const getRecommendationStyle = () => {
    switch (recommendation) {
      case 'BUY_YES':
        return {
          bgClass: 'bg-emerald-500/20 border-emerald-500/40',
          textClass: 'text-emerald-400',
          icon: '\u2191', // Up arrow
          label: 'YES',
        };
      case 'BUY_NO':
        return {
          bgClass: 'bg-red-500/20 border-red-500/40',
          textClass: 'text-red-400',
          icon: '\u2193', // Down arrow
          label: 'NO',
        };
      default:
        return {
          bgClass: 'bg-gray-500/20 border-gray-500/40',
          textClass: 'text-gray-400',
          icon: '\u2194', // Bidirectional arrow
          label: 'HOLD',
        };
    }
  };

  // Determine confidence styling
  const getConfidenceStyle = () => {
    switch (confidence) {
      case 'high':
        return {
          bgClass: 'bg-amber-500/15',
          textClass: 'text-amber-400',
          label: 'HIGH',
        };
      case 'medium':
        return {
          bgClass: 'bg-blue-500/15',
          textClass: 'text-blue-400',
          label: 'MED',
        };
      default:
        return {
          bgClass: 'bg-gray-500/15',
          textClass: 'text-gray-500',
          label: 'LOW',
        };
    }
  };

  const recStyle = getRecommendationStyle();
  const confStyle = getConfidenceStyle();

  // Only show badge if there's a meaningful recommendation
  if (recommendation === 'HOLD' && absEdge < 5) {
    return null;
  }

  return (
    <div
      onClick={(e) => {
        e.stopPropagation();
        setExpanded(!expanded);
      }}
      className="mt-2 cursor-pointer"
    >
      {/* Main badge row */}
      <div className={`
        flex items-center justify-between gap-2 px-2.5 py-1.5 rounded-lg
        border backdrop-blur-sm
        transition-all duration-200
        hover:scale-[1.01]
        ${recStyle.bgClass}
      `}>
        {/* Left: AI icon + probability */}
        <div className="flex items-center gap-2">
          <span className="text-purple-400 text-xs font-medium flex items-center gap-1">
            <svg className="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 20 20">
              <path d="M10 2a1 1 0 011 1v1.323l3.954 1.582 1.599-.8a1 1 0 01.894 1.79l-1.233.617 1.738 5.42a1 1 0 01-.285 1.05A3.989 3.989 0 0115 15a3.989 3.989 0 01-2.667-1.018 1 1 0 01-.285-1.05l1.715-5.349L11 6.477V16h2a1 1 0 110 2H7a1 1 0 110-2h2V6.477L6.237 7.582l1.715 5.349a1 1 0 01-.285 1.05A3.989 3.989 0 015 15a3.989 3.989 0 01-2.667-1.018 1 1 0 01-.285-1.05l1.738-5.42-1.233-.617a1 1 0 01.894-1.79l1.599.8L9 4.323V3a1 1 0 011-1z" />
            </svg>
            AI
          </span>
          <span className="font-mono text-white text-sm font-medium">
            {Math.round(evidence_probability * 100)}%
          </span>
          <span className="text-gray-500 text-xs">vs</span>
          <span className="font-mono text-gray-400 text-sm">
            {Math.round(market_probability * 100)}%
          </span>
        </div>

        {/* Right: Edge + Recommendation */}
        <div className="flex items-center gap-2">
          {/* Edge badge */}
          <span className={`
            font-mono text-xs font-bold px-1.5 py-0.5 rounded
            ${edgePercent > 0 ? 'text-emerald-400 bg-emerald-500/15' : 'text-red-400 bg-red-500/15'}
          `}>
            {edgePercent > 0 ? '+' : ''}{edgePercent}%
          </span>

          {/* Recommendation badge */}
          <span className={`
            flex items-center gap-1 px-1.5 py-0.5 rounded text-xs font-bold
            ${recStyle.bgClass} ${recStyle.textClass}
          `}>
            <span>{recStyle.icon}</span>
            <span>{recStyle.label}</span>
          </span>

          {/* Confidence indicator */}
          <span className={`
            text-[10px] font-medium px-1.5 py-0.5 rounded
            ${confStyle.bgClass} ${confStyle.textClass}
          `}>
            {confStyle.label}
          </span>
        </div>
      </div>

      {/* Expanded details */}
      {expanded && (
        <div className={`
          mt-1.5 px-2.5 py-2 rounded-lg text-xs
          bg-gray-800/60 border border-gray-700/40
          animate-expand-in
        `}>
          {/* Primary driver */}
          {primaryDriver && (
            <div className="mb-2">
              <span className="text-gray-500 font-medium">Key Driver: </span>
              <span className="text-gray-300">{primaryDriver}</span>
            </div>
          )}

          {/* Edge explanation */}
          {edge_explanation && (
            <div className="text-gray-400 leading-relaxed">
              {edge_explanation}
            </div>
          )}

          {/* Research timestamp */}
          {research.researchedAt && (
            <div className="mt-2 text-gray-600 text-[10px]">
              Researched {formatTimeAgo(research.researchedAt)}
            </div>
          )}
        </div>
      )}

      {/* Inline animation styles */}
      <style>{`
        @keyframes expand-in {
          0% { opacity: 0; transform: translateY(-4px); }
          100% { opacity: 1; transform: translateY(0); }
        }
        .animate-expand-in {
          animation: expand-in 150ms ease-out;
        }
      `}</style>
    </div>
  );
};

/**
 * Format timestamp as relative time (e.g., "2m ago", "1h ago")
 */
function formatTimeAgo(timestamp) {
  const seconds = Math.floor(Date.now() / 1000 - timestamp);
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

export default memo(MarketResearchBadge);
