import React, { memo, useState } from 'react';
import { TrendingUp, ChevronDown, ChevronRight, Target, Activity, Check } from 'lucide-react';

/**
 * SectionHeader - Reusable section header component
 */
const SectionHeader = ({ icon: Icon, title }) => (
  <div className="flex items-center space-x-2 mb-3">
    <Icon className="w-4 h-4 text-slate-500" />
    <h4 className="text-xs uppercase tracking-wider text-slate-500 font-semibold">
      {title}
    </h4>
  </div>
);

/**
 * ProbabilityBar - Visual probability comparison bar
 */
const ProbabilityBar = memo(({ marketProb, evidenceProb }) => {
  const marketPercent = Math.round((marketProb || 0) * 100);
  const evidencePercent = Math.round((evidenceProb || 0) * 100);

  return (
    <div className="space-y-1">
      <div className="relative h-2 bg-slate-700 rounded-full overflow-hidden">
        {/* Market probability (gray background) */}
        <div
          className="absolute h-full bg-slate-500 rounded-full transition-all duration-300"
          style={{ width: `${marketPercent}%` }}
        />
        {/* Evidence probability (colored overlay) */}
        <div
          className={`
            absolute h-full rounded-full opacity-70 transition-all duration-300
            ${evidencePercent > marketPercent ? 'bg-emerald-400' : 'bg-red-400'}
          `}
          style={{ width: `${evidencePercent}%` }}
        />
      </div>
      <div className="flex justify-between text-[10px]">
        <span className="text-slate-500">Market: {marketPercent}%</span>
        <span className={evidencePercent > marketPercent ? 'text-emerald-400' : 'text-red-400'}>
          AI: {evidencePercent}%
        </span>
      </div>
    </div>
  );
});

ProbabilityBar.displayName = 'ProbabilityBar';

/**
 * RecommendationBadge - Shows BUY_YES, BUY_NO, or HOLD
 */
const RecommendationBadge = memo(({ recommendation }) => {
  const config = {
    BUY_YES: {
      bgClass: 'bg-emerald-500/5',
      borderClass: 'border-emerald-500/30',
      textClass: 'text-emerald-400',
      label: 'YES',
      icon: TrendingUp,
    },
    BUY_NO: {
      bgClass: 'bg-red-500/5',
      borderClass: 'border-red-500/30',
      textClass: 'text-red-400',
      label: 'NO',
      icon: TrendingUp,
      iconClass: 'rotate-180',
    },
    HOLD: {
      bgClass: 'bg-slate-500/5',
      borderClass: 'border-slate-500/30',
      textClass: 'text-slate-400',
      label: 'HOLD',
      icon: Activity,
    },
  };

  const { bgClass, borderClass, textClass, label, icon: Icon, iconClass } = config[recommendation] || config.HOLD;

  return (
    <span className={`
      inline-flex items-center space-x-1.5 px-2.5 py-1 rounded-lg text-xs font-bold
      ${bgClass} ${borderClass} ${textClass} border
    `}>
      <Icon className={`w-3.5 h-3.5 ${iconClass || ''}`} />
      <span>{label}</span>
    </span>
  );
});

RecommendationBadge.displayName = 'RecommendationBadge';

/**
 * ConfidenceBadge - Shows confidence level
 */
const ConfidenceBadge = memo(({ confidence }) => {
  const config = {
    high: {
      bgClass: 'bg-amber-500/10',
      textClass: 'text-amber-400',
      label: 'HIGH',
    },
    medium: {
      bgClass: 'bg-blue-500/10',
      textClass: 'text-blue-400',
      label: 'MED',
    },
    low: {
      bgClass: 'bg-slate-500/10',
      textClass: 'text-slate-500',
      label: 'LOW',
    },
  };

  const normalizedConf = (confidence || 'medium').toLowerCase();
  const { bgClass, textClass, label } = config[normalizedConf] || config.medium;

  return (
    <span className={`
      px-2 py-0.5 rounded text-[10px] font-semibold
      ${bgClass} ${textClass}
    `}>
      {label}
    </span>
  );
});

ConfidenceBadge.displayName = 'ConfidenceBadge';

/**
 * MarketAssessmentCard - Individual market assessment card
 */
const MarketAssessmentCard = memo(({ market }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  const {
    ticker,
    title,
    evidence_probability,
    market_probability,
    mispricing_magnitude,
    recommendation,
    confidence,
    edge_explanation,
  } = market;

  // Calculate edge percentage
  const edgePercent = Math.round((mispricing_magnitude || 0) * 100);
  const absEdge = Math.abs(edgePercent);
  const hasSignificantEdge = absEdge >= 5;

  // Check if market meets actionable trading criteria
  // Rule: Edge >= 5% + Confidence >= medium + Evidence >= medium = SHOULD trade
  const isActionable = (() => {
    const hasEdge = Math.abs(mispricing_magnitude || 0) >= 0.05;
    const hasConfidence = ['high', 'medium'].includes((confidence || '').toLowerCase());
    const hasEvidence = ['high', 'medium'].includes((market.evidence_quality || 'medium').toLowerCase());
    return hasEdge && hasConfidence && hasEvidence;
  })();

  // Calculate EV per contract in cents
  // EV = edge * potential_profit, where potential_profit = 100 - entry_price
  const evCents = (() => {
    const edge = mispricing_magnitude || 0;
    const entryPrice = recommendation === 'BUY_YES'
      ? (market_probability || 0) * 100
      : (1 - (market_probability || 0)) * 100;
    const potentialProfit = 100 - entryPrice;
    return edge * potentialProfit;
  })();

  // Determine card highlight based on recommendation
  const getCardStyle = () => {
    switch (recommendation) {
      case 'BUY_YES':
        return 'bg-emerald-500/5 border-emerald-500/20 hover:border-emerald-500/40';
      case 'BUY_NO':
        return 'bg-red-500/5 border-red-500/20 hover:border-red-500/40';
      default:
        return 'bg-slate-800/30 border-slate-700/30 hover:border-slate-600/40';
    }
  };

  return (
    <div className={`
      rounded-xl p-4 border transition-all duration-200
      ${getCardStyle()}
    `}>
      {/* Header Row */}
      <div className="flex items-start justify-between mb-3">
        <div className="flex-1 min-w-0">
          <h5 className="text-sm font-medium text-white truncate" title={title}>
            {title}
          </h5>
          <span className="text-[10px] font-mono text-slate-500">
            {ticker}
          </span>
        </div>
        <div className="flex items-center space-x-2 flex-shrink-0 ml-3">
          {isActionable && (
            <span className="inline-flex items-center space-x-1 px-1.5 py-0.5 rounded-full text-[9px] font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/40">
              <Check className="w-2.5 h-2.5" />
              <span>TRADE</span>
            </span>
          )}
          <RecommendationBadge recommendation={recommendation} />
          <ConfidenceBadge confidence={confidence} />
        </div>
      </div>

      {/* Probability Bar */}
      <ProbabilityBar
        marketProb={market_probability}
        evidenceProb={evidence_probability}
      />

      {/* Edge Display */}
      <div className="flex items-center justify-between mt-3">
        <span className="text-[10px] text-slate-500 uppercase tracking-wider">Edge</span>
        <div className="text-right">
          <span className={`
            text-sm font-mono font-bold
            ${edgePercent > 0 ? 'text-emerald-400' : edgePercent < 0 ? 'text-red-400' : 'text-slate-400'}
          `}>
            {edgePercent > 0 ? '+' : ''}{edgePercent}%
          </span>
          <div className={`
            text-[10px] font-mono
            ${evCents > 0.5 ? 'text-emerald-400' : evCents < -0.5 ? 'text-red-400' : 'text-slate-500'}
          `}>
            EV: {evCents > 0 ? '+' : ''}{Math.round(evCents)}c
          </div>
        </div>
      </div>

      {/* Expandable Edge Explanation */}
      {edge_explanation && (
        <div className="mt-3">
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="flex items-center space-x-1 text-[10px] text-slate-500 hover:text-slate-400 transition-colors"
          >
            {isExpanded ? (
              <ChevronDown className="w-3 h-3" />
            ) : (
              <ChevronRight className="w-3 h-3" />
            )}
            <span>Why?</span>
          </button>

          {isExpanded && (
            <div className="mt-2 p-3 bg-slate-900/50 rounded-lg border border-slate-700/20">
              <p className="text-xs text-slate-400 leading-relaxed">
                {edge_explanation}
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
});

MarketAssessmentCard.displayName = 'MarketAssessmentCard';

/**
 * MarketAssessmentsSection - Grid of market assessment cards
 *
 * Shows all markets in the event with:
 * - Probability comparison bars
 * - Recommendation badges
 * - Edge calculations
 * - Expandable explanations
 */
const MarketAssessmentsSection = ({ markets }) => {
  const [showAll, setShowAll] = useState(false);

  if (!markets || markets.length === 0) return null;

  // Sort markets: recommendations first (BUY_YES, BUY_NO), then by edge magnitude
  const sortedMarkets = [...markets].sort((a, b) => {
    const recOrder = { BUY_YES: 0, BUY_NO: 1, HOLD: 2 };
    const aOrder = recOrder[a.recommendation] ?? 3;
    const bOrder = recOrder[b.recommendation] ?? 3;
    if (aOrder !== bOrder) return aOrder - bOrder;
    return Math.abs(b.mispricing_magnitude || 0) - Math.abs(a.mispricing_magnitude || 0);
  });

  // Show first 6 markets by default, or all if showAll
  const displayedMarkets = showAll ? sortedMarkets : sortedMarkets.slice(0, 6);
  const hasMore = sortedMarkets.length > 6;

  // Count recommendations
  const buyYesCount = markets.filter(m => m.recommendation === 'BUY_YES').length;
  const buyNoCount = markets.filter(m => m.recommendation === 'BUY_NO').length;

  return (
    <div className="bg-slate-900/40 rounded-xl p-4 border border-slate-700/30">
      <div className="flex items-center justify-between mb-4">
        <SectionHeader icon={Target} title="Market Assessments" />
        <div className="flex items-center space-x-3">
          <span className="text-xs text-slate-500">
            {markets.length} markets
          </span>
          {buyYesCount > 0 && (
            <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
              {buyYesCount} YES
            </span>
          )}
          {buyNoCount > 0 && (
            <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-red-500/10 text-red-400 border border-red-500/20">
              {buyNoCount} NO
            </span>
          )}
        </div>
      </div>

      {/* Market Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
        {displayedMarkets.map((market, idx) => (
          <MarketAssessmentCard
            key={market.ticker || idx}
            market={market}
          />
        ))}
      </div>

      {/* Show More/Less Button */}
      {hasMore && (
        <div className="mt-4 text-center">
          <button
            onClick={() => setShowAll(!showAll)}
            className="text-sm text-emerald-400 hover:text-emerald-300 transition-colors"
          >
            {showAll ? 'Show less' : `Show all ${sortedMarkets.length} markets`}
          </button>
        </div>
      )}
    </div>
  );
};

export default memo(MarketAssessmentsSection);
