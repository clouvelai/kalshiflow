import React, { memo, useState } from 'react';
import { Search, ChevronDown, ChevronRight, Shield, CheckCircle, AlertCircle, MessageSquare, Heart, Users, BadgeCheck } from 'lucide-react';

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
 * ReliabilityBadge - Shows evidence reliability level
 */
const ReliabilityBadge = memo(({ reliability }) => {
  const config = {
    high: {
      icon: CheckCircle,
      bgClass: 'bg-emerald-500/10',
      borderClass: 'border-emerald-500/30',
      textClass: 'text-emerald-400',
      label: 'HIGH',
    },
    medium: {
      icon: Shield,
      bgClass: 'bg-amber-500/10',
      borderClass: 'border-amber-500/30',
      textClass: 'text-amber-400',
      label: 'MEDIUM',
    },
    low: {
      icon: AlertCircle,
      bgClass: 'bg-red-500/10',
      borderClass: 'border-red-500/30',
      textClass: 'text-red-400',
      label: 'LOW',
    },
  };

  const normalizedReliability = (reliability || 'medium').toLowerCase();
  const { icon: Icon, bgClass, borderClass, textClass, label } = config[normalizedReliability] || config.medium;

  return (
    <span className={`
      inline-flex items-center space-x-1.5 px-2.5 py-1 rounded-lg text-xs font-semibold
      ${bgClass} ${borderClass} ${textClass} border
    `}>
      <Icon className="w-3.5 h-3.5" />
      <span>{label} RELIABILITY</span>
    </span>
  );
});

ReliabilityBadge.displayName = 'ReliabilityBadge';

/**
 * TruthSocialSourceBadge - Shows Truth Social source with stats
 */
const TruthSocialSourceBadge = memo(({ truthSocialData }) => {
  if (!truthSocialData || truthSocialData.status === 'unavailable') {
    return null;
  }

  const { signals_emitted = 0, unique_authors = 0, verified_count = 0, status } = truthSocialData;

  // No matches found
  if (status === 'no_matches' || signals_emitted === 0) {
    return (
      <span className="inline-flex items-center space-x-1.5 px-2.5 py-1 rounded-lg text-xs font-medium bg-slate-700/30 text-slate-500 border border-slate-600/20">
        <MessageSquare className="w-3.5 h-3.5" />
        <span>No Truth Social signals found</span>
      </span>
    );
  }

  return (
    <span className="inline-flex items-center space-x-2 px-2.5 py-1 rounded-lg text-xs font-medium bg-pink-500/10 text-pink-400 border border-pink-500/20">
      <MessageSquare className="w-3.5 h-3.5" />
      <span>{signals_emitted} signals</span>
      <span className="text-pink-500/50">•</span>
      <Users className="w-3 h-3" />
      <span>{unique_authors} authors</span>
      {verified_count > 0 && (
        <>
          <span className="text-pink-500/50">•</span>
          <BadgeCheck className="w-3 h-3" />
          <span>{verified_count}</span>
        </>
      )}
    </span>
  );
});

TruthSocialSourceBadge.displayName = 'TruthSocialSourceBadge';

/**
 * TruthSocialSignalCard - Displays a single distilled Truth Social signal
 */
const TruthSocialSignalCard = memo(({ signal }) => {
  const {
    author_handle,
    is_verified,
    claim,
    claim_type,
    confidence,
    reasoning_short,
    engagement_score,
    entities,
    source_url,
  } = signal || {};

  const confidencePct = typeof confidence === 'number' ? Math.round(confidence * 100) : null;

  return (
    <div className="bg-slate-800/30 rounded-lg p-3 border border-slate-700/20">
      <div className="flex items-start justify-between gap-3 mb-2">
        <div className="flex items-center space-x-2 min-w-0">
          <span className="text-xs font-medium text-pink-400 truncate">
            @{author_handle || 'unknown'}
          </span>
          {is_verified && <BadgeCheck className="w-3.5 h-3.5 text-blue-400 flex-shrink-0" />}
        </div>
        <div className="flex items-center space-x-2 text-xs text-slate-500 flex-shrink-0">
          {typeof engagement_score === 'number' && (
            <span className="flex items-center space-x-1">
              <Heart className="w-3 h-3" />
              <span>{Math.round(engagement_score)}</span>
            </span>
          )}
          {confidencePct !== null && (
            <span className="px-2 py-0.5 rounded-md bg-slate-700/30 border border-slate-600/20 text-slate-400">
              {confidencePct}% conf
            </span>
          )}
          {claim_type && (
            <span className="px-2 py-0.5 rounded-md bg-pink-500/10 border border-pink-500/20 text-pink-400">
              {String(claim_type).toUpperCase()}
            </span>
          )}
        </div>
      </div>

      {claim && (
        <p className="text-xs text-slate-300 leading-relaxed">
          {claim}
        </p>
      )}

      {reasoning_short && (
        <p className="text-[11px] text-slate-500 mt-2 leading-relaxed">
          {reasoning_short}
        </p>
      )}

      {Array.isArray(entities) && entities.length > 0 && (
        <div className="flex flex-wrap gap-1 mt-2">
          {entities.slice(0, 6).map((e, idx) => (
            <span
              key={`${e}-${idx}`}
              className="text-[10px] px-2 py-0.5 rounded-md bg-slate-700/30 text-slate-400 border border-slate-600/20"
            >
              {e}
            </span>
          ))}
        </div>
      )}

      {source_url && (
        <a
          href={source_url}
          target="_blank"
          rel="noopener noreferrer"
          className="text-xs text-pink-400/70 hover:text-pink-400 mt-2 inline-block"
        >
          View on Truth Social →
        </a>
      )}
    </div>
  );
});

TruthSocialSignalCard.displayName = 'TruthSocialSignalCard';

/**
 * TruthSocialSignalsSection - Expandable list of distilled signals
 */
const TruthSocialSignalsSection = memo(({ topSignals }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  if (!topSignals || topSignals.length === 0) {
    return null;
  }

  return (
    <div className="border border-pink-500/20 rounded-lg overflow-hidden">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center justify-between px-4 py-3 bg-pink-500/5 hover:bg-pink-500/10 transition-colors"
      >
        <div className="flex items-center space-x-2">
          <MessageSquare className="w-4 h-4 text-pink-400" />
          <span className="text-xs uppercase tracking-wider text-pink-400 font-medium">
            Truth Social Signals
          </span>
          <span className="text-xs text-pink-500/50">({topSignals.length})</span>
        </div>
        {isExpanded ? (
          <ChevronDown className="w-4 h-4 text-pink-400" />
        ) : (
          <ChevronRight className="w-4 h-4 text-pink-400" />
        )}
      </button>

      {isExpanded && (
        <div className="px-4 py-3 bg-slate-900/30 space-y-2">
          {topSignals.slice(0, 5).map((signal, idx) => (
            <TruthSocialSignalCard key={signal.signal_id || idx} signal={signal} />
          ))}
        </div>
      )}
    </div>
  );
});

TruthSocialSignalsSection.displayName = 'TruthSocialSignalsSection';

/**
 * EvidenceSection - Displays gathered evidence and its reliability
 *
 * Shows:
 * - Reliability badge
 * - Truth Social source badge (signals, authors, verified)
 * - Evidence summary paragraph
 * - Key evidence bullet points
 * - Truth Social signals (expandable)
 */
const EvidenceSection = ({ evidenceSummary, reliability, keyEvidence, evidenceMetadata }) => {
  const [isEvidenceExpanded, setIsEvidenceExpanded] = useState(false);

  // Extract Truth Social data from metadata
  const truthSocialData = evidenceMetadata?.truth_social;
  const hasTruthSocialSignals = truthSocialData?.top_signals?.length > 0;

  const hasContent = evidenceSummary || (keyEvidence && keyEvidence.length > 0) || hasTruthSocialSignals;

  if (!hasContent) return null;

  return (
    <div className="bg-slate-900/40 rounded-xl p-4 border border-slate-700/30">
      <div className="flex items-center justify-between mb-3">
        <SectionHeader icon={Search} title="Evidence" />
        <div className="flex items-center space-x-2">
          <TruthSocialSourceBadge truthSocialData={truthSocialData} />
          <ReliabilityBadge reliability={reliability} />
        </div>
      </div>

      <div className="space-y-4">
        {/* Evidence Summary */}
        {evidenceSummary && (
          <div className="bg-slate-800/40 rounded-lg p-4 border border-slate-700/20">
            <p className="text-sm text-slate-300 leading-relaxed">
              {evidenceSummary}
            </p>
          </div>
        )}

        {/* Truth Social Signals (expandable) */}
        {hasTruthSocialSignals && (
          <TruthSocialSignalsSection topSignals={truthSocialData.top_signals} />
        )}

        {/* Key Evidence Points */}
        {keyEvidence && keyEvidence.length > 0 && (
          <div className="border border-slate-700/20 rounded-lg overflow-hidden">
            <button
              onClick={() => setIsEvidenceExpanded(!isEvidenceExpanded)}
              className="w-full flex items-center justify-between px-4 py-3 bg-slate-800/30 hover:bg-slate-800/50 transition-colors"
            >
              <div className="flex items-center space-x-2">
                <CheckCircle className="w-4 h-4 text-cyan-400" />
                <span className="text-xs uppercase tracking-wider text-slate-400 font-medium">
                  Key Evidence Points
                </span>
                <span className="text-xs text-slate-600">({keyEvidence.length})</span>
              </div>
              {isEvidenceExpanded ? (
                <ChevronDown className="w-4 h-4 text-slate-500" />
              ) : (
                <ChevronRight className="w-4 h-4 text-slate-500" />
              )}
            </button>

            {isEvidenceExpanded && (
              <div className="px-4 py-3 bg-slate-900/30 space-y-3">
                {keyEvidence.map((evidence, idx) => (
                  <div
                    key={idx}
                    className="flex items-start space-x-3"
                  >
                    <span className="flex-shrink-0 mt-1.5 w-1.5 h-1.5 rounded-full bg-cyan-500/50 border border-cyan-500/30" />
                    <p className="text-xs text-slate-400 leading-relaxed">
                      {evidence}
                    </p>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default memo(EvidenceSection);
