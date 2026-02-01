import React, { memo, useState } from 'react';
import { Search, ChevronDown, ChevronRight, Shield, CheckCircle, AlertCircle } from 'lucide-react';

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
 * EvidenceSection - Displays gathered evidence and its reliability
 *
 * Shows:
 * - Reliability badge
 * - Evidence summary paragraph
 * - Key evidence bullet points
 */
const EvidenceSection = ({ evidenceSummary, reliability, keyEvidence }) => {
  const [isEvidenceExpanded, setIsEvidenceExpanded] = useState(false);

  const hasContent = evidenceSummary || (keyEvidence && keyEvidence.length > 0);

  if (!hasContent) return null;

  return (
    <div className="bg-slate-900/40 rounded-xl p-4 border border-slate-700/30">
      <div className="flex items-center justify-between mb-3">
        <SectionHeader icon={Search} title="Evidence" />
        <ReliabilityBadge reliability={reliability} />
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
