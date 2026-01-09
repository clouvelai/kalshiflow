import React, { memo, useState } from 'react';
import { Brain, ChevronDown, ChevronRight, AlertTriangle, TrendingUp, Link2 } from 'lucide-react';

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
 * ExpandableList - Collapsible list of items
 */
const ExpandableList = memo(({ title, items, icon: Icon, accentColor = 'slate' }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  if (!items || items.length === 0) return null;

  const colorClasses = {
    amber: 'text-amber-400 bg-amber-500/10 border-amber-500/20',
    red: 'text-red-400 bg-red-500/10 border-red-500/20',
    cyan: 'text-cyan-400 bg-cyan-500/10 border-cyan-500/20',
    slate: 'text-slate-400 bg-slate-500/10 border-slate-500/20',
  };

  return (
    <div className="border border-slate-700/20 rounded-lg overflow-hidden">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center justify-between px-3 py-2 bg-slate-800/30 hover:bg-slate-800/50 transition-colors"
      >
        <div className="flex items-center space-x-2">
          <Icon className={`w-3.5 h-3.5 ${colorClasses[accentColor].split(' ')[0]}`} />
          <span className="text-[10px] uppercase tracking-wider text-slate-500 font-medium">
            {title}
          </span>
          <span className="text-[10px] text-slate-600">({items.length})</span>
        </div>
        {isExpanded ? (
          <ChevronDown className="w-3.5 h-3.5 text-slate-500" />
        ) : (
          <ChevronRight className="w-3.5 h-3.5 text-slate-500" />
        )}
      </button>

      {isExpanded && (
        <div className="px-3 py-2 bg-slate-900/30 space-y-1.5">
          {items.map((item, idx) => (
            <div
              key={idx}
              className={`flex items-start space-x-2 text-xs leading-relaxed`}
            >
              <span className={`mt-1 w-1.5 h-1.5 rounded-full flex-shrink-0 ${colorClasses[accentColor].split(' ')[1]} border ${colorClasses[accentColor].split(' ')[2]}`} />
              <span className="text-slate-400">{item}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
});

ExpandableList.displayName = 'ExpandableList';

/**
 * KeyDriverSection - Displays primary driver analysis
 *
 * Shows the single most important factor determining the outcome,
 * along with reasoning, base rate, secondary factors, and tail risks.
 */
const KeyDriverSection = ({
  primaryDriver,
  reasoning,
  baseRate,
  secondaryFactors,
  tailRisks,
  causalChain,
}) => {
  const hasContent = primaryDriver || reasoning;

  if (!hasContent) return null;

  // Format base rate as percentage
  const baseRatePercent = baseRate != null ? Math.round(baseRate * 100) : null;

  return (
    <div className="bg-slate-900/40 rounded-xl p-4 border border-slate-700/30">
      <SectionHeader icon={Brain} title="Key Driver Analysis" />

      <div className="space-y-4">
        {/* Primary Driver */}
        {primaryDriver && (
          <div className="bg-gradient-to-r from-emerald-500/10 to-slate-900/20 rounded-lg p-4 border border-emerald-500/20">
            <div className="flex items-center space-x-2 mb-2">
              <span className="text-[10px] uppercase tracking-wider text-emerald-400 font-bold">
                Primary Driver
              </span>
              {baseRatePercent != null && (
                <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-slate-700/50 text-slate-300 border border-slate-600/30">
                  {baseRatePercent}% base rate
                </span>
              )}
            </div>
            <p className="text-sm text-white font-medium leading-relaxed">
              {primaryDriver}
            </p>
          </div>
        )}

        {/* Reasoning */}
        {reasoning && (
          <div>
            <div className="flex items-center space-x-2 mb-2">
              <span className="text-[10px] uppercase tracking-wider text-slate-500 font-medium">
                Reasoning
              </span>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              {reasoning}
            </p>
          </div>
        )}

        {/* Causal Chain */}
        {causalChain && (
          <div className="bg-slate-800/40 rounded-lg p-3 border border-slate-700/20">
            <div className="flex items-center space-x-2 mb-2">
              <Link2 className="w-3.5 h-3.5 text-cyan-400" />
              <span className="text-[10px] uppercase tracking-wider text-slate-500 font-medium">
                Causal Chain
              </span>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed font-mono">
              {causalChain}
            </p>
          </div>
        )}

        {/* Expandable Lists */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <ExpandableList
            title="Secondary Factors"
            items={secondaryFactors}
            icon={TrendingUp}
            accentColor="amber"
          />
          <ExpandableList
            title="Tail Risks"
            items={tailRisks}
            icon={AlertTriangle}
            accentColor="red"
          />
        </div>
      </div>
    </div>
  );
};

export default memo(KeyDriverSection);
