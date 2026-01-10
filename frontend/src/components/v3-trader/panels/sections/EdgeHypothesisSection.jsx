import React, { memo } from 'react';
import { Lightbulb } from 'lucide-react';

/**
 * SectionHeader - Reusable section header component
 */
const SectionHeader = ({ icon: Icon, title }) => (
  <div className="flex items-center space-x-2 mb-3">
    <Icon className="w-4 h-4 text-amber-400" />
    <h4 className="text-xs uppercase tracking-wider text-slate-500 font-semibold">
      {title}
    </h4>
  </div>
);

/**
 * EdgeHypothesisSection - Displays where the market might be wrong
 *
 * Amber/gold theme to highlight profit-focused thinking.
 * Shows the AI's hypothesis about information asymmetry.
 */
const EdgeHypothesisSection = ({ edgeHypothesis }) => {
  if (!edgeHypothesis) return null;

  return (
    <div className="bg-amber-500/10 rounded-xl p-4 border border-amber-500/20">
      <SectionHeader icon={Lightbulb} title="Edge Hypothesis" />

      <div className="bg-slate-900/40 rounded-lg p-4 border border-amber-500/10">
        <p className="text-sm text-amber-100 leading-relaxed">
          {edgeHypothesis}
        </p>
      </div>
    </div>
  );
};

export default memo(EdgeHypothesisSection);
