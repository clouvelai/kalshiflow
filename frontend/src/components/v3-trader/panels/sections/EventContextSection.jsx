import React, { memo } from 'react';
import { FileText, Clock, Target } from 'lucide-react';

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
 * EventContextSection - Displays event description, resolution criteria, and time horizon
 *
 * Provides the foundational understanding of what the event is about
 * and how it will be resolved.
 */
const EventContextSection = ({ description, resolutionCriteria, timeHorizon }) => {
  const hasContent = description || resolutionCriteria || timeHorizon;

  if (!hasContent) return null;

  return (
    <div className="bg-slate-900/40 rounded-xl p-4 border border-slate-700/30">
      <SectionHeader icon={FileText} title="Event Context" />

      <div className="space-y-4">
        {/* Description */}
        {description && (
          <div>
            <p className="text-sm text-slate-300 leading-relaxed">
              {description}
            </p>
          </div>
        )}

        {/* Resolution and Time Horizon */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Resolution Criteria */}
          {resolutionCriteria && (
            <div className="bg-slate-800/40 rounded-lg p-3 border border-slate-700/20">
              <div className="flex items-center space-x-2 mb-2">
                <Target className="w-3.5 h-3.5 text-cyan-400" />
                <span className="text-[10px] uppercase tracking-wider text-slate-500 font-medium">
                  Resolution Criteria
                </span>
              </div>
              <p className="text-xs text-slate-400 leading-relaxed">
                {resolutionCriteria}
              </p>
            </div>
          )}

          {/* Time Horizon */}
          {timeHorizon && (
            <div className="bg-slate-800/40 rounded-lg p-3 border border-slate-700/20">
              <div className="flex items-center space-x-2 mb-2">
                <Clock className="w-3.5 h-3.5 text-amber-400" />
                <span className="text-[10px] uppercase tracking-wider text-slate-500 font-medium">
                  Time Horizon
                </span>
              </div>
              <p className="text-xs text-slate-400 leading-relaxed">
                {timeHorizon}
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default memo(EventContextSection);
