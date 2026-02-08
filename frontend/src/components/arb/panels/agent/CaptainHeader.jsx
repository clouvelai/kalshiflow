import React, { memo } from 'react';
import { Brain, Zap } from 'lucide-react';

const CaptainHeader = memo(({ isRunning, cycleCount, currentSubagent }) => {
  const subagentLabel =
    currentSubagent && currentSubagent !== 'single_arb_captain'
      ? currentSubagent.replace(/_/g, ' ')
      : null;

  return (
    <div className="flex items-center justify-between px-3 py-2.5 border-b border-gray-800/30 shrink-0">
      <div className="flex items-center gap-2">
        <div className={`p-1 rounded-md transition-colors duration-200 ${
          isRunning ? 'bg-violet-500/15 shadow-sm shadow-violet-500/20' : 'bg-gray-800/40'
        }`}>
          <Brain className={`w-3.5 h-3.5 transition-colors duration-200 ${
            isRunning ? 'text-violet-400' : 'text-gray-500'
          }`} />
        </div>
        <div className="flex items-center gap-1.5">
          <span className="text-[12px] font-semibold text-gray-200">Captain</span>
          {subagentLabel && (
            <span className="text-[9px] text-violet-400/70 font-mono truncate max-w-[120px]">
              {subagentLabel}
            </span>
          )}
        </div>
      </div>
      <div className="flex items-center gap-2">
        {cycleCount > 0 && (
          <span data-testid="agent-cycle-count" className="text-[9px] text-gray-600 font-mono tabular-nums">
            C{cycleCount}
          </span>
        )}
        <span
          data-testid="agent-running-badge"
          data-running={isRunning}
          className={`flex items-center gap-1 px-1.5 py-0.5 rounded-full text-[8px] font-semibold uppercase tracking-wider transition-colors duration-200 ${
            isRunning
              ? 'bg-violet-500/15 text-violet-300 border border-violet-500/20'
              : 'bg-gray-800/40 text-gray-600 border border-gray-700/20'
          }`}
        >
          {isRunning && <Zap className="w-2 h-2" />}
          {isRunning ? 'Active' : 'Idle'}
        </span>
      </div>
    </div>
  );
});

CaptainHeader.displayName = 'CaptainHeader';

export default CaptainHeader;
