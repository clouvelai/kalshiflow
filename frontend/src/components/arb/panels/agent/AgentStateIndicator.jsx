import React, { memo, useMemo } from 'react';
import { motion } from 'framer-motion';
import { AGENT_STATE_COLORS } from '../../utils/styleConstants';

const PHASES = ['idle', 'thinking', 'acting', 'complete'];

const AgentStateIndicator = memo(({ isRunning, activeToolCall, thinking }) => {
  const currentPhase = useMemo(() => {
    if (!isRunning) return 'idle';
    if (activeToolCall) return 'acting';
    if (thinking?.streaming || thinking?.text) return 'thinking';
    return 'thinking';
  }, [isRunning, activeToolCall, thinking]);

  const activeIdx = PHASES.indexOf(currentPhase);

  return (
    <div className="flex items-center gap-2 px-3 py-1.5 border-b border-gray-800/20">
      <div className="flex items-center gap-1.5 flex-1">
        {PHASES.map((phase, i) => {
          const isActive = i === activeIdx;
          const isPast = i < activeIdx;
          const colors = AGENT_STATE_COLORS[phase];

          return (
            <div key={phase} className="flex items-center gap-1.5 flex-1">
              <div className="relative flex items-center justify-center w-4 h-4">
                {isActive && (
                  <motion.div
                    layoutId="agent-state-highlight"
                    className={`absolute inset-0 rounded-full ${colors.bg} opacity-30`}
                    transition={{ type: 'spring', stiffness: 500, damping: 30 }}
                  />
                )}
                <div className={`w-2 h-2 rounded-full transition-colors duration-200 ${
                  isActive ? colors.bg
                    : isPast ? 'bg-gray-600'
                    : 'bg-gray-800'
                }`} />
              </div>
              <span className={`text-[8px] uppercase tracking-wider font-semibold transition-colors duration-200 ${
                isActive ? colors.text : 'text-gray-700'
              }`}>
                {colors.label}
              </span>
              {i < PHASES.length - 1 && (
                <div className={`flex-1 h-px ${isPast ? 'bg-gray-600' : 'bg-gray-800/50'}`} />
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
});

AgentStateIndicator.displayName = 'AgentStateIndicator';

export default AgentStateIndicator;
