import React, { memo } from 'react';
import { Brain, Zap, Pause, AlertTriangle } from 'lucide-react';

const CaptainHeader = memo(({ isRunning, cycleCount, captainPaused, exchangeStatus }) => {
  const exchangeDown = exchangeStatus && exchangeStatus.active === false;
  const isPaused = captainPaused;

  const getBadgeStyle = () => {
    if (exchangeDown) return 'bg-red-500/15 text-red-300 border border-red-500/20';
    if (isPaused) return 'bg-amber-500/15 text-amber-300 border border-amber-500/20';
    if (isRunning) return 'bg-violet-500/15 text-violet-300 border border-violet-500/20';
    return 'bg-gray-800/40 text-gray-600 border border-gray-700/20';
  };

  const getBadgeText = () => {
    if (exchangeDown) return 'Exchange Down';
    if (isPaused) return 'Paused';
    if (isRunning) return 'Active';
    return 'Idle';
  };

  const getBadgeIcon = () => {
    if (exchangeDown) return <AlertTriangle className="w-2 h-2" />;
    if (isPaused) return <Pause className="w-2 h-2" />;
    if (isRunning) return <Zap className="w-2 h-2" />;
    return null;
  };

  return (
    <div className="flex items-center justify-between px-3 py-2.5 border-b border-gray-800/30 shrink-0">
      <div className="flex items-center gap-2">
        <div className={`p-1 rounded-md transition-colors duration-200 ${
          exchangeDown ? 'bg-red-500/15 shadow-sm shadow-red-500/20' :
          isRunning ? 'bg-violet-500/15 shadow-sm shadow-violet-500/20' : 'bg-gray-800/40'
        }`}>
          <Brain className={`w-3.5 h-3.5 transition-colors duration-200 ${
            exchangeDown ? 'text-red-400' :
            isRunning ? 'text-violet-400' : 'text-gray-500'
          }`} />
        </div>
        <div className="flex items-center gap-1.5">
          <span className="text-[12px] font-semibold text-gray-200">Captain</span>
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
          className={`flex items-center gap-1 px-1.5 py-0.5 rounded-full text-[8px] font-semibold uppercase tracking-wider transition-colors duration-200 ${getBadgeStyle()}`}
        >
          {getBadgeIcon()}
          {getBadgeText()}
        </span>
      </div>
    </div>
  );
});

CaptainHeader.displayName = 'CaptainHeader';

export default CaptainHeader;
