import React, { memo, useMemo } from 'react';
import { Eye, Zap, BarChart3, Scan, Brain, Terminal } from 'lucide-react';
import { AGENT_STATE_COLORS } from '../../utils/styleConstants';

const AgentStateIndicator = memo(({ isRunning, activeToolCall, thinking, cycleMode, attentionCount }) => {
  const state = useMemo(() => {
    if (!isRunning) return 'observing';
    if (cycleMode === 'reactive') return 'reactive';
    if (cycleMode === 'strategic') return 'strategic';
    if (cycleMode === 'deep_scan') return 'deep_scan';
    if (activeToolCall) return 'acting';
    if (thinking?.streaming || thinking?.text) return 'thinking';
    return 'thinking';
  }, [isRunning, activeToolCall, thinking, cycleMode]);

  const colors = AGENT_STATE_COLORS[state] || AGENT_STATE_COLORS.observing;

  const getIcon = () => {
    switch (state) {
      case 'observing': return <Eye className="w-3 h-3" />;
      case 'reactive': return <Zap className="w-3 h-3" />;
      case 'strategic': return <BarChart3 className="w-3 h-3" />;
      case 'deep_scan': return <Scan className="w-3 h-3" />;
      case 'thinking': return <Brain className="w-3 h-3" />;
      case 'acting': return <Terminal className="w-3 h-3" />;
      default: return <Eye className="w-3 h-3" />;
    }
  };

  const getDetail = () => {
    if (state === 'observing' && attentionCount > 0) {
      return `${attentionCount} signal${attentionCount !== 1 ? 's' : ''} pending`;
    }
    if (state === 'reactive') return `Responding to attention signals`;
    if (state === 'strategic') return 'Portfolio review';
    if (state === 'deep_scan') return 'Comprehensive scan';
    if (state === 'acting' && activeToolCall) return activeToolCall.tool_name + '()';
    if (state === 'thinking') return 'Reasoning...';
    return '';
  };

  return (
    <div className="flex items-center gap-2 px-3 py-1.5 border-b border-gray-800/20">
      <div className="relative flex items-center justify-center w-4 h-4">
        <div className={`absolute inset-0 rounded-full ${colors.bg} opacity-20 ${isRunning ? 'animate-pulse' : ''}`} />
        <div className={`w-2 h-2 rounded-full ${colors.bg}`} />
      </div>
      <span className={`${colors.text} shrink-0`}>
        {getIcon()}
      </span>
      <span className={`text-[9px] uppercase tracking-wider font-semibold ${colors.text}`}>
        {colors.label}
      </span>
      {getDetail() && (
        <>
          <span className="text-gray-700 text-[9px]">&mdash;</span>
          <span className="text-[9px] text-gray-500 font-mono">{getDetail()}</span>
        </>
      )}
    </div>
  );
});

AgentStateIndicator.displayName = 'AgentStateIndicator';

export default AgentStateIndicator;
