import { useState, useCallback, useEffect } from 'react';

export function useMMAgent(agentMessages) {
  const [isRunning, setIsRunning] = useState(false);
  const [cycleCount, setCycleCount] = useState(0);
  const [cycleMode, setCycleMode] = useState(null);
  const [thinking, setThinking] = useState(null);
  const [toolCalls, setToolCalls] = useState([]);

  useEffect(() => {
    if (!agentMessages || agentMessages.length === 0) return;

    const latest = agentMessages[0];
    if (!latest) return;

    if (latest.agent === 'admiral') {
      setIsRunning(true);
      if (latest.cycle) setCycleCount(latest.cycle);
      if (latest.mode) setCycleMode(latest.mode);
    }

    // Process subtypes
    const subtype = latest.subtype || latest.type;

    switch (subtype) {
      case 'thinking_delta':
      case 'thinking_complete':
        setThinking({
          text: latest.text || latest.content || '',
          streaming: subtype === 'thinking_delta',
        });
        break;

      case 'tool_call':
        setToolCalls(prev => [{
          tool: latest.tool || latest.name || '',
          input: latest.input || {},
          timestamp: latest.timestamp,
          status: 'running',
        }, ...prev].slice(0, 30));
        break;

      case 'tool_result':
        setToolCalls(prev => {
          const updated = [...prev];
          const idx = updated.findIndex(t => t.tool === (latest.tool || latest.name));
          if (idx >= 0) {
            updated[idx] = { ...updated[idx], status: 'complete', result: latest.result };
          }
          return updated;
        });
        break;

      case 'cycle_complete':
        setIsRunning(false);
        setThinking(null);
        break;

      default:
        break;
    }
  }, [agentMessages]);

  return {
    isRunning,
    cycleCount,
    cycleMode,
    thinking,
    toolCalls,
  };
}
