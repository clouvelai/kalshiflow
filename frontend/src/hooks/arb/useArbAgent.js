import { useState, useEffect, useRef, useCallback } from 'react';

const MAX_TOOL_CALLS = 30;

/**
 * useArbAgent - Categorizes the raw agent message stream into structured buckets.
 *
 * Returns:
 *   isRunning       - Whether an agent is actively running
 *   currentSubagent - Name of the currently active agent
 *   cycleCount      - Number of captain cycles observed
 *   thinking        - { text, agent, streaming } from thinking_delta/thinking_complete
 *   activeToolCall  - Currently executing tool { tool_name, tool_input, agent, timestamp }
 *   toolCalls       - Recent tool invocations (capped)
 */
export const useArbAgent = (agentMessages = []) => {
  const [isRunning, setIsRunning] = useState(false);
  const [currentSubagent, setCurrentSubagent] = useState(null);
  const [cycleCount, setCycleCount] = useState(0);
  const [thinking, setThinking] = useState({ text: '', agent: null, streaming: false });
  const [activeToolCall, setActiveToolCall] = useState(null);
  const [toolCalls, setToolCalls] = useState([]);

  const lastProcessedIdRef = useRef(null);
  const processedCountRef = useRef(0);

  const processMessage = useCallback((msg) => {
    switch (msg.subtype) {
      case 'thinking_delta':
        setThinking({ text: msg.text || '', agent: msg.agent, streaming: true });
        break;

      case 'thinking_complete':
        setThinking({ text: msg.text || '', agent: msg.agent, streaming: false });
        break;

      case 'tool_call':
        setActiveToolCall({
          tool_name: msg.tool_name,
          tool_input: msg.tool_input,
          agent: msg.agent,
          timestamp: msg.timestamp,
        });
        break;

      case 'tool_result':
        setActiveToolCall(null);
        setToolCalls(prev => [{
          id: msg.id,
          tool_name: msg.tool_name,
          tool_input: msg.tool_input,
          tool_output: msg.tool_output,
          agent: msg.agent,
          timestamp: msg.timestamp,
        }, ...prev].slice(0, MAX_TOOL_CALLS));
        break;

      case 'subagent_start':
        setIsRunning(true);
        setCurrentSubagent(msg.agent || null);
        // Captain starts increment cycle count
        if (msg.agent === 'captain') {
          setCycleCount(c => c + 1);
          // Reset thinking for new cycle
          setThinking({ text: '', agent: msg.agent, streaming: false });
        }
        break;

      case 'subagent_complete':
        // Only mark idle if captain completes (subagents complete mid-cycle)
        if (msg.agent === 'captain') {
          setIsRunning(false);
          setCurrentSubagent(null);
        }
        break;

      case 'subagent_error':
        if (msg.agent === 'captain') {
          setIsRunning(false);
          setCurrentSubagent(null);
        }
        break;

      default:
        break;
    }
  }, []);

  // Incremental processing: only process new messages
  useEffect(() => {
    if (agentMessages.length === 0) return;
    // agentMessages is newest-first, so find where we left off
    const startIdx = lastProcessedIdRef.current
      ? agentMessages.findIndex(m => m.id === lastProcessedIdRef.current)
      : agentMessages.length;

    // Process new messages (they're at indices 0..startIdx-1, newest first)
    // Process oldest-first for correct ordering
    const newCount = startIdx === -1 ? agentMessages.length : startIdx;
    if (newCount === 0) return;

    const newMessages = agentMessages.slice(0, newCount).reverse();
    for (const msg of newMessages) {
      processMessage(msg);
    }
    lastProcessedIdRef.current = agentMessages[0].id;
    processedCountRef.current = agentMessages.length;
  }, [agentMessages, processMessage]);

  return {
    isRunning,
    currentSubagent,
    cycleCount,
    thinking,
    activeToolCall,
    toolCalls,
  };
};

export default useArbAgent;
