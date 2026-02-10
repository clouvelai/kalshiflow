import { useState, useEffect, useRef, useCallback } from 'react';

const MAX_TOOL_CALLS = 30;
const MAX_MEMORY_OPS = 20;
const STORAGE_KEY = 'arb_agent_state';

/* ─── Session Storage Persistence ─── */
const saveSnapshot = (state) => {
  try {
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  } catch (e) {
    /* ignore quota errors */
  }
};

const loadSnapshot = () => {
  try {
    const saved = sessionStorage.getItem(STORAGE_KEY);
    return saved ? JSON.parse(saved) : null;
  } catch (e) {
    return null;
  }
};

/**
 * useArbAgent - Categorizes the raw agent message stream into structured buckets.
 *
 * Returns:
 *   isRunning        - Whether the Captain agent is actively running
 *   cycleCount       - Number of captain cycles observed
 *   thinking         - { text, agent, streaming } from thinking_delta/thinking_complete
 *   activeToolCall   - Currently executing tool { tool_name, tool_input, agent, timestamp }
 *   toolCalls        - Recent tool invocations (capped, arb category only)
 *   todos            - Current TODO list [{text, status}]
 *   memoryOps        - Recent memory operations [{id, type, tool_name, ...}]
 *
 * Persistence: State is persisted to sessionStorage and restored on mount.
 */
export const useArbAgent = (agentMessages = []) => {
  // Load snapshot once on mount
  const snapshotRef = useRef(loadSnapshot());
  const snapshot = snapshotRef.current;

  const [isRunning, setIsRunning] = useState(false);
  const [cycleCount, setCycleCount] = useState(snapshot?.cycleCount ?? 0);
  const [thinking, setThinking] = useState(
    snapshot?.thinking ?? { text: '', agent: null, streaming: false }
  );
  const [activeToolCall, setActiveToolCall] = useState(null);
  const [toolCalls, setToolCalls] = useState(snapshot?.toolCalls ?? []);
  const [todos, setTodos] = useState(snapshot?.todos ?? []);
  const [memoryOps, setMemoryOps] = useState(snapshot?.memoryOps ?? []);

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

      case 'todo_update':
        if (msg.todos) {
          setTodos(msg.todos);
        }
        break;

      case 'tool_call':
        setActiveToolCall({
          tool_name: msg.tool_name,
          tool_input: msg.tool_input,
          agent: msg.agent,
          category: msg.category,
          timestamp: msg.timestamp,
        });
        // Track memory category ops
        if (msg.category === 'memory') {
          setMemoryOps(prev => [{
            id: msg.id,
            type: 'call',
            tool_name: msg.tool_name,
            tool_input: msg.tool_input,
            timestamp: msg.timestamp,
          }, ...prev].slice(0, MAX_MEMORY_OPS));
        }
        break;

      case 'tool_result':
        setActiveToolCall(null);
        // Track memory results
        if (msg.category === 'memory') {
          setMemoryOps(prev => [{
            id: msg.id,
            type: 'result',
            tool_name: msg.tool_name,
            tool_output: msg.tool_output,
            timestamp: msg.timestamp,
          }, ...prev].slice(0, MAX_MEMORY_OPS));
        }
        // Only add non-memory tool calls to the toolCalls list
        if (msg.category !== 'memory' && msg.category !== 'todo') {
          setToolCalls(prev => [{
            id: msg.id,
            tool_name: msg.tool_name,
            tool_input: msg.tool_input,
            tool_output: msg.tool_output,
            agent: msg.agent,
            category: msg.category,
            timestamp: msg.timestamp,
          }, ...prev].slice(0, MAX_TOOL_CALLS));
        }
        break;

      case 'subagent_start':
        setIsRunning(true);
        if (msg.agent === 'single_arb_captain') {
          setCycleCount(c => c + 1);
          setThinking({ text: '', agent: msg.agent, streaming: false });
        }
        break;

      case 'subagent_complete':
        if (msg.agent === 'single_arb_captain') {
          setIsRunning(false);
        }
        break;

      case 'subagent_error':
        setIsRunning(false);
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

  // Persist state to sessionStorage (debounced)
  useEffect(() => {
    const timer = setTimeout(() => {
      saveSnapshot({
        cycleCount,
        thinking: { ...thinking, streaming: false }, // Always save as non-streaming
        toolCalls,
        todos,
        memoryOps,
      });
    }, 500);
    return () => clearTimeout(timer);
  }, [cycleCount, thinking, toolCalls, todos, memoryOps]);

  return {
    isRunning,
    cycleCount,
    thinking,
    activeToolCall,
    toolCalls,
    todos,
    memoryOps,
  };
};

export default useArbAgent;
