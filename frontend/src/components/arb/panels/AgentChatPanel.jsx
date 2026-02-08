import React, { memo } from 'react';
import { Brain, Zap } from 'lucide-react';
import { panelClasses } from '../utils/styleConstants';
import {
  ThinkingStream,
  TodoListSection,
  ToolCallsSection,
  TradesSection,
  CommandoSection,
  MemorySection,
} from './agent';

/**
 * AgentChatPanel - Main Captain agent interface panel.
 *
 * Split layout: Thinking stream (left) | Sidebar (right).
 * Below: Commando sessions + trades.
 */
const AgentChatPanel = ({
  thinking = { text: '', agent: null, streaming: false },
  activeToolCall = null,
  toolCalls = [],
  arbTrades = [],
  isRunning = false,
  currentSubagent = null,
  cycleCount = 0,
  todos = [],
  memoryOps = [],
  commandoSessions = [],
}) => {
  const subagentLabel =
    currentSubagent && currentSubagent !== 'single_arb_captain'
      ? currentSubagent.replace(/_/g, ' ')
      : null;

  return (
    <div
      id="agent-panel"
      data-testid="agent-panel"
      data-running={isRunning}
      className={`${panelClasses('violet')} flex flex-col max-h-[620px]`}
    >
      {/* ── Header ── */}
      <div className="flex items-center justify-between px-5 py-3.5 border-b border-violet-800/15 shrink-0">
        <div className="flex items-center gap-3">
          <div className={`p-1.5 rounded-lg ${isRunning ? 'bg-violet-500/15' : 'bg-gray-800/40'} transition-colors duration-300`}>
            <Brain className={`w-4 h-4 transition-colors duration-300 ${isRunning ? 'text-violet-400' : 'text-gray-500'}`} />
          </div>
          <div className="flex items-baseline gap-2">
            <h3 className="text-[13px] font-semibold text-gray-200 tracking-wide">Captain</h3>
            {isRunning && (
              <span className="flex items-center gap-1.5">
                <span className="relative flex h-1.5 w-1.5">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-violet-400 opacity-60" />
                  <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-violet-400" />
                </span>
                <span data-testid="agent-status-label" className="text-[10px] text-violet-400/80 font-medium">
                  {subagentLabel || 'Thinking'}
                </span>
              </span>
            )}
            {!isRunning && (
              <span data-testid="agent-status-label" className="text-[10px] text-gray-600">Idle</span>
            )}
          </div>
        </div>

        <div className="flex items-center gap-2.5">
          {cycleCount > 0 && (
            <span id="agent-cycle-count" data-testid="agent-cycle-count" className="text-[10px] text-gray-600 font-mono tabular-nums">
              Cycle {cycleCount}
            </span>
          )}
          <span
            id="agent-running-badge"
            data-testid="agent-running-badge"
            data-running={isRunning}
            className={`flex items-center gap-1 px-2 py-0.5 rounded-full text-[9px] font-semibold uppercase tracking-wider transition-colors duration-300 ${
              isRunning
                ? 'bg-violet-500/15 text-violet-300 border border-violet-500/20'
                : 'bg-gray-800/40 text-gray-600 border border-gray-700/20'
            }`}
          >
            {isRunning && <Zap className="w-2.5 h-2.5" />}
            {isRunning ? 'Active' : 'Idle'}
          </span>
        </div>
      </div>

      {/* ── Content ── */}
      <div className="flex-1 min-h-0 overflow-y-auto px-5 py-4 flex flex-col gap-4">
        {/* Split view: Thinking | Sidebar */}
        <div
          id="agent-split-view"
          data-testid="agent-split-view"
          className="grid grid-cols-5 gap-4 min-h-[280px] max-h-[420px]"
        >
          {/* Left: Thinking Stream (3 cols) */}
          <div id="thinking-container" data-testid="thinking-container" className="col-span-3 flex flex-col min-h-0">
            <ThinkingStream
              thinking={thinking}
              activeToolCall={activeToolCall}
              isRunning={isRunning}
            />
          </div>

          {/* Right: Sidebar (2 cols) */}
          <div id="agent-sidebar" data-testid="agent-sidebar" className="col-span-2 flex flex-col gap-2.5 overflow-y-auto min-h-0">
            <TodoListSection todos={todos} />
            <ToolCallsSection toolCalls={toolCalls} />
            <MemorySection memoryOps={memoryOps} />
            {todos.length === 0 && toolCalls.length === 0 && memoryOps.length === 0 && (
              <div className="flex-1 flex items-center justify-center">
                <span className="text-[11px] text-gray-700">Waiting for activity...</span>
              </div>
            )}
          </div>
        </div>

        {/* Commando Sessions */}
        <CommandoSection sessions={commandoSessions} />

        {/* Trades */}
        <TradesSection trades={arbTrades} />
      </div>
    </div>
  );
};

export default memo(AgentChatPanel);
