import React, { memo, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Brain, Loader2, Eye, Terminal } from 'lucide-react';
import SidebarCollapseToggle from './SidebarCollapseToggle';
import CaptainHeader from '../panels/agent/CaptainHeader';
import AgentStateIndicator from '../panels/agent/AgentStateIndicator';
import ActivityFeed from '../panels/agent/ActivityFeed';
import TodoListSection from '../panels/agent/TodoListSection';
import { CommandoSection } from '../panels/agent/CommandoSection';
import QuickTrades from '../panels/agent/QuickTrades';
import renderThinkingMarkdown from '../../../utils/renderThinkingMarkdown';
import { RIGHT_SIDEBAR_WIDTH, RIGHT_SIDEBAR_COLLAPSED } from '../utils/styleConstants';

const ThinkingPanel = memo(({ thinking, activeToolCall, isRunning }) => {
  const scrollRef = useRef(null);

  useEffect(() => {
    if (scrollRef.current && thinking.streaming) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [thinking.text, thinking.streaming]);

  const isActive = isRunning || thinking.text || activeToolCall;

  return (
    <motion.div
      animate={{ height: isActive ? 'auto' : 48 }}
      transition={{ duration: 0.15, ease: 'easeOut' }}
      className="shrink-0 overflow-hidden border-b border-gray-800/20"
      style={{ maxHeight: '45%' }}
    >
      <div
        ref={scrollRef}
        data-testid="thinking-stream"
        className="h-full overflow-y-auto px-3 py-2"
      >
        {thinking.text ? (
          <AnimatePresence mode="wait">
            {thinking.streaming ? (
              <motion.div
                key="streaming"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.15 }}
              >
                <pre className="whitespace-pre-wrap font-mono text-[11px] text-gray-300 leading-relaxed">
                  {thinking.text}
                  <span
                    className="inline-block w-[2px] h-[13px] bg-violet-400 ml-0.5 align-text-bottom"
                    style={{ animation: 'blink 1s step-end infinite' }}
                  />
                </pre>
                <style>{`@keyframes blink { 50% { opacity: 0; } }`}</style>
              </motion.div>
            ) : (
              <motion.div
                key="markdown"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.15 }}
                className="prose-agent space-y-1"
              >
                {renderThinkingMarkdown(thinking.text)}
              </motion.div>
            )}
          </AnimatePresence>
        ) : activeToolCall ? (
          <div className="flex items-center gap-2 py-0.5">
            <Loader2 className="w-3 h-3 text-cyan-400 animate-spin" />
            <span className="text-[11px] font-mono text-cyan-400/80">{activeToolCall.tool_name}()</span>
          </div>
        ) : isRunning ? (
          <div className="flex items-center gap-2 py-0.5 text-gray-500">
            <Loader2 className="w-3 h-3 animate-spin" />
            <span className="text-[11px]">Reasoning...</span>
          </div>
        ) : (
          <div className="flex items-center gap-2 py-0.5 text-gray-600">
            <Eye className="w-3 h-3" />
            <span className="text-[11px]">Observing</span>
          </div>
        )}
      </div>

      {/* Active tool overlay */}
      {activeToolCall && thinking.text && (
        <div className="px-3 pb-1.5 flex items-center gap-1.5 text-cyan-400/80">
          <Terminal className="w-2.5 h-2.5" />
          <span className="text-[10px] font-mono">{activeToolCall.tool_name}()</span>
        </div>
      )}
    </motion.div>
  );
});
ThinkingPanel.displayName = 'ThinkingPanel';

const RightSidebar = memo(({
  collapsed, onToggle,
  isRunning, currentSubagent, cycleCount,
  thinking, activeToolCall, toolCalls,
  todos, memoryOps, commandoSessions, arbTrades,
}) => {
  return (
    <motion.div
      className="shrink-0 flex flex-col bg-gray-950/60 border-l border-gray-800/40 overflow-hidden"
      animate={{ width: collapsed ? RIGHT_SIDEBAR_COLLAPSED : RIGHT_SIDEBAR_WIDTH }}
      transition={{ type: 'spring', stiffness: 400, damping: 30 }}
    >
      {/* Toggle bar */}
      <div className="flex items-center justify-between px-2 py-1.5 border-b border-gray-800/30 shrink-0">
        {!collapsed && (
          <span className="text-[10px] font-semibold text-gray-500 uppercase tracking-wider pl-1">Agent</span>
        )}
        <SidebarCollapseToggle collapsed={collapsed} onToggle={onToggle} side="right" />
      </div>

      {collapsed ? (
        <div className="flex flex-col items-center gap-3 py-3">
          <Brain className={`w-4 h-4 ${isRunning ? 'text-violet-400' : 'text-gray-600'}`} />
        </div>
      ) : (
        <>
          <CaptainHeader isRunning={isRunning} cycleCount={cycleCount} currentSubagent={currentSubagent} />
          <AgentStateIndicator isRunning={isRunning} activeToolCall={activeToolCall} thinking={thinking} />
          <ThinkingPanel thinking={thinking} activeToolCall={activeToolCall} isRunning={isRunning} />

          {/* Todo + Commando */}
          {(todos.length > 0 || commandoSessions.length > 0) && (
            <div className="px-2 py-1.5 space-y-1.5 border-b border-gray-800/20 shrink-0">
              <TodoListSection todos={todos} />
              <CommandoSection sessions={commandoSessions} />
            </div>
          )}

          {/* Activity Feed */}
          <ActivityFeed toolCalls={toolCalls} memoryOps={memoryOps} />

          {/* Quick Trades */}
          <QuickTrades trades={arbTrades} />
        </>
      )}
    </motion.div>
  );
});

RightSidebar.displayName = 'RightSidebar';

export default RightSidebar;
