import React, { memo, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Loader2, Eye, Terminal, Crosshair, AlertTriangle, Radio } from 'lucide-react';
import CaptainHeader from '../panels/agent/CaptainHeader';
import AgentStateIndicator from '../panels/agent/AgentStateIndicator';
import AttentionFeed from '../panels/agent/AttentionFeed';
import AutoActionStrip from '../panels/agent/AutoActionStrip';
import ActivityFeed from '../panels/agent/ActivityFeed';
import TodoListSection from '../panels/agent/TodoListSection';
import renderThinkingMarkdown from '../../../utils/renderThinkingMarkdown';

/**
 * ThinkingPanel - Captain's thinking stream, now full-width with flex layout.
 * No longer capped at 45% - uses flex-1 with min/max constraints.
 */
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
      style={{ maxHeight: '50%' }}
    >
      <div
        ref={scrollRef}
        data-testid="thinking-stream"
        className="h-full overflow-y-auto px-4 py-2.5"
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
        <div className="px-4 pb-1.5 flex items-center gap-1.5 text-cyan-400/80">
          <Terminal className="w-2.5 h-2.5" />
          <span className="text-[10px] font-mono">{activeToolCall.tool_name}()</span>
        </div>
      )}
    </motion.div>
  );
});
ThinkingPanel.displayName = 'ThinkingPanel';

/**
 * SniperStatusStrip - Compact sniper telemetry bar.
 */
const SniperStatusStrip = memo(({ sniperState }) => {
  if (!sniperState || !sniperState.enabled) {
    return (
      <div className="flex items-center gap-2 px-3 py-1.5 border-b border-gray-800/20 shrink-0">
        <Crosshair className="w-3 h-3 text-gray-600" />
        <span className="text-[9px] text-gray-600 uppercase tracking-wider font-medium">Sniper Off</span>
      </div>
    );
  }

  const last = sniperState.lastAction;
  const hasError = last?.error;
  const actionCount = sniperState.recentActions?.length || 0;

  return (
    <div className={`flex items-center gap-3 px-3 py-1.5 border-b shrink-0 ${
      hasError ? 'border-amber-500/20 bg-amber-500/5' : 'border-gray-800/20'
    }`}>
      <div className="flex items-center gap-1.5">
        <Crosshair className={`w-3 h-3 ${hasError ? 'text-amber-400' : 'text-emerald-400'}`} />
        <span className="text-[9px] text-emerald-400 uppercase tracking-wider font-semibold">Sniper</span>
      </div>
      {last && (
        <>
          <span className="text-[9px] font-mono text-gray-500">
            {last.direction?.toUpperCase()} {last.event_ticker?.slice(0, 12)}
          </span>
          <span className={`text-[9px] font-mono ${hasError ? 'text-amber-400' : 'text-emerald-400'}`}>
            {hasError ? `Rejected` : `${last.legs_filled}/${last.legs_attempted} legs`}
          </span>
          {last.latency_ms && (
            <span className="text-[9px] font-mono text-gray-600">{Math.round(last.latency_ms)}ms</span>
          )}
        </>
      )}
      <span className="text-[9px] font-mono text-gray-600 ml-auto">{actionCount} actions</span>
    </div>
  );
});
SniperStatusStrip.displayName = 'SniperStatusStrip';

/**
 * FreshnessBar - Shows exchange status, data freshness from feed_stats.
 */
const FreshnessBar = memo(({ exchangeStatus, feedStats }) => {
  const exchangeDown = exchangeStatus && exchangeStatus.active === false;
  if (!exchangeDown && !feedStats) return null;

  const obAge = feedStats?.last_orderbook_at
    ? Math.round((Date.now() / 1000 - feedStats.last_orderbook_at))
    : null;

  return (
    <div className={`flex items-center gap-3 px-3 py-1 border-b shrink-0 ${
      exchangeDown ? 'border-red-500/20 bg-red-500/5' : 'border-gray-800/15'
    }`}>
      {exchangeDown && (
        <div className="flex items-center gap-1.5">
          <AlertTriangle className="w-3 h-3 text-red-400" />
          <span className="text-[9px] text-red-400 font-semibold uppercase">Exchange Down</span>
          {exchangeStatus.error && (
            <span className="text-[9px] text-red-400/60 truncate max-w-[200px]">{exchangeStatus.error}</span>
          )}
        </div>
      )}
      {obAge != null && (
        <div className="flex items-center gap-1 ml-auto">
          <Radio className="w-2.5 h-2.5 text-gray-600" />
          <span className={`text-[9px] font-mono ${obAge > 30 ? 'text-amber-400' : 'text-gray-600'}`}>
            OB: {obAge}s ago
          </span>
        </div>
      )}
    </div>
  );
});
FreshnessBar.displayName = 'FreshnessBar';

/**
 * AgentPanel - Full-width agent view for the main content area.
 *
 * Renders CaptainHeader, AgentStateIndicator, SniperStatusStrip, ThinkingPanel,
 * then Activity feed (todos + tool timeline).
 */
const AgentPanel = memo(({
  isRunning, cycleCount,
  thinking, activeToolCall, toolCalls,
  todos, memoryOps,
  sniperState, captainPaused, exchangeStatus, feedStats,
  attentionItems, attentionStats, autoActions, captainMode, captainTiming, cycleMode,
  connectionStatus,
}) => {
  const attentionCount = attentionItems?.length || 0;

  return (
    <div className="flex-1 min-h-0 flex flex-col bg-gray-950/30">
      <CaptainHeader isRunning={isRunning} cycleCount={cycleCount} captainPaused={captainPaused} exchangeStatus={exchangeStatus} captainMode={captainMode} captainTiming={captainTiming} attentionCount={attentionCount} connectionStatus={connectionStatus} />
      <FreshnessBar exchangeStatus={exchangeStatus} feedStats={feedStats} />
      <SniperStatusStrip sniperState={sniperState} />
      <AutoActionStrip autoActions={autoActions} />
      <AgentStateIndicator isRunning={isRunning} activeToolCall={activeToolCall} thinking={thinking} cycleMode={cycleMode} attentionCount={attentionCount} />
      <AttentionFeed attentionItems={attentionItems} attentionStats={attentionStats} />
      <ThinkingPanel thinking={thinking} activeToolCall={activeToolCall} isRunning={isRunning} />

      {/* Activity content */}
      <div className="flex-1 min-h-0 flex flex-col overflow-hidden">
        {todos.length > 0 && (
          <div className="px-3 py-2 shrink-0">
            <TodoListSection todos={todos} />
          </div>
        )}
        <ActivityFeed toolCalls={toolCalls} memoryOps={memoryOps} />
      </div>
    </div>
  );
});

AgentPanel.displayName = 'AgentPanel';

export default AgentPanel;
