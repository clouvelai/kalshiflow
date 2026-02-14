import React, { memo, useRef, useEffect } from 'react';
import { Loader2, Eye, Terminal } from 'lucide-react';

/**
 * ThinkingPanel - Admiral's thinking stream.
 */
const ThinkingPanel = memo(({ thinking, isRunning }) => {
  const scrollRef = useRef(null);

  useEffect(() => {
    if (scrollRef.current && thinking?.streaming) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [thinking?.text, thinking?.streaming]);

  return (
    <div className="shrink-0 overflow-hidden border-b border-gray-800/20" style={{ maxHeight: '40%' }}>
      <div ref={scrollRef} className="h-full overflow-y-auto px-4 py-2.5">
        {thinking?.text ? (
          thinking.streaming ? (
            <pre className="whitespace-pre-wrap font-mono text-[11px] text-gray-300 leading-relaxed">
              {thinking.text}
              <span
                className="inline-block w-[2px] h-[13px] bg-amber-400 ml-0.5 align-text-bottom"
                style={{ animation: 'blink 1s step-end infinite' }}
              />
            </pre>
          ) : (
            <pre className="whitespace-pre-wrap font-mono text-[11px] text-gray-400 leading-relaxed">
              {thinking.text}
            </pre>
          )
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
      <style>{`@keyframes blink { 50% { opacity: 0; } }`}</style>
    </div>
  );
});
ThinkingPanel.displayName = 'ThinkingPanel';

/**
 * ToolCallItem - Single tool call in the activity feed.
 */
const ToolCallItem = memo(({ call }) => {
  const statusColor = call.status === 'complete' ? 'text-emerald-400' : 'text-cyan-400';

  return (
    <div className="flex items-center gap-2 px-3 py-1 hover:bg-gray-800/20 transition-colors">
      <Terminal className={`w-3 h-3 ${statusColor} shrink-0`} />
      <span className="text-[10px] font-mono text-gray-300">{call.tool}()</span>
      {call.status === 'running' && (
        <Loader2 className="w-2.5 h-2.5 text-cyan-400 animate-spin ml-auto" />
      )}
      {call.status === 'complete' && (
        <span className="text-[9px] text-emerald-400/60 ml-auto">done</span>
      )}
    </div>
  );
});
ToolCallItem.displayName = 'ToolCallItem';

/**
 * AdmiralAgentPanel - Simplified agent view: thinking + tool call activity feed.
 */
const AdmiralAgentPanel = ({
  isRunning, cycleCount, cycleMode, thinking, toolCalls = [],
}) => {
  return (
    <div className="flex-1 min-h-0 flex flex-col bg-gray-950/30">
      {/* Status bar */}
      <div className="flex items-center gap-2 px-3 py-2 border-b border-gray-800/20 shrink-0">
        <span className={`w-1.5 h-1.5 rounded-full ${isRunning ? 'bg-emerald-500 animate-pulse' : 'bg-gray-600'}`} />
        <span className="text-[10px] font-semibold text-gray-300 uppercase tracking-wider">
          Admiral
        </span>
        {cycleMode && (
          <span className={`text-[9px] px-1.5 py-0.5 rounded font-mono ${
            cycleMode === 'reactive' ? 'bg-amber-500/10 text-amber-400'
              : cycleMode === 'strategic' ? 'bg-blue-500/10 text-blue-400'
              : 'bg-violet-500/10 text-violet-400'
          }`}>
            {cycleMode}
          </span>
        )}
        <span className="text-[9px] font-mono text-gray-600 ml-auto">
          cycle #{cycleCount}
        </span>
      </div>

      {/* Thinking */}
      <ThinkingPanel thinking={thinking || { text: '', streaming: false }} isRunning={isRunning} />

      {/* Tool calls activity feed */}
      <div className="flex-1 min-h-0 overflow-y-auto">
        {toolCalls.length === 0 ? (
          <div className="flex items-center justify-center h-full text-gray-600">
            <span className="text-[11px]">No tool calls yet</span>
          </div>
        ) : (
          <div className="py-1">
            {toolCalls.map((call, i) => (
              <ToolCallItem key={`${call.tool}-${call.timestamp || i}`} call={call} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default memo(AdmiralAgentPanel);
