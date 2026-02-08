import React, { memo, useRef, useEffect } from 'react';
import { Loader2, Eye, Terminal } from 'lucide-react';
import renderThinkingMarkdown from '../../../../utils/renderThinkingMarkdown';

/**
 * ThinkingStream - Displays Captain's reasoning with auto-scroll.
 *
 * States:
 *   streaming  -> Raw text with blinking caret
 *   complete   -> Rich markdown rendering
 *   executing  -> Tool name overlay
 *   thinking   -> Subtle spinner
 *   idle       -> Observing placeholder
 */
const ThinkingStream = memo(({ thinking, activeToolCall, isRunning }) => {
  const scrollRef = useRef(null);

  useEffect(() => {
    if (scrollRef.current && thinking.streaming) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [thinking.text, thinking.streaming]);

  const hasContent = thinking.text || activeToolCall || isRunning;

  return (
    <div id="thinking-stream" data-testid="thinking-stream" className="relative flex-1 min-h-0">
      <div
        ref={scrollRef}
        data-testid="thinking-content"
        className="h-full overflow-y-auto rounded-lg bg-gray-900/50 border border-violet-500/8 p-4"
      >
        {thinking.text ? (
          thinking.streaming ? (
            /* ── Streaming: raw text + thin blinking caret ── */
            <div>
              <pre className="whitespace-pre-wrap font-mono text-[12px] text-gray-300 leading-relaxed">
                {thinking.text}
                <span
                  className="inline-block w-[2px] h-[14px] bg-violet-400 ml-0.5 align-text-bottom"
                  style={{ animation: 'blink 1s step-end infinite' }}
                />
              </pre>
              <style>{`@keyframes blink { 50% { opacity: 0; } }`}</style>
            </div>
          ) : (
            /* ── Complete: rich markdown ── */
            <div className="prose-agent space-y-1">
              {renderThinkingMarkdown(thinking.text)}
            </div>
          )
        ) : activeToolCall ? (
          /* ── Executing tool (no thinking text yet) ── */
          <div className="flex items-center gap-3 py-1">
            <Loader2 className="w-4 h-4 text-cyan-400 animate-spin" />
            <span className="text-[12px] font-mono text-cyan-400/80">
              {activeToolCall.tool_name}()
            </span>
          </div>
        ) : isRunning ? (
          /* ── Thinking (no text yet) ── */
          <div className="flex items-center gap-3 py-1 text-gray-500">
            <Loader2 className="w-4 h-4 animate-spin" />
            <span className="text-[12px]">Reasoning...</span>
          </div>
        ) : (
          /* ── Idle ── */
          <div className="flex items-center gap-2 py-1 text-gray-600">
            <Eye className="w-4 h-4" />
            <span className="text-[12px]">Observing markets</span>
          </div>
        )}
      </div>

      {/* Active tool overlay at bottom */}
      {activeToolCall && thinking.text && (
        <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-gray-900 via-gray-900/90 to-transparent pt-8 pb-2.5 px-4 rounded-b-lg pointer-events-none">
          <div className="flex items-center gap-2 text-cyan-400/90">
            <Terminal className="w-3 h-3" />
            <span className="text-[11px] font-mono">
              {activeToolCall.tool_name}()
            </span>
          </div>
        </div>
      )}
    </div>
  );
});

ThinkingStream.displayName = 'ThinkingStream';

export default ThinkingStream;
