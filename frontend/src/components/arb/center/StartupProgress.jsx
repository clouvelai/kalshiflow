import React, { memo, useEffect, useRef } from 'react';

const formatTime = (ts) => {
  const d = new Date(ts);
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
};

const StartupProgress = memo(({ messages = [] }) => {
  const scrollRef = useRef(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages.length]);

  const latest = messages[messages.length - 1];
  const progress = latest?.step && latest?.totalSteps
    ? latest.step / latest.totalSteps
    : null;

  return (
    <div className="flex-1 flex items-center justify-center p-6">
      <div className="w-full max-w-md bg-gray-900/80 border border-gray-800/60 rounded-xl p-6 shadow-xl">
        {/* Header with pulse */}
        <div className="flex items-center gap-3 mb-5">
          <span className="relative flex h-3 w-3">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-400 opacity-75" />
            <span className="relative inline-flex rounded-full h-3 w-3 bg-blue-500" />
          </span>
          <span className="text-sm font-semibold text-gray-200 tracking-wide">
            System Starting...
          </span>
        </div>

        {/* Message log */}
        <div
          ref={scrollRef}
          className="space-y-1.5 max-h-60 overflow-y-auto mb-4 scrollbar-thin scrollbar-thumb-gray-700"
        >
          {messages.length === 0 && (
            <p className="text-xs text-gray-600 italic">Waiting for backend...</p>
          )}
          {messages.map((msg, i) => (
            <div
              key={i}
              className="flex items-start gap-2 text-xs animate-in fade-in duration-300"
              style={{ animationDelay: `${i * 30}ms` }}
            >
              <span className="text-gray-600 font-mono shrink-0 tabular-nums">
                {formatTime(msg.timestamp)}
              </span>
              <span className={i === messages.length - 1 ? 'text-gray-200' : 'text-gray-400'}>
                {msg.message}
              </span>
            </div>
          ))}
        </div>

        {/* Progress bar */}
        {progress != null && (
          <div className="mt-2">
            <div className="flex items-center justify-between mb-1">
              <div className="h-1.5 flex-1 bg-gray-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-blue-500/80 rounded-full transition-all duration-500 ease-out"
                  style={{ width: `${Math.min(progress * 100, 100)}%` }}
                />
              </div>
              <span className="text-[10px] text-gray-500 ml-3 tabular-nums font-mono">
                {latest.step}/{latest.totalSteps}
              </span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
});

StartupProgress.displayName = 'StartupProgress';

export default StartupProgress;
