import React, { useRef, useEffect, useCallback, useState, memo } from 'react';
import { Activity, Copy, Check } from 'lucide-react';
import ConsoleMessage from '../ui/ConsoleMessage';

/**
 * V3ConsoleOutput - Console output panel with message display
 */
const V3ConsoleOutput = ({ messages, expandedMessages, onToggleExpand }) => {
  const [copied, setCopied] = useState(false);
  const [autoScroll, setAutoScroll] = useState(false);
  const messagesContainerRef = useRef(null);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  // Handle scroll events to detect if user is near bottom
  const handleScroll = useCallback(() => {
    if (!messagesContainerRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = messagesContainerRef.current;
    const isNearBottom = scrollHeight - scrollTop - clientHeight < 100;
    setAutoScroll(isNearBottom);
  }, []);

  // Only auto-scroll if user is near bottom
  useEffect(() => {
    if (autoScroll) {
      scrollToBottom();
    }
  }, [messages, autoScroll]);

  const copyConsoleOutput = () => {
    const output = messages.map(msg => {
      const state = msg.metadata?.state ? `[${msg.metadata.state}] ` : '';
      const status = msg.metadata?.status ? `[${msg.metadata.status}] ` : '';
      const transition = msg.metadata?.isTransition
        ? `[${msg.metadata.fromState} -> ${msg.metadata.toState}] `
        : '';
      return `${msg.timestamp} ${transition || state}${status}${msg.content}`;
    }).join('\n');

    navigator.clipboard.writeText(output).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  return (
    <div className="bg-gray-900/50 backdrop-blur-sm rounded-xl border border-gray-800 overflow-hidden">
      {/* Console Header */}
      <div className="bg-black/50 px-6 py-3 border-b border-gray-800 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="flex space-x-1.5">
            <div className="w-3 h-3 rounded-full bg-red-500/80 hover:bg-red-500 transition-colors cursor-pointer" />
            <div className="w-3 h-3 rounded-full bg-yellow-500/80 hover:bg-yellow-500 transition-colors cursor-pointer" />
            <div className="w-3 h-3 rounded-full bg-green-500/80 hover:bg-green-500 transition-colors cursor-pointer" />
          </div>
          <span className="text-xs text-gray-400 font-mono uppercase tracking-wider">System Console</span>
        </div>
        <div className="flex items-center space-x-4">
          <button
            onClick={copyConsoleOutput}
            className="flex items-center space-x-2 px-3 py-1.5 text-xs text-gray-400 hover:text-gray-200 hover:bg-gray-800 rounded-lg transition-all"
            title="Copy console output"
          >
            {copied ? (
              <>
                <Check className="w-3.5 h-3.5" />
                <span className="font-medium">Copied!</span>
              </>
            ) : (
              <>
                <Copy className="w-3.5 h-3.5" />
                <span className="font-medium">Copy</span>
              </>
            )}
          </button>
          <span className="text-xs text-gray-500 font-mono">
            {messages.length} messages
          </span>
        </div>
      </div>

      {/* Messages */}
      <div
        ref={messagesContainerRef}
        onScroll={handleScroll}
        className="h-[600px] overflow-y-auto p-4 font-mono text-sm bg-black/20"
      >
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-gray-600">
            <Activity className="w-8 h-8 mb-3 animate-pulse" />
            <div>Waiting for messages...</div>
          </div>
        ) : (
          <div className="space-y-2">
            {messages.map((message) => (
              <ConsoleMessage
                key={message.id}
                message={message}
                isExpanded={expandedMessages.has(message.id)}
                onToggleExpand={onToggleExpand}
              />
            ))}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>
    </div>
  );
};

export default memo(V3ConsoleOutput);
