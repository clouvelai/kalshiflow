import React, { memo, useRef, useEffect, useState } from 'react';
import {
  Brain, Wrench, CheckCircle, RefreshCw,
  ArrowUpCircle, ChevronDown, ChevronRight, Eye,
  Search, Database, Globe, FileText, ShoppingCart,
} from 'lucide-react';
import renderThinkingMarkdown from '../../../utils/renderThinkingMarkdown';

const fmtTime = (ts) => {
  if (!ts) return '';
  try {
    const d = new Date(ts);
    return d.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  } catch {
    return '';
  }
};

const TOOL_ICONS = {
  get_pair_snapshot: Database,
  get_spread_snapshot: Database,
  get_event_codex: FileText,
  get_validation_status: CheckCircle,
  get_system_state: Database,
  memory_store: Database,
  memory_search: Search,
  buy_arb_position: ShoppingCart,
  sell_arb_position: ShoppingCart,
  delegate_event_analyst: Brain,
  delegate_memory_curator: Brain,
  kalshi_get_events: Globe,
  kalshi_get_markets: Globe,
  kalshi_get_orderbook: Globe,
  kalshi_get_positions: Globe,
  kalshi_get_fills: Globe,
  poly_get_events: Globe,
  poly_get_markets: Globe,
  poly_search_events: Search,
  get_pair_history: Database,
  save_validation: FileText,
};

const getToolIcon = (toolName) => TOOL_ICONS[toolName] || Wrench;

/* ─── ThinkingStream ─── */
const ThinkingStream = memo(({ thinking, activeToolCall, isRunning }) => {
  const scrollRef = useRef(null);

  useEffect(() => {
    if (scrollRef.current && thinking.streaming) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [thinking.text, thinking.streaming]);

  const hasContent = thinking.text || activeToolCall || isRunning;

  return (
    <div className="relative">
      <div
        ref={scrollRef}
        className="max-h-[400px] overflow-y-auto rounded-lg bg-gray-900/60 border border-violet-800/20 p-4"
      >
        {thinking.text ? (
          <div className="space-y-0">
            {renderThinkingMarkdown(thinking.text)}
            {thinking.streaming && (
              <span className="inline-block w-2 h-4 bg-violet-400 animate-pulse ml-0.5 align-text-bottom" />
            )}
          </div>
        ) : activeToolCall ? (
          <div className="flex items-center gap-3 text-cyan-400">
            <RefreshCw className="w-4 h-4 animate-spin" />
            <span className="text-sm font-mono">
              Executing: {activeToolCall.tool_name}()
            </span>
          </div>
        ) : isRunning ? (
          <div className="flex items-center gap-3 text-gray-500">
            <RefreshCw className="w-4 h-4 animate-spin" />
            <span className="text-sm">Agent is thinking...</span>
          </div>
        ) : (
          <div className="flex items-center gap-2 text-gray-600 py-2">
            <Eye className="w-4 h-4" />
            <span className="text-sm">Agent is observing...</span>
          </div>
        )}
      </div>

      {/* Active tool call overlay at bottom */}
      {activeToolCall && thinking.text && (
        <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-gray-900 to-transparent pt-6 pb-2 px-4 rounded-b-lg">
          <div className="flex items-center gap-2 text-cyan-400">
            <RefreshCw className="w-3.5 h-3.5 animate-spin" />
            <span className="text-xs font-mono">
              {activeToolCall.tool_name}()
            </span>
          </div>
        </div>
      )}
    </div>
  );
});
ThinkingStream.displayName = 'ThinkingStream';

/* ─── ToolCallRow ─── */
const ToolCallRow = memo(({ call }) => {
  const Icon = getToolIcon(call.tool_name);
  return (
    <div className="flex items-center gap-2 py-1.5 border-b border-gray-800/50 last:border-0">
      <Icon className="w-3.5 h-3.5 text-cyan-500 shrink-0" />
      <span className="text-[11px] font-mono text-gray-300 shrink-0">
        {call.tool_name}
      </span>
      {call.agent && call.agent !== 'captain' && (
        <span className="text-[9px] px-1 py-0.5 bg-violet-900/40 text-violet-400 rounded shrink-0">
          {call.agent}
        </span>
      )}
      {call.tool_input && (
        <span className="text-[10px] text-gray-600 truncate flex-1 font-mono">
          {call.tool_input}
        </span>
      )}
      <span className="text-[9px] text-gray-700 font-mono shrink-0 ml-auto">
        {fmtTime(call.timestamp)}
      </span>
    </div>
  );
});
ToolCallRow.displayName = 'ToolCallRow';

/* ─── ToolCallsSection ─── */
const ToolCallsSection = memo(({ toolCalls }) => {
  const [expanded, setExpanded] = useState(false);
  const visible = toolCalls.slice(0, 10);

  if (toolCalls.length === 0) return null;

  return (
    <div className="rounded-lg border border-gray-800/40 bg-gray-900/30">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center gap-2 px-3 py-2 text-left hover:bg-gray-800/30 transition-colors rounded-lg"
      >
        {expanded ? (
          <ChevronDown className="w-3.5 h-3.5 text-gray-500" />
        ) : (
          <ChevronRight className="w-3.5 h-3.5 text-gray-500" />
        )}
        <Wrench className="w-3.5 h-3.5 text-cyan-500" />
        <span className="text-[11px] font-semibold text-gray-400 uppercase tracking-wider">
          Tool Calls
        </span>
        <span className="text-[10px] text-gray-600 font-mono ml-auto">
          {toolCalls.length}
        </span>
      </button>
      {expanded && (
        <div className="px-3 pb-2 max-h-[200px] overflow-y-auto">
          {visible.map(call => (
            <ToolCallRow key={call.id} call={call} />
          ))}
          {toolCalls.length > 10 && (
            <div className="text-[10px] text-gray-600 text-center py-1">
              +{toolCalls.length - 10} more
            </div>
          )}
        </div>
      )}
    </div>
  );
});
ToolCallsSection.displayName = 'ToolCallsSection';

/* ─── TradeCard (arbTrades shape) ─── */
const TradeCard = memo(({ trade }) => {
  const isYes = trade.side?.toLowerCase() === 'yes';

  return (
    <div className="rounded-lg border border-emerald-800/30 bg-emerald-950/20 p-3">
      <div className="flex items-center gap-2 flex-wrap">
        <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-emerald-900/60 text-emerald-300">
          BUY
        </span>
        <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${
          isYes ? 'bg-blue-900/60 text-blue-300' : 'bg-orange-900/60 text-orange-300'
        }`}>
          {trade.side?.toUpperCase() || '?'}
        </span>
        <span className="text-[11px] font-mono text-gray-300">
          {trade.kalshi_ticker || 'unknown'}
        </span>
        {trade.contracts && (
          <span className="text-[10px] text-gray-500">
            x{trade.contracts}
          </span>
        )}
        {trade.price_cents != null && (
          <span className="text-[10px] text-gray-400 font-mono">
            @{trade.price_cents}c
          </span>
        )}
        {trade.spread_at_entry != null && (
          <span className="text-[10px] text-cyan-500 font-mono">
            spread {trade.spread_at_entry.toFixed(1)}c
          </span>
        )}
        <CheckCircle className="w-3.5 h-3.5 text-emerald-400 ml-auto" />
        <span className="text-[9px] text-gray-600 font-mono">
          {fmtTime(trade.timestamp)}
        </span>
      </div>
    </div>
  );
});
TradeCard.displayName = 'TradeCard';

/* ─── TradesSection ─── */
const TradesSection = memo(({ trades }) => {
  if (trades.length === 0) return null;

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2 px-1">
        <ArrowUpCircle className="w-3.5 h-3.5 text-amber-500" />
        <span className="text-[11px] font-semibold text-gray-400 uppercase tracking-wider">
          Trades
        </span>
        <span className="text-[10px] text-gray-600 font-mono ml-auto">
          {trades.length}
        </span>
      </div>
      <div className="max-h-[300px] overflow-y-auto space-y-2">
        {trades.map(trade => (
          <TradeCard key={trade.id} trade={trade} />
        ))}
      </div>
    </div>
  );
});
TradesSection.displayName = 'TradesSection';

/* ─── AgentChatPanel ─── */
const AgentChatPanel = ({
  thinking = { text: '', agent: null, streaming: false },
  activeToolCall = null,
  toolCalls = [],
  arbTrades = [],
  isRunning = false,
  currentSubagent = null,
  cycleCount = 0,
}) => {
  return (
    <div className="
      bg-gradient-to-br from-gray-900/70 via-gray-900/50 to-gray-950/70
      backdrop-blur-md rounded-2xl
      border border-violet-800/30
      shadow-xl shadow-black/20
      p-5 flex flex-col gap-4
    ">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="p-2 rounded-lg bg-gradient-to-br from-violet-900/40 to-violet-950/40 border border-violet-700/30">
            <Brain className={`w-4 h-4 ${isRunning ? 'text-violet-400 animate-pulse' : 'text-violet-500'}`} />
          </div>
          <div>
            <h3 className="text-sm font-bold text-gray-200 uppercase tracking-wider">Agent</h3>
            <div className="flex items-center gap-2 mt-0.5">
              {isRunning ? (
                <>
                  <RefreshCw className="w-3 h-3 text-violet-400 animate-spin" />
                  <span className="text-[10px] text-violet-400 font-medium">
                    {currentSubagent ? `Running: ${currentSubagent}` : 'Running'}
                  </span>
                </>
              ) : (
                <span className="text-[10px] text-gray-500">Idle</span>
              )}
            </div>
          </div>
        </div>
        <div className="flex items-center gap-3">
          {cycleCount > 0 && (
            <span className="text-[10px] text-gray-500 font-mono">
              Cycle #{cycleCount}
            </span>
          )}
          <span className={`px-2 py-0.5 rounded-full text-[9px] font-bold uppercase tracking-wider ${
            isRunning
              ? 'bg-violet-900/50 text-violet-300 border border-violet-700/40'
              : 'bg-gray-800/50 text-gray-500 border border-gray-700/30'
          }`}>
            {isRunning ? 'Running' : 'Idle'}
          </span>
        </div>
      </div>

      {/* Thinking Stream */}
      <ThinkingStream
        thinking={thinking}
        activeToolCall={activeToolCall}
        isRunning={isRunning}
      />

      {/* Tool Calls (collapsible) */}
      <ToolCallsSection toolCalls={toolCalls} />

      {/* Trades */}
      <TradesSection trades={arbTrades} />
    </div>
  );
};

export default memo(AgentChatPanel);
