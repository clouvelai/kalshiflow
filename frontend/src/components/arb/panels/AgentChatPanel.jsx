import React, { memo, useRef, useEffect, useState, useMemo } from 'react';
import {
  Brain, Wrench, CheckCircle, RefreshCw,
  ArrowUpCircle, ChevronDown, ChevronRight, Eye,
  Search, Database, Globe, ShoppingCart, ListTodo,
  ArrowUpRight, ArrowDownLeft, FileText, Wallet,
  Crosshair, Clock, AlertTriangle, Copy, Check,
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
  get_event_snapshot: Database,
  get_all_events: Globe,
  get_events_summary: Globe,
  get_market_orderbook: Database,
  get_recent_trades: Globe,
  get_trade_history: FileText,
  execute_arb: ShoppingCart,
  place_order: Crosshair,
  cancel_order: AlertTriangle,
  get_resting_orders: Clock,
  memory_store: Database,
  memory_search: Search,
  get_positions: Wallet,
  get_balance: Wallet,
};

const getToolIcon = (toolName) => TOOL_ICONS[toolName] || Wrench;

/* ─── Tool Category Detection ─── */
const getToolCategory = (toolName) => {
  if (['execute_arb', 'place_order', 'cancel_order', 'get_resting_orders'].includes(toolName)) return 'arb';
  if (['memory_store', 'memory_search', 'edit_file', 'write_file', 'read_file'].includes(toolName)) return 'memory';
  if (['get_positions', 'get_balance', 'get_trade_history'].includes(toolName)) return 'portfolio';
  if (['get_event_snapshot', 'get_events_summary', 'get_all_events'].includes(toolName)) return 'surveillance';
  if (['get_market_orderbook', 'get_recent_trades'].includes(toolName)) return 'market';
  return 'other';
};

const CATEGORY_STYLES = {
  arb: { bg: 'bg-cyan-900/30', text: 'text-cyan-400', dot: 'bg-cyan-500', label: 'ARB' },
  memory: { bg: 'bg-violet-900/30', text: 'text-violet-400', dot: 'bg-violet-500', label: 'MEM' },
  portfolio: { bg: 'bg-emerald-900/30', text: 'text-emerald-400', dot: 'bg-emerald-500', label: 'PORT' },
  surveillance: { bg: 'bg-orange-900/30', text: 'text-orange-400', dot: 'bg-orange-500', label: 'SURV' },
  market: { bg: 'bg-blue-900/30', text: 'text-blue-400', dot: 'bg-blue-500', label: 'MKT' },
  other: { bg: 'bg-gray-800', text: 'text-gray-400', dot: 'bg-gray-500', label: 'TOOL' },
};

const CategoryBadge = ({ category }) => {
  const style = CATEGORY_STYLES[category] || CATEGORY_STYLES.other;
  return (
    <span className={`ml-2 px-1.5 py-0.5 rounded text-[8px] font-bold uppercase ${style.bg} ${style.text}`}>
      {style.label}
    </span>
  );
};

/* ─── JSON Syntax Highlighting ─── */
const highlightJson = (str) => {
  if (!str) return str;
  // Simple JSON highlighting without dependencies
  return str.replace(
    /("(?:[^"\\]|\\.)*")\s*:/g,
    '<span class="text-violet-400">$1</span>:'
  ).replace(
    /:\s*("(?:[^"\\]|\\.)*")/g,
    ': <span class="text-emerald-400">$1</span>'
  ).replace(
    /:\s*(true|false|null)/g,
    ': <span class="text-amber-400">$1</span>'
  ).replace(
    /:\s*(-?\d+\.?\d*)/g,
    ': <span class="text-cyan-400">$1</span>'
  );
};

const JsonBlock = ({ label, content }) => {
  const [copied, setCopied] = useState(false);

  const formatted = useMemo(() => {
    try {
      const obj = typeof content === 'string' ? JSON.parse(content) : content;
      return JSON.stringify(obj, null, 2);
    } catch {
      return typeof content === 'string' ? content : JSON.stringify(content);
    }
  }, [content]);

  const handleCopy = () => {
    navigator.clipboard.writeText(formatted);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div>
      <div className="flex items-center justify-between mb-1">
        <div className="text-[9px] text-gray-500 uppercase tracking-wider">{label}</div>
        <button
          onClick={handleCopy}
          className="p-1 hover:bg-gray-700/50 rounded transition-colors"
          title="Copy to clipboard"
        >
          {copied ? (
            <Check className="w-3 h-3 text-emerald-400" />
          ) : (
            <Copy className="w-3 h-3 text-gray-500 hover:text-gray-300" />
          )}
        </button>
      </div>
      <pre
        className="text-[11px] font-mono bg-gray-900/60 rounded p-2 overflow-x-auto max-h-[200px] overflow-y-auto text-gray-300"
        dangerouslySetInnerHTML={{ __html: highlightJson(formatted) }}
      />
    </div>
  );
};

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
    <div id="thinking-stream" data-testid="thinking-stream" className="relative flex-1 min-h-0">
      <div
        ref={scrollRef}
        data-testid="thinking-content"
        className="h-full overflow-y-auto rounded-lg bg-gray-900/60 border border-violet-800/20 p-4"
      >
        {thinking.text ? (
          thinking.streaming ? (
            // Streaming mode: raw text with cursor for performance
            <div className="space-y-0">
              <pre className="whitespace-pre-wrap font-mono text-[12px] text-gray-300 leading-relaxed">
                {thinking.text}
              </pre>
              <span className="inline-block w-2 h-4 bg-violet-400 animate-pulse ml-0.5 align-text-bottom" />
            </div>
          ) : (
            // Complete mode: rich markdown rendering
            <div className="prose-agent space-y-1">
              {renderThinkingMarkdown(thinking.text)}
            </div>
          )
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

/* ─── TodoListSection ─── */
const TODO_STATUS_ICONS = {
  completed: { icon: '✓', color: 'text-emerald-400' },
  in_progress: { icon: '◎', color: 'text-amber-400' },
  pending: { icon: '○', color: 'text-gray-500' },
};

const TodoListSection = memo(({ todos }) => {
  if (!todos || todos.length === 0) return null;

  return (
    <div className="rounded-lg border border-gray-800/40 bg-gray-900/30 p-3">
      <div className="flex items-center gap-2 mb-2">
        <ListTodo className="w-3.5 h-3.5 text-amber-500" />
        <span className="text-[11px] font-semibold text-gray-400 uppercase tracking-wider">
          TODO
        </span>
        <span className="text-[10px] text-gray-600 font-mono ml-auto">
          {todos.filter(t => t.status === 'completed').length}/{todos.length}
        </span>
      </div>
      <div className="space-y-1 max-h-[160px] overflow-y-auto">
        {todos.map((todo, i) => {
          const status = TODO_STATUS_ICONS[todo.status] || TODO_STATUS_ICONS.pending;
          return (
            <div key={i} className="flex items-start gap-2 py-0.5">
              <span className={`text-[12px] ${status.color} shrink-0 leading-5`}>
                {status.icon}
              </span>
              <span className={`text-[11px] leading-5 ${
                todo.status === 'completed' ? 'text-gray-600 line-through' :
                todo.status === 'in_progress' ? 'text-gray-300' :
                'text-gray-400'
              }`}>
                {todo.text || todo.content || todo.description || JSON.stringify(todo)}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
});
TodoListSection.displayName = 'TodoListSection';

/* ─── ToolCallCard (Expandable) ─── */
const ToolCallCard = memo(({ call }) => {
  const [expanded, setExpanded] = useState(false);
  const Icon = getToolIcon(call.tool_name);
  const category = getToolCategory(call.tool_name);
  const categoryStyle = CATEGORY_STYLES[category];

  return (
    <div className="border border-gray-800/40 rounded-lg overflow-hidden bg-gray-900/20 hover:bg-gray-900/40 transition-colors">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center gap-2 p-2 text-left"
      >
        <Icon className={`w-3.5 h-3.5 ${categoryStyle.text} shrink-0`} />
        <span className="text-[11px] font-mono text-gray-300 shrink-0">
          {call.tool_name}
        </span>
        <CategoryBadge category={category} />
        <span className="text-[9px] text-gray-600 ml-auto mr-2 font-mono">
          {fmtTime(call.timestamp)}
        </span>
        <ChevronDown className={`w-3 h-3 text-gray-500 transition-transform ${expanded ? 'rotate-180' : ''}`} />
      </button>

      {expanded && (
        <div className="border-t border-gray-800/40 p-3 space-y-3 bg-gray-950/50">
          {call.tool_input && (
            <JsonBlock label="Input" content={call.tool_input} />
          )}
          {call.tool_output && (
            <JsonBlock label="Output" content={call.tool_output} />
          )}
          {!call.tool_input && !call.tool_output && (
            <div className="text-[10px] text-gray-600 italic">No data available</div>
          )}
        </div>
      )}
    </div>
  );
});
ToolCallCard.displayName = 'ToolCallCard';

/* ─── ToolTimeline ─── */
const ToolTimeline = memo(({ toolCalls }) => {
  const [showAll, setShowAll] = useState(false);
  const visible = showAll ? toolCalls : toolCalls.slice(0, 8);

  return (
    <div className="relative pl-4">
      {/* Gradient timeline line */}
      <div className="absolute left-1.5 top-0 bottom-0 w-px bg-gradient-to-b from-cyan-500/50 via-violet-500/30 to-gray-800/20" />

      <div className="space-y-2">
        {visible.map((call) => {
          const category = getToolCategory(call.tool_name);
          const categoryStyle = CATEGORY_STYLES[category];
          return (
            <div key={call.id} className="relative">
              {/* Timeline dot */}
              <div className={`absolute -left-[11px] top-3 w-2 h-2 rounded-full ${categoryStyle.dot} ring-2 ring-gray-900`} />
              <ToolCallCard call={call} />
            </div>
          );
        })}
      </div>

      {toolCalls.length > 8 && !showAll && (
        <button
          onClick={() => setShowAll(true)}
          className="mt-2 ml-2 text-[10px] text-gray-500 hover:text-gray-300 transition-colors"
        >
          Show {toolCalls.length - 8} more...
        </button>
      )}
    </div>
  );
});
ToolTimeline.displayName = 'ToolTimeline';

/* ─── ToolCallsSection ─── */
const ToolCallsSection = memo(({ toolCalls }) => {
  const [expanded, setExpanded] = useState(true);

  if (toolCalls.length === 0) return null;

  return (
    <div id="tool-calls-section" data-testid="tool-calls-section" data-count={toolCalls.length} className="rounded-lg border border-gray-800/40 bg-gray-900/30">
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
          Tools
        </span>
        <span className="text-[10px] text-gray-600 font-mono ml-auto">
          {toolCalls.length}
        </span>
      </button>
      {expanded && (
        <div className="px-3 pb-3 max-h-[350px] overflow-y-auto">
          <ToolTimeline toolCalls={toolCalls} />
        </div>
      )}
    </div>
  );
});
ToolCallsSection.displayName = 'ToolCallsSection';

/* ─── MemorySection ─── */
const MemorySection = memo(({ memoryOps }) => {
  if (!memoryOps || memoryOps.length === 0) return null;

  return (
    <div className="rounded-lg border border-gray-800/40 bg-gray-900/30 p-3">
      <div className="flex items-center gap-2 mb-2">
        <Brain className="w-3.5 h-3.5 text-violet-500" />
        <span className="text-[11px] font-semibold text-gray-400 uppercase tracking-wider">
          Memory
        </span>
        <span className="text-[10px] text-gray-600 font-mono ml-auto">
          {memoryOps.length}
        </span>
      </div>
      <div className="space-y-1 max-h-[140px] overflow-y-auto">
        {memoryOps.slice(0, 10).map((op, i) => {
          const isStore = op.tool_name === 'memory_store' || (op.tool_name === 'edit_file' || op.tool_name === 'write_file');
          const isSearch = op.tool_name === 'memory_search' || op.tool_name === 'read_file';
          const Icon = isStore ? ArrowUpRight : isSearch ? ArrowDownLeft : FileText;
          const iconColor = isStore ? 'text-emerald-500' : isSearch ? 'text-cyan-500' : 'text-violet-500';
          const preview = op.type === 'call'
            ? (op.tool_input || '').slice(0, 60)
            : (op.tool_output || '').slice(0, 60);

          return (
            <div key={`${op.id}-${i}`} className="flex items-center gap-2 py-0.5">
              <Icon className={`w-3 h-3 ${iconColor} shrink-0`} />
              <span className="text-[10px] font-mono text-gray-500 shrink-0">
                {op.tool_name?.replace('memory_', '')}
              </span>
              {preview && (
                <span className="text-[10px] text-gray-600 truncate flex-1">
                  {preview}
                </span>
              )}
              <span className="text-[9px] text-gray-700 font-mono shrink-0 ml-auto">
                {fmtTime(op.timestamp)}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
});
MemorySection.displayName = 'MemorySection';

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
    <div id="trades-section" data-testid="trades-section" data-count={trades.length} className="space-y-2">
      <div className="flex items-center gap-2 px-1">
        <ArrowUpCircle className="w-3.5 h-3.5 text-amber-500" />
        <span className="text-[11px] font-semibold text-gray-400 uppercase tracking-wider">
          Trades
        </span>
        <span data-testid="trades-count" className="text-[10px] text-gray-600 font-mono ml-auto">
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

/* ─── CommandoSessionPanel ─── */
const CommandoElapsed = ({ startedAt, completedAt, active }) => {
  const [elapsed, setElapsed] = React.useState(0);

  React.useEffect(() => {
    if (!startedAt) return;
    if (!active && completedAt) {
      setElapsed(Math.round((completedAt - startedAt) / 1000));
      return;
    }
    const tick = () => setElapsed(Math.round((Date.now() - startedAt) / 1000));
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, [startedAt, completedAt, active]);

  return (
    <span className="text-[10px] font-mono text-gray-400">
      {elapsed}s
    </span>
  );
};

const CommandoSessionCard = memo(({ session }) => {
  const { active, startedAt, completedAt, error, prompt, ops } = session;

  return (
    <div className={`rounded-lg border p-3 ${
      active
        ? 'border-amber-700/40 bg-amber-950/20'
        : error
          ? 'border-red-800/30 bg-red-950/10'
          : 'border-emerald-800/30 bg-emerald-950/10'
    }`}>
      {/* Header */}
      <div className="flex items-center gap-2 mb-2">
        <Crosshair className={`w-3.5 h-3.5 ${active ? 'text-amber-400 animate-pulse' : error ? 'text-red-400' : 'text-emerald-400'}`} />
        <span className="text-[11px] font-bold text-gray-300 uppercase tracking-wider">
          Commando
        </span>
        {active && <RefreshCw className="w-3 h-3 text-amber-400 animate-spin" />}
        <span className={`ml-auto px-1.5 py-0.5 rounded text-[9px] font-bold uppercase ${
          active ? 'bg-amber-900/50 text-amber-300' : error ? 'bg-red-900/50 text-red-300' : 'bg-emerald-900/50 text-emerald-300'
        }`}>
          {active ? 'Executing' : error ? 'Error' : 'Done'}
        </span>
        <CommandoElapsed startedAt={startedAt} completedAt={completedAt} active={active} />
      </div>

      {/* Prompt */}
      {prompt && (
        <div className="text-[10px] text-gray-500 mb-2 truncate font-mono">
          {prompt.slice(0, 120)}
        </div>
      )}

      {/* Ops timeline */}
      {ops.length > 0 && (
        <div className="space-y-1 max-h-[120px] overflow-y-auto">
          {ops.map((op, i) => {
            const Icon = TOOL_ICONS[op.tool_name] || Wrench;
            const isCall = op.type === 'call';
            return (
              <div key={`${op.id}-${i}`} className="flex items-center gap-2 py-0.5">
                <Icon className={`w-3 h-3 ${isCall ? 'text-cyan-500' : 'text-gray-500'} shrink-0`} />
                <span className="text-[10px] font-mono text-gray-400 shrink-0">
                  {op.tool_name}
                </span>
                <span className={`text-[9px] truncate flex-1 ${isCall ? 'text-gray-500' : 'text-gray-600'}`}>
                  {isCall ? (op.tool_input || '').slice(0, 60) : (op.tool_output || '').slice(0, 60)}
                </span>
              </div>
            );
          })}
        </div>
      )}

      {/* Empty state */}
      {ops.length === 0 && active && (
        <div className="text-[10px] text-gray-600 flex items-center gap-2">
          <RefreshCw className="w-3 h-3 animate-spin" />
          Preparing execution...
        </div>
      )}
    </div>
  );
});
CommandoSessionCard.displayName = 'CommandoSessionCard';

const CommandoSection = memo(({ sessions }) => {
  if (!sessions || sessions.length === 0) return null;

  return (
    <div className="space-y-2">
      {sessions.map(session => (
        <CommandoSessionCard key={session.id} session={session} />
      ))}
    </div>
  );
});
CommandoSection.displayName = 'CommandoSection';

/* ─── AgentChatPanel ─── */
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
  return (
    <div
      id="agent-panel"
      data-testid="agent-panel"
      data-running={isRunning}
      className="
        bg-gradient-to-br from-gray-900/70 via-gray-900/50 to-gray-950/70
        backdrop-blur-md rounded-2xl
        border border-violet-800/30
        shadow-xl shadow-black/20
        p-5 flex flex-col gap-4
        max-h-[600px]
      "
    >
      {/* Header - stays fixed at top */}
      <div className="flex items-center justify-between shrink-0">
        <div className="flex items-center space-x-3">
          <div className="p-2 rounded-lg bg-gradient-to-br from-violet-900/40 to-violet-950/40 border border-violet-700/30">
            <Brain className={`w-4 h-4 ${isRunning ? 'text-violet-400 animate-pulse' : 'text-violet-500'}`} />
          </div>
          <div>
            <h3 className="text-sm font-bold text-gray-200 uppercase tracking-wider">Agent</h3>
            <div id="agent-status" data-testid="agent-status" className="flex items-center gap-2 mt-0.5">
              {isRunning ? (
                <>
                  <RefreshCw className="w-3 h-3 text-violet-400 animate-spin" />
                  <span data-testid="agent-status-label" className="text-[10px] text-violet-400 font-medium">
                    {currentSubagent && currentSubagent !== 'single_arb_captain'
                      ? `Subagent: ${currentSubagent}`
                      : 'Running'}
                  </span>
                </>
              ) : (
                <span data-testid="agent-status-label" className="text-[10px] text-gray-500">Idle</span>
              )}
            </div>
          </div>
        </div>
        <div className="flex items-center gap-3">
          {cycleCount > 0 && (
            <span id="agent-cycle-count" data-testid="agent-cycle-count" className="text-[10px] text-gray-500 font-mono">
              Cycle #{cycleCount}
            </span>
          )}
          <span
            id="agent-running-badge"
            data-testid="agent-running-badge"
            data-running={isRunning}
            className={`px-2 py-0.5 rounded-full text-[9px] font-bold uppercase tracking-wider ${
              isRunning
                ? 'bg-violet-900/50 text-violet-300 border border-violet-700/40'
                : 'bg-gray-800/50 text-gray-500 border border-gray-700/30'
            }`}
          >
            {isRunning ? 'Running' : 'Idle'}
          </span>
        </div>
      </div>

      {/* Scrollable content area */}
      <div className="flex-1 min-h-0 overflow-y-auto flex flex-col gap-4">
        {/* Split-view: Thinking (left) | TODOs + Tools + Memory (right) */}
        <div id="agent-split-view" data-testid="agent-split-view" className="grid grid-cols-5 gap-4" style={{ minHeight: '300px', maxHeight: '450px' }}>
          {/* Left: Thinking Stream (3 cols) */}
          <div id="thinking-container" data-testid="thinking-container" className="col-span-3 flex flex-col min-h-0">
            <ThinkingStream
              thinking={thinking}
              activeToolCall={activeToolCall}
              isRunning={isRunning}
            />
          </div>

          {/* Right: TODOs + Tools + Memory (2 cols) */}
          <div id="agent-sidebar" data-testid="agent-sidebar" className="col-span-2 flex flex-col gap-3 overflow-y-auto min-h-0">
            <TodoListSection todos={todos} />
            <ToolCallsSection toolCalls={toolCalls} />
            <MemorySection memoryOps={memoryOps} />
            {/* If no right-panel content yet, show placeholder */}
            {todos.length === 0 && toolCalls.length === 0 && memoryOps.length === 0 && (
              <div className="flex-1 flex items-center justify-center text-gray-700 text-[11px]">
                Waiting for agent activity...
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
