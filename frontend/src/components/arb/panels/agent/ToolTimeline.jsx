import React, { memo, useState, useMemo } from 'react';
import { ChevronDown, ChevronRight, Wrench, Copy, Check } from 'lucide-react';
import { fmtTime } from '../../utils/formatters';
import { getToolCategory, getToolIcon, CATEGORY_STYLES } from '../../utils/colorMaps';

/**
 * CategoryBadge - Compact category label.
 */
const CategoryBadge = ({ category }) => {
  const style = CATEGORY_STYLES[category] || CATEGORY_STYLES.other;
  return (
    <span className={`px-1.5 py-px rounded text-[8px] font-semibold uppercase tracking-wider ${style.bg} ${style.text}`}>
      {style.label}
    </span>
  );
};

/**
 * Highlight JSON keys, strings, booleans, numbers.
 */
const highlightJson = (str) => {
  if (!str) return str;
  return str.replace(
    /("(?:[^"\\]|\\.)*")\s*:/g,
    '<span class="text-violet-400/80">$1</span>:'
  ).replace(
    /:\s*("(?:[^"\\]|\\.)*")/g,
    ': <span class="text-emerald-400/80">$1</span>'
  ).replace(
    /:\s*(true|false|null)/g,
    ': <span class="text-amber-400/80">$1</span>'
  ).replace(
    /:\s*(-?\d+\.?\d*)/g,
    ': <span class="text-cyan-400/80">$1</span>'
  );
};

/**
 * JsonBlock - Formatted JSON with copy.
 */
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
    setTimeout(() => setCopied(false), 1500);
  };

  return (
    <div>
      <div className="flex items-center justify-between mb-1">
        <span className="text-[8px] text-gray-500 uppercase tracking-wider font-semibold">{label}</span>
        <button
          onClick={handleCopy}
          className="p-0.5 hover:bg-gray-700/40 rounded transition-colors"
          title="Copy"
        >
          {copied
            ? <Check className="w-2.5 h-2.5 text-emerald-400" />
            : <Copy className="w-2.5 h-2.5 text-gray-600 hover:text-gray-400" />
          }
        </button>
      </div>
      <pre
        className="text-[10px] font-mono bg-gray-950/50 rounded p-2 overflow-x-auto max-h-[180px] overflow-y-auto text-gray-400 leading-relaxed border border-gray-800/20"
        dangerouslySetInnerHTML={{ __html: highlightJson(formatted) }}
      />
    </div>
  );
};

/**
 * ToolCallCard - Expandable card for a single tool call.
 */
const ToolCallCard = memo(({ call }) => {
  const [expanded, setExpanded] = useState(false);
  const Icon = getToolIcon(call.tool_name);
  const category = getToolCategory(call.tool_name);
  const categoryStyle = CATEGORY_STYLES[category];

  return (
    <div className="rounded-lg overflow-hidden bg-gray-900/20 border border-gray-800/25 hover:border-gray-700/30 transition-colors">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center gap-2 px-2.5 py-1.5 text-left"
      >
        <Icon className={`w-3 h-3 ${categoryStyle.text} shrink-0 opacity-70`} />
        <span className="text-[10px] font-mono text-gray-300 shrink-0">
          {call.tool_name}
        </span>
        <CategoryBadge category={category} />
        <span className="text-[9px] text-gray-600 ml-auto mr-1.5 font-mono tabular-nums">
          {fmtTime(call.timestamp)}
        </span>
        <ChevronDown className={`w-2.5 h-2.5 text-gray-600 transition-transform duration-150 ${expanded ? 'rotate-180' : ''}`} />
      </button>

      {expanded && (
        <div className="border-t border-gray-800/25 p-2.5 space-y-2.5 bg-gray-950/30">
          {call.tool_input && <JsonBlock label="Input" content={call.tool_input} />}
          {call.tool_output && <JsonBlock label="Output" content={call.tool_output} />}
          {!call.tool_input && !call.tool_output && (
            <span className="text-[10px] text-gray-600 italic">No data</span>
          )}
        </div>
      )}
    </div>
  );
});
ToolCallCard.displayName = 'ToolCallCard';

/**
 * ToolTimeline - Vertical timeline with category-colored dots.
 */
const ToolTimeline = memo(({ toolCalls }) => {
  const [showAll, setShowAll] = useState(false);
  const visible = showAll ? toolCalls : toolCalls.slice(0, 8);

  return (
    <div className="relative pl-3.5">
      {/* Timeline line */}
      <div className="absolute left-[5px] top-0 bottom-0 w-px bg-gradient-to-b from-violet-500/30 via-gray-700/20 to-transparent" />

      <div className="space-y-1.5">
        {visible.map((call) => {
          const category = getToolCategory(call.tool_name);
          const categoryStyle = CATEGORY_STYLES[category];
          return (
            <div key={call.id} className="relative">
              <div className={`absolute -left-[9px] top-2.5 w-1.5 h-1.5 rounded-full ${categoryStyle.dot} ring-2 ring-gray-900/80`} />
              <ToolCallCard call={call} />
            </div>
          );
        })}
      </div>

      {toolCalls.length > 8 && !showAll && (
        <button
          onClick={() => setShowAll(true)}
          className="mt-1.5 ml-1 text-[10px] text-gray-500 hover:text-gray-300 transition-colors"
        >
          +{toolCalls.length - 8} more
        </button>
      )}
    </div>
  );
});
ToolTimeline.displayName = 'ToolTimeline';

/**
 * ToolCallsSection - Collapsible tool timeline.
 */
const ToolCallsSection = memo(({ toolCalls }) => {
  const [expanded, setExpanded] = useState(true);

  if (toolCalls.length === 0) return null;

  return (
    <div id="tool-calls-section" data-testid="tool-calls-section" data-count={toolCalls.length} className="rounded-lg border border-gray-800/30 bg-gray-900/25">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center gap-2 px-3 py-2 text-left hover:bg-gray-800/20 transition-colors rounded-lg"
      >
        {expanded
          ? <ChevronDown className="w-3 h-3 text-gray-500" />
          : <ChevronRight className="w-3 h-3 text-gray-500" />
        }
        <Wrench className="w-3 h-3 text-cyan-500/70" />
        <span className="text-[10px] font-semibold text-gray-400 uppercase tracking-wider">
          Tools
        </span>
        <span className="text-[10px] text-gray-600 font-mono tabular-nums ml-auto">
          {toolCalls.length}
        </span>
      </button>
      {expanded && (
        <div className="px-3 pb-2.5 max-h-[320px] overflow-y-auto">
          <ToolTimeline toolCalls={toolCalls} />
        </div>
      )}
    </div>
  );
});
ToolCallsSection.displayName = 'ToolCallsSection';

export { ToolTimeline, ToolCallCard, ToolCallsSection, CategoryBadge, JsonBlock };
export default ToolCallsSection;
