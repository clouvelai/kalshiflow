import React, { memo, useState, useMemo } from 'react';
import { ChevronDown, ChevronRight, Copy, Check } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { fmtTime } from '../../utils/formatters';
import { getToolCategory, getToolIcon, CATEGORY_STYLES } from '../../utils/colorMaps';

const ActivityEntry = memo(({ entry }) => {
  const [expanded, setExpanded] = useState(false);
  const Icon = getToolIcon(entry.tool_name);
  const category = getToolCategory(entry.tool_name);
  const style = CATEGORY_STYLES[category];
  const isMemory = entry.source === 'memory';

  const preview = entry.tool_output
    ? String(entry.tool_output).slice(0, 80)
    : entry.tool_input
      ? String(entry.tool_input).slice(0, 80)
      : '';

  return (
    <motion.div
      initial={{ opacity: 0, height: 0 }}
      animate={{ opacity: 1, height: 'auto' }}
      exit={{ opacity: 0, height: 0 }}
      transition={{ duration: 0.15, ease: 'easeOut' }}
      className="overflow-hidden"
    >
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center gap-1.5 px-2 py-1 text-left hover:bg-gray-800/20 transition-colors rounded"
      >
        <Icon className={`w-2.5 h-2.5 ${style.text} shrink-0 opacity-70`} />
        <span className={`px-1 py-px rounded text-[7px] font-semibold uppercase tracking-wider ${style.bg} ${style.text}`}>
          {isMemory ? 'MEM' : style.label}
        </span>
        <span className="text-[9px] font-mono text-gray-400 truncate flex-1 min-w-0">
          {entry.tool_name}
        </span>
        <span className="text-[8px] text-gray-700 font-mono tabular-nums shrink-0">
          {fmtTime(entry.timestamp)}
        </span>
        <ChevronDown className={`w-2.5 h-2.5 text-gray-600 transition-transform duration-100 shrink-0 ${expanded ? 'rotate-180' : ''}`} />
      </button>

      {expanded && (
        <div className="ml-4 mr-2 mb-1 p-2 bg-gray-950/40 rounded border border-gray-800/20">
          {entry.tool_input && (
            <div className="mb-1.5">
              <span className="text-[7px] text-gray-600 uppercase tracking-wider font-semibold">Input</span>
              <pre className="text-[9px] font-mono text-gray-400 mt-0.5 whitespace-pre-wrap break-all max-h-[100px] overflow-y-auto">
                {typeof entry.tool_input === 'string' ? entry.tool_input : JSON.stringify(entry.tool_input, null, 2)}
              </pre>
            </div>
          )}
          {entry.tool_output && (
            <div>
              <span className="text-[7px] text-gray-600 uppercase tracking-wider font-semibold">Output</span>
              <pre className="text-[9px] font-mono text-gray-500 mt-0.5 whitespace-pre-wrap break-all max-h-[100px] overflow-y-auto">
                {typeof entry.tool_output === 'string' ? entry.tool_output.slice(0, 500) : JSON.stringify(entry.tool_output, null, 2)?.slice(0, 500)}
              </pre>
            </div>
          )}
        </div>
      )}
    </motion.div>
  );
});
ActivityEntry.displayName = 'ActivityEntry';

const ActivityFeed = memo(({ toolCalls = [], memoryOps = [] }) => {
  const [showAll, setShowAll] = useState(false);

  const merged = useMemo(() => {
    const tools = toolCalls.map(t => ({ ...t, source: 'tool' }));
    const memory = memoryOps.map(m => ({ ...m, source: 'memory' }));
    return [...tools, ...memory]
      .sort((a, b) => {
        const ta = new Date(a.timestamp || 0).getTime();
        const tb = new Date(b.timestamp || 0).getTime();
        return tb - ta;
      });
  }, [toolCalls, memoryOps]);

  const visible = showAll ? merged : merged.slice(0, 12);
  const hasMore = merged.length > 12;

  if (merged.length === 0) {
    return (
      <div className="px-3 py-4 text-center">
        <span className="text-[10px] text-gray-700">No activity yet</span>
      </div>
    );
  }

  return (
    <div className="flex-1 min-h-0 flex flex-col">
      <div className="flex items-center gap-2 px-3 py-1.5 border-b border-gray-800/20 shrink-0">
        <span className="text-[9px] font-semibold text-gray-500 uppercase tracking-wider">Activity</span>
        <span className="text-[9px] font-mono text-gray-700 ml-auto tabular-nums">{merged.length}</span>
      </div>
      <div className="flex-1 overflow-y-auto min-h-0 px-1 py-0.5">
        <AnimatePresence initial={false}>
          {visible.map((entry) => (
            <ActivityEntry key={`${entry.id}-${entry.source}`} entry={entry} />
          ))}
        </AnimatePresence>
        {hasMore && !showAll && (
          <button
            onClick={() => setShowAll(true)}
            className="w-full py-1 text-[9px] text-gray-600 hover:text-gray-400 transition-colors"
          >
            +{merged.length - 12} more
          </button>
        )}
      </div>
    </div>
  );
});

ActivityFeed.displayName = 'ActivityFeed';

export default ActivityFeed;
