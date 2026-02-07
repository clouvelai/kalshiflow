import React, { memo } from 'react';
import { Brain, ArrowUpRight, ArrowDownLeft, FileText } from 'lucide-react';
import { fmtTime } from '../../utils/formatters';

/**
 * MemorySection - Memory operations (store, search, file ops).
 */
const MemorySection = memo(({ memoryOps }) => {
  if (!memoryOps || memoryOps.length === 0) return null;

  return (
    <div className="rounded-lg border border-gray-800/30 bg-gray-900/25 p-2.5">
      <div className="flex items-center gap-2 mb-2">
        <Brain className="w-3 h-3 text-violet-500/70" />
        <span className="text-[10px] font-semibold text-gray-400 uppercase tracking-wider">
          Memory
        </span>
        <span className="text-[10px] text-gray-600 font-mono tabular-nums ml-auto">
          {memoryOps.length}
        </span>
      </div>
      <div className="space-y-0.5 max-h-[130px] overflow-y-auto">
        {memoryOps.slice(0, 10).map((op, i) => {
          const isStore = op.tool_name === 'memory_store' || op.tool_name === 'edit_file' || op.tool_name === 'write_file';
          const isSearch = op.tool_name === 'memory_search' || op.tool_name === 'read_file';
          const Icon = isStore ? ArrowUpRight : isSearch ? ArrowDownLeft : FileText;
          const iconColor = isStore ? 'text-emerald-500/60' : isSearch ? 'text-cyan-500/60' : 'text-violet-500/60';
          const preview = op.type === 'call'
            ? (op.tool_input || '').slice(0, 50)
            : (op.tool_output || '').slice(0, 50);

          return (
            <div key={`${op.id}-${i}`} className="flex items-center gap-2 py-0.5">
              <Icon className={`w-2.5 h-2.5 ${iconColor} shrink-0`} />
              <span className="text-[10px] font-mono text-gray-500 shrink-0">
                {op.tool_name?.replace('memory_', '')}
              </span>
              {preview && (
                <span className="text-[9px] text-gray-600 truncate flex-1">
                  {preview}
                </span>
              )}
              <span className="text-[9px] text-gray-700 font-mono tabular-nums shrink-0 ml-auto">
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

export default MemorySection;
