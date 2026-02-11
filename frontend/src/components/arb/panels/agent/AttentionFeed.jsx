import React, { memo, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Radio, ChevronDown, ChevronUp, Check } from 'lucide-react';
import { ATTENTION_URGENCY_STYLES } from '../../utils/colorMaps';

/**
 * AttentionFeed - Real-time feed of attention signals from the AttentionRouter.
 *
 * Shows scored signals flowing through the system. Collapsed to 2 items by default.
 * Items are color-coded by urgency. Auto-handled items show a checkmark.
 */
const AttentionFeed = memo(({ attentionItems = [], attentionStats }) => {
  const [expanded, setExpanded] = useState(false);

  // Hide entirely when no signals and no stats - saves vertical space
  if (!attentionItems.length && !attentionStats) {
    return null;
  }

  const visibleItems = expanded ? attentionItems : attentionItems.slice(0, 2);
  const hasMore = attentionItems.length > 2;

  return (
    <div className="border-b border-gray-800/15 shrink-0">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-1 border-b border-gray-800/10">
        <div className="flex items-center gap-1.5">
          <Radio className="w-2.5 h-2.5 text-gray-500" />
          <span className="text-[9px] text-gray-500 uppercase tracking-wider font-medium">Attention</span>
        </div>
        <div className="flex items-center gap-2">
          {attentionStats && (
            <span className="text-[8px] text-gray-600 font-mono tabular-nums">
              {attentionStats.total_evaluated || 0} eval
            </span>
          )}
          <span className="text-[9px] text-gray-500 font-mono tabular-nums">
            {attentionItems.length} signal{attentionItems.length !== 1 ? 's' : ''}
          </span>
          {hasMore && (
            <button
              onClick={() => setExpanded(!expanded)}
              className="text-gray-600 hover:text-gray-400 transition-colors"
            >
              {expanded ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
            </button>
          )}
        </div>
      </div>

      {/* Items */}
      <div className="px-2 py-1 space-y-0.5">
        <AnimatePresence mode="popLayout">
          {visibleItems.map((item, i) => {
            const style = ATTENTION_URGENCY_STYLES[item.urgency] || ATTENTION_URGENCY_STYLES.normal;
            const isAutoHandled = !!item.data?.auto_handled;
            const ticker = item.event_ticker || '';
            const shortTicker = ticker.length > 16 ? ticker.slice(0, 16) + '...' : ticker;

            return (
              <motion.div
                key={`${item.event_ticker}:${item.category}:${i}`}
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -10 }}
                transition={{ duration: 0.15 }}
                className={`flex items-center gap-2 px-2 py-0.5 rounded ${style.bg}`}
              >
                <div className={`w-1.5 h-1.5 rounded-full ${style.dot} ${style.pulse ? 'animate-pulse' : ''} shrink-0`} />
                <span className={`text-[8px] font-semibold uppercase tracking-wider w-[62px] shrink-0 ${style.text}`}>
                  {item.urgency}
                </span>
                <span className="text-[9px] font-mono text-gray-400 shrink-0 w-[110px] truncate" title={ticker}>
                  {shortTicker}
                </span>
                <span className="text-[9px] text-gray-500 truncate flex-1" title={item.summary}>
                  {item.summary}
                </span>
                {isAutoHandled && (
                  <div className="flex items-center gap-0.5 shrink-0">
                    <Check className="w-2.5 h-2.5 text-emerald-500" />
                    <span className="text-[8px] text-emerald-500/80">auto</span>
                  </div>
                )}
                <span className="text-[8px] text-gray-600 font-mono tabular-nums shrink-0">
                  {Math.round(item.score)}
                </span>
              </motion.div>
            );
          })}
        </AnimatePresence>
      </div>
    </div>
  );
});

AttentionFeed.displayName = 'AttentionFeed';

export default AttentionFeed;
