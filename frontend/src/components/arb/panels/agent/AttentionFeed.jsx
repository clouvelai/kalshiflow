import React, { memo, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Radio, ChevronDown, ChevronUp, Check, Sunrise } from 'lucide-react';
import { ATTENTION_URGENCY_STYLES, ATTENTION_CATEGORY_STYLES } from '../../utils/colorMaps';

/**
 * EarlyBirdScoreBar - Mini horizontal bar for a single score component.
 */
const EarlyBirdScoreBar = ({ label, value, max, color }) => {
  const pct = max > 0 ? Math.min((value / max) * 100, 100) : 0;
  return (
    <div className="flex items-center gap-1.5">
      <span className="text-[7px] text-gray-500 uppercase tracking-wider w-[38px] shrink-0 text-right">{label}</span>
      <div className="flex-1 h-[3px] bg-gray-800 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full ${color}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="text-[7px] font-mono text-gray-500 w-[16px] shrink-0 tabular-nums">{Math.round(value)}</span>
    </div>
  );
};

/**
 * EarlyBirdItem - Expanded attention item for early bird opportunities.
 * Shows score breakdown bars, strategy, and fair value estimate.
 */
const EarlyBirdItem = memo(({ item, style, shortTicker, ticker }) => {
  const [expanded, setExpanded] = useState(false);
  const data = item.data || {};
  const strategy = data.strategy || 'unknown';
  const fairValue = data.fair_value;
  const ebScore = data.early_bird_score;

  const strategyLabels = {
    complement: { text: 'Complement', color: 'text-emerald-400' },
    series: { text: 'Series', color: 'text-blue-400' },
    news: { text: 'News', color: 'text-orange-400' },
    captain_decide: { text: 'Captain', color: 'text-violet-400' },
    unknown: { text: 'Evaluate', color: 'text-gray-400' },
  };

  const strat = strategyLabels[strategy] || strategyLabels.unknown;

  return (
    <motion.div
      key={`${item.event_ticker}:${item.category}`}
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: -10 }}
      transition={{ duration: 0.2 }}
      className="rounded overflow-hidden"
    >
      <button
        onClick={() => setExpanded(!expanded)}
        className={`w-full flex items-center gap-2 px-2 py-1 ${style.bg} border-l-2 border-lime-500/40 hover:border-lime-500/60 transition-colors text-left`}
      >
        <Sunrise className={`w-3 h-3 ${style.text} shrink-0 ${style.pulse ? 'animate-pulse' : ''}`} />
        <span className={`text-[8px] font-bold uppercase tracking-wider shrink-0 ${style.text}`}>
          {style.label || 'EARLY BIRD'}
        </span>
        <span className="text-[9px] font-mono text-gray-300 shrink-0 truncate" title={ticker}>
          {shortTicker}
        </span>
        <span className={`text-[8px] font-semibold shrink-0 ${strat.color}`}>
          {strat.text}
        </span>
        {fairValue != null && (
          <span className="text-[9px] font-mono text-emerald-400/80 shrink-0">
            FV:{Math.round(fairValue)}c
          </span>
        )}
        <span className="flex-1" />
        {ebScore != null && (
          <span className={`text-[9px] font-mono font-semibold tabular-nums shrink-0 ${
            ebScore >= 60 ? 'text-emerald-400' : ebScore >= 40 ? 'text-amber-400' : 'text-gray-500'
          }`}>
            {Math.round(ebScore)}
          </span>
        )}
        <ChevronDown className={`w-2.5 h-2.5 text-gray-600 transition-transform duration-100 shrink-0 ${expanded ? 'rotate-180' : ''}`} />
      </button>

      {expanded && (
        <div className="px-3 py-1.5 bg-gray-900/60 border-l-2 border-lime-500/20 space-y-0.5">
          {item.summary && (
            <p className="text-[9px] text-gray-400 mb-1">{item.summary}</p>
          )}
          <EarlyBirdScoreBar label="Comp" value={data.complement_score || 0} max={25} color="bg-emerald-500" />
          <EarlyBirdScoreBar label="News" value={data.news_score || 0} max={20} color="bg-orange-500" />
          <EarlyBirdScoreBar label="Cat" value={data.category_score || 0} max={15} color="bg-blue-500" />
          <EarlyBirdScoreBar label="Time" value={data.timing_score || 0} max={10} color="bg-violet-500" />
          <EarlyBirdScoreBar label="Risk" value={data.risk_score || 0} max={10} color="bg-teal-500" />
        </div>
      )}
    </motion.div>
  );
});
EarlyBirdItem.displayName = 'EarlyBirdItem';

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
            const isEarlyBird = item.category === 'early_bird';
            const categoryStyle = ATTENTION_CATEGORY_STYLES[item.category];
            const style = categoryStyle || ATTENTION_URGENCY_STYLES[item.urgency] || ATTENTION_URGENCY_STYLES.normal;
            const isAutoHandled = !!item.data?.auto_handled;
            const ticker = item.event_ticker || '';
            const shortTicker = ticker.length > 16 ? ticker.slice(0, 16) + '...' : ticker;

            if (isEarlyBird) {
              return (
                <EarlyBirdItem
                  key={`${item.event_ticker}:${item.category}:${i}`}
                  item={item}
                  style={style}
                  shortTicker={shortTicker}
                  ticker={ticker}
                />
              );
            }

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
