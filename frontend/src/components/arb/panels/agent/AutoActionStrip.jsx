import React, { memo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { AlertTriangle, Clock, Shield } from 'lucide-react';
import { AUTO_ACTION_STYLES } from '../../utils/colorMaps';

const ACTION_ICONS = {
  stop_loss: AlertTriangle,
  time_exit: Clock,
  regime_gate: Shield,
};

/**
 * AutoActionStrip - Compact bar showing recent auto-actions.
 *
 * Shows last 3 auto-actions in a horizontal layout. Hidden when empty.
 */
const AutoActionStrip = memo(({ autoActions = [] }) => {
  if (!autoActions.length) return null;

  const recent = autoActions.slice(0, 3);

  return (
    <div className="flex items-center gap-2 px-3 py-1 border-b border-gray-800/15 shrink-0 overflow-x-auto">
      <AnimatePresence mode="popLayout">
        {recent.map((action, i) => {
          const style = AUTO_ACTION_STYLES[action.action] || AUTO_ACTION_STYLES.stop_loss;
          const Icon = ACTION_ICONS[action.action] || AlertTriangle;
          const ticker = action.ticker || '';
          const shortTicker = ticker.length > 14 ? ticker.slice(0, 14) + '...' : ticker;

          return (
            <motion.div
              key={`${action.action}:${action.ticker}:${i}`}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              transition={{ duration: 0.15 }}
              className={`flex items-center gap-1.5 px-2 py-0.5 rounded ${style.bg} shrink-0`}
            >
              <Icon className={`w-2.5 h-2.5 ${style.text}`} />
              <span className={`text-[8px] font-semibold uppercase tracking-wider ${style.text}`}>
                {style.label}
              </span>
              <span className="text-[9px] font-mono text-gray-400 truncate max-w-[100px]" title={ticker}>
                {shortTicker}
              </span>
            </motion.div>
          );
        })}
      </AnimatePresence>
    </div>
  );
});

AutoActionStrip.displayName = 'AutoActionStrip';

export default AutoActionStrip;
