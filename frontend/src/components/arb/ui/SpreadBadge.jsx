import React, { memo } from 'react';

/**
 * SpreadBadge - Color-coded spread indicator
 *
 * Green  : spread >= highThreshold (profitable)
 * Yellow : spread >= lowThreshold and < highThreshold (near-actionable)
 * Gray   : spread < lowThreshold (not actionable)
 */
const SpreadBadge = ({ spreadCents, highThreshold = 5, lowThreshold = 3 }) => {
  const absSpread = Math.abs(spreadCents ?? 0);

  let colorClasses;
  if (absSpread >= highThreshold) {
    colorClasses = 'bg-emerald-900/40 text-emerald-400 border-emerald-600/40';
  } else if (absSpread >= lowThreshold) {
    colorClasses = 'bg-amber-900/40 text-amber-400 border-amber-600/40';
  } else {
    colorClasses = 'bg-gray-800/40 text-gray-500 border-gray-700/40';
  }

  const sign = (spreadCents ?? 0) >= 0 ? '+' : '-';

  return (
    <span
      className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-mono font-bold border ${colorClasses}`}
    >
      {sign}{absSpread}c
    </span>
  );
};

export default memo(SpreadBadge);
