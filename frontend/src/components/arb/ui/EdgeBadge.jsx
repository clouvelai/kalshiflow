import React, { memo } from 'react';

/**
 * EdgeBadge - Color-coded edge indicator for single-event arb
 *
 * Green  : edge >= highThreshold (profitable after fees)
 * Amber  : edge >= lowThreshold and < highThreshold (marginal)
 * Gray   : edge < lowThreshold (not actionable)
 */
const EdgeBadge = ({ edgeCents, direction, highThreshold = 3, lowThreshold = 1 }) => {
  const absEdge = Math.abs(edgeCents ?? 0);

  let colorClasses;
  if (absEdge >= highThreshold) {
    colorClasses = 'bg-emerald-500/10 text-emerald-400 border-emerald-500/15';
  } else if (absEdge >= lowThreshold) {
    colorClasses = 'bg-amber-500/10 text-amber-400 border-amber-500/15';
  } else {
    colorClasses = 'bg-gray-800/25 text-gray-500 border-gray-700/20';
  }

  const label = direction === 'long' ? 'L' : direction === 'short' ? 'S' : '';
  const sign = (edgeCents ?? 0) >= 0 ? '+' : '-';

  return (
    <span
      className={`inline-flex items-center px-2 py-0.5 rounded text-[10px] font-mono font-semibold border tabular-nums ${colorClasses}`}
      title={`${direction || ''} edge: ${sign}${absEdge.toFixed(1)}c after fees`}
    >
      {label && <span className="mr-0.5 text-[9px] opacity-70">{label}</span>}
      {sign}{absEdge.toFixed(0)}c
    </span>
  );
};

export default memo(EdgeBadge);
