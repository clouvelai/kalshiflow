import React, { memo } from 'react';
import { getStateColor, getStatusColor } from '../../../utils/v3-trader';

/**
 * StateBadge - Displays a trader state badge
 */
export const StateBadge = memo(({ state }) => {
  if (!state) return null;

  return (
    <span className={`px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider ${getStateColor(state)}`}>
      {state}
    </span>
  );
});

StateBadge.displayName = 'StateBadge';

/**
 * StatusBadge - Displays a status badge (SUCCESS, FAILED, etc.)
 */
export const StatusBadge = memo(({ status }) => {
  if (!status) return null;

  return (
    <span className={`px-2 py-0.5 rounded text-xs font-bold ${getStatusColor(status)}`}>
      {status}
    </span>
  );
});

StatusBadge.displayName = 'StatusBadge';

/**
 * TransitionBadge - Displays a state transition (fromState -> toState)
 */
export const TransitionBadge = memo(({ fromState, toState }) => {
  if (!fromState || !toState) return null;

  return (
    <div className="flex items-center space-x-1 text-xs">
      <span className={`px-2 py-0.5 rounded font-bold ${getStateColor(fromState)}`}>
        {fromState}
      </span>
      <span className="text-gray-400">-&gt;</span>
      <span className={`px-2 py-0.5 rounded font-bold ${getStateColor(toState)}`}>
        {toState}
      </span>
    </div>
  );
});

TransitionBadge.displayName = 'TransitionBadge';

export default StateBadge;
