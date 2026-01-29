import { useState, useEffect } from 'react';

/**
 * Hook that increments a tick counter at a given interval.
 * Any component calling this hook re-renders on each tick,
 * causing inline formatRelativeTimestamp() calls to produce fresh strings.
 *
 * @param {number} intervalMs - Tick interval in milliseconds (default 30000 = 30s)
 * @returns {number} Current tick count (for dependency tracking)
 */
const useRelativeTime = (intervalMs = 30000) => {
  const [tick, setTick] = useState(0);

  useEffect(() => {
    const id = setInterval(() => {
      setTick((t) => t + 1);
    }, intervalMs);
    return () => clearInterval(id);
  }, [intervalMs]);

  return tick;
};

export default useRelativeTime;
