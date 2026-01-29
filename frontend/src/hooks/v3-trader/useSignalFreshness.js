import { useState, useEffect, useRef, useCallback } from 'react';

/**
 * Hook that tracks new signal arrivals and freshness tiers.
 *
 * 1. New arrival detection - Compares current signal IDs against previous render.
 *    Newly seen IDs are flagged for 1.2s (entry animation window), then auto-cleared.
 *
 * 2. Freshness tier - getFreshnessTier(signal) returns:
 *    - 'fresh' if signal < 2 min old (pulsing orange glow)
 *    - 'recent' if signal < 10 min old (subtle glow)
 *    - 'normal' otherwise (no special styling)
 *
 * @param {Array} signals - Array of signal objects with .id and .timestamp (unix seconds)
 * @returns {{ newSignalIds: Set<string>, getFreshnessTier: (signal) => string }}
 */
const useSignalFreshness = (signals = []) => {
  const [newSignalIds, setNewSignalIds] = useState(new Set());
  const prevIdsRef = useRef(new Set());

  // Detect newly arrived signals
  useEffect(() => {
    const currentIds = new Set(signals.map((s) => s.id || s.signal_id));
    const arrivals = new Set();

    for (const id of currentIds) {
      if (id && !prevIdsRef.current.has(id)) {
        arrivals.add(id);
      }
    }

    prevIdsRef.current = currentIds;

    if (arrivals.size > 0) {
      setNewSignalIds((prev) => new Set([...prev, ...arrivals]));

      // Clear after 1.2s (animation duration)
      const timer = setTimeout(() => {
        setNewSignalIds((prev) => {
          const next = new Set(prev);
          for (const id of arrivals) {
            next.delete(id);
          }
          return next;
        });
      }, 1200);

      return () => clearTimeout(timer);
    }
  }, [signals]);

  // Determine freshness tier based on signal age
  const getFreshnessTier = useCallback((signal) => {
    const ts = signal?.timestamp || signal?.created_at;
    if (ts == null) return 'normal';

    const ageSec = Date.now() / 1000 - ts;
    if (ageSec < 120) return 'fresh';   // < 2 min
    if (ageSec < 600) return 'recent';  // < 10 min
    return 'normal';
  }, []);

  return { newSignalIds, getFreshnessTier };
};

export default useSignalFreshness;
