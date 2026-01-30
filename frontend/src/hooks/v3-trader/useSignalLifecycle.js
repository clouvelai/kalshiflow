import { useState, useCallback, useMemo } from 'react';

/**
 * useSignalLifecycle - Manages signal evaluation lifecycle state
 *
 * Tracks each signal's evaluation progress:
 * - new: Just arrived, not yet evaluated
 * - evaluating: Agent has seen it, evaluation in progress (1/3, 2/3, etc.)
 * - traded: Agent executed a trade on this signal
 * - passed: Agent decided not to trade
 * - expired: Hit max evaluations without trading
 * - historical: Pre-restart signal, never evaluated this session
 */
export const useSignalLifecycle = () => {
  // Map<signalId, lifecycleState>
  const [lifecycleMap, setLifecycleMap] = useState(new Map());

  /**
   * Handle a single signal_lifecycle_update message
   */
  const handleLifecycleUpdate = useCallback((data) => {
    if (!data?.signal_id) return;

    setLifecycleMap(prev => {
      const next = new Map(prev);
      next.set(data.signal_id, {
        signalId: data.signal_id,
        status: data.status,
        evaluationCount: data.evaluation_count || 0,
        maxEvaluations: data.max_evaluations || 3,
        latestDecision: data.latest_decision || null,
        evaluations: (data.evaluations || []).map(e => ({
          cycle: e.cycle,
          decision: e.decision,
          reason: e.reason,
          timestamp: e.timestamp,
        })),
        isHistorical: data.is_historical || false,
      });
      return next;
    });
  }, []);

  /**
   * Handle snapshot restore: populate lifecycle map from array of signal states
   */
  const handleLifecycleSnapshot = useCallback((signals) => {
    if (!Array.isArray(signals) || signals.length === 0) return;

    setLifecycleMap(() => {
      const next = new Map();
      for (const s of signals) {
        if (!s.signal_id) continue;
        next.set(s.signal_id, {
          signalId: s.signal_id,
          status: s.status,
          evaluationCount: s.evaluation_count || 0,
          maxEvaluations: s.max_evaluations || 3,
          latestDecision: s.latest_decision || null,
          evaluations: (s.evaluations || []).map(e => ({
            cycle: e.cycle,
            decision: e.decision,
            reason: e.reason,
            timestamp: e.timestamp,
          })),
          isHistorical: s.is_historical || false,
        });
      }
      return next;
    });
  }, []);

  /**
   * Get lifecycle state for a specific signal
   */
  const getLifecycle = useCallback((signalId) => {
    return lifecycleMap.get(signalId) || null;
  }, [lifecycleMap]);

  /**
   * Get summary counts for section header
   */
  const summaryCounts = useMemo(() => {
    const counts = {
      new: 0,
      evaluating: 0,
      traded: 0,
      passed: 0,
      expired: 0,
      historical: 0,
    };
    for (const s of lifecycleMap.values()) {
      if (s.status in counts) {
        counts[s.status]++;
      }
    }
    return counts;
  }, [lifecycleMap]);

  /**
   * Sort priority for signal statuses (lower = higher priority = shown first)
   */
  const getStatusSortPriority = useCallback((signalId) => {
    const lifecycle = lifecycleMap.get(signalId);
    if (!lifecycle) return 1; // Unknown signals get "new" priority
    const priorities = {
      evaluating: 0,
      new: 1,
      traded: 2,
      passed: 3,
      expired: 4,
      historical: 5,
    };
    return priorities[lifecycle.status] ?? 3;
  }, [lifecycleMap]);

  return {
    lifecycleMap,
    handleLifecycleUpdate,
    handleLifecycleSnapshot,
    getLifecycle,
    summaryCounts,
    getStatusSortPriority,
  };
};

export default useSignalLifecycle;
