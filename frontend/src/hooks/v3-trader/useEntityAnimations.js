import { useState, useEffect, useRef, useMemo, useCallback } from 'react';

/**
 * useEntityAnimations - Hook for managing entity animation states
 *
 * Detects changes in entity index and returns animation states:
 * - newEntityIds: Set of IDs for newly discovered entities (animate in)
 * - updatedEntityIds: Map of ID → sentiment delta for updates (pulse)
 * - recentActivityIds: Set of IDs with activity in last 60s (breathing glow)
 */
export const useEntityAnimations = ({ entityIndex, entitySystemActive }) => {
  // Track previous entity state for change detection
  const prevEntitiesRef = useRef(new Map());
  const prevSentimentsRef = useRef(new Map());

  // Animation states
  const [newEntityIds, setNewEntityIds] = useState(new Set());
  const [updatedEntityIds, setUpdatedEntityIds] = useState(new Map());
  const [recentActivityIds, setRecentActivityIds] = useState(new Set());

  // Timeouts for animation cleanup
  const newEntityTimeoutsRef = useRef(new Map());
  const updateTimeoutsRef = useRef(new Map());

  // Track activity timestamps for recent activity detection
  const activityTimestampsRef = useRef(new Map());

  /**
   * Calculate aggregate sentiment for an entity from reddit_signals
   */
  const getEntitySentiment = useCallback((entity) => {
    if (!entity.reddit_signals) return 0;

    const signals = entity.reddit_signals;
    if (typeof signals === 'object' && signals.aggregate_sentiment !== undefined) {
      return signals.aggregate_sentiment;
    }
    if (typeof signals === 'number') {
      return signals;
    }
    return 0;
  }, []);

  /**
   * Detect new entities and sentiment updates
   */
  useEffect(() => {
    if (!entityIndex?.entities?.length) {
      prevEntitiesRef.current.clear();
      prevSentimentsRef.current.clear();
      return;
    }

    const currentEntities = new Map();
    const currentSentiments = new Map();
    const newIds = new Set();
    const updatedIds = new Map();

    entityIndex.entities.forEach(entity => {
      const id = entity.entity_id;
      currentEntities.set(id, entity);

      const sentiment = getEntitySentiment(entity);
      currentSentiments.set(id, sentiment);

      // Check if this is a new entity
      if (!prevEntitiesRef.current.has(id)) {
        newIds.add(id);
        activityTimestampsRef.current.set(id, Date.now());
      }
      // Check if sentiment changed
      else if (prevSentimentsRef.current.has(id)) {
        const prevSentiment = prevSentimentsRef.current.get(id);
        if (prevSentiment !== sentiment) {
          const delta = sentiment - prevSentiment;
          updatedIds.set(id, delta);
          activityTimestampsRef.current.set(id, Date.now());
        }
      }
    });

    // Batch update new entity animations
    if (newIds.size > 0) {
      setNewEntityIds(prev => {
        const next = new Set(prev);
        newIds.forEach(id => next.add(id));
        return next;
      });

      // Schedule cleanup after animation duration (800ms + buffer)
      newIds.forEach(id => {
        if (newEntityTimeoutsRef.current.has(id)) {
          clearTimeout(newEntityTimeoutsRef.current.get(id));
        }
        const timeout = setTimeout(() => {
          setNewEntityIds(prev => {
            const next = new Set(prev);
            next.delete(id);
            return next;
          });
          newEntityTimeoutsRef.current.delete(id);
        }, 1200);
        newEntityTimeoutsRef.current.set(id, timeout);
      });
    }

    // Batch update sentiment animations
    if (updatedIds.size > 0) {
      setUpdatedEntityIds(prev => {
        const next = new Map(prev);
        updatedIds.forEach((delta, id) => next.set(id, delta));
        return next;
      });

      // Schedule cleanup after animation duration (600ms + buffer)
      updatedIds.forEach((_, id) => {
        if (updateTimeoutsRef.current.has(id)) {
          clearTimeout(updateTimeoutsRef.current.get(id));
        }
        const timeout = setTimeout(() => {
          setUpdatedEntityIds(prev => {
            const next = new Map(prev);
            next.delete(id);
            return next;
          });
          updateTimeoutsRef.current.delete(id);
        }, 800);
        updateTimeoutsRef.current.set(id, timeout);
      });
    }

    // Update refs for next comparison
    prevEntitiesRef.current = currentEntities;
    prevSentimentsRef.current = currentSentiments;
  }, [entityIndex?.entities, getEntitySentiment]);

  /**
   * Update recent activity set every 10 seconds
   * Entities with activity in last 60s get breathing glow
   */
  useEffect(() => {
    const updateRecentActivity = () => {
      const now = Date.now();
      const recentThreshold = 60000; // 60 seconds

      const recent = new Set();
      activityTimestampsRef.current.forEach((timestamp, id) => {
        if (now - timestamp < recentThreshold) {
          recent.add(id);
        } else {
          activityTimestampsRef.current.delete(id);
        }
      });

      setRecentActivityIds(recent);
    };

    // Update immediately and then every 10 seconds
    updateRecentActivity();
    const interval = setInterval(updateRecentActivity, 10000);

    return () => clearInterval(interval);
  }, []);

  /**
   * Cleanup timeouts on unmount
   */
  useEffect(() => {
    return () => {
      newEntityTimeoutsRef.current.forEach(timeout => clearTimeout(timeout));
      updateTimeoutsRef.current.forEach(timeout => clearTimeout(timeout));
    };
  }, []);

  /**
   * Get animation class names for an entity
   */
  const getAnimationClasses = useCallback((entityId) => {
    const classes = [];

    if (newEntityIds.has(entityId)) {
      classes.push('animate-entity-discovered');
    }

    if (updatedEntityIds.has(entityId)) {
      const delta = updatedEntityIds.get(entityId);
      if (delta > 0) {
        classes.push('animate-sentiment-positive');
      } else {
        classes.push('animate-sentiment-negative');
      }
    }

    if (recentActivityIds.has(entityId) && entitySystemActive) {
      classes.push('animate-recent-activity');
    }

    return classes.join(' ');
  }, [newEntityIds, updatedEntityIds, recentActivityIds, entitySystemActive]);

  /**
   * Check if entity has any active animation
   */
  const hasActiveAnimation = useCallback((entityId) => {
    return newEntityIds.has(entityId) || updatedEntityIds.has(entityId);
  }, [newEntityIds, updatedEntityIds]);

  return {
    newEntityIds,
    updatedEntityIds,
    recentActivityIds,
    getAnimationClasses,
    hasActiveAnimation,
    getEntitySentiment,
  };
};

export default useEntityAnimations;
