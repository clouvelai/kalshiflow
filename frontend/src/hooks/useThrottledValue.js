import { useState, useEffect, useRef } from 'react';

/**
 * useThrottledValue - Throttle rapid value updates
 *
 * Limits how frequently a value can change in the UI to prevent
 * flickering and visual chaos from rapid updates (e.g., real-time prices).
 *
 * @param {any} value - The value to throttle
 * @param {number} throttleMs - Minimum interval between updates (default 500ms)
 * @returns {any} - The throttled value
 *
 * @example
 * // Throttle price updates to max 2 per second
 * const throttledPrice = useThrottledValue(livePrice, 500);
 */
export function useThrottledValue(value, throttleMs = 500) {
  const [throttledValue, setThrottledValue] = useState(value);
  const lastUpdateRef = useRef(0);
  const pendingValueRef = useRef(value);
  const timeoutRef = useRef(null);
  const isFirstRenderRef = useRef(true);

  useEffect(() => {
    // Always update immediately on first render
    if (isFirstRenderRef.current) {
      isFirstRenderRef.current = false;
      setThrottledValue(value);
      lastUpdateRef.current = Date.now();
      return;
    }

    pendingValueRef.current = value;
    const now = Date.now();
    const timeSinceLastUpdate = now - lastUpdateRef.current;

    if (timeSinceLastUpdate >= throttleMs) {
      // Enough time has passed, update immediately
      setThrottledValue(value);
      lastUpdateRef.current = now;
    } else {
      // Schedule update for remaining time
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }

      timeoutRef.current = setTimeout(() => {
        setThrottledValue(pendingValueRef.current);
        lastUpdateRef.current = Date.now();
      }, throttleMs - timeSinceLastUpdate);
    }

    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, [value, throttleMs]);

  return throttledValue;
}

export default useThrottledValue;
