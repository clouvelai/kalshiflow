import { useState, useEffect, useRef, useCallback } from 'react';

/**
 * Hook that returns isPulsing boolean when version/data changes
 * Pulse clears after a configurable duration
 *
 * @param {number} version - Version field that triggers pulse on change
 * @param {Object} options - Configuration options
 * @param {number} options.duration - Pulse duration in ms (default: 500)
 * @returns {boolean} - Whether pulse animation is active
 */
export function useDataUpdatePulse(version, options = {}) {
  const { duration = 500 } = options;

  const [isPulsing, setIsPulsing] = useState(false);
  const previousVersionRef = useRef(version);
  const timeoutRef = useRef(null);
  const isFirstRender = useRef(true);

  // Memoized pulse trigger function
  const triggerPulse = useCallback(() => {
    // Clear any existing timeout
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }

    // Start pulse
    setIsPulsing(true);

    // Clear pulse after duration
    timeoutRef.current = setTimeout(() => {
      setIsPulsing(false);
    }, duration);
  }, [duration]);

  useEffect(() => {
    // Skip pulse on first render
    if (isFirstRender.current) {
      isFirstRender.current = false;
      previousVersionRef.current = version;
      return;
    }

    // Only pulse if version changed
    if (version === previousVersionRef.current) {
      return;
    }

    previousVersionRef.current = version;
    // Pulse hooks intentionally trigger state updates on version changes
    // eslint-disable-next-line react-hooks/set-state-in-effect
    triggerPulse();

    // Cleanup timeout on unmount
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, [version, triggerPulse]);

  return isPulsing;
}

export default useDataUpdatePulse;
