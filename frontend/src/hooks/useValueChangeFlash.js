import { useState, useEffect, useRef } from 'react';

/**
 * Hook that returns flash state ('increase' | 'decrease' | null) on value change
 * Flash clears after a configurable duration
 *
 * @param {number} value - The value to watch for changes
 * @param {Object} options - Configuration options
 * @param {number} options.duration - Flash duration in ms (default: 800)
 * @param {number} options.version - Optional version field for change detection
 * @returns {'increase' | 'decrease' | null} - Flash state
 */
export function useValueChangeFlash(value, options = {}) {
  const { duration = 800, version } = options;

  const [flashState, setFlashState] = useState(null);
  const previousValueRef = useRef(value);
  const previousVersionRef = useRef(version);
  const timeoutRef = useRef(null);
  const isFirstRender = useRef(true);

  useEffect(() => {
    // Skip flash on first render
    if (isFirstRender.current) {
      isFirstRender.current = false;
      previousValueRef.current = value;
      previousVersionRef.current = version;
      return;
    }

    // Determine if we should check for changes
    // If version is provided, only flash if version changed
    const shouldCheck = version !== undefined
      ? version !== previousVersionRef.current
      : value !== previousValueRef.current;

    if (!shouldCheck) {
      return;
    }

    // Clear any existing timeout
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }

    // Determine flash direction
    // Flash hooks intentionally set state on value changes
    const previousValue = previousValueRef.current;
    if (value > previousValue) {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setFlashState('increase');
    } else if (value < previousValue) {
      setFlashState('decrease');
    }

    // Update refs
    previousValueRef.current = value;
    previousVersionRef.current = version;

    // Clear flash after duration
    timeoutRef.current = setTimeout(() => {
      setFlashState(null);
    }, duration);

    // Cleanup timeout on unmount
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, [value, version, duration]);

  return flashState;
}

export default useValueChangeFlash;
