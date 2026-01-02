import { useState, useEffect, useRef, useCallback } from 'react';

/**
 * Hook for smooth number counting animation
 * Uses requestAnimationFrame for smooth 60fps animation
 *
 * @param {number} targetValue - The final value to animate to
 * @param {Object} options - Animation options
 * @param {number} options.duration - Animation duration in ms (default: 400)
 * @param {number} options.threshold - Minimum change percentage to trigger animation (default: 0.1)
 * @returns {{ displayValue: number, isAnimating: boolean }}
 */
export function useAnimatedValue(targetValue, options = {}) {
  const { duration = 400, threshold = 0.1 } = options;

  const [displayValue, setDisplayValue] = useState(targetValue);
  const [isAnimating, setIsAnimating] = useState(false);

  const animationRef = useRef(null);
  const startValueRef = useRef(targetValue);
  const startTimeRef = useRef(null);
  const previousTargetRef = useRef(targetValue);
  const isMountedRef = useRef(true);

  // Easing function: easeOutCubic for smooth deceleration
  const easeOutCubic = useCallback((t) => {
    return 1 - Math.pow(1 - t, 3);
  }, []);

  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, []);

  useEffect(() => {
    // Skip if target hasn't changed
    if (targetValue === previousTargetRef.current) {
      return;
    }

    // Calculate percentage change to skip tiny changes
    const previousValue = previousTargetRef.current;
    if (previousValue !== 0) {
      const percentChange = Math.abs((targetValue - previousValue) / previousValue) * 100;
      if (percentChange < threshold) {
        // Skip animation for tiny changes, schedule update in next frame
        previousTargetRef.current = targetValue;
        requestAnimationFrame(() => {
          if (isMountedRef.current) {
            setDisplayValue(targetValue);
          }
        });
        return;
      }
    }

    // Cancel any existing animation
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
    }

    // Start new animation
    startValueRef.current = displayValue;
    startTimeRef.current = null;
    previousTargetRef.current = targetValue;
    // Animation hooks intentionally set state on prop changes
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setIsAnimating(true);

    const animate = (timestamp) => {
      if (!isMountedRef.current) return;

      if (!startTimeRef.current) {
        startTimeRef.current = timestamp;
      }

      const elapsed = timestamp - startTimeRef.current;
      const progress = Math.min(elapsed / duration, 1);
      const easedProgress = easeOutCubic(progress);

      const currentValue = startValueRef.current +
        (targetValue - startValueRef.current) * easedProgress;

      setDisplayValue(currentValue);

      if (progress < 1) {
        animationRef.current = requestAnimationFrame(animate);
      } else {
        setDisplayValue(targetValue);
        setIsAnimating(false);
        animationRef.current = null;
      }
    };

    animationRef.current = requestAnimationFrame(animate);

    // Cleanup on effect re-run
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [targetValue, duration, threshold, easeOutCubic, displayValue]);

  return { displayValue, isAnimating };
}

export default useAnimatedValue;
