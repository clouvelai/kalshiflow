/**
 * formatters.js - Shared formatting utilities for the arb dashboard.
 *
 * Consolidates time, volume, and other formatting functions used across components.
 */

/**
 * Format a timestamp to HH:MM:SS time string.
 * @param {number|string|Date} ts - Timestamp (ms, s, ISO string, or Date)
 * @returns {string} Formatted time string
 */
export const fmtTime = (ts) => {
  if (!ts) return '';
  try {
    // Handle seconds vs milliseconds
    const timestamp = typeof ts === 'number' && ts < 1e12 ? ts * 1000 : ts;
    const d = new Date(timestamp);
    return d.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    });
  } catch {
    return '';
  }
};

/**
 * Format volume with appropriate suffix (K, M, B).
 * @param {number} vol - Volume value
 * @param {number} decimals - Decimal places (default: 1)
 * @returns {string} Formatted volume string
 */
export const formatVol = (vol, decimals = 1) => {
  if (vol == null) return '--';
  if (vol >= 1_000_000_000) return (vol / 1_000_000_000).toFixed(decimals) + 'B';
  if (vol >= 1_000_000) return (vol / 1_000_000).toFixed(decimals) + 'M';
  if (vol >= 1_000) return (vol / 1_000).toFixed(decimals) + 'K';
  return vol.toFixed(0);
};

/**
 * Get freshness display for cache/data age.
 * @param {number} ms - Age in milliseconds
 * @returns {{ label: string, color: string }} Display info
 */
export const getFreshnessDisplay = (ms) => {
  if (ms == null) return { label: '--', color: 'text-gray-600' };
  const secs = Math.floor(ms / 1000);
  if (secs < 60) return { label: `${secs}s`, color: 'text-emerald-400' };
  const mins = Math.floor(secs / 60);
  if (mins < 5) return { label: `${mins}m`, color: 'text-amber-400' };
  return { label: `${mins}m`, color: 'text-red-400' };
};

/**
 * Truncate text with ellipsis.
 * @param {string} text - Text to truncate
 * @param {number} maxLen - Maximum length (default: 50)
 * @returns {string} Truncated text
 */
export const truncate = (text, maxLen = 50) => {
  if (!text || text.length <= maxLen) return text || '';
  return text.substring(0, maxLen - 3) + '...';
};
