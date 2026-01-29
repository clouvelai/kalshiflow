/**
 * Formatting utilities for V3 Trader Console
 */

/**
 * Format a dollar amount as currency
 * @param {number} dollars - Amount in dollars
 * @returns {string} Formatted currency string
 */
export const formatCurrency = (dollars) => {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2
  }).format(dollars);
};

/**
 * Format cents as currency (converts to dollars first)
 * @param {number} cents - Amount in cents
 * @returns {string} Formatted currency string
 */
export const formatCentsAsCurrency = (cents) => {
  const dollars = cents / 100;
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2
  }).format(dollars);
};

/**
 * Format cents as P&L with +/- prefix
 * @param {number} cents - Amount in cents (positive or negative)
 * @returns {string} Formatted P&L string with prefix
 */
export const formatPnLCurrency = (cents) => {
  const dollars = cents / 100;
  const prefix = cents >= 0 ? '+' : '';
  return prefix + new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2
  }).format(dollars);
};

/**
 * Format cents with +/- prefix for settlement display
 * @param {number} cents - Amount in cents
 * @returns {string} Formatted string like "+$1.50" or "-$0.75"
 */
export const formatSettlementCurrency = (cents) => {
  const dollars = Math.abs(cents) / 100;
  const prefix = cents >= 0 ? '+' : '-';
  return `${prefix}$${dollars.toFixed(2)}`;
};

/**
 * Format cents as simple cents display
 * @param {number} cents - Amount in cents
 * @returns {string} Formatted string like "50c"
 */
export const formatCents = (cents) => `${cents}c`;

/**
 * Format age in seconds as human-readable string
 * @param {number} ageSeconds - Age in seconds
 * @returns {string} Formatted age string like "30s ago", "5m ago", "2h ago"
 */
export const formatAge = (ageSeconds) => {
  if (ageSeconds < 60) {
    return `${Math.floor(ageSeconds)}s ago`;
  } else if (ageSeconds < 3600) {
    return `${Math.floor(ageSeconds / 60)}m ago`;
  } else {
    return `${Math.floor(ageSeconds / 3600)}h ago`;
  }
};

/**
 * Format Unix timestamp as time string
 * @param {number} timestamp - Unix timestamp in seconds
 * @returns {string} Formatted time string like "14:30:45"
 */
export const formatTime = (timestamp) => {
  if (!timestamp) return 'N/A';
  const date = new Date(timestamp * 1000);
  return date.toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit'
  });
};

/**
 * Format ISO timestamp as relative time string
 * @param {string} isoString - ISO 8601 timestamp
 * @returns {string} Relative time like "2h 30m", "45m", "Closed"
 */
export const formatRelativeTime = (isoString) => {
  if (!isoString) return '-';
  const closeTime = new Date(isoString);
  const now = new Date();
  const diffMs = closeTime - now;

  if (diffMs <= 0) return 'Closed';

  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMins / 60);
  const diffDays = Math.floor(diffHours / 24);

  if (diffDays > 0) {
    const remainingHours = diffHours % 24;
    return remainingHours > 0 ? `${diffDays}d ${remainingHours}h` : `${diffDays}d`;
  }
  if (diffHours > 0) {
    const remainingMins = diffMins % 60;
    return remainingMins > 0 ? `${diffHours}h ${remainingMins}m` : `${diffHours}h`;
  }
  return `${diffMins}m`;
};

/**
 * Format current time as console timestamp
 * @returns {string} Time string like "14:30:45"
 */
export const formatConsoleTimestamp = () => {
  return new Date().toLocaleTimeString('en-US', {
    hour12: false,
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit'
  });
};

/**
 * Format a Unix timestamp as a relative time string.
 * Recomputes delta on each call - designed for re-evaluation on render ticks.
 * @param {number|null} unixSeconds - Unix timestamp in seconds
 * @returns {string|null} Relative time like "12s ago", "3m ago", "2h ago", or "Jan 15"
 */
export const formatRelativeTimestamp = (unixSeconds) => {
  if (unixSeconds == null) return null;
  const now = Date.now() / 1000;
  const delta = now - unixSeconds;
  if (delta < 0) return 'just now';
  if (delta < 60) return `${Math.floor(delta)}s ago`;
  if (delta < 3600) return `${Math.floor(delta / 60)}m ago`;
  if (delta < 86400) return `${Math.floor(delta / 3600)}h ago`;
  const date = new Date(unixSeconds * 1000);
  return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
};

/**
 * Compute processing latency between source creation and signal detection.
 * @param {number|null} sourceTs - Source creation Unix timestamp (seconds)
 * @param {number|null} signalTs - Signal detection Unix timestamp (seconds)
 * @returns {string|null} Latency like "45s", "3m 12s", or null if timestamps missing
 */
export const formatLatency = (sourceTs, signalTs) => {
  if (sourceTs == null || signalTs == null) return null;
  const delta = Math.max(0, signalTs - sourceTs);
  if (delta < 60) return `${Math.floor(delta)}s`;
  const mins = Math.floor(delta / 60);
  const secs = Math.floor(delta % 60);
  return secs > 0 ? `${mins}m ${secs}s` : `${mins}m`;
};
