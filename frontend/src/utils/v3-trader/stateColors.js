/**
 * State and status color mappings for V3 Trader Console
 */

/**
 * Color classes for trader states
 */
export const STATE_COLORS = {
  startup: 'text-gray-400 bg-gray-900/50',
  initializing: 'text-yellow-400 bg-yellow-900/30',
  orderbook_connect: 'text-blue-400 bg-blue-900/30',
  trading_client_connect: 'text-purple-400 bg-purple-900/30',
  kalshi_data_sync: 'text-cyan-400 bg-cyan-900/30',
  ready: 'text-green-400 bg-green-900/30',
  error: 'text-red-400 bg-red-900/30',
  shutdown: 'text-gray-500 bg-gray-900/50'
};

/**
 * Color classes for status badges
 */
export const STATUS_COLORS = {
  SUCCESS: 'text-green-400 bg-green-900/30',
  FAILED: 'text-red-400 bg-red-900/30',
  ERROR: 'text-red-400 bg-red-900/30',
  READY: 'text-green-400 bg-green-900/30',
  INITIALIZING: 'text-yellow-400 bg-yellow-900/30',
  CONNECTING: 'text-blue-400 bg-blue-900/30',
  CALIBRATING: 'text-purple-400 bg-purple-900/30'
};

/**
 * Color classes for decision actions in audit panel
 */
export const ACTION_COLORS = {
  followed: 'bg-green-900/30 text-green-400 border-green-700/50',
  skipped_age: 'bg-yellow-900/30 text-yellow-400 border-yellow-700/50',
  skipped_position: 'bg-purple-900/30 text-purple-400 border-purple-700/50',
  skipped_orders: 'bg-blue-900/30 text-blue-400 border-blue-700/50',
  already_followed: 'bg-gray-900/30 text-gray-400 border-gray-700/50',
  rate_limited: 'bg-red-900/30 text-red-400 border-red-700/50'
};

/**
 * Labels for decision actions
 */
export const ACTION_LABELS = {
  followed: 'FOLLOWED',
  skipped_age: 'TOO OLD',
  skipped_position: 'HAS POSITION',
  skipped_orders: 'HAS ORDERS',
  already_followed: 'ALREADY DONE',
  rate_limited: 'RATE LIMITED'
};

/**
 * Get state color classes
 * @param {string} state - The trader state
 * @returns {string} Tailwind CSS classes for the state
 */
export const getStateColor = (state) => {
  const stateKey = state?.toLowerCase();
  return STATE_COLORS[stateKey] || 'text-gray-400 bg-gray-900/50';
};

/**
 * Get status color classes
 * @param {string} status - The status value
 * @returns {string} Tailwind CSS classes for the status
 */
export const getStatusColor = (status) => {
  return STATUS_COLORS[status] || 'text-gray-400 bg-gray-900/50';
};

/**
 * Get action style classes for decision audit
 * @param {string} action - The action type
 * @returns {string} Tailwind CSS classes for the action
 */
export const getActionStyle = (action) => {
  return ACTION_COLORS[action] || 'bg-gray-900/30 text-gray-400 border-gray-700/50';
};

/**
 * Get action label for display
 * @param {string} action - The action type
 * @returns {string} Human-readable label
 */
export const getActionLabel = (action) => {
  return ACTION_LABELS[action] || action.toUpperCase();
};

/**
 * Get ping health color classes
 * @param {string} health - Health status ('healthy', 'degraded', 'unhealthy')
 * @returns {string} Tailwind CSS classes
 */
export const getPingHealthColor = (health) => {
  switch (health) {
    case 'healthy':
      return 'bg-green-900/30 text-green-400';
    case 'degraded':
      return 'bg-yellow-900/30 text-yellow-400';
    case 'unhealthy':
      return 'bg-red-900/30 text-red-400';
    default:
      return 'bg-gray-900/30 text-gray-400';
  }
};

/**
 * Get last ping age color classes
 * @param {number|null} age - Age in seconds
 * @returns {string} Tailwind CSS classes
 */
export const getPingAgeColor = (age) => {
  if (age === null) return 'text-gray-400';
  if (age < 10) return 'text-green-400';
  if (age < 30) return 'text-yellow-400';
  return 'text-red-400';
};

/**
 * Get side badge classes (yes/no)
 * @param {string} side - 'yes' or 'no'
 * @returns {string} Tailwind CSS classes
 */
export const getSideClasses = (side) => {
  return side === 'yes'
    ? 'bg-green-900/30 text-green-400 border border-green-700/50'
    : 'bg-red-900/30 text-red-400 border border-red-700/50';
};

/**
 * Get P&L color class
 * @param {number} value - P&L value (positive or negative)
 * @returns {string} Tailwind CSS color class
 */
export const getPnLColor = (value) => {
  return value >= 0 ? 'text-green-400' : 'text-red-400';
};
