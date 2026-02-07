/**
 * colorMaps.js - Shared color mappings and category styling.
 *
 * Consolidates tool categories, edge colors, and status styles used across components.
 */

import {
  Database, Globe, FileText, ShoppingCart, Crosshair,
  AlertTriangle, Clock, Search, Wallet, Wrench,
} from 'lucide-react';

// ─── Tool Category Mapping ───────────────────────────────────────────────────

/**
 * Map tool name to category.
 * @param {string} toolName - Name of the tool
 * @returns {'arb'|'memory'|'portfolio'|'surveillance'|'market'|'mentions'|'other'}
 */
export const getToolCategory = (toolName) => {
  if (['execute_arb', 'place_order', 'cancel_order', 'get_resting_orders'].includes(toolName)) return 'arb';
  if (['memory_store', 'memory_search', 'edit_file', 'write_file', 'read_file'].includes(toolName)) return 'memory';
  if (['get_positions', 'get_balance', 'get_trade_history'].includes(toolName)) return 'portfolio';
  if (['get_event_snapshot', 'get_events_summary', 'get_all_events'].includes(toolName)) return 'surveillance';
  if (['get_market_orderbook', 'get_recent_trades'].includes(toolName)) return 'market';
  if (['simulate_probability', 'trigger_simulation', 'compute_edge', 'query_wordnet', 'get_event_context', 'get_mention_context'].includes(toolName)) return 'mentions';
  return 'other';
};

/**
 * Category styling for badges and indicators.
 */
export const CATEGORY_STYLES = {
  arb: { bg: 'bg-cyan-500/10', text: 'text-cyan-400/80', dot: 'bg-cyan-500', label: 'ARB' },
  memory: { bg: 'bg-violet-500/10', text: 'text-violet-400/80', dot: 'bg-violet-500', label: 'MEM' },
  portfolio: { bg: 'bg-emerald-500/10', text: 'text-emerald-400/80', dot: 'bg-emerald-500', label: 'PORT' },
  surveillance: { bg: 'bg-orange-500/10', text: 'text-orange-400/80', dot: 'bg-orange-500', label: 'SURV' },
  market: { bg: 'bg-blue-500/10', text: 'text-blue-400/80', dot: 'bg-blue-500', label: 'MKT' },
  mentions: { bg: 'bg-amber-500/10', text: 'text-amber-400/80', dot: 'bg-amber-500', label: 'MENT' },
  other: { bg: 'bg-gray-800/40', text: 'text-gray-400', dot: 'bg-gray-500', label: 'TOOL' },
};

// ─── Tool Icons ──────────────────────────────────────────────────────────────

export const TOOL_ICONS = {
  get_event_snapshot: Database,
  get_all_events: Globe,
  get_events_summary: Globe,
  get_market_orderbook: Database,
  get_recent_trades: Globe,
  get_trade_history: FileText,
  execute_arb: ShoppingCart,
  place_order: Crosshair,
  cancel_order: AlertTriangle,
  get_resting_orders: Clock,
  memory_store: Database,
  memory_search: Search,
  get_positions: Wallet,
  get_balance: Wallet,
};

/**
 * Get icon component for a tool.
 * @param {string} toolName - Name of the tool
 * @returns {React.Component} Lucide icon component
 */
export const getToolIcon = (toolName) => TOOL_ICONS[toolName] || Wrench;

// ─── Edge Color Logic ────────────────────────────────────────────────────────

/**
 * Get edge color class based on edge value.
 * @param {number} edge - Edge value in cents
 * @param {{ high?: number, low?: number }} thresholds - Color thresholds
 * @returns {{ text: string, bg: string }} Tailwind color classes
 */
export const getEdgeColor = (edge, thresholds = { high: 10, low: 5 }) => {
  const absEdge = Math.abs(edge || 0);
  const positive = (edge || 0) > 0;

  if (absEdge >= thresholds.high) {
    return positive
      ? { text: 'text-emerald-400', bg: 'bg-emerald-950/15' }
      : { text: 'text-red-400', bg: 'bg-red-950/15' };
  }
  if (absEdge >= thresholds.low) {
    return positive
      ? { text: 'text-emerald-400/80', bg: '' }
      : { text: 'text-red-400/80', bg: '' };
  }
  return { text: 'text-gray-400', bg: '' };
};

// ─── Status Badge Colors ─────────────────────────────────────────────────────

export const STATUS_STYLES = {
  filled: { bg: 'bg-emerald-500/15', text: 'text-emerald-400/80', label: 'FILLED' },
  partial: { bg: 'bg-amber-500/15', text: 'text-amber-400/80', label: 'PARTIAL' },
  cancelled: { bg: 'bg-red-500/15', text: 'text-red-400/80', label: 'CANCEL' },
  pending: { bg: 'bg-gray-500/15', text: 'text-gray-400/80', label: 'PEND' },
  resting: { bg: 'bg-blue-500/15', text: 'text-blue-400/80', label: 'REST' },
};

/**
 * Get status style for a trade/order status.
 * @param {string} status - Status string
 * @returns {{ bg: string, text: string, label: string }}
 */
export const getStatusStyle = (status) => {
  return STATUS_STYLES[status?.toLowerCase()] || STATUS_STYLES.pending;
};

// ─── Todo Status Icons ───────────────────────────────────────────────────────

export const TODO_STATUS_ICONS = {
  completed: { icon: '\u2713', color: 'text-emerald-400' },
  in_progress: { icon: '\u25CE', color: 'text-amber-400' },
  pending: { icon: '\u25CB', color: 'text-gray-500' },
};
