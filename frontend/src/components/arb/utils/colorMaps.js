/**
 * colorMaps.js - Shared color mappings and category styling.
 *
 * Consolidates tool categories, edge colors, and status styles used across components.
 */

import {
  Database, Globe, FileText, ShoppingCart, Crosshair,
  AlertTriangle, Clock, Search, Wallet, Wrench, HeartPulse, Sunrise,
} from 'lucide-react';

// ─── Tool Category Mapping ───────────────────────────────────────────────────

/**
 * Map tool name to category.
 * @param {string} toolName - Name of the tool
 * @returns {'arb'|'memory'|'portfolio'|'surveillance'|'market'|'sniper'|'other'}
 */
export const getToolCategory = (toolName) => {
  if (['execute_arb', 'place_order', 'cancel_order', 'get_resting_orders'].includes(toolName)) return 'arb';
  if (['recall_memory', 'store_insight'].includes(toolName)) return 'memory';
  if (['get_portfolio'].includes(toolName)) return 'portfolio';
  if (['get_events_summary', 'get_market_state', 'search_news', 'get_market_movers'].includes(toolName)) return 'surveillance';
  if (['get_market_orderbook', 'get_recent_trades'].includes(toolName)) return 'market';
  if (['configure_sniper', 'configure_automation'].includes(toolName)) return 'sniper';
  if (['configure_quotes', 'pull_quotes', 'resume_quotes', 'get_quote_performance'].includes(toolName)) return 'mm';
  if (['get_account_health'].includes(toolName)) return 'system';
  if (['get_early_bird_opportunities'].includes(toolName)) return 'early_bird';
  if (['write_todos'].includes(toolName)) return 'todo';
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
  sniper: { bg: 'bg-rose-500/10', text: 'text-rose-400/80', dot: 'bg-rose-500', label: 'SNPR' },
  mm: { bg: 'bg-fuchsia-500/10', text: 'text-fuchsia-400/80', dot: 'bg-fuchsia-500', label: 'MM' },
  system: { bg: 'bg-teal-500/10', text: 'text-teal-400/80', dot: 'bg-teal-500', label: 'SYS' },
  early_bird: { bg: 'bg-lime-500/10', text: 'text-lime-400/80', dot: 'bg-lime-500', label: 'BIRD' },
  todo: { bg: 'bg-amber-500/10', text: 'text-amber-400/80', dot: 'bg-amber-500', label: 'TODO' },
  other: { bg: 'bg-gray-800/40', text: 'text-gray-400', dot: 'bg-gray-500', label: 'TOOL' },
};

// ─── Tool Icons ──────────────────────────────────────────────────────────────

export const TOOL_ICONS = {
  execute_arb: ShoppingCart,
  place_order: Crosshair,
  cancel_order: AlertTriangle,
  get_resting_orders: Clock,
  // V2 tools
  get_events_summary: Globe,
  get_market_state: Globe,
  get_market_orderbook: Database,
  get_recent_trades: Globe,
  get_portfolio: Wallet,
  search_news: Search,
  recall_memory: Search,
  store_insight: Database,
  configure_sniper: Crosshair,
  configure_automation: Wrench,
  get_account_health: HeartPulse,
  get_market_movers: Globe,
  get_early_bird_opportunities: Sunrise,
  configure_quotes: Wrench,
  pull_quotes: AlertTriangle,
  resume_quotes: Crosshair,
  get_quote_performance: Database,
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

// ─── Attention Urgency Styles ────────────────────────────────────────────────

export const ATTENTION_URGENCY_STYLES = {
  immediate: { bg: 'bg-red-500/15', text: 'text-red-400', dot: 'bg-red-500', pulse: true },
  high:      { bg: 'bg-amber-500/15', text: 'text-amber-400', dot: 'bg-amber-500', pulse: false },
  normal:    { bg: 'bg-gray-800/40', text: 'text-gray-400', dot: 'bg-gray-600', pulse: false },
};

// ─── Attention Category Styles (override urgency when category is special) ──

export const ATTENTION_CATEGORY_STYLES = {
  early_bird: { bg: 'bg-lime-500/12', text: 'text-lime-400', dot: 'bg-lime-500', pulse: true, label: 'EARLY BIRD' },
  mm_fill: { bg: 'bg-fuchsia-500/12', text: 'text-fuchsia-400', dot: 'bg-fuchsia-500', pulse: false, label: 'MM FILL' },
  mm_vpin_spike: { bg: 'bg-fuchsia-500/12', text: 'text-fuchsia-400', dot: 'bg-fuchsia-500', pulse: true, label: 'MM VPIN' },
  mm_inventory_warning: { bg: 'bg-fuchsia-500/12', text: 'text-fuchsia-400', dot: 'bg-fuchsia-500', pulse: false, label: 'MM INV' },
};

// ─── Auto-Action Styles ─────────────────────────────────────────────────────

export const AUTO_ACTION_STYLES = {
  stop_loss:    { bg: 'bg-red-500/10', text: 'text-red-400', label: 'STOP' },
  time_exit:    { bg: 'bg-amber-500/10', text: 'text-amber-400', label: 'TIME' },
  regime_gate:  { bg: 'bg-violet-500/10', text: 'text-violet-400', label: 'GATE' },
};

// ─── Captain Mode Styles ────────────────────────────────────────────────────

export const CAPTAIN_MODE_STYLES = {
  reactive:  { bg: 'bg-amber-500/15', text: 'text-amber-300', border: 'border-amber-500/20' },
  strategic: { bg: 'bg-violet-500/15', text: 'text-violet-300', border: 'border-violet-500/20' },
  deep_scan: { bg: 'bg-blue-500/15', text: 'text-blue-300', border: 'border-blue-500/20' },
};

// ─── Todo Status Icons ───────────────────────────────────────────────────────

export const TODO_STATUS_ICONS = {
  completed: { icon: '\u2713', color: 'text-emerald-400' },
  in_progress: { icon: '\u25CE', color: 'text-amber-400' },
  pending: { icon: '\u25CB', color: 'text-gray-500' },
  stale: { icon: '\u26A0', color: 'text-red-400' },
};
