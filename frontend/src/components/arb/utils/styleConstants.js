/**
 * styleConstants.js - Shared Tailwind style patterns for arb dashboard.
 *
 * Consolidates repeated panel styles and common class patterns.
 */

// ─── Panel Base Styles ───────────────────────────────────────────────────────

export const PANEL_BASE = 'bg-gradient-to-br from-gray-900/60 via-gray-900/40 to-gray-950/60';
export const PANEL_BLUR = 'backdrop-blur-sm';
export const PANEL_ROUNDED = 'rounded-xl';
export const PANEL_SHADOW = 'shadow-lg shadow-black/10';

// ─── Border Variants ─────────────────────────────────────────────────────────

export const BORDER_VARIANTS = {
  cyan: 'border border-cyan-500/8 shadow-lg shadow-cyan-500/3',
  violet: 'border border-violet-500/15',
  amber: 'border border-amber-500/10',
  emerald: 'border border-emerald-500/15',
  gray: 'border border-gray-800/30',
};

// ─── Combined Panel Classes ──────────────────────────────────────────────────

/**
 * Get combined panel classes for a variant.
 * @param {'cyan'|'violet'|'amber'|'emerald'|'gray'} variant - Border variant
 * @returns {string} Combined Tailwind classes
 */
export const panelClasses = (variant = 'gray') => {
  return `${PANEL_BASE} ${PANEL_BLUR} ${PANEL_ROUNDED} ${PANEL_SHADOW} ${BORDER_VARIANTS[variant] || BORDER_VARIANTS.gray}`;
};

// ─── Section Container Styles ────────────────────────────────────────────────

export const SECTION_CONTAINER = 'rounded-lg border border-gray-800/30 bg-gray-900/25';
export const SECTION_HEADER = 'flex items-center gap-2 px-3 py-2';
export const SECTION_LABEL = 'text-[11px] font-semibold text-gray-400 uppercase tracking-wider';

// ─── Card Styles ─────────────────────────────────────────────────────────────

export const CARD_BASE = 'rounded-lg overflow-hidden bg-gray-900/20 hover:bg-gray-900/30 transition-colors';
export const CARD_BORDERED = 'border border-gray-800/25';

// ─── Text Styles ─────────────────────────────────────────────────────────────

export const TEXT_MUTED = 'text-gray-500';
export const TEXT_MONO = 'font-mono';
export const TEXT_TINY = 'text-[9px]';
export const TEXT_SMALL = 'text-[10px]';
export const TEXT_MEDIUM = 'text-[11px]';

// ─── Animation Classes ───────────────────────────────────────────────────────

export const ANIMATE_SPIN = 'animate-spin';
export const ANIMATE_PULSE = 'animate-pulse';

// ─── Layout Dimensions ──────────────────────────────────────────────────────

export const LEFT_SIDEBAR_WIDTH = 300;
export const LEFT_SIDEBAR_COLLAPSED = 40;
export const RIGHT_SIDEBAR_WIDTH = 360;
export const RIGHT_SIDEBAR_COLLAPSED = 40;
export const HEADER_HEIGHT = 44;

// ─── Panel Dividers ─────────────────────────────────────────────────────────

export const PANEL_DIVIDER = 'border-r border-gray-800/40';

// ─── Agent State Colors ─────────────────────────────────────────────────────

export const AGENT_STATE_COLORS = {
  idle: { bg: 'bg-gray-500', text: 'text-gray-400', label: 'Idle' },
  thinking: { bg: 'bg-violet-500', text: 'text-violet-400', label: 'Thinking' },
  acting: { bg: 'bg-cyan-500', text: 'text-cyan-400', label: 'Acting' },
  complete: { bg: 'bg-emerald-500', text: 'text-emerald-400', label: 'Complete' },
};
