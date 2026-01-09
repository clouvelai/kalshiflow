import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import LifecycleMarketCard from './LifecycleMarketCard';

/**
 * MarketGrid - Grid of market cards, optionally grouped by category
 *
 * Props:
 *   - marketsByCategory: Object mapping category -> array of markets
 *   - showCategoryHeaders: Boolean to show category headers
 *   - rlmStates: Object mapping ticker -> RLM state
 *   - tradePulses: Object mapping ticker -> { side, ts } for pulse animation
 *   - rlmConfig: RLM strategy config from backend { min_trades, yes_threshold, min_price_drop }
 *   - eventExposure: Event exposure data { event_groups: { event_ticker -> EventGroup } }
 *   - getMarketResearch: Function (ticker) => research assessment | null
 */

/**
 * Extract abbreviated category from event_ticker.
 * @param {string} eventTicker - e.g., "KXNFL-25JAN05", "KXUNITEDCUP-26"
 * @returns {string|null} - e.g., "NFL", "UCUP"
 */
function parseEventAbbrev(eventTicker) {
  if (!eventTicker) return null;

  // Format: KX{CATEGORY}-{DATE/ID}
  // Extract category between "KX" and first "-"
  const match = eventTicker.match(/^KX([A-Z]+)/i);
  if (!match) return null;

  const category = match[1].toUpperCase();

  // Shorten long categories (max 4 chars)
  if (category.length > 4) {
    return category.slice(0, 4);
  }
  return category;
}

/**
 * Get event exposure data for a specific market.
 *
 * Looks up the market's event_ticker in the event_groups and adds
 * market_index based on position within the event.
 */
function getMarketEventExposure(market, eventExposure) {
  if (!eventExposure?.event_groups || !market.event_ticker) {
    return null;
  }

  const eventGroup = eventExposure.event_groups[market.event_ticker];
  if (!eventGroup) {
    return null;
  }

  // Calculate market index within the event (1-based for display)
  // Sort tickers alphabetically for stable ordering
  const marketTickers = Object.keys(eventGroup.markets || {}).sort();
  const marketIndex = marketTickers.indexOf(market.ticker) + 1;

  // Parse event abbreviation from event_ticker (e.g., "KXNFL-25JAN05" -> "NFL")
  const eventAbbrev = parseEventAbbrev(market.event_ticker);

  return {
    ...eventGroup,
    market_index: marketIndex > 0 ? marketIndex : 1,
    event_abbrev: eventAbbrev,
  };
}

// Animation variants for market cards
const cardVariants = {
  initial: { opacity: 0, y: -20, scale: 0.95 },
  animate: { opacity: 1, y: 0, scale: 1 },
  exit: { opacity: 0, scale: 0.9, transition: { duration: 0.2 } }
};

const MarketGrid = ({ marketsByCategory, showCategoryHeaders, groupBy = 'category', rlmStates = {}, tradePulses = {}, rlmConfig, eventExposure, getMarketResearch }) => {
  const categories = Object.entries(marketsByCategory);

  if (categories.length === 0) {
    return (
      <div className="bg-gray-900/30 rounded-lg p-8 text-center border border-gray-800">
        <div className="text-gray-500 mb-2">
          <svg className="w-12 h-12 mx-auto mb-3 opacity-50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
              d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
          </svg>
        </div>
        <p className="text-gray-400 text-sm">No markets tracked yet</p>
        <p className="text-gray-600 text-xs mt-1">
          Markets will appear here as they're discovered via lifecycle events
        </p>
      </div>
    );
  }

  // Single category view (filtered)
  if (!showCategoryHeaders && categories.length === 1) {
    const [_, markets] = categories[0];
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
        <AnimatePresence mode="popLayout">
          {markets.map(market => (
            <motion.div
              key={market.ticker}
              layout
              variants={cardVariants}
              initial="initial"
              animate="animate"
              exit="exit"
              transition={{ duration: 0.3, type: "spring", bounce: 0.2 }}
            >
              <LifecycleMarketCard
                market={market}
                rlmState={rlmStates[market.ticker]}
                tradePulse={tradePulses[market.ticker]}
                rlmConfig={rlmConfig}
                eventExposure={getMarketEventExposure(market, eventExposure)}
                research={getMarketResearch?.(market.ticker)}
              />
            </motion.div>
          ))}
        </AnimatePresence>
      </div>
    );
  }

  // Multi-category view with headers
  return (
    <div className="space-y-6">
      {categories.map(([groupKey, markets]) => (
        <div key={groupKey}>
          {/* Group header (category or event) */}
          {showCategoryHeaders && (
            <div className="flex items-center gap-3 mb-3">
              <h2 className={`text-sm font-semibold uppercase tracking-wide ${
                groupBy === 'event' ? 'text-blue-400' : 'text-gray-300'
              }`}>
                {formatGroupName(groupKey, groupBy)}
              </h2>
              <span className="text-xs text-gray-500 font-mono">
                {markets.length} market{markets.length !== 1 ? 's' : ''}
              </span>
              <div className="flex-1 border-t border-gray-800" />
            </div>
          )}

          {/* Market cards grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
            <AnimatePresence mode="popLayout">
              {markets.map(market => (
                <motion.div
                  key={market.ticker}
                  layout
                  variants={cardVariants}
                  initial="initial"
                  animate="animate"
                  exit="exit"
                  transition={{ duration: 0.3, type: "spring", bounce: 0.2 }}
                >
                  <LifecycleMarketCard
                    market={market}
                    rlmState={rlmStates[market.ticker]}
                    tradePulse={tradePulses[market.ticker]}
                    rlmConfig={rlmConfig}
                    eventExposure={getMarketEventExposure(market, eventExposure)}
                    research={getMarketResearch?.(market.ticker)}
                  />
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        </div>
      ))}
    </div>
  );
};

/**
 * Format group header name based on groupBy mode.
 * @param {string} key - The group key (category slug or event_ticker)
 * @param {string} groupBy - 'category' or 'event'
 * @returns {string} Formatted display name
 */
function formatGroupName(key, groupBy = 'category') {
  if (groupBy === 'event') {
    // Format event_ticker: "KXNFL-26JAN05-HOUPIT" -> "NFL: 26JAN05-HOUPIT"
    if (key === 'no-event') return 'No Event';

    const abbrev = parseEventAbbrev(key);
    if (abbrev) {
      // Remove the "KX{CATEGORY}-" prefix for cleaner display
      const suffix = key.replace(/^KX[A-Z]+-/i, '');
      return `${abbrev}: ${suffix}`;
    }
    return key;
  }

  // Category names (default)
  const names = {
    sports: 'Sports',
    crypto: 'Crypto',
    entertainment: 'Entertainment',
    media_mentions: 'Media',
    politics: 'Politics',
    economics: 'Economics',
    climate: 'Climate',
    other: 'Other'
  };
  return names[key] || key;
}

export default MarketGrid;
