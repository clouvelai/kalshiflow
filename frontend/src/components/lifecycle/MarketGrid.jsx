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
 */

// Animation variants for market cards
const cardVariants = {
  initial: { opacity: 0, y: -20, scale: 0.95 },
  animate: { opacity: 1, y: 0, scale: 1 },
  exit: { opacity: 0, scale: 0.9, transition: { duration: 0.2 } }
};

const MarketGrid = ({ marketsByCategory, showCategoryHeaders, rlmStates = {}, tradePulses = {} }) => {
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
      {categories.map(([category, markets]) => (
        <div key={category}>
          {/* Category header */}
          {showCategoryHeaders && (
            <div className="flex items-center gap-3 mb-3">
              <h2 className="text-sm font-semibold text-gray-300 uppercase tracking-wide">
                {formatCategoryName(category)}
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

function formatCategoryName(category) {
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
  return names[category] || category;
}

export default MarketGrid;
