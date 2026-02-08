import React, { memo, useState, useMemo } from 'react';
import { X, ExternalLink } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import OverviewTab from './tabs/OverviewTab';
import MarketsTab from './tabs/MarketsTab';
import BookTradesTab from './tabs/BookTradesTab';
import HeatmapTab from './tabs/HeatmapTab';

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'markets', label: 'Markets' },
  { id: 'book', label: 'Book & Trades' },
  { id: 'heatmap', label: 'Heatmap' },
];

const EventView = memo(({ event, eventTrades, arbTrades, positionsByTicker, mentionsData, onClose }) => {
  const [activeTab, setActiveTab] = useState('markets');
  const [selectedMarket, setSelectedMarket] = useState(null);

  if (!event) return null;

  const { event_ticker, title, category, mutually_exclusive } = event;

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.15 }}
      className="flex flex-col h-full"
    >
      {/* Header */}
      <div className="flex items-center gap-3 px-4 py-2.5 border-b border-gray-800/40 shrink-0 bg-gray-950/40">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-0.5">
            <span className="text-[10px] font-mono text-gray-500">{event_ticker}</span>
            {category && (
              <span className="text-[9px] font-mono bg-gray-800/40 text-gray-400 rounded-full px-2 py-0.5">{category}</span>
            )}
            <span className={`text-[9px] font-mono rounded-full px-2 py-0.5 ${
              mutually_exclusive
                ? 'bg-emerald-500/10 text-emerald-400/70'
                : 'bg-amber-500/10 text-amber-400/70'
            }`}>
              {mutually_exclusive ? 'mut. excl.' : 'independent'}
            </span>
          </div>
          <div className="flex items-center gap-2">
            {event.image_url && (
              <img
                src={event.image_url}
                alt=""
                className="w-6 h-6 rounded-md object-cover opacity-80 flex-shrink-0"
                onError={(e) => { e.target.style.display = 'none'; }}
              />
            )}
            <h3 className="text-sm font-medium text-gray-200 leading-tight truncate">{title || event_ticker}</h3>
          </div>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          {(event.kalshi_url || event_ticker) && (
            <a
              href={event.kalshi_url || `https://kalshi.com/markets/${event_ticker.toLowerCase()}`}
              target="_blank"
              rel="noopener noreferrer"
              className="text-gray-600 hover:text-cyan-400 transition-colors"
              title="View on Kalshi"
            >
              <ExternalLink className="w-3.5 h-3.5" />
            </a>
          )}
          <button
            onClick={onClose}
            className="p-1 rounded-md hover:bg-gray-800/50 text-gray-500 hover:text-gray-300 transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Tab bar */}
      <div className="flex items-center gap-1 px-4 py-1.5 border-b border-gray-800/30 shrink-0 bg-gray-950/20">
        {TABS.map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`px-3 py-1 text-[11px] font-medium rounded-md transition-colors ${
              activeTab === tab.id
                ? 'bg-cyan-500/12 text-cyan-400 border border-cyan-500/20'
                : 'text-gray-500 hover:text-gray-300 hover:bg-gray-800/30'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div className="flex-1 overflow-y-auto min-h-0 p-4">
        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, x: 8 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -8 }}
            transition={{ duration: 0.1 }}
          >
            {activeTab === 'overview' && <OverviewTab event={event} />}
            {activeTab === 'markets' && (
              <MarketsTab
                event={event}
                positionsByTicker={positionsByTicker}
                selectedMarket={selectedMarket}
                onSelectMarket={setSelectedMarket}
                mentionsData={mentionsData}
              />
            )}
            {activeTab === 'book' && (
              <BookTradesTab
                event={event}
                eventTrades={eventTrades}
                arbTrades={arbTrades}
                selectedMarket={selectedMarket}
                onSelectMarket={setSelectedMarket}
              />
            )}
            {activeTab === 'heatmap' && <HeatmapTab event={event} />}
          </motion.div>
        </AnimatePresence>
      </div>
    </motion.div>
  );
});

EventView.displayName = 'EventView';

export default EventView;
