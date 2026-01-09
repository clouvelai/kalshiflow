import React, { memo } from 'react';
import { Sparkles, X } from 'lucide-react';

/**
 * ResearchAlertToast - Shows notification for high-confidence AI research findings
 *
 * Displays when Agentic Research identifies a market with:
 * - High confidence rating
 * - Significant mispricing (>15%)
 *
 * Positioned in the bottom-right area of the screen.
 */
const ResearchAlertToast = ({ alert, onDismiss }) => {
  if (!alert) return null;

  const { eventTitle, market, marketsWithEdge } = alert;

  // Extract market details
  const {
    ticker,
    evidence_probability,
    market_probability,
    mispricing_magnitude,
    recommendation,
  } = market || {};

  // Calculate edge percentage
  const edgePercent = Math.round((mispricing_magnitude || 0) * 100);

  // Determine recommendation styling
  const isYes = recommendation === 'BUY_YES';
  const recText = isYes ? 'YES' : 'NO';
  const recArrow = isYes ? '\u2191' : '\u2193';

  return (
    <div className={`
      fixed bottom-28 right-4 p-4 rounded-lg shadow-xl z-50
      border backdrop-blur-sm max-w-sm
      bg-purple-900/90 border-purple-600
      animate-slide-in
    `}>
      <div className="flex items-start space-x-3">
        {/* AI Icon with pulse */}
        <div className="flex-shrink-0">
          <Sparkles className="w-6 h-6 text-purple-400 animate-pulse" />
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0">
          {/* Header */}
          <div className="text-sm font-semibold text-purple-200 truncate">
            AI Research Alert
          </div>

          {/* Event title */}
          <div className="text-xs text-purple-300/80 truncate mt-0.5">
            {eventTitle}
          </div>

          {/* Market details */}
          <div className="mt-2 p-2 bg-purple-800/40 rounded-md">
            <div className="text-sm font-mono text-white truncate">
              {ticker}
            </div>

            {/* Probability comparison */}
            <div className="flex items-center gap-2 mt-1 text-xs">
              <span className="text-purple-300">AI:</span>
              <span className="font-mono text-white font-medium">
                {Math.round((evidence_probability || 0) * 100)}%
              </span>
              <span className="text-purple-400">vs</span>
              <span className="font-mono text-purple-300">
                {Math.round((market_probability || 0) * 100)}%
              </span>
              <span className="text-purple-400">mkt</span>
            </div>

            {/* Edge and recommendation */}
            <div className="flex items-center justify-between mt-2">
              <span className={`
                font-mono text-sm font-bold px-2 py-0.5 rounded
                ${edgePercent > 0 ? 'text-emerald-300 bg-emerald-500/20' : 'text-red-300 bg-red-500/20'}
              `}>
                {edgePercent > 0 ? '+' : ''}{edgePercent}% edge
              </span>

              <span className={`
                flex items-center gap-1 px-2 py-0.5 rounded text-sm font-bold
                ${isYes
                  ? 'text-emerald-300 bg-emerald-500/25 border border-emerald-500/40'
                  : 'text-red-300 bg-red-500/25 border border-red-500/40'
                }
              `}>
                <span>{recArrow}</span>
                <span>BUY {recText}</span>
              </span>
            </div>
          </div>

          {/* Additional markets indicator */}
          {marketsWithEdge > 1 && (
            <div className="text-xs text-purple-400 mt-1.5">
              + {marketsWithEdge - 1} more market{marketsWithEdge > 2 ? 's' : ''} with edge
            </div>
          )}
        </div>

        {/* Dismiss button */}
        <button
          onClick={onDismiss}
          className="flex-shrink-0 text-purple-400 hover:text-white transition-colors"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Animation styles */}
      <style>{`
        @keyframes slide-in {
          0% {
            opacity: 0;
            transform: translateX(100px);
          }
          100% {
            opacity: 1;
            transform: translateX(0);
          }
        }
        .animate-slide-in {
          animation: slide-in 0.3s ease-out;
        }
      `}</style>
    </div>
  );
};

export default memo(ResearchAlertToast);
