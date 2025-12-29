import React, { useState, memo } from 'react';
import { ChevronRight, ChevronDown, FileText } from 'lucide-react';
import { formatAge, getActionStyle, getActionLabel } from '../../../utils/v3-trader';

/**
 * DecisionRow - Memoized row component for decision history table
 */
const DecisionRow = memo(({ decision, index }) => (
  <tr
    className={`border-b border-gray-700/30 hover:bg-gray-800/50 transition-colors ${
      decision.action === 'followed' ? 'bg-green-900/10' : ''
    }`}
  >
    <td className="px-3 py-2 text-center">
      <span className={`px-2 py-0.5 rounded text-xs font-bold uppercase border ${getActionStyle(decision.action)}`}>
        {getActionLabel(decision.action)}
      </span>
    </td>
    <td className="px-3 py-2 font-mono text-gray-300 text-xs">{decision.market_ticker}</td>
    <td className="px-3 py-2 text-center">
      <span className={`px-2 py-0.5 rounded text-xs font-bold uppercase ${
        decision.side === 'yes'
          ? 'bg-green-900/30 text-green-400 border border-green-700/50'
          : 'bg-red-900/30 text-red-400 border border-red-700/50'
      }`}>
        {decision.side}
      </span>
    </td>
    <td className="px-3 py-2 text-right font-mono text-gray-400 text-xs">
      ${decision.whale_size_dollars?.toFixed(0) || 0}
    </td>
    <td className="px-3 py-2 text-gray-400 text-xs truncate max-w-xs" title={decision.reason}>
      {decision.reason}
    </td>
    <td className="px-3 py-2 text-right font-mono text-gray-500 text-xs">
      {formatAge(decision.age_seconds)}
    </td>
  </tr>
));

DecisionRow.displayName = 'DecisionRow';

/**
 * DecisionAuditPanel - Shows why whales were followed/skipped
 */
const DecisionAuditPanel = ({ decisionHistory, decisionStats }) => {
  const [isExpanded, setIsExpanded] = useState(true);

  // Calculate stats summary
  const stats = decisionStats || {};
  const detected = stats.whales_detected || 0;
  const followed = stats.whales_followed || 0;
  const skipped = stats.whales_skipped || 0;
  const followRate = detected > 0 ? ((followed / detected) * 100).toFixed(1) : 0;

  return (
    <div className="bg-gray-900/50 backdrop-blur-sm rounded-xl border border-gray-800 p-4 mt-4">
      {/* Header with stats summary */}
      <div
        className="flex items-center justify-between cursor-pointer"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center space-x-2">
          <FileText className="w-4 h-4 text-orange-400" />
          <h3 className="text-sm font-bold text-gray-300 uppercase tracking-wider">Decision Audit</h3>
        </div>
        <div className="flex items-center space-x-4">
          {/* Stats summary row */}
          <div className="flex items-center space-x-3 text-xs">
            <span className="font-mono">
              <span className="text-gray-500">Detected:</span> <span className="text-white font-bold">{detected}</span>
            </span>
            <span className="text-gray-600">|</span>
            <span className="font-mono">
              <span className="text-gray-500">Followed:</span> <span className="text-green-400 font-bold">{followed}</span>
            </span>
            <span className="text-gray-600">|</span>
            <span className="font-mono">
              <span className="text-gray-500">Skipped:</span> <span className="text-yellow-400 font-bold">{skipped}</span>
            </span>
            <span className="text-gray-600">|</span>
            <span className="font-mono">
              <span className="text-gray-500">Rate:</span> <span className="text-cyan-400 font-bold">{followRate}%</span>
            </span>
          </div>
          {isExpanded ? (
            <ChevronDown className="w-4 h-4 text-gray-500" />
          ) : (
            <ChevronRight className="w-4 h-4 text-gray-500" />
          )}
        </div>
      </div>

      {/* Skip reason breakdown */}
      {isExpanded && stats && (
        <div className="grid grid-cols-6 gap-2 mt-4 mb-4">
          <div className="bg-gray-800/30 rounded-lg p-2 border border-gray-700/50 text-center">
            <div className="text-xs text-gray-500 uppercase">Age</div>
            <div className="text-sm font-mono font-bold text-yellow-400">{stats.skipped_age || 0}</div>
          </div>
          <div className="bg-gray-800/30 rounded-lg p-2 border border-gray-700/50 text-center">
            <div className="text-xs text-gray-500 uppercase">Position</div>
            <div className="text-sm font-mono font-bold text-purple-400">{stats.skipped_position || 0}</div>
          </div>
          <div className="bg-gray-800/30 rounded-lg p-2 border border-gray-700/50 text-center">
            <div className="text-xs text-gray-500 uppercase">Orders</div>
            <div className="text-sm font-mono font-bold text-blue-400">{stats.skipped_orders || 0}</div>
          </div>
          <div className="bg-gray-800/30 rounded-lg p-2 border border-gray-700/50 text-center">
            <div className="text-xs text-gray-500 uppercase">Already</div>
            <div className="text-sm font-mono font-bold text-gray-400">{stats.already_followed || 0}</div>
          </div>
          <div className="bg-gray-800/30 rounded-lg p-2 border border-gray-700/50 text-center">
            <div className="text-xs text-gray-500 uppercase">Rate Limit</div>
            <div className="text-sm font-mono font-bold text-red-400">{stats.rate_limited || 0}</div>
          </div>
          <div className="bg-green-900/20 rounded-lg p-2 border border-green-700/30 text-center">
            <div className="text-xs text-gray-500 uppercase">Followed</div>
            <div className="text-sm font-mono font-bold text-green-400">{stats.whales_followed || 0}</div>
          </div>
        </div>
      )}

      {/* Decision history table */}
      {isExpanded && decisionHistory && decisionHistory.length > 0 && (
        <div className="bg-gray-800/30 rounded-lg border border-gray-700/50 overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-gray-900/50 border-b border-gray-700/50">
                <th className="px-3 py-2 text-center text-xs text-gray-500 uppercase font-medium">Action</th>
                <th className="px-3 py-2 text-left text-xs text-gray-500 uppercase font-medium">Market</th>
                <th className="px-3 py-2 text-center text-xs text-gray-500 uppercase font-medium">Side</th>
                <th className="px-3 py-2 text-right text-xs text-gray-500 uppercase font-medium">Size</th>
                <th className="px-3 py-2 text-left text-xs text-gray-500 uppercase font-medium">Reason</th>
                <th className="px-3 py-2 text-right text-xs text-gray-500 uppercase font-medium">Age</th>
              </tr>
            </thead>
            <tbody>
              {decisionHistory.slice(0, 10).map((decision, index) => (
                <DecisionRow
                  key={`${decision.whale_id}-${index}`}
                  decision={decision}
                  index={index}
                />
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Empty state */}
      {isExpanded && (!decisionHistory || decisionHistory.length === 0) && (
        <div className="bg-gray-800/30 rounded-lg p-6 border border-gray-700/50 text-center mt-4">
          <FileText className="w-6 h-6 text-gray-600 mx-auto mb-2" />
          <div className="text-gray-500 text-sm">No decisions yet</div>
          <div className="text-gray-600 text-xs mt-1">Waiting for whale queue evaluations</div>
        </div>
      )}
    </div>
  );
};

export default memo(DecisionAuditPanel);
