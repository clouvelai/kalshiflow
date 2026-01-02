import React, { memo } from 'react';
import {
  Activity, Zap, Database, ChevronRight, ChevronDown,
  AlertCircle, CheckCircle, XCircle, ArrowRight, Info
} from 'lucide-react';
import { getStateColor, getStatusColor } from '../../../utils/v3-trader';

/**
 * Get icon for message type
 */
const getMessageIcon = (type, metadata) => {
  if (metadata?.isTransition) {
    return <ArrowRight className="w-4 h-4 text-purple-400" />;
  }

  // Handle activity messages based on activity_type
  if (type === 'activity' && metadata?.activity_type) {
    switch (metadata.activity_type) {
      case 'state_transition': return <ArrowRight className="w-4 h-4 text-purple-400" />;
      case 'sync': return <Activity className="w-4 h-4 text-blue-400" />;
      case 'health_check': return <Activity className="w-4 h-4 text-green-400" />;
      case 'operation': return <ChevronRight className="w-4 h-4 text-gray-400" />;
      default: return <ChevronRight className="w-4 h-4 text-gray-500" />;
    }
  }

  switch (type) {
    case 'state': return <Zap className="w-4 h-4 text-purple-400" />;
    case 'data': return <Database className="w-4 h-4 text-blue-400" />;
    case 'success': return <CheckCircle className="w-4 h-4 text-green-400" />;
    case 'warning': return <AlertCircle className="w-4 h-4 text-yellow-400" />;
    case 'error': return <XCircle className="w-4 h-4 text-red-400" />;
    case 'info': return <Info className="w-4 h-4 text-blue-400" />;
    default: return <ChevronRight className="w-4 h-4 text-gray-500" />;
  }
};

/**
 * Get color for message type
 */
const getMessageColor = (type, metadata) => {
  if (type === 'activity' && metadata?.activity_type) {
    switch (metadata.activity_type) {
      case 'state_transition': return 'text-purple-200';
      case 'sync': return 'text-blue-200';
      case 'health_check': return 'text-green-200';
      case 'operation': return 'text-gray-200';
      default: return 'text-gray-200';
    }
  }

  switch (type) {
    case 'state': return 'text-purple-200';
    case 'data': return 'text-blue-200';
    case 'success': return 'text-green-200';
    case 'warning': return 'text-yellow-200';
    case 'error': return 'text-red-200';
    case 'info': return 'text-blue-200';
    case 'activity': return 'text-gray-200';
    default: return 'text-gray-200';
  }
};

/**
 * ConsoleMessage - Single message in the console output
 */
const ConsoleMessage = memo(({ message, isExpanded, onToggleExpand }) => {
  const hasMetadata = message.metadata?.formattedMetadata &&
    Object.keys(message.metadata.formattedMetadata).length > 0;

  return (
    <div className="group hover:bg-gray-800/20 rounded-lg transition-all duration-200">
      <div className="flex items-start space-x-3 p-2">
        {/* Timestamp */}
        <span className="text-gray-500 text-xs w-20 flex-shrink-0 font-mono">
          [{message.timestamp}]
        </span>

        {/* State Transition or State Badge */}
        <div className="w-44 flex-shrink-0">
          {message.metadata?.isTransition && message.metadata?.fromState && message.metadata?.toState ? (
            <div className="flex items-center space-x-1 text-xs">
              <span className={`px-2 py-0.5 rounded font-bold ${getStateColor(message.metadata.fromState)}`}>
                {message.metadata.fromState}
              </span>
              <ArrowRight className="w-3 h-3 text-gray-400" />
              <span className={`px-2 py-0.5 rounded font-bold ${getStateColor(message.metadata.toState)}`}>
                {message.metadata.toState}
              </span>
            </div>
          ) : message.metadata?.state ? (
            <span className={`px-2 py-0.5 rounded text-xs font-bold ${getStateColor(message.metadata.state)}`}>
              [{message.metadata.state}]
            </span>
          ) : null}
        </div>

        {/* Status Badge */}
        <div className="w-24 flex-shrink-0">
          {message.metadata?.status && (
            <span className={`px-2 py-0.5 rounded text-xs font-bold ${getStatusColor(message.metadata.status)}`}>
              {message.metadata.status}
            </span>
          )}
        </div>

        {/* Icon */}
        <div className="flex-shrink-0 mt-0.5">
          {getMessageIcon(message.type, message.metadata)}
        </div>

        {/* Message Content */}
        {message.content && (
          <div className={`flex-1 ${getMessageColor(message.type, message.metadata)}`}>
            <div className="leading-relaxed">{message.content}</div>
          </div>
        )}
        {!message.content && <div className="flex-1"></div>}

        {/* Expand/Collapse Button */}
        {hasMetadata && (
          <button
            onClick={() => onToggleExpand(message.id)}
            className="flex-shrink-0 p-1 text-gray-500 hover:text-gray-300 transition-colors"
            title={isExpanded ? "Collapse metadata" : "Expand metadata"}
          >
            {isExpanded ? (
              <ChevronDown className="w-4 h-4" />
            ) : (
              <ChevronRight className="w-4 h-4" />
            )}
          </button>
        )}
      </div>

      {/* Expandable Metadata Section */}
      {hasMetadata && isExpanded && (
        <div className="ml-24 mb-2 mr-4">
          <div className="bg-gray-800/30 rounded-lg p-3 border border-gray-700/50">
            <div className="text-xs text-gray-400 font-mono space-y-1">
              {Object.entries(message.metadata.formattedMetadata).map(([key, value]) => (
                <div key={key} className="flex">
                  <span className="text-gray-500 mr-2">{key}:</span>
                  <span className="text-gray-300 break-all whitespace-pre-wrap">{value}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
});

ConsoleMessage.displayName = 'ConsoleMessage';

export default ConsoleMessage;
