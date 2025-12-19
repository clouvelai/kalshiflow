import React, { useState } from 'react';
import { CheckCircleIcon, XCircleIcon, ClockIcon, ExclamationTriangleIcon } from '@heroicons/react/24/solid';
import { CheckCircleIcon as CheckCircleOutline, XCircleIcon as XCircleOutline, ClockIcon as ClockOutline, ChevronDownIcon, ChevronRightIcon } from '@heroicons/react/24/outline';

const SystemHealth = ({ initializationStatus, componentHealth }) => {
  // State to track which phases are expanded/collapsed
  const [expandedPhases, setExpandedPhases] = useState(new Set());
  const getStatusIcon = (status) => {
    switch (status) {
      case 'complete':
        return <CheckCircleIcon className="h-5 w-5 text-green-400" />;
      case 'failed':
        return <XCircleIcon className="h-5 w-5 text-red-400" />;
      case 'in_progress':
        return <ClockOutline className="h-5 w-5 text-blue-400 animate-spin" />;
      case 'pending':
        return <ClockOutline className="h-5 w-5 text-gray-500" />;
      default:
        return null;
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'complete':
        return 'text-green-400';
      case 'failed':
        return 'text-red-400';
      case 'in_progress':
        return 'text-blue-400';
      case 'pending':
        return 'text-gray-400';
      default:
        return 'text-gray-400';
    }
  };

  const formatTime = (timestamp) => {
    if (!timestamp) return 'N/A';
    return new Date(timestamp * 1000).toLocaleTimeString();
  };

  const formatDuration = (seconds) => {
    if (!seconds) return '0s';
    if (seconds < 60) return `${seconds.toFixed(1)}s`;
    const minutes = Math.floor(seconds / 60);
    const secs = (seconds % 60).toFixed(0);
    return `${minutes}m ${secs}s`;
  };

  // Phase grouping for better visualization
  const phases = [
    {
      name: 'Connection & Health Checks',
      steps: ['orderbook_health', 'trader_client_health', 'fill_listener_health', 'event_bus_health']
    },
    {
      name: 'State Discovery & Sync',
      steps: ['sync_balance', 'sync_positions', 'sync_orders']
    },
    {
      name: 'Listener Verification',
      steps: ['verify_orderbook_subscriptions', 'verify_fill_listener_subscription', 'verify_listeners']
    },
    {
      name: 'Ready to Resume',
      steps: ['initialization_complete']
    }
  ];

  const getStepDetails = (stepId) => {
    if (!initializationStatus || !initializationStatus.steps) return null;
    return initializationStatus.steps[stepId];
  };

  const togglePhase = (phaseName) => {
    setExpandedPhases(prev => {
      const next = new Set(prev);
      if (next.has(phaseName)) {
        next.delete(phaseName);
      } else {
        next.add(phaseName);
      }
      return next;
    });
  };

  const isPhaseExpanded = (phaseName) => {
    return expandedPhases.has(phaseName);
  };

  // Calculate completion counts for each phase
  const getPhaseStats = (phase) => {
    const steps = phase.steps.map(stepId => getStepDetails(stepId)).filter(Boolean);
    const completed = steps.filter(step => step.status === 'complete').length;
    const total = steps.length;
    return { completed, total };
  };

  const isInitializing = initializationStatus && !initializationStatus.completed_at;
  
  // If no initialization status yet, show placeholder
  if (!initializationStatus) {
    return (
      <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
        <div className="text-center py-8">
          <ClockOutline className="h-12 w-12 text-gray-500 mx-auto mb-4 animate-pulse" />
          <div className="text-gray-400">Waiting for initialization to start...</div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Initialization Status Header */}
      <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-xl font-semibold text-gray-100">System Initialization</h3>
            {initializationStatus.completed_at ? (
              <div className="flex items-center space-x-2 px-4 py-2 bg-green-900/30 border border-green-700/50 rounded-lg">
                <CheckCircleIcon className="h-5 w-5 text-green-400" />
                <span className="font-medium text-green-400">
                  Ready
                </span>
              </div>
            ) : (
              <div className="flex items-center space-x-2 px-4 py-2 bg-blue-900/30 border border-blue-700/50 rounded-lg">
                <ClockOutline className="h-5 w-5 text-blue-400 animate-spin" />
                <span className="font-medium text-blue-400">
                  Initializing...
                </span>
              </div>
            )}
          </div>
          
          {initializationStatus.completed_at && (
            <div className="mb-6 p-4 bg-gray-900/50 rounded-lg border border-gray-700">
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-sm text-gray-400">Initialization completed in</div>
                  <div className="text-lg font-semibold text-gray-100">
                    {formatDuration(initializationStatus.duration_seconds)} at {formatTime(initializationStatus.completed_at)}
                  </div>
                </div>
                {initializationStatus.warnings && initializationStatus.warnings.length > 0 && (
                  <div className="flex items-start">
                    <ExclamationTriangleIcon className="h-5 w-5 text-yellow-400 mr-2 mt-0.5" />
                    <div>
                      <div className="font-semibold text-yellow-400 text-sm">Warnings:</div>
                      <ul className="list-disc list-inside text-xs text-yellow-300 mt-1">
                        {initializationStatus.warnings.map((warning, idx) => (
                          <li key={idx}>{warning}</li>
                        ))}
                      </ul>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Initialization Steps by Phase - 3 Column Grid */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {phases.map((phase, phaseIdx) => {
              const stats = getPhaseStats(phase);
              const isExpanded = isPhaseExpanded(phase.name);
              
              return (
                <div key={phaseIdx} className="bg-gray-900/30 rounded-lg border border-gray-700/50">
                  {/* Phase Header - Collapsible */}
                  <button
                    onClick={() => togglePhase(phase.name)}
                    className="w-full flex items-center justify-between p-4 hover:bg-gray-800/50 transition-colors rounded-t-lg"
                  >
                    <div className="flex items-center space-x-3">
                      {isExpanded ? (
                        <ChevronDownIcon className="h-5 w-5 text-gray-400" />
                      ) : (
                        <ChevronRightIcon className="h-5 w-5 text-gray-400" />
                      )}
                      <h4 className="text-sm font-semibold text-gray-300 uppercase tracking-wide">
                        {phase.name}
                      </h4>
                    </div>
                    <div className={`px-2 py-1 rounded text-xs font-semibold ${
                      stats.completed === stats.total
                        ? 'bg-green-900/50 text-green-400 border border-green-700/50'
                        : stats.completed > 0
                        ? 'bg-yellow-900/50 text-yellow-400 border border-yellow-700/50'
                        : 'bg-gray-800/50 text-gray-400 border border-gray-700/50'
                    }`}>
                      {stats.completed}/{stats.total}
                    </div>
                  </button>
                  
                  {/* Phase Content - Collapsible */}
                  {isExpanded && (
                    <div className="p-4 pt-0 space-y-3">
                      {phase.steps.map((stepId) => {
                        const step = getStepDetails(stepId);
                        if (!step) return null;

                        const statusBg = {
                          'complete': 'bg-green-900/20 border-green-700/30',
                          'failed': 'bg-red-900/20 border-red-700/30',
                          'in_progress': 'bg-blue-900/20 border-blue-700/30',
                          'pending': 'bg-gray-800/50 border-gray-700/50'
                        }[step.status] || 'bg-gray-800/50 border-gray-700/50';

                        // Format context details for card body
                        const formatContextValue = (key, value) => {
                          // Handle nested details object
                          if (key === 'details' && typeof value === 'object' && value !== null) {
                            return Object.entries(value).map(([nestedKey, nestedValue]) => {
                              return formatContextValue(nestedKey, nestedValue);
                            }).filter(Boolean);
                          }
                          if (key === 'balance' || key === 'balance_before') {
                            return `${key === 'balance' ? 'Cash Balance' : 'Previous Balance'}: $${Number(value).toFixed(2)}`;
                          }
                          if (key === 'portfolio_value') {
                            return `Portfolio Value: $${Number(value).toFixed(2)}`;
                          }
                          if (key === 'positions_count') {
                            return `Positions: ${value}`;
                          }
                          if (key === 'orders_in_kalshi' || key === 'orders_in_memory') {
                            return `${key === 'orders_in_kalshi' ? 'Kalshi' : 'Local'} Orders: ${value}`;
                          }
                          if (key === 'markets_subscribed' || key === 'snapshots_received' || key === 'deltas_received') {
                            return `${key === 'markets_subscribed' ? 'Markets' : key === 'snapshots_received' ? 'Snapshots' : 'Deltas'}: ${value}`;
                          }
                          if (key === 'api_url' || key === 'ws_url' || key === 'fill_listener_ws_url' || key === 'position_listener_ws_url') {
                            const label = key === 'api_url' ? 'API URL' : 
                                        key === 'ws_url' ? 'WebSocket URL' :
                                        key === 'fill_listener_ws_url' ? 'Fill Listener WS URL' :
                                        'Position Listener WS URL';
                            return `${label}: ${String(value)}`;
                          }
                          if (key === 'connected' || key === 'fill_listener_connected' || key === 'position_listener_connected') {
                            const label = key === 'connected' ? 'Connected' :
                                        key === 'fill_listener_connected' ? 'Fill Listener Connected' :
                                        'Position Listener Connected';
                            return `${label}: ${value ? 'true' : 'false'}`;
                          }
                          if (key === 'fill_listener' || key === 'position_listener') {
                            return `${key === 'fill_listener' ? 'Fill Listener' : 'Position Listener'}: ${value ? 'active' : 'inactive'}`;
                          }
                          if (key === 'total_steps' || key === 'completed_steps' || key === 'failed_steps') {
                            return `${key === 'total_steps' ? 'Total Steps' : key === 'completed_steps' ? 'Completed Steps' : 'Failed Steps'}: ${value}`;
                          }
                          if (key === 'warnings_count') {
                            return `Warnings: ${value}`;
                          }
                          if (typeof value === 'object' && value !== null) {
                            return null;
                          }
                          if (typeof value === 'boolean') {
                            return `${key}: ${value ? 'true' : 'false'}`;
                          }
                          return `${key}: ${String(value)}`;
                        };

                        return (
                          <div
                            key={stepId}
                            className={`rounded-lg border transition-all ${statusBg}`}
                          >
                            {/* Card Header */}
                            <div className="flex items-center space-x-2 p-3 border-b border-gray-700/50">
                              <div className="flex-shrink-0">
                                {getStatusIcon(step.status)}
                              </div>
                              <div className={`font-medium text-sm ${getStatusColor(step.status)} flex-1`}>
                                {step.name}
                              </div>
                            </div>
                            
                            {/* Card Body - Context Details */}
                            {step.details && Object.keys(step.details).length > 0 && (
                              <div className="p-3 text-xs text-gray-400 space-y-1.5">
                                {Object.entries(step.details).map(([key, value]) => {
                                  const formatted = formatContextValue(key, value);
                                  if (!formatted) return null;
                                  // Handle array of formatted values (from nested details)
                                  if (Array.isArray(formatted)) {
                                    return formatted.map((item, idx) => (
                                      <div key={`${key}-${idx}`} className="font-mono truncate">
                                        {item}
                                      </div>
                                    ));
                                  }
                                  return (
                                    <div key={key} className="font-mono truncate">
                                      {formatted}
                                    </div>
                                  );
                                })}
                              </div>
                            )}
                            
                            {/* Card Footer - Timestamp */}
                            {step.completed_at && (
                              <div className="px-3 py-2 border-t border-gray-700/50 text-xs text-gray-500">
                                {formatTime(step.completed_at)}
                              </div>
                            )}
                            
                            {/* Error Display */}
                            {step.error && (
                              <div className="px-3 py-2 border-t border-gray-700/50 text-xs text-red-400 font-medium">
                                {step.error}
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>

      {/* Component Health Status */}
      {componentHealth && Object.keys(componentHealth).length > 0 && (
        <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
          <h3 className="text-lg font-semibold text-gray-100 mb-4">Component Health Status</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {Object.entries(componentHealth).map(([component, health]) => (
              <div
                key={component}
                className={`p-4 rounded-lg border ${
                  health.status === 'healthy'
                    ? 'bg-green-900/20 border-green-700/30'
                    : health.status === 'unhealthy'
                    ? 'bg-red-900/20 border-red-700/30'
                    : 'bg-yellow-900/20 border-yellow-700/30'
                }`}
              >
                <div className="flex items-center justify-between mb-3">
                  <div className="font-medium text-gray-200 capitalize">{component.replace(/_/g, ' ')}</div>
                  {health.status === 'healthy' ? (
                    <CheckCircleIcon className="h-5 w-5 text-green-400" />
                  ) : (
                    <XCircleIcon className="h-5 w-5 text-red-400" />
                  )}
                </div>
                <div className="text-xs text-gray-400 space-y-1.5">
                  <div className="flex items-center justify-between">
                    <span>Status:</span>
                    <span className={`font-semibold ${
                      health.status === 'healthy' ? 'text-green-400' : 'text-red-400'
                    }`}>
                      {health.status}
                    </span>
                  </div>
                  {health.last_update && (
                    <div className="flex items-center justify-between">
                      <span>Last Update:</span>
                      <span className="font-mono">{formatTime(health.last_update)}</span>
                    </div>
                  )}
                  {health.details && health.details.api_url && (
                    <div className="pt-1 border-t border-gray-700">
                      <div className="text-gray-500 mb-1">API:</div>
                      <div className="truncate font-mono text-gray-300">{health.details.api_url}</div>
                    </div>
                  )}
                  {health.details && health.details.ws_url && (
                    <div className="pt-1 border-t border-gray-700">
                      <div className="text-gray-500 mb-1">WebSocket:</div>
                      <div className="truncate font-mono text-gray-300">{health.details.ws_url}</div>
                    </div>
                  )}
                  {health.details && health.details.markets_subscribed !== undefined && (
                    <div className="flex items-center justify-between">
                      <span>Markets:</span>
                      <span className="font-semibold text-gray-300">{health.details.markets_subscribed}</span>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default SystemHealth;

