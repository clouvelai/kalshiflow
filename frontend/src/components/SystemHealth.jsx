import React, { useState } from 'react';
import { CheckCircleIcon, XCircleIcon, ClockIcon, ExclamationTriangleIcon } from '@heroicons/react/24/solid';
import { CheckCircleIcon as CheckCircleOutline, XCircleIcon as XCircleOutline, ClockIcon as ClockOutline, ChevronDownIcon, ChevronRightIcon } from '@heroicons/react/24/outline';
import TraderStatePanel from './TraderStatePanel';

const SystemHealth = ({ initializationStatus, componentHealth, traderStatus, traderStatusHistory }) => {
  // Phase grouping for better visualization (defined early for useState initialization)
  const phases = [
    {
      name: 'Connection & Health Checks',
      steps: ['orderbook_health', 'trader_client_health', 'fill_listener_health', 'position_listener_health', 'event_bus_health']
    },
    {
      name: 'State Discovery & Sync',
      steps: ['sync_balance', 'sync_positions', 'sync_settlements', 'sync_orders']
    },
    {
      name: 'Listener Verification',
      steps: ['verify_orderbook_subscriptions', 'verify_fill_listener_subscription', 'verify_position_listener_subscription', 'verify_listeners']
    }
  ];

  // Initialize expandedPhases as empty set (collapsed by default)
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

  // Calculate summary stats for hero view dynamically from steps
  const allSteps = initializationStatus.steps || {};
  const stepEntries = Object.values(allSteps);
  
  // Calculate total expected steps from all phases (including initialization_complete)
  const allExpectedStepIds = phases.flatMap(phase => phase.steps);
  allExpectedStepIds.push('initialization_complete'); // Add the final step that's not in phases anymore
  const expectedTotalSteps = allExpectedStepIds.length;
  
  // Use the count of actual steps if we have them, otherwise use expected total or provided total_steps
  const actualStepCount = stepEntries.length;
  const totalSteps = actualStepCount > 0 ? actualStepCount : 
                     (initializationStatus.total_steps || expectedTotalSteps);
  
  const completedSteps = stepEntries.filter(step => step.status === 'complete').length;
  const failedSteps = stepEntries.filter(step => step.status === 'failed').length;
  const inProgressSteps = stepEntries.filter(step => step.status === 'in_progress').length;
  const hasFailures = failedSteps > 0;
  const isComplete = initializationStatus.completed_at !== null && !hasFailures;

  return (
    <div className="space-y-6">
      {/* Hero View - System Status */}
      <div className={`rounded-lg border p-6 ${
        isComplete 
          ? 'bg-gradient-to-r from-green-900/20 to-green-800/10 border-green-700/50' 
          : hasFailures
          ? 'bg-gradient-to-r from-red-900/20 to-red-800/10 border-red-700/50'
          : 'bg-gradient-to-r from-blue-900/20 to-blue-800/10 border-blue-700/50'
      }`}>
        <div className="flex items-start justify-between">
          <div className="flex items-start space-x-4 flex-1">
            {/* Status Icon */}
            <div className="flex-shrink-0 mt-1">
              {isComplete ? (
                <CheckCircleIcon className="h-8 w-8 text-green-400" />
              ) : hasFailures ? (
                <XCircleIcon className="h-8 w-8 text-red-400" />
              ) : (
                <ClockOutline className="h-8 w-8 text-blue-400 animate-spin" />
              )}
            </div>

            {/* Status Content */}
            <div className="flex-1">
              <div className="flex items-center gap-3 mb-2">
                <h2 className={`text-2xl font-bold ${
                  isComplete ? 'text-green-400' : hasFailures ? 'text-red-400' : 'text-blue-400'
                }`}>
                  {isComplete 
                    ? 'System Checks and Calibration Complete' 
                    : hasFailures
                    ? 'Initialization Failed'
                    : 'Initializing System...'}
                </h2>
                {traderStatus?.current_status?.includes('low_cash') && (
                  <div className="flex items-center space-x-1.5 px-2.5 py-1 bg-red-500/10 border border-red-500/30 rounded-md">
                    <span className="text-red-400 text-xs">⚠️</span>
                    <span className="text-red-400 text-xs font-medium">Low Cash</span>
                  </div>
                )}
              </div>

              {/* Show progress during initialization or summary on completion */}
              {initializationStatus.completed_at ? (
                <div className="space-y-3">
                  {/* Summary Stats - Shown on completion */}
                  <div className="flex flex-wrap gap-4 text-sm">
                    <div className="flex items-center space-x-2">
                      <span className="text-gray-400">Completed:</span>
                      <span className={`font-semibold ${isComplete ? 'text-green-400' : 'text-gray-300'}`}>
                        {completedSteps}/{totalSteps} steps
                      </span>
                    </div>
                    {failedSteps > 0 && (
                      <div className="flex items-center space-x-2">
                        <span className="text-gray-400">Failed:</span>
                        <span className="font-semibold text-red-400">{failedSteps}</span>
                      </div>
                    )}
                    <div className="flex items-center space-x-2">
                      <span className="text-gray-400">Duration:</span>
                      <span className="font-semibold text-gray-300">
                        {formatDuration(initializationStatus.duration_seconds)}
                      </span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <span className="text-gray-400">Completed at:</span>
                      <span className="font-mono text-gray-300">
                        {formatTime(initializationStatus.completed_at)}
                      </span>
                    </div>
                  </div>

                  {/* Trading Session Details */}
                  {(initializationStatus.starting_cash !== undefined || 
                    initializationStatus.starting_portfolio_value !== undefined ||
                    initializationStatus.positions_resumed !== undefined ||
                    initializationStatus.orders_resumed !== undefined ||
                    initializationStatus.markets_trading) && (
                    <div className="pt-2 border-t border-gray-700/50">
                      <div className="text-sm text-gray-400 font-semibold mb-2">Session Details:</div>
                      <div className="flex flex-wrap gap-4 text-sm">
                        {initializationStatus.starting_cash !== undefined && (
                          <div className="flex items-center space-x-2">
                            <span className="text-gray-400">Starting Cash:</span>
                            <span className="font-semibold text-gray-300">
                              ${Number(initializationStatus.starting_cash).toFixed(2)}
                            </span>
                          </div>
                        )}
                        {initializationStatus.starting_portfolio_value !== undefined && (
                          <div className="flex items-center space-x-2">
                            <span className="text-gray-400">Starting Portfolio:</span>
                            <span className="font-semibold text-gray-300">
                              ${Number(initializationStatus.starting_portfolio_value).toFixed(2)}
                            </span>
                          </div>
                        )}
                        {initializationStatus.positions_resumed !== undefined && (
                          <div className="flex items-center space-x-2">
                            <span className="text-gray-400">Positions Resumed:</span>
                            <span className="font-semibold text-gray-300">
                              {initializationStatus.positions_resumed}
                            </span>
                          </div>
                        )}
                        {initializationStatus.orders_resumed !== undefined && (
                          <div className="flex items-center space-x-2">
                            <span className="text-gray-400">Orders Resumed:</span>
                            <span className="font-semibold text-gray-300">
                              {initializationStatus.orders_resumed}
                            </span>
                          </div>
                        )}
                        {initializationStatus.markets_trading && (
                          <div className="flex items-center space-x-2">
                            <span className="text-gray-400">Markets:</span>
                            <span className="font-semibold text-gray-300">
                              {Array.isArray(initializationStatus.markets_trading) 
                                ? initializationStatus.markets_trading.length 
                                : initializationStatus.markets_trading}
                            </span>
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Warnings */}
                  {initializationStatus.warnings && initializationStatus.warnings.length > 0 && (
                    <div className="pt-2 border-t border-gray-700/50">
                      <div className="flex items-start space-x-2">
                        <ExclamationTriangleIcon className="h-5 w-5 text-yellow-400 mt-0.5 flex-shrink-0" />
                        <div>
                          <div className="font-semibold text-yellow-400 text-sm mb-1">Warnings:</div>
                          <ul className="text-sm text-yellow-300 space-y-0.5">
                            {initializationStatus.warnings.map((warning, idx) => (
                              <li key={idx}>• {warning}</li>
                            ))}
                          </ul>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Failed Steps Details */}
                  {hasFailures && (
                    <div className="pt-2 border-t border-red-700/50">
                      <div className="text-sm text-red-400 font-semibold mb-2">Failed Steps:</div>
                      <ul className="text-sm text-red-300 space-y-1">
                        {Object.entries(initializationStatus.steps || {}).map(([stepId, step]) => {
                          if (step.status === 'failed') {
                            return (
                              <li key={stepId} className="flex items-start space-x-2">
                                <span>•</span>
                                <span>{step.name}: {step.error || 'Unknown error'}</span>
                              </li>
                            );
                          }
                          return null;
                        })}
                      </ul>
                    </div>
                  )}
                </div>
              ) : (
                <div className="space-y-2">
                  {/* Progress during initialization */}
                  <div className="text-gray-400 text-sm">
                    System initialization in progress...
                  </div>
                  <div className="flex flex-wrap gap-4 text-sm">
                    <div className="flex items-center space-x-2">
                      <span className="text-gray-400">Completed:</span>
                      <span className="font-semibold text-green-400">
                        {completedSteps}/{totalSteps} steps
                      </span>
                    </div>
                    {inProgressSteps > 0 && (
                      <div className="flex items-center space-x-2">
                        <span className="text-gray-400">In progress:</span>
                        <span className="font-semibold text-blue-400">{inProgressSteps}</span>
                      </div>
                    )}
                    {failedSteps > 0 && (
                      <div className="flex items-center space-x-2">
                        <span className="text-gray-400">Failed:</span>
                        <span className="font-semibold text-red-400">{failedSteps}</span>
                      </div>
                    )}
                    {initializationStatus.started_at && (
                      <div className="flex items-center space-x-2">
                        <span className="text-gray-400">Started at:</span>
                        <span className="font-mono text-gray-300">
                          {formatTime(initializationStatus.started_at)}
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

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

      {/* Trader Status */}
      <TraderStatePanel 
        state={null}
        executionStats={null}
        showExecutionStats={false}
        showPositions={false}
        showOrders={false}
        showActionBreakdown={false}
        showPortfolioStats={false}
        showSessionCashflow={false}
        traderStatus={traderStatus}
        traderStatusHistory={traderStatusHistory}
      />
    </div>
  );
};

export default SystemHealth;

