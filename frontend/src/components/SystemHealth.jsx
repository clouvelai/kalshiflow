import React from 'react';
import { CheckCircleIcon, XCircleIcon, ClockIcon, ExclamationTriangleIcon } from '@heroicons/react/24/solid';
import { CheckCircleIcon as CheckCircleOutline, XCircleIcon as XCircleOutline, ClockIcon as ClockOutline } from '@heroicons/react/24/outline';

const SystemHealth = ({ initializationStatus, componentHealth }) => {
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
                  <div className="text-lg font-semibold text-gray-100">{formatDuration(initializationStatus.duration_seconds)}</div>
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

          {/* Initialization Steps by Phase */}
          <div className="space-y-6">
            {phases.map((phase, phaseIdx) => (
              <div key={phaseIdx} className="border-b border-gray-700 pb-6 last:border-b-0 last:pb-0">
                <h4 className="text-sm font-semibold text-gray-300 mb-4 uppercase tracking-wide">{phase.name}</h4>
                <div className="space-y-2">
                  {phase.steps.map((stepId) => {
                    const step = getStepDetails(stepId);
                    if (!step) return null;

                    const statusBg = {
                      'complete': 'bg-green-900/20 border-green-700/30',
                      'failed': 'bg-red-900/20 border-red-700/30',
                      'in_progress': 'bg-blue-900/20 border-blue-700/30',
                      'pending': 'bg-gray-800/50 border-gray-700/50'
                    }[step.status] || 'bg-gray-800/50 border-gray-700/50';

                    return (
                      <div
                        key={stepId}
                        className={`flex items-start justify-between p-4 rounded-lg border transition-all ${statusBg}`}
                      >
                        <div className="flex items-start space-x-4 flex-1">
                          <div className="mt-0.5">
                            {getStatusIcon(step.status)}
                          </div>
                          <div className="flex-1 min-w-0">
                            <div className={`font-medium text-sm ${getStatusColor(step.status)} mb-1`}>
                              {step.name}
                            </div>
                            {step.details && Object.keys(step.details).length > 0 && (
                              <div className="text-xs text-gray-400 mt-2 space-y-1">
                                {Object.entries(step.details).map(([key, value]) => {
                                  if (key === 'details' && typeof value === 'object') {
                                    return null;
                                  }
                                  if (key === 'balance' || key === 'balance_before') {
                                    return (
                                      <div key={key} className="font-mono">
                                        {key === 'balance' ? 'Cash Balance' : 'Previous Balance'}: ${Number(value).toFixed(2)}
                                      </div>
                                    );
                                  }
                                  if (key === 'portfolio_value') {
                                    return (
                                      <div key={key} className="font-mono">
                                        Portfolio Value: ${Number(value).toFixed(2)}
                                      </div>
                                    );
                                  }
                                  if (key === 'positions_count') {
                                    return (
                                      <div key={key}>
                                        Positions: {value}
                                      </div>
                                    );
                                  }
                                  if (key === 'orders_in_kalshi' || key === 'orders_in_memory') {
                                    return (
                                      <div key={key}>
                                        {key === 'orders_in_kalshi' ? 'Kalshi' : 'Local'} Orders: {value}
                                      </div>
                                    );
                                  }
                                  if (key === 'markets_subscribed' || key === 'snapshots_received' || key === 'deltas_received') {
                                    return (
                                      <div key={key}>
                                        {key === 'markets_subscribed' ? 'Markets' : key === 'snapshots_received' ? 'Snapshots' : 'Deltas'}: {value}
                                      </div>
                                    );
                                  }
                                  if (typeof value === 'object') {
                                    return null;
                                  }
                                  return (
                                    <div key={key} className="truncate">
                                      {key}: {String(value)}
                                    </div>
                                  );
                                })}
                              </div>
                            )}
                            {step.error && (
                              <div className="text-xs text-red-400 mt-2 font-medium">{step.error}</div>
                            )}
                          </div>
                        </div>
                        {step.completed_at && (
                          <div className="text-xs text-gray-500 ml-4 whitespace-nowrap">
                            {formatTime(step.completed_at)}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>
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

