import React from 'react';

const SystemHealth = ({ 
  initializationStatus, 
  componentHealth, 
  traderStatus, 
  traderStatusHistory = [],
  stateTransitions = [] 
}) => {
  // Helper function to format timestamp
  const formatTimestamp = (timestamp) => {
    if (!timestamp) return '--:--:--';
    const date = new Date(timestamp * 1000);
    return date.toLocaleTimeString('en-US', { 
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    });
  };

  // Helper function to get status color
  const getStatusColor = (status) => {
    // Ensure status is a string
    const statusStr = String(status || '').toLowerCase();
    
    switch (statusStr) {
      case 'ready':
      case 'healthy':
      case 'complete':
        return 'text-green-400';
      case 'calibrating':
      case 'syncing':
      case 'starting':
        return 'text-blue-400';
      case 'idle':
      case 'pending':
        return 'text-gray-400';
      case 'acting':
      case 'trading':
        return 'text-yellow-400';
      case 'error':
      case 'failed':
        return 'text-red-400';
      default:
        return 'text-gray-400';
    }
  };

  // Helper to format status history entry
  const formatStatusEntry = (entry) => {
    if (!entry) return null;
    
    // Extract relevant information based on entry type
    if (entry.entry_type === 'state_transition') {
      const from = entry.details?.from_state || 'unknown';
      const to = entry.details?.to_state || 'unknown';
      const reason = entry.details?.reason || '';
      const timeInState = entry.details?.time_in_previous_state;
      
      return (
        <span>
          <span className="text-gray-500">{from}</span>
          <span className="text-gray-400 mx-1">→</span>
          <span className={getStatusColor(to)}>{to}</span>
          {reason && <span className="text-gray-500 ml-2">({reason})</span>}
          {timeInState && <span className="text-gray-600 ml-2">[{timeInState.toFixed(1)}s]</span>}
        </span>
      );
    }
    
    if (entry.entry_type === 'calibration_step') {
      const step = entry.details?.step || 'unknown';
      const status = entry.details?.status || 'unknown';
      const duration = entry.details?.duration;
      const error = entry.details?.error;
      
      return (
        <span>
          <span className="text-blue-400">Calibration:</span>
          <span className="text-gray-400 ml-2">{step}</span>
          <span className={`ml-2 ${getStatusColor(status)}`}>{status}</span>
          {duration && <span className="text-gray-600 ml-2">({duration.toFixed(1)}s)</span>}
          {error && <span className="text-red-400 ml-2">{error}</span>}
        </span>
      );
    }
    
    if (entry.entry_type === 'service_status') {
      const service = entry.details?.service || 'unknown';
      const status = entry.details?.status || 'unknown';
      const details = entry.details?.details;
      
      return (
        <span>
          <span className="text-purple-400">{service}:</span>
          <span className={`ml-2 ${getStatusColor(status)}`}>{status}</span>
          {details?.error && <span className="text-red-400 ml-2">{details.error}</span>}
        </span>
      );
    }
    
    if (entry.entry_type === 'action_result') {
      const action = entry.details?.action || 'unknown';
      const result = entry.details?.result || 'unknown';
      const duration = entry.details?.duration;
      const error = entry.details?.error;
      
      return (
        <span>
          <span className="text-cyan-400">{action}:</span>
          <span className={`ml-2 ${error ? 'text-red-400' : 'text-gray-400'}`}>{result}</span>
          {duration && <span className="text-gray-600 ml-2">({duration.toFixed(1)}s)</span>}
          {error && <span className="text-red-400 ml-2">{error}</span>}
        </span>
      );
    }
    
    // Default fallback
    return <span className="text-gray-400">{JSON.stringify(entry.details || entry)}</span>;
  };

  return (
    <div className="space-y-4">
      {/* Trader Status Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Current Status */}
        <div className="bg-gray-700/30 rounded-lg p-4 border border-gray-700">
          <h3 className="text-sm font-semibold text-gray-300 mb-3 uppercase tracking-wider">
            TRADER STATUS
          </h3>
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-gray-500 text-sm">Current State:</span>
              <span className={`font-mono font-medium text-lg ${getStatusColor(traderStatus?.current_status)}`}>
                {traderStatus?.current_status || 'unknown'}
              </span>
            </div>
            {traderStatus?.last_transition && (
              <div className="text-xs text-gray-500 mt-2 pt-2 border-t border-gray-700/50">
                <div>From: {traderStatus.last_transition.from_state}</div>
                <div>To: {traderStatus.last_transition.to_state}</div>
                {traderStatus.last_transition.reason && (
                  <div>Reason: {traderStatus.last_transition.reason}</div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Status History */}
        <div className="bg-gray-700/30 rounded-lg p-4 border border-gray-700">
          <h3 className="text-sm font-semibold text-gray-300 mb-3 uppercase tracking-wider">
            STATUS HISTORY
          </h3>
          <div className="space-y-1 max-h-40 overflow-y-auto">
            {traderStatusHistory.length > 0 ? (
              traderStatusHistory.slice().reverse().slice(-10).map((entry, idx) => (
                <div key={idx} className="text-xs flex items-start space-x-2">
                  <span className="text-gray-600 font-mono flex-shrink-0">
                    {formatTimestamp(entry.timestamp)}
                  </span>
                  <span className="flex-1">
                    {formatStatusEntry(entry)}
                  </span>
                </div>
              ))
            ) : (
              <div className="text-gray-500 text-xs">No status history available</div>
            )}
          </div>
        </div>
      </div>

      {/* Component Health Grid */}
      {Object.keys(componentHealth).length > 0 && (
        <div className="bg-gray-700/30 rounded-lg p-4 border border-gray-700">
          <h3 className="text-sm font-semibold text-gray-300 mb-3 uppercase tracking-wider">
            COMPONENT HEALTH
          </h3>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            {Object.entries(componentHealth).map(([component, health]) => (
              <div key={component} className="bg-gray-800/50 rounded-lg p-3 border border-gray-700/50">
                <div className="text-xs text-gray-500 mb-1">{component}</div>
                <div className={`text-sm font-medium ${getStatusColor(health.status)}`}>
                  {health.status}
                </div>
                {health.details && (
                  <div className="text-xs text-gray-600 mt-1">
                    {JSON.stringify(health.details).slice(0, 50)}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* State Transitions */}
      {stateTransitions.length > 0 && (
        <div className="bg-gray-700/30 rounded-lg p-4 border border-gray-700">
          <h3 className="text-sm font-semibold text-gray-300 mb-3 uppercase tracking-wider">
            STATE TRANSITIONS
          </h3>
          <div className="space-y-2 max-h-60 overflow-y-auto">
            {stateTransitions.slice(0, 20).map((transition) => (
              <div key={transition.id} className="text-xs p-2 bg-gray-800/50 rounded border border-gray-700/50">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-gray-600 font-mono">
                    {formatTimestamp(transition.timestamp)}
                  </span>
                  {transition.time_in_previous_state && (
                    <span className="text-gray-600">
                      Duration: {transition.time_in_previous_state.toFixed(1)}s
                    </span>
                  )}
                </div>
                <div className="flex items-center space-x-2">
                  <span className={getStatusColor(transition.from_state)}>
                    {transition.from_state}
                  </span>
                  <span className="text-gray-500">→</span>
                  <span className={getStatusColor(transition.to_state)}>
                    {transition.to_state}
                  </span>
                  {transition.reason && (
                    <span className="text-gray-500 ml-2">({transition.reason})</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Initialization/Calibration Status */}
      {initializationStatus && (
        <div className="bg-gray-700/30 rounded-lg p-4 border border-gray-700">
          <h3 className="text-sm font-semibold text-gray-300 mb-3 uppercase tracking-wider">
            INITIALIZATION STATUS
          </h3>
          <div className="space-y-3">
            {initializationStatus.is_complete ? (
              <div className="text-green-400 text-sm">
                ✅ Initialization Complete
                {initializationStatus.duration_seconds && (
                  <span className="text-gray-500 ml-2">
                    ({initializationStatus.duration_seconds.toFixed(1)}s)
                  </span>
                )}
              </div>
            ) : (
              <div className="text-blue-400 text-sm animate-pulse">
                ⏳ Initializing...
              </div>
            )}
            
            {initializationStatus.steps && Object.keys(initializationStatus.steps).length > 0 && (
              <div className="space-y-1">
                {Object.entries(initializationStatus.steps).map(([stepId, step]) => (
                  <div key={stepId} className="text-xs flex items-center justify-between">
                    <span className="text-gray-400">{step.step_name || stepId}:</span>
                    <span className={`${
                      step.status === 'complete' ? 'text-green-400' :
                      step.status === 'error' ? 'text-red-400' :
                      step.status === 'running' ? 'text-blue-400' :
                      'text-gray-500'
                    }`}>
                      {step.status}
                      {step.duration && <span className="text-gray-600 ml-1">({step.duration.toFixed(1)}s)</span>}
                    </span>
                  </div>
                ))}
              </div>
            )}
            
            {initializationStatus.warnings && initializationStatus.warnings.length > 0 && (
              <div className="mt-2 space-y-1">
                {initializationStatus.warnings.map((warning, idx) => (
                  <div key={idx} className="text-xs text-amber-400">
                    ⚠️ {warning}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default SystemHealth;