import React, { memo } from 'react';
import { AlertCircle, ShieldOff, X } from 'lucide-react';

/**
 * WatchdogToast - Shows notification when the deep agent watchdog detects issues
 *
 * Warning (amber): Cycle task died and was restarted, shows restart counter
 * Critical (rose): Agent permanently stopped after too many restarts
 * Positioned at bottom-44 to stack above OrderCancelledToast
 */
const WatchdogToast = ({ event, onDismiss }) => {
  if (!event) return null;

  const isCritical = event.severity === 'critical' || event.permanentlyStopped;

  const bgClass = isCritical
    ? 'bg-rose-900/90 border-rose-500'
    : 'bg-amber-900/90 border-amber-500';

  const Icon = isCritical ? ShieldOff : AlertCircle;
  const iconColor = isCritical ? 'text-rose-400' : 'text-amber-400';
  const titleColor = isCritical ? 'text-rose-300' : 'text-amber-300';

  return (
    <div className={`fixed bottom-44 right-4 p-4 rounded-lg shadow-lg z-50 border backdrop-blur-sm max-w-sm ${bgClass} animate-toast-enter`}>
      <div className="flex items-start space-x-3">
        <Icon className={`w-5 h-5 ${iconColor} flex-shrink-0 mt-0.5`} />
        <div className="flex-1 min-w-0">
          <div className="text-sm font-medium text-white">
            {isCritical ? 'Agent Permanently Stopped' : 'Watchdog Restart'}
          </div>
          <div className={`text-xs ${titleColor} mt-0.5 line-clamp-2`}>
            {event.message}
          </div>
          {!isCritical && (
            <div className="text-xs font-mono text-amber-400/80 mt-1">
              Restarts: {event.restartsThisHour}/{event.maxRestartsPerHour}
            </div>
          )}
        </div>
        <button
          onClick={onDismiss}
          className="text-gray-400 hover:text-white flex-shrink-0"
        >
          <X className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
};

export default memo(WatchdogToast);
