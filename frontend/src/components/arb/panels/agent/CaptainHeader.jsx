import React, { memo, useState, useEffect } from 'react';
import { Brain, Zap, Pause, AlertTriangle, Activity, Clock, Timer, WifiOff } from 'lucide-react';
import { CAPTAIN_MODE_STYLES } from '../../utils/colorMaps';

/**
 * NextRunCountdown - Shows countdown to next strategic or deep_scan cycle.
 *
 * Ticks every second to show a live countdown. Falls back to "Waiting for signal"
 * when the system hasn't reported any timing yet.
 */
const NextRunCountdown = memo(({ captainTiming, isRunning, captainPaused }) => {
  const [now, setNow] = useState(Date.now());

  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, []);

  if (captainPaused) {
    return (
      <div className="flex items-center gap-1">
        <Pause className="w-2.5 h-2.5 text-amber-400/60" />
        <span className="text-[9px] text-amber-400/80 font-mono">Paused</span>
      </div>
    );
  }

  if (isRunning) {
    return (
      <div className="flex items-center gap-1">
        <Activity className="w-2.5 h-2.5 text-violet-400/60 animate-pulse" />
        <span className="text-[9px] text-violet-400/80 font-mono">Running</span>
      </div>
    );
  }

  if (!captainTiming?.lastStrategic && !captainTiming?.lastDeepScan) {
    return (
      <div className="flex items-center gap-1">
        <Timer className="w-2.5 h-2.5 text-gray-600" />
        <span className="text-[9px] text-gray-600 font-mono">Waiting for signal</span>
      </div>
    );
  }

  // Calculate seconds until next strategic and deep_scan
  const strategicMs = captainTiming.lastStrategic
    ? captainTiming.lastStrategic + (captainTiming.strategicInterval * 1000) - now
    : null;
  const deepScanMs = captainTiming.lastDeepScan
    ? captainTiming.lastDeepScan + (captainTiming.deepScanInterval * 1000) - now
    : null;

  // Pick whichever is sooner
  let nextMs = null;
  let nextLabel = '';
  if (strategicMs != null && deepScanMs != null) {
    if (deepScanMs <= strategicMs) {
      nextMs = deepScanMs;
      nextLabel = 'Deep';
    } else {
      nextMs = strategicMs;
      nextLabel = 'Strat';
    }
  } else if (strategicMs != null) {
    nextMs = strategicMs;
    nextLabel = 'Strat';
  } else if (deepScanMs != null) {
    nextMs = deepScanMs;
    nextLabel = 'Deep';
  }

  if (nextMs == null || nextMs <= 0) {
    return (
      <div className="flex items-center gap-1">
        <Timer className="w-2.5 h-2.5 text-emerald-400/60 animate-pulse" />
        <span className="text-[9px] text-emerald-400/80 font-mono">Due now</span>
      </div>
    );
  }

  const totalSec = Math.ceil(nextMs / 1000);
  const min = Math.floor(totalSec / 60);
  const sec = totalSec % 60;
  const display = min > 0
    ? `${min}:${sec.toString().padStart(2, '0')}`
    : `${sec}s`;

  const isUrgent = totalSec <= 30;
  const textColor = isUrgent ? 'text-amber-400/80' : 'text-gray-500';
  const iconColor = isUrgent ? 'text-amber-400/60' : 'text-gray-600';

  return (
    <div className="flex items-center gap-1" title={`Next ${nextLabel === 'Deep' ? 'deep scan' : 'strategic'} cycle in ${display}`}>
      <Clock className={`w-2.5 h-2.5 ${iconColor}`} />
      <span className={`text-[9px] font-mono tabular-nums ${textColor}`}>
        {nextLabel} {display}
      </span>
    </div>
  );
});
NextRunCountdown.displayName = 'NextRunCountdown';

const CaptainHeader = memo(({ isRunning, cycleCount, captainPaused, exchangeStatus, captainMode, captainTiming, attentionCount, connectionStatus }) => {
  const exchangeDown = exchangeStatus && exchangeStatus.active === false;
  const isPaused = captainPaused;
  const mode = captainMode?.mode;
  const isDisconnected = connectionStatus === 'disconnected' || connectionStatus === 'error';

  const getBadgeStyle = () => {
    if (isDisconnected) return 'bg-gray-800/40 text-gray-600 border border-gray-700/20';
    if (exchangeDown) return 'bg-red-500/15 text-red-300 border border-red-500/20';
    if (isPaused) return 'bg-amber-500/15 text-amber-300 border border-amber-500/20';
    if (isRunning && mode) {
      const modeStyle = CAPTAIN_MODE_STYLES[mode];
      if (modeStyle) return `${modeStyle.bg} ${modeStyle.text} border ${modeStyle.border}`;
    }
    if (isRunning) return 'bg-violet-500/15 text-violet-300 border border-violet-500/20';
    return 'bg-gray-800/40 text-gray-500 border border-gray-700/20';
  };

  const getBadgeText = () => {
    if (isDisconnected) return 'Offline';
    if (exchangeDown) return 'Exchange Down';
    if (isPaused) return 'Paused';
    if (isRunning && mode === 'reactive') return 'Reactive';
    if (isRunning && mode === 'strategic') return 'Strategic';
    if (isRunning && mode === 'deep_scan') return 'Deep Scan';
    if (isRunning) return 'Active';
    return 'Observing';
  };

  const getBadgeIcon = () => {
    if (isDisconnected) return <WifiOff className="w-2 h-2" />;
    if (exchangeDown) return <AlertTriangle className="w-2 h-2" />;
    if (isPaused) return <Pause className="w-2 h-2" />;
    if (isRunning && mode === 'reactive') return <Zap className="w-2 h-2" />;
    if (isRunning) return <Activity className="w-2 h-2" />;
    return null;
  };

  return (
    <div className="flex items-center justify-between px-3 py-2.5 border-b border-gray-800/30 shrink-0">
      <div className="flex items-center gap-2">
        <div className={`p-1 rounded-md transition-colors duration-200 ${
          isDisconnected ? 'bg-gray-800/40' :
          exchangeDown ? 'bg-red-500/15 shadow-sm shadow-red-500/20' :
          isRunning && mode === 'reactive' ? 'bg-amber-500/15 shadow-sm shadow-amber-500/20' :
          isRunning ? 'bg-violet-500/15 shadow-sm shadow-violet-500/20' : 'bg-gray-800/40'
        }`}>
          <Brain className={`w-3.5 h-3.5 transition-colors duration-200 ${
            isDisconnected ? 'text-gray-600' :
            exchangeDown ? 'text-red-400' :
            isRunning && mode === 'reactive' ? 'text-amber-400' :
            isRunning ? 'text-violet-400' : 'text-gray-500'
          }`} />
        </div>
        <div className="flex items-center gap-1.5">
          <span className="text-[12px] font-semibold text-gray-200">Captain</span>
        </div>
      </div>
      <div className="flex items-center gap-2.5">
        {/* Next run countdown */}
        {!isDisconnected && (
          <NextRunCountdown captainTiming={captainTiming} isRunning={isRunning} captainPaused={isPaused} />
        )}
        {/* Attention signal count when idle */}
        {attentionCount > 0 && !isRunning && (
          <div className="flex items-center gap-1">
            <div className="w-1.5 h-1.5 rounded-full bg-amber-500 animate-pulse" />
            <span className="text-[9px] text-amber-400/80 font-mono tabular-nums">{attentionCount}</span>
          </div>
        )}
        {cycleCount > 0 && (
          <span data-testid="agent-cycle-count" className="text-[9px] text-gray-600 font-mono tabular-nums">
            C{cycleCount}
          </span>
        )}
        <span
          data-testid="agent-running-badge"
          data-running={isRunning}
          className={`flex items-center gap-1 px-1.5 py-0.5 rounded-full text-[8px] font-semibold uppercase tracking-wider transition-colors duration-200 ${getBadgeStyle()}`}
        >
          {getBadgeIcon()}
          {getBadgeText()}
        </span>
      </div>
    </div>
  );
});

CaptainHeader.displayName = 'CaptainHeader';

export default CaptainHeader;
