import React, { memo, useState, useEffect } from 'react';
import { Crosshair, Loader2, Wrench } from 'lucide-react';
import { TOOL_ICONS } from '../../utils/colorMaps';

/**
 * CommandoElapsed - Live elapsed timer.
 */
const CommandoElapsed = ({ startedAt, completedAt, active }) => {
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    if (!startedAt) return;
    if (!active && completedAt) {
      setElapsed(Math.round((completedAt - startedAt) / 1000));
      return;
    }
    const tick = () => setElapsed(Math.round((Date.now() - startedAt) / 1000));
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, [startedAt, completedAt, active]);

  return (
    <span className="text-[10px] font-mono text-gray-500 tabular-nums">
      {elapsed}s
    </span>
  );
};

/**
 * CommandoSessionCard - Single commando execution.
 */
const CommandoSessionCard = memo(({ session }) => {
  const { active, startedAt, completedAt, error, prompt, ops } = session;

  const borderColor = active
    ? 'border-amber-500/15'
    : error
      ? 'border-red-500/12'
      : 'border-emerald-500/12';

  const bgColor = active
    ? 'bg-amber-950/10'
    : error
      ? 'bg-red-950/8'
      : 'bg-emerald-950/8';

  return (
    <div className={`rounded-lg border ${borderColor} ${bgColor} p-3`}>
      {/* Header */}
      <div className="flex items-center gap-2 mb-2">
        <Crosshair className={`w-3.5 h-3.5 ${
          active ? 'text-amber-400' : error ? 'text-red-400' : 'text-emerald-400'
        }`} />
        <span className="text-[10px] font-semibold text-gray-300 uppercase tracking-wider">
          Commando
        </span>
        {active && <Loader2 className="w-3 h-3 text-amber-400 animate-spin" />}
        <span className={`ml-auto px-1.5 py-px rounded text-[8px] font-semibold uppercase tracking-wider ${
          active ? 'bg-amber-500/15 text-amber-300/80' :
          error ? 'bg-red-500/15 text-red-300/80' :
          'bg-emerald-500/15 text-emerald-300/80'
        }`}>
          {active ? 'Running' : error ? 'Error' : 'Done'}
        </span>
        <CommandoElapsed startedAt={startedAt} completedAt={completedAt} active={active} />
      </div>

      {/* Prompt */}
      {prompt && (
        <div className="text-[10px] text-gray-500 mb-2 truncate font-mono">
          {prompt.slice(0, 120)}
        </div>
      )}

      {/* Ops */}
      {ops.length > 0 && (
        <div className="space-y-0.5 max-h-[110px] overflow-y-auto">
          {ops.map((op, i) => {
            const Icon = TOOL_ICONS[op.tool_name] || Wrench;
            const isCall = op.type === 'call';
            return (
              <div key={`${op.id}-${i}`} className="flex items-center gap-2 py-0.5">
                <Icon className={`w-2.5 h-2.5 ${isCall ? 'text-cyan-500/60' : 'text-gray-600'} shrink-0`} />
                <span className="text-[10px] font-mono text-gray-400 shrink-0">
                  {op.tool_name}
                </span>
                <span className={`text-[9px] truncate flex-1 ${isCall ? 'text-gray-500' : 'text-gray-600'}`}>
                  {isCall ? (op.tool_input || '').slice(0, 60) : (op.tool_output || '').slice(0, 60)}
                </span>
              </div>
            );
          })}
        </div>
      )}

      {ops.length === 0 && active && (
        <div className="text-[10px] text-gray-600 flex items-center gap-1.5">
          <Loader2 className="w-2.5 h-2.5 animate-spin" />
          Preparing...
        </div>
      )}
    </div>
  );
});
CommandoSessionCard.displayName = 'CommandoSessionCard';

/**
 * CommandoSection - List of commando sessions.
 */
const CommandoSection = memo(({ sessions }) => {
  if (!sessions || sessions.length === 0) return null;

  return (
    <div className="space-y-2">
      {sessions.map(session => (
        <CommandoSessionCard key={session.id} session={session} />
      ))}
    </div>
  );
});
CommandoSection.displayName = 'CommandoSection';

export { CommandoElapsed, CommandoSessionCard, CommandoSection };
export default CommandoSection;
