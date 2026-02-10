import React, { memo, useMemo } from 'react';
import { ListTodo, Check, Loader2, Circle, AlertTriangle } from 'lucide-react';

/**
 * StatusIcon - Proper icon for todo status.
 */
const StatusIcon = ({ status }) => {
  if (status === 'completed') return <Check className="w-3 h-3 text-emerald-400 shrink-0" />;
  if (status === 'in_progress') return <Loader2 className="w-3 h-3 text-amber-400 animate-spin shrink-0" />;
  if (status === 'stale') return <AlertTriangle className="w-3 h-3 text-red-400 shrink-0" />;
  return <Circle className="w-3 h-3 text-gray-600 shrink-0" />;
};

/**
 * PriorityBadge - Small badge for high/low priority tasks.
 */
const PriorityBadge = ({ priority }) => {
  if (priority === 3) return <span className="text-[9px] font-bold text-rose-400/80 mr-1">H</span>;
  if (priority === 1) return <span className="text-[9px] font-bold text-gray-500/80 mr-1">L</span>;
  return null;
};

const STATUS_ORDER = { in_progress: 0, pending: 1, stale: 2, completed: 3 };

/**
 * TodoListSection - Captain's enriched task list.
 * Sorted by priority (high first), then status (in_progress > pending > stale > completed).
 */
const TodoListSection = memo(({ todos }) => {
  if (!todos || todos.length === 0) return null;

  const sorted = useMemo(() => {
    return [...todos].sort((a, b) => {
      const prioDiff = (b.priority || 2) - (a.priority || 2);
      if (prioDiff !== 0) return prioDiff;
      return (STATUS_ORDER[a.status] ?? 9) - (STATUS_ORDER[b.status] ?? 9);
    });
  }, [todos]);

  const completed = todos.filter(t => t.status === 'completed').length;
  const stale = todos.filter(t => t.status === 'stale').length;

  return (
    <div className="rounded-lg border border-gray-800/30 bg-gray-900/25 p-2.5">
      <div className="flex items-center gap-2 mb-2">
        <ListTodo className="w-3 h-3 text-amber-500/70" />
        <span className="text-[10px] font-semibold text-gray-400 uppercase tracking-wider">
          Tasks
        </span>
        <span className="text-[10px] text-gray-600 font-mono tabular-nums ml-auto">
          {completed}/{todos.length}
          {stale > 0 && <span className="text-red-400/70 ml-1">({stale} stale)</span>}
        </span>
      </div>
      <div className="space-y-0.5 max-h-[200px] overflow-y-auto">
        {sorted.map((todo, i) => (
          <div key={todo.id || i} className={`flex items-start gap-2 py-0.5 px-0.5 rounded ${
            todo.status === 'stale' ? 'bg-red-950/10' : ''
          }`}>
            <div className="mt-0.5">
              <StatusIcon status={todo.status} />
            </div>
            <div className="flex-1 min-w-0">
              <span className={`text-[11px] leading-relaxed ${
                todo.status === 'completed' ? 'text-gray-600 line-through' :
                todo.status === 'stale' ? 'text-red-400/80' :
                todo.status === 'in_progress' ? 'text-gray-300' :
                'text-gray-400'
              }`}>
                <PriorityBadge priority={todo.priority} />
                {todo.text || todo.content || todo.description || JSON.stringify(todo)}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
});

TodoListSection.displayName = 'TodoListSection';

export default TodoListSection;
