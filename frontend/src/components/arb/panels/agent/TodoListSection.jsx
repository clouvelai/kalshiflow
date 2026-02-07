import React, { memo } from 'react';
import { ListTodo, Check, Loader2, Circle } from 'lucide-react';

/**
 * StatusIcon - Proper icon for todo status instead of unicode.
 */
const StatusIcon = ({ status }) => {
  if (status === 'completed') return <Check className="w-3 h-3 text-emerald-400 shrink-0" />;
  if (status === 'in_progress') return <Loader2 className="w-3 h-3 text-amber-400 animate-spin shrink-0" />;
  return <Circle className="w-3 h-3 text-gray-600 shrink-0" />;
};

/**
 * TodoListSection - Captain's task list.
 */
const TodoListSection = memo(({ todos }) => {
  if (!todos || todos.length === 0) return null;

  const completed = todos.filter(t => t.status === 'completed').length;

  return (
    <div className="rounded-lg border border-gray-800/30 bg-gray-900/25 p-2.5">
      <div className="flex items-center gap-2 mb-2">
        <ListTodo className="w-3 h-3 text-amber-500/70" />
        <span className="text-[10px] font-semibold text-gray-400 uppercase tracking-wider">
          Tasks
        </span>
        <span className="text-[10px] text-gray-600 font-mono tabular-nums ml-auto">
          {completed}/{todos.length}
        </span>
      </div>
      <div className="space-y-0.5 max-h-[150px] overflow-y-auto">
        {todos.map((todo, i) => (
          <div key={i} className="flex items-start gap-2 py-0.5 px-0.5">
            <div className="mt-0.5">
              <StatusIcon status={todo.status} />
            </div>
            <span className={`text-[11px] leading-relaxed ${
              todo.status === 'completed' ? 'text-gray-600 line-through' :
              todo.status === 'in_progress' ? 'text-gray-300' :
              'text-gray-400'
            }`}>
              {todo.text || todo.content || todo.description || JSON.stringify(todo)}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
});

TodoListSection.displayName = 'TodoListSection';

export default TodoListSection;
