import React, { memo } from 'react';
import { ChevronLeft, ChevronRight } from 'lucide-react';

const SidebarCollapseToggle = memo(({ collapsed, onToggle, side = 'left' }) => {
  const Icon = side === 'left'
    ? (collapsed ? ChevronRight : ChevronLeft)
    : (collapsed ? ChevronLeft : ChevronRight);

  return (
    <button
      onClick={onToggle}
      className="flex items-center justify-center w-6 h-6 rounded-md hover:bg-gray-800/50 transition-colors text-gray-500 hover:text-gray-300"
      title={collapsed ? 'Expand' : 'Collapse'}
    >
      <Icon className="w-3.5 h-3.5" />
    </button>
  );
});

SidebarCollapseToggle.displayName = 'SidebarCollapseToggle';

export default SidebarCollapseToggle;
