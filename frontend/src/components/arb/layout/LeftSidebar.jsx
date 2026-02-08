import React, { memo } from 'react';
import { motion } from 'framer-motion';
import { Layers } from 'lucide-react';
import SidebarCollapseToggle from './SidebarCollapseToggle';
import PortfolioSummary from '../panels/PortfolioSummary';
import EventList from '../panels/EventList';
import { LEFT_SIDEBAR_WIDTH, LEFT_SIDEBAR_COLLAPSED, PANEL_DIVIDER } from '../utils/styleConstants';

const LeftSidebar = memo(({ collapsed, onToggle, tradingState, events, selectedEventTicker, onSelectEvent }) => {
  return (
    <motion.div
      className={`shrink-0 flex flex-col bg-gray-950/60 ${PANEL_DIVIDER} overflow-hidden`}
      animate={{ width: collapsed ? LEFT_SIDEBAR_COLLAPSED : LEFT_SIDEBAR_WIDTH }}
      transition={{ type: 'spring', stiffness: 400, damping: 30 }}
    >
      {/* Toggle bar */}
      <div className="flex items-center justify-between px-2 py-1.5 border-b border-gray-800/30 shrink-0">
        {!collapsed && (
          <span className="text-[10px] font-semibold text-gray-500 uppercase tracking-wider pl-1">Portfolio</span>
        )}
        <SidebarCollapseToggle collapsed={collapsed} onToggle={onToggle} side="left" />
      </div>

      {collapsed ? (
        /* Icon rail */
        <div className="flex flex-col items-center gap-3 py-3">
          <Layers className="w-4 h-4 text-gray-600" />
        </div>
      ) : (
        <>
          <PortfolioSummary tradingState={tradingState} />
          <EventList
            events={events}
            selectedEventTicker={selectedEventTicker}
            onSelectEvent={onSelectEvent}
          />
        </>
      )}
    </motion.div>
  );
});

LeftSidebar.displayName = 'LeftSidebar';

export default LeftSidebar;
