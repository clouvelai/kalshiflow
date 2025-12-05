import React from 'react';

const Header = ({ connectionStatus, ...props }) => {
  return (
    <header className="bg-gradient-to-r from-slate-900 via-slate-800 to-slate-900 text-white shadow-xl border-b border-slate-700/50 backdrop-blur-sm" {...props} data-testid={props['data-testid'] || "header"}>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-20">
          {/* Logo and Title */}
          <div className="flex items-center space-x-4" data-testid="header-logo">
            <div className="w-10 h-10 bg-gradient-to-r from-blue-500 to-cyan-400 rounded-xl flex items-center justify-center shadow-lg shadow-blue-500/25">
              <svg className="w-6 h-6 text-white" fill="currentColor" viewBox="0 0 24 24">
                <path d="M3 13h8V3H9v6H5V3H3v10zm10-6h8V5h-8v2zm0 6h8v-2h-8v2zm0 6h8v-2h-8v2zM3 21h8v-6H9v2H5v-2H3v6z"/>
              </svg>
            </div>
            <div className="space-y-0.5">
              <h1 className="text-2xl font-semibold tracking-tight" data-testid="header-title">Kalshi Flowboard</h1>
              <p className="text-slate-300 text-sm font-medium tracking-wide" data-testid="header-subtitle">Real-time Market Analytics</p>
            </div>
          </div>

          {/* Connection Status */}
          <div className="flex items-center space-x-3" data-testid="connection-status">
            <div className={`w-2.5 h-2.5 rounded-full ${
              connectionStatus === 'connected' 
                ? 'bg-emerald-400 shadow-lg shadow-emerald-400/40 animate-pulse' 
                : connectionStatus === 'connecting'
                ? 'bg-amber-400 shadow-lg shadow-amber-400/40 animate-pulse'
                : 'bg-red-400 shadow-lg shadow-red-400/40'
            }`} data-testid="connection-indicator"></div>
            <span className={`px-3 py-1.5 rounded-full text-sm font-semibold tracking-wide ${
              connectionStatus === 'connected'
                ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                : connectionStatus === 'connecting'
                ? 'bg-amber-500/20 text-amber-300 border border-amber-500/30'
                : 'bg-red-500/20 text-red-300 border border-red-500/30'
            }`} data-testid="connection-status-text">
              {connectionStatus === 'connected' 
                ? 'Live' 
                : connectionStatus === 'connecting'
                ? 'Connecting'
                : 'Disconnected'}
            </span>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;