import React from 'react';

const Header = ({ connectionStatus, ...props }) => {
  return (
    <header className="bg-gradient-to-r from-blue-900 to-indigo-900 text-white shadow-lg" {...props} data-testid={props['data-testid'] || "header"}>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo and Title */}
          <div className="flex items-center space-x-3" data-testid="header-logo">
            <div className="w-8 h-8 bg-gradient-to-r from-green-400 to-blue-500 rounded-lg flex items-center justify-center">
              <svg className="w-5 h-5 text-white" fill="currentColor" viewBox="0 0 20 20">
                <path d="M13 6a3 3 0 11-6 0 3 3 0 016 0zM18 8a2 2 0 11-4 0 2 2 0 014 0zM14 15a4 4 0 00-8 0v3h8v-3z" />
              </svg>
            </div>
            <div>
              <h1 className="text-xl font-bold" data-testid="header-title">Kalshi Flowboard</h1>
              <p className="text-blue-200 text-sm" data-testid="header-subtitle">Real-time Market Flow</p>
            </div>
          </div>

          {/* Connection Status */}
          <div className="flex items-center space-x-2" data-testid="connection-status">
            <div className={`w-3 h-3 rounded-full ${
              connectionStatus === 'connected' 
                ? 'bg-green-500 animate-pulse' 
                : connectionStatus === 'connecting'
                ? 'bg-yellow-500 animate-pulse'
                : 'bg-red-500'
            }`} data-testid="connection-indicator"></div>
            <span className="text-sm font-medium" data-testid="connection-status-text">
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