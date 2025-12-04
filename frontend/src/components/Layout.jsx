import React from 'react';
import Header from './Header';

const Layout = ({ 
  children, 
  connectionStatus = 'disconnected'
}) => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100" data-testid="app-layout">
      <Header 
        connectionStatus={connectionStatus}
        data-testid="app-header"
      />
      
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6" data-testid="main-content">
        <div className="space-y-8">
          {children}
        </div>
      </main>
    </div>
  );
};

export default Layout;