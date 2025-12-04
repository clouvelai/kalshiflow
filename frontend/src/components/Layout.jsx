import React from 'react';
import Header from './Header';

const Layout = ({ 
  children, 
  connectionStatus = 'disconnected', 
  tradeCount = 0 
}) => {
  return (
    <div className="min-h-screen bg-gray-50">
      <Header 
        connectionStatus={connectionStatus}
        tradeCount={tradeCount}
      />
      
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {children}
        </div>
      </main>
    </div>
  );
};

export default Layout;