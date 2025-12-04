import React, { createContext, useContext } from 'react';
import useTradeData from '../hooks/useTradeData';

const TradeContext = createContext();

export const useTradeContext = () => {
  const context = useContext(TradeContext);
  if (!context) {
    throw new Error('useTradeContext must be used within a TradeProvider');
  }
  return context;
};

export const TradeProvider = ({ children }) => {
  const tradeData = useTradeData();

  return (
    <TradeContext.Provider value={tradeData}>
      {children}
    </TradeContext.Provider>
  );
};

export default TradeContext;