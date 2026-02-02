import { useState, useCallback, useRef } from 'react';
import { formatConsoleTimestamp } from '../../utils/v3-trader';

/**
 * useConsoleMessages - Hook for managing console message state
 * Handles message buffering, deduplication, and expansion state
 */
export const useConsoleMessages = () => {
  const [messages, setMessages] = useState([]);
  const [expandedMessages, setExpandedMessages] = useState(new Set());
  const lastMessageRef = useRef(null);
  const messageIdCounter = useRef(0);

  /**
   * Add a message to the console
   */
  const addMessage = useCallback((type, content, metadata = {}) => {
    // Skip messages with UNKNOWN or undefined state
    if (metadata.state === 'UNKNOWN' || metadata.state === 'unknown' || metadata.state === 'undefined') {
      console.log('Skipping message with UNKNOWN state:', content, metadata);
      return;
    }

    // Deduplicate rapid repeated messages
    const messageKey = `${type}-${content}-${metadata.state || ''}`;
    const now = Date.now();

    if (lastMessageRef.current) {
      const { key: lastKey, time: lastTime, content: lastContent } = lastMessageRef.current;
      if ((lastKey === messageKey && (now - lastTime) < 1000) ||
          (lastContent === content && (now - lastTime) < 1000)) {
        return;
      }
    }

    lastMessageRef.current = { key: messageKey, time: now, content: content };

    const timestamp = formatConsoleTimestamp();

    // Parse state transition info
    let fromState = null;
    let toState = null;
    let isTransition = false;

    const contentStr = typeof content === 'string' ? content : String(content ?? '');
    if (metadata.to_state || contentStr.includes('\u2192')) {
      isTransition = true;
      if (metadata.from_state && metadata.to_state) {
        fromState = metadata.from_state;
        toState = metadata.to_state;
      } else if (contentStr.includes('\u2192')) {
        const match = contentStr.match(/(\w+)\s*\u2192\s*(\w+)/);
        if (match) {
          fromState = match[1];
          toState = match[2];
        }
      }
    }

    // Extract status from the message
    let status = null;
    if (contentStr.includes('SUCCESS')) status = 'SUCCESS';
    else if (contentStr.includes('FAILED')) status = 'FAILED';
    else if (contentStr.includes('ERROR')) status = 'ERROR';
    else if (contentStr.includes('READY')) status = 'READY';
    else if (contentStr.includes('INITIALIZING')) status = 'INITIALIZING';
    else if (contentStr.includes('CONNECTING')) status = 'CONNECTING';
    else if (contentStr.includes('CALIBRATING')) status = 'CALIBRATING';

    // Clean up the content for display
    let cleanContent = contentStr;
    if (isTransition && fromState && toState) {
      cleanContent = contentStr
        .replace(new RegExp(`${fromState}\\s*\u2192\\s*${toState}:?\\s*`, 'gi'), '')
        .replace(/\u2192\s*State:\s*/gi, '')
        .replace(/State:\s*/gi, '')
        .replace(/\u2192\s*\w+/gi, '')
        .trim();

      if (cleanContent.toLowerCase() === toState.toLowerCase() ||
          cleanContent.toLowerCase() === fromState.toLowerCase() ||
          cleanContent === 'State' ||
          cleanContent === '\u2192') {
        cleanContent = '';
      }
    }

    // Format metadata for display
    const formattedMetadata = {};
    if (metadata.metadata && typeof metadata.metadata === 'object') {
      for (const [key, value] of Object.entries(metadata.metadata)) {
        if (key === 'markets' && Array.isArray(value)) {
          formattedMetadata[key] = value.join(', ');
        } else if (typeof value === 'object') {
          formattedMetadata[key] = JSON.stringify(value, null, 2);
        } else {
          formattedMetadata[key] = value;
        }
      }
    }

    if (metadata.context && metadata.context !== 'State transition') {
      formattedMetadata.context = metadata.context;
    }

    messageIdCounter.current += 1;
    const newMessage = {
      id: `${Date.now()}-${messageIdCounter.current}`,
      type,
      content: cleanContent,
      timestamp,
      metadata: {
        ...metadata,
        formattedMetadata: Object.keys(formattedMetadata).length > 0 ? formattedMetadata : null,
        isTransition,
        fromState,
        toState,
        status,
        state: metadata.state || metadata.to_state || toState
      }
    };

    setMessages(prev => [...prev.slice(-100), newMessage]);

    // Auto-expand messages with metadata
    if (Object.keys(formattedMetadata).length > 0) {
      setExpandedMessages(prev => {
        const newExpanded = new Set(prev);
        newExpanded.add(newMessage.id);
        return newExpanded;
      });
    }
  }, []);

  /**
   * Toggle message expansion
   */
  const toggleMessageExpansion = useCallback((messageId) => {
    setExpandedMessages(prev => {
      const newSet = new Set(prev);
      if (newSet.has(messageId)) {
        newSet.delete(messageId);
      } else {
        newSet.add(messageId);
      }
      return newSet;
    });
  }, []);

  /**
   * Clear all messages
   */
  const clearMessages = useCallback(() => {
    setMessages([]);
    setExpandedMessages(new Set());
    lastMessageRef.current = null;
  }, []);

  return {
    messages,
    expandedMessages,
    addMessage,
    toggleMessageExpansion,
    clearMessages
  };
};

export default useConsoleMessages;
