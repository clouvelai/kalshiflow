import React, { useState } from 'react';
import { faqData } from '../data/faqData';

const ChevronIcon = ({ isOpen, className = "" }) => (
  <svg 
    className={`w-5 h-5 transition-transform duration-300 ${isOpen ? 'rotate-180' : ''} ${className}`}
    fill="none" 
    stroke="currentColor" 
    viewBox="0 0 24 24"
  >
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
  </svg>
);

const FAQ = () => {
  const [openItems, setOpenItems] = useState(new Set());

  const toggleItem = (itemId) => {
    setOpenItems(prev => {
      const newSet = new Set(prev);
      if (newSet.has(itemId)) {
        newSet.delete(itemId);
      } else {
        newSet.add(itemId);
      }
      return newSet;
    });
  };

  const formatAnswerText = (text) => {
    return text.split('\n').map((line, index) => {
      // Handle section headers (lines ending with colon)
      if (line.trim().endsWith(':') && !line.trim().startsWith('•')) {
        return (
          <h4 key={index} className="font-semibold text-gray-800 mt-4 mb-3 text-base">
            {line.trim()}
          </h4>
        );
      }
      // Handle bullet points
      if (line.trim().startsWith('•')) {
        const content = line.trim().substring(1).trim();
        // Check if bullet point contains example text in parentheses or key-value pattern
        const hasExample = content.includes('(') || content.includes(':');
        return (
          <li key={index} className="ml-6 mb-3 flex items-start">
            <span className="text-green-500 mr-3 mt-1 font-bold text-sm">▸</span>
            <span className="flex-1 text-gray-700 leading-relaxed">
              {hasExample ? (
                // Parse content with examples or key phrases
                content.split(/(\([^)]+\)|(?:^|\s)[A-Z][^:]+:|[-+]?\$\d+k)/).map((part, i) => {
                  if (part.startsWith('(') && part.endsWith(')')) {
                    return <span key={i} className="text-gray-500 text-sm italic">{part}</span>;
                  }
                  if (part.includes('$')) {
                    return <span key={i} className="font-mono font-semibold text-blue-600">{part}</span>;
                  }
                  if (part.endsWith(':') && /^[A-Z]/.test(part.trim())) {
                    return <span key={i} className="font-semibold text-gray-800">{part}</span>;
                  }
                  return part;
                })
              ) : content}
            </span>
          </li>
        );
      }
      // Handle empty lines
      if (line.trim() === '') {
        return <div key={index} className="mb-3" />;
      }
      // Regular paragraph
      return (
        <p key={index} className="mb-3 text-gray-700 leading-relaxed">
          {line.trim()}
        </p>
      );
    });
  };

  return (
    <div className="bg-white rounded-xl shadow-lg border border-gray-200 overflow-hidden">
      {/* FAQ Header */}
      <div className="bg-gradient-to-r from-blue-600 to-blue-700 px-4 sm:px-6 py-6 text-center">
        <h2 className="text-xl sm:text-2xl font-bold text-white mb-2">
          Frequently Asked Questions
        </h2>
        <p className="text-blue-100 text-sm">
          Everything you need to know about Kalshi Flowboard
        </p>
      </div>

      {/* FAQ Items */}
      <div className="divide-y divide-gray-200">
        {faqData.map((item, index) => {
          const isOpen = openItems.has(item.id);
          
          return (
            <div key={item.id} className="group">
              {/* Question Header */}
              <button
                onClick={() => toggleItem(item.id)}
                className="w-full px-4 sm:px-6 py-4 text-left focus:outline-none focus:bg-gray-50 hover:bg-gray-50 transition-colors duration-200"
                aria-expanded={isOpen}
                aria-controls={`faq-answer-${item.id}`}
              >
                <div className="flex items-center justify-between">
                  <h3 className="text-base sm:text-lg font-semibold text-gray-900 pr-4">
                    {item.question}
                  </h3>
                  <ChevronIcon 
                    isOpen={isOpen} 
                    className="text-gray-400 group-hover:text-gray-600 flex-shrink-0"
                  />
                </div>
              </button>

              {/* Answer Content */}
              <div
                id={`faq-answer-${item.id}`}
                className={`overflow-hidden transition-all duration-300 ease-in-out ${
                  isOpen ? 'opacity-100' : 'max-h-0 opacity-0'
                }`}
                style={{
                  maxHeight: isOpen ? '2000px' : '0'
                }}
              >
                <div className="px-4 sm:px-8 pb-6 pt-2">
                  <div className="border-l-4 border-blue-100 pl-6 py-2">
                    <div className="text-gray-600 leading-relaxed space-y-0 text-sm sm:text-base">
                      {formatAnswerText(item.answer)}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Contact Info Footer */}
      <div className="bg-gray-50 px-4 sm:px-6 py-6 border-t border-gray-200">
        <div className="text-center mb-4">
          <h3 className="text-lg font-semibold text-gray-900 mb-1">
            Samuel Clark
          </h3>
          <p className="text-sm text-gray-600">
            Get in touch with the developer
          </p>
        </div>
        
        <div className="space-y-3 max-w-md mx-auto">
          {/* Email */}
          <div className="group">
            <a
              href="mailto:samuelmacarthurclark@gmail.com"
              className="flex items-center gap-3 p-3 bg-white hover:bg-blue-50 border border-gray-200 hover:border-blue-300 rounded-lg transition-all duration-200 text-decoration-none"
            >
              <svg className="w-4 h-4 text-gray-500 group-hover:text-blue-600 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 8l7.89 7.89a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
              </svg>
              <div className="flex-1 text-left min-w-0">
                <div className="font-medium text-gray-900 group-hover:text-blue-700 text-sm truncate">
                  Email
                </div>
                <div className="text-xs text-gray-500 group-hover:text-blue-500 truncate">
                  samuelmacarthurclark@gmail.com
                </div>
              </div>
            </a>
          </div>

          {/* Website */}
          <div className="group">
            <a
              href="https://www.samuelclark.dev/"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-3 p-3 bg-white hover:bg-purple-50 border border-gray-200 hover:border-purple-300 rounded-lg transition-all duration-200 text-decoration-none"
            >
              <svg className="w-4 h-4 text-gray-500 group-hover:text-purple-600 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9v-9m0-9v9m0 9c-5 0-9-4-9-9s4-9 9-9" />
              </svg>
              <div className="flex-1 text-left min-w-0">
                <div className="font-medium text-gray-900 group-hover:text-purple-700 text-sm truncate">
                  Portfolio
                </div>
                <div className="text-xs text-gray-500 group-hover:text-purple-500 truncate">
                  samuelclark.dev
                </div>
              </div>
              <svg className="w-3 h-3 text-gray-400 group-hover:text-purple-500 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
              </svg>
            </a>
          </div>

          {/* LinkedIn */}
          <div className="group">
            <a
              href="https://www.linkedin.com/in/samclark77/"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-3 p-3 bg-white hover:bg-blue-50 border border-gray-200 hover:border-blue-300 rounded-lg transition-all duration-200 text-decoration-none"
            >
              <svg className="w-4 h-4 text-gray-500 group-hover:text-blue-600 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 8a6 6 0 016 6v7h-4v-7a2 2 0 00-2-2 2 2 0 00-2 2v7h-4v-7a6 6 0 016-6zM2 9h4v12H2z" />
                <circle cx="4" cy="4" r="2" strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} />
              </svg>
              <div className="flex-1 text-left min-w-0">
                <div className="font-medium text-gray-900 group-hover:text-blue-700 text-sm truncate">
                  LinkedIn
                </div>
                <div className="text-xs text-gray-500 group-hover:text-blue-500 truncate">
                  Connect with me
                </div>
              </div>
              <svg className="w-3 h-3 text-gray-400 group-hover:text-blue-500 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
              </svg>
            </a>
          </div>
        </div>
      </div>
    </div>
  );
};

export default FAQ;