import React from 'react';

/**
 * Renders markdown text into formatted JSX with visual hierarchy.
 * Handles headers, subheaders, bold text, bullets, and preserves emojis.
 *
 * Shared between DeepAgentPanel (Trader tab) and AgentPage (dedicated view).
 */
const renderThinkingMarkdown = (text) => {
  if (!text) return null;

  const lines = text.split('\n');
  const elements = [];
  let currentList = [];
  let listKey = 0;

  const flushList = () => {
    if (currentList.length > 0) {
      elements.push(
        <ul key={`list-${listKey++}`} className="space-y-1 ml-4 my-2">
          {currentList}
        </ul>
      );
      currentList = [];
    }
  };

  const renderInlineFormatting = (lineText) => {
    // Handle **bold** text
    const parts = lineText.split(/(\*\*[^*]+\*\*)/g);
    return parts.map((part, i) => {
      if (part.startsWith('**') && part.endsWith('**')) {
        return (
          <strong key={i} className="text-gray-200 font-semibold">
            {part.slice(2, -2)}
          </strong>
        );
      }
      return part;
    });
  };

  lines.forEach((line, index) => {
    const trimmed = line.trim();

    // Skip empty lines but preserve spacing
    if (!trimmed) {
      flushList();
      elements.push(<div key={`space-${index}`} className="h-2" />);
      return;
    }

    // ## Header (large, with border)
    if (trimmed.startsWith('## ')) {
      flushList();
      const headerText = trimmed.slice(3);
      elements.push(
        <div
          key={`h2-${index}`}
          className="text-violet-300 font-semibold text-sm border-b border-violet-700/30 pb-1 mb-2 mt-3 first:mt-0"
        >
          {renderInlineFormatting(headerText)}
        </div>
      );
      return;
    }

    // ### Subheader (medium, subtle background)
    if (trimmed.startsWith('### ')) {
      flushList();
      const subheaderText = trimmed.slice(4);
      elements.push(
        <div
          key={`h3-${index}`}
          className="text-violet-400 font-medium text-xs bg-violet-900/30 px-2 py-1 rounded mt-3 mb-1"
        >
          {renderInlineFormatting(subheaderText)}
        </div>
      );
      return;
    }

    // Bullet points (- item)
    if (trimmed.startsWith('- ')) {
      const bulletText = trimmed.slice(2);
      currentList.push(
        <li
          key={`bullet-${index}`}
          className="text-sm text-gray-300 flex items-start gap-2"
        >
          <span className="text-violet-500 mt-1.5 text-[8px]">●</span>
          <span className="flex-1">{renderInlineFormatting(bulletText)}</span>
        </li>
      );
      return;
    }

    // Numbered list (1. item, 2. item, etc.)
    const numberedMatch = trimmed.match(/^(\d+)\.\s+(.+)$/);
    if (numberedMatch) {
      currentList.push(
        <li
          key={`num-${index}`}
          className="text-sm text-gray-300 flex items-start gap-2"
        >
          <span className="text-violet-400 font-mono text-xs min-w-[16px]">{numberedMatch[1]}.</span>
          <span className="flex-1">{renderInlineFormatting(numberedMatch[2])}</span>
        </li>
      );
      return;
    }

    // Regular text
    flushList();
    elements.push(
      <div key={`text-${index}`} className="text-sm text-gray-300 my-0.5">
        {renderInlineFormatting(trimmed)}
      </div>
    );
  });

  // Flush any remaining list items
  flushList();

  return elements;
};

export default renderThinkingMarkdown;
