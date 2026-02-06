import React from 'react';

/**
 * Renders Captain thinking output into formatted JSX.
 * Optimized for Captain's actual output patterns:
 * - Headers (## / ###) for cycle summaries and sections
 * - Bold (**text**) for emphasis on key findings
 * - Bullets (- item) for lists of observations/actions
 * - Inline code (`ticker`) for market tickers and values
 * - Code blocks for structured data (orderbook snapshots, etc.)
 *
 * Removed: table support (Captain doesn't output tables)
 */
const renderThinkingMarkdown = (text) => {
  if (!text) return null;

  const lines = text.split('\n');
  const elements = [];
  let currentList = [];
  let listKey = 0;
  let codeBlockLines = null;
  let codeBlockLang = '';

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
    // Handle **bold** and `inline code` (common for tickers like `INXD-25FEB05-T3475`)
    const parts = lineText.split(/(\*\*[^*]+\*\*|`[^`]+`)/g);
    return parts.map((part, i) => {
      if (part.startsWith('**') && part.endsWith('**')) {
        return (
          <strong key={i} className="text-gray-200 font-semibold">
            {part.slice(2, -2)}
          </strong>
        );
      }
      if (part.startsWith('`') && part.endsWith('`')) {
        return (
          <code key={i} className="px-1 py-0.5 bg-gray-800/60 rounded text-cyan-300 text-[11px] font-mono">
            {part.slice(1, -1)}
          </code>
        );
      }
      return part;
    });
  };

  lines.forEach((line, index) => {
    const trimmed = line.trim();

    // Handle code block start/end (for orderbook snapshots, JSON data)
    if (trimmed.startsWith('```')) {
      if (codeBlockLines === null) {
        codeBlockLines = [];
        codeBlockLang = trimmed.slice(3).trim();
        flushList();
      } else {
        elements.push(
          <div key={`code-${index}`} className="my-2">
            {codeBlockLang && (
              <div className="text-[9px] text-gray-500 uppercase tracking-wider mb-1 font-mono">
                {codeBlockLang}
              </div>
            )}
            <pre className="text-[11px] font-mono bg-gray-950/80 border border-gray-800/50 rounded-lg p-3 overflow-x-auto text-gray-300 leading-relaxed">
              {codeBlockLines.join('\n')}
            </pre>
          </div>
        );
        codeBlockLines = null;
        codeBlockLang = '';
      }
      return;
    }

    // Inside code block - collect lines
    if (codeBlockLines !== null) {
      codeBlockLines.push(line);
      return;
    }

    // Empty lines - preserve spacing
    if (!trimmed) {
      flushList();
      elements.push(<div key={`space-${index}`} className="h-2" />);
      return;
    }

    // Horizontal rule (--- for section breaks)
    if (/^-{3,}$/.test(trimmed)) {
      flushList();
      elements.push(
        <hr key={`hr-${index}`} className="border-t border-gray-700/50 my-3" />
      );
      return;
    }

    // ## Header - Cycle summaries, major sections
    if (trimmed.startsWith('## ')) {
      flushList();
      elements.push(
        <div
          key={`h2-${index}`}
          className="text-violet-300 font-semibold text-sm border-b border-violet-700/30 pb-1 mb-2 mt-3 first:mt-0"
        >
          {renderInlineFormatting(trimmed.slice(3))}
        </div>
      );
      return;
    }

    // ### Subheader - Subsections like "Key Findings", "Next Steps"
    if (trimmed.startsWith('### ')) {
      flushList();
      elements.push(
        <div
          key={`h3-${index}`}
          className="text-violet-400 font-medium text-xs bg-violet-900/30 px-2 py-1 rounded mt-3 mb-1"
        >
          {renderInlineFormatting(trimmed.slice(4))}
        </div>
      );
      return;
    }

    // Bullet points (- item) - observations, action items
    if (trimmed.startsWith('- ')) {
      currentList.push(
        <li
          key={`bullet-${index}`}
          className="text-sm text-gray-300 flex items-start gap-2"
        >
          <span className="text-violet-500 mt-1.5 text-[8px]">●</span>
          <span className="flex-1">{renderInlineFormatting(trimmed.slice(2))}</span>
        </li>
      );
      return;
    }

    // Numbered list (1. item) - ordered steps
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

  // Flush any remaining list
  flushList();

  return elements;
};

export default renderThinkingMarkdown;
