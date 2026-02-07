import React from 'react';

/**
 * Renders Captain thinking output into formatted JSX.
 *
 * Patterns: Headers (##/###), bold (**text**), bullets (- item),
 * inline code (`ticker`), code blocks, numbered lists.
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
        <ul key={`list-${listKey++}`} className="space-y-0.5 ml-3 my-1.5">
          {currentList}
        </ul>
      );
      currentList = [];
    }
  };

  const renderInline = (lineText) => {
    const parts = lineText.split(/(\*\*[^*]+\*\*|`[^`]+`)/g);
    return parts.map((part, i) => {
      if (part.startsWith('**') && part.endsWith('**')) {
        return (
          <strong key={i} className="text-gray-200 font-medium">
            {part.slice(2, -2)}
          </strong>
        );
      }
      if (part.startsWith('`') && part.endsWith('`')) {
        return (
          <code key={i} className="px-1 py-px bg-gray-800/50 rounded text-cyan-300/90 text-[11px] font-mono">
            {part.slice(1, -1)}
          </code>
        );
      }
      return part;
    });
  };

  lines.forEach((line, index) => {
    const trimmed = line.trim();

    // Code block boundaries
    if (trimmed.startsWith('```')) {
      if (codeBlockLines === null) {
        codeBlockLines = [];
        codeBlockLang = trimmed.slice(3).trim();
        flushList();
      } else {
        elements.push(
          <div key={`code-${index}`} className="my-2">
            {codeBlockLang && (
              <div className="text-[8px] text-gray-500 uppercase tracking-wider mb-1 font-mono">
                {codeBlockLang}
              </div>
            )}
            <pre className="text-[11px] font-mono bg-gray-950/60 border border-gray-800/30 rounded-lg p-3 overflow-x-auto text-gray-300 leading-relaxed">
              {codeBlockLines.join('\n')}
            </pre>
          </div>
        );
        codeBlockLines = null;
        codeBlockLang = '';
      }
      return;
    }

    if (codeBlockLines !== null) {
      codeBlockLines.push(line);
      return;
    }

    // Empty line
    if (!trimmed) {
      flushList();
      elements.push(<div key={`space-${index}`} className="h-1.5" />);
      return;
    }

    // Horizontal rule
    if (/^-{3,}$/.test(trimmed)) {
      flushList();
      elements.push(
        <hr key={`hr-${index}`} className="border-t border-gray-700/30 my-2.5" />
      );
      return;
    }

    // ## Header
    if (trimmed.startsWith('## ')) {
      flushList();
      elements.push(
        <div
          key={`h2-${index}`}
          className="text-violet-300/90 font-medium text-[13px] border-b border-violet-800/20 pb-1 mb-1.5 mt-2.5 first:mt-0"
        >
          {renderInline(trimmed.slice(3))}
        </div>
      );
      return;
    }

    // ### Subheader
    if (trimmed.startsWith('### ')) {
      flushList();
      elements.push(
        <div
          key={`h3-${index}`}
          className="text-violet-400/80 font-medium text-[12px] bg-violet-900/20 px-2 py-0.5 rounded mt-2 mb-1"
        >
          {renderInline(trimmed.slice(4))}
        </div>
      );
      return;
    }

    // Bullet points
    if (trimmed.startsWith('- ')) {
      currentList.push(
        <li key={`bullet-${index}`} className="text-[12px] text-gray-300 flex items-start gap-2">
          <span className="text-violet-500/60 mt-[7px] text-[6px]">&#9679;</span>
          <span className="flex-1 leading-relaxed">{renderInline(trimmed.slice(2))}</span>
        </li>
      );
      return;
    }

    // Numbered list
    const numberedMatch = trimmed.match(/^(\d+)\.\s+(.+)$/);
    if (numberedMatch) {
      currentList.push(
        <li key={`num-${index}`} className="text-[12px] text-gray-300 flex items-start gap-2">
          <span className="text-violet-400/60 font-mono text-[11px] min-w-[14px]">{numberedMatch[1]}.</span>
          <span className="flex-1 leading-relaxed">{renderInline(numberedMatch[2])}</span>
        </li>
      );
      return;
    }

    // Regular text
    flushList();
    elements.push(
      <div key={`text-${index}`} className="text-[12px] text-gray-300 leading-relaxed my-0.5">
        {renderInline(trimmed)}
      </div>
    );
  });

  flushList();
  return elements;
};

export default renderThinkingMarkdown;
