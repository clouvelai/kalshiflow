import React from 'react';

/**
 * CategoryHealthBar - Clickable category pills with mini capacity bars
 */
const CategoryHealthBar = ({ stats, activeCategory, onCategoryClick }) => {
  const categoryData = stats?.by_category || {};
  const totalTracked = stats?.tracked || 0;

  // Define category display info with beautiful colors
  const categoryConfig = {
    sports: { label: 'Sports', color: 'bg-blue-500', textColor: 'text-blue-400', bgColor: 'bg-blue-500/20' },
    crypto: { label: 'Crypto', color: 'bg-orange-500', textColor: 'text-orange-400', bgColor: 'bg-orange-500/20' },
    entertainment: { label: 'Entertainment', color: 'bg-purple-500', textColor: 'text-purple-400', bgColor: 'bg-purple-500/20' },
    media_mentions: { label: 'Media', color: 'bg-pink-500', textColor: 'text-pink-400', bgColor: 'bg-pink-500/20' },
    politics: { label: 'Politics', color: 'bg-red-500', textColor: 'text-red-400', bgColor: 'bg-red-500/20' },
    economics: { label: 'Economics', color: 'bg-green-500', textColor: 'text-green-400', bgColor: 'bg-green-500/20' },
    climate: { label: 'Climate', color: 'bg-cyan-500', textColor: 'text-cyan-400', bgColor: 'bg-cyan-500/20' },
    financials: { label: 'Financials', color: 'bg-emerald-500', textColor: 'text-emerald-400', bgColor: 'bg-emerald-500/20' },
    science: { label: 'Science', color: 'bg-indigo-500', textColor: 'text-indigo-400', bgColor: 'bg-indigo-500/20' },
    world: { label: 'World', color: 'bg-teal-500', textColor: 'text-teal-400', bgColor: 'bg-teal-500/20' },
    tech: { label: 'Tech', color: 'bg-violet-500', textColor: 'text-violet-400', bgColor: 'bg-violet-500/20' },
    culture: { label: 'Culture', color: 'bg-fuchsia-500', textColor: 'text-fuchsia-400', bgColor: 'bg-fuchsia-500/20' },
    other: { label: 'Other', color: 'bg-gray-500', textColor: 'text-gray-400', bgColor: 'bg-gray-500/20' }
  };

  // Get categories with counts, sorted by count
  const categories = Object.entries(categoryData)
    .filter(([_, count]) => count > 0)
    .sort((a, b) => b[1] - a[1]);

  if (categories.length === 0) {
    return (
      <div className="bg-gray-900/50 rounded-lg p-3 mb-4 border border-gray-800">
        <p className="text-gray-500 text-sm text-center">
          Waiting for markets to be tracked...
        </p>
      </div>
    );
  }

  return (
    <div className="bg-gray-900/50 rounded-lg p-3 mb-4 border border-gray-800">
      <div className="flex flex-wrap gap-2">
        {/* All button */}
        <button
          onClick={() => onCategoryClick(null)}
          className={`flex items-center gap-2 px-3 py-1.5 rounded-lg transition-all ${
            !activeCategory
              ? 'bg-gray-700 text-white ring-1 ring-gray-500'
              : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
          }`}
        >
          <span className="text-sm font-medium">All</span>
          <span className="text-xs font-mono bg-gray-600 px-1.5 py-0.5 rounded">
            {totalTracked}
          </span>
        </button>

        {/* Category pills */}
        {categories.map(([category, count]) => {
          // Case-insensitive lookup
          const configKey = category.toLowerCase();
          const config = categoryConfig[configKey] || categoryConfig.other;
          const isActive = activeCategory === category;
          const proportion = totalTracked > 0 ? (count / totalTracked) * 100 : 0;

          return (
            <button
              key={category}
              onClick={() => onCategoryClick(category)}
              className={`flex items-center gap-2 px-3 py-1.5 rounded-lg transition-all border ${
                isActive
                  ? `${config.bgColor} border-current ring-1 ring-current ${config.textColor}`
                  : `bg-gray-800/50 border-gray-700 hover:border-gray-600 hover:bg-gray-700/50`
              }`}
            >
              {/* Category name */}
              <span className={`text-sm font-medium ${config.textColor}`}>
                {config.label}
              </span>

              {/* Count badge */}
              <span className={`text-xs font-mono px-1.5 py-0.5 rounded ${
                isActive ? `${config.color}/30` : 'bg-gray-700'
              } ${config.textColor}`}>
                {count}
              </span>

              {/* Mini proportion bar */}
              <div className="w-10 h-1.5 bg-gray-700 rounded-full overflow-hidden">
                <div
                  className={`h-full ${config.color} transition-all duration-300`}
                  style={{ width: `${proportion}%` }}
                />
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
};

export default CategoryHealthBar;
