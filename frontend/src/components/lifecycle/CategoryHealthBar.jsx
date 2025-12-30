import React from 'react';

/**
 * CategoryHealthBar - Clickable category pills with mini capacity bars
 */
const CategoryHealthBar = ({ stats, activeCategory, onCategoryClick }) => {
  const categoryData = stats?.by_category || {};
  const totalTracked = stats?.tracked || 0;

  // Define category display info
  const categoryConfig = {
    sports: { label: 'Sports', color: 'bg-blue-500', textColor: 'text-blue-400' },
    crypto: { label: 'Crypto', color: 'bg-orange-500', textColor: 'text-orange-400' },
    entertainment: { label: 'Entertainment', color: 'bg-purple-500', textColor: 'text-purple-400' },
    media_mentions: { label: 'Media', color: 'bg-pink-500', textColor: 'text-pink-400' },
    politics: { label: 'Politics', color: 'bg-red-500', textColor: 'text-red-400' },
    economics: { label: 'Economics', color: 'bg-green-500', textColor: 'text-green-400' },
    climate: { label: 'Climate', color: 'bg-cyan-500', textColor: 'text-cyan-400' },
    other: { label: 'Other', color: 'bg-gray-500', textColor: 'text-gray-400' }
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
          const config = categoryConfig[category] || categoryConfig.other;
          const isActive = activeCategory === category;
          const proportion = totalTracked > 0 ? (count / totalTracked) * 100 : 0;

          return (
            <button
              key={category}
              onClick={() => onCategoryClick(category)}
              className={`flex items-center gap-2 px-3 py-1.5 rounded-lg transition-all ${
                isActive
                  ? 'bg-gray-700 text-white ring-1 ring-gray-500'
                  : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
              }`}
            >
              {/* Category name */}
              <span className={`text-sm font-medium ${isActive ? 'text-white' : config.textColor}`}>
                {config.label}
              </span>

              {/* Count badge */}
              <span className={`text-xs font-mono px-1.5 py-0.5 rounded ${
                isActive ? 'bg-gray-500' : 'bg-gray-700'
              }`}>
                {count}
              </span>

              {/* Mini proportion bar */}
              <div className="w-8 h-1.5 bg-gray-700 rounded-full overflow-hidden">
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
