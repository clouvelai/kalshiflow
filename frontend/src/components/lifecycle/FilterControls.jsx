import React from 'react';

/**
 * FilterControls - Search, status, spread, and sort filters
 */
const FilterControls = ({ filters, onFilterChange, marketCount, totalCount }) => {
  const updateFilter = (key, value) => {
    onFilterChange(prev => ({ ...prev, [key]: value }));
  };

  return (
    <div className="bg-gray-900/50 rounded-lg p-3 border border-gray-800 flex flex-wrap items-center gap-4">
      {/* Search */}
      <div className="flex-1 min-w-48">
        <input
          type="text"
          value={filters.search}
          onChange={(e) => updateFilter('search', e.target.value)}
          placeholder="Search markets..."
          className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-1.5 text-sm text-white placeholder-gray-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
        />
      </div>

      {/* Status filter */}
      <div className="flex items-center gap-2">
        <label className="text-xs text-gray-500 uppercase tracking-wide">Status</label>
        <select
          value={filters.status}
          onChange={(e) => updateFilter('status', e.target.value)}
          className="bg-gray-800 border border-gray-700 rounded-lg px-2 py-1.5 text-sm text-white focus:outline-none focus:ring-1 focus:ring-blue-500"
        >
          <option value="all">All</option>
          <option value="active">Active</option>
          <option value="determined">Determined</option>
        </select>
      </div>

      {/* Spread filter */}
      <div className="flex items-center gap-2">
        <label className="text-xs text-gray-500 uppercase tracking-wide">Spread</label>
        <select
          value={filters.spread}
          onChange={(e) => updateFilter('spread', e.target.value)}
          className="bg-gray-800 border border-gray-700 rounded-lg px-2 py-1.5 text-sm text-white focus:outline-none focus:ring-1 focus:ring-blue-500"
        >
          <option value="any">Any</option>
          <option value="lt2">&le;2c</option>
          <option value="lt5">&le;5c</option>
          <option value="lt10">&le;10c</option>
        </select>
      </div>

      {/* Sort */}
      <div className="flex items-center gap-2">
        <label className="text-xs text-gray-500 uppercase tracking-wide">Sort</label>
        <select
          value={filters.sort}
          onChange={(e) => updateFilter('sort', e.target.value)}
          className="bg-gray-800 border border-gray-700 rounded-lg px-2 py-1.5 text-sm text-white focus:outline-none focus:ring-1 focus:ring-blue-500"
        >
          <option value="newest">Newest</option>
          <option value="volume_delta">Volume Delta</option>
          <option value="spread">Tightest Spread</option>
          <option value="price_move">Price Move</option>
        </select>
      </div>

      {/* Results count */}
      <div className="text-sm text-gray-400 ml-auto">
        {marketCount === totalCount ? (
          <span>{totalCount} markets</span>
        ) : (
          <span>{marketCount} of {totalCount} markets</span>
        )}
      </div>
    </div>
  );
};

export default FilterControls;
