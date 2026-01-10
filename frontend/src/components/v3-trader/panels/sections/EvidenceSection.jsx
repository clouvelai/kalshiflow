import React, { memo, useState } from 'react';
import { Search, ChevronDown, ChevronRight, Shield, CheckCircle, AlertCircle, MessageSquare, Heart, Repeat, Users, BadgeCheck } from 'lucide-react';

/**
 * SectionHeader - Reusable section header component
 */
const SectionHeader = ({ icon: Icon, title }) => (
  <div className="flex items-center space-x-2 mb-3">
    <Icon className="w-4 h-4 text-slate-500" />
    <h4 className="text-xs uppercase tracking-wider text-slate-500 font-semibold">
      {title}
    </h4>
  </div>
);

/**
 * ReliabilityBadge - Shows evidence reliability level
 */
const ReliabilityBadge = memo(({ reliability }) => {
  const config = {
    high: {
      icon: CheckCircle,
      bgClass: 'bg-emerald-500/10',
      borderClass: 'border-emerald-500/30',
      textClass: 'text-emerald-400',
      label: 'HIGH',
    },
    medium: {
      icon: Shield,
      bgClass: 'bg-amber-500/10',
      borderClass: 'border-amber-500/30',
      textClass: 'text-amber-400',
      label: 'MEDIUM',
    },
    low: {
      icon: AlertCircle,
      bgClass: 'bg-red-500/10',
      borderClass: 'border-red-500/30',
      textClass: 'text-red-400',
      label: 'LOW',
    },
  };

  const normalizedReliability = (reliability || 'medium').toLowerCase();
  const { icon: Icon, bgClass, borderClass, textClass, label } = config[normalizedReliability] || config.medium;

  return (
    <span className={`
      inline-flex items-center space-x-1.5 px-2.5 py-1 rounded-lg text-xs font-semibold
      ${bgClass} ${borderClass} ${textClass} border
    `}>
      <Icon className="w-3.5 h-3.5" />
      <span>{label} RELIABILITY</span>
    </span>
  );
});

ReliabilityBadge.displayName = 'ReliabilityBadge';

/**
 * TruthSocialSourceBadge - Shows Truth Social source with stats
 */
const TruthSocialSourceBadge = memo(({ truthSocialData }) => {
  if (!truthSocialData || truthSocialData.status === 'unavailable') {
    return null;
  }

  const { posts_found = 0, unique_authors = 0, verified_count = 0, status } = truthSocialData;

  // No matches found
  if (status === 'no_matches' || posts_found === 0) {
    return (
      <span className="inline-flex items-center space-x-1.5 px-2.5 py-1 rounded-lg text-xs font-medium bg-slate-700/30 text-slate-500 border border-slate-600/20">
        <MessageSquare className="w-3.5 h-3.5" />
        <span>No Truth Social posts found</span>
      </span>
    );
  }

  return (
    <span className="inline-flex items-center space-x-2 px-2.5 py-1 rounded-lg text-xs font-medium bg-pink-500/10 text-pink-400 border border-pink-500/20">
      <MessageSquare className="w-3.5 h-3.5" />
      <span>{posts_found} posts</span>
      <span className="text-pink-500/50">•</span>
      <Users className="w-3 h-3" />
      <span>{unique_authors} authors</span>
      {verified_count > 0 && (
        <>
          <span className="text-pink-500/50">•</span>
          <BadgeCheck className="w-3 h-3" />
          <span>{verified_count}</span>
        </>
      )}
    </span>
  );
});

TruthSocialSourceBadge.displayName = 'TruthSocialSourceBadge';

/**
 * TruthSocialPostCard - Displays a single Truth Social post with engagement
 */
const TruthSocialPostCard = memo(({ post }) => {
  const { author, likes = 0, reblogs = 0, replies = 0, is_verified, url, content } = post;

  // Truncate content for display
  const truncatedContent = content && content.length > 200
    ? content.substring(0, 200) + '...'
    : content;

  return (
    <div className="bg-slate-800/30 rounded-lg p-3 border border-slate-700/20">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center space-x-2">
          <span className="text-xs font-medium text-pink-400">@{author}</span>
          {is_verified && <BadgeCheck className="w-3.5 h-3.5 text-blue-400" />}
        </div>
        <div className="flex items-center space-x-3 text-xs text-slate-500">
          <span className="flex items-center space-x-1">
            <Heart className="w-3 h-3" />
            <span>{likes}</span>
          </span>
          <span className="flex items-center space-x-1">
            <Repeat className="w-3 h-3" />
            <span>{reblogs}</span>
          </span>
          <span className="flex items-center space-x-1">
            <MessageSquare className="w-3 h-3" />
            <span>{replies}</span>
          </span>
        </div>
      </div>
      {truncatedContent && (
        <p className="text-xs text-slate-400 leading-relaxed">
          {truncatedContent}
        </p>
      )}
      {url && (
        <a
          href={url}
          target="_blank"
          rel="noopener noreferrer"
          className="text-xs text-pink-400/70 hover:text-pink-400 mt-2 inline-block"
        >
          View on Truth Social →
        </a>
      )}
    </div>
  );
});

TruthSocialPostCard.displayName = 'TruthSocialPostCard';

/**
 * TruthSocialPostsSection - Expandable list of Truth Social posts
 */
const TruthSocialPostsSection = memo(({ topPosts }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  if (!topPosts || topPosts.length === 0) {
    return null;
  }

  return (
    <div className="border border-pink-500/20 rounded-lg overflow-hidden">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center justify-between px-4 py-3 bg-pink-500/5 hover:bg-pink-500/10 transition-colors"
      >
        <div className="flex items-center space-x-2">
          <MessageSquare className="w-4 h-4 text-pink-400" />
          <span className="text-xs uppercase tracking-wider text-pink-400 font-medium">
            Truth Social Posts
          </span>
          <span className="text-xs text-pink-500/50">({topPosts.length})</span>
        </div>
        {isExpanded ? (
          <ChevronDown className="w-4 h-4 text-pink-400" />
        ) : (
          <ChevronRight className="w-4 h-4 text-pink-400" />
        )}
      </button>

      {isExpanded && (
        <div className="px-4 py-3 bg-slate-900/30 space-y-2">
          {topPosts.slice(0, 5).map((post, idx) => (
            <TruthSocialPostCard key={post.post_id || idx} post={post} />
          ))}
        </div>
      )}
    </div>
  );
});

TruthSocialPostsSection.displayName = 'TruthSocialPostsSection';

/**
 * EvidenceSection - Displays gathered evidence and its reliability
 *
 * Shows:
 * - Reliability badge
 * - Truth Social source badge (posts found, authors, verified)
 * - Evidence summary paragraph
 * - Key evidence bullet points
 * - Truth Social posts (expandable)
 */
const EvidenceSection = ({ evidenceSummary, reliability, keyEvidence, evidenceMetadata }) => {
  const [isEvidenceExpanded, setIsEvidenceExpanded] = useState(false);

  // Extract Truth Social data from metadata
  const truthSocialData = evidenceMetadata?.truth_social;
  const hasTruthSocialPosts = truthSocialData?.top_posts?.length > 0;

  const hasContent = evidenceSummary || (keyEvidence && keyEvidence.length > 0) || hasTruthSocialPosts;

  if (!hasContent) return null;

  return (
    <div className="bg-slate-900/40 rounded-xl p-4 border border-slate-700/30">
      <div className="flex items-center justify-between mb-3">
        <SectionHeader icon={Search} title="Evidence" />
        <div className="flex items-center space-x-2">
          <TruthSocialSourceBadge truthSocialData={truthSocialData} />
          <ReliabilityBadge reliability={reliability} />
        </div>
      </div>

      <div className="space-y-4">
        {/* Evidence Summary */}
        {evidenceSummary && (
          <div className="bg-slate-800/40 rounded-lg p-4 border border-slate-700/20">
            <p className="text-sm text-slate-300 leading-relaxed">
              {evidenceSummary}
            </p>
          </div>
        )}

        {/* Truth Social Posts (expandable) */}
        {hasTruthSocialPosts && (
          <TruthSocialPostsSection topPosts={truthSocialData.top_posts} />
        )}

        {/* Key Evidence Points */}
        {keyEvidence && keyEvidence.length > 0 && (
          <div className="border border-slate-700/20 rounded-lg overflow-hidden">
            <button
              onClick={() => setIsEvidenceExpanded(!isEvidenceExpanded)}
              className="w-full flex items-center justify-between px-4 py-3 bg-slate-800/30 hover:bg-slate-800/50 transition-colors"
            >
              <div className="flex items-center space-x-2">
                <CheckCircle className="w-4 h-4 text-cyan-400" />
                <span className="text-xs uppercase tracking-wider text-slate-400 font-medium">
                  Key Evidence Points
                </span>
                <span className="text-xs text-slate-600">({keyEvidence.length})</span>
              </div>
              {isEvidenceExpanded ? (
                <ChevronDown className="w-4 h-4 text-slate-500" />
              ) : (
                <ChevronRight className="w-4 h-4 text-slate-500" />
              )}
            </button>

            {isEvidenceExpanded && (
              <div className="px-4 py-3 bg-slate-900/30 space-y-3">
                {keyEvidence.map((evidence, idx) => (
                  <div
                    key={idx}
                    className="flex items-start space-x-3"
                  >
                    <span className="flex-shrink-0 mt-1.5 w-1.5 h-1.5 rounded-full bg-cyan-500/50 border border-cyan-500/30" />
                    <p className="text-xs text-slate-400 leading-relaxed">
                      {evidence}
                    </p>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default memo(EvidenceSection);
