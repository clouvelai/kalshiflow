import React, { memo } from 'react';
import { Network, Users, Briefcase, Target, HelpCircle } from 'lucide-react';

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
 * EntityCard - Displays a single entity (actor, object, or candidate)
 */
const EntityCard = memo(({ entity, type }) => {
  const typeConfig = {
    actor: {
      icon: Users,
      color: 'emerald',
      bgClass: 'bg-emerald-500/5',
      borderClass: 'border-emerald-500/20',
      textClass: 'text-emerald-400',
    },
    object: {
      icon: Briefcase,
      color: 'cyan',
      bgClass: 'bg-cyan-500/5',
      borderClass: 'border-cyan-500/20',
      textClass: 'text-cyan-400',
    },
    candidate: {
      icon: Target,
      color: 'violet',
      bgClass: 'bg-violet-500/5',
      borderClass: 'border-violet-500/20',
      textClass: 'text-violet-400',
    },
  };

  const config = typeConfig[type] || typeConfig.candidate;
  const Icon = config.icon;

  return (
    <div className={`
      ${config.bgClass} border ${config.borderClass} rounded-lg p-3
      transition-all duration-200 hover:scale-[1.02]
    `}>
      <div className="flex items-start space-x-3">
        <div className={`
          p-1.5 rounded-lg bg-slate-800/50 flex-shrink-0
        `}>
          <Icon className={`w-4 h-4 ${config.textClass}`} />
        </div>
        <div className="flex-1 min-w-0">
          <h5 className="text-sm font-medium text-white truncate">
            {entity.canonical_name}
          </h5>
          {entity.role && (
            <p className="text-[10px] text-slate-500 uppercase tracking-wider mt-0.5">
              {entity.role}
            </p>
          )}
          {entity.role_description && (
            <p className="text-xs text-slate-400 mt-1 leading-relaxed">
              {entity.role_description}
            </p>
          )}
          {entity.market_ticker && (
            <div className="mt-2">
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-700/50 text-slate-400 border border-slate-600/30">
                {entity.market_ticker}
              </span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
});

EntityCard.displayName = 'EntityCard';

/**
 * EntityColumn - Column of entities with a header
 */
const EntityColumn = memo(({ title, entities, type, icon: Icon }) => {
  if (!entities || entities.length === 0) return null;

  const typeColors = {
    actor: 'text-emerald-400',
    object: 'text-cyan-400',
    candidate: 'text-violet-400',
  };

  return (
    <div>
      <div className="flex items-center space-x-2 mb-3">
        <Icon className={`w-3.5 h-3.5 ${typeColors[type]}`} />
        <span className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold">
          {title}
        </span>
        <span className="text-[10px] text-slate-600">({entities.length})</span>
      </div>
      <div className="space-y-2">
        {entities.slice(0, 5).map((entity, idx) => (
          <EntityCard
            key={entity.entity_id || idx}
            entity={entity}
            type={type}
          />
        ))}
        {entities.length > 5 && (
          <p className="text-xs text-slate-500 text-center py-2">
            +{entities.length - 5} more
          </p>
        )}
      </div>
    </div>
  );
});

EntityColumn.displayName = 'EntityColumn';

/**
 * SemanticFrameSection - Displays the semantic frame structure
 *
 * Shows actors, objects, and candidates in a 3-column grid,
 * along with the question template and resolution trigger.
 */
const SemanticFrameSection = ({ semanticFrame }) => {
  if (!semanticFrame) return null;

  const {
    frame_type,
    question_template,
    actors,
    objects,
    candidates,
    resolution_trigger,
  } = semanticFrame;

  const hasEntities = (actors?.length > 0) || (objects?.length > 0) || (candidates?.length > 0);

  if (!hasEntities && !question_template) return null;

  return (
    <div className="bg-slate-900/40 rounded-xl p-4 border border-slate-700/30">
      <SectionHeader icon={Network} title="Semantic Frame" />

      {/* Question Template */}
      {question_template && (
        <div className="bg-slate-800/40 rounded-lg p-3 border border-slate-700/20 mb-4">
          <div className="flex items-center space-x-2 mb-2">
            <HelpCircle className="w-3.5 h-3.5 text-amber-400" />
            <span className="text-[10px] uppercase tracking-wider text-slate-500 font-medium">
              Question Structure
            </span>
          </div>
          <p className="text-sm text-white font-mono leading-relaxed">
            {question_template}
          </p>
        </div>
      )}

      {/* Resolution Trigger */}
      {resolution_trigger && (
        <div className="bg-slate-800/40 rounded-lg p-3 border border-slate-700/20 mb-4">
          <div className="flex items-center space-x-2 mb-2">
            <Target className="w-3.5 h-3.5 text-cyan-400" />
            <span className="text-[10px] uppercase tracking-wider text-slate-500 font-medium">
              Resolution Trigger
            </span>
          </div>
          <p className="text-xs text-slate-400 leading-relaxed">
            {resolution_trigger}
          </p>
        </div>
      )}

      {/* 3-Column Grid for Entities */}
      {hasEntities && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <EntityColumn
            title="Actors"
            entities={actors}
            type="actor"
            icon={Users}
          />
          <EntityColumn
            title="Objects"
            entities={objects}
            type="object"
            icon={Briefcase}
          />
          <EntityColumn
            title="Candidates"
            entities={candidates}
            type="candidate"
            icon={Target}
          />
        </div>
      )}
    </div>
  );
};

export default memo(SemanticFrameSection);
