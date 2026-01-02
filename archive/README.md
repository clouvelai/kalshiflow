# Archive Directory Organization

This directory contains historical files from the Kalshi Flowboard project development process. The archive has been reorganized for better discoverability and logical grouping.

## Directory Structure

```
archive/
├── README.md                 # This file - organization guide
├── planning/                 # All planning and design documents
│   └── project/              # Project-level planning and feature docs
├── scripts/                  # Deprecated/experimental scripts
│   ├── rl/                   # RL-related utility scripts
│   ├── database/             # Database analysis and debugging scripts
│   └── deployment/           # Old deployment configuration scripts
├── experimental/             # Experimental code and prototypes
│   └── api/                  # API experiments and test results
└── notes/                    # Development journals and notes
    └── notes.md              # Historical development journal
```

## What's In Each Directory

### `/planning/`
Contains all planning documents, design specs, and roadmaps:

#### `/planning/project/` - Project-level Planning
- **Feature planning**: `feature_plan.json` - Main project roadmap
- **Architecture docs**: `app_overview.md` - Application architecture overview
- **Infrastructure setup**: Supabase, Railway WebSocket configuration docs
- **Performance analysis**: Post-deployment performance reports

### `/scripts/`
Historical utility scripts no longer in active use:

#### `/scripts/rl/` - RL Utilities
- `demo_curriculum_learning.py` - Curriculum learning experiment
- `query_rl_orderbook.py` - Orderbook data querying utilities
- `visualize_observation.py` - RL observation visualization tools

#### `/scripts/database/` - Database Tools
- `check_database.py` - Database health checking
- `check_session_data.py` - RL session data validation

#### `/scripts/deployment/` - Deployment Tools
- `set-railway-vars.sh` - Railway environment variable setup

### `/experimental/`
Experimental code and prototypes:

#### `/experimental/api/` - API Experiments  
- `demo_account_test_results.py` - Kalshi demo API testing results (comprehensive test suite)

### `/notes/`
Development journals and informal documentation:
- `notes.md` - Historical development journal (Dec 11 entries about RL environment issues)

## Archive History

This archive was reorganized on December 15, 2024, from a deeply nested structure with significant duplication. The previous structure had multiple overlapping `rl/` directories and planning documents scattered across various paths up to 5 levels deep.

**Previous issues resolved:**
- Consolidated multiple scattered RL planning directories and duplicate files
- Flattened deeply nested paths (some were `/planning/backend-archive/rl/backend-planning/`)
- Removed duplicate files and empty directories  
- Separated planning docs from scripts and experimental code
- Organized RL-related scripts into logical categories (database, deployment, RL utilities)

## Usage Guidelines

**When to reference the archive:**
- Looking for historical context on design decisions
- Understanding why certain approaches were tried/abandoned
- Finding old utility scripts that might be useful
- Reviewing past RL training experiments and results

**What NOT to do:**
- Don't use archived scripts in production - they're here for historical reference
- Don't treat planning docs as current - check main project docs first
- Don't modify files here - create new versions in the main project if needed

## Maintenance

This archive should be treated as read-only historical reference. If you need to:
- **Add new files**: Consider if they belong in the main project instead
- **Update documentation**: Update current project docs, not archived versions  
- **Remove files**: Only remove truly obsolete files, keep for historical value
- **Reorganize**: Only if the current structure becomes significantly problematic

Last organized: December 15, 2024