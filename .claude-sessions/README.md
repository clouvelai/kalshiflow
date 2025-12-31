# Claude Code Session Tracking

This directory enables coordination between multiple Claude Code sessions working on the same codebase.

## How It Works

Each Claude Code session registers itself by creating a JSON file in `active/`. Other sessions can read these files to see what's happening across the project.

## Session File Format

```json
{
  "session_id": "unique-id",
  "started_at": "2025-12-30T10:15:00Z",
  "last_updated": "2025-12-30T10:45:00Z",
  "branch": "feature/my-branch",
  "status": "active",
  "current_task": "Brief description of current work",
  "files_touched": ["path/to/file.py"],
  "sub_agents": [
    {
      "type": "agent-type",
      "task": "What the agent is doing",
      "started_at": "2025-12-30T10:40:00Z",
      "status": "running"
    }
  ]
}
```

## Commands

```bash
# View all active sessions
./scripts/claude-sessions.sh list

# Clean up stale sessions (>4 hours old)
./scripts/claude-sessions.sh clean
```

## For Claude Code

When starting a session:
1. Generate a unique session ID (use first 8 chars of a UUID)
2. Create `active/{session_id}.json` with your initial state
3. Update the file when switching tasks or spawning sub-agents
4. Check other session files before making major changes to shared files

## File Locations

- `active/*.json` - Live session files (git-ignored)
- `README.md` - This documentation (tracked in git)
