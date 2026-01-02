#!/bin/bash
# Claude Code Session Tracker
# View and manage active Claude Code sessions

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( dirname "$SCRIPT_DIR" )"
SESSIONS_DIR="$PROJECT_ROOT/.claude-sessions/active"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'
BOLD='\033[1m'

usage() {
    echo "Usage: $0 <command>"
    echo ""
    echo "Commands:"
    echo "  list    Show all active Claude Code sessions"
    echo "  clean   Remove stale sessions (>4 hours old)"
    echo "  help    Show this help message"
    echo ""
}

list_sessions() {
    echo -e "${BOLD}${CYAN}Active Claude Code Sessions${NC}"
    echo -e "${CYAN}============================${NC}"
    echo ""

    if [ ! -d "$SESSIONS_DIR" ]; then
        echo -e "${YELLOW}No sessions directory found.${NC}"
        return
    fi

    session_count=0
    now=$(date +%s)

    shopt -s nullglob
    for session_file in "$SESSIONS_DIR"/*.json; do
        session_count=$((session_count + 1))

        # Parse JSON with jq if available
        if command -v jq > /dev/null 2>&1; then
            session_id=$(jq -r '.session_id // "unknown"' "$session_file")
            branch=$(jq -r '.branch // "unknown"' "$session_file")
            status=$(jq -r '.status // "unknown"' "$session_file")
            current_task=$(jq -r '.current_task // "No task specified"' "$session_file")
            last_updated=$(jq -r '.last_updated // ""' "$session_file")
            sub_agents=$(jq -r '.sub_agents // [] | length' "$session_file")
        else
            # Fallback: basic grep parsing
            session_id=$(grep -o '"session_id"[[:space:]]*:[[:space:]]*"[^"]*"' "$session_file" | cut -d'"' -f4)
            branch=$(grep -o '"branch"[[:space:]]*:[[:space:]]*"[^"]*"' "$session_file" | cut -d'"' -f4)
            status=$(grep -o '"status"[[:space:]]*:[[:space:]]*"[^"]*"' "$session_file" | cut -d'"' -f4)
            current_task=$(grep -o '"current_task"[[:space:]]*:[[:space:]]*"[^"]*"' "$session_file" | cut -d'"' -f4)
            last_updated=""
            sub_agents="?"
        fi

        # Calculate age
        age_str=""
        if [ -n "$last_updated" ]; then
            if updated_ts=$(date -j -f "%Y-%m-%dT%H:%M:%SZ" "$last_updated" +%s 2>/dev/null); then
                age_secs=$((now - updated_ts))
                if [ $age_secs -lt 60 ]; then
                    age_str="${age_secs}s ago"
                elif [ $age_secs -lt 3600 ]; then
                    age_str="$((age_secs / 60))m ago"
                else
                    age_str="$((age_secs / 3600))h ago"
                fi
            fi
        fi

        # Status color
        case "$status" in
            "active") status_color="${GREEN}" ;;
            "idle") status_color="${YELLOW}" ;;
            *) status_color="${NC}" ;;
        esac

        echo -e "${BOLD}Session: ${BLUE}$session_id${NC}"
        echo -e "  Branch:  ${CYAN}$branch${NC}"
        echo -e "  Status:  ${status_color}$status${NC} ${age_str:+($age_str)}"
        echo -e "  Task:    $current_task"
        if [ "$sub_agents" != "0" ] && [ "$sub_agents" != "?" ]; then
            echo -e "  Agents:  ${YELLOW}$sub_agents running${NC}"
        fi
        echo ""
    done
    shopt -u nullglob

    if [ $session_count -eq 0 ]; then
        echo -e "${YELLOW}No active sessions found.${NC}"
        echo ""
        echo "Sessions are registered when Claude Code starts working."
    else
        echo -e "${CYAN}----------------------------${NC}"
        echo -e "Total: ${BOLD}$session_count${NC} session(s)"
    fi
}

clean_sessions() {
    echo -e "${BOLD}Cleaning stale sessions...${NC}"

    if [ ! -d "$SESSIONS_DIR" ]; then
        echo "No sessions directory found."
        return
    fi

    now=$(date +%s)
    max_age=$((4 * 3600))
    cleaned=0

    shopt -s nullglob
    for session_file in "$SESSIONS_DIR"/*.json; do
        # Get file modification time (macOS)
        if stat_time=$(stat -f %m "$session_file" 2>/dev/null); then
            age=$((now - stat_time))
            if [ $age -gt $max_age ]; then
                session_id=$(basename "$session_file" .json)
                echo -e "  ${RED}Removing${NC} stale session: $session_id ($(($age / 3600))h old)"
                rm "$session_file"
                cleaned=$((cleaned + 1))
            fi
        fi
    done
    shopt -u nullglob

    if [ $cleaned -eq 0 ]; then
        echo -e "${GREEN}No stale sessions found.${NC}"
    else
        echo -e "${GREEN}Cleaned $cleaned stale session(s).${NC}"
    fi
}

# Main
case "${1:-}" in
    list|ls|"")
        list_sessions
        ;;
    clean|cleanup)
        clean_sessions
        ;;
    help|--help|-h)
        usage
        ;;
    *)
        echo "Unknown command: $1"
        usage
        exit 1
        ;;
esac
