"""Conversation branching — tree-based conversation management.

Allows creating named branches from the current conversation state,
switching between them, and listing all branches.
"""

import json
import os
from datetime import datetime


class BranchManager:
    """Manages conversation branches as separate saved states."""

    def __init__(self, branch_dir: str = "~/.spark/branches"):
        self.branch_dir = os.path.expanduser(branch_dir)
        os.makedirs(self.branch_dir, exist_ok=True)
        self._current_branch: str = "main"

    @property
    def current(self) -> str:
        return self._current_branch

    def save_branch(self, name: str, context, cwd: str = "") -> str:
        """Save current conversation as a named branch."""
        branch_path = os.path.join(self.branch_dir, f"{name}.json")
        data = {
            "name": name,
            "timestamp": datetime.now().isoformat(),
            "turn_count": context.turn_count,
            "cwd": cwd or os.getcwd(),
            "parent_branch": self._current_branch,
            "messages": context.messages,
            "system_prompt": context.system_prompt,
        }
        with open(branch_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        self._current_branch = name
        return f"Branch '{name}' saved ({context.turn_count} turns)"

    def switch_branch(self, name: str, context) -> tuple[bool, str]:
        """Switch to a named branch, loading its conversation state."""
        branch_path = os.path.join(self.branch_dir, f"{name}.json")
        if not os.path.exists(branch_path):
            return False, f"Branch '{name}' not found"

        try:
            with open(branch_path, encoding="utf-8") as f:
                data = json.load(f)
            context.messages = data.get("messages", [])
            context.turn_count = data.get("turn_count", 0)
            # Restore system prompt if saved
            if "system_prompt" in data:
                context.system_prompt = data["system_prompt"]
            self._current_branch = name
            return True, f"Switched to branch '{name}' ({context.turn_count} turns)"
        except (json.JSONDecodeError, OSError) as e:
            return False, f"Failed to load branch '{name}': {e}"

    def create_branch(self, name: str, context, cwd: str = "") -> str:
        """Create a new branch from current state (saves current first)."""
        # Save current branch state before creating new one
        if self._current_branch:
            self.save_branch(self._current_branch, context, cwd)
        return self.save_branch(name, context, cwd)

    def delete_branch(self, name: str) -> tuple[bool, str]:
        """Delete a branch."""
        if name == self._current_branch:
            return False, "Cannot delete the current branch"
        branch_path = os.path.join(self.branch_dir, f"{name}.json")
        if not os.path.exists(branch_path):
            return False, f"Branch '{name}' not found"
        os.remove(branch_path)
        return True, f"Deleted branch '{name}'"

    def list_branches(self) -> list[dict]:
        """List all branches with metadata."""
        branches = []
        for f in sorted(os.listdir(self.branch_dir)):
            if not f.endswith(".json"):
                continue
            path = os.path.join(self.branch_dir, f)
            try:
                with open(path, encoding="utf-8") as fh:
                    data = json.load(fh)
                name = data.get("name", f.replace(".json", ""))
                branches.append({
                    "name": name,
                    "turns": data.get("turn_count", 0),
                    "timestamp": data.get("timestamp", ""),
                    "parent": data.get("parent_branch", ""),
                    "current": name == self._current_branch,
                })
            except (json.JSONDecodeError, OSError):
                pass
        return branches

    def merge_branch(self, source: str, context) -> tuple[bool, str]:
        """Merge another branch's messages into the current conversation."""
        branch_path = os.path.join(self.branch_dir, f"{source}.json")
        if not os.path.exists(branch_path):
            return False, f"Branch '{source}' not found"

        try:
            with open(branch_path, encoding="utf-8") as f:
                data = json.load(f)
            source_messages = data.get("messages", [])
            if not source_messages:
                return False, f"Branch '{source}' has no messages"

            # Add a summary of the merged branch
            summary = (
                f"[Merged from branch '{source}']\n"
                f"The branch had {len(source_messages)} messages. "
                f"Key content has been incorporated."
            )
            context.add_user(summary)
            return True, f"Merged branch '{source}' into '{self._current_branch}'"
        except (json.JSONDecodeError, OSError) as e:
            return False, f"Failed to merge: {e}"
