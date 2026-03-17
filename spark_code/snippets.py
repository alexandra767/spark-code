"""Snippet library — save and reuse common prompts."""

import json
import os


class SnippetLibrary:
    """User-defined reusable prompts."""

    def __init__(self, path: str = "~/.spark/snippets.json"):
        self.path = os.path.expanduser(path)
        self._snippets: dict[str, str] = {}
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, encoding="utf-8") as f:
                    self._snippets = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._snippets = {}

    def _save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._snippets, f, indent=2)

    def add(self, name: str, prompt: str) -> str:
        self._snippets[name] = prompt
        self._save()
        return f"Saved snippet: {name}"

    def get(self, name: str) -> str | None:
        return self._snippets.get(name)

    def remove(self, name: str) -> str:
        if name in self._snippets:
            del self._snippets[name]
            self._save()
            return f"Removed snippet: {name}"
        return f"Snippet not found: {name}"

    def list(self) -> dict[str, str]:
        return dict(self._snippets)
