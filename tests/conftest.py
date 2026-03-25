"""Shared test fixtures for Spark Code tests."""

import asyncio
import json
import os
import tempfile
import shutil

import pytest


@pytest.fixture
def tmp_dir():
    """Create a temporary directory for file operation tests."""
    d = tempfile.mkdtemp(prefix="spark_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def tmp_file(tmp_dir):
    """Create a temporary file with sample content."""
    path = os.path.join(tmp_dir, "test_file.py")
    content = "line 1\nline 2\nline 3\nline 4\nline 5\n"
    with open(path, "w") as f:
        f.write(content)
    return path


@pytest.fixture
def sample_config():
    """Return a sample config dict."""
    return {
        "model": {
            "endpoint": "http://localhost:11434",
            "name": "qwen2.5:72b",
            "temperature": 0.7,
            "max_tokens": 4096,
            "context_window": 32768,
            "api_key": "",
            "provider": "ollama",
        },
        "permissions": {
            "mode": "auto",
            "always_allow": ["read_file", "glob", "grep", "list_dir"],
        },
        "ui": {
            "theme": "dark",
        },
    }
