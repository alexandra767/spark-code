"""Detect platform info for system prompt injection."""

import os
import platform
import shutil


def get_platform_info() -> dict:
    """Gather platform details. Called once at startup."""
    system = platform.system()
    os_name = {"Darwin": "macOS", "Linux": "Linux", "Windows": "Windows"}.get(system, system)

    shell = os.environ.get("SHELL", "")
    if shell:
        shell = os.path.basename(shell)

    python_ver = platform.python_version()

    pkg_mgr = ""
    if system == "Darwin" and shutil.which("brew"):
        pkg_mgr = "brew"
    elif system == "Linux":
        if shutil.which("apt"):
            pkg_mgr = "apt"
        elif shutil.which("dnf"):
            pkg_mgr = "dnf"
        elif shutil.which("pacman"):
            pkg_mgr = "pacman"

    return {
        "os": os_name,
        "system": system,
        "shell": shell,
        "python": python_ver,
        "package_manager": pkg_mgr,
    }


def format_platform_prompt(cwd: str) -> str:
    """Format platform info as a system prompt prefix."""
    info = get_platform_info()
    parts = [
        f"Platform: {info['os']} ({info['system']})",
        f"Shell: {info['shell']}" if info["shell"] else None,
        f"CWD: {cwd}",
        f"Python: {info['python']}",
        f"Package manager: {info['package_manager']}" if info["package_manager"] else None,
    ]
    return ", ".join(p for p in parts if p)
