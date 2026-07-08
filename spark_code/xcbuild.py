"""iOS build pipeline primitives (archive/export/validate/upload) for Spark Code."""
from __future__ import annotations

import asyncio
import json
import plistlib
from pathlib import Path


class XcBuildError(Exception):
    pass


def archive_argv(project, scheme, archive_path, is_workspace=False) -> list[str]:
    flag = "-workspace" if is_workspace else "-project"
    return ["xcodebuild", flag, str(project), "-scheme", scheme,
            "-configuration", "Release", "-destination", "generic/platform=iOS",
            "-archivePath", str(archive_path), "-allowProvisioningUpdates", "archive"]


def export_argv(archive_path, export_path, export_options_path) -> list[str]:
    return ["xcodebuild", "-exportArchive", "-archivePath", str(archive_path),
            "-exportPath", str(export_path), "-exportOptionsPlist", str(export_options_path),
            "-allowProvisioningUpdates"]


def altool_argv(action, ipa_path, key_id, issuer_id) -> list[str]:
    return ["xcrun", "altool", f"--{action}-app", "-f", str(ipa_path), "-t", "ios",
            "--apiKey", key_id, "--apiIssuer", issuer_id, "--output-format", "json"]


def write_default_export_options(path, team_id=None) -> None:
    opts = {"method": "app-store", "destination": "export",
            "signingStyle": "automatic", "uploadSymbols": True}
    if team_id:
        opts["teamID"] = team_id
    Path(path).write_bytes(plistlib.dumps(opts))


def parse_altool_json(stdout) -> tuple[bool, list[str]]:
    try:
        data = json.loads(stdout)
    except (json.JSONDecodeError, TypeError):
        return False, ["Could not parse altool output"]
    if not isinstance(data, dict):
        return False, ["Unexpected altool output (not a JSON object)"]
    errs = data.get("product-errors") or []
    if errs:
        return False, [e.get("message", str(e)) if isinstance(e, dict) else str(e) for e in errs]
    return True, []


def read_archive_props(archive_path) -> dict:
    p = Path(archive_path) / "Info.plist"
    data = plistlib.loads(p.read_bytes())
    props = data.get("ApplicationProperties", {})
    return {"bundle_id": props.get("CFBundleIdentifier"),
            "marketing_version": props.get("CFBundleShortVersionString"),
            "build_number": props.get("CFBundleVersion")}


async def run_step(argv, cwd=None, timeout=1800.0) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(
        *argv, cwd=cwd,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise XcBuildError(f"step timed out after {timeout}s: {' '.join(map(str, argv[:3]))} …")
    return proc.returncode, out.decode("utf-8", "replace")
