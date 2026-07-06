"""Tests for platform detection."""

from unittest.mock import patch

from spark_code.platform_info import format_platform_prompt, get_platform_info


def test_get_platform_info_returns_dict():
    info = get_platform_info()
    assert "os" in info
    assert "shell" in info
    assert "python" in info


def test_get_platform_info_os():
    info = get_platform_info()
    assert info["os"] in ("macOS", "Linux", "Windows")


def test_format_platform_prompt_contains_os():
    prompt = format_platform_prompt("/some/dir")
    assert "Platform:" in prompt
    assert "CWD:" in prompt
    assert "/some/dir" in prompt


def test_format_platform_prompt_contains_python():
    prompt = format_platform_prompt("/tmp")
    assert "Python:" in prompt


@patch("platform.system", return_value="Darwin")
def test_macos_detection(mock_sys):
    info = get_platform_info()
    assert info["os"] == "macOS"


@patch("platform.system", return_value="Linux")
def test_linux_detection(mock_sys):
    info = get_platform_info()
    assert info["os"] == "Linux"
