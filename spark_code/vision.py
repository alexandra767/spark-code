"""Describe an image via a vision-capable provider (Gemini) so a text-only
primary model can work with what's on screen."""
from __future__ import annotations

import base64
from pathlib import Path

from .config import load_config, load_keys, resolve_provider_key
from .model import ModelClient

DEFAULT_VISION_PROMPT = (
    "You are the eyes for another AI. Describe this screen in detail: all visible "
    "text, UI elements, buttons, layout, current state, and anything notable. Be "
    "thorough and literal.")

# Appended by capture tools to signal the terminal should DISPLAY this image
# inline (see agent._store_tool_result / _flush_display_images). The path is
# stripped from the model-facing text — the model only ever sees the description.
DISPLAY_IMAGE_SENTINEL = "\n__SPARK_DISPLAY_IMAGE__:"


class VisionError(Exception):
    pass


async def describe_image(image_path, prompt=None, provider_name="gemini-pro", config=None) -> str:
    cfg = config or load_config()
    pconf = (cfg.get("providers", {}) or {}).get(provider_name)
    if not pconf:
        raise VisionError(f"vision provider '{provider_name}' is not configured in ~/.spark/config.yaml")
    try:
        data = Path(image_path).read_bytes()
    except OSError as e:
        raise VisionError(f"could not read image {image_path}: {e}") from e
    client = ModelClient(
        endpoint=pconf.get("endpoint", ""),
        model=pconf.get("model", ""),
        temperature=0.2,
        max_tokens=pconf.get("max_tokens", 1024),
        api_key=resolve_provider_key(provider_name, pconf, load_keys()),
        provider=provider_name,
        supports_vision=True)
    b64 = base64.b64encode(data).decode("utf-8")
    messages = [{"role": "user", "content": [
        {"type": "text", "text": prompt or DEFAULT_VISION_PROMPT},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]}]
    text = ""
    try:
        err_detail = ""
        async for chunk in client.chat(messages, stream=False):
            if chunk.get("type") == "text":
                text += chunk.get("content", "")
            elif chunk.get("type") == "error":
                # chat() yields (never raises) an error chunk on API failure;
                # capture it so a bad key / rate limit / oversized image surfaces
                # the real cause instead of a generic "no description".
                err_detail = chunk.get("content", "")
    finally:
        await client.close()
    text = text.strip()
    if not text:
        raise VisionError(err_detail or "vision provider returned no description")
    return text
