
import pytest

from spark_code.vision import DEFAULT_VISION_PROMPT, VisionError, describe_image

CFG = {"providers": {"gemini-pro": {"endpoint": "https://gen.example/v1beta/openai",
    "model": "gemini-2.5-pro", "max_tokens": 1024, "api_key": "k"}}}

async def test_describe_image_sends_image_and_returns_text(tmp_path, monkeypatch):
    img = tmp_path / "s.png"
    img.write_bytes(b"\x89PNG_fake_bytes")
    seen = {}
    class FakeClient:
        def __init__(self, **kw): seen["init"] = kw
        async def chat(self, messages, tools=None, stream=True):
            seen["messages"] = messages
            yield {"type": "text", "content": "A login screen with "}
            yield {"type": "text", "content": "an email field and a blue Submit button."}
            yield {"type": "done", "usage": {}}
        async def close(self): seen["closed"] = True
    monkeypatch.setattr("spark_code.vision.ModelClient", FakeClient)
    out = await describe_image(str(img), config=CFG)
    assert "blue Submit button" in out and out == out.strip()
    assert seen["init"]["supports_vision"] is True and seen["init"]["provider"] == "gemini-pro"
    content = seen["messages"][0]["content"]
    assert content[0]["type"] == "text" and DEFAULT_VISION_PROMPT[:10] in content[0]["text"]
    assert content[1]["type"] == "image_url" and content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert seen["closed"] is True

async def test_describe_image_custom_prompt(tmp_path, monkeypatch):
    img = tmp_path / "s.png"
    img.write_bytes(b"x")
    class FakeClient:
        def __init__(self, **kw): pass
        async def chat(self, messages, tools=None, stream=True):
            FakeClient.msgs = messages
            yield {"type": "text", "content": "ok"}
        async def close(self): pass
    monkeypatch.setattr("spark_code.vision.ModelClient", FakeClient)
    await describe_image(str(img), prompt="Just read the error text.", config=CFG)
    assert FakeClient.msgs[0]["content"][0]["text"] == "Just read the error text."

async def test_missing_provider_raises(tmp_path):
    img = tmp_path / "s.png"
    img.write_bytes(b"x")
    with pytest.raises(VisionError):
        await describe_image(str(img), config={"providers": {}})

async def test_unreadable_image_raises(tmp_path):
    with pytest.raises(VisionError):
        await describe_image(str(tmp_path / "nope.png"), config=CFG)

async def test_api_error_surfaces_real_cause(tmp_path, monkeypatch):
    img = tmp_path / "s.png"
    img.write_bytes(b"x")
    class FakeClient:
        def __init__(self, **kw): pass
        async def chat(self, messages, tools=None, stream=True):
            yield {"type": "error", "content": "API error (429): rate limit exceeded"}
        async def close(self): pass
    monkeypatch.setattr("spark_code.vision.ModelClient", FakeClient)
    with pytest.raises(VisionError) as ei:
        await describe_image(str(img), config=CFG)
    assert "429" in str(ei.value) and "rate limit" in str(ei.value)


async def test_empty_description_raises(tmp_path, monkeypatch):
    img = tmp_path / "s.png"
    img.write_bytes(b"x")
    class FakeClient:
        def __init__(self, **kw): pass
        async def chat(self, messages, tools=None, stream=True):
            yield {"type": "done", "usage": {}}
        async def close(self): pass
    monkeypatch.setattr("spark_code.vision.ModelClient", FakeClient)
    with pytest.raises(VisionError):
        await describe_image(str(img), config=CFG)


async def test_describe_image_uses_jpeg_mime_for_jpg(tmp_path, monkeypatch):
    # The data URL media type must be derived from the file, not hardcoded to
    # PNG — some providers reject a JPEG mislabeled as image/png.
    img = tmp_path / "shot.jpg"
    img.write_bytes(b"\xff\xd8\xff_fake_jpeg")
    seen = {}

    class FakeClient:
        def __init__(self, **kw):
            pass

        async def chat(self, messages, tools=None, stream=True):
            seen["messages"] = messages
            yield {"type": "text", "content": "ok"}

        async def close(self):
            pass
    monkeypatch.setattr("spark_code.vision.ModelClient", FakeClient)
    await describe_image(str(img), config=CFG)
    url = seen["messages"][0]["content"][1]["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")


async def test_describe_image_falls_back_to_png_for_unknown_ext(tmp_path, monkeypatch):
    img = tmp_path / "capture.unknownext"
    img.write_bytes(b"whatever")
    seen = {}

    class FakeClient:
        def __init__(self, **kw):
            pass

        async def chat(self, messages, tools=None, stream=True):
            seen["messages"] = messages
            yield {"type": "text", "content": "ok"}

        async def close(self):
            pass
    monkeypatch.setattr("spark_code.vision.ModelClient", FakeClient)
    await describe_image(str(img), config=CFG)
    url = seen["messages"][0]["content"][1]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")


async def test_describe_image_honors_config_vision_provider(tmp_path, monkeypatch):
    # cost knob: model.vision_provider picks the model (e.g. cheap flash) when
    # the caller doesn't pass one explicitly.
    img = tmp_path / "s.png"
    img.write_bytes(b"x")
    cfg = {"model": {"vision_provider": "gemini"},
           "providers": {"gemini": {"endpoint": "e", "model": "flash", "api_key": "k"}}}
    seen = {}

    class FakeClient:
        def __init__(self, **kw):
            seen["provider"] = kw.get("provider")

        async def chat(self, messages, tools=None, stream=True):
            yield {"type": "text", "content": "ok"}

        async def close(self):
            pass
    monkeypatch.setattr("spark_code.vision.ModelClient", FakeClient)
    await describe_image(str(img), config=cfg)
    assert seen["provider"] == "gemini"
