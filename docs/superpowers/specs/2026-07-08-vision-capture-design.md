# Spark Code — phone screenshot capture + Gemini vision-describe

- **Date:** 2026-07-08
- **Status:** Design approved (verbal); spec awaiting user review before writing-plans

## Goal

Let Spark Code capture a screenshot of a real phone (**Android** via `adb`, **iPhone** via `idevicescreenshot`), send it to **Gemini**, and return a **text description** the text-only local 80B can read and work with. Core insight: the 80B can't see pixels; Gemini can, so any captured image becomes text for the 80B.

## Runtime / prerequisites

- `adb` at `~/Library/Android/sdk/platform-tools/adb` (or on PATH / `$ANDROID_HOME`). `idevicescreenshot` (libimobiledevice) on PATH.
- Gemini provider `gemini-pro` (gemini-2.5-pro, multimodal) already in `~/.spark/config.yaml`.
- `ModelClient.chat(messages)` sends messages verbatim → supports OpenAI-style `image_url` content parts to Gemini. No new deps.
- **No device is connected as of this writing** — the capture tools must fail cleanly with "no device connected"; a real-screen capture needs a device plugged in.

## Components

### 1. `spark_code/vision.py` — Gemini image describer (the reusable core)
- `VisionError(Exception)`.
- `async describe_image(image_path, prompt=None, provider_name="gemini-pro", config=None) -> str`:
  - load `config` (via `config.load_config()` if not passed); resolve `providers[provider_name]` (error if missing).
  - build a `ModelClient` for that provider (`resolve_provider_key`/`load_keys`, `supports_vision=True`).
  - read the image, base64-encode, send one user message: `[{"type":"text","text": prompt or DEFAULT_VISION_PROMPT}, {"type":"image_url","image_url":{"url":"data:image/png;base64,<b64>"}}]`.
  - collect text chunks from `chat(messages, stream=False)`; `await client.close()`; return the description. Raise `VisionError` on empty/failed response.
  - `DEFAULT_VISION_PROMPT`: "You are the eyes for another AI. Describe this screen in detail: all visible text, UI elements, buttons, layout, current state, and anything notable. Be thorough and literal."

### 2. `spark_code/tools/android_screenshot.py`
`name="android_screenshot"`, `is_read_only=True`, `requires_permission=False`, optional param `device` (serial), optional `prompt`.
- Resolve adb (PATH → `~/Library/Android/sdk/platform-tools/adb` → `$ANDROID_HOME/platform-tools/adb`); if none → clear error.
- `adb [-s device] exec-out screencap -p` capturing raw PNG bytes from stdout → temp file. If no device / adb error (`no devices/emulators found`) → return a clean "no Android device connected (enable USB debugging + plug in)" message.
- `describe_image(png)` → return "📱 Android screenshot — Gemini sees:\n<description>". Clean up the temp file.

### 3. `spark_code/tools/ios_screenshot.py`
`name="ios_screenshot"`, `is_read_only=True`, `requires_permission=False`, optional `prompt`.
- `idevicescreenshot <tmp.png>` (real paired iPhone). If not installed or no device → clean error ("no paired iPhone connected / trust the Mac"). idevicescreenshot may emit `.tiff` for some devices — request a `.png` path; if it writes tiff, note it. 
- `describe_image(png)` → return "📱 iPhone screenshot — Gemini sees:\n<description>". Clean up temp.

Both tools registered in `cli.py` `build_tools`.

## Safety

- Read-only observation (capture a screen, call Gemini) — no writes, no gate. Gemini token cost is minor and per-invocation.
- Temp images deleted after describe. Nothing persisted into the repo/project tree.

## Testing

- **Unit (mock subprocess + `describe_image`/`ModelClient`):** `describe_image` builds the correct Gemini message (text + `image_url` data URL) and returns the collected text; adb resolution order; android/ios tools capture argv; **no-device error path** returns a clean message and does NOT call `describe_image`.
- **Live acceptance (controller-run), no phone needed:**
  1. **Vision core:** generate a PNG containing a known random token, call `describe_image` → assert Gemini's description contains the token (ground-truth: image→Gemini→text works end to end).
  2. **Capture no-device path:** run `android_screenshot` / `ios_screenshot` with nothing connected → confirm the clean "no device" message (real, since none is attached).
- **Full real-screen capture** (Android via `adb`, iPhone via pairing) is deferred to when Alexandra plugs a device in.

## Out of scope

Routing the existing `simulator_screenshot` / web `browser_take_screenshot` through `describe_image` (the helper is reusable for those later); video/scrcpy; tapping/driving the device.

## Risks / notes

- No device connected now → full capture untestable until one is; the vision core + no-device paths ARE testable now.
- `idevicescreenshot` output format (png vs tiff) varies by device; handle both or request png.
- Gemini vision cost per screenshot (paid key); occasional use.
- Changes land on branch `feat/vision-capture` in the live `~/spark-code`.
