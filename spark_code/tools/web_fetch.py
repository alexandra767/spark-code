"""Web fetch tool — fetch and parse web pages with streaming support."""

import asyncio
import ipaddress
import socket
from urllib.parse import urljoin, urlparse

import httpx

from .base import Tool

_MAX_REDIRECTS = 5
# RFC 6598 shared/CGNAT space — Tailscale's 100.x range lives here and older
# Python's ipaddress didn't flag it as private. Checked explicitly.
_CGNAT = ipaddress.ip_network("100.64.0.0/10")


async def _resolves_to_internal(host: str) -> bool:
    """True if *host* resolves to any non-public address (loopback, private,
    link-local — incl. the 169.254.169.254 cloud-metadata endpoint — reserved,
    unspecified, multicast, or CGNAT/tailnet). Unresolvable → treated as
    blocked (fail closed)."""
    loop = asyncio.get_event_loop()
    try:
        infos = await loop.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except (socket.gaierror, UnicodeError, OSError):
        return True
    if not infos:
        return True
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            return True
        if (ip.is_loopback or ip.is_private or ip.is_link_local
                or ip.is_reserved or ip.is_unspecified or ip.is_multicast):
            return True
        if ip.version == 4 and ip in _CGNAT:
            return True
    return False


async def _validate_url(url: str) -> str | None:
    """Return an error string if *url* is unsafe to fetch, else None.

    SECURITY (SSRF guard): web_fetch is treated as read-only, so in auto mode
    it fetches with no permission prompt. Without this a prompt-injected URL —
    or a redirect from an innocent-looking one — could reach the cloud-metadata
    endpoint or an unauthenticated tailnet service (RAG :8010, Orrery :4200,
    revdash). Only http(s) to a public address is allowed. (Residual: DNS
    rebinding between this check and httpx's own resolution is not defended —
    out of scope for a local coding assistant.)"""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return f"Error: refusing to fetch non-http(s) URL scheme '{parsed.scheme or '(none)'}'"
    host = parsed.hostname
    if not host:
        return "Error: URL has no host"
    if await _resolves_to_internal(host):
        return ("Error: refusing to fetch a private/internal address "
                "(SSRF guard — loopback, private, link-local, metadata, or tailnet)")
    return None


class WebFetchTool(Tool):
    name = "web_fetch"
    description = "Fetch a web page and return its text content. Useful for reading documentation, articles, and API references."
    is_read_only = True

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch",
                },
                "max_length": {
                    "type": "integer",
                    "description": "Maximum characters to return (default: 10000)",
                },
            },
            "required": ["url"],
        }

    async def execute(self, url: str, max_length: int = 10000, **kw) -> str:
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            return "Error: beautifulsoup4 not installed. Run: pip install beautifulsoup4"

        headers = {"User-Agent": "Mozilla/5.0 (compatible; SparkCode/1.0)"}
        try:
            # follow_redirects=False + a manual, per-hop validated loop so a
            # redirect can't bounce us onto an internal address after the
            # first URL passed the SSRF check.
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=False) as client:
                current = url
                response = None
                for _ in range(_MAX_REDIRECTS + 1):
                    err = await _validate_url(current)
                    if err:
                        return err
                    response = await client.get(current, headers=headers)
                    if response.is_redirect:
                        loc = response.headers.get("location")
                        if not loc:
                            break
                        current = urljoin(current, loc)
                        continue
                    break
                else:
                    return "Error fetching URL: too many redirects"
                response.raise_for_status()
        except Exception as e:
            return f"Error fetching URL: {e}"

        content_type = response.headers.get("content-type", "")
        if "html" in content_type:
            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
        else:
            text = response.text

        if len(text) > max_length:
            text = text[:max_length] + f"\n\n... (truncated at {max_length} chars)"

        return f"URL: {url}\n\n{text}"

    async def execute_streaming(self, url: str, max_length: int = 10000,
                                callback=None, **kw) -> str:
        """Fetch with streaming progress updates."""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            return "Error: beautifulsoup4 not installed."

        headers = {"User-Agent": "Mozilla/5.0 (compatible; SparkCode/1.0)"}
        try:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=False) as client:
                # Resolve redirects with cheap validated GETs first, then stream
                # the final destination (each hop SSRF-checked).
                current = url
                for _ in range(_MAX_REDIRECTS + 1):
                    err = await _validate_url(current)
                    if err:
                        return err
                    head = await client.get(current, headers=headers)
                    if head.is_redirect:
                        loc = head.headers.get("location")
                        if not loc:
                            break
                        current = urljoin(current, loc)
                        continue
                    break
                else:
                    return "Error fetching URL: too many redirects"

                async with client.stream("GET", current, headers=headers) as response:
                    response.raise_for_status()

                    total = int(response.headers.get("content-length", 0))
                    chunks = []
                    downloaded = 0

                    async for chunk in response.aiter_bytes(1024):
                        chunks.append(chunk)
                        downloaded += len(chunk)
                        if callback and total > 0:
                            pct = downloaded / total * 100
                            callback(f"Downloading... {downloaded:,}/{total:,} bytes ({pct:.0f}%)")
                        elif callback:
                            callback(f"Downloading... {downloaded:,} bytes")

                    body = b"".join(chunks)
                    text = body.decode("utf-8", errors="replace")

        except Exception as e:
            return f"Error fetching URL: {e}"

        content_type = response.headers.get("content-type", "")
        if "html" in content_type:
            soup = BeautifulSoup(text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)

        if len(text) > max_length:
            text = text[:max_length] + f"\n\n... (truncated at {max_length} chars)"

        if callback:
            callback(f"Fetched {len(text):,} chars from {url}")

        return f"URL: {url}\n\n{text}"
