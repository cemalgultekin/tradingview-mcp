"""Smoke-check that the server module imports and registers its tools.

Run via ``uv run python scripts/verify_server.py``.

This is deliberately a real file rather than inline ``python -c`` in the setup
scripts: Windows PowerShell 5.1 strips double quotes when forwarding arguments
to a native executable, which silently corrupts inlined Python source.

It catches a failure mode the pytest suite can mask -- the server module not
importing at all (a missing service module, a broken tool registration) -- and
exits non-zero so the calling script can stop.
"""
from __future__ import annotations

import sys

MIN_TOOLS = 30


def main() -> int:
    try:
        from tradingview_mcp.server import mcp
    except Exception as exc:  # noqa: BLE001 - want the reason, whatever it is
        print(f"    FAIL - server did not import: {type(exc).__name__}: {exc}")
        return 1

    tools = mcp._tool_manager.list_tools()
    if len(tools) < MIN_TOOLS:
        print(f"    FAIL - tool count suspiciously low: {len(tools)} (expected >= {MIN_TOOLS})")
        return 1

    unannotated = [t.name for t in tools if t.annotations is None or not (t.annotations.title or "").strip()]
    if unannotated:
        print(f"    FAIL - tools missing annotations/title: {unannotated}")
        return 1

    print(f"    OK - {len(tools)} tools registered, all annotated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
