"""Human names for map UIDs.

`VO0nIzBIyHoVjYWmHNvNnl2IUCd` is Nadeo's own identifier for a map - globally
unique, and what telemetry reports, so every config / line / occupancy dump is
keyed by it on disk. That is correct for storage and unreadable for people.

This module is a DISPLAY layer only. Files stay UID-keyed; the panel, the
logs and the CLI show the alias, and accept either form as input.

    maps/names.json :  { "<uid>": "<slug>", ... }

stdlib only - the web panel imports this before the torch venv exists.
"""
from __future__ import annotations

import json
import os
import re
import threading

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PATH = os.path.join(_ROOT, "maps", "names.json")

_lock = threading.Lock()
_cache: dict | None = None
_mtime = 0.0

# A real Nadeo map uid: 27 chars of URL-safe base64, no separators.
_UID_RE = re.compile(r"^[A-Za-z0-9_-]{20,32}$")


def _load() -> dict:
    global _cache, _mtime
    with _lock:
        try:
            m = os.path.getmtime(_PATH)
        except OSError:
            _cache, _mtime = {}, 0.0
            return _cache
        if _cache is None or m != _mtime:
            try:
                with open(_PATH) as f:
                    data = json.load(f)
                _cache = {str(k): str(v) for k, v in data.items()}
            except (OSError, ValueError):
                _cache = {}
            _mtime = m
        return _cache


def looks_like_uid(s: str) -> bool:
    return bool(s) and bool(_UID_RE.match(s))


def name_for(uid: str | None, *, short: bool = False) -> str:
    """UID -> alias, or the UID itself if there is no alias.

    short=False (default) returns "alias (VO0nIz…)" so the uid is still
    visible; short=True returns just "alias".
    """
    if not uid:
        return "(no map)"
    alias = _load().get(uid)
    if not alias:
        return uid
    if short:
        return alias
    return f"{alias} ({uid[:6]}…)"


def uid_for(name: str | None) -> str | None:
    """alias -> UID. A string that is already a UID is returned unchanged, so
    callers can pass whichever the user typed."""
    if not name:
        return None
    if looks_like_uid(name):
        return name
    for uid, alias in _load().items():
        if alias == name:
            return uid
    return None


def set_name(uid: str, alias: str) -> None:
    """Add or change an alias and persist it. Slugified: lowercase, spaces and
    punctuation to hyphens, so it is safe in a filename later."""
    slug = re.sub(r"[^a-z0-9]+", "-", alias.strip().lower()).strip("-")
    if not slug or not looks_like_uid(uid):
        raise ValueError(f"bad uid/alias: {uid!r} / {alias!r}")
    data = dict(_load())
    data[uid] = slug
    tmp = _PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, _PATH)
    with _lock:
        globals()["_cache"] = None      # force reload next call


def all_names() -> dict:
    return dict(_load())


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 2:
        print(name_for(sys.argv[1]))
    elif len(sys.argv) == 3:
        set_name(sys.argv[1], sys.argv[2])
        print(f"{sys.argv[1]} -> {name_for(sys.argv[1])}")
    else:
        for u, a in all_names().items():
            print(f"{a:24s} {u}")
