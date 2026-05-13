#!/usr/bin/env python3
# Throw-away helper for the utils refactor. Deleted at the end of the branch.
#
# Usage:
#   python scripts/refactor/rewrite_imports.py
#
# Reads MAPPINGS below and rewrites every .py file under the repo (excluding
# the script itself, .git, and __pycache__).
#
# For each (old_mod, new_mod) pair it rewrites:
#   from old_mod import X         -> from new_mod import X
#   from old_mod.sub import X     -> from new_mod.sub import X
#   import old_mod                -> import new_mod
#   import old_mod.sub            -> import new_mod.sub
#   import old_mod as Z           -> import new_mod as Z
#
# The "from miles.utils import old_leaf" pattern is NOT handled here; those
# cases (about 6 of them) are rewritten by hand.

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# (old_module_dotted_path, new_module_dotted_path)
# Order matters: longer/more-specific entries should come first so they
# don't get clobbered by a shorter prefix replacement.
MAPPINGS: list[tuple[str, str]] = []


def load_mappings(arg: str | None) -> None:
    """Allow loading mappings from a file path (one 'old new' per line)."""
    if not arg:
        return
    path = Path(arg)
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        old, new = line.split()
        MAPPINGS.append((old, new))


def walk_py_files() -> list[Path]:
    skip_parts = {".git", "__pycache__", "node_modules", ".venv", "venv", "build", "dist"}
    out: list[Path] = []
    for p in REPO.rglob("*.py"):
        if any(part in skip_parts for part in p.parts):
            continue
        # Don't rewrite the rewriter itself.
        if p.resolve() == Path(__file__).resolve():
            continue
        out.append(p)
    return out


def rewrite_text(text: str, mappings: list[tuple[str, str]]) -> tuple[str, int]:
    n = 0
    for old, new in mappings:
        old_re = re.escape(old)
        # from X.something import ...  /  from X import ...
        pattern1 = re.compile(rf"\bfrom\s+{old_re}(\b)")
        text, c1 = pattern1.subn(lambda m: f"from {new}{m.group(1)}", text)
        # import X.something  /  import X  (boundary that isn't a dot continuation)
        # We use a negative lookahead to avoid consuming partial matches mid-identifier.
        pattern2 = re.compile(rf"\bimport\s+{old_re}(?=[\s.,\n#]|$)")
        text, c2 = pattern2.subn(f"import {new}", text)
        n += c1 + c2
    return text, n


def main() -> int:
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    load_mappings(arg)
    if not MAPPINGS:
        print("No mappings loaded. Pass a mapping file path.", file=sys.stderr)
        return 2

    # Longest old-paths first so prefix mappings don't clobber longer rewrites
    # that produced extended paths in the same pass.
    MAPPINGS.sort(key=lambda pair: -len(pair[0]))

    total_files = 0
    total_subs = 0
    for path in walk_py_files():
        original = path.read_text()
        new_text, n = rewrite_text(original, MAPPINGS)
        if n and new_text != original:
            path.write_text(new_text)
            total_files += 1
            total_subs += n
            print(f"{path.relative_to(REPO)}: {n} subs")
    print(f"-- {total_files} files, {total_subs} substitutions --")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
