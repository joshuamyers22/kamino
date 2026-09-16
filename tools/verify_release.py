"""Fail closed unless a release tag matches reviewed package metadata."""

from __future__ import annotations

import argparse
import re
import subprocess
import tomllib
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).parents[1]


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    arguments = parser.parse_args()
    tag = str(arguments.tag)
    if re.fullmatch(r"v(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)", tag) is None:
        raise SystemExit("release tag must be an exact vMAJOR.MINOR.PATCH value")
    document = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = cast(dict[str, Any], document["project"])
    version = str(project["version"])
    if tag != f"v{version}":
        raise SystemExit(
            f"release tag {tag!r} does not match package version {version!r}"
        )
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    heading = rf"^## \[{re.escape(version)}\] - \d{{4}}-\d{{2}}-\d{{2}}$"
    if re.search(heading, changelog, re.MULTILINE) is None:
        raise SystemExit(f"CHANGELOG.md has no dated [{version}] release heading")
    head = _git("rev-parse", "HEAD")
    tagged = _git("rev-list", "-n", "1", tag)
    if tagged != head:
        raise SystemExit(
            f"release tag points to {tagged}, not checked-out commit {head}"
        )
    if _git("status", "--short", "--untracked-files=no"):
        raise SystemExit("tracked worktree changes are present")
    print(f"verified {tag} at {head}")


if __name__ == "__main__":
    main()
