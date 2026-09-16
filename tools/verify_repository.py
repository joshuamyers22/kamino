"""Verify offline repository governance and workflow invariants."""

from __future__ import annotations

import re
import subprocess
import tomllib
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).parents[1]
ACTION = re.compile(r"^\s*-\s+uses:\s+([^\s#]+)", re.MULTILINE)
LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
SHA = re.compile(r"[0-9a-f]{40}")


def _repository_files() -> tuple[Path, ...]:
    output = subprocess.check_output(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        stderr=subprocess.DEVNULL,
    )
    return tuple(ROOT / value.decode("utf-8") for value in output.split(b"\0") if value)


def _verify_links(paths: tuple[Path, ...]) -> None:
    missing: list[str] = []
    for path in paths:
        if path.suffix.lower() != ".md" or "docs/archive" in path.as_posix():
            continue
        text = path.read_text(encoding="utf-8")
        for raw in LINK.findall(text):
            target = raw.strip().strip("<>").split("#", 1)[0]
            if not target or re.match(r"(?:https?|mailto):", target):
                continue
            resolved = (path.parent / unquote(target)).resolve()
            try:
                resolved.relative_to(ROOT)
            except ValueError:
                missing.append(f"{path.relative_to(ROOT)} -> outside repository: {raw}")
                continue
            if not resolved.exists():
                missing.append(f"{path.relative_to(ROOT)} -> missing: {raw}")
    if missing:
        raise SystemExit("invalid local Markdown links:\n" + "\n".join(missing))


def _verify_actions(paths: tuple[Path, ...]) -> None:
    invalid: list[str] = []
    for path in paths:
        if path.parent != ROOT / ".github" / "workflows" or path.suffix != ".yml":
            continue
        for reference in ACTION.findall(path.read_text(encoding="utf-8")):
            if reference.startswith("./"):
                continue
            if "@" not in reference:
                invalid.append(f"{path.name}: {reference}")
                continue
            revision = reference.rsplit("@", 1)[1]
            if SHA.fullmatch(revision) is None:
                invalid.append(f"{path.name}: {reference}")
    if invalid:
        raise SystemExit(
            "workflow actions must use full commit SHAs:\n" + "\n".join(invalid)
        )


def _verify_distribution_boundary() -> None:
    document = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    sdist = document["tool"]["hatch"]["build"]["targets"]["sdist"]
    included = set(sdist["include"])
    expected = {"/CHANGELOG.md", "/LICENSE", "/README.md", "/pyproject.toml", "/src/**"}
    if included != expected:
        raise SystemExit(f"sdist inclusion boundary drifted: {sorted(included)}")
    extras = set(document["project"].get("optional-dependencies", {}))
    if extras != {"postfit-statsmodels"}:
        raise SystemExit(f"published optional extras drifted: {sorted(extras)}")


def main() -> None:
    paths = _repository_files()
    _verify_links(paths)
    _verify_actions(paths)
    _verify_distribution_boundary()
    print("verified local links, immutable actions, and distribution boundary")


if __name__ == "__main__":
    main()
