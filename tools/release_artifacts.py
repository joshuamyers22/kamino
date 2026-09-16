"""Validate release archives and emit reproducible supply-chain metadata."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import os
import re
import stat
import subprocess
import tarfile
import tomllib
import uuid
import zipfile
from collections.abc import Iterable
from email import policy
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from typing import Any, cast

ROOT = Path(__file__).parents[1]
DIST = ROOT / "dist"
OUTPUT = ROOT / "build" / "release"
REPOSITORY = "https://github.com/joshuamyers22/kamino"
BUILD_TYPE = f"{REPOSITORY}/.github/workflows/release.yml@v1"
IMPORT_PACKAGE = "kamino"
MAX_SDIST_BYTES = 8 * 1024 * 1024
MAX_WHEEL_BYTES = 8 * 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project() -> tuple[str, str]:
    document = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = cast(dict[str, Any], document["project"])
    return str(project["name"]), str(project["version"])


def _archive_name(name: str) -> str:
    """Return the wheel/sdist filename form of a distribution name."""
    return re.sub(r"[-_.]+", "_", name).lower()


def _safe_member(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise SystemExit(f"unsafe archive member: {name!r}")
    return path


def _one_artifact(pattern: str) -> Path:
    matches = tuple(sorted(DIST.glob(pattern)))
    if len(matches) != 1:
        raise SystemExit(f"expected exactly one {pattern!r} artifact, found {matches}")
    return matches[0]


def _verify_record(archive: zipfile.ZipFile, record_name: str) -> None:
    rows = csv.reader(io.StringIO(archive.read(record_name).decode("utf-8")))
    recorded: set[str] = set()
    for name, encoded_hash, encoded_size in rows:
        if name in recorded:
            raise SystemExit(f"duplicate wheel RECORD row: {name}")
        recorded.add(name)
        if name == record_name:
            if encoded_hash or encoded_size:
                raise SystemExit("wheel RECORD must not hash itself")
            continue
        try:
            payload = archive.read(name)
        except KeyError as error:
            raise SystemExit(f"wheel RECORD names missing member: {name}") from error
        expected_size = int(encoded_size)
        if len(payload) != expected_size:
            raise SystemExit(f"wheel RECORD size mismatch: {name}")
        algorithm, expected = encoded_hash.split("=", 1)
        if algorithm != "sha256":
            raise SystemExit(f"wheel RECORD uses unsupported hash: {algorithm}")
        actual = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=")
        if actual.decode("ascii") != expected:
            raise SystemExit(f"wheel RECORD hash mismatch: {name}")
    members = {value.filename for value in archive.infolist()}
    if recorded != members:
        raise SystemExit("wheel RECORD membership differs from the archive")


def _verify_wheel(path: Path, name: str, version: str) -> None:
    if path.stat().st_size > MAX_WHEEL_BYTES:
        raise SystemExit("wheel exceeds the release size limit")
    distribution = f"{_archive_name(name)}-{version}.dist-info"
    license_name = f"{distribution}/licenses/LICENSE"
    metadata_name = f"{distribution}/METADATA"
    record_name = f"{distribution}/RECORD"
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = [value.filename for value in infos]
        if len(names) != len(set(names)):
            raise SystemExit("wheel contains duplicate members")
        for info in infos:
            member = _safe_member(info.filename)
            mode = (info.external_attr >> 16) & 0o170000
            if mode == stat.S_IFLNK:
                raise SystemExit(f"wheel contains a symbolic link: {info.filename}")
            if member.parts[0] not in {IMPORT_PACKAGE, distribution}:
                raise SystemExit(
                    f"wheel contains development-only member: {info.filename}"
                )
            if member.suffix in {".pyc", ".pyo", ".pickle", ".pkl"}:
                raise SystemExit(
                    f"wheel contains unsafe generated data: {info.filename}"
                )
        for required in (license_name, metadata_name, record_name):
            if required not in names:
                raise SystemExit(f"wheel is missing {required}")
        if archive.read(license_name) != (ROOT / "LICENSE").read_bytes():
            raise SystemExit("wheel license differs from repository LICENSE")
        metadata = BytesParser(policy=policy.default).parsebytes(
            archive.read(metadata_name)
        )
        if metadata["Name"] != name or metadata["Version"] != version:
            raise SystemExit("wheel metadata identity differs from pyproject.toml")
        if metadata["License-Expression"] != "MIT":
            raise SystemExit("wheel does not declare the reviewed MIT license")
        extras = set(metadata.get_all("Provides-Extra", []))
        if extras != {"postfit-statsmodels"}:
            raise SystemExit(f"unexpected published optional extras: {sorted(extras)}")
        _verify_record(archive, record_name)


def _verify_sdist(path: Path, name: str, version: str) -> None:
    if path.stat().st_size > MAX_SDIST_BYTES:
        raise SystemExit("source distribution exceeds the release size limit")
    root = f"{_archive_name(name)}-{version}"
    forbidden = {
        ".git",
        ".github",
        ".work",
        "benchmarks",
        "oracle",
        "statistical",
        "tests",
        "tools",
    }
    names: set[str] = set()
    total = 0
    with tarfile.open(path, mode="r:gz") as archive:
        for info in archive.getmembers():
            member = _safe_member(info.name)
            if member.parts[0] != root:
                raise SystemExit(f"sdist has an unexpected root: {info.name}")
            if len(member.parts) > 1 and member.parts[1] in forbidden:
                raise SystemExit(f"sdist contains development-only member: {info.name}")
            if info.issym() or info.islnk() or info.isdev():
                raise SystemExit(f"sdist contains a special member: {info.name}")
            if info.name in names:
                raise SystemExit(f"sdist contains a duplicate member: {info.name}")
            names.add(info.name)
            total += info.size
        if total > MAX_SDIST_BYTES:
            raise SystemExit("unpacked sdist exceeds the release size limit")
        required = {
            f"{root}/LICENSE",
            f"{root}/README.md",
            f"{root}/pyproject.toml",
            f"{root}/src/kamino/__init__.py",
        }
        missing = required - names
        if missing:
            raise SystemExit(f"sdist is missing required members: {sorted(missing)}")
        license_member = archive.extractfile(f"{root}/LICENSE")
        if (
            license_member is None
            or license_member.read() != (ROOT / "LICENSE").read_bytes()
        ):
            raise SystemExit("sdist license differs from repository LICENSE")


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()


def _normalize_sbom(path: Path, scope: str) -> None:
    raw = json.loads(path.read_text(encoding="utf-8"))
    document = cast(dict[str, Any], raw)
    if document.get("bomFormat") != "CycloneDX" or document.get("specVersion") != "1.5":
        raise SystemExit(f"uv emitted an unexpected SBOM format for {scope}")
    metadata = cast(dict[str, Any], document.setdefault("metadata", {}))
    metadata.pop("timestamp", None)
    properties = cast(list[dict[str, str]], metadata.setdefault("properties", []))
    properties.extend(
        [
            {"name": "kamino:dependency-scope", "value": scope},
            {"name": "kamino:uv-lock-sha256", "value": _sha256(ROOT / "uv.lock")},
        ]
    )
    document["serialNumber"] = "urn:uuid:00000000-0000-0000-0000-000000000000"
    seed = json.dumps(document, allow_nan=False, sort_keys=True, separators=(",", ":"))
    document["serialNumber"] = f"urn:uuid:{uuid.uuid5(uuid.NAMESPACE_URL, seed)}"
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _export_sbom(path: Path, *, all_extras: bool) -> None:
    command = [
        "uv",
        "export",
        "--quiet",
        "--frozen",
        "--no-dev",
        "--format",
        "cyclonedx1.5",
        "--output-file",
        str(path),
    ]
    if all_extras:
        command.append("--all-extras")
    subprocess.run(command, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    _normalize_sbom(path, "all-extras" if all_extras else "core")


def _artifact_records(paths: Iterable[Path]) -> list[dict[str, object]]:
    return [
        {
            "path": str(path.relative_to(ROOT)),
            "sha256": _sha256(path),
            "size": path.stat().st_size,
        }
        for path in sorted(paths)
    ]


def main() -> None:
    name, version = _project()
    archive_name = _archive_name(name)
    wheel = _one_artifact(f"{archive_name}-{version}-*.whl")
    sdist = _one_artifact(f"{archive_name}-{version}.tar.gz")
    _verify_wheel(wheel, name, version)
    _verify_sdist(sdist, name, version)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    core_sbom = OUTPUT / "kamino-core.cdx.json"
    extras_sbom = OUTPUT / "kamino-all-extras.cdx.json"
    _export_sbom(core_sbom, all_extras=False)
    _export_sbom(extras_sbom, all_extras=True)

    artifacts = (wheel, sdist, core_sbom, extras_sbom)
    records = _artifact_records(artifacts)
    checksums = "".join(f"{record['sha256']}  {record['path']}\n" for record in records)
    (OUTPUT / "SHA256SUMS").write_text(checksums, encoding="utf-8")
    try:
        source_commit = _git("rev-parse", "HEAD")
        source_remote = _git("remote", "get-url", "origin")
        dirty = bool(_git("status", "--short", "--untracked-files=no"))
    except (OSError, subprocess.CalledProcessError):
        source_commit = os.environ.get("GITHUB_SHA", "unknown")
        source_remote = REPOSITORY
        dirty = True
    manifest = {
        "format": "kamino-release-manifest",
        "schema_version": "1.0.0",
        "package": {"name": name, "version": version},
        "source": {
            "commit": source_commit,
            "dirty_tracked_tree": dirty,
            "repository": source_remote,
        },
        "build": {
            "build_type": BUILD_TYPE,
            "uv_lock_sha256": _sha256(ROOT / "uv.lock"),
            "workflow_ref": os.environ.get("GITHUB_WORKFLOW_REF"),
            "run_id": os.environ.get("GITHUB_RUN_ID"),
        },
        "artifacts": records,
    }
    (OUTPUT / "release-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"verified {wheel.relative_to(ROOT)}")
    print(f"verified {sdist.relative_to(ROOT)}")
    print(f"wrote {OUTPUT.relative_to(ROOT)} supply-chain evidence")


if __name__ == "__main__":
    main()
