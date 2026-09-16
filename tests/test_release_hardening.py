# pyright: reportPrivateUsage=false

from __future__ import annotations

import io
import json
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from tools.release_artifacts import (
    _archive_name,
    _normalize_sbom,
    _safe_member,
    _verify_sdist,
    _verify_wheel,
)


def test_distribution_filename_normalization_preserves_import_boundary() -> None:
    assert _archive_name("kamino-lme") == "kamino_lme"


def test_archive_paths_reject_absolute_and_parent_members() -> None:
    with pytest.raises(SystemExit, match="unsafe archive member"):
        _safe_member("/absolute")
    with pytest.raises(SystemExit, match="unsafe archive member"):
        _safe_member("kamino-0.0.1/../private")


def test_sdist_rejects_development_oracle_member(tmp_path: Path) -> None:
    target = tmp_path / "kamino-0.0.1.tar.gz"
    payload = b"not distributable"
    with tarfile.open(target, "w:gz") as archive:
        info = tarfile.TarInfo("kamino-0.0.1/oracle/fixture.json")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    with pytest.raises(SystemExit, match="development-only member"):
        _verify_sdist(target, "kamino", "0.0.1")


def test_wheel_rejects_development_member(tmp_path: Path) -> None:
    target = tmp_path / "kamino-0.0.1-py3-none-any.whl"
    with zipfile.ZipFile(target, "w") as archive:
        archive.writestr("oracle/private.json", "{}")
    with pytest.raises(SystemExit, match="development-only member"):
        _verify_wheel(target, "kamino", "0.0.1")


def test_sbom_normalization_removes_time_and_random_identity(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    base: dict[str, object] = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "version": 1,
        "metadata": {"timestamp": "one"},
        "components": [],
        "dependencies": [],
    }
    first.write_text(
        json.dumps({**base, "serialNumber": "urn:uuid:first"}), encoding="utf-8"
    )
    second.write_text(
        json.dumps(
            {
                **base,
                "serialNumber": "urn:uuid:second",
                "metadata": {"timestamp": "two"},
            }
        ),
        encoding="utf-8",
    )
    _normalize_sbom(first, "core")
    _normalize_sbom(second, "core")
    assert first.read_bytes() == second.read_bytes()
    normalized = json.loads(first.read_text(encoding="utf-8"))
    assert "timestamp" not in normalized["metadata"]
    assert normalized["serialNumber"].startswith("urn:uuid:")
