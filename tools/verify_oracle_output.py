"""Verify that regenerated R oracle output has the reviewed manifest hash."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parents[1]
MANIFEST = ROOT / "oracle" / "manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", action="store_true")
    arguments = parser.parse_args()
    manifest: dict[str, Any] = json.loads(MANIFEST.read_text(encoding="utf-8"))
    source_hashes = {
        "oracle/generate_fixtures.R": manifest["fixture_generator_sha256"],
        "oracle/Dockerfile": manifest["container"]["dockerfile_sha256"],
    }
    for relative_path, expected in source_hashes.items():
        path = ROOT / relative_path
        actual = sha256(path)
        if actual != expected:
            message = f"source hash mismatch for {path}: {expected=} {actual=}"
            raise SystemExit(message)
        print(f"verified {relative_path} sha256:{actual}")
    for output in manifest["outputs"]:
        path = ROOT / output["path"]
        actual = sha256(path)
        expected = output["sha256"]
        if actual != expected:
            raise SystemExit(
                f"oracle hash mismatch for {path}: expected {expected}, got {actual}"
            )
        print(f"verified {path.relative_to(ROOT)} sha256:{actual}")
    if arguments.image:
        actual_image = subprocess.check_output(
            [
                "docker",
                "image",
                "inspect",
                "kamino-oracle:phase0",
                "--format",
                "{{.Id}}",
            ],
            text=True,
        ).strip()
        expected_image = manifest["container"]["built_image_digest"]
        if actual_image != expected_image:
            raise SystemExit(
                f"oracle image mismatch: expected {expected_image}, got {actual_image}"
            )
        print(f"verified kamino-oracle:phase0 {actual_image}")


if __name__ == "__main__":
    main()
