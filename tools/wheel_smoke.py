"""Install the built wheel into a clean environment and import it off-tree."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parents[1]


def main() -> None:
    wheels = sorted((ROOT / "dist").glob("kamino-*.whl"))
    if len(wheels) != 1:
        raise SystemExit(f"expected exactly one Kamino wheel, found {len(wheels)}")
    with tempfile.TemporaryDirectory(prefix="kamino-wheel-smoke-") as directory:
        environment = Path(directory) / "venv"
        subprocess.run(
            ["uv", "venv", "--python", sys.executable, str(environment)], check=True
        )
        python = (
            environment / "Scripts" / "python.exe"
            if sys.platform == "win32"
            else environment / "bin" / "python"
        )
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(python),
                str(wheels[0]),
            ],
            check=True,
        )
        subprocess.run(
            [
                str(python),
                "-I",
                "-c",
                "import kamino; print(kamino.__version__)",
            ],
            cwd=directory,
            check=True,
        )


if __name__ == "__main__":
    main()
