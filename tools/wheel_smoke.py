"""Install the wheel and exercise the public fitter off-tree."""

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
                (
                    "import numpy as np; import kamino; "
                    "fit=kamino.lmer('y ~ 1 + (1 | g)', "
                    "{'y':[0.,.1,-.1,10.,10.1,9.9],"
                    "'g':['a','a','a','b','b','b']}, reml=False); "
                    "pred=fit.predict({'g':np.array(['a','new'])}, "
                    "mode='conditional', allow_new_groups=True); "
                    "assert fit.diagnostics.converged and fit.theta[0] > 0; "
                    "assert pred.new_group == (False, True); "
                    "assert np.isfinite(pred.values).all(); "
                    "print(kamino.__version__, fit.objective)"
                ),
            ],
            cwd=directory,
            check=True,
        )


if __name__ == "__main__":
    main()
