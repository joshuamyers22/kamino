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
                    "boundary=kamino.lmer('y ~ 1 + (1 | g)', "
                    "{'y':[1.,2.,1.,2.,1.,2.],"
                    "'g':['a','a','b','b','c','c']}); "
                    "assert boundary.theta[0] == 0 and "
                    "boundary.diagnostics.boundary; "
                    "slope=kamino.lmer('y ~ x + (1 + x | g)', "
                    "{'y':[0.,1.,2.,10.,12.,14.,-5.,-3.,-1.],"
                    "'x':[0.,1.,2.]*3,"
                    "'g':['a']*3+['b']*3+['c']*3}, reml=False); "
                    "slope_pred=slope.predict({'x':[0.,1.],"
                    "'g':['a','new']}, mode='conditional', "
                    "allow_new_groups=True); "
                    "assert slope.theta.shape == (3,) and "
                    "slope.random_covariance.shape == (2,2); "
                    "assert np.isfinite(slope_pred.values).all(); "
                    "independent=kamino.lmer('y ~ x + (1 + x || g)', "
                    "{'y':[0.,1.,2.,10.,12.,14.,-5.,-3.,-1.],"
                    "'x':[0.,1.,2.]*3,"
                    "'g':['a']*3+['b']*3+['c']*3}, reml=False); "
                    "assert independent.theta.shape == (2,) and "
                    "independent.covariance_term_sizes == (1,1) and "
                    "independent.random_covariance[0,1] == 0.; "
                    "independent.save('smoke.kamino'); "
                    "loaded=kamino.load_model_bundle('smoke.kamino'); "
                    "loaded_pred=loaded.predict({'x':[0.,1.],"
                    "'g':['a','new']}, mode='conditional', "
                    "allow_new_groups=True); "
                    "assert np.array_equal(loaded_pred.values, "
                    "independent.predict({'x':[0.,1.],"
                    "'g':['a','new']}, mode='conditional', "
                    "allow_new_groups=True).values); "
                    "assert loaded.covariance_term_sizes == (1,1); "
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
