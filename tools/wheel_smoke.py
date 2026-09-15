"""Install the wheel and exercise the public fitter off-tree."""

from __future__ import annotations

import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).parents[1]


def main() -> None:
    wheels = sorted((ROOT / "dist").glob("kamino-*.whl"))
    if len(wheels) != 1:
        raise SystemExit(f"expected exactly one Kamino wheel, found {len(wheels)}")
    sdists = sorted((ROOT / "dist").glob("kamino-*.tar.gz"))
    if len(sdists) != 1:
        raise SystemExit(f"expected exactly one Kamino sdist, found {len(sdists)}")
    with tarfile.open(sdists[0], mode="r:gz") as archive:
        if any("/oracle/fixtures/" in name for name in archive.getnames()):
            raise SystemExit("GPL oracle fixtures must be absent from the MIT sdist")
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
                    "{'y':[.1,.9,2.2,10.1,11.8,14.2,-4.9,-3.2,-.8],"
                    "'x':[0.,1.,2.]*3,"
                    "'g':['a']*3+['b']*3+['c']*3}, reml=False); "
                    "slope_pred=slope.predict({'x':[0.,1.],"
                    "'g':['a','new']}, mode='conditional', "
                    "allow_new_groups=True); "
                    "assert slope.theta.shape == (3,) and "
                    "slope.random_covariance.shape == (2,2); "
                    "assert np.isfinite(slope_pred.values).all(); "
                    "independent=kamino.lmer('y ~ x + (1 + x || g)', "
                    "{'y':[.1,.9,2.2,10.1,11.8,14.2,-4.9,-3.2,-.8],"
                    "'x':[0.,1.,2.]*3,"
                    "'g':['a']*3+['b']*3+['c']*3}, reml=False); "
                    "assert independent.theta.shape == (2,) and "
                    "independent.covariance_term_sizes == (1,1) and "
                    "independent.random_covariance[0,1] == 0.; "
                    "cat_data={'x':[-1.,0.,1.]*4,"
                    "'f':['base','a','b','a','b','base','b','base','a',"
                    "'base','b','a'],'g':['a']*3+['b']*3+['c']*3+['d']*3,"
                    "'o':[.1]*12}; "
                    "cat_data['y']=[2.+.6*x+{'base':0.,'a':.3,'b':-.2}[f]+"
                    "{'a':-.5,'b':.2,'c':.6,'d':-.1}[g]+.1-.05 "
                    "+[.1,-.05,.02,-.08,.07,-.03,.04,-.09,.06,-.02,.08,-.04][i] "
                    "for i,(x,f,g) in enumerate(zip(cat_data['x'],cat_data['f'],"
                    "cat_data['g']))]; "
                    "categorical=kamino.lmer("
                    "'y ~ x + f + offset(o) + (1 | g)',cat_data,reml=False,"
                    "weights=[1.,1.2,.8]*4,offset=[-.05]*12,"
                    "contrasts={'f':'sum'}); "
                    "cat_pred=categorical.predict({'x':[.25],'f':['a'],"
                    "'g':['b'],'o':[.2]},mode='conditional',offset=[-.1]); "
                    "assert categorical.fixed_names == "
                    "('(Intercept)','x','f1','f2') and "
                    "np.isfinite(cat_pred.values).all(); "
                    "crossed=kamino.lmer('y ~ 1 + (1 | g) + (1 | h)',"
                    "{'y':[1.,1.2,2.1,2.3,3.2,3.0,1.1,1.4,2.4,2.0,3.3,3.1],"
                    "'g':['a']*4+['b']*4+['c']*4,"
                    "'h':['u','v','u','v']*3},reml=False); "
                    "crossed_pred=crossed.predict({'g':['a','new'],"
                    "'h':['u','v']},mode='conditional',allow_new_groups=True); "
                    "assert crossed.diagnostics.backend == "
                    "'scipy-superlu-symmetric-sparse' and "
                    "crossed.theta.shape == (2,) and "
                    "crossed_pred.new_group == (False,True); "
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
                    "categorical.save('categorical.kamino'); "
                    "cat_loaded=kamino.load_model_bundle('categorical.kamino'); "
                    "assert np.array_equal(cat_loaded.predict({'x':[.25],"
                    "'f':['a'],'g':['b'],'o':[.2]},mode='conditional',"
                    "offset=[-.1]).values,cat_pred.values); "
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
