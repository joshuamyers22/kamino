"""Install the wheel and exercise the public fitter off-tree."""

from __future__ import annotations

import subprocess
import sys
import tarfile
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[1]


def main() -> None:
    document = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    distribution = str(document["project"]["name"]).replace("-", "_")
    wheels = sorted((ROOT / "dist").glob(f"{distribution}-*.whl"))
    if len(wheels) != 1:
        raise SystemExit(f"expected exactly one Kamino wheel, found {len(wheels)}")
    sdists = sorted((ROOT / "dist").glob(f"{distribution}-*.tar.gz"))
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
                    "satt=fit.satterthwaite(); "
                    "assert satt.available and satt.test([1.]).available; "
                    "cr2=fit.cluster_robust(); "
                    "assert cr2.test([1.]).available and "
                    "cr2.covariance_type == 'CR2'; "
                    "kr=fit.kenward_roger(); "
                    "assert kr.available and kr.reml_refit and "
                    "kr.test([1.]).available; "
                    "prof=fit.profile(targets=['.sigma'],values={'.sigma':"
                    "[fit.sigma*.9,fit.sigma*1.1]}); "
                    "assert prof.baseline_kind is kamino.ObjectiveKind.ML and "
                    "len(prof.trace('.sigma').points)==3; "
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
                    "rank_data=dict(cat_data); "
                    "rank_data['duplicate']=[2.*x for x in cat_data['x']]; "
                    "ranked=kamino.lmer('y ~ x + duplicate + (1 | g)',"
                    "rank_data,reml=False); "
                    "rank_pred=ranked.predict({'x':[.25],"
                    "'duplicate':[.6],'g':['a']},mode='population'); "
                    "assert ranked.dropped_fixed_names == ('duplicate',) and "
                    "rank_pred.estimable == (False,) and "
                    "np.isnan(rank_pred.values[0]); "
                    "cr_f=(['a']*3+['b']*3+['c']*3)*10; "
                    "cr_g=[f'g{i}' for i in range(10) for _ in range(9)]; "
                    "cr_y=[5.+{'a':0.,'b':1.,'c':-.6}[f]+.12*i+"
                    ".3*np.sin(j*1.7) for j,(f,g) in "
                    "enumerate(zip(cr_f,cr_g)) for i in [int(g[1:])]]; "
                    "categorical_random=kamino.lmer('y ~ f + (1 + f | g)',"
                    "{'y':cr_y,'f':cr_f,'g':cr_g},reml=False); "
                    "assert categorical_random.random_coefficient_names == "
                    "('(Intercept)','fb','fc') and "
                    "categorical_random.theta.shape == (6,); "
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
                    "draw=fit.simulate(1,seed=123,mode='unconditional').draws[0]; "
                    "refitted=fit.refit(draw.response); "
                    "bootstrap=fit.parametric_bootstrap(2,seed=124,workers=2,"
                    "ledger_path='smoke-ledger'); "
                    "assert np.isfinite(refitted.beta).all() and "
                    "bootstrap.complete and bootstrap.failure_rate == 0.; "
                    "assert bootstrap.interval(level=.8).lower.shape == (1,); "
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
                    "ranked.save('ranked.kamino'); "
                    "rank_loaded=kamino.load_model_bundle('ranked.kamino'); "
                    "rank_loaded_pred=rank_loaded.predict({'x':[.25],"
                    "'duplicate':[.6],'g':['a']},mode='population'); "
                    "assert rank_loaded_pred.estimable == (False,) and "
                    "np.isnan(rank_loaded_pred.values[0]); "
                    "categorical.save('categorical.kamino'); "
                    "cat_loaded=kamino.load_model_bundle('categorical.kamino'); "
                    "assert np.array_equal(cat_loaded.predict({'x':[.25],"
                    "'f':['a'],'g':['b'],'o':[.2]},mode='conditional',"
                    "offset=[-.1]).values,cat_pred.values); "
                    "categorical_random.save('categorical-random.kamino'); "
                    "cat_random_loaded=kamino.load_model_bundle("
                    "'categorical-random.kamino'); "
                    "cat_random_data={'f':['a','c'],'g':['g0','new']}; "
                    "assert np.array_equal(cat_random_loaded.predict("
                    "cat_random_data,mode='conditional',allow_new_groups=True)"
                    ".values,categorical_random.predict(cat_random_data,"
                    "mode='conditional',allow_new_groups=True).values); "
                    "crossed.save('crossed.kamino'); "
                    "crossed_loaded=kamino.load_model_bundle('crossed.kamino'); "
                    "assert np.array_equal(crossed_loaded.predict("
                    "{'g':['a','new'],'h':['u','v']},mode='conditional',"
                    "allow_new_groups=True).values,crossed_pred.values); "
                    "assert pred.new_group == (False, True); "
                    "assert np.isfinite(pred.values).all(); "
                    "postfit=fit.postfit(); "
                    "assert postfit.tidy()[0].available and "
                    "postfit.performance().get('conditional_r2') > 0.; "
                    "\ntry:\n kamino.adapt_statsmodels_ols(object())\n"
                    "except kamino.PostfitError as error:\n"
                    " assert 'postfit-statsmodels' in str(error)\n"
                    "else:\n"
                    " raise AssertionError('optional adapter did not fail closed')\n"
                    "print(kamino.__version__, fit.objective)"
                ),
            ],
            cwd=directory,
            check=True,
        )


if __name__ == "__main__":
    main()
