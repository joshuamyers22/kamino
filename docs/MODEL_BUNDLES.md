# Prediction-only model bundles

Kamino's `.kamino` bundle is an explicit persistence format for prediction. It
does not save enough caller data to refit a model.

```python
from kamino import lmer, load_model_bundle

fit = lmer("y ~ 1 + (1 | g)", data)
fit.save("model.kamino")

model = load_model_bundle("model.kamino")
prediction = model.predict(
    {"g": ["known", "new"]},
    mode="conditional",
    allow_new_groups=True,
)
```

Existing files are not replaced unless `overwrite=True`. Publication is atomic.
Loading validates the versioned schema, internal paths, dtypes, shapes, finite
values, size limits, per-array SHA-256 hashes, the model-content hash, ordered
coefficient/group identity, and covariance consistency. Pickle and NumPy object
arrays are never loaded. `BundleLimits` can impose tighter file, member, or
element ceilings before allocation.

Schema `1.1.0` records covariance-term boundaries so independent terms sharing a
grouping factor remain independent after reload. Schema `1.2.0` also records the
owned fixed-effect encoder and formula-offset names, including ordered factor
levels and treatment/sum coding. The loader continues to accept schemas `1.0.0`
and `1.1.0` for their original intercept/numeric fixed designs. Saving always
writes the current schema.

Schema 1.2 does not encode the full-to-retained rank map/null-space basis,
categorical random encoders, or multiple grouping structures. Saving a
rank-deficient, categorical-random, nested, or crossed model fails explicitly
until the Phase 2 artifact-recovery schema is separately reviewed and gated.

The loaded `PredictionOnlyModel` requires explicit new data. It exposes only
population and conditional prediction. It cannot produce training predictions,
refit, bootstrap, or perform inference. Models fitted with any argument offset,
including an all-zero vector, require an explicit offset for every new
prediction. Formula offsets are reevaluated from the named new-data columns and
added to that vector.

## Privacy and lifecycle

The bundle omits response values, fitted values, residuals, row identifiers,
training design rows, and training group rows. It does contain model parameters,
fitted random effects, formula/column names, fixed factor levels, and canonical
group labels. Those labels and effects may identify people, devices,
institutions, or locations.
Apply the same access, retention, backup, and sharing controls as for sensitive
derived research data.

Kamino does not upload bundles or print their contents. Deleting the file is the
only Kamino-side cleanup required; operating-system trash, snapshots, backups,
cloud version history, and SSD wear leveling may retain copies, so use the
storage provider's approved deletion process when secure erasure is required.
No training response or RNG checkpoint is hidden in the artifact.

SHA-256 checksums detect accidental or malicious modification after creation;
they do not authenticate who created the bundle. Accept bundles only from a
trusted source.

## Private bootstrap ledgers are separate

`fit.parametric_bootstrap(..., ledger_path=...)` creates a private directory,
not a `.kamino` prediction bundle. The ledger stores every simulated response in
a non-object float64 `.npy` file plus an atomically replaced canonical JSON
manifest. It validates response hashes, exact fit/request/RNG identity, installed
source identity, and the reviewed project-plan hash before resuming.

Because those response arrays may be sensitive, the ledger directory and files
are created with owner-only permissions where the platform supports them. Apply
the caller's approved access, encryption, retention, backup, and deletion policy.
The ledger is not self-contained: resuming requires the same live fit or an
exactly reconstructed fit with the same input response and model state.

Completed, warning, singular, optimizer-failure, numerical-failure, and
statistic-failure outcomes remain visible. Failed replicates are never silently
redrawn. A singular fit is a valid completed outcome. This private ledger does
not change the prediction-only capabilities or privacy claims of `.kamino`
bundles.
