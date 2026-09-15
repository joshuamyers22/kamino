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
grouping factor remain independent after reload. The loader continues to accept
schema `1.0.0`, whose missing term metadata unambiguously denotes its original
single correlated covariance term. Saving always writes the current schema.

The loaded `PredictionOnlyModel` requires explicit new data. It exposes only
population and conditional prediction. It cannot produce training predictions,
refit, bootstrap, or perform inference. Models fitted with a nonzero argument
offset continue to require an explicit offset for every new prediction.

## Privacy and lifecycle

The bundle omits response values, fitted values, residuals, row identifiers,
training design rows, and training group rows. It does contain model parameters,
fitted random effects, formula/column names, and canonical group labels. Those
labels and effects may identify people, devices, institutions, or locations.
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
