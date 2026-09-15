"""Safe, versioned prediction-only model bundles."""

# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnnecessaryIsInstance=false

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import tempfile
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path, PurePosixPath
from typing import Any, cast

import numpy as np

from kamino.errors import BundleError
from kamino.formula import FixedEncoder, FixedRank, FixedVariable
from kamino.model import FloatArray, ObjectiveKind
from kamino.results import (
    LinearMixedModelResult,
    OptimizerDiagnostics,
    PredictionOnlyModel,
)

BUNDLE_FORMAT = "kamino-prediction-bundle"
BUNDLE_SCHEMA_VERSION = "1.2.0"
_SUPPORTED_BUNDLE_SCHEMA_VERSIONS = {"1.0.0", "1.1.0", BUNDLE_SCHEMA_VERSION}
REFERENCE_PROFILE = "lme4-2.0.6-unstructured-gaussian-v1"
# Canonical LF digest; updated whenever the reviewed project plan changes.
PROJECT_PLAN_SHA256 = "b2856c3a877f174b6591ffdf192a6128185827929e395c6caf289d3e749b716b"

_MANIFEST_PATH = "manifest.json"
_ARRAY_NAMES = (
    "theta",
    "beta",
    "beta_covariance",
    "random_covariance",
    "random_effects",
)
_ARRAY_PATHS = {name: f"arrays/{name}.npy" for name in _ARRAY_NAMES}


@dataclass(frozen=True, slots=True)
class BundleLimits:
    """Hard limits applied before bundle members or arrays are allocated."""

    maximum_bundle_bytes: int = 64 * 1024 * 1024
    maximum_manifest_bytes: int = 1024 * 1024
    maximum_member_bytes: int = 32 * 1024 * 1024
    maximum_array_elements: int = 4_000_000

    def __post_init__(self) -> None:
        values = (
            self.maximum_bundle_bytes,
            self.maximum_manifest_bytes,
            self.maximum_member_bytes,
            self.maximum_array_elements,
        )
        if not all(type(value) is int and value > 0 for value in values):
            raise BundleError("bundle limits must be positive integers")
        if self.maximum_manifest_bytes > self.maximum_member_bytes:
            raise BundleError("manifest limit cannot exceed the member limit")
        if self.maximum_member_bytes > self.maximum_bundle_bytes:
            raise BundleError("member limit cannot exceed the bundle limit")


def _canonical_json(value: object) -> bytes:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise BundleError(f"bundle metadata is not canonical JSON: {error}") from error
    return encoded.encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _implementation_sha256() -> str:
    """Hash the exact installed Kamino Python sources that wrote the bundle."""
    package_directory = Path(__file__).resolve().parent
    sources = sorted(package_directory.glob("*.py"), key=lambda path: path.name)
    if not sources:
        raise BundleError("installed Kamino sources are unavailable for provenance")
    digest = hashlib.sha256()
    try:
        for source in sources:
            name = source.name.encode("utf-8")
            payload = source.read_bytes()
            digest.update(len(name).to_bytes(4, "big"))
            digest.update(name)
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
    except OSError as error:
        raise BundleError(
            f"could not hash installed Kamino sources: {error}"
        ) from error
    return digest.hexdigest()


def _array_bytes(value: FloatArray) -> tuple[FloatArray, bytes]:
    array = np.array(value, dtype="<f8", copy=True)
    if not np.isfinite(array).all():
        raise BundleError("model arrays must contain only finite float64 values")
    stream = io.BytesIO()
    np.save(stream, array, allow_pickle=False)
    return array, stream.getvalue()


def _diagnostics_metadata(value: OptimizerDiagnostics) -> dict[str, object]:
    return {
        "converged": value.converged,
        "message": value.message,
        "evaluations": value.evaluations,
        "boundary": value.boundary,
        "lower_bound": value.lower_bound,
        "search_upper_bound": value.search_upper_bound,
        "optimizer": value.optimizer,
        "parameter_count": value.parameter_count,
        "backend": value.backend,
        "control": {
            "initial_upper_bound": value.initial_upper_bound,
            "maximum_upper_bound": value.maximum_upper_bound,
            "absolute_theta_tolerance": value.absolute_theta_tolerance,
            "maximum_evaluations": value.maximum_evaluations,
            "boundary_tolerance": value.boundary_tolerance,
        },
    }


def _model_metadata(value: LinearMixedModelResult) -> dict[str, object]:
    return {
        "formula": value.formula,
        "kind": value.kind.value,
        "objective": value.objective,
        "log_likelihood": value.log_likelihood,
        "sigma2": value.sigma2,
        "random_variance": value.random_variance,
        "fixed_names": list(value.fixed_names),
        "random_names": list(value.random_names),
        "group_name": value.group_name,
        "group_levels": list(value.group_levels),
        "random_coefficient_names": list(value.random_coefficient_names),
        "covariance_term_sizes": list(value.covariance_term_sizes),
        "predictor_name": value.predictor_name,
        "requires_explicit_offset": value.requires_explicit_offset,
        "design": {
            "encoding": "owned-fixed-v1",
            "variables": [
                {
                    "name": variable.name,
                    "kind": variable.kind,
                    "levels": list(variable.levels),
                    "contrast": variable.contrast,
                }
                for variable in value.fixed_encoder.variables
            ],
            "terms": [list(term) for term in value.fixed_encoder.terms],
            "formula_offsets": list(value.formula_offset_names),
            "transforms": [],
        },
        "diagnostics": _diagnostics_metadata(value.diagnostics),
    }


def _array_metadata(name: str, array: FloatArray, payload: bytes) -> dict[str, object]:
    return {
        "path": _ARRAY_PATHS[name],
        "sha256": _sha256(payload),
        "dtype": "<f8",
        "shape": list(array.shape),
        "elements": int(array.size),
        "nbytes": int(array.nbytes),
    }


def _build_manifest(
    model: dict[str, object], arrays: dict[str, dict[str, object]]
) -> dict[str, object]:
    content = {"model": model, "arrays": arrays}
    return {
        "format": BUNDLE_FORMAT,
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "producer": {
            "package": "kamino",
            "version": version("kamino"),
            "reference_profile": REFERENCE_PROFILE,
            "implementation_sha256": _implementation_sha256(),
            "project_plan_sha256": PROJECT_PLAN_SHA256,
        },
        "capabilities": {
            "prediction": ["population", "conditional"],
            "training_rows": False,
            "refit": False,
            "inference": False,
        },
        **content,
        "integrity": {"model_sha256": _sha256(_canonical_json(content))},
    }


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o600 << 16
    return info


def _write_archive(path: Path, manifest: bytes, payloads: Mapping[str, bytes]) -> None:
    with zipfile.ZipFile(path, mode="w", allowZip64=False) as archive:
        archive.writestr(_zip_info(_MANIFEST_PATH), manifest)
        for name in _ARRAY_NAMES:
            archive.writestr(_zip_info(_ARRAY_PATHS[name]), payloads[name])
    # Windows requires a writable descriptor for fsync; the archive is already
    # closed, so update-binary mode does not alter its deterministic contents.
    with path.open("r+b") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save_model_bundle(
    model: LinearMixedModelResult,
    path: str | Path,
    *,
    overwrite: bool = False,
    limits: BundleLimits | None = None,
) -> Path:
    """Atomically save a fitted model without training observations or response."""
    if not isinstance(model, LinearMixedModelResult):
        raise BundleError("save_model_bundle requires a fitted Kamino result")
    if model.random_terms:
        raise BundleError(
            "prediction bundles for nested/crossed models require the Phase 2 "
            "artifact-recovery schema milestone"
        )
    if model.fixed_rank.dropped_indices:
        raise BundleError(
            "prediction bundles for rank-deficient models require the Phase 2 "
            "artifact-recovery schema milestone"
        )
    if model.random_encoder != _legacy_fixed_encoder(model.predictor_name):
        raise BundleError(
            "prediction bundles for categorical random terms require the Phase 2 "
            "artifact-recovery schema milestone"
        )
    if not isinstance(overwrite, bool):
        raise BundleError("overwrite must be a boolean")
    active_limits = limits or BundleLimits()
    if not isinstance(active_limits, BundleLimits):
        raise BundleError("limits must be a BundleLimits instance")
    target = Path(path)
    if not target.name:
        raise BundleError("bundle path must name a file")
    parent = target.parent
    if not parent.is_dir():
        raise BundleError("bundle parent directory does not exist")
    if target.is_symlink() or (target.exists() and not target.is_file()):
        raise BundleError("bundle target must be a regular file path")
    if target.exists() and not overwrite:
        raise BundleError("bundle target already exists; set overwrite=True")

    source_arrays = {
        "theta": model.theta,
        "beta": model.beta,
        "beta_covariance": model.beta_covariance,
        "random_covariance": model.random_covariance,
        "random_effects": model.random_effects,
    }
    arrays: dict[str, FloatArray] = {}
    payloads: dict[str, bytes] = {}
    metadata: dict[str, dict[str, object]] = {}
    for name in _ARRAY_NAMES:
        array, payload = _array_bytes(source_arrays[name])
        if array.size > active_limits.maximum_array_elements:
            raise BundleError(f"array {name!r} exceeds the configured element limit")
        if len(payload) > active_limits.maximum_member_bytes:
            raise BundleError(f"array {name!r} exceeds the configured member limit")
        arrays[name] = array
        payloads[name] = payload
        metadata[name] = _array_metadata(name, array, payload)
    model_metadata = _model_metadata(model)
    _validate_model(model_metadata, arrays)
    manifest = _canonical_json(_build_manifest(model_metadata, metadata)) + b"\n"
    if len(manifest) > active_limits.maximum_manifest_bytes:
        raise BundleError("bundle manifest exceeds the configured size limit")

    temporary: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=parent
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        _write_archive(temporary, manifest, payloads)
        if temporary.stat().st_size > active_limits.maximum_bundle_bytes:
            raise BundleError("bundle exceeds the configured file-size limit")
        if overwrite:
            os.replace(temporary, target)
        else:
            os.link(temporary, target)
            temporary.unlink()
        temporary = None
        _fsync_directory(parent)
    except BundleError:
        raise
    except OSError as error:
        raise BundleError(f"bundle publication failed: {error}") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target


def _duplicate_rejecting_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BundleError(f"bundle JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise BundleError(f"bundle JSON contains invalid constant {value}")


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise BundleError(f"{name} must be a JSON object")
    return cast(dict[str, Any], value)


def _exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    actual = set(value)
    if actual != expected:
        raise BundleError(f"{name} has unsupported or missing fields")


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise BundleError(f"{name} must be a nonempty string")
    return value


def _string_tuple(
    value: Any, name: str, *, allow_empty: bool = False
) -> tuple[str, ...]:
    if not isinstance(value, list) or (not value and not allow_empty):
        qualifier = "string list" if allow_empty else "nonempty string list"
        raise BundleError(f"{name} must be a {qualifier}")
    result = tuple(_string(item, name) for item in value)
    if len(set(result)) != len(result):
        raise BundleError(f"{name} must contain unique labels")
    return result


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise BundleError(f"{name} must be a boolean")
    return value


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise BundleError(f"{name} must be an integer of at least {minimum}")
    return value


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BundleError(f"{name} must be numeric")
    result = float(value)
    if not np.isfinite(result) or (positive and result <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise BundleError(f"{name} must be {qualifier}")
    return result


def _optional_number(value: Any, name: str) -> float | None:
    return None if value is None else _number(value, name)


def _validate_diagnostics(value: Any) -> OptimizerDiagnostics:
    metadata = _mapping(value, "model.diagnostics")
    _exact_keys(
        metadata,
        {
            "converged",
            "message",
            "evaluations",
            "boundary",
            "lower_bound",
            "search_upper_bound",
            "optimizer",
            "parameter_count",
            "backend",
            "control",
        },
        "model.diagnostics",
    )
    control = _mapping(metadata["control"], "model.diagnostics.control")
    _exact_keys(
        control,
        {
            "initial_upper_bound",
            "maximum_upper_bound",
            "absolute_theta_tolerance",
            "maximum_evaluations",
            "boundary_tolerance",
        },
        "model.diagnostics.control",
    )
    converged = _boolean(metadata["converged"], "diagnostics.converged")
    if not converged:
        raise BundleError("prediction bundles require a converged fitted model")
    return OptimizerDiagnostics(
        converged=converged,
        message=_string(metadata["message"], "diagnostics.message"),
        evaluations=_integer(
            metadata["evaluations"], "diagnostics.evaluations", minimum=1
        ),
        boundary=_boolean(metadata["boundary"], "diagnostics.boundary"),
        lower_bound=_number(metadata["lower_bound"], "diagnostics.lower_bound"),
        search_upper_bound=_optional_number(
            metadata["search_upper_bound"], "diagnostics.search_upper_bound"
        ),
        optimizer=_string(metadata["optimizer"], "diagnostics.optimizer"),
        parameter_count=_integer(
            metadata["parameter_count"], "diagnostics.parameter_count", minimum=1
        ),
        backend=_string(metadata["backend"], "diagnostics.backend"),
        initial_upper_bound=_number(
            control["initial_upper_bound"], "control.initial_upper_bound", positive=True
        ),
        maximum_upper_bound=_number(
            control["maximum_upper_bound"], "control.maximum_upper_bound", positive=True
        ),
        absolute_theta_tolerance=_number(
            control["absolute_theta_tolerance"],
            "control.absolute_theta_tolerance",
            positive=True,
        ),
        maximum_evaluations=_integer(
            control["maximum_evaluations"], "control.maximum_evaluations", minimum=1
        ),
        boundary_tolerance=_number(
            control["boundary_tolerance"], "control.boundary_tolerance", positive=True
        ),
    )


def _legacy_fixed_encoder(predictor: str | None) -> FixedEncoder:
    return FixedEncoder(
        variables=(
            ()
            if predictor is None
            else (FixedVariable(name=predictor, kind="numeric"),)
        ),
        terms=((),) if predictor is None else ((), (predictor,)),
    )


def _identity_fixed_rank(fixed_names: tuple[str, ...]) -> FixedRank:
    null_basis = np.empty((len(fixed_names), 0), dtype=np.float64)
    null_basis.setflags(write=False)
    indices = tuple(range(len(fixed_names)))
    return FixedRank(
        full_names=fixed_names,
        retained_indices=indices,
        dropped_indices=(),
        pivot=indices,
        tolerance=1e-7,
        estimability_tolerance=1e-8,
        null_basis=null_basis,
    )


def _validate_fixed_encoder(
    value: Any, fixed_names: tuple[str, ...]
) -> tuple[FixedEncoder, tuple[str, ...]]:
    design = _mapping(value, "model.design")
    _exact_keys(
        design,
        {"encoding", "variables", "terms", "formula_offsets", "transforms"},
        "model.design",
    )
    if design["encoding"] != "owned-fixed-v1" or design["transforms"] != []:
        raise BundleError("bundle contains unsupported design state")
    raw_variables = design["variables"]
    if not isinstance(raw_variables, list):
        raise BundleError("model.design.variables must be a JSON array")
    variables: list[FixedVariable] = []
    names: set[str] = set()
    for index, raw_variable in enumerate(raw_variables):
        metadata = _mapping(raw_variable, f"model.design.variables[{index}]")
        _exact_keys(metadata, {"name", "kind", "levels", "contrast"}, "variable")
        name = _string(metadata["name"], "fixed variable name")
        if re.fullmatch(r"[A-Za-z_]\w*", name) is None or name in names:
            raise BundleError("fixed variable names must be unique identifiers")
        names.add(name)
        kind = metadata["kind"]
        raw_levels = metadata["levels"]
        if not isinstance(raw_levels, list):
            raise BundleError("fixed variable levels must be a JSON array")
        levels = _string_tuple(raw_levels, "fixed variable levels", allow_empty=True)
        contrast = metadata["contrast"]
        if kind == "numeric":
            if levels or contrast is not None:
                raise BundleError("numeric fixed variable metadata is inconsistent")
            variables.append(FixedVariable(name=name, kind="numeric"))
        elif kind == "categorical":
            if len(levels) < 2 or contrast not in ("treatment", "sum"):
                raise BundleError("categorical fixed variable metadata is inconsistent")
            variables.append(
                FixedVariable(
                    name=name,
                    kind="categorical",
                    levels=levels,
                    contrast=cast(Any, contrast),
                )
            )
        else:
            raise BundleError("fixed variable kind is unsupported")
    raw_terms = design["terms"]
    if not isinstance(raw_terms, list):
        raise BundleError("model.design.terms must be a JSON array")
    terms: list[tuple[str, ...]] = []
    for raw_term in raw_terms:
        if not isinstance(raw_term, list):
            raise BundleError("fixed terms must be JSON arrays")
        term = tuple(_string(item, "fixed term variable") for item in raw_term)
        if len(term) > 2 or len(set(term)) != len(term) or not set(term) <= names:
            raise BundleError("fixed term metadata is inconsistent")
        terms.append(term)
    if not terms or terms[0] != () or len(set(terms)) != len(terms):
        raise BundleError("fixed terms must contain one unique leading intercept")
    if {name for term in terms for name in term} != names:
        raise BundleError("fixed variable and term metadata are inconsistent")
    encoder = FixedEncoder(tuple(variables), tuple(terms))
    if encoder.fixed_names != fixed_names:
        raise BundleError("fixed encoder and coefficient labels are inconsistent")
    offsets = _string_tuple(
        design["formula_offsets"], "formula offsets", allow_empty=True
    )
    if any(re.fullmatch(r"[A-Za-z_]\w*", name) is None for name in offsets):
        raise BundleError("formula offset names must be identifiers")
    return encoder, offsets


def _validate_model(
    value: Any,
    arrays: Mapping[str, FloatArray],
    *,
    schema_version: str = BUNDLE_SCHEMA_VERSION,
) -> tuple[
    dict[str, Any],
    OptimizerDiagnostics,
    tuple[int, ...],
    FixedEncoder,
    tuple[str, ...],
]:
    model = _mapping(value, "model")
    expected_keys = {
        "formula",
        "kind",
        "objective",
        "log_likelihood",
        "sigma2",
        "random_variance",
        "fixed_names",
        "random_names",
        "group_name",
        "group_levels",
        "random_coefficient_names",
        "predictor_name",
        "requires_explicit_offset",
        "design",
        "diagnostics",
    }
    if schema_version != "1.0.0":
        expected_keys.add("covariance_term_sizes")
    _exact_keys(
        model,
        expected_keys,
        "model",
    )
    fixed_names = _string_tuple(model["fixed_names"], "model.fixed_names")
    random_names = _string_tuple(model["random_names"], "model.random_names")
    group_levels = _string_tuple(model["group_levels"], "model.group_levels")
    coefficients = _string_tuple(
        model["random_coefficient_names"], "model.random_coefficient_names"
    )
    predictor_raw = model["predictor_name"]
    predictor = (
        None if predictor_raw is None else _string(predictor_raw, "predictor_name")
    )
    if schema_version == BUNDLE_SCHEMA_VERSION:
        fixed_encoder, formula_offsets = _validate_fixed_encoder(
            model["design"], fixed_names
        )
    else:
        design = _mapping(model["design"], "model.design")
        _exact_keys(design, {"encoding", "contrasts", "transforms"}, "model.design")
        expected_encoding = (
            "intercept" if predictor is None else "intercept-plus-numeric"
        )
        if (
            design["encoding"] != expected_encoding
            or design["contrasts"] != []
            or design["transforms"] != []
        ):
            raise BundleError("bundle contains unsupported design state")
        fixed_encoder = _legacy_fixed_encoder(predictor)
        formula_offsets = ()
        if fixed_names != coefficients:
            raise BundleError("fixed and random coefficient labels are inconsistent")
    if predictor is None:
        if coefficients != ("(Intercept)",):
            raise BundleError("intercept bundle has inconsistent coefficient labels")
    elif coefficients != ("(Intercept)", predictor):
        raise BundleError("slope bundle has inconsistent coefficient labels")
    if predictor is not None:
        encoded_predictor = next(
            (
                variable
                for variable in fixed_encoder.variables
                if variable.name == predictor
            ),
            None,
        )
        if encoded_predictor is None or encoded_predictor.kind != "numeric":
            raise BundleError("random-slope predictor encoding is inconsistent")
    k = len(coefficients)
    groups = len(group_levels)
    if schema_version == "1.0.0":
        term_sizes = (k,)
    else:
        raw_term_sizes = model["covariance_term_sizes"]
        if not isinstance(raw_term_sizes, list):
            raise BundleError("model.covariance_term_sizes must be a JSON array")
        term_sizes = tuple(
            _integer(item, "model.covariance_term_sizes", minimum=1)
            for item in raw_term_sizes
        )
        if not term_sizes or sum(term_sizes) != k:
            raise BundleError(
                "covariance term sizes must sum to the random coefficient count"
            )
    theta_count = sum(size * (size + 1) // 2 for size in term_sizes)
    p = len(fixed_names)
    expected_shapes = {
        "theta": (theta_count,),
        "beta": (p,),
        "beta_covariance": (p, p),
        "random_covariance": (k, k),
        "random_effects": (groups * k,),
    }
    for name, expected in expected_shapes.items():
        if arrays[name].shape != expected:
            raise BundleError(f"array {name!r} has an inconsistent model shape")
    if len(random_names) != groups * k:
        raise BundleError("random_names has an inconsistent length")
    group_name = _string(model["group_name"], "model.group_name")
    expected_random_names = tuple(
        f"{group_name}[{level}]:{coefficient}"
        for level in group_levels
        for coefficient in coefficients
    )
    if random_names != expected_random_names:
        raise BundleError("random_names does not preserve the group/coefficient map")

    objective = _number(model["objective"], "model.objective")
    log_likelihood = _number(model["log_likelihood"], "model.log_likelihood")
    if not np.isclose(log_likelihood, -0.5 * objective, rtol=1e-14, atol=1e-14):
        raise BundleError("objective and log likelihood are inconsistent")
    sigma2 = _number(model["sigma2"], "model.sigma2", positive=True)
    random_variance = _number(model["random_variance"], "model.random_variance")
    if random_variance < 0.0 or random_variance != float(
        arrays["random_covariance"][0, 0]
    ):
        raise BundleError("random variance and covariance are inconsistent")
    for name in ("beta_covariance", "random_covariance"):
        covariance = arrays[name]
        if not np.allclose(covariance, covariance.T, rtol=1e-12, atol=1e-12):
            raise BundleError(f"array {name!r} must be symmetric")
        if float(np.linalg.eigvalsh(covariance).min()) < -1e-10:
            raise BundleError(f"array {name!r} must be positive semidefinite")
    theta = arrays["theta"]
    factor = np.zeros((k, k), dtype=np.float64)
    cursor = 0
    block_start = 0
    for size in term_sizes:
        for column in range(size):
            width = size - column
            factor[
                block_start + column : block_start + size,
                block_start + column,
            ] = theta[cursor : cursor + width]
            cursor += width
        block_start += size
    if (factor.diagonal() < 0.0).any() or not np.allclose(
        arrays["random_covariance"], sigma2 * factor @ factor.T, rtol=1e-12, atol=1e-12
    ):
        raise BundleError("theta and random covariance are inconsistent")
    diagnostics = _validate_diagnostics(model["diagnostics"])
    if diagnostics.parameter_count != theta_count:
        raise BundleError("optimizer parameter count is inconsistent with theta")
    if diagnostics.maximum_upper_bound <= diagnostics.initial_upper_bound:
        raise BundleError("optimizer bounds are inconsistent")
    _string(model["formula"], "model.formula")
    try:
        ObjectiveKind(_string(model["kind"], "model.kind"))
    except ValueError as error:
        raise BundleError("model.kind is unsupported") from error
    _boolean(model["requires_explicit_offset"], "model.requires_explicit_offset")
    return model, diagnostics, term_sizes, fixed_encoder, formula_offsets


def _safe_member_name(name: str) -> bool:
    path = PurePosixPath(name)
    return (
        bool(name)
        and not path.is_absolute()
        and "\\" not in name
        and all(part not in ("", ".", "..") for part in path.parts)
    )


def _read_manifest(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> dict[str, Any]:
    try:
        raw = archive.read(info)
        parsed = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_duplicate_rejecting_object,
            parse_constant=_invalid_constant,
        )
    except BundleError:
        raise
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        RuntimeError,
        zipfile.BadZipFile,
    ) as error:
        raise BundleError(f"bundle manifest is invalid: {error}") from error
    return _mapping(parsed, "manifest")


def _load_array(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    metadata: Mapping[str, Any],
    limits: BundleLimits,
) -> FloatArray:
    shape_raw = metadata["shape"]
    if not isinstance(shape_raw, list) or not all(
        type(item) is int and item >= 0 for item in shape_raw
    ):
        raise BundleError(f"array metadata for {info.filename!r} has an invalid shape")
    shape = tuple(shape_raw)
    elements = _integer(metadata["elements"], "array.elements")
    nbytes = _integer(metadata["nbytes"], "array.nbytes")
    if elements > limits.maximum_array_elements:
        raise BundleError(f"array member {info.filename!r} exceeds the element limit")
    if math.prod(shape) != elements or nbytes != elements * 8:
        raise BundleError(f"array metadata for {info.filename!r} is inconsistent")
    payload = archive.read(info)
    if _sha256(payload) != metadata["sha256"]:
        raise BundleError(f"checksum mismatch for {info.filename!r}")
    try:
        loaded = np.load(io.BytesIO(payload), allow_pickle=False)
    except (ValueError, OSError, EOFError) as error:
        raise BundleError(
            f"array member {info.filename!r} is invalid: {error}"
        ) from error
    if not isinstance(loaded, np.ndarray) or loaded.dtype != np.dtype("<f8"):
        raise BundleError(f"array member {info.filename!r} must be float64")
    if loaded.shape != shape or loaded.size != elements or loaded.nbytes != nbytes:
        raise BundleError(f"array metadata for {info.filename!r} is inconsistent")
    if not np.isfinite(loaded).all():
        raise BundleError(f"array member {info.filename!r} contains non-finite values")
    result = np.array(loaded, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def load_model_bundle(
    path: str | Path, *, limits: BundleLimits | None = None
) -> PredictionOnlyModel:
    """Validate and load a non-executable prediction-only model bundle."""
    active_limits = limits or BundleLimits()
    if not isinstance(active_limits, BundleLimits):
        raise BundleError("limits must be a BundleLimits instance")
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise BundleError("bundle source must be a regular file")
    try:
        if source.stat().st_size > active_limits.maximum_bundle_bytes:
            raise BundleError("bundle exceeds the configured file-size limit")
        with zipfile.ZipFile(source, mode="r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise BundleError("bundle contains duplicate member names")
            if not all(_safe_member_name(name) for name in names):
                raise BundleError("bundle contains an unsafe member path")
            expected_members = {_MANIFEST_PATH, *_ARRAY_PATHS.values()}
            if set(names) != expected_members:
                raise BundleError("bundle has unsupported or missing members")
            by_name = {info.filename: info for info in infos}
            total_size = 0
            for info in infos:
                if info.flag_bits & 0x1:
                    raise BundleError("encrypted bundle members are unsupported")
                if info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                    raise BundleError("bundle uses an unsupported compression method")
                if info.file_size > active_limits.maximum_member_bytes:
                    raise BundleError(
                        f"bundle member {info.filename!r} exceeds its size limit"
                    )
                total_size += info.file_size
            if total_size > active_limits.maximum_bundle_bytes:
                raise BundleError("uncompressed bundle members exceed the size limit")
            manifest_info = by_name[_MANIFEST_PATH]
            if manifest_info.file_size > active_limits.maximum_manifest_bytes:
                raise BundleError("bundle manifest exceeds the configured size limit")
            manifest = _read_manifest(archive, manifest_info)
            _exact_keys(
                manifest,
                {
                    "format",
                    "schema_version",
                    "producer",
                    "capabilities",
                    "model",
                    "arrays",
                    "integrity",
                },
                "manifest",
            )
            if manifest["format"] != BUNDLE_FORMAT:
                raise BundleError("bundle format is unsupported")
            schema_version = _string(manifest["schema_version"], "schema_version")
            if schema_version not in _SUPPORTED_BUNDLE_SCHEMA_VERSIONS:
                raise BundleError("bundle schema version is unsupported")
            producer = _mapping(manifest["producer"], "producer")
            _exact_keys(
                producer,
                {
                    "package",
                    "version",
                    "reference_profile",
                    "implementation_sha256",
                    "project_plan_sha256",
                },
                "producer",
            )
            if (
                producer["package"] != "kamino"
                or producer["reference_profile"] != REFERENCE_PROFILE
            ):
                raise BundleError("bundle producer profile is unsupported")
            _string(producer["version"], "producer.version")
            for key in ("implementation_sha256", "project_plan_sha256"):
                recorded_hash = _string(producer[key], f"producer.{key}")
                if len(recorded_hash) != 64 or any(
                    character not in "0123456789abcdef" for character in recorded_hash
                ):
                    raise BundleError(f"producer {key} is invalid")
            capabilities = _mapping(manifest["capabilities"], "capabilities")
            if capabilities != {
                "prediction": ["population", "conditional"],
                "training_rows": False,
                "refit": False,
                "inference": False,
            }:
                raise BundleError("bundle capabilities are unsupported")
            array_metadata = _mapping(manifest["arrays"], "arrays")
            _exact_keys(array_metadata, set(_ARRAY_NAMES), "arrays")
            arrays: dict[str, FloatArray] = {}
            normalized_array_metadata: dict[str, dict[str, Any]] = {}
            for name in _ARRAY_NAMES:
                metadata = _mapping(array_metadata[name], f"arrays.{name}")
                _exact_keys(
                    metadata,
                    {"path", "sha256", "dtype", "shape", "elements", "nbytes"},
                    f"arrays.{name}",
                )
                if metadata["path"] != _ARRAY_PATHS[name] or metadata["dtype"] != "<f8":
                    raise BundleError(f"array metadata for {name!r} is unsupported")
                checksum = _string(metadata["sha256"], f"arrays.{name}.sha256")
                if len(checksum) != 64 or any(
                    character not in "0123456789abcdef" for character in checksum
                ):
                    raise BundleError(f"array checksum for {name!r} is invalid")
                normalized_array_metadata[name] = metadata
                arrays[name] = _load_array(
                    archive, by_name[_ARRAY_PATHS[name]], metadata, active_limits
                )
            integrity = _mapping(manifest["integrity"], "integrity")
            _exact_keys(integrity, {"model_sha256"}, "integrity")
            content = {"model": manifest["model"], "arrays": normalized_array_metadata}
            if integrity["model_sha256"] != _sha256(_canonical_json(content)):
                raise BundleError("bundle model checksum mismatch")
            (
                model,
                diagnostics,
                term_sizes,
                fixed_encoder,
                formula_offsets,
            ) = _validate_model(
                manifest["model"], arrays, schema_version=schema_version
            )
    except BundleError:
        raise
    except (OSError, KeyError, TypeError, ValueError, zipfile.BadZipFile) as error:
        raise BundleError(f"bundle is invalid or unreadable: {error}") from error

    predictor_raw = model["predictor_name"]
    predictor_name = None if predictor_raw is None else cast(str, predictor_raw)
    return PredictionOnlyModel(
        formula=cast(str, model["formula"]),
        kind=ObjectiveKind(cast(str, model["kind"])),
        objective=float(model["objective"]),
        log_likelihood=float(model["log_likelihood"]),
        theta=arrays["theta"],
        beta=arrays["beta"],
        beta_covariance=arrays["beta_covariance"],
        sigma2=float(model["sigma2"]),
        random_variance=float(model["random_variance"]),
        random_covariance=arrays["random_covariance"],
        random_effects=arrays["random_effects"],
        fixed_names=tuple(cast(list[str], model["fixed_names"])),
        random_names=tuple(cast(list[str], model["random_names"])),
        group_name=cast(str, model["group_name"]),
        group_levels=tuple(cast(list[str], model["group_levels"])),
        random_coefficient_names=tuple(
            cast(list[str], model["random_coefficient_names"])
        ),
        covariance_term_sizes=term_sizes,
        fixed_encoder=fixed_encoder,
        fixed_rank=_identity_fixed_rank(tuple(cast(list[str], model["fixed_names"]))),
        random_encoder=_legacy_fixed_encoder(predictor_name),
        formula_offset_names=formula_offsets,
        diagnostics=diagnostics,
        predictor_name=predictor_name,
        requires_explicit_offset=cast(bool, model["requires_explicit_offset"]),
    )


__all__ = [
    "BUNDLE_FORMAT",
    "BUNDLE_SCHEMA_VERSION",
    "BundleLimits",
    "load_model_bundle",
    "save_model_bundle",
]
