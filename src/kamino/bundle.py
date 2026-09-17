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
from kamino.formula import FixedEncoder, FixedRank, FixedVariable, RandomTermDesign
from kamino.model import FloatArray, ObjectiveKind
from kamino.results import (
    LinearMixedModelResult,
    OptimizerDiagnostics,
    PredictionOnlyModel,
)

BUNDLE_FORMAT = "kamino-prediction-bundle"
BUNDLE_SCHEMA_VERSION = "1.3.0"
_SUPPORTED_BUNDLE_SCHEMA_VERSIONS = {
    "1.0.0",
    "1.1.0",
    "1.2.0",
    BUNDLE_SCHEMA_VERSION,
}
REFERENCE_PROFILE = "lme4-2.0.6-unstructured-gaussian-v1"
# Canonical LF digest; updated whenever the reviewed project plan changes.
PROJECT_PLAN_SHA256 = "483967de16800f63aabfbf991f04580ea6d8218929e8c9651fbb3b6a8e1789c6"

_MANIFEST_PATH = "manifest.json"
_LEGACY_ARRAY_NAMES = (
    "theta",
    "beta",
    "beta_covariance",
    "random_covariance",
    "random_effects",
)
_ARRAY_NAMES = (*_LEGACY_ARRAY_NAMES, "fixed_null_basis")
_ARRAY_PATHS = {name: f"arrays/{name}.npy" for name in _ARRAY_NAMES}
_FIXED_RANK_TOLERANCE = 1e-7
_ESTIMABILITY_TOLERANCE = 1e-8


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


def _encoder_metadata(value: FixedEncoder) -> dict[str, object]:
    return {
        "encoding": "owned-fixed-v1",
        "variables": [
            {
                "name": variable.name,
                "kind": variable.kind,
                "levels": list(variable.levels),
                "contrast": variable.contrast,
            }
            for variable in value.variables
        ],
        "terms": [list(term) for term in value.terms],
    }


def _rank_metadata(value: FixedRank) -> dict[str, object]:
    return {
        "full_names": list(value.full_names),
        "retained_indices": list(value.retained_indices),
        "dropped_indices": list(value.dropped_indices),
        "pivot": list(value.pivot),
        "tolerance": value.tolerance,
        "estimability_tolerance": value.estimability_tolerance,
    }


def _random_term_metadata(value: RandomTermDesign) -> dict[str, object]:
    return {
        "group_name": value.group_name,
        "source_names": list(value.source_names),
        "group_levels": list(value.group_levels),
        "random_coefficient_names": list(value.random_coefficient_names),
        "predictor_name": value.predictor_name,
        "random_encoding": _encoder_metadata(value.random_encoder),
    }


def _model_metadata(value: LinearMixedModelResult) -> dict[str, object]:
    fixed_design = _encoder_metadata(value.fixed_encoder)
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
            **fixed_design,
            "formula_offsets": list(value.formula_offset_names),
            "transforms": [],
        },
        "fixed_rank": _rank_metadata(value.fixed_rank),
        "random_encoding": _encoder_metadata(value.random_encoder),
        "random_terms": [_random_term_metadata(term) for term in value.random_terms],
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
            "version": version("kamino-lme"),
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
        "fixed_null_basis": model.fixed_rank.null_basis,
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
        tolerance=_FIXED_RANK_TOLERANCE,
        estimability_tolerance=_ESTIMABILITY_TOLERANCE,
        null_basis=null_basis,
    )


def _validate_encoder(
    value: Any, expected_names: tuple[str, ...], name: str
) -> FixedEncoder:
    encoding = _mapping(value, name)
    _exact_keys(encoding, {"encoding", "variables", "terms"}, name)
    if encoding["encoding"] != "owned-fixed-v1":
        raise BundleError("bundle contains unsupported design state")
    raw_variables = encoding["variables"]
    if not isinstance(raw_variables, list):
        raise BundleError(f"{name}.variables must be a JSON array")
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
    raw_terms = encoding["terms"]
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
    if encoder.fixed_names != expected_names:
        raise BundleError("encoder and coefficient labels are inconsistent")
    return encoder


def _validate_fixed_encoder(
    value: Any, full_fixed_names: tuple[str, ...]
) -> tuple[FixedEncoder, tuple[str, ...]]:
    design = _mapping(value, "model.design")
    _exact_keys(
        design,
        {"encoding", "variables", "terms", "formula_offsets", "transforms"},
        "model.design",
    )
    if design["transforms"] != []:
        raise BundleError("bundle contains unsupported design state")
    encoder = _validate_encoder(
        {
            "encoding": design["encoding"],
            "variables": design["variables"],
            "terms": design["terms"],
        },
        full_fixed_names,
        "model.design",
    )
    offsets = _string_tuple(
        design["formula_offsets"], "formula offsets", allow_empty=True
    )
    if any(re.fullmatch(r"[A-Za-z_]\w*", name) is None for name in offsets):
        raise BundleError("formula offset names must be identifiers")
    return encoder, offsets


def _integer_tuple(value: Any, name: str, *, allow_empty: bool) -> tuple[int, ...]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise BundleError(f"{name} must be a JSON integer array")
    return tuple(_integer(item, name) for item in value)


def _validate_fixed_rank(
    value: Any,
    fixed_names: tuple[str, ...],
    null_basis: FloatArray,
) -> FixedRank:
    metadata = _mapping(value, "model.fixed_rank")
    _exact_keys(
        metadata,
        {
            "full_names",
            "retained_indices",
            "dropped_indices",
            "pivot",
            "tolerance",
            "estimability_tolerance",
        },
        "model.fixed_rank",
    )
    full_names = _string_tuple(metadata["full_names"], "fixed rank full_names")
    retained = _integer_tuple(
        metadata["retained_indices"], "fixed rank retained_indices", allow_empty=False
    )
    dropped = _integer_tuple(
        metadata["dropped_indices"], "fixed rank dropped_indices", allow_empty=True
    )
    pivot = _integer_tuple(metadata["pivot"], "fixed rank pivot", allow_empty=False)
    size = len(full_names)
    if (
        len(retained) != len(fixed_names)
        or tuple(full_names[index] for index in retained if index < size) != fixed_names
        or sorted((*retained, *dropped)) != list(range(size))
        or pivot != (*retained, *dropped)
        or len(set(pivot)) != size
    ):
        raise BundleError("fixed rank map is inconsistent")
    if null_basis.shape != (size, len(dropped)):
        raise BundleError("fixed null-space basis has an inconsistent model shape")
    if dropped:
        norms = np.linalg.norm(null_basis, axis=0)
        if not np.allclose(norms, 1.0, rtol=1e-10, atol=1e-10):
            raise BundleError("fixed null-space basis must have normalized columns")
        dropped_block = null_basis[np.asarray(dropped, dtype=np.int64), :]
        if not np.allclose(
            dropped_block,
            np.diag(np.diag(dropped_block)),
            rtol=0.0,
            atol=1e-14,
        ) or np.any(np.diag(dropped_block) <= 0.0):
            raise BundleError("fixed null-space basis does not preserve drop identity")
    tolerance = _number(metadata["tolerance"], "fixed rank tolerance", positive=True)
    estimability_tolerance = _number(
        metadata["estimability_tolerance"],
        "fixed rank estimability_tolerance",
        positive=True,
    )
    if (
        tolerance != _FIXED_RANK_TOLERANCE
        or estimability_tolerance != _ESTIMABILITY_TOLERANCE
    ):
        raise BundleError("fixed rank tolerances are unsupported")
    return FixedRank(
        full_names=full_names,
        retained_indices=retained,
        dropped_indices=dropped,
        pivot=pivot,
        tolerance=tolerance,
        estimability_tolerance=estimability_tolerance,
        null_basis=null_basis,
    )


def _validate_random_term(value: Any, index: int) -> RandomTermDesign:
    name = f"model.random_terms[{index}]"
    metadata = _mapping(value, name)
    _exact_keys(
        metadata,
        {
            "group_name",
            "source_names",
            "group_levels",
            "random_coefficient_names",
            "predictor_name",
            "random_encoding",
        },
        name,
    )
    group_name = _string(metadata["group_name"], f"{name}.group_name")
    source_names = _string_tuple(metadata["source_names"], f"{name}.source_names")
    if any(re.fullmatch(r"[A-Za-z_]\w*", source) is None for source in source_names):
        raise BundleError("random-term source names must be identifiers")
    if group_name != ":".join(source_names):
        raise BundleError("random-term group and source identity is inconsistent")
    group_levels = _string_tuple(metadata["group_levels"], f"{name}.group_levels")
    coefficients = _string_tuple(
        metadata["random_coefficient_names"], f"{name}.random_coefficient_names"
    )
    predictor_raw = metadata["predictor_name"]
    predictor = (
        None
        if predictor_raw is None
        else _string(predictor_raw, f"{name}.predictor_name")
    )
    encoder = _validate_encoder(
        metadata["random_encoding"], coefficients, f"{name}.random_encoding"
    )
    expected_terms = ((),) if predictor is None else ((), (predictor,))
    if (
        encoder.terms != expected_terms
        or (predictor is None and encoder.variables)
        or (
            predictor is not None
            and tuple(variable.name for variable in encoder.variables) != (predictor,)
        )
    ):
        raise BundleError("random-term predictor and encoder are inconsistent")
    return RandomTermDesign(
        group_name=group_name,
        source_names=source_names,
        group_levels=group_levels,
        training_groups=(),
        random_coefficient_names=coefficients,
        predictor_name=predictor,
        random_encoder=encoder,
    )


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
    FixedRank,
    FixedEncoder,
    tuple[str, ...],
    tuple[RandomTermDesign, ...],
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
    if schema_version == BUNDLE_SCHEMA_VERSION:
        expected_keys.update({"fixed_rank", "random_encoding", "random_terms"})
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
    if schema_version in ("1.2.0", BUNDLE_SCHEMA_VERSION):
        full_fixed_names = (
            _string_tuple(
                _mapping(model["fixed_rank"], "model.fixed_rank")["full_names"],
                "fixed rank full_names",
            )
            if schema_version == BUNDLE_SCHEMA_VERSION
            else fixed_names
        )
        fixed_encoder, formula_offsets = _validate_fixed_encoder(
            model["design"], full_fixed_names
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
    if schema_version == BUNDLE_SCHEMA_VERSION:
        fixed_rank = _validate_fixed_rank(
            model["fixed_rank"], fixed_names, arrays["fixed_null_basis"]
        )
        random_encoder = _validate_encoder(
            model["random_encoding"], coefficients, "model.random_encoding"
        )
        raw_random_terms = model["random_terms"]
        if not isinstance(raw_random_terms, list):
            raise BundleError("model.random_terms must be a JSON array")
        random_terms = tuple(
            _validate_random_term(term, index)
            for index, term in enumerate(raw_random_terms)
        )
    else:
        fixed_rank = _identity_fixed_rank(fixed_names)
        random_encoder = _legacy_fixed_encoder(predictor)
        random_terms = ()

    if not random_terms:
        expected_random_encoder = _legacy_fixed_encoder(predictor)
        if schema_version != BUNDLE_SCHEMA_VERSION:
            if coefficients != expected_random_encoder.fixed_names:
                raise BundleError("random-slope coefficient labels are inconsistent")
        elif (
            random_encoder.terms != expected_random_encoder.terms
            or (predictor is None and random_encoder.variables)
            or (
                predictor is not None
                and tuple(variable.name for variable in random_encoder.variables)
                != (predictor,)
            )
        ):
            raise BundleError("random-slope predictor encoding is inconsistent")

    if schema_version == "1.0.0":
        term_sizes = (len(coefficients),)
    else:
        raw_term_sizes = model["covariance_term_sizes"]
        if not isinstance(raw_term_sizes, list):
            raise BundleError("model.covariance_term_sizes must be a JSON array")
        term_sizes = tuple(
            _integer(item, "model.covariance_term_sizes", minimum=1)
            for item in raw_term_sizes
        )
        if not term_sizes:
            raise BundleError("covariance term sizes must not be empty")
    if random_terms:
        random_width = sum(len(term.random_coefficient_names) for term in random_terms)
        effect_count = sum(
            len(term.group_levels) * len(term.random_coefficient_names)
            for term in random_terms
        )
        expected_random_names = tuple(
            f"{term.group_name}[{level}]:{coefficient}"
            for term in random_terms
            for level in term.group_levels
            for coefficient in term.random_coefficient_names
        )
        if term_sizes != tuple(
            len(term.random_coefficient_names) for term in random_terms
        ):
            raise BundleError(
                "covariance term sizes do not preserve random-term boundaries"
            )
        first_term = random_terms[0]
        if (
            _string(model["group_name"], "model.group_name") != first_term.group_name
            or group_levels != first_term.group_levels
            or coefficients != first_term.random_coefficient_names
            or random_encoder != first_term.random_encoder
        ):
            raise BundleError("primary random-term metadata is inconsistent")
    else:
        random_width = len(coefficients)
        effect_count = len(group_levels) * random_width
        if sum(term_sizes) != random_width:
            raise BundleError(
                "covariance term sizes must sum to the random coefficient count"
            )
        group_name = _string(model["group_name"], "model.group_name")
        expected_random_names = tuple(
            f"{group_name}[{level}]:{coefficient}"
            for level in group_levels
            for coefficient in coefficients
        )
    theta_count = sum(size * (size + 1) // 2 for size in term_sizes)
    p = len(fixed_names)
    expected_shapes = {
        "theta": (theta_count,),
        "beta": (p,),
        "beta_covariance": (p, p),
        "random_covariance": (random_width, random_width),
        "random_effects": (effect_count,),
    }
    for name, expected in expected_shapes.items():
        if arrays[name].shape != expected:
            raise BundleError(f"array {name!r} has an inconsistent model shape")
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
    factor = np.zeros((random_width, random_width), dtype=np.float64)
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
    return (
        model,
        diagnostics,
        term_sizes,
        fixed_encoder,
        fixed_rank,
        random_encoder,
        formula_offsets,
        random_terms,
    )


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
            if _MANIFEST_PATH not in names:
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
            array_names = (
                _ARRAY_NAMES
                if schema_version == BUNDLE_SCHEMA_VERSION
                else _LEGACY_ARRAY_NAMES
            )
            expected_members = {
                _MANIFEST_PATH,
                *(_ARRAY_PATHS[name] for name in array_names),
            }
            if set(names) != expected_members:
                raise BundleError("bundle has unsupported or missing members")
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
            _exact_keys(array_metadata, set(array_names), "arrays")
            arrays: dict[str, FloatArray] = {}
            normalized_array_metadata: dict[str, dict[str, Any]] = {}
            for name in array_names:
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
                fixed_rank,
                random_encoder,
                formula_offsets,
                random_terms,
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
        fixed_rank=fixed_rank,
        random_encoder=random_encoder,
        formula_offset_names=formula_offsets,
        diagnostics=diagnostics,
        predictor_name=predictor_name,
        requires_explicit_offset=cast(bool, model["requires_explicit_offset"]),
        random_terms=random_terms,
    )


__all__ = [
    "BUNDLE_FORMAT",
    "BUNDLE_SCHEMA_VERSION",
    "BundleLimits",
    "load_model_bundle",
    "save_model_bundle",
]
