"""Deterministic Gaussian simulation and resumable parametric bootstrap."""

# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportMissingTypeStubs=false, reportPrivateUsage=false
# pyright: reportUnnecessaryIsInstance=false

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import warnings
from collections.abc import Generator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from threading import Event
from typing import Any, Literal, TypeAlias, cast

import numpy as np
from scipy.stats import beta as beta_distribution
from threadpoolctl import threadpool_limits

from kamino.bundle import PROJECT_PLAN_SHA256, REFERENCE_PROFILE
from kamino.errors import (
    BootstrapError,
    ConvergenceError,
    ModelSpecificationError,
    NumericalError,
    ResourceLimitError,
)
from kamino.formula import SingleGroupDesign
from kamino.model import FloatArray
from kamino.results import LinearMixedModelResult

SimulationMode: TypeAlias = Literal["unconditional", "conditional"]
BootstrapStatus: TypeAlias = Literal[
    "success",
    "warning",
    "singular",
    "optimizer_failure",
    "numerical_failure",
    "statistic_failure",
]
IntervalMethod: TypeAlias = Literal["percentile", "basic"]

_LEDGER_FORMAT = "kamino-bootstrap-ledger"
_LEDGER_SCHEMA_VERSION = "1.0.0"
_BIT_GENERATOR = "PCG64DXSM"
_RANDOM_EFFECT_PURPOSE = 0x4B414D01
_RESIDUAL_PURPOSE = 0x4B414D02
_SUCCESS_STATUSES = frozenset({"success", "warning", "singular"})


def _readonly(value: Sequence[float] | FloatArray) -> FloatArray:
    result = np.array(value, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class InferenceLimits:
    """Preflight bounds for simulation, workers, and private ledger payloads."""

    maximum_replicates: int = 100_000
    maximum_workers: int = 32
    maximum_response_values: int = 100_000_000
    maximum_manifest_bytes: int = 64 * 1024 * 1024
    maximum_response_bytes: int = 512 * 1024 * 1024

    def __post_init__(self) -> None:
        values = (
            self.maximum_replicates,
            self.maximum_workers,
            self.maximum_response_values,
            self.maximum_manifest_bytes,
            self.maximum_response_bytes,
        )
        if not all(type(value) is int and value > 0 for value in values):
            raise ModelSpecificationError("inference limits must be positive integers")


@dataclass(frozen=True, slots=True)
class SimulationDraw:
    """One reproducible response draw with a content identity."""

    replicate_id: int
    response: FloatArray
    response_sha256: str


@dataclass(frozen=True, slots=True)
class SimulationBatch:
    """A deterministic set of Gaussian mixed-model response draws."""

    mode: SimulationMode
    root_seed: int
    bit_generator: str
    numpy_version: str
    draws: tuple[SimulationDraw, ...]


@dataclass(frozen=True, slots=True)
class BootstrapReplicate:
    """Auditable outcome for one requested bootstrap replicate."""

    replicate_id: int
    status: BootstrapStatus
    response_sha256: str
    statistics: FloatArray | None
    boundary: bool | None
    warnings: tuple[str, ...]
    message: str | None

    @property
    def usable(self) -> bool:
        """Whether this replicate produced a valid fitted statistic."""
        return self.status in _SUCCESS_STATUSES


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    """Percentile or basic interval with its finite-replicate resolution."""

    method: IntervalMethod
    level: float
    statistic_names: tuple[str, ...]
    lower: FloatArray
    upper: FloatArray
    completed_replicates: int
    failed_replicates: int
    quantile_convention: str
    tail_probability_standard_error: float
    incomplete: bool


@dataclass(frozen=True, slots=True)
class FailureAccounting:
    """Observed failures and a one-sided exact binomial upper bound."""

    requested_replicates: int
    completed_replicates: int
    failed_replicates: int
    observed_failure_rate: float
    confidence_level: float
    upper_failure_rate: float
    statuses: tuple[tuple[BootstrapStatus, int], ...]


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    """Fixed-effect parametric-bootstrap estimates and complete failure ledger."""

    mode: SimulationMode
    root_seed: int
    bit_generator: str
    numpy_version: str
    statistic_names: tuple[str, ...]
    observed: FloatArray
    estimates: FloatArray
    records: tuple[BootstrapReplicate, ...]
    ledger_path: Path | None
    allow_incomplete: bool

    @property
    def completed_replicates(self) -> int:
        return sum(record.usable for record in self.records)

    @property
    def failed_replicates(self) -> int:
        return len(self.records) - self.completed_replicates

    @property
    def complete(self) -> bool:
        return self.failed_replicates == 0

    @property
    def failure_rate(self) -> float:
        return self.failed_replicates / len(self.records)

    def failure_accounting(
        self, *, confidence_level: float = 0.95
    ) -> FailureAccounting:
        """Return exact one-sided binomial failure accounting for every draw."""
        if not np.isfinite(confidence_level) or not 0.0 < confidence_level < 1.0:
            raise BootstrapError(
                "failure confidence level must be strictly between zero and one"
            )
        failed = self.failed_replicates
        total = len(self.records)
        upper = (
            1.0
            if failed == total
            else float(
                beta_distribution.ppf(
                    confidence_level,
                    failed + 1,
                    total - failed,
                )
            )
        )
        counts: dict[BootstrapStatus, int] = {}
        for record in self.records:
            counts[record.status] = counts.get(record.status, 0) + 1
        return FailureAccounting(
            requested_replicates=total,
            completed_replicates=self.completed_replicates,
            failed_replicates=failed,
            observed_failure_rate=self.failure_rate,
            confidence_level=float(confidence_level),
            upper_failure_rate=upper,
            statuses=tuple(sorted(counts.items())),
        )

    def interval(
        self,
        *,
        level: float = 0.95,
        method: IntervalMethod = "percentile",
    ) -> BootstrapInterval:
        """Construct a named bootstrap interval from completed replicates."""
        if not np.isfinite(level) or not 0.0 < level < 1.0:
            raise BootstrapError("interval level must be strictly between zero and one")
        if method not in ("percentile", "basic"):
            raise BootstrapError("interval method must be 'percentile' or 'basic'")
        if self.failed_replicates and not self.allow_incomplete:
            raise BootstrapError(
                "bootstrap has unresolved failed replicates; rerun with "
                "allow_incomplete=True to inspect a partial interval"
            )
        usable = np.array([record.usable for record in self.records], dtype=np.bool_)
        values = self.estimates[usable]
        if values.shape[0] < 2:
            raise BootstrapError("at least two completed replicates are required")
        alpha = 1.0 - level
        quantiles = np.quantile(
            values,
            [alpha / 2.0, 1.0 - alpha / 2.0],
            axis=0,
            method="linear",
        )
        if method == "percentile":
            lower, upper = quantiles[0], quantiles[1]
        else:
            lower = 2.0 * self.observed - quantiles[1]
            upper = 2.0 * self.observed - quantiles[0]
        tail = alpha / 2.0
        return BootstrapInterval(
            method=method,
            level=float(level),
            statistic_names=self.statistic_names,
            lower=_readonly(lower),
            upper=_readonly(upper),
            completed_replicates=int(values.shape[0]),
            failed_replicates=self.failed_replicates,
            quantile_convention="numpy-linear",
            tail_probability_standard_error=float(
                np.sqrt(tail * (1.0 - tail) / values.shape[0])
            ),
            incomplete=self.failed_replicates != 0,
        )


def _validated_request(
    model: LinearMixedModelResult,
    replicates: int,
    seed: int,
    mode: SimulationMode,
    limits: InferenceLimits,
) -> None:
    if not isinstance(model, LinearMixedModelResult):
        raise ModelSpecificationError("simulation requires a fitted Kamino result")
    if type(replicates) is not int or replicates <= 0:
        raise ModelSpecificationError("replicates must be a positive integer")
    if replicates > limits.maximum_replicates:
        raise ResourceLimitError("replicate count exceeds the inference limit")
    if type(seed) is not int or seed < 0 or seed >= 2**128:
        raise ModelSpecificationError("seed must be an integer in [0, 2**128)")
    if mode not in ("unconditional", "conditional"):
        raise ModelSpecificationError(
            "simulation mode must be 'unconditional' or 'conditional'"
        )
    n = model._training_design.spec.n
    if replicates * n > limits.maximum_response_values:
        raise ResourceLimitError("simulated response size exceeds the inference limit")


def _rng(seed: int, replicate_id: int, purpose: int) -> np.random.Generator:
    sequence = np.random.SeedSequence([seed, replicate_id, purpose])
    return np.random.Generator(np.random.PCG64DXSM(sequence))


def _factor_blocks(
    theta: FloatArray, term_sizes: tuple[int, ...], sigma: float
) -> tuple[FloatArray, ...]:
    blocks: list[FloatArray] = []
    cursor = 0
    for size in term_sizes:
        factor = np.zeros((size, size), dtype=np.float64)
        for column in range(size):
            width = size - column
            factor[column:, column] = theta[cursor : cursor + width]
            cursor += width
        blocks.append(sigma * factor)
    if cursor != theta.shape[0]:
        raise BootstrapError("covariance parameter identity is inconsistent")
    return tuple(blocks)


def _random_contribution(
    model: LinearMixedModelResult,
    replicate_id: int,
    seed: int,
    mode: SimulationMode,
) -> FloatArray:
    design = model._training_design
    contribution = np.zeros(design.spec.n, dtype=np.float64)
    if mode == "conditional":
        if isinstance(design, SingleGroupDesign):
            effects = model.random_effects.reshape(
                design.spec.group_count, design.spec.k
            )
            contribution += np.einsum(
                "nk,nk->n",
                design.spec.random_design,
                effects[design.spec.group_indices],
            )
            return contribution
        cursor = 0
        for term in design.spec.terms:
            stop = cursor + term.q
            effects = model.random_effects[cursor:stop].reshape(
                term.group_count, term.k
            )
            cursor = stop
            contribution += np.einsum(
                "nk,nk->n", term.random_design, effects[term.group_indices]
            )
        return contribution

    generator = _rng(seed, replicate_id, _RANDOM_EFFECT_PURPOSE)
    blocks = _factor_blocks(model.theta, model.covariance_term_sizes, model.sigma)
    if isinstance(design, SingleGroupDesign):
        column = 0
        for size, factor in zip(model.covariance_term_sizes, blocks, strict=True):
            effects = generator.standard_normal((design.spec.group_count, size))
            effects = effects @ factor.T
            random_design = design.spec.random_design[:, column : column + size]
            contribution += np.einsum(
                "nk,nk->n", random_design, effects[design.spec.group_indices]
            )
            column += size
        return contribution

    for term, factor in zip(design.spec.terms, blocks, strict=True):
        effects = generator.standard_normal((term.group_count, term.k)) @ factor.T
        contribution += np.einsum(
            "nk,nk->n", term.random_design, effects[term.group_indices]
        )
    return contribution


def _array_payload(value: FloatArray) -> bytes:
    stream = io.BytesIO()
    np.save(stream, np.asarray(value, dtype="<f8"), allow_pickle=False)
    return stream.getvalue()


def _simulate_one(
    model: LinearMixedModelResult,
    replicate_id: int,
    seed: int,
    mode: SimulationMode,
) -> SimulationDraw:
    design = model._training_design
    mean = design.spec.offset + design.spec.x @ model.beta
    mean = mean + _random_contribution(model, replicate_id, seed, mode)
    residual_rng = _rng(seed, replicate_id, _RESIDUAL_PURPOSE)
    residual = (
        model.sigma
        * residual_rng.standard_normal(design.spec.n)
        / np.sqrt(design.spec.weights)
    )
    response = _readonly(mean + residual)
    payload = _array_payload(response)
    return SimulationDraw(
        replicate_id=replicate_id,
        response=response,
        response_sha256=hashlib.sha256(payload).hexdigest(),
    )


def simulate(
    model: LinearMixedModelResult,
    replicates: int,
    *,
    seed: int,
    mode: SimulationMode = "unconditional",
    limits: InferenceLimits | None = None,
) -> SimulationBatch:
    """Draw responses with replicate/purpose-separated deterministic streams."""
    active_limits = limits or InferenceLimits()
    if not isinstance(active_limits, InferenceLimits):
        raise ModelSpecificationError("limits must be an InferenceLimits instance")
    _validated_request(model, replicates, seed, mode, active_limits)
    draws = tuple(
        _simulate_one(model, replicate_id, seed, mode)
        for replicate_id in range(replicates)
    )
    return SimulationBatch(
        mode=mode,
        root_seed=seed,
        bit_generator=_BIT_GENERATOR,
        numpy_version=np.__version__,
        draws=draws,
    )


def _update_array_hash(digest: Any, name: str, value: np.ndarray[Any, Any]) -> None:
    array = np.ascontiguousarray(value)
    label = name.encode("utf-8")
    shape = json.dumps(list(array.shape), separators=(",", ":")).encode("ascii")
    dtype = array.dtype.str.encode("ascii")
    digest.update(len(label).to_bytes(4, "big"))
    digest.update(label)
    digest.update(len(shape).to_bytes(4, "big"))
    digest.update(shape)
    digest.update(len(dtype).to_bytes(4, "big"))
    digest.update(dtype)
    digest.update(array.tobytes(order="C"))


def _fit_fingerprint(model: LinearMixedModelResult) -> str:
    digest = hashlib.sha256()
    design = model._training_design
    metadata = {
        "formula": model.formula,
        "kind": model.kind.value,
        "row_ids": list(model.row_ids),
        "fixed_names": list(model.fixed_names),
        "random_names": list(model.random_names),
        "covariance_term_sizes": list(model.covariance_term_sizes),
        "control": {
            "initial_upper_bound": model._fit_control.initial_upper_bound,
            "maximum_upper_bound": model._fit_control.maximum_upper_bound,
            "absolute_theta_tolerance": model._fit_control.absolute_theta_tolerance,
            "maximum_evaluations": model._fit_control.maximum_evaluations,
            "boundary_tolerance": model._fit_control.boundary_tolerance,
        },
    }
    digest.update(_canonical_json(metadata))
    for name, value in (
        ("y", design.spec.y),
        ("x", design.spec.x),
        ("weights", design.spec.weights),
        ("offset", design.spec.offset),
        ("theta", model.theta),
        ("beta", model.beta),
    ):
        _update_array_hash(digest, name, value)
    if isinstance(design, SingleGroupDesign):
        _update_array_hash(digest, "group_indices", design.spec.group_indices)
        _update_array_hash(digest, "random_design", design.spec.random_design)
    else:
        for index, term in enumerate(design.spec.terms):
            _update_array_hash(digest, f"group_indices_{index}", term.group_indices)
            _update_array_hash(digest, f"random_design_{index}", term.random_design)
    return digest.hexdigest()


def _implementation_sha256() -> str:
    package_directory = Path(__file__).resolve().parent
    sources = sorted(package_directory.glob("*.py"), key=lambda path: path.name)
    if not sources:
        raise BootstrapError("installed Kamino sources are unavailable for provenance")
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
        raise BootstrapError(
            f"could not hash installed Kamino sources: {error}"
        ) from error
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    try:
        result = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise BootstrapError(
            f"ledger metadata is not canonical JSON: {error}"
        ) from error
    return result.encode("utf-8")


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            os.chmod(temporary, 0o600)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except OSError as error:
        with suppress(OSError):
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        raise BootstrapError(f"could not atomically write ledger: {error}") from error


def _record_metadata(record: BootstrapReplicate) -> dict[str, object]:
    return {
        "replicate_id": record.replicate_id,
        "status": record.status,
        "response_path": f"responses/{record.replicate_id:08d}.npy",
        "response_sha256": record.response_sha256,
        "statistics": (
            None if record.statistics is None else record.statistics.tolist()
        ),
        "boundary": record.boundary,
        "warnings": list(record.warnings),
        "message": record.message,
    }


def _producer_metadata() -> dict[str, object]:
    return {
        "package": "kamino",
        "version": version("kamino"),
        "reference_profile": REFERENCE_PROFILE,
        "implementation_sha256": _implementation_sha256(),
        "project_plan_sha256": PROJECT_PLAN_SHA256,
    }


def _request_metadata(
    model: LinearMixedModelResult,
    replicates: int,
    seed: int,
    mode: SimulationMode,
) -> dict[str, object]:
    return {
        "fit_sha256": _fit_fingerprint(model),
        "replicates": replicates,
        "root_seed": str(seed),
        "mode": mode,
        "bit_generator": _BIT_GENERATOR,
        "numpy_version": np.__version__,
        "statistic": {
            "kind": "retained_fixed_effects",
            "names": list(model.fixed_names),
        },
    }


def _manifest(
    producer: dict[str, object],
    request: dict[str, object],
    records: Sequence[BootstrapReplicate],
) -> dict[str, object]:
    content = {
        **request,
        "records": [_record_metadata(record) for record in records],
    }
    return {
        "format": _LEDGER_FORMAT,
        "schema_version": _LEDGER_SCHEMA_VERSION,
        "producer": producer,
        **content,
        "integrity": {
            "content_sha256": hashlib.sha256(_canonical_json(content)).hexdigest()
        },
    }


def _prepare_ledger(path: str | Path, limits: InferenceLimits) -> Path:
    requested = Path(path).expanduser()
    if requested.is_symlink():
        raise BootstrapError("bootstrap ledger path must not be a symbolic link")
    ledger = requested.absolute()
    try:
        ledger.mkdir(mode=0o700, parents=False, exist_ok=True)
        responses = ledger / "responses"
        if responses.exists() and responses.is_symlink():
            raise BootstrapError("ledger responses path must not be a symbolic link")
        responses.mkdir(mode=0o700, exist_ok=True)
        if os.name != "nt":
            os.chmod(ledger, 0o700)
            os.chmod(responses, 0o700)
    except OSError as error:
        raise BootstrapError(f"could not create bootstrap ledger: {error}") from error
    manifest_path = ledger / "manifest.json"
    if (
        manifest_path.exists()
        and manifest_path.stat().st_size > limits.maximum_manifest_bytes
    ):
        raise ResourceLimitError("bootstrap manifest exceeds the configured limit")
    return ledger


@contextmanager
def _ledger_lock(ledger: Path) -> Generator[None]:
    """Hold a crash-released, cross-process lock for one ledger writer."""
    lock_path = ledger / ".writer.lock"
    try:
        handle = lock_path.open("a+b")
        if os.name != "nt":
            os.chmod(lock_path, 0o600)
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
            os.fsync(handle.fileno())
        handle.seek(0)
    except OSError as error:
        raise BootstrapError(
            f"could not open bootstrap ledger lock: {error}"
        ) from error

    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        handle.close()
        raise BootstrapError("bootstrap ledger already has an active writer") from error

    try:
        yield
    finally:
        with suppress(OSError):
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _exact_keys(value: dict[str, Any], expected: set[str], location: str) -> None:
    if set(value) != expected:
        raise BootstrapError(f"ledger {location} has unexpected or missing fields")


def _load_records(
    ledger: Path,
    model: LinearMixedModelResult,
    replicates: int,
    expected_producer: dict[str, object],
    expected_request: dict[str, object],
    limits: InferenceLimits,
) -> dict[int, BootstrapReplicate]:
    path = ledger / "manifest.json"
    if path.is_symlink():
        raise BootstrapError("ledger manifest must be a regular file")
    if not path.exists():
        if any((ledger / "responses").iterdir()):
            raise BootstrapError("ledger has response files but no manifest")
        return {}
    if not path.is_file():
        raise BootstrapError("ledger manifest must be a regular file")
    try:
        if path.stat().st_size > limits.maximum_manifest_bytes:
            raise ResourceLimitError("bootstrap manifest exceeds the configured limit")
        payload = path.read_bytes()
        if len(payload) > limits.maximum_manifest_bytes:
            raise ResourceLimitError("bootstrap manifest exceeds the configured limit")
        raw = json.loads(payload)
    except (OSError, json.JSONDecodeError) as error:
        raise BootstrapError(f"could not read bootstrap manifest: {error}") from error
    if not isinstance(raw, dict):
        raise BootstrapError("bootstrap manifest must contain a JSON object")
    manifest = cast(dict[str, Any], raw)
    _exact_keys(
        manifest,
        {
            "format",
            "schema_version",
            "producer",
            "fit_sha256",
            "replicates",
            "root_seed",
            "mode",
            "bit_generator",
            "numpy_version",
            "statistic",
            "records",
            "integrity",
        },
        "manifest",
    )
    if (
        manifest["format"] != _LEDGER_FORMAT
        or manifest["schema_version"] != _LEDGER_SCHEMA_VERSION
    ):
        raise BootstrapError("unsupported bootstrap ledger format or schema")
    producer = manifest["producer"]
    if not isinstance(producer, dict):
        raise BootstrapError("ledger producer metadata is invalid")
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
    if producer != expected_producer:
        raise BootstrapError("ledger producer identity does not match this runtime")
    content = {
        key: manifest[key]
        for key in (
            "fit_sha256",
            "replicates",
            "root_seed",
            "mode",
            "bit_generator",
            "numpy_version",
            "statistic",
            "records",
        )
    }
    integrity = manifest["integrity"]
    if not isinstance(integrity, dict) or set(integrity) != {"content_sha256"}:
        raise BootstrapError("ledger integrity metadata is invalid")
    content_hash = hashlib.sha256(_canonical_json(content)).hexdigest()
    if integrity["content_sha256"] != content_hash:
        raise BootstrapError("bootstrap manifest integrity check failed")
    for key, expected_value in expected_request.items():
        if manifest[key] != expected_value:
            raise BootstrapError(f"ledger {key} does not match this bootstrap request")
    if not isinstance(manifest["records"], list):
        raise BootstrapError("ledger records must be a list")

    records: dict[int, BootstrapReplicate] = {}
    for item in manifest["records"]:
        if not isinstance(item, dict):
            raise BootstrapError("ledger record must be an object")
        record = cast(dict[str, Any], item)
        _exact_keys(
            record,
            {
                "replicate_id",
                "status",
                "response_path",
                "response_sha256",
                "statistics",
                "boundary",
                "warnings",
                "message",
            },
            "record",
        )
        replicate_id = record["replicate_id"]
        if type(replicate_id) is not int or not 0 <= replicate_id < replicates:
            raise BootstrapError("ledger record has an invalid replicate ID")
        if replicate_id in records:
            raise BootstrapError("ledger contains a duplicate replicate ID")
        expected_path = f"responses/{replicate_id:08d}.npy"
        if record["response_path"] != expected_path:
            raise BootstrapError("ledger response path is not canonical")
        response_path = ledger / expected_path
        if response_path.is_symlink() or not response_path.is_file():
            raise BootstrapError("ledger response payload is missing or unsafe")
        try:
            if response_path.stat().st_size > limits.maximum_response_bytes:
                raise ResourceLimitError("ledger response exceeds the configured limit")
            payload = response_path.read_bytes()
            if len(payload) > limits.maximum_response_bytes:
                raise ResourceLimitError("ledger response exceeds the configured limit")
        except OSError as error:
            raise BootstrapError(f"could not read ledger response: {error}") from error
        if hashlib.sha256(payload).hexdigest() != record["response_sha256"]:
            raise BootstrapError("ledger response checksum failed")
        try:
            response = np.load(io.BytesIO(payload), allow_pickle=False)
        except (ValueError, OSError) as error:
            raise BootstrapError("ledger response payload is invalid") from error
        if response.dtype != np.dtype("<f8") or response.shape != (
            model._training_design.spec.n,
        ):
            raise BootstrapError("ledger response shape or dtype is invalid")
        if not np.isfinite(response).all():
            raise BootstrapError("ledger response contains non-finite values")
        status = record["status"]
        if not isinstance(status, str) or status not in {
            "success",
            "warning",
            "singular",
            "optimizer_failure",
            "numerical_failure",
            "statistic_failure",
        }:
            raise BootstrapError("ledger record has an invalid status")
        raw_statistics = record["statistics"]
        statistics = None
        if raw_statistics is not None:
            try:
                statistics = _readonly(raw_statistics)
            except (TypeError, ValueError) as error:
                raise BootstrapError("ledger statistics are invalid") from error
            if (
                statistics.shape != (len(model.fixed_names),)
                or not np.isfinite(statistics).all()
            ):
                raise BootstrapError("ledger statistics have an invalid shape")
        if (status in _SUCCESS_STATUSES) != (statistics is not None):
            raise BootstrapError("ledger status and statistics disagree")
        raw_warnings = record["warnings"]
        if not isinstance(raw_warnings, list) or not all(
            isinstance(value, str) for value in raw_warnings
        ):
            raise BootstrapError("ledger warnings are invalid")
        boundary = record["boundary"]
        if boundary is not None and type(boundary) is not bool:
            raise BootstrapError("ledger boundary status is invalid")
        message = record["message"]
        if message is not None and not isinstance(message, str):
            raise BootstrapError("ledger message is invalid")
        if status in _SUCCESS_STATUSES:
            if type(boundary) is not bool or message is not None:
                raise BootstrapError("ledger completed status metadata is inconsistent")
            if status == "singular" and not boundary:
                raise BootstrapError("ledger singular status metadata is inconsistent")
            if status != "singular" and boundary:
                raise BootstrapError("ledger regular status metadata is inconsistent")
            if status == "warning" and not raw_warnings:
                raise BootstrapError("ledger warning status metadata is inconsistent")
            if status == "success" and raw_warnings:
                raise BootstrapError("ledger success status metadata is inconsistent")
        elif boundary is not None or message is None:
            raise BootstrapError("ledger failure status metadata is inconsistent")
        records[replicate_id] = BootstrapReplicate(
            replicate_id=replicate_id,
            status=cast(BootstrapStatus, status),
            response_sha256=record["response_sha256"],
            statistics=statistics,
            boundary=boundary,
            warnings=tuple(raw_warnings),
            message=message,
        )
    return records


def _write_ledger(
    ledger: Path,
    producer: dict[str, object],
    request: dict[str, object],
    records: dict[int, BootstrapReplicate],
    limits: InferenceLimits,
) -> None:
    ordered = tuple(records[index] for index in sorted(records))
    payload = _canonical_json(_manifest(producer, request, ordered)) + b"\n"
    if len(payload) > limits.maximum_manifest_bytes:
        raise ResourceLimitError("bootstrap manifest exceeds the configured limit")
    _atomic_write(ledger / "manifest.json", payload)


def _write_response(
    ledger: Path, draw: SimulationDraw, limits: InferenceLimits
) -> None:
    payload = _array_payload(draw.response)
    if len(payload) > limits.maximum_response_bytes:
        raise ResourceLimitError("ledger response exceeds the configured limit")
    if hashlib.sha256(payload).hexdigest() != draw.response_sha256:
        raise BootstrapError("simulated response identity changed before ledger write")
    _atomic_write(ledger / "responses" / f"{draw.replicate_id:08d}.npy", payload)


def _fit_replicate(
    model: LinearMixedModelResult,
    replicate_id: int,
    seed: int,
    mode: SimulationMode,
) -> tuple[SimulationDraw, BootstrapReplicate]:
    draw = _simulate_one(model, replicate_id, seed, mode)
    found: list[warnings.WarningMessage] = []
    captured: tuple[str, ...] = ()
    try:
        with warnings.catch_warnings(record=True) as recorded:
            found = recorded
            warnings.simplefilter("always")
            fitted = model.refit(draw.response)
        captured = tuple(str(item.message) for item in found)
        status: BootstrapStatus
        if fitted.diagnostics.boundary:
            status = "singular"
        elif captured:
            status = "warning"
        else:
            status = "success"
        record = BootstrapReplicate(
            replicate_id=replicate_id,
            status=status,
            response_sha256=draw.response_sha256,
            statistics=_readonly(fitted.beta),
            boundary=fitted.diagnostics.boundary,
            warnings=captured,
            message=None,
        )
    except ConvergenceError as error:
        captured = tuple(str(item.message) for item in found)
        record = BootstrapReplicate(
            replicate_id,
            "optimizer_failure",
            draw.response_sha256,
            None,
            None,
            captured,
            str(error),
        )
    except NumericalError as error:
        captured = tuple(str(item.message) for item in found)
        record = BootstrapReplicate(
            replicate_id,
            "numerical_failure",
            draw.response_sha256,
            None,
            None,
            captured,
            str(error),
        )
    except Exception as error:
        captured = tuple(str(item.message) for item in found)
        record = BootstrapReplicate(
            replicate_id,
            "statistic_failure",
            draw.response_sha256,
            None,
            None,
            captured,
            f"{type(error).__name__}: {error}",
        )
    return draw, record


def _result(
    model: LinearMixedModelResult,
    records: dict[int, BootstrapReplicate],
    replicates: int,
    seed: int,
    mode: SimulationMode,
    ledger: Path | None,
    allow_incomplete: bool,
) -> BootstrapResult:
    if len(records) != replicates:
        raise BootstrapError(
            f"bootstrap stopped after {len(records)} of {replicates} replicates"
        )
    ordered = tuple(records[index] for index in range(replicates))
    estimates = np.full((replicates, len(model.fixed_names)), np.nan, dtype=np.float64)
    for record in ordered:
        if record.statistics is not None:
            estimates[record.replicate_id] = record.statistics
    estimates.setflags(write=False)
    return BootstrapResult(
        mode=mode,
        root_seed=seed,
        bit_generator=_BIT_GENERATOR,
        numpy_version=np.__version__,
        statistic_names=model.fixed_names,
        observed=_readonly(model.beta),
        estimates=estimates,
        records=ordered,
        ledger_path=ledger,
        allow_incomplete=allow_incomplete,
    )


def _execute_bootstrap(
    model: LinearMixedModelResult,
    replicates: int,
    seed: int,
    mode: SimulationMode,
    workers: int,
    ledger: Path | None,
    allow_incomplete: bool,
    cancel_event: Event | None,
    active_limits: InferenceLimits,
) -> BootstrapResult:
    producer: dict[str, object] | None = None
    request: dict[str, object] | None = None
    if ledger is None:
        records: dict[int, BootstrapReplicate] = {}
    else:
        producer = _producer_metadata()
        request = _request_metadata(model, replicates, seed, mode)
        records = _load_records(
            ledger,
            model,
            replicates,
            producer,
            request,
            active_limits,
        )
        if not (ledger / "manifest.json").exists():
            _write_ledger(ledger, producer, request, records, active_limits)
    pending = [index for index in range(replicates) if index not in records]

    def accept(draw: SimulationDraw, record: BootstrapReplicate) -> None:
        if ledger is not None:
            assert producer is not None and request is not None
            _write_response(ledger, draw, active_limits)
        records[record.replicate_id] = record
        if ledger is not None:
            assert producer is not None and request is not None
            _write_ledger(ledger, producer, request, records, active_limits)

    if workers == 1:
        for replicate_id in pending:
            if cancel_event is not None and cancel_event.is_set():
                break
            draw, record = _fit_replicate(model, replicate_id, seed, mode)
            accept(draw, record)
    elif pending:
        with (
            threadpool_limits(limits=1),
            ThreadPoolExecutor(
                max_workers=min(workers, len(pending)),
                thread_name_prefix="kamino-bootstrap",
            ) as executor,
        ):
            futures: dict[Future[tuple[SimulationDraw, BootstrapReplicate]], int] = {}
            iterator = iter(pending)
            for _ in range(min(workers, len(pending))):
                replicate_id = next(iterator)
                futures[
                    executor.submit(_fit_replicate, model, replicate_id, seed, mode)
                ] = replicate_id
            while futures:
                future = next(as_completed(futures))
                futures.pop(future)
                draw, record = future.result()
                accept(draw, record)
                if cancel_event is not None and cancel_event.is_set():
                    continue
                try:
                    replicate_id = next(iterator)
                except StopIteration:
                    continue
                futures[
                    executor.submit(_fit_replicate, model, replicate_id, seed, mode)
                ] = replicate_id

    return _result(
        model,
        records,
        replicates,
        seed,
        mode,
        ledger,
        allow_incomplete,
    )


def parametric_bootstrap(
    model: LinearMixedModelResult,
    replicates: int,
    *,
    seed: int,
    mode: SimulationMode = "unconditional",
    workers: int = 1,
    ledger_path: str | Path | None = None,
    allow_incomplete: bool = False,
    cancel_event: Event | None = None,
    limits: InferenceLimits | None = None,
) -> BootstrapResult:
    """Simulate and refit retained fixed effects with an optional private ledger."""
    active_limits = limits or InferenceLimits()
    if not isinstance(active_limits, InferenceLimits):
        raise ModelSpecificationError("limits must be an InferenceLimits instance")
    _validated_request(model, replicates, seed, mode, active_limits)
    if type(workers) is not int or workers <= 0:
        raise ModelSpecificationError("workers must be a positive integer")
    if workers > active_limits.maximum_workers:
        raise ResourceLimitError("worker count exceeds the inference limit")
    if not isinstance(allow_incomplete, bool):
        raise ModelSpecificationError("allow_incomplete must be a boolean")
    if cancel_event is not None and not isinstance(cancel_event, Event):
        raise ModelSpecificationError("cancel_event must be a threading.Event")

    ledger = (
        None if ledger_path is None else _prepare_ledger(ledger_path, active_limits)
    )
    arguments = (
        model,
        replicates,
        seed,
        mode,
        workers,
        ledger,
        allow_incomplete,
        cancel_event,
        active_limits,
    )
    if ledger is None:
        return _execute_bootstrap(*arguments)
    with _ledger_lock(ledger):
        return _execute_bootstrap(*arguments)


__all__ = [
    "BootstrapInterval",
    "BootstrapReplicate",
    "BootstrapResult",
    "FailureAccounting",
    "InferenceLimits",
    "SimulationBatch",
    "SimulationDraw",
    "SimulationMode",
    "parametric_bootstrap",
    "simulate",
]
