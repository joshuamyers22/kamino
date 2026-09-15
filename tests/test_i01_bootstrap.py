# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportPrivateUsage=false

from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import warnings
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Event
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

import kamino.inference as inference_module
from kamino import (
    BootstrapError,
    BootstrapReplicate,
    BootstrapResult,
    InferenceLimits,
    LinearMixedModelResult,
    ResourceLimitError,
    lmer,
    parametric_bootstrap,
    simulate,
)
from kamino.errors import ConvergenceError, ModelSpecificationError, NumericalError
from kamino.formula import ColumnInput

FIXTURE_PATH = (
    Path(__file__).parents[1] / "oracle" / "fixtures" / "v1" / "dyestuff.json"
)
I01_FIXTURE_PATH = FIXTURE_PATH.with_name("i01_bootstrap.json")


def _frame() -> pd.DataFrame:
    fixture: dict[str, Any] = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    data = fixture["data"]
    return pd.DataFrame(
        {"Yield": data["response"], "Batch": data["groups"]},
        index=data["row_ids"],
    )


def _fit() -> LinearMixedModelResult:
    return lmer("Yield ~ 1 + (1 | Batch)", _frame(), reml=False)


def _rewrite_manifest(
    ledger: Path, mutation: Any, *, repair_integrity: bool = True
) -> None:
    path = ledger / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    mutation(manifest)
    if repair_integrity:
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
        payload = json.dumps(
            content,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        manifest["integrity"]["content_sha256"] = hashlib.sha256(payload).hexdigest()
    path.write_text(
        json.dumps(manifest, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _replace_response_payload(
    ledger: Path, value: np.ndarray[Any, np.dtype[np.float64]]
) -> None:
    stream = io.BytesIO()
    np.save(stream, value, allow_pickle=False)
    payload = stream.getvalue()
    (ledger / "responses" / "00000000.npy").write_bytes(payload)

    def update(manifest: dict[str, Any]) -> None:
        manifest["records"][0]["response_sha256"] = hashlib.sha256(payload).hexdigest()

    _rewrite_manifest(ledger, update)


def test_refit_reuses_exact_model_identity() -> None:
    fitted = _fit()
    refitted = fitted.refit(fitted.fitted_values + fitted.residuals)

    assert refitted.formula == fitted.formula
    assert refitted.kind is fitted.kind
    assert refitted.row_ids == fitted.row_ids
    assert refitted.fixed_names == fitted.fixed_names
    assert refitted.random_names == fitted.random_names
    assert refitted.objective == pytest.approx(fitted.objective, abs=1e-10)
    np.testing.assert_allclose(refitted.theta, fitted.theta, atol=1e-10, rtol=0.0)
    np.testing.assert_allclose(refitted.beta, fitted.beta, atol=1e-12, rtol=0.0)


@pytest.mark.parametrize(
    "case",
    json.loads(I01_FIXTURE_PATH.read_text(encoding="utf-8"))["cases"],
    ids=lambda case: case["id"],
)
def test_simulated_response_refits_match_pinned_lme4(case: dict[str, Any]) -> None:
    fitted = lmer(
        "Yield ~ 1 + (1 | Batch)",
        _frame(),
        reml=case["kind"] == "reml",
    )
    draw = fitted.simulate(
        1,
        seed=int(case["root_seed"]),
        mode=case["mode"],
    ).draws[0]
    assert draw.response_sha256 == case["response_sha256"]
    np.testing.assert_array_equal(draw.response, case["response"])

    refitted = fitted.refit(draw.response)
    expected = case["fit"]
    assert refitted.objective == pytest.approx(expected["objective"], abs=1e-8)
    assert refitted.log_likelihood == pytest.approx(
        expected["log_likelihood"], abs=1e-8
    )
    np.testing.assert_allclose(refitted.theta, expected["theta"], atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(refitted.beta, expected["beta"], atol=1e-8, rtol=0.0)
    assert refitted.sigma == pytest.approx(expected["sigma"], abs=1e-6)
    assert refitted.random_variance == pytest.approx(
        expected["random_variance"], abs=1e-4
    )
    assert refitted.diagnostics.boundary is expected["boundary"]


@pytest.mark.parametrize(
    "response",
    [[1.0], [1.0] * 30 + [float("nan")]],
)
def test_refit_rejects_invalid_response(response: list[float]) -> None:
    with pytest.raises(ModelSpecificationError, match="refit response"):
        _fit().refit(response)


@pytest.mark.parametrize("mode", ["unconditional", "conditional"])
def test_simulation_is_replicate_deterministic_and_immutable(mode: str) -> None:
    fitted = _fit()
    first = simulate(fitted, 3, seed=4102, mode=mode)  # type: ignore[arg-type]
    second = fitted.simulate(3, seed=4102, mode=mode)  # type: ignore[arg-type]

    assert first.bit_generator == "PCG64DXSM"
    assert first.numpy_version == np.__version__
    assert tuple(draw.replicate_id for draw in first.draws) == (0, 1, 2)
    for left, right in zip(first.draws, second.draws, strict=True):
        np.testing.assert_array_equal(left.response, right.response)
        assert left.response_sha256 == right.response_sha256
        assert not left.response.flags.writeable
    assert first.draws[0].response_sha256 != first.draws[1].response_sha256


def test_unconditional_and_conditional_simulation_have_declared_moments() -> None:
    fitted = _fit()
    count = 5_000
    unconditional = fitted.simulate(count, seed=901, mode="unconditional")
    conditional = fitted.simulate(count, seed=901, mode="conditional")
    unconditional_values = np.stack([draw.response for draw in unconditional.draws])
    conditional_values = np.stack([draw.response for draw in conditional.draws])

    population = fitted.predict(mode="population").values
    conditional_mean = fitted.predict(mode="conditional").values
    np.testing.assert_allclose(
        unconditional_values.mean(axis=0), population, atol=3.0, rtol=0.0
    )
    np.testing.assert_allclose(
        conditional_values.mean(axis=0), conditional_mean, atol=3.0, rtol=0.0
    )
    same_group_covariance = np.cov(
        unconditional_values[:, 0], unconditional_values[:, 1]
    )[0, 1]
    different_group_covariance = np.cov(
        unconditional_values[:, 0], unconditional_values[:, 5]
    )[0, 1]
    assert same_group_covariance == pytest.approx(fitted.random_variance, abs=70.0)
    assert different_group_covariance == pytest.approx(0.0, abs=170.0)
    assert np.var(conditional_values[:, 0], ddof=1) == pytest.approx(
        fitted.sigma2, rel=0.06
    )


def test_simulation_preflights_seed_mode_and_allocation() -> None:
    fitted = _fit()
    with pytest.raises(ModelSpecificationError, match="replicates"):
        simulate(fitted, 0, seed=1)
    with pytest.raises(ModelSpecificationError, match="seed"):
        simulate(fitted, 1, seed=-1)
    with pytest.raises(ModelSpecificationError, match="mode"):
        simulate(fitted, 1, seed=1, mode="mystery")  # type: ignore[arg-type]
    with pytest.raises(ResourceLimitError, match="response size"):
        simulate(
            fitted,
            2,
            seed=1,
            limits=InferenceLimits(maximum_response_values=59),
        )


def test_weighted_conditional_simulation_scales_residual_variance() -> None:
    groups = [f"g{index // 5}" for index in range(20)]
    weights = np.tile([1.0, 4.0, 1.0, 4.0, 1.0], 4)
    response = np.array(
        [10.0 + 0.8 * (index // 5) + (-1.0) ** index for index in range(20)]
    )
    fitted = lmer(
        "y ~ 1 + (1 | g)",
        {"y": response, "g": groups},
        weights=weights,
        reml=False,
    )
    draws = fitted.simulate(4_000, seed=202, mode="conditional")
    values = np.stack([draw.response for draw in draws.draws])
    variance_weight_one = np.var(values[:, 0], ddof=1)
    variance_weight_four = np.var(values[:, 1], ddof=1)
    assert variance_weight_one / variance_weight_four == pytest.approx(4.0, rel=0.1)


def test_refit_supports_correlated_slope_and_coupled_sparse_designs(
    tmp_path: Path,
) -> None:
    sleep_fixture = json.loads(
        FIXTURE_PATH.with_name("sleepstudy.json").read_text(encoding="utf-8")
    )
    sleep_data = sleep_fixture["data"]
    slope = lmer(
        "Reaction ~ Days + (1 + Days | Subject)",
        {
            "Reaction": sleep_data["response"],
            "Days": sleep_data["predictor"],
            "Subject": sleep_data["groups"],
        },
        reml=False,
    )
    slope_draw = slope.simulate(1, seed=5).draws[0]
    slope_refit = slope.refit(slope_draw.response)
    assert slope_refit.diagnostics.backend == "single-group-block-cholesky"
    assert slope_refit.theta.shape == (3,)

    sparse_fixture = json.loads(
        FIXTURE_PATH.with_name("penicillin_sparse.json").read_text(encoding="utf-8")
    )
    response_name = sparse_fixture["response_name"]
    sparse_data: dict[str, ColumnInput] = {
        response_name: sparse_fixture["data"]["response"]
    }
    for name in sparse_fixture["group_columns"]:
        sparse_data[name] = sparse_fixture["data"][name]
    sparse = lmer(sparse_fixture["formula"], sparse_data, reml=False)
    sparse_draw = sparse.simulate(1, seed=6).draws[0]
    sparse_refit = sparse.refit(sparse_draw.response)
    assert sparse_refit.diagnostics.backend == "scipy-superlu-symmetric-sparse"
    assert sparse_refit.theta.shape == sparse.theta.shape
    conditional_sparse_draw = sparse.simulate(1, seed=8, mode="conditional").draws[0]
    assert conditional_sparse_draw.response.shape == sparse.fitted_values.shape
    sparse_bootstrap = sparse.parametric_bootstrap(
        1, seed=7, ledger_path=tmp_path / "sparse-ledger", allow_incomplete=True
    )
    assert len(sparse_bootstrap.records) == 1


def test_bootstrap_is_worker_count_invariant() -> None:
    fitted = _fit()
    serial = fitted.parametric_bootstrap(4, seed=77, workers=1)
    parallel = fitted.parametric_bootstrap(4, seed=77, workers=2)

    assert serial.complete
    assert parallel.complete
    assert serial.failure_rate == 0.0
    accounting = serial.failure_accounting()
    assert accounting.requested_replicates == 4
    assert accounting.completed_replicates == 4
    assert accounting.failed_replicates == 0
    assert sum(count for _, count in accounting.statuses) == 4
    assert {status for status, _ in accounting.statuses} <= {"success", "singular"}
    assert accounting.upper_failure_rate == pytest.approx(1.0 - 0.05 ** (1.0 / 4.0))
    assert tuple(record.response_sha256 for record in serial.records) == tuple(
        record.response_sha256 for record in parallel.records
    )
    np.testing.assert_allclose(serial.estimates, parallel.estimates, atol=0.0, rtol=0.0)


def test_private_ledger_resumes_without_refitting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fitted = _fit()
    ledger = tmp_path / "private-bootstrap"
    first = fitted.parametric_bootstrap(3, seed=18, workers=2, ledger_path=ledger)

    assert first.complete
    assert (ledger / "manifest.json").is_file()
    assert len(tuple((ledger / "responses").glob("*.npy"))) == 3
    if os.name != "nt":
        assert stat.S_IMODE(ledger.stat().st_mode) == 0o700
        assert stat.S_IMODE((ledger / "manifest.json").stat().st_mode) == 0o600

    saved_response = np.load(ledger / "responses" / "00000000.npy", allow_pickle=False)
    replayed = fitted.refit(saved_response)
    first_statistics = first.records[0].statistics
    assert first_statistics is not None
    np.testing.assert_allclose(replayed.beta, first_statistics, atol=0.0, rtol=0.0)

    def unexpected_refit(*args: object, **kwargs: object) -> object:
        raise AssertionError("completed ledger entries must not be refitted")

    monkeypatch.setattr(inference_module, "_fit_replicate", unexpected_refit)
    resumed = parametric_bootstrap(fitted, 3, seed=18, workers=1, ledger_path=ledger)
    np.testing.assert_array_equal(resumed.estimates, first.estimates)
    assert resumed.records == first.records


def test_private_ledger_rejects_a_concurrent_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fitted = _fit()
    ledger = tmp_path / "private-bootstrap"
    started = Event()
    release = Event()
    original = inference_module._fit_replicate

    def blocked_refit(*args: Any, **kwargs: Any) -> Any:
        started.set()
        if not release.wait(timeout=10.0):
            raise AssertionError("test did not release the ledger writer")
        return original(*args, **kwargs)

    monkeypatch.setattr(inference_module, "_fit_replicate", blocked_refit)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            fitted.parametric_bootstrap,
            1,
            seed=807,
            ledger_path=ledger,
        )
        assert started.wait(timeout=10.0)
        with pytest.raises(BootstrapError, match="active writer"):
            fitted.parametric_bootstrap(1, seed=807, ledger_path=ledger)
        release.set()
        assert future.result().complete


def test_ledger_rejects_request_drift_and_corrupt_payload(tmp_path: Path) -> None:
    fitted = _fit()
    ledger = tmp_path / "private-bootstrap"
    fitted.parametric_bootstrap(2, seed=55, ledger_path=ledger)

    with pytest.raises(BootstrapError, match="root_seed"):
        fitted.parametric_bootstrap(2, seed=56, ledger_path=ledger)

    payload = ledger / "responses" / "00000000.npy"
    payload.write_bytes(payload.read_bytes() + b"corrupt")
    with pytest.raises(BootstrapError, match="checksum"):
        fitted.parametric_bootstrap(2, seed=55, ledger_path=ledger)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("format", "unsupported"),
        ("producer_fields", "producer has unexpected"),
        ("producer_identity", "producer identity"),
        ("integrity_fields", "integrity metadata"),
        ("integrity_hash", "integrity check"),
        ("fit", "fit_sha256"),
        ("records_type", "records must be a list"),
        ("record_type", "record must be an object"),
        ("replicate_id", "invalid replicate ID"),
        ("response_path", "not canonical"),
        ("status", "invalid status"),
        ("statistics_type", "statistics are invalid"),
        ("statistics_shape", "invalid shape"),
        ("status_statistics", "status and statistics disagree"),
        ("warnings", "warnings are invalid"),
        ("boundary", "boundary status is invalid"),
        ("message", "message is invalid"),
    ],
)
def test_ledger_rejects_malformed_metadata(
    tmp_path: Path, case: str, message: str
) -> None:
    fitted = _fit()
    ledger = tmp_path / case
    fitted.parametric_bootstrap(1, seed=808, ledger_path=ledger)

    def mutate(manifest: dict[str, Any]) -> None:
        record = manifest["records"][0]
        if case == "format":
            manifest["format"] = "not-kamino"
        elif case == "producer_fields":
            manifest["producer"].pop("reference_profile")
        elif case == "producer_identity":
            manifest["producer"]["version"] = "99.0"
        elif case == "integrity_fields":
            manifest["integrity"]["extra"] = "bad"
        elif case == "integrity_hash":
            manifest["integrity"]["content_sha256"] = "0" * 64
        elif case == "fit":
            manifest["fit_sha256"] = "0" * 64
        elif case == "records_type":
            manifest["records"] = {}
        elif case == "record_type":
            manifest["records"] = [7]
        elif case == "replicate_id":
            record["replicate_id"] = 3
        elif case == "response_path":
            record["response_path"] = "../escape.npy"
        elif case == "status":
            record["status"] = "redrawn"
        elif case == "statistics_type":
            record["statistics"] = "not-numeric"
        elif case == "statistics_shape":
            record["statistics"] = [1.0, 2.0]
        elif case == "status_statistics":
            record["status"] = "optimizer_failure"
        elif case == "warnings":
            record["warnings"] = "warning"
        elif case == "boundary":
            record["boundary"] = "yes"
        elif case == "message":
            record["message"] = 5

    _rewrite_manifest(
        ledger,
        mutate,
        repair_integrity=case not in {"integrity_fields", "integrity_hash"},
    )
    with pytest.raises(BootstrapError, match=message):
        fitted.parametric_bootstrap(1, seed=808, ledger_path=ledger)


def test_ledger_rejects_duplicate_missing_and_unsafe_response(tmp_path: Path) -> None:
    fitted = _fit()
    duplicate = tmp_path / "duplicate"
    fitted.parametric_bootstrap(2, seed=809, ledger_path=duplicate)

    def duplicate_record(manifest: dict[str, Any]) -> None:
        manifest["records"][1]["replicate_id"] = 0

    _rewrite_manifest(duplicate, duplicate_record)
    with pytest.raises(BootstrapError, match="duplicate"):
        fitted.parametric_bootstrap(2, seed=809, ledger_path=duplicate)

    missing = tmp_path / "missing"
    fitted.parametric_bootstrap(1, seed=810, ledger_path=missing)
    (missing / "responses" / "00000000.npy").unlink()
    with pytest.raises(BootstrapError, match="missing or unsafe"):
        fitted.parametric_bootstrap(1, seed=810, ledger_path=missing)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (np.array([1.0, 2.0]), "shape or dtype"),
        (np.full(30, np.nan), "non-finite"),
    ],
)
def test_ledger_rejects_invalid_response_arrays(
    tmp_path: Path,
    value: np.ndarray[Any, np.dtype[np.float64]],
    message: str,
) -> None:
    fitted = _fit()
    ledger = tmp_path / message.replace(" ", "-")
    fitted.parametric_bootstrap(1, seed=811, ledger_path=ledger)
    _replace_response_payload(ledger, value)
    with pytest.raises(BootstrapError, match=message):
        fitted.parametric_bootstrap(1, seed=811, ledger_path=ledger)


def test_ledger_rejects_non_npy_and_size_limit(tmp_path: Path) -> None:
    fitted = _fit()
    invalid = tmp_path / "invalid-npy"
    fitted.parametric_bootstrap(1, seed=812, ledger_path=invalid)
    payload = b"not a numpy array"
    response_path = invalid / "responses" / "00000000.npy"
    response_path.write_bytes(payload)

    def update_hash(manifest: dict[str, Any]) -> None:
        manifest["records"][0]["response_sha256"] = hashlib.sha256(payload).hexdigest()

    _rewrite_manifest(invalid, update_hash)
    with pytest.raises(BootstrapError, match="payload is invalid"):
        fitted.parametric_bootstrap(1, seed=812, ledger_path=invalid)

    limited = tmp_path / "limited"
    fitted.parametric_bootstrap(1, seed=813, ledger_path=limited)
    with pytest.raises(ResourceLimitError, match="response exceeds"):
        fitted.parametric_bootstrap(
            1,
            seed=813,
            ledger_path=limited,
            limits=InferenceLimits(maximum_response_bytes=8),
        )

    fresh = tmp_path / "fresh-limited"
    with pytest.raises(ResourceLimitError, match="response exceeds"):
        fitted.parametric_bootstrap(
            1,
            seed=819,
            ledger_path=fresh,
            limits=InferenceLimits(maximum_response_bytes=8),
        )
    assert (fresh / "manifest.json").is_file()
    assert not any((fresh / "responses").iterdir())


def test_ledger_rejects_symlinks_orphan_payloads_and_invalid_json(
    tmp_path: Path,
) -> None:
    fitted = _fit()
    if os.name != "nt":
        target = tmp_path / "target"
        target.mkdir()
        linked = tmp_path / "linked-ledger"
        linked.symlink_to(target, target_is_directory=True)
        with pytest.raises(BootstrapError, match="symbolic link"):
            fitted.parametric_bootstrap(1, seed=814, ledger_path=linked)

        response_target = tmp_path / "response-target"
        response_target.mkdir()
        unsafe = tmp_path / "unsafe-responses"
        unsafe.mkdir()
        (unsafe / "responses").symlink_to(response_target, target_is_directory=True)
        with pytest.raises(BootstrapError, match="responses path"):
            fitted.parametric_bootstrap(1, seed=815, ledger_path=unsafe)

    orphan = tmp_path / "orphan"
    (orphan / "responses").mkdir(parents=True)
    (orphan / "responses" / "00000000.npy").write_bytes(b"orphan")
    with pytest.raises(BootstrapError, match="no manifest"):
        fitted.parametric_bootstrap(1, seed=816, ledger_path=orphan)

    invalid = tmp_path / "invalid-json"
    (invalid / "responses").mkdir(parents=True)
    (invalid / "manifest.json").write_text("not-json", encoding="utf-8")
    with pytest.raises(BootstrapError, match="could not read"):
        fitted.parametric_bootstrap(1, seed=817, ledger_path=invalid)

    non_object = tmp_path / "non-object"
    (non_object / "responses").mkdir(parents=True)
    (non_object / "manifest.json").write_text("[]", encoding="utf-8")
    with pytest.raises(BootstrapError, match="JSON object"):
        fitted.parametric_bootstrap(1, seed=818, ledger_path=non_object)


def test_cancelled_bootstrap_preserves_resumable_empty_ledger(tmp_path: Path) -> None:
    fitted = _fit()
    ledger = tmp_path / "private-bootstrap"
    cancelled = Event()
    cancelled.set()
    with pytest.raises(BootstrapError, match="stopped after 0 of 2"):
        parametric_bootstrap(
            fitted,
            2,
            seed=100,
            ledger_path=ledger,
            cancel_event=cancelled,
        )
    assert ledger.is_dir()
    assert (ledger / "manifest.json").is_file()
    resumed = fitted.parametric_bootstrap(2, seed=100, ledger_path=ledger)
    assert resumed.complete


def test_intervals_fail_closed_for_unresolved_replicates() -> None:
    records = (
        BootstrapReplicate(0, "success", "a", np.array([0.5]), False, (), None),
        BootstrapReplicate(
            1, "optimizer_failure", "b", None, None, (), "did not converge"
        ),
        BootstrapReplicate(2, "success", "c", np.array([1.5]), False, (), None),
    )
    estimates = np.array([[0.5], [np.nan], [1.5]])
    strict = BootstrapResult(
        "unconditional",
        1,
        "PCG64DXSM",
        np.__version__,
        ("(Intercept)",),
        np.array([1.0]),
        estimates,
        records,
        None,
        False,
    )
    with pytest.raises(BootstrapError, match="unresolved failed"):
        strict.interval()

    partial = BootstrapResult(
        "unconditional",
        1,
        "PCG64DXSM",
        np.__version__,
        ("(Intercept)",),
        np.array([1.0]),
        estimates,
        records,
        None,
        True,
    )
    interval = partial.interval(level=0.8, method="basic")
    assert interval.incomplete
    assert interval.completed_replicates == 2
    assert interval.failed_replicates == 1
    assert interval.quantile_convention == "numpy-linear"
    assert interval.tail_probability_standard_error == pytest.approx(
        np.sqrt(0.1 * 0.9 / 2)
    )

    with pytest.raises(BootstrapError, match="interval level"):
        partial.interval(level=1.0)
    with pytest.raises(BootstrapError, match="interval method"):
        partial.interval(method="studentized")  # type: ignore[arg-type]
    with pytest.raises(BootstrapError, match="failure confidence"):
        partial.failure_accounting(confidence_level=0.0)
    all_failed = replace(
        partial,
        estimates=np.full((1, 1), np.nan),
        records=(records[1],),
    )
    assert all_failed.failure_accounting().upper_failure_rate == 1.0
    one_completed = replace(
        partial,
        estimates=np.array([[0.5], [np.nan]]),
        records=(records[0], records[1]),
    )
    with pytest.raises(BootstrapError, match="at least two"):
        one_completed.interval()


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (ConvergenceError("no optimum"), "optimizer_failure"),
        (NumericalError("factor failed"), "numerical_failure"),
        (ValueError("statistic failed"), "statistic_failure"),
    ],
)
def test_bootstrap_classifies_refit_failures_without_redraw(
    monkeypatch: pytest.MonkeyPatch, error: Exception, status: str
) -> None:
    def failed_refit(*args: object, **kwargs: object) -> object:
        raise error

    monkeypatch.setattr(type(_fit()), "refit", failed_refit)
    result = _fit().parametric_bootstrap(1, seed=901, allow_incomplete=True)
    assert result.records[0].status == status
    assert result.records[0].statistics is None
    assert result.failed_replicates == 1


def test_bootstrap_records_nonfatal_warnings(monkeypatch: pytest.MonkeyPatch) -> None:
    fitted = _fit()
    original = cast(Callable[..., LinearMixedModelResult], type(fitted).refit)

    def warned_refit(self: Any, *args: object, **kwargs: object) -> Any:
        warnings.warn("assessment warning", RuntimeWarning, stacklevel=2)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(type(fitted), "refit", warned_refit)
    result = fitted.parametric_bootstrap(1, seed=902)
    assert result.records[0].status == "warning"
    assert result.records[0].warnings == ("assessment warning",)


def test_parallel_cancellation_leaves_a_resumable_partial_ledger(
    tmp_path: Path,
) -> None:
    fitted = _fit()
    ledger = tmp_path / "parallel-cancel"
    cancelled = Event()
    cancelled.set()
    with pytest.raises(BootstrapError, match="stopped after 2 of 5"):
        fitted.parametric_bootstrap(
            5,
            seed=903,
            workers=2,
            ledger_path=ledger,
            cancel_event=cancelled,
        )
    resumed = fitted.parametric_bootstrap(5, seed=903, workers=2, ledger_path=ledger)
    assert resumed.complete


def test_inference_argument_and_resource_limits() -> None:
    fitted = _fit()
    with pytest.raises(ModelSpecificationError, match="inference limits"):
        InferenceLimits(maximum_workers=0)
    with pytest.raises(ResourceLimitError, match="replicate count"):
        simulate(
            fitted,
            2,
            seed=1,
            limits=InferenceLimits(maximum_replicates=1),
        )
    with pytest.raises(ResourceLimitError, match="replicate count"):
        fitted.simulate(
            2,
            seed=1,
            limits=InferenceLimits(maximum_replicates=1),
        )
    with pytest.raises(ModelSpecificationError, match="limits"):
        simulate(fitted, 1, seed=1, limits="bad")  # type: ignore[arg-type]
    with pytest.raises(ModelSpecificationError, match="limits"):
        parametric_bootstrap(
            fitted,
            1,
            seed=1,
            limits="bad",  # type: ignore[arg-type]
        )
    with pytest.raises(ModelSpecificationError, match="allow_incomplete"):
        parametric_bootstrap(
            fitted,
            1,
            seed=1,
            allow_incomplete=1,  # type: ignore[arg-type]
        )
    with pytest.raises(ModelSpecificationError, match="cancel_event"):
        parametric_bootstrap(
            fitted,
            1,
            seed=1,
            cancel_event="stop",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("workers", [0, 33])
def test_bootstrap_worker_limits(workers: int) -> None:
    expected = ModelSpecificationError if workers == 0 else ResourceLimitError
    with pytest.raises(expected):
        _fit().parametric_bootstrap(1, seed=1, workers=workers)
