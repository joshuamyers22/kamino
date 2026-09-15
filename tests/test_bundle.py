# pyright: reportMissingTypeStubs=false, reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

import hashlib
import io
import json
import warnings
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import kamino.bundle as bundle_module
from kamino import BundleError, BundleLimits, FitControl, lmer, load_model_bundle
from kamino.errors import PredictionError
from kamino.results import LinearMixedModelResult, PredictionOnlyModel

FIXTURE_DIRECTORY = Path(__file__).parents[1] / "oracle" / "fixtures" / "v1"


def _fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURE_DIRECTORY / name).read_text(encoding="utf-8"))


def _dyestuff_fit(name: str, *, reml: bool) -> LinearMixedModelResult:
    data = _fixture(name)["data"]
    return lmer(
        "Yield ~ 1 + (1 | Batch)",
        pd.DataFrame(
            {
                "Yield": data["response"],
                "Batch": pd.Categorical(
                    data["groups"], categories=data["group_levels"], ordered=True
                ),
            }
        ),
        reml=reml,
    )


def _sleepstudy_fit(*, reml: bool) -> LinearMixedModelResult:
    data = _fixture("sleepstudy.json")["data"]
    return lmer(
        "Reaction ~ Days + (1 + Days | Subject)",
        pd.DataFrame(
            {
                "Reaction": data["response"],
                "Days": data["predictor"],
                "Subject": pd.Categorical(
                    data["groups"], categories=data["group_levels"], ordered=True
                ),
            }
        ),
        reml=reml,
    )


def _sleepstudy_independent_fit(*, reml: bool) -> LinearMixedModelResult:
    data = _fixture("sleepstudy_independent.json")["data"]
    return lmer(
        "Reaction ~ Days + (1 + Days || Subject)",
        pd.DataFrame(
            {
                "Reaction": data["response"],
                "Days": data["predictor"],
                "Subject": pd.Categorical(
                    data["groups"], categories=data["group_levels"], ordered=True
                ),
            }
        ),
        reml=reml,
    )


def _regular_fit(*, control: FitControl | None = None) -> LinearMixedModelResult:
    return lmer(
        "y ~ 1 + (1 | g)",
        {
            "y": [0.0, 0.1, -0.1, 10.0, 10.1, 9.9],
            "g": ["a", "a", "a", "b", "b", "b"],
        },
        reml=False,
        control=control,
    )


def _assert_retained_state(
    fitted: LinearMixedModelResult, loaded: PredictionOnlyModel
) -> None:
    assert loaded.formula == fitted.formula
    assert loaded.kind is fitted.kind
    assert loaded.objective == fitted.objective
    assert loaded.log_likelihood == fitted.log_likelihood
    assert loaded.sigma2 == fitted.sigma2
    assert loaded.random_variance == fitted.random_variance
    assert loaded.fixed_names == fitted.fixed_names
    assert loaded.random_names == fitted.random_names
    assert loaded.group_name == fitted.group_name
    assert loaded.group_levels == fitted.group_levels
    assert loaded.random_coefficient_names == fitted.random_coefficient_names
    assert loaded.covariance_term_sizes == fitted.covariance_term_sizes
    assert loaded.predictor_name == fitted.predictor_name
    assert loaded.requires_explicit_offset == fitted.requires_explicit_offset
    assert loaded.diagnostics == fitted.diagnostics
    for name in (
        "theta",
        "beta",
        "beta_covariance",
        "random_covariance",
        "random_effects",
    ):
        actual = getattr(loaded, name)
        np.testing.assert_array_equal(actual, getattr(fitted, name))
        assert not actual.flags.writeable


@pytest.mark.parametrize("reml", [False, True], ids=["ml", "reml"])
@pytest.mark.parametrize("fixture_name", ["dyestuff.json", "dyestuff2.json"])
def test_random_intercept_bundle_round_trip_preserves_predictions(
    fixture_name: str, reml: bool, tmp_path: Path
) -> None:
    fitted = _dyestuff_fit(fixture_name, reml=reml)
    path = fitted.save(tmp_path / f"{fixture_name}-{reml}.kamino")
    loaded = load_model_bundle(path)
    _assert_retained_state(fitted, loaded)

    data = {"Batch": [fitted.group_levels[0], "new"]}
    for mode in ("population", "conditional"):
        expected = fitted.predict(
            data,
            mode=mode,
            allow_new_groups=True,  # type: ignore[arg-type]
        )
        actual = loaded.predict(
            data,
            mode=mode,
            allow_new_groups=True,  # type: ignore[arg-type]
        )
        np.testing.assert_array_equal(actual.values, expected.values)
        assert actual.new_group == expected.new_group


@pytest.mark.parametrize("reml", [False, True], ids=["ml", "reml"])
def test_sleepstudy_bundle_round_trip_preserves_predictions(
    reml: bool, tmp_path: Path
) -> None:
    fitted = _sleepstudy_fit(reml=reml)
    loaded = load_model_bundle(fitted.save(tmp_path / f"sleepstudy-{reml}.kamino"))
    _assert_retained_state(fitted, loaded)

    data = {
        "Days": [0.0, 5.0, 10.0],
        "Subject": [fitted.group_levels[0], fitted.group_levels[1], "new"],
    }
    for mode in ("population", "conditional"):
        expected = fitted.predict(
            data,
            mode=mode,
            allow_new_groups=True,  # type: ignore[arg-type]
        )
        actual = loaded.predict(
            data,
            mode=mode,
            allow_new_groups=True,  # type: ignore[arg-type]
        )
        np.testing.assert_array_equal(actual.values, expected.values)
        assert actual.new_group == expected.new_group


@pytest.mark.parametrize("reml", [False, True], ids=["ml", "reml"])
def test_independent_sleepstudy_bundle_round_trip_preserves_predictions(
    reml: bool, tmp_path: Path
) -> None:
    fitted = _sleepstudy_independent_fit(reml=reml)
    loaded = load_model_bundle(
        fitted.save(tmp_path / f"sleepstudy-independent-{reml}.kamino")
    )
    _assert_retained_state(fitted, loaded)

    assert loaded.covariance_term_sizes == (1, 1)
    data = {
        "Days": [0.0, 5.0, 10.0],
        "Subject": [fitted.group_levels[0], fitted.group_levels[1], "new"],
    }
    for mode in ("population", "conditional"):
        expected = fitted.predict(
            data,
            mode=mode,
            allow_new_groups=True,  # type: ignore[arg-type]
        )
        actual = loaded.predict(
            data,
            mode=mode,
            allow_new_groups=True,  # type: ignore[arg-type]
        )
        np.testing.assert_array_equal(actual.values, expected.values)
        assert actual.new_group == expected.new_group


def test_bundle_is_deterministic_and_omits_training_data(tmp_path: Path) -> None:
    fitted = _sleepstudy_fit(reml=True)
    first = fitted.save(tmp_path / "first.kamino")
    second = fitted.save(tmp_path / "second.kamino")
    assert first.read_bytes() == second.read_bytes()

    with zipfile.ZipFile(first) as archive:
        assert set(archive.namelist()) == {
            "manifest.json",
            "arrays/theta.npy",
            "arrays/beta.npy",
            "arrays/beta_covariance.npy",
            "arrays/random_covariance.npy",
            "arrays/random_effects.npy",
        }
        manifest = json.loads(archive.read("manifest.json"))
    serialized = json.dumps(manifest, sort_keys=True)
    for forbidden in (
        "response",
        "residuals",
        "fitted_values",
        "row_ids",
        "training_groups",
        "training_offset",
        "training_fixed_design",
        "training_random_design",
        '"u"',
    ):
        assert forbidden not in serialized
    assert manifest["capabilities"] == {
        "prediction": ["population", "conditional"],
        "training_rows": False,
        "refit": False,
        "inference": False,
    }
    assert manifest["schema_version"] == "1.1.0"
    assert manifest["model"]["covariance_term_sizes"] == [2]
    project_plan = Path(__file__).parents[1] / "PROJECT_PLAN.md"
    canonical_plan = project_plan.read_bytes().replace(b"\r\n", b"\n")
    assert (
        manifest["producer"]["project_plan_sha256"]
        == hashlib.sha256(canonical_plan).hexdigest()
    )
    assert len(manifest["producer"]["implementation_sha256"]) == 64


def test_loaded_bundle_requires_explicit_data(tmp_path: Path) -> None:
    loaded = load_model_bundle(_regular_fit().save(tmp_path / "model.kamino"))
    with pytest.raises(BundleError):
        load_model_bundle(tmp_path / "missing.kamino")
    with pytest.raises(PredictionError, match="do not store training rows"):
        loaded.predict(mode="conditional")


def test_bundle_preserves_offset_requirement_and_fit_controls(tmp_path: Path) -> None:
    control = FitControl(
        initial_upper_bound=2.0,
        maximum_upper_bound=2048.0,
        absolute_theta_tolerance=1e-9,
        maximum_evaluations=777,
        boundary_tolerance=1e-6,
    )
    fitted = lmer(
        "y ~ 1 + (1 | g)",
        {
            "y": [0.1, 0.2, 0.0, 10.1, 10.2, 10.0],
            "g": ["a", "a", "a", "b", "b", "b"],
        },
        reml=False,
        offset=[0.1] * 6,
        control=control,
    )
    loaded = load_model_bundle(fitted.save(tmp_path / "offset.kamino"))
    assert loaded.requires_explicit_offset
    assert loaded.diagnostics.initial_upper_bound == 2.0
    assert loaded.diagnostics.maximum_upper_bound == 2048.0
    assert loaded.diagnostics.absolute_theta_tolerance == 1e-9
    assert loaded.diagnostics.maximum_evaluations == 777
    assert loaded.diagnostics.boundary_tolerance == 1e-6
    with pytest.raises(PredictionError, match="requires an explicit offset"):
        loaded.predict({"g": ["a"]}, mode="conditional")
    expected = fitted.predict({"g": ["a"]}, mode="conditional", offset=[0.2])
    actual = loaded.predict({"g": ["a"]}, mode="conditional", offset=[0.2])
    np.testing.assert_array_equal(actual.values, expected.values)


def _rewrite_bundle(
    source: Path,
    target: Path,
    *,
    edit_manifest: Callable[[dict[str, Any]], None] | None = None,
    replace_member: tuple[str, bytes] | None = None,
    extra_member: tuple[str, bytes] | None = None,
) -> None:
    with zipfile.ZipFile(source) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(members["manifest.json"])
    if edit_manifest is not None:
        edit_manifest(manifest)
    if replace_member is not None:
        members[replace_member[0]] = replace_member[1]
    members["manifest.json"] = (
        json.dumps(manifest, allow_nan=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode()
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
        if extra_member is not None:
            archive.writestr(*extra_member)


def _replace_raw_member(source: Path, target: Path, name: str, payload: bytes) -> None:
    with zipfile.ZipFile(source) as archive:
        members = {member: archive.read(member) for member in archive.namelist()}
    members[name] = payload
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for member, content in members.items():
            archive.writestr(member, content)


def _resign(manifest: dict[str, Any]) -> None:
    content = {"model": manifest["model"], "arrays": manifest["arrays"]}
    encoded = json.dumps(
        content, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode()
    manifest["integrity"]["model_sha256"] = hashlib.sha256(encoded).hexdigest()


def _set_nested(manifest: dict[str, Any], path: tuple[str, ...], value: object) -> None:
    target = manifest
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = value


def test_bundle_loads_legacy_schema_with_one_correlated_term(tmp_path: Path) -> None:
    fitted = _sleepstudy_fit(reml=True)
    source = fitted.save(tmp_path / "current.kamino")

    def downgrade(manifest: dict[str, Any]) -> None:
        manifest["schema_version"] = "1.0.0"
        del manifest["model"]["covariance_term_sizes"]
        _resign(manifest)

    legacy = tmp_path / "legacy.kamino"
    _rewrite_bundle(source, legacy, edit_manifest=downgrade)
    loaded = load_model_bundle(legacy)

    _assert_retained_state(fitted, loaded)
    assert loaded.covariance_term_sizes == (2,)


@pytest.mark.parametrize("term_sizes", [[], [1, 1], [0, 1], [True]])
def test_bundle_rejects_invalid_covariance_term_sizes(
    term_sizes: list[int], tmp_path: Path
) -> None:
    source = _regular_fit().save(tmp_path / "source.kamino")

    def mutate(manifest: dict[str, Any]) -> None:
        manifest["model"]["covariance_term_sizes"] = term_sizes
        _resign(manifest)

    invalid = tmp_path / "invalid.kamino"
    _rewrite_bundle(source, invalid, edit_manifest=mutate)
    with pytest.raises(BundleError, match="covariance.term|positive integer"):
        load_model_bundle(invalid)


@pytest.mark.parametrize(
    ("path", "value", "resign", "message"),
    [
        (("format",), "other", False, "format is unsupported"),
        (("producer", "package"), "other", False, "producer profile"),
        (("producer", "implementation_sha256"), "bad", False, "is invalid"),
        (("capabilities", "refit"), True, False, "capabilities"),
        (("arrays", "beta", "path"), "../beta.npy", False, "metadata"),
        (("arrays", "beta", "sha256"), "bad", False, "checksum"),
        (("model", "objective"), 0.0, False, "model checksum"),
        (("model", "design", "encoding"), "other", True, "design state"),
        (("model", "fixed_names"), ["wrong"], True, "coefficient labels"),
        (("model", "group_levels"), ["a", "a"], True, "unique labels"),
        (
            ("model", "random_names"),
            ["wrong", "also-wrong"],
            True,
            "group/coefficient map",
        ),
        (("model", "log_likelihood"), 0.0, True, "log likelihood"),
        (("model", "sigma2"), 0.0, True, "finite and positive"),
        (("model", "diagnostics", "converged"), False, True, "converged fitted model"),
        (("model", "diagnostics", "parameter_count"), 2, True, "parameter count"),
        (
            ("model", "diagnostics", "control", "maximum_upper_bound"),
            0.5,
            True,
            "bounds",
        ),
        (("model", "kind"), "unsupported", True, "kind is unsupported"),
        (("model", "requires_explicit_offset"), "yes", True, "must be a boolean"),
        (("model", "diagnostics", "evaluations"), 0, True, "at least 1"),
        (("model", "diagnostics", "message"), "", True, "nonempty string"),
    ],
)
def test_bundle_rejects_manifest_contract_mutations(
    path: tuple[str, ...],
    value: object,
    resign: bool,
    message: str,
    tmp_path: Path,
) -> None:
    source = _regular_fit().save(tmp_path / "source.kamino")

    def mutate(manifest: dict[str, Any]) -> None:
        _set_nested(manifest, path, value)
        if resign:
            _resign(manifest)

    invalid = tmp_path / "invalid.kamino"
    _rewrite_bundle(source, invalid, edit_manifest=mutate)
    with pytest.raises(BundleError, match=message):
        load_model_bundle(invalid)


def _npy_payload(value: np.ndarray[Any, Any]) -> bytes:
    stream = io.BytesIO()
    np.save(stream, value, allow_pickle=False)
    return stream.getvalue()


@pytest.mark.parametrize(
    ("name", "array", "message"),
    [
        ("beta", np.array([1.0, 2.0]), "model shape"),
        ("beta", np.array([1.0], dtype=np.float32), "must be float64"),
        ("beta", np.array([np.nan]), "non-finite"),
        ("beta_covariance", np.array([[-1.0]]), "positive semidefinite"),
        ("theta", np.array([-1.0]), "theta and random covariance"),
    ],
)
def test_bundle_rejects_invalid_numeric_payloads(
    name: str, array: np.ndarray[Any, Any], message: str, tmp_path: Path
) -> None:
    source = _regular_fit().save(tmp_path / "source.kamino")
    payload = _npy_payload(array)

    def replace_array(manifest: dict[str, Any]) -> None:
        metadata = manifest["arrays"][name]
        metadata["sha256"] = hashlib.sha256(payload).hexdigest()
        metadata["shape"] = list(array.shape)
        metadata["elements"] = int(array.size)
        metadata["nbytes"] = int(array.size * 8)
        _resign(manifest)

    invalid = tmp_path / "invalid.kamino"
    _rewrite_bundle(
        source,
        invalid,
        edit_manifest=replace_array,
        replace_member=(f"arrays/{name}.npy", payload),
    )
    with pytest.raises(BundleError, match=message):
        load_model_bundle(invalid)


def test_bundle_rejects_unsupported_schema_and_extra_members(tmp_path: Path) -> None:
    source = _regular_fit().save(tmp_path / "source.kamino")
    unsupported = tmp_path / "unsupported.kamino"
    _rewrite_bundle(
        source,
        unsupported,
        edit_manifest=lambda manifest: manifest.update(schema_version="999.0.0"),
    )
    with pytest.raises(BundleError, match="schema version"):
        load_model_bundle(unsupported)

    extra = tmp_path / "extra.kamino"
    _rewrite_bundle(source, extra, extra_member=("unexpected.txt", b"data"))
    with pytest.raises(BundleError, match="unsupported or missing members"):
        load_model_bundle(extra)


@pytest.mark.parametrize(
    ("manifest_payload", "message"),
    [
        (b"{", "manifest is invalid"),
        (b'{"value":NaN}', "invalid constant"),
    ],
)
def test_bundle_rejects_malformed_manifest_json(
    manifest_payload: bytes, message: str, tmp_path: Path
) -> None:
    source = _regular_fit().save(tmp_path / "source.kamino")
    invalid = tmp_path / "invalid.kamino"
    _replace_raw_member(source, invalid, "manifest.json", manifest_payload)
    with pytest.raises(BundleError, match=message):
        load_model_bundle(invalid)


def test_bundle_rejects_duplicate_json_keys_and_zip_members(tmp_path: Path) -> None:
    source = _regular_fit().save(tmp_path / "source.kamino")
    with zipfile.ZipFile(source) as archive:
        manifest = archive.read("manifest.json")
        members = {name: archive.read(name) for name in archive.namelist()}

    duplicate_json = tmp_path / "duplicate-json.kamino"
    _replace_raw_member(
        source,
        duplicate_json,
        "manifest.json",
        b'{"format":"duplicate",' + manifest[1:],
    )
    with pytest.raises(BundleError, match="duplicate key"):
        load_model_bundle(duplicate_json)

    duplicate_member = tmp_path / "duplicate-member.kamino"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(
            duplicate_member, "w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            for name, payload in members.items():
                archive.writestr(name, payload)
            archive.writestr("manifest.json", manifest)
    with pytest.raises(BundleError, match="duplicate member"):
        load_model_bundle(duplicate_member)


def test_bundle_rejects_unsafe_paths_and_checksum_corruption(tmp_path: Path) -> None:
    source = _regular_fit().save(tmp_path / "source.kamino")
    unsafe = tmp_path / "unsafe.kamino"
    _rewrite_bundle(source, unsafe, extra_member=("../escape", b"data"))
    with pytest.raises(BundleError, match="unsafe member path"):
        load_model_bundle(unsafe)

    corrupt = tmp_path / "corrupt.kamino"
    with zipfile.ZipFile(source) as archive:
        beta = bytearray(archive.read("arrays/beta.npy"))
    beta[-1] ^= 1
    _rewrite_bundle(source, corrupt, replace_member=("arrays/beta.npy", bytes(beta)))
    with pytest.raises(BundleError, match="checksum mismatch"):
        load_model_bundle(corrupt)


def test_bundle_rejects_pickle_arrays_before_model_construction(tmp_path: Path) -> None:
    source = _regular_fit().save(tmp_path / "source.kamino")
    stream = io.BytesIO()
    np.save(stream, np.array([{"payload": "not executable"}], dtype=object))
    payload = stream.getvalue()

    def use_object_array(manifest: dict[str, Any]) -> None:
        metadata = manifest["arrays"]["beta"]
        metadata["sha256"] = hashlib.sha256(payload).hexdigest()
        metadata["shape"] = [1]
        metadata["elements"] = 1
        metadata["nbytes"] = 8
        _resign(manifest)

    malicious = tmp_path / "object.kamino"
    _rewrite_bundle(
        source,
        malicious,
        edit_manifest=use_object_array,
        replace_member=("arrays/beta.npy", payload),
    )
    with pytest.raises(BundleError, match="Object arrays cannot be loaded"):
        load_model_bundle(malicious)


def test_bundle_rejects_inconsistent_covariance_identity(tmp_path: Path) -> None:
    source = _regular_fit().save(tmp_path / "source.kamino")
    stream = io.BytesIO()
    np.save(stream, np.array([[-1.0]], dtype=np.float64), allow_pickle=False)
    payload = stream.getvalue()

    def replace_covariance(manifest: dict[str, Any]) -> None:
        metadata = manifest["arrays"]["random_covariance"]
        metadata["sha256"] = hashlib.sha256(payload).hexdigest()
        _resign(manifest)

    invalid = tmp_path / "covariance.kamino"
    _rewrite_bundle(
        source,
        invalid,
        edit_manifest=replace_covariance,
        replace_member=("arrays/random_covariance.npy", payload),
    )
    with pytest.raises(BundleError, match="random variance and covariance"):
        load_model_bundle(invalid)


def test_bundle_enforces_overwrite_atomicity_and_resource_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fitted = _regular_fit()
    target = fitted.save(tmp_path / "model.kamino")
    original = target.read_bytes()
    with pytest.raises(BundleError, match="already exists"):
        fitted.save(target)

    replacement = tmp_path / "replacement.kamino"
    replacement.write_bytes(b"old artifact")
    fitted.save(replacement, overwrite=True)
    _assert_retained_state(fitted, load_model_bundle(replacement))

    with pytest.raises(BundleError, match="element limit"):
        bundle_module.save_model_bundle(
            fitted,
            tmp_path / "small.kamino",
            limits=BundleLimits(maximum_array_elements=1),
        )
    tiny_limits = BundleLimits(
        maximum_bundle_bytes=1,
        maximum_manifest_bytes=1,
        maximum_member_bytes=1,
        maximum_array_elements=1,
    )
    with pytest.raises(BundleError, match="file-size limit"):
        load_model_bundle(target, limits=tiny_limits)

    def fail_replace(_source: Path, _target: Path) -> None:
        raise OSError("synthetic publication failure")

    monkeypatch.setattr(bundle_module.os, "replace", fail_replace)
    with pytest.raises(BundleError, match="publication failed"):
        fitted.save(target, overwrite=True)
    assert target.read_bytes() == original
    assert list(tmp_path.glob(".model.kamino.*.tmp")) == []


@pytest.mark.parametrize(
    "arguments",
    [
        {"maximum_bundle_bytes": 0},
        {"maximum_manifest_bytes": 2, "maximum_member_bytes": 1},
        {"maximum_member_bytes": 2, "maximum_bundle_bytes": 1},
    ],
)
def test_bundle_limits_fail_closed(arguments: dict[str, int]) -> None:
    with pytest.raises(BundleError, match="limit"):
        BundleLimits(**arguments)
