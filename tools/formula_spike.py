"""Compare candidate formula backends against the pinned Phase 0 R matrices."""

# pyright: reportMissingTypeStubs=false, reportUnknownVariableType=false
# pyright: reportUnknownLambdaType=false, reportUnknownMemberType=false

from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import re
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from formulae import design_matrices
from formulaic import model_matrix

ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "oracle" / "fixtures" / "v1" / "fixed_theta.json"
FORMULA_CONTRACT = ROOT / "oracle" / "fixtures" / "v1" / "formula_contract.json"


def package_version(name: str) -> str:
    return importlib.metadata.version(name)


def formulae_random_labels(group_matrix: Any) -> list[str]:
    labels: list[str] = []
    for name, term in group_matrix.terms.items():
        variable = "(Intercept)" if term.kind == "intercept" else name.split("|")[0]
        labels.extend(f"g[{group}]:{variable}" for group in term.groups)
    return labels


def matrix_sha256(matrix: np.ndarray) -> str:
    canonical = np.array(matrix, dtype="<f8", order="F", copy=True)
    canonical[canonical == 0.0] = 0.0
    return hashlib.sha256(canonical.tobytes(order="F")).hexdigest()


def random_key(grouping: str, level: str, column: str) -> tuple[Any, ...]:
    assignments = tuple(sorted(zip(grouping.split(":"), level.split(":"), strict=True)))
    return assignments, column


def reference_random_keys(case: dict[str, Any]) -> list[tuple[Any, ...]]:
    keys: list[tuple[Any, ...]] = []
    for term in case["random_terms"]:
        for level in term["levels"]:
            for column in term["columns"]:
                keys.append(random_key(term["grouping"], level, column))
    return keys


def formulae_random_keys(group_matrix: Any) -> list[tuple[Any, ...]]:
    keys: list[tuple[Any, ...]] = []
    for name, term in group_matrix.terms.items():
        column = "(Intercept)" if term.kind == "intercept" else name.split("|")[0]
        grouping = name.split("|", maxsplit=1)[1]
        keys.extend(random_key(grouping, level, column) for level in term.groups)
    return keys


def formulae_fixed_labels(common_matrix: Any) -> list[str]:
    labels: list[str] = []
    for term in common_matrix.terms.values():
        for label in term.labels:
            if label == "Intercept":
                labels.append("(Intercept)")
            else:
                labels.append(label.replace("[", "").replace("]", ""))
    return labels


def caught_error(function: Any) -> dict[str, str] | None:
    try:
        function()
    except Exception as error:  # noqa: BLE001 - evidence records vendor failures.
        message = re.sub(r"\x1b\[[0-9;]*m", "", str(error))
        return {"type": type(error).__name__, "message": message}
    return None


def main() -> None:
    fixture: dict[str, Any] = json.loads(FIXTURE.read_text(encoding="utf-8"))
    contract: dict[str, Any] = json.loads(FORMULA_CONTRACT.read_text(encoding="utf-8"))
    source = fixture["data"]
    frame = pd.DataFrame(
        {
            "y": source["y"],
            "x": [row[1] for row in source["X"]],
            "o": source["offset"],
            "g": pd.Categorical(
                ["beta"] * 4 + ["alpha"] * 4 + ["gamma"] * 4,
                categories=["gamma", "alpha", "beta"],
                ordered=True,
            ),
            "f": pd.Categorical(
                [
                    "middle",
                    "low",
                    "high",
                    "low",
                    "high",
                    "middle",
                    "low",
                    "high",
                    "low",
                    "high",
                    "middle",
                    "middle",
                ],
                categories=["middle", "high", "low"],
                ordered=True,
            ),
            "h": pd.Categorical(
                ["one", "two"] * 6,
                categories=["two", "one"],
                ordered=True,
            ),
        },
        index=source["row_ids"],
    )
    mixed_formula = "y ~ x + (1 + x | g)"
    matrices: Any = design_matrices(mixed_formula, frame)
    raw_offset_matrices: Any = design_matrices("y ~ x + offset(o) + (1 + x | g)", frame)
    x = np.asarray(matrices.common).astype(np.float64, copy=False)
    z_raw = np.asarray(matrices.group).astype(np.float64, copy=False)
    raw_labels = formulae_random_labels(matrices.group)
    desired_labels: list[str] = source["random_names"]
    permutation = [raw_labels.index(label) for label in desired_labels]
    z_reordered = z_raw[:, permutation]

    missing_frame = frame.copy()
    missing_frame.loc["r03", "x"] = np.nan
    vendor_output = io.StringIO()
    with redirect_stdout(vendor_output), redirect_stderr(vendor_output):
        missing: Any = design_matrices(mixed_formula, missing_frame, na_action="drop")

    new_frame = pd.DataFrame(
        {
            "y": [1.0, 2.0],
            "x": [0.25, -0.25],
            "g": ["alpha", "new-level"],
        },
        index=["known", "unknown"],
    )
    known_group: Any = matrices.group.evaluate_new_data(new_frame.iloc[[0]])
    new_level_error = caught_error(lambda: matrices.group.evaluate_new_data(new_frame))

    formulaic_error = caught_error(lambda: model_matrix(mixed_formula, frame))
    formulae_double_bar_error = caught_error(
        lambda: design_matrices("y ~ x + (1 + x || g)", frame)
    )

    corpus_results: list[dict[str, Any]] = []
    for case in contract["cases"]:
        case_frame = frame.copy()
        if missing_row := case.get("missing_x_row"):
            case_frame.loc[missing_row, "x"] = np.nan
        adapter_formula = case.get("adapter_formula", case["formula"])
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            candidate: Any = design_matrices(adapter_formula, case_frame)
        candidate_x = np.asarray(candidate.common).astype(np.float64, copy=False)
        candidate_z = np.asarray(candidate.group).astype(np.float64, copy=False)
        raw_keys = formulae_random_keys(candidate.group)
        expected_keys = reference_random_keys(case)
        candidate_z = candidate_z[:, [raw_keys.index(key) for key in expected_keys]]
        corpus_results.append(
            {
                "id": case["id"],
                "rows_exact": list(candidate.data.index)
                == case.get("retained_rows", list(frame.index)),
                "fixed_labels_exact": formulae_fixed_labels(candidate.common)
                == case["fixed_names"],
                "X_exact": matrix_sha256(candidate_x) == case["X_sha256"],
                "Z_exact_after_label_permutation": matrix_sha256(candidate_z)
                == case["Z_sha256"],
                "adapter_formula": adapter_formula,
            }
        )

    evidence: dict[str, Any] = {
        "schema_version": "1.0.0",
        "all_contract_cases_pass": all(
            result[check]
            for result in corpus_results
            for check in (
                "rows_exact",
                "fixed_labels_exact",
                "X_exact",
                "Z_exact_after_label_permutation",
            )
        ),
        "reference": fixture["reference"],
        "versions": {
            "formulae": package_version("formulae"),
            "formulaic": package_version("formulaic"),
            "pandas": package_version("pandas"),
        },
        "formulae": {
            "fixed_matrix_exact": bool(np.array_equal(x, np.asarray(source["X"]))),
            "raw_offset_formula_fixed_columns": int(
                np.asarray(raw_offset_matrices.common).shape[1]
            ),
            "offset_requires_owned_extraction": bool(
                np.asarray(raw_offset_matrices.common).shape[1] != x.shape[1]
            ),
            "random_matrix_raw_exact": bool(
                np.array_equal(z_raw, np.asarray(source["Z"]))
            ),
            "random_matrix_exact_after_label_permutation": bool(
                np.array_equal(z_reordered, np.asarray(source["Z"]))
            ),
            "raw_random_labels": raw_labels,
            "reference_random_labels": desired_labels,
            "reference_to_raw_permutation": permutation,
            "retained_rows_after_missing_x": list(missing.data.index),
            "missing_row_message": vendor_output.getvalue().strip(),
            "known_level_preserves_column_count": bool(
                np.asarray(known_group).shape[1] == z_raw.shape[1]
            ),
            "new_level_error": new_level_error,
            "double_bar_error": formulae_double_bar_error,
        },
        "formulaic": {"mixed_formula_error": formulaic_error},
        "formula_corpus": corpus_results,
        "decision": {
            "backend": "formulae-adapter",
            "status": "accepted-for-phase-1-subset",
            "reason": (
                "Formulae parses lme4-style grouped terms and reproduces the pinned "
                "X/Z values after an explicit label permutation; Formulaic rejects "
                "the grouped-term operator without an owned parser extension."
            ),
            "required_adapter_controls": [
                "construct one shared complete-case frame before matrix evaluation",
                "canonicalize random columns by persisted group-and-term labels",
                "extract offset terms before Formulae builds the fixed matrix",
                "reject or explicitly expand double-bar syntax before evaluation",
                "preserve the vendor's default rejection of new grouping levels",
                "allowlist names and transforms before invoking the vendor evaluator",
            ],
            "not_yet_claimed": [
                "categorical random effects",
                "custom and non-treatment contrasts",
                "general transform allowlist",
                "stateful prediction parity",
            ],
        },
    }
    print(json.dumps(evidence, indent=2, sort_keys=True))
    if not evidence["all_contract_cases_pass"]:
        raise SystemExit("formula contract failed")


if __name__ == "__main__":
    main()
