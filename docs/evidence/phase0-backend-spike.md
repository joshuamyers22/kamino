# Phase 0 numerical backend evidence

Local environment: macOS arm64, CPython 3.12.14, NumPy 2.5.3 with Accelerate,
SciPy 1.18.1. Timings are median of three warm in-process evaluations and are
feasibility observations, not release performance promises.

At eight groups, the loop-based block prototype matched the production dense
criterion exactly; SciPy sparse LU differed by `1.42e-14`. Maximum beta errors
were `3.55e-15` and `1.11e-15`, respectively.

| groups | n | q | dense C estimate | block prototype | sparse LU |
|---:|---:|---:|---:|---:|---:|
| 100 | 400 | 200 | 0.32 MB | 0.0038 s | 0.00034 s |
| 1,000 | 4,000 | 2,000 | 32 MB | 0.067 s | 0.00094 s |
| 5,000 | 20,000 | 10,000 | 800 MB | 0.91 s | 0.0041 s |

The structure has only three stored entries per 2-by-2 diagonal C block. Dense
allocation is therefore rejected as the Phase 1 scaling design. The Python-loop
block prototype is also too slow to ship as-is; production work must batch the
factorizations. General sparse LU remains a comparator because an SPD Cholesky
contract and cross-platform distribution evidence are unresolved.

Re-run on the current machine with `make backend-spike`.
