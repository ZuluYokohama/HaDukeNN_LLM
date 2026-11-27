# AGENTS.md — HaDukeNN_LLM (`v0.8.0` Antigravity Workspace Specification)

This repository is configured for autonomous local development in **Google Antigravity** (`Antigravity App`, `Antigravity IDE`, and `Antigravity CLI` `agy`). Read and honor all invariants below on every turn before modifying code, documentation, or iteration artifacts.

## 1. Repository Architecture & Dual-Package Boundary

- **`src/sheaf_verify/` (`sheaf-verify` CLI & verification harness — PURE PYTHON 3.11+ STDLIB)**:
  - Zero runtime dependencies (`dependencies = []` in `pyproject.toml`). Never import `numpy`, `scipy`, `torch`, `pytest`, or any third-party package inside `src/sheaf_verify/`.
  - Owns the training (G/H/P) and HAL control graphs (`gates.py`, `control_graph.py`), 33 reviewer checklist objections (`checklist.py`), hash-chained sealed ledger (`ledger.py`, `seal.py`, `verify.py`), cellular sheaf cohomology & fault isolation (`sheaf_cohomology.py`), SAVI empirical-Bayes inference (`savi.py`), regulatory [IETF SCITT RFC 9943](https://www.rfc-editor.org/info/rfc9943/) + EU AI Act Art. 12 + FINRA 3110 WORM bundles (`compliance.py`), zero-black-box CTW-PY/Q-nullspace engine (`glassbox.py`), campaign verifiers (`campaign.py`), mutation adequacy (`mutation.py`), and the zero-dependency universal verifier (`selftest.py`).
- **`src/hadukenn_core/` (Core Neuro-Symbolic, Hardware & Sheaf-Bridge Stack — Optional `numpy`/`scipy`/`torch`)**:
  - Owns byte-level structural tokenization (`tokenization.py`), hypergraph incidence & Chodrow configuration nulls (`hypergraph.py`), spectral & paradigmatic partitions (`partitions.py`), Cluster-IS & dense heads (`heads.py`, `fast_ops.py` C11 AVX-512), 12 GiB RTX 5070 & 13 GiB looped INT8 NPU runtime (`rtx5070_frontier.py`), incremental O(1) multiset hypergraph bridge (`exec_bridge.py`), T-43 All-Realms suite (`all_realms_suite.py`), T-44 Glass-Box benchmark (`glassbox_benchmark.py`), and the T-45 **Inverted Oracle-to-Small-LLM Relational Sheaf Bridge** (`sheaf_bridge.py`).
- **`iterations/exec-hypergraph-v01/` (Merged [PR #9](https://github.com/ZuluYokohama/HaDukeNN_LLM/pull/9) Executable Hypergraph Baseline)**:
  - Standalone 71-test Python 3.11+ stdlib package (`hadukenn_exec`) implementing causal byte-to-span representations, typed relational execution, scoped counterfactual branching, revision invalidation, and semantic replay.

## 2. Non-Negotiable Gates & Hard Invariants

1. **Canonical Fixture Anchor**:
   - `sheaf-verify anchor --store fixtures/passing` must always output:
     `sheaf-verify.anchor.v1 e247484042475d381eb454df95b718f55739a6cd0385006f142be257b8b86567`
2. **All Control-Graph Gates & Coverage Floor**:
   - `src/sheaf_verify` statement coverage must stay **`>= 85.0%`**.
   - Every gate in `all_gates()` (G/H/P and HAL0–HAL6) must record both a `pass` trace and an own-criteria `fail` trace under `SHEAF_GATE_TRACE`.
   - `sheaf-verify mutation-score` must achieve **`12/12` killed mutants (`mutation_score == 1.0`)**.
3. **Lint, Documentation Budget & Export Hygiene**:
   - Zero Ruff `E`, `F`, `I`, `UP` lint errors (`line-length = 120`, `E501` ignored).
   - `docs/` total size `< 1,000,000` bytes, `< 300` KiB per HTML page, and zero external `<script src="...">` URLs (`.github/scripts/docs_budget.py`).
   - Never commit model weights/binaries (`.pt`, `.pth`, `.ckpt`, `.safetensors`, `.onnx`, `.bin`), private credentials, MIT license grant text (`LICENSE` is PolyForm Noncommercial 1.0.0; commercial terms live in `NOTICE` and `COMMERCIAL_LICENSING.md`), Windows drive-letter backslash paths, or private workstation hostnames (`tests/test_docs.py`).
   - `main` is protected (`CONTRIBUTING.md`): deliver all changes on feature/iteration branches via pull request using `.github/PULL_REQUEST_TEMPLATE.md`.

## 3. Empirical Ground Truth (Do Not Re-Propose Falsified Paths)

- **Proven Neural Winners**:
  - **Input byte-to-span tokenization (`struct32k`)**: Passed `16/16` scaling cells (`-0.0337` BPB at `M-1.5x`, `p = 5.1e-5`), with gain growing with model scale (`dlog2k = +0.1246`).
  - **Dimension-free Oracle-to-Small-LLM Sheaf Bridge (`sheaf_bridge.py`, `T-45`)**: Whitened spectral warm-start + sparse LLE barycentric & Kruskal H0 MST Laplacian (`L_bridge`) + sheaf-filtered context register stalks keep a **100% dense flat output head (`~41 TFLOPS`)** with `<= 0.01` BPB symmetric INT8 drift.
  - **Active sheaf KV-prefix registers (`U3`) & hyperedge-scoped kNN-LM (`E2`)**: Evaluated on the decontaminated `27,902`-doc / `125,572`-edge `wt103-v3` hypergraph against degree-matched [Chodrow (2020)](https://arxiv.org/abs/1902.09302) nulls.
- **Retired / Scoped Paths**:
  - **Do NOT use `182x181` [Goodman class-factored](https://arxiv.org/abs/cs/0108006) output heads for `V = 32,768` GPU pretraining**: Confirmed `KILL` in `T-38` (`+0.0320` BPB at `XS`, `+0.0247` at `S`, `+0.0188` at `M`; `~25%` effective backbone capacity tax with identical scaling exponent `alpha_body ~ 0.233`, plus `-23.9%` Tensor-Core utilization drop). Reserve factored / Cluster-IS heads strictly for **`V >= 128k..262k`** frontier vocabularies (`T-41`) or **`batch=1` DRAM-bound 13 GiB looped INT8 edge NPUs** (`T-43`).
  - **Do NOT use passive `E1` article-order shuffling**: Tied the Chodrow null (`1.0957` vs `1.0960` `xb_bpb`).

## 4. One-Command Verification (Online or Air-Gapped Sandbox)

Run the universal verifier before and after every task (works both with `pytest`/`ruff` installed and in pure-stdlib `--sandbox` mode):

```bash
PYTHONPATH=.:src python3 -m sheaf_verify.selftest --report-out ANTIGRAVITY_VERIFICATION_REPORT.json
```

When `pytest` and `ruff` are installed locally:

```bash
ruff check .
pytest --cov=sheaf_verify --cov-report=term-missing --cov-fail-under=85
sheaf-verify verify --store fixtures/passing --seal-key-file fixtures/test_seal.key
python3 .github/scripts/docs_budget.py
```
