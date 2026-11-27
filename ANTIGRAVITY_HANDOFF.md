# Antigravity Local Builder Handoff Guide (`HaDukeNN_LLM` `v0.8.0`)

This package is a clean-room, self-verifying **Google Antigravity** workspace bundle ready to open directly in **Antigravity 2.0 App**, **Antigravity IDE**, or **Antigravity CLI (`agy`)**—supporting both cloud-assisted models and offline local models (LiteRT / local GPU / looped NPU workflows).

## 1. Included Antigravity Control Plane

| Path | Antigravity Surface | Purpose |
| :--- | :--- | :--- |
| `AGENTS.md` | Always-on root context | Defines package boundaries, all 22 control-graph gates, export-hygiene bans, empirical `T-37`/`T-38` laws, and verification commands. |
| `.agents/rules/00-repository-invariants.md` | `always_on` rule | Protects `fixtures/passing` anchor `307711a4...bd7c4`, PolyForm Noncommercial + commercial dual-licensing, and PR-only `main` hygiene. |
| `.agents/rules/10-sheaf-verify-stdlib.md` | `glob` (`src/sheaf_verify/**`, `tests/**`) | Enforces zero third-party runtime imports inside `sheaf_verify`, `>= 85%` statement coverage, and `12/12` mutant kill adequacy. |
| `.agents/rules/20-empirical-architecture-guardrails.md` | `always_on` rule | Steers local builders away from retired `32k` class-factored heads and passive `E1` packing toward `struct32k`, `Sheaf-Bridge` (`T-45`), active sheaf registers (`U3`), `muP` + sampled negatives, and `SCITT` compliance. |
| `.agents/skills/verify-all-gates/` | On-demand skill (`/verify-all-gates`) | Executes all 8 repository gates (`selftest.py`) and writes `ANTIGRAVITY_VERIFICATION_REPORT.json`. |
| `.agents/skills/run-sheaf-bridge/` | On-demand skill (`/run-sheaf-bridge`) | Runs the 5-seed `T-45` Inverted Oracle-to-Small-LLM Relational Sheaf Bridge benchmark and verifies its `SCITT RFC 9943` bundle. |
| `.agents/skills/prepare-iteration-pr/` | On-demand skill (`/prepare-iteration-pr`) | Audits git status, export hygiene, `docs_budget.py`, and `.github/PULL_REQUEST_TEMPLATE.md` fields before opening a PR on `ZuluYokohama/HaDukeNN_LLM`. |
| `.agents/agents/gate-auditor/agent.md` | Custom subagent | Read-only verification & reviewer-objection auditor subagent. |
| `.agents/agents/sheaf-bridge-builder/agent.md` | Custom subagent | Neuro-symbolic & small-LLM Sheaf-Bridge implementation subagent. |

## 2. Quickstart Bootstrap on Your Local Workstation

1. **Unpack into a dedicated git worktree or clean project folder** (keep separate from any active training working tree while `T-38` finishes):
   ```bash
   unzip HaDukeNN_LLM-v0.8.0-antigravity-handoff.zip -d HaDukeNN_LLM-v0.8.0
   cd HaDukeNN_LLM-v0.8.0
   ```
2. **Run the zero-dependency 8-gate verification suite immediately** (works out-of-the-box on Python 3.11+ even before `pip install` or inside `agy --sandbox`):
   ```bash
   PYTHONPATH=.:src python3 -m sheaf_verify.selftest
   ```
3. **(Optional) Install editable dev + scientific stack** for full `ruff` / `pytest` / `numpy` / `scipy` execution:
   ```bash
   pip install -e ".[dev,stack]"
   ruff check .
   pytest --cov=sheaf_verify --cov-report=term-missing --cov-fail-under=85
   ```
4. **Launch in Google Antigravity**:
   - **Antigravity App / IDE**: Open the `HaDukeNN_LLM-v0.8.0` folder as a Workspace Project; verify under **Customizations -> Skills & Rules** that `AGENTS.md`, the 3 `.agents/rules` modules, and the 3 `.agents/skills` packages appear automatically.
   - **Antigravity CLI**: Run `agy` (or `agy --sandbox`) from the project root and invoke `/verify-all-gates` or `/run-sheaf-bridge`.

## 3. Ready-to-Dispatch Antigravity Local Builder Objectives

Once `HaDukeNN_LLM-v0.8.0` is open in Antigravity, dispatch these sequenced tasks to local agents:

1. **Task 1 — Stack `v0.8.0` PR Atop Merged [PR #9](https://github.com/ZuluYokohama/HaDukeNN_LLM/pull/9)**:
   - Prompt: *"Use `/prepare-iteration-pr` to branch `iter/v080-sheaf-bridge-and-compliance` off `origin/main` (including merged PR #9), layer `src/sheaf_verify/{compliance,glassbox,selftest}.py`, `src/hadukenn_core/{all_realms_suite,glassbox_benchmark,sheaf_bridge}.py`, `COMMERCIAL_LICENSING.md`, and `.agents/`, verify all gates via `/verify-all-gates`, and generate the PR body matching `.github/PULL_REQUEST_TEMPLATE.md`."*
2. **Task 2 — Offline 8B Oracle `struct32k` Barycentric Cache Extractor**:
   - Prompt: *"Extend `src/hadukenn_core/sheaf_bridge.py` with a streaming chunked CLI exporter that reads precomputed `struct32k` token embeddings (`[32768, D_oracle]`), writes the whitened SVD warm-start + sparse CSR `(I - W_lle)^T(I - W_lle) + gamma * L_mst` matrix (`<= 8 MiB`), and registers its SHA-256 manifest via `sheaf-verify register`."*
3. **Task 3 — Post-`T-38` Integration of `T-38` 5-Rung Scaling & FLOP-Control Records**:
   - Prompt: *"Once `T-38` Stage 3 (`L`) and the 3-seed horizon-matched FLOP control (`25f13f2d`) finish, ingest the final 5-rung `N_body` scaling JSON into `iterations/t38-design/`, run `sheaf-verify campaign --mode t38-prereg`, and update `iterations/INDEX.md` while keeping all `test_docs.py` G1 invariants intact."*
