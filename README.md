<h1>HaDukeN //\\// HGNLLM Experiments</h1>

[![PolyForm Noncommercial 1.0.0](https://img.shields.io/badge/license-PolyForm%20Noncommercial%201.0.0-blue)](LICENSE)

Verification harness for hypergraph and sheaf language-model experiments. The Python package and the command remain `sheaf-verify`.

The harness does not train a model. A trainer, running on its own machine, registers the bytes it actually used. This repository stores manifests, a hash-chained ledger, synthetic fixtures, and the recorded G0 metadata. It does not store model weights, a corpus, tokenizer tables, or credentials.

Licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE). Commercial use requires a separate license from the author. See `NOTICE`. Contact via GitHub [@ZuluYokohama](https://github.com/ZuluYokohama).

## Public priority record

Annotated tag `v0.1.0-priority` points at the commit in this fingerprint. The tag message carries the same four lines. This section was added in the following commit, so `git archive` of the tag hashes the tree that does not yet include this section.

```
commit: 7403b03b699ab02e0cbaac15d738bb7da345e257
utc: 2026-11-23T21:55:18Z
git-archive-sha256: 64583710435b2831eb660401910205436e38aacefe379df4d7c39a5e80fc9c74
sheaf-verify-ledger-head: sheaf-verify.anchor.v1 307711a4d841c2c8fa0bbd954d15b2a585012608528c9f53b9ea08ad717bd7c4
```

Reproduce the archive hash with `git archive --format=tar v0.1.0-priority | sha256sum`. The ledger line is `sheaf-verify anchor --store fixtures/passing` on that commit.

The in-tree `fixtures/passing` ledger was later regenerated when each ledger record began carrying an explicit `profile` field (`training` or `hal`). The current fixture head is `sheaf-verify.anchor.v1 e247484042475d381eb454df95b718f55739a6cd0385006f142be257b8b86567`. The four-line `v0.1.0-priority` fingerprint above is unchanged and still describes that tag's tree.

The tag is an SSH-signed annotated tag. The key available here is `ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIPqiJsBjAsv4KymedFcUR891X1lgC90DW8yMtjcHJ/p0` (Cursor Agent). It is not a personal key of Blake Jones.

## Install

```bash
pip install -e ".[dev]"
ruff check .
pytest --cov=sheaf_verify --cov-report=term-missing --cov-fail-under=85
sheaf-verify verify --store fixtures/passing --seal-key-file fixtures/test_seal.key
SHEAF_GATE_TRACE=gate-trace.jsonl pytest
python -m sheaf_verify.gate_coverage gate-trace.jsonl --matrix gate-coverage.md
```

Python 3.11 or newer. The runtime dependencies are the standard library. pytest, pytest-cov, and ruff are the development dependencies. CI fails when line coverage of `src/sheaf_verify` is under 85%, and when any control-graph node lacks a pass test or an own-criteria fail test. `CONTRIBUTING.md` is the merge rule: iteration branches merge only by pull request, and `main` is protected.

The file `fixtures/test_seal.key` is a public test vector, `sheaf-verify-public-test-vector`. It is not a production key. A production key stays off the trainer and off this repository. After `verify` succeeds, anchor the ledger head outside the store. Anyone who holds the key can build a new chain that verifies, so the printed hash has to live where that key cannot rewrite it:

```bash
sheaf-verify anchor --store .sheaf --out ~/sheaf-anchors.txt
```

That command only reads `ledger/head.json` and, if you pass `--out`, appends one line to a file outside the store. It does not contact an external service. Printing the line is the same anchor.

## What a trainer runs

From the training checkout, with artifacts outside this store:

```bash
sheaf-verify register --store .sheaf --kind raw_data --role corpus.train --path data/train.bin \
  --metadata '{"document_ids":["doc-001"],"order":0,"split":"train"}'
sheaf-verify register --store .sheaf --kind raw_data --role corpus.valid --path data/valid.bin \
  --metadata '{"document_ids":["doc-101"],"order":1,"split":"valid"}'
sheaf-verify register --store .sheaf --kind raw_data --role corpus.test --path data/test.bin \
  --metadata '{"document_ids":["doc-201"],"order":2,"split":"test"}'

sheaf-verify register --store .sheaf --kind vocabulary --role vocabulary --path vocab.json \
  --input <train-manifest-sha256>
sheaf-verify register --store .sheaf --kind hypergraph --role hypergraph --path graph.json \
  --input <train-manifest-sha256>
sheaf-verify register --store .sheaf --kind hypergraph --role hypergraph.random --path graph-random.json \
  --input <train-manifest-sha256>

sheaf-verify register-tree --store .sheaf --role code --path .
sheaf-verify register --store .sheaf --kind kernel --role kernel --path kernels/apply.py \
  --metadata '{"backend":"cpu","entry":"apply"}'
sheaf-verify register-env --store .sheaf --python 3.12.3 --cuda 12.4 --driver 550.54.14 --gpu "NVIDIA example"
sheaf-verify register --store .sheaf --kind config --role config --path config.json
sheaf-verify register --store .sheaf --kind seeds --role seeds --path seeds.json
sheaf-verify register --store .sheaf --kind claim --role claim --path claim.json

sheaf-verify register-external --store .sheaf --role checkpoint.hypergraph \
  --payload-sha256 <sha256-of-checkpoint-bytes> --size-bytes <nbytes> \
  --locator external://trainer/checkpoints/hypergraph \
  --input <code-sha> --input <config-sha> --input <seeds-sha> --input <kernel-sha>

sheaf-verify register --store .sheaf --kind log --role log.hypergraph --path logs/hypergraph.log \
  --input <checkpoint-manifest-sha256>
sheaf-verify register --store .sheaf --kind trace --role trace.causality --path causality.json \
  --input <hypergraph-manifest-sha256>
sheaf-verify register --store .sheaf --kind metrics --role metrics --path metrics.json \
  --input <train-sha> --input <valid-sha> --input <test-sha> --input <seeds-sha> --input <vocab-sha>
sheaf-verify register --store .sheaf --kind figure --role figure --path figure.svg \
  --metadata '{"plots":"bpb","metrics_payload_sha256":"<metrics-payload-sha>"}' \
  --input <metrics-manifest-sha256>

sheaf-verify run --store .sheaf --seal-key-file ~/.sheaf-seal-key
sheaf-verify verify --store .sheaf --seal-key-file ~/.sheaf-seal-key
sheaf-verify anchor --store .sheaf --out ~/sheaf-anchors.txt
sheaf-verify bundle --store .sheaf --seal-key-file ~/.sheaf-seal-key --out reviewer-bundle
```

Repeat the checkpoint and log registration for `superword_bpe` and `randomized_hypergraph`. A CUDA, OpenCL, or C kernel must set `host_reference_sha256` to the payload hash of an input manifest that holds the host reference source. The harness records that binding. It does not launch the kernel.

`run --profile training` (the default) writes every G/H/P gate, including failures, then registers a result manifest. `run --profile hal` writes only HAL0–HAL6 for a hardware or microbench study; it cannot be used to skip the training gates. Exit 0 means every gate passed. Exit 1 means a gate failed and the ledger still holds the failure. Exit 2 means the command itself was refused.

`verify` recomputes hashes, the ledger, and gate replay. It exits 1 on a mismatch. A scientifically failed study can still verify: the failure is what the ledger says. `bundle` exits 1 unless verification is clean, every gate passed, and every checklist objection is answered.

`--strict` fails when a filesystem checkpoint path is missing. An `external://` locator stays a note either way, because the weight bytes live on the trainer.

## Layout

```
src/sheaf_verify/    library and the sheaf-verify command
tests/               pytest suite
scripts/             host drivers (live HAL run, receipted jovin CPU checks)
fixtures/passing/    synthetic study, regenerated by python -m sheaf_verify.synthetic
fixtures/g0-recorded/  recorded G0 metadata, not a corpus
iterations/          index on main; one experiment per iter/<id> branch
docs/                static briefing site for GitHub Pages, including the control graph
CONTRIBUTING.md      pull-request rule for iteration branches
schemas/             JSON Schema drafts for the stored records
```

A store contains `manifests/`, `payloads/`, `snapshots/`, and `ledger/`. Checkpoint bytes are not a payload. `.sheaf/` is the default store path and is gitignored.

Read `ARCHITECTURE.md` for the hash and ledger rules, `GATES.md` for the pass conditions, and `REVIEWER_GUIDE.md` for the objections a reviewer is expected to raise. The public briefing is `docs/index.html`.


## Receipted CPU checks

`scripts/run_jovin_cpu_checks.sh` runs a stdlib-only CPU slice of the HAL checks inside the jovin-harness sandbox (`python:3.12-slim`, no numpy, no CUDA, 256 KiB input cap, 30 s timeout ceiling) and records a hash-chained receipt. It does **not** run the host pytest suite or the GPU / HAL `.so` build; those stay on the host. When `~/hadukenn/logs/live-gated-sidecars/` is present, the claim, parity, timings, and guards documents are embedded and recomputed inside the sandbox.

```bash
./scripts/run_jovin_cpu_checks.sh
# or without live sidecars:
./scripts/run_jovin_cpu_checks.sh --no-live
```


## Live campaign evaluation

`sheaf-verify campaign` JSON-only invocations are labeled **replay**: they score a registered document and do not re-measure. For T-46, `--live` re-runs the in-process suite, records raw timing repeats, recomputes CCE against the stdlib float64 reference, samples thermal/power, and writes `iterations/t46-tri-machine-hal/T46_TRI_MACHINE_RESULTS.lapbeast-live.json` without touching the sealed `T46_TRI_MACHINE_RESULTS.json`. `--replay` forces JSON-only. `--live` is also implemented for the T-49 HST *toy* campaign (48,000 training bytes, typically about a minute); it is opt-in, not the default. Other campaign modes stay replay-only.

```bash
sheaf-verify campaign --mode t46-tri-machine-hal \
  --path iterations/t46-tri-machine-hal/T46_TRI_MACHINE_RESULTS.json --replay --json
sheaf-verify campaign --mode t46-tri-machine-hal \
  --path iterations/t46-tri-machine-hal/T46_TRI_MACHINE_RESULTS.json --live --json
```

The v1.2.0 drop ships `src/sheaf_verify/hst_standard.py` and a T-49 briefing. The package version in `pyproject.toml` is `1.0.0`. Sealed confirmation hashes and the sealed T-46 JSON are unchanged.

The official claim status for that briefing is [`docs/HST_V1_CLAIMS_AUDIT.md`](docs/HST_V1_CLAIMS_AUDIT.md) and the corrected copy [`docs/HST-v1-briefing.html`](docs/HST-v1-briefing.html). The T-49 5-seed BPB result (`p = 4.81e-4`, 20% fewer optimizer steps) is a **48,000-byte synthetic toy campaign**, reproduced on this host, not a large-corpus pretrain. The briefing’s “221/221 checks”, “20–35% compute left on the table”, “125M–1.3B sweet spot”, and “regulated-domain readiness” statements are hypotheses or contradicted counts, not measured facts. The T-37 struct32k `-0.0337` BPB figure remains a recorded `iterations/t37-grid/` result; it was not re-run for this audit.

## Iterations

`main` holds the harness and `iterations/INDEX.md`. An experiment does not land on `main`. Cut a branch named `iter/<id>` from `main` and add:

- `iterations/<id>/BRIEFING.md` — an OODA block (Observe, Orient, Decide, Act), then the op-order sections Situation, Mission, Execution, Sustainment, and Command & Signal
- `iterations/<id>/MODE_METHOD.md` — the mode, the method, the controls, and the gates that iteration is allowed to speak to
- `iterations/<id>/results/` — records from that iteration

`iterations/TEMPLATE/` is the blank form. It is not an iteration. Copy it onto the branch. Do not put weights, a corpus, a tokenizer merge table, a credential, or a machine path in `results/`.

The feasibility counts are iteration `g0` on branch `iter/g0`. That branch is listed from `iterations/INDEX.md`. The files for it are on the branch, not on `main`.

## Briefing site

`docs/` is a static site: `index.html` is the overview op-order, `briefing.html` is the detailed briefing. There is no build step and no JavaScript. GitHub Pages can publish the `docs/` folder on the default branch.

## Recorded results

WikiText-103 test bits per byte. Lower is better. These are measured campaign numbers. They are not the synthetic `fixtures/passing` ledger, and that ledger is unchanged.

- G0 is partial. The recorded valid-split counts meet the unit, fallback, CPU-hour, and RAM limits. V=65,536 is over the written cap of 64,000.
- G1 failed. hnt64k test BPB 1.452254 is above struct32k plus one standard deviation (1.329303). Train FLOPs/byte 3.5706e7 is above struct32k and sw64k.
- G1b failed on BPB. Structured hnt40k test BPB 1.477304 is above 1.329303. Train FLOPs/byte 1.1929e7 is about 61% below struct32k, so only the FLOPs side moved.
- G1c passed as a compute-matched comparison, n=3 seeds. hnt40k-struct-v2 is 1.269376 ± 0.004417 at 1.2061e7 train FLOPs per byte and 4.2006e15 train FLOPs. The sw40k-struct-v2 control is 1.322698 ± 0.006710. struct32k is 1.322354. The GPT-2 tokenizer arm is 1.328219. hnt versus that control is -0.0533 (p 7.1e-4). The hnt mean is under the G1 line of 1.329303. This is a compute-efficiency win, not a per-byte modeling win.
- g1d is the equal-data follow-up. struct32k trained on 348.3M bytes, the same train bytes hnt40k-struct-v2 saw, scored 1.194155 ± 0.000824 at 1.0756e16 train FLOPs. That beats hnt by 0.075 at equal data.
- g1g, hnt40k-struct-v2 FLOPs-matched to g1d at 1.0756e16 train FLOPs, scored 1.174315 ± 0.000564 (seeds 1.174805, 1.173698, 1.174444). That is 0.019839 below g1d (Welch p 1.3e-5), That is a win at equal analytic train FLOPs only (the 6N + attention + head count). The structured model swaps the flat 40k output softmax for a factored cluster-then-unit head (200 clusters × 200 units). At d=384 the flat head is about 56.6% of plain-model FLOPs per token, so struct-v2 counts about 2.22× fewer FLOPs per token. At equal FLOPs, g1g therefore saw 891.7M train bytes (about 1.65 epochs, against g1d's 0.64), and it took about 1,488 s per seed against about 501–505 s for g1d and g1h, so it is about 3× slower on wall clock.
- g1h, plain hnt40k with the structured path off (struct none, mean compose, low-rank residual, no spectral init) at the same 1.0756e16 train FLOPs, scored 1.197637 ± 0.000517 (seeds 1.198146, 1.197111, 1.197653). That is 0.023321 above g1g (p 8.4e-7) and 0.003482 above g1d (p 0.006): at equal analytic FLOPs and the same tokenizer, the structured run scores lower, but it also saw about 2.2× the data. At equal data the order reverses. G1c, the same hnt40k-struct-v2 model at 348.3M train bytes, is 1.269376 ± 0.004417. That is 0.0579 worse than plain hnt40k at the same bytes (g1i, Welch p 8.8e-5) and 0.0752 worse than struct32k (g1d, p 8.2e-4). The mechanism resembles a class-factored softmax (Goodman 2001, arXiv:cs/0108006). This is not evidence that the structure layer models text better. Wall-clock-matched and 2× data-ladder controls are pending.
- g1i, the same plain hnt40k at g1d's 348.35M train bytes (9.350e15 train FLOPs), scored 1.211442 ± 0.003395 (seeds 1.210431, 1.215228, 1.208666). That is 0.017287 above g1d (p 0.0094), so at equal bytes plain hnt40k loses to struct32k. g1i and g1d use different tokenizers, so this is a tokenizer contrast, not a structure ablation.
- A3, the random-cluster control (hnt40k with random clusters, FLOPs-matched), finished as g1j: 1.176685 ± 0.002032 (n=3). That is 0.002369 above g1g (Welch two-sided p 0.17). It is not a clean contrast, because spectral init was on in g1g and off in g1j. Untested.

### Since g1i (Nov 24–25, 2026)

- **T-37 tokenizer grid** (`iterations/t37-grid/`): FineWeb-Edu, flat heads, fwedu-hnt40k against fwedu-struct32k at 2 sizes × 4 byte budgets, n=3. struct32k is better in all 8 contrasts, and Holm (α 0.01) keeps 5 of 8. The edge shrinks with more data (log2 k slope −0.232 at S, −0.137 at M; both CIs below 0) and is larger at M than at S (Δlog2 k +0.125, 95% CI +0.080 to +0.173). At M-1.5× the gap is 0.0337 (1.323749 against 1.290067, one-sided p 5.1e-5). These are tokenizer effects, not structured-head effects.
- **T-38 design** (`iterations/t38-design/`): frozen pre-registration (SHA-256 `524fca093dc6c00e803b6f9998e69277f2804cc38e44ad71f94ffd0b88d63590`) for plain against the structure component on the same 32k BPE tokenizer, plus a paired-seed addendum (reported secondary only). No T-38 result exists yet.
- **E1 packing control** (`iterations/hg-e1-pack-random/`): random article order, E1 held-out-article BPB 1.093936 ± 0.003312 (n=3). It has no pass/fail of its own.
- **Ledger** (`iterations/LEDGER_since_g1i.csv`): every iteration recorded after g1i, numbers only, with no verdict labels.

Caveats for G1c, g1d, g1g, g1h, and g1i: n=3, one model size, WikiText-103 only. Parameter totals are verified against the per-seed trainer records (the trainer itself is not in this repository): hnt40k-struct-v2 39,943,040, struct32k 23,204,736, and plain hnt40k 25,981,824, each with 10,621,824 non-embedding. Every run applied RoPE. The trainer's structured model path applies RoPE only and ignores `--pe heat`, so no heat positional encoding has been tested. G1c does not reverse G1 or G1b. There is no `iter/g1c` branch. The g1d, g1g, g1h, and g1i records are folders under `iterations/` on `main`.

## Gaps

`GATES.md` now follows the architecture spec and the paper plan. G0 is the valid-split unit reduction (15% fewer than GPT-2 BPE, byte fallback at most 1%, vocabulary at most 64,000, build at most 2 CPU-hours and 16 GB). The feasibility build's `V = 65,536` is flagged against that cap in `fixtures/g0-recorded/`; that file is recorded metadata, not data. G1 is hnt64k test BPB_corpus at or below struct32k plus one standard deviation, with lower train FLOPs per byte than struct32k and the equal-size sw64k control. H0–H7 are the spec stages, and P0–P6 are the paper-plan phases. Pilots use `PILOT_SEEDS = 3`. The headline uses `HEADLINE_SEEDS = 5`. A single `MIN_SEEDS = 5` is not applied to G1.

The seal is still HMAC-SHA256, not an Ed25519 signature. `sheaf-verify anchor` prints the ledger head so it can be copied out of band. It does not add a timestamp service.

The causality trace and the per-seed BPB are checks of submitted records. A trainer can still submit a trace that was not produced by the named code. An independent rerun is still required. The git commit id, when present, is recorded and not recomputed unless that checkout is still present. The git tree SHA-1 and the SHA-256 listing are what `verify` recomputes from the snapshot.

## v0.2.0 Verification Upgrades

`sheaf-verify` v0.2.0 keeps zero runtime dependencies and preserves `fixtures/passing` byte-for-byte while adding five verification subsystems:

1. **Cellular Sheaf Cohomology & Grounded Mapping-Cone (`sheaf_cohomology.py`, `sheaf-verify sheaf-check`):** Exact rational coboundary rank (`dim H^0`, `dim H^1`), Hodge Laplacian spectral gap, Yokoyama grounded mapping-cone energy against external reference anchors, O(1)-in-n partitioned local updates, and 1-vs-1 symmetric tie abstention (`ABSTAIN_TIE`) versus recoverable multi-witness fault localization (`RECOVERABLE_FAULT`).
2. **Certified Interval/Rational Numerics & Precision Envelopes (`numerics.py`):** Exact `Fraction` + outward IEEE-754 `nextafter` `CertifiedBall` enclosures, Higham tree-reduction drift envelopes across `fp64` through `bf16`, `mxfp8`, and `nvfp4`, RL learner-sampler precision parity gates, and Tensor-Core hardware roofline checks.
3. **Sequential Anytime-Valid Inference & Small-Sample Power (`savi.py`):** Predictable plug-in Kelly-betting e-processes under Ville's inequality, e-BH FDR control under arbitrary dependence, Smyth Empirical-Bayes tier-pooled paired-seed t-tests alongside Welch, and Hoeffding-Bentkus Learn-Then-Test (LTT) conformal threshold calibration.
4. **Standards-Native Attestation, Merkle Receipts & Anti-Cheat (`attestation.py`, `anticheat.py`):** Strict RFC 7493/8785 JSON validation, in-toto Statement v1 wrapped in DSSE v1.0 PAE (`sheaf-verify attest` / `verify-attestation`), RFC 9162 binary Merkle inclusion receipts (`sheaf-verify receipt` / `verify-receipt`), and AST static kernel security auditing (`sheaf-verify audit-kernel`).
5. **Campaign Protocol Audit & Mutation Adequacy (`campaign.py`, `mutation.py`):** Domain of Calibration (`PROGRESSIVE`, `STAGNANT`, `IMPROPER`) transition checks, `256 × 128` Tensor-Core cluster geometry verification, independent T-37 / T-38 / `hg-v3` audits (`sheaf-verify campaign`), and a 12-mutant scientific and cryptographic adequacy suite (`sheaf-verify mutation-score`).

## v0.3.0 & v0.4.0 Frontier-5070 Stack (`src/hadukenn_core/`)

Optional package extra `pip install -e ".[stack]"` (`numpy>=2.0`, `scipy>=1.11`, keeping `sheaf-verify` core `dependencies = []` pure stdlib) provides the executable hypergraph-sheaf LLM stack and single-GPU **12 GB RTX 5070 (`Blackwell sm_120`)** engine:

1. **Micro-Sliced `2048 × 128` (`V = 262,144`) `Cluster-IS` Output Head (`heads.py`, `csrc/fast_head_avx512.c`, `torch_bridge.py`):** Eliminates both the `8.0 GiB` flat BF16 logit+grad wall (`102.4x` buffer reduction to `0.0781 GiB` at 8k tokens, `174.68x` native C11 AVX-512 speedup over `Tiled-CCE`) and the `16.0 GiB` 3D `torch.bmm` gather trap via `chunk_tokens = 256` micro-slicing (`ParadigmaticClusterISHeadTorch`).
2. **12 GB RTX 5070 Quantized Training & 8B Checkpoint Surgery (`rtx5070_frontier.py`):** Combines `Blockwise8BitAdamW` (`3.857x` moment compression), `NF4BlockQuantizer` / `NF4FrozenLinearTorch` (`4.1319 bits/param`), and norm-preserving zero-corpus SVD `factorize_pretrained_lm_head` (`-0.2703` step-0 BPB, `p = 1.0e-6`; `-0.1887` post-NF4 `Cluster-IS` BPB, `p = 5.4e-5` vs. random across 5 seeds), fitting `1.1B` from-scratch (`6.30 GiB`) and `8B` `262k`-vocab surgery (`8.78 GiB`, `3.22 GiB` headroom, `43.48 MiB <= 48.0 MiB` L2 cache) inside 12.0 GiB GDDR7.
3. **RoPE-Decoupled Biorthogonal Sheaf `KV-Prefix` + Symmetric `LP-FT` (`transformer_lm.py`, `rtx5070_frontier.py`):** Decouples positionless sheaf content addressing (`q_unrot @ k_mem^T + mem_score_bias`) from within-window `RoPE` attention across non-hapax consensus hyperedge stalks (`0.0` bitwise self-inclusion drift), beating the degree/size-matched Chodrow configuration null across all 5 seeds (`-0.0162` domain BPB, `p = 4.7e-5`) and window-isolated NF4 (`-0.0253` domain BPB, `p = 2.0e-5`; `-0.0044` all-token BPB). Verify sealed records via `sheaf-verify campaign --mode t41-frontier5070 --path iterations/t41-frontier-5070/T41_FRONTIER5070_RESULTS.json`.

## v0.5.0 Unified Neuro-Symbolic Executable Hypergraph (`exec-hypergraph-v01` + `src/hadukenn_core/exec_bridge.py`)

`v0.5.0` integrates the 41-file executable relational hypergraph prototype (`iterations/exec-hypergraph-v01/`, `tests/test_executable_iteration.py`) with the `hadukenn_core` neural-sheaf stack and `sheaf-verify`:

1. **$O(1)$ Homomorphic Multiset State Commitments (`IncrementalHypergraphKernel`):** Replaces per-command $O(N \log N)$ full-dictionary JSON hashing (`151.49 ms` at depth `128`, `551.61 ms` at depth `256`) with a 256-bit (`4 × uint64`) `BLAKE2b` multiset accumulator plus domain-separated `SHA-256` finalization and dirty-flag read caching ([Clarke et al.](https://www.researchgate.net/publication/2481704_Incremental_Multiset_Hash_Functions_and_Their_Application_to_Memory_Integrity_Checking); [Solana SIMD-0215](https://github.com/solana-foundation/solana-improvement-documents/blob/main/proposals/0215-accounts-lattice-hash.md)). Preserves `66/66` scenarios, `5,992/5,992` outcomes, `66/66` semantic replays, and `8/8` intervention pairs while reaching **`19.57 ms` (`7.74x` speedup) at depth `128`** and **`41.53 ms` (`13.28x` speedup; `8.51x` faster than an equally-audited `AuditedCachedDAG`) at depth `256`**.
2. **Causal `span_graph <-> OESpan` Alignment & Cellular Sheaf Pre-Gate:** Connects `hadukenn.span_graph.v1` byte/span nodes to `OESpan` BPE tokens with `0.0` unseen-suffix drift (`306/306` tokens anchored) and gates conflicting sources via `sheaf_cohomology.py` (`ABSTAIN_TIE` on `1-vs-1` ties; `RECOVERABLE_FAULT` isolating `src_corrupt` under `2-vs-1` witness consensus before executing verified exact conservation arithmetic).
3. **5-Seed Learned Policy on Held-Out OOD Multi-Step Compositions (`T-42`):** Under matched beam budget (`K = 4`) across 5 seeds (`13, 21, 34, 55, 89`), training on `depth 2..3` episodes and evaluating on `100` held-out `depth 4..6` OOD episodes with unseen entities, unit-incompatible distractors, and counterfactual `Suppose instead` forks yields **`100.0%` OOD exact accuracy (`100.0%` counterfactual branch accuracy)** for `hypergraph_sheaf_controller`, versus **`15.2%`** for `flat_seq_controller_dag` (`delta_error = -0.8480`, Smyth EB `t = -127.84`, `p < 1e-12`, SAVI Kelly `E = 6.1029`) and **`4.4%`** for `direct_neural_blackbox`. Verify via `sheaf-verify campaign --mode t42-exec-bridge --path iterations/t42-unified-neuro-symbolic/T42_UNIFIED_EXEC_RESULTS.json`.

## v0.6.0 All-Realms Competitive Suite (`src/sheaf_verify/compliance.py` + `src/hadukenn_core/all_realms_suite.py`)

`v0.6.0` extends the zero-dependency `sheaf-verify` core and `hadukenn_core` across four deployment realms (`T-43`, 5 seeds `13, 21, 34, 55, 89`, commercial licensing details in `COMMERCIAL_LICENSING.md`):

1. **Pure-Stdlib Six-Verdict Action Ontology & Regulatory Bundles (`compliance.py`):** Adds `ExactUnitLattice` (`Fraction` scaling across volume, currency, mass, time, energy, and ratios), `LocalActionOntology` (`VALID`, `INVALID`, `IMPOSSIBLE`, `TRANSLATION_AMBIGUOUS`, `ABSTAIN_TIE`, `RECOVERABLE_FAULT`), and `emit_regulatory_bundle` / `verify_regulatory_bundle` covering [IETF SCITT RFC 9943](https://www.rfc-editor.org/info/rfc9943/), [RFC 9162](https://www.rfc-editor.org/info/rfc9162/), [DSSE v1.0](https://github.com/secure-systems-lab/dsse), [EU AI Act Article 12](https://artificialintelligenceact.com/article-12-record-keeping/), and [FINRA Rule 3110](https://www.finra.org/rules-guidance/guidance/reports/2026-finra-annual-regulatory-oversight-report/gen-ai).
2. **Noisy Multi-Vertical NL-to-Hypergraph Compiler (`all_realms_suite.py` Realm A):** Compiles messy mixed-unit prose across finance, clinical, energy, and supply-chain episodes with **`100.0%` 5-seed exact accuracy and `0.0%` silent conflict corruption**, outperforming post-hoc single-turn SMT filtering (`40.0%`, `E = 6.3281`) and mutable state-dict routing (`0.0%`).
3. **Drop-In `VerifiableGraphCheckpointer` & Heterogeneous Edge Silicon (Realms B & C):** Executes 200 multi-branch supersteps in `15.88 ms` (`79.41 us/step`) catching `4/4` tamper and conservation attacks, while `QuantizedINT8NPULoopEngine` (`Snapdragon X1P-42-100` `13.0 GiB` LPDDR5x @ `14.5 W`) matches FP32 Stage-1 `LP` within `0.000588` BPB alongside the `12.0 GiB` RTX 5070 engine (`8.78 GiB` at `8.9B`).
4. **Same-Tokenizer (`ByteBPETokenizer`) `T-38` Bridge (Realm D):** Evaluates `Plain Flat Tied` against `Paradigmatic Cluster-IS + OE-Span` on an identical `ByteBPETokenizer`, achieving `-0.0281` ΔBPB at matched analytic FLOPs (`eb_p = 0.0051`) while transparently reporting short equal-byte warmup parity (`+0.0075` ΔBPB). Verify via `sheaf-verify campaign --mode t43-all-realms --path iterations/t43-all-realms-dominance/T43_ALL_REALMS_RESULTS.json`.
