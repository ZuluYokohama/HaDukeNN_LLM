# Architecture

sheaf-verify is a content-addressed store plus an append-only ledger of gate decisions. Training code stays outside the store. The store answers a narrower question: given these bytes, do the declared checks pass, and does a later edit show up?

## Manifests

A manifest is canonical JSON, format `sheaf-verify.manifest.v1`. Canonical means UTF-8, keys sorted, separators `(",", ":")`, no ASCII escaping, and no NaN. The file name is the SHA-256 of those exact bytes. Verification reads the file back and rejects it when the bytes are not that form.

The payload hash is separate. Inline bytes live at `payloads/<sha256>`. A checkpoint uses `storage: external` and records `payload_sha256`, `size_bytes`, and a locator. The register command refuses checkpoint bytes. `external://...` is a note, not a file. A filesystem locator that exists on the verifying machine is hashed and compared. A missing filesystem path is a note unless `--strict` is set.

`inputs` are manifest hashes that must already exist. That rule keeps provenance a DAG. The current binding for a role is the manifest with the latest `registered_at`, then the greater hash. Timestamps are `YYYY-MM-DDTHH:MM:SSZ`, so lexicographic order is time order.

Roles are dotted identifiers such as `corpus.train` and `checkpoint.hypergraph`. Kinds are `raw_data`, `hypergraph`, `vocabulary`, `code_tree`, `environment`, `config`, `seeds`, `checkpoint`, `log`, `metrics`, `figure`, `kernel`, `result`, `claim`, and `trace`.

## Code identity

`register-tree` writes a canonical listing of `{mode, path, sha256, size}` and a snapshot of those bytes at `snapshots/<content sha>/`. The manifest also stores the git tree SHA-1 of that listing, computed with Git's object format: blob headers, mode `100644` or `100755`, and directory entries sorted as `name + "/"`. Symlinks are refused. `.git`, `__pycache__`, `.sheaf`, `.venv`, and `.pytest_cache` are skipped. If the directory is a git checkout, the commit id is recorded and `verify` says so. The value it recomputes is the tree id, from the snapshot.

## Ledger

`ledger/ledger.jsonl` is one canonical record per line. `prev_sha256` sits inside the hashed body and names the previous line's `record_sha256`. Editing, deleting, or reordering a line breaks the chain. `ledger/head.json` must name the last line. A non-empty `ledger/.lock` is a failure. The lock is only for `flock` during append, and the fixture deletes the empty file.

The optional seal is HMAC-SHA256 over the canonical body with the signature and the record hash removed. The record hash then covers the body including the signature. The key id is `HMAC-SHA256(key, "sheaf-verify.key-id.v1")` and is not the key. This is not a public-key signature. A holder of the key can write a different chain that verifies. Unsigned records are rejected unless `--allow-unsigned` is set. A chain cannot mix signed and unsigned lines.

`sheaf-verify anchor` prints `sheaf-verify.anchor.v1` and the `head_sha256` from `ledger/head.json`. `--out` appends that same line to a path outside the store. Printing the line, or copying the file onto another machine by hand, is the out-of-band anchor. The command reads the head file and, when asked, writes the path you name. It does not call a timestamp service, a notary, or any other network service. The anchor matters only if the copy is kept where the seal key cannot rewrite it. `verify` is still what checks the chain.

## Gates and replay

The graph is declared in `src/sheaf_verify/gates.py` and drawn in `GATES.md`. `run` evaluates every gate in topological order and appends the decision, pass or fail. A gate passes only when every criterion passed and every upstream dependency has a passing ledger record whose bound manifest and payload hashes are the hashes bound now. Re-running an upstream gate without re-running the downstream gate makes `verify` fail, because replay recomputes the latest decision against the full ledger.

After the gates, `run` registers a `result` manifest. Its payload names the outcome, the ledger head, and each gate's record hash. Its inputs are the current binding, excluding any previous result. `verify` checks that the latest result matches the latest records and that every result's ledger head is some record in the chain.

## What the checks do not do

They do not execute the model, the CUDA or OpenCL kernel, or a second training run. H4 checks a submitted causality trace. G1 checks the raw-byte BPB identity. G2 and G4 recompute one-sided Welch tests. G4 also recomputes the two-tailed Welch or paired test, the bootstrap interval, and Hedges' g from the submitted per-seed numbers. A trace or a seed list can be invented. The harness makes that invention explicit and comparable. It does not make it true.

Statistics use a Student-t tail from the regularized incomplete beta, so SciPy is not a dependency. Bootstrap draws use `random.Random` with the recorded seed. On CPython that sequence is deterministic.

## Design precedents

Ideas below were read and not copied. None of those repositories were modified. Their licenses were not adopted. This tree stays under the notice in `NOTICE`.

### ZuluYokohama/mission-lab, `research/mission_lab_v0_1`

Used. Gates are predicates over a sealed record, and a failed upstream predicate blocks a downstream one. The seal is HMAC-SHA256 with a key id, and the documentation says that is not a public signature. Manifests are canonical JSON. Symlinks are refused. A worker may submit a result, but the controller is what decides the predicate.

Not used. The PolyForm license. The crystal-kernel and equation-graph code. Any trading or protein-structure implementation. The worker, planner, and executor modules as code.

### ZuluYokohama/jovin-harness

Used. The ledger is an append-only hash chain. Verification detects an edit, a deletion, and a reorder. A content hash is not treated as a signature. A failed attempt is still a record.

Not used. The PolyForm license. The receipt module as code.

### ZuluYokohama/btc-mining-sim, `src/btc_mining_sim/research_protocol.py`

Used. A claim can be a preregistered hypothesis whose canonical JSON is the hashed payload (`sheaf-verify.claim.v1`), frozen before metrics exist. Corpus splits carry integer orders and must satisfy train < valid < test, the chronological half of that file's holdout check, in addition to disjoint document ids. A superiority claim fails when the superword-BPE contrast and the randomized-hypergraph contrast disagree. Agreement is required only for a claim of superiority. A study that claims nothing still records the disagreement and can pass G4. That is the "controls disagree, so the result is not supported" rule, applied to the two controls this harness already computes. The seal comparison uses `hmac.compare_digest`.

Not used. SHA-256d, midstate caching, AsicBoost, difficulty retargeting, the mempool, Merkle coinbase branches, mining strategies, and the mission-lab bridge that drives those campaigns. Circular-shift and permutation nulls over paired numeric series. Pearson correlation and mean squared error. Those belong to a mining-math simulator, not to a bits-per-byte study.

### ZuluYokohama/network-analysis-engine, branch `sheaf-gauge-engine` (`dc51a77`)

Used. A hypergraph manifest may carry an oriented 2-complex (`num_vertices`, `edges`, `faces`). H0 then checks the boundary identity, the maximum absolute entry of B1 times B2, which `FaceComplex.verify_boundary_identity` checks in that branch. The check here is a few dozen lines of integer arithmetic. It does not import that class.

A study hypergraph may also carry a sheaf block: state ids, operations with a string restriction map, and one or more pairs of paths. H4 composes the maps and fails when converging paths disagree. That is the loop-closure idea in `network_engine/sheaf/cocycle.py`. The composition is the obvious substitution of string maps. The process-graph classes, NumPy stalks, and cocycle subgraph report were not imported.

A kernel whose `backend` is `cuda`, `opencl`, or `c` must name `host_reference_sha256`, and that hash must be the payload of an input manifest. The branch's device code is OpenCL in `packages/ne_adreno_gpu/ne_adreno_gpu/kernels.cl` and C in `packages/ne_oryon_kernels/csrc/sheaf_kernels.c`. No `.cu` file is on that branch. The harness still accepts `cuda` as a backend name so a later CUDA source is bound the same way: source hash in the manifest, host reference as a separate payload, no launch. CPU kernels are their own reference.

The branch's `Check` record (name, passed, detail) is the same shape as a criterion. Severity and category were left off. Every criterion in this harness is blocking. Their router fails closed when a caller forces the flat regime on a frustrated complex. The same posture is why a superiority flag cannot be set without both contrasts supporting it.

Not used. The desktop and web GUI. Sensor-telemetry ingestion and quarantine. Involvement barcodes, hierarchy line graphs, and the gauge numerics in `network_engine/gauge/curved.py` and `flat.py`. The OpenCL and C kernel sources. The mainnet mempool experiments. Their `network_engine/meta/seal.py` reuses the mission-lab key-id label; this harness uses `sheaf-verify.key-id.v1` and does not call that function.
