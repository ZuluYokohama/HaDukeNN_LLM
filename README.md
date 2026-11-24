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

`run` writes every gate, including failures, then registers a result manifest. Exit 0 means every gate passed. Exit 1 means a gate failed and the ledger still holds the failure. Exit 2 means the command itself was refused.

`verify` recomputes hashes, the ledger, and gate replay. It exits 1 on a mismatch. A scientifically failed study can still verify: the failure is what the ledger says. `bundle` exits 1 unless verification is clean, every gate passed, and every checklist objection is answered.

`--strict` fails when a filesystem checkpoint path is missing. An `external://` locator stays a note either way, because the weight bytes live on the trainer.

## Layout

```
src/sheaf_verify/    library and the sheaf-verify command
tests/               pytest suite
fixtures/passing/    synthetic study, regenerated by python -m sheaf_verify.synthetic
fixtures/g0-recorded/  recorded G0 metadata, not a corpus
iterations/          index on main; one experiment per iter/<id> branch
docs/                static briefing site for GitHub Pages, including the control graph
CONTRIBUTING.md      pull-request rule for iteration branches
schemas/             JSON Schema drafts for the stored records
```

A store contains `manifests/`, `payloads/`, `snapshots/`, and `ledger/`. Checkpoint bytes are not a payload. `.sheaf/` is the default store path and is gitignored.

Read `ARCHITECTURE.md` for the hash and ledger rules, `GATES.md` for the pass conditions, and `REVIEWER_GUIDE.md` for the objections a reviewer is expected to raise. The public briefing is `docs/index.html`.

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
- g1g is a FLOPs-matched hnt run at 1.0756e16 train FLOPs, the same budget as g1d. It is pending. No result is recorded.

Caveats for G1c, g1d, and the pending g1g comparison: n=3, one model size, WikiText-103 only. Parameter totals are unverified in this repository: it contains no trainer. The g1b record reports 39,943,040 total and 10,621,824 non-embedding for one hnt40k-struct-v2 seed. The G1 briefing reports struct32k at 23,204,736. G1c does not reverse G1 or G1b. There is no `iter/g1c`, `iter/g1d`, or `iter/g1g` branch in this repository.

## Gaps

`GATES.md` now follows the architecture spec and the paper plan. G0 is the valid-split unit reduction (15% fewer than GPT-2 BPE, byte fallback at most 1%, vocabulary at most 64,000, build at most 2 CPU-hours and 16 GB). The feasibility build's `V = 65,536` is flagged against that cap in `fixtures/g0-recorded/`; that file is recorded metadata, not data. G1 is hnt64k test BPB_corpus at or below struct32k plus one standard deviation, with lower train FLOPs per byte than struct32k and the equal-size sw64k control. H0–H7 are the spec stages, and P0–P6 are the paper-plan phases. Pilots use `PILOT_SEEDS = 3`. The headline uses `HEADLINE_SEEDS = 5`. A single `MIN_SEEDS = 5` is not applied to G1.

The seal is still HMAC-SHA256, not an Ed25519 signature. `sheaf-verify anchor` prints the ledger head so it can be copied out of band. It does not add a timestamp service.

The causality trace and the per-seed BPB are checks of submitted records. A trainer can still submit a trace that was not produced by the named code. An independent rerun is still required. The git commit id, when present, is recorded and not recomputed unless that checkout is still present. The git tree SHA-1 and the SHA-256 listing are what `verify` recomputes from the snapshot.
