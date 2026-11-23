<h1>HaDukeN //\\// HGNLLM Experiments</h1>

[![PolyForm Noncommercial 1.0.0](https://img.shields.io/badge/license-PolyForm%20Noncommercial%201.0.0-blue)](LICENSE)

Verification harness for hypergraph and sheaf language-model experiments. The Python package and the command remain `sheaf-verify`.

The harness does not train a model. A trainer, running on its own machine, registers the bytes it actually used. This repository stores manifests, a hash-chained ledger, synthetic fixtures, and the recorded G0 metadata. It does not store model weights, a corpus, tokenizer tables, or credentials.

Licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE). Commercial use requires a separate license from the author. See `NOTICE`. Contact via GitHub [@ZuluYokohama](https://github.com/ZuluYokohama).

## Install

```bash
pip install -e ".[dev]"
ruff check .
pytest
sheaf-verify verify --store fixtures/passing --seal-key-file fixtures/test_seal.key
```

Python 3.11 or newer. The runtime dependencies are the standard library. pytest and ruff are the development dependencies.

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
docs/                static briefing site for GitHub Pages
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

## Gaps

`GATES.md` now follows the architecture spec and the paper plan. G0 is the valid-split unit reduction (15% fewer than GPT-2 BPE, byte fallback at most 1%, vocabulary at most 64,000, build at most 2 CPU-hours and 16 GB). The feasibility build's `V = 65,536` is flagged against that cap in `fixtures/g0-recorded/`; that file is recorded metadata, not data. G1 is hnt64k test BPB_corpus at or below struct32k plus one standard deviation, with lower train FLOPs per byte than struct32k and the equal-size sw64k control. H0–H7 are the spec stages, and P0–P6 are the paper-plan phases. Pilots use `PILOT_SEEDS = 3`. The headline uses `HEADLINE_SEEDS = 5`. A single `MIN_SEEDS = 5` is not applied to G1.

The seal is still HMAC-SHA256, not an Ed25519 signature. `sheaf-verify anchor` prints the ledger head so it can be copied out of band. It does not add a timestamp service.

The causality trace and the per-seed BPB are checks of submitted records. A trainer can still submit a trace that was not produced by the named code. An independent rerun is still required. The git commit id, when present, is recorded and not recomputed unless that checkout is still present. The git tree SHA-1 and the SHA-256 listing are what `verify` recomputes from the snapshot.
