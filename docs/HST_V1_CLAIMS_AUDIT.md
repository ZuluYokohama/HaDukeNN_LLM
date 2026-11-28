# HST-v1 claims audit (LapBeast, 2026-11-26)

This is the authoritative status of claims that appeared in the v1.2.0 HST-v1 briefing HTML (sha256 `e57ce3d06ba1926581a93b6b61f569c1458f49c92707ee84590d806cdc2a7319`, not committed; the file is 1.6 MiB and is not in `docs/`) and in `docs/HST_V1_SPEC.md` / `iterations/t49-hst-foundation-stage/BRIEFING.md` as shipped. Numbers below were checked on LapBeast (RTX 4080 Laptop, CUDA sm_89). A corrected briefing is `docs/HST-v1-briefing.html`.

No claim in this table is a forecast for a large public-corpus pretrain. A separate, pre-registered equal-bytes run on a public corpus is documented under `iterations/t49-hst-foundation-stage/` when that protocol is committed.

## Verified on this host

| Claim | Evidence |
| --- | --- |
| Digit atomicity, lossless UTF-8/binary roundtrip, causal prefix invariance, Merkle manifest `10d7198bfcda0cada0d332a979a00004cea7b632c016af61557a239077b45ba8`, tamper rejection, fixed control IDs 256–263 | Unit tests in `tests/test_hst_foundation_v120.py` and `verify_hst_manifest_integrity` on `iterations/t49-hst-foundation-stage/HST_V1_REFERENCE_MANIFEST.json` |
| 19.85% token-sequence reduction at vocabulary size `V = 416` versus Byte-BPE on the T-49 toy corpus | Sealed and live T-49 JSON; re-measured with `sheaf-verify campaign --mode t49-hst-foundation --live` |

## Reproduced, but only as a 48 KB toy campaign

The sealed T-49 pillar-2 numbers (`0.876562` vs `1.255984` mean test BPB, gain `-0.379422`, `5/5` seeds, paired SE `0.043612`, `p = 4.81e-4`, `100` vs `125` optimizer steps, rank-16 bottleneck `3.0032` BPB) match `iterations/t49-hst-foundation-stage/T49_HST_FOUNDATION_RESULTS.json`. A live re-run on this host still won `5/5` with `p ≈ 4.26e-4` and gain `0.374723` (small numeric drift). Pillar-3 muP width numbers (`d = 32, 64, 128`) are produced by the same in-repo function `run_t49_hst_foundation_campaign()`.

**These results use a 48,000-byte synthetic corpus and 100–125 optimizer steps (about 15 seconds to one minute). They are not a WikiText-103 or FineWeb-Edu foundation-model pretrain, and they must not be cited as one.**

## Recorded in campaign JSON, not re-run on this host during the v1.2.0 work

| Claim | Where it lives |
| --- | --- |
| T-37 struct32k `-0.0337` BPB at M-1.5×, `p = 5.1e-5`, `16/16` cells, `dlog2k = +0.1246` | `iterations/t37-grid/` and the README campaign summary. Labeled there as tokenizer-grid measurements. Not re-run for this audit. |
| T-38 Stage-3 L 1-epoch `+0.018610` BPB and `45.36` vs `39.00` TFLOPS (WBeast / RTX 5070 sheet) | Recorded constants in `src/hadukenn_core/hst_foundation_lm.py` and `src/sheaf_verify/t38_scaling_audit.py`. Not a LapBeast measurement. |

## Unsupported as stated (hypotheses or contradicted counts)

| Claim from the v1.2.0 briefing / spec | Status |
| --- | --- |
| “221/221 automated checks passing” and “150 pytest functions (0 failures) + 71 PR #9 unittests” | **Unsupported as a current LapBeast fact.** As-delivered v1.2.0 on this host: pytest **148 passed, 2 failed** (INT8 qgemm exactness and Zenbook-loop cosine `0.986 < 0.995`) and 8 ruff errors. After `v1.2.0-lapbeast` integration (phase-2 INT8/HAL retained): pytest **227 passed**, ruff clean, `src/sheaf_verify` coverage **86.58%**. The drop’s `ANTIGRAVITY_VERIFICATION_REPORT.json` still contains the 150/0 figure; treat it as a source-drop artifact, not a live receipt. |
| “20–35% compute left on the table” / “burning 20–35% extra sequence tokens” | **Hypothesis**, citing SuperBPE (Liu et al., 2025), not a HaDukeNN measurement on a large public corpus. The only in-repo measured reduction is **19.85% at V = 416** on the 48 KB toy set. The briefing’s “25–33% at 32k–65k” has no training log in this repository. |
| “125M–1.3B sweet spot” and the five-tier ladder as a demonstrated optimum | **Planned hardware targets** in `STAGED_FOUNDATION_LADDER`, not a fitted scaling law or a trained 125M–1.3B model in this repo. |
| “Regulated-domain readiness” / “auditable 360M–1.3B … beats an unverified 70B” | **Product hypothesis**, not a demonstrated property of any trained HaDukeNN-LM checkpoint. |
| Sealed T-46 / briefing hardware inventory “RTX 5060 / sm_120 / 275HX” as this machine | **Not this host.** LapBeast is an i9-13980HX + RTX 4080 Laptop (sm_89, 12282 MiB). |

The sealed toy T-49 JSON and the sealed T-46 JSON were not rewritten.
