# T-49 Staged Foundation Release: `HST-v1` Open Tokenization Standard & `HaDukeNN-LM` Pretraining Engine

> **Scope (read this first).** Pillar 2 is a **48,000-byte synthetic equal-bytes toy campaign** (100 vs 125 optimizer steps, typically under a minute). It is reproducible in-repo (`run_t49_hst_foundation_campaign`) and must not be cited as a WikiText-103, FineWeb-Edu, or multi-hour foundation-model result. Tokenizer invariants (digit atomicity, Merkle root `ad400f75…`, 19.85% reduction at `V = 416`) are verified. The v1.2.0 HTML briefing’s “221/221 checks”, “20–35% compute”, “125M–1.3B sweet spot”, and “regulated-domain readiness” claims are **not supported** by this campaign; see [`docs/HST_V1_CLAIMS_AUDIT.md`](../../docs/HST_V1_CLAIMS_AUDIT.md).

## 1. Objective & Empirical Synthesis

`T-49` stages the open **`HST-v1` (`HaDukeNN Span-Trace v1.0`) Tokenization Standard** (`src/sheaf_verify/hst_standard.py`, `schemas/hst-v1.schema.json`, `docs/HST_V1_SPEC.md`) together with the reference **`HaDukeNN-LM` Foundation Pretraining Stack** (`src/hadukenn_core/hst_foundation_lm.py`), incorporating every empirical law verified across `G1`, `T-30`, `T-37`, and `T-38` Stages 1–3 ([Master Results Sheet](https://docs.google.com/spreadsheets/d/1Y74fgo8BTV6em61pOoHUiOsTh3jhco7i1FSwL-ADqi4/edit), including the November 26 `7:48 PM CT` `Stage 3 L` `1`-epoch readout `+0.018610` BPB at `45.36` vs `39.00` TFLOPS).

## 2. Verified 5-Seed Results (`seeds = 13, 21, 34, 55, 89`)

1. **Pillar 1 — `HST-v1` Open Tokenization Standard (`hadukenn.hst.manifest.v1`)**:
   - Universal fixed layout (`IDs 0..255` raw byte atoms, `IDs 256..263` universal verifiable control tokens `<|hst:bos|>..<|hst:receipt|>`, `94` Stage A digit-atomic lexical merges, `58` Stage B PMI-gated cross-boundary super-spans; Merkle root `ad400f750a973371ded6bcd68e064ad271557a3d4995d7f3c1f91bcf0eafa079`).
   - Achieves **`19.85%` token sequence reduction** (`2.4889` causal bytes/token and `2.4907` DP bytes/token vs `1.9949` standard Byte-BPE bytes/token at identical vocabulary size `V = 416`), with `100%` lossless binary+UTF-8 roundtrip, `100%` single-digit arithmetic atomicity, and `0` causal prefix-invariance violations outside `max_span_bytes`.
2. **Pillar 2 — Strict Equal-Bytes toy LM probe (`48,000` synthetic training bytes, not a public-corpus pretrain)**:
   - Combining `HST-v1` + `CountShrunkSpanComposer` + cosine `StochasTok` + [cellular-sheaf](https://arxiv.org/abs/1808.01513) `KV-Prefix` registers + contiguous tiled [Cut Cross-Entropy](https://arxiv.org/abs/2411.00909) (`100` optimizer steps) achieves **`0.876562` mean test BPB** versus **`1.255984` mean test BPB** for standard `Byte-BPE + Dense SP` (`125` optimizer steps)—winning **all `5/5` seeds by `-0.379422` BPB (`paired SE = 0.043612`, `p = 4.81e-4`)** while executing **`20.0%` fewer optimizer steps**, whereas a rank-constrained output bottleneck loses by `+1.747203` BPB.
3. **Pillar 3 — `muP` Zero-Shot Width Scaling (`d = 32 -> 64 -> 128`)**:
   - Directly resolves the unscaled high-LR width spike observed on `Stage 3 L`: Standard Parametrization (`SP`) degrades at `d = 128` (`2.2411 -> 2.0092 -> 2.4768` BPB), whereas `HaDukeNN-LM` [Tensor Programs V (`muP`)](https://arxiv.org/abs/2203.03466) with RMSNorm-matched `sqrt(d_base / d)` output scaling improves monotonically (`2.2669 -> 1.7819 -> 1.4706` BPB, beating `SP` at `d = 128` by **`-1.006259` BPB**).
4. **Pillar 4 — Inline Verifiable Control-Token Decoding & SCITT Compliance**:
   - Intercepts `<|hst:fork|>` and `<|hst:assert|>` tokens inline via `IncrementalHypergraphKernel` ([PR #9](https://github.com/ZuluYokohama/HaDukeNN_LLM/pull/9)) and `ExactUnitLattice` (`20 L + 5000 mL == 25 L`), achieving **`100.0%` (`5/5` seeds) spoof refusal**, **`100.0%` deterministic semantic replay**, and verified [IETF SCITT RFC 9943](https://www.rfc-editor.org/info/rfc9943/) receipts.

## 3. Reproducing & Verifying

```bash
PYTHONPATH=.:src python3 -m sheaf_verify campaign \
  --mode t49-hst-foundation \
  --path iterations/t49-hst-foundation-stage/T49_HST_FOUNDATION_RESULTS.json --json
```
