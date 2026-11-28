# HaDukeNN Span-Trace v1 (`HST-v1`) Open Tokenization Standard & `HaDukeNN-LM` Architecture Specification

Claim status for the v1.2.0 briefing is in [HST_V1_CLAIMS_AUDIT.md](HST_V1_CLAIMS_AUDIT.md) and the corrected briefing [HST-v1-briefing.html](HST-v1-briefing.html). Do not cite the 48 KB T-49 probe as a foundation-model result.

## 1. Design Rationale

Standard subword tokenizers ([SentencePiece](https://arxiv.org/abs/1808.06226) and regex-constrained Byte-BPE) suffer from four structural flaws that tax both training compute and downstream reasoning:

1. **Whitespace Prison & Wasted Context**: Pre-tokenization regexes prevent high-information multi-word technical phrases and value-unit idioms from merging. The figure that this wastes `20-35%` extra sequence tokens and quadratic attention compute is a **literature hypothesis** from [SuperBPE, Liu et al., 2025](https://arxiv.org/abs/2503.13423), not a HaDukeNN measurement on a large public corpus. The only in-repo measured reduction is `19.85%` at `V = 416` on the 48,000-byte T-49 toy campaign; see [HST_V1_CLAIMS_AUDIT.md](HST_V1_CLAIMS_AUDIT.md).
2. **Inconsistent Multi-Digit Arithmetic Chunking**: Arbitrary 2- and 3-digit merges fragment numbers inconsistently (`1042` vs `10420`), degrading exact arithmetic and unit conversion.
3. **Unverified Prefix Drift & Uncommitted Vocabularies**: Distributed training and streaming inference lack a cryptographic vocabulary commitment (`SHA-256` Merkle root) and formal lookahead bounds (`max_span_bytes`) guaranteeing that unseen suffix bytes never mutate committed prefix tokens.
4. **String-Prompted Tool & Verification Hacks**: Verifiable reasoning actions (`fork`, `assert`, `unit`, `receipt`) are bolted onto raw text via fragile JSON wrappers rather than native reserved control slots coupled directly to an execution kernel.

At the same time, empirical scaling campaigns across `G1`, `T-30`, `T-37`, and the 5-rung `T-38` ladder (`XS, S, M, M+, L` in [the master results sheet](https://docs.google.com/spreadsheets/d/1Y74fgo8BTV6em61pOoHUiOsTh3jhco7i1FSwL-ADqi4/edit)) established three non-negotiable hardware laws for `HaDukeNN-LM`:

- **Put structure in input path contraction, not narrow output bottlenecks**: The recorded T-37 tokenizer grid (`iterations/t37-grid/`, not re-run for the v1.2.0 audit) reports that `struct32k` improved FineWeb-Edu equal-bytes compression across `16/16` scaling cells (`-0.0337` BPB at `M-1.5x`, `p = 5.1e-5`, growing with scale `dlog2k = +0.1246`), whereas `182x181` [Goodman class-factored](https://arxiv.org/abs/cs/0108006) output heads lost every equal-bytes cell (`XS +0.0320`, `S +0.0247`, `M +0.0188`, `M+ +0.0153`; 1-epoch `L +0.0186`) with exponent parity (`alpha_H - alpha_P ~ 0`).
- **Preserve contiguous Tensor-Core GEMMs via Tiled Cut Cross-Entropy**: Chunked vocabulary cross-entropy ([Wijmans et al., 2024](https://arxiv.org/abs/2411.00909)) sustains `41-45 TFLOPS` without materializing full `[B*W, V]` logit buffers, reserving cluster importance sampling strictly for `V >= 128k` or `batch=1` DRAM-bound edge NPU loops.
- **Enforce `muP` Zero-Shot Width Scaling**: Unscaled Standard Parametrization (`SP`) shifted its optimal LR downward and spiked to `1.5611` BPB at `lr = 0.002` on Stage 3 rung `L` (`d=768, L=12`), motivating exact [Tensor Programs V (`muP`)](https://arxiv.org/abs/2203.03466) `d_base / d` output-logit and hidden-LR scaling.

## 2. Normative `HST-v1` Wire Format (`hadukenn.hst.manifest.v1`)

Every compliant `HST-v1` tokenizer (`schemas/hst-v1.schema.json`, `src/sheaf_verify/hst_standard.py`) guarantees:

1. **Identity Byte Base (`IDs 0..255`)**: Exact 1-to-1 mapping to raw octets `0x00..0xFF` (`0%` unknown-token fallback on arbitrary UTF-8 or binary streams).
2. **Digit-Atomic Pre-Tokenization (`digit_atomic_policy = true`)**: ASCII digits `0..9` (`0x30..0x39`) remain isolated single-byte atoms in Stage A and are blocked from pure digit-to-digit merges in Stage B.
3. **Two-Stage Curriculum Merge Table (`merges`)**:
   - `A_lexical`: Intra-chunk byte-level BPE merges inside pre-tokenized lexical pieces.
   - `B_superspan`: Cross-boundary super-spans bridging spaces and unit tokens solely when empirical Pointwise Mutual Information satisfies `PMI(u, v) >= tau_pmi` and total span byte length `<= max_span_bytes` (`default 16`).
4. **Bounded Causal Prefix Invariance**: Under `causal_local` encoding, appending any suffix `S` after byte index `N` is mathematically guaranteed never to alter any token ending at or before `N - max_span_bytes`.
5. **Merge-DAG `StochasTok` Training Schedule**: During pretraining, merged spans decompose along their binary merge tree with cosine-annealed probability `p(t): 0.12 -> 0.00`, preserving sub-span representation sharpness.
6. **Reserved Verifiable Control Slots & Canonical Merkle Commitment**: Top-of-vocabulary slots (`<|hst:bos|>`, `<|hst:eos|>`, `<|hst:span_open|>`, `<|hst:span_close|>`, `<|hst:fork|>`, `<|hst:assert|>`, `<|hst:unit|>`, `<|hst:receipt|>`) bind inline to `IncrementalHypergraphKernel` ([PR #9](https://github.com/ZuluYokohama/HaDukeNN_LLM/pull/9)) and `ExactUnitLattice`, sealed under `merkle_root_sha256` and [IETF SCITT RFC 9943](https://www.rfc-editor.org/info/rfc9943/).

## 3. Staged `HaDukeNN-LM` Hardware Scaling Ladder

The tiers below are **planned hardware targets**, not a measured “125M–1.3B sweet spot” and not proof of regulated-domain readiness. No 125M–1.3B HaDukeNN-LM pretrain is in this repository. VRAM estimates are design figures.


| Tier ID | Architecture (`d / L / H / d_ff`) | Vocab (`HST-v1`) | Output Head | Target Hardware | Est. BF16 Train VRAM |
| :--- | :--- | ---: | :--- | :--- | ---: |
| `HST-Proxy-15M` | `256 / 6 / 4 / 704` | `32,768` | `tiled_cce` | Single 12 GB GPU (`muP` proxy tune) | `1.6 GiB` |
| `HST-Core-125M` | `768 / 12 / 12 / 2,048` | `32,768` | `tiled_cce` | Single 12 GB GPU (`~45 TFLOPS`) | `4.8 GiB` |
| `HST-Edge-360M` | `1,024 / 20 / 16 / 2,816` | `49,152` | `tiled_cce` | Single 12 GB GPU / `13 GB` Looped INT8 NPU | `8.9 GiB` |
| `HST-Base-1.3B` | `2,048 / 24 / 16 / 5,504` | `65,536` | `tiled_cce` | Single 24-48 GB GPU or 4xGPU Node | `11.4 GiB` (sharded) |
| `HST-Wide-3.2B` | `2,560 / 32 / 20 / 6,912` | `131,072` | `cluster_is` | 8xGPU Node / `14.5 W` INT8 Edge Loop | `19.8 GiB` (sharded) |
