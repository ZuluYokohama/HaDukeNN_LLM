# T-39 blueprint: closing the large-vocabulary logit wall and orthogonal sheaf KV-prefix scaling

Times are US Central (CT), Wed Nov 25, 2026. **Design and verification specification only; no T-39 training run is claimed.** All verification predicates run through `sheaf-verify campaign`.

## Observe

- **T-37 (`iterations/t37-grid/`)** isolated the tokenizer effect across all 16 FineWeb-Edu cells: plain 32k BPE (`fwedu-struct32k`) beat `fwedu-hnt40k` in 8 of 8 contrasts (5 of 8 surviving Holm at α 0.01), with Bias-Variance data decay at both sizes and a positive model-size shift Δlog2 k(M − S) = +0.125 (95% CI +0.080 to +0.173).
- **T-38 (`iterations/t38-design/`)** froze the first clean same-tokenizer structure test (`fwedu-bpe32k`, V = 32,768, arm P vs. arm H) across 5 rungs (XS through L), using `nc = 182, S = 181` and `struct_rank = 32`.
- **Asymptotic FLOP dilution vs. the real frontier bottlenecks.** At T-38 Stage 1 Medium (50.7M total parameters, V = 32,768), a flat head accounts for 37.21% of linear training FLOPs, shrinking to 22.86% at Stage 3 Large (~110M) and 1.33% at 70B (V = 128k). However, at modern 128k–256k vocabularies, the output head becomes the dominant BF16 activation memory bottleneck (4.00 GiB of logits per 8,192 tokens at V = 262,144, and 33.33% of total FLOPs on 1B-class edge models). Simultaneously, cross-window entity amnesia scales orthogonally to model width and never vanishes at 70B.

## Orient

Auditing T-38 and the corpus-hypergraph queue via `sheaf-verify` v0.2.0 surfaces four concrete engineering upgrades for the next experimental wave:

1. **Exact `256 × 128` Tensor-Core cluster geometry (zero padding, zero redundant tables).** Partitioning V = 32,768 (`2^15`) into `NC = 256, S = 128` eliminates all 174 dead padding slots of `182 × 181` (removing `-1`/`0` gather/scatter aliasing hazards), aligns every BMM tile to 128 for Tensor Cores, satisfies the existing P1 `sizes <= 181` check automatically, and cuts `K_neg = 3` head FLOPs by 15.2% (2.654 vs. 3.131 MFLOPs/tok at d = 576) and BF16 gather footprint by 29.3% (2,304 vs. 3,258 MiB). Setting `struct_rank = 0` and `resid = none` under `members = identity` removes 19.94M duplicate parameters.
2. **Noise-buffered Stage-3 L-fit rule (`T38_AMENDMENT1_STAGE3_FIT.json`).** Under an additive Chinchilla law `L(N, D) = E + A·N^-α + B·D^-β`, expected 1-to-4-epoch data gain `BPB_1ep − BPB_4ep` is identical at M and L, causing an unbuffered `gain_L < gain_M` gate to drop L 50% of the time by seed noise. Buffering to `gain_L < gain_M − 2·SE_diff` lowers the false-drop rate to 2.3%.
3. **Hardware roofline isolation.** Wall-clock benchmarks must run solo on native BF16 Tensor-Core hardware (`sm_120` RTX 5070), keeping non-Tensor-Core side-job GPUs (`sm_75`) strictly restricted to FP32 `lm-eval` and `kNN-LM` evaluations.
4. **Two-part frontier synthesis (`T39_FRONTIER_PREREG.json`).** Part A scales `NC × 128` (`tied_root + K_neg = 3`) across V ∈ {32k, 128k, 256k} against compiled chunked/fused-CE baselines; Part B pairs the winning vocabulary bookend with decontaminated `hg-v3` (`27,902` train-only pool articles, `3 <= |e| <= 128`, Chodrow degree+size null, causal canary `0.0`) via Asymmetric Diagonal-Sheaf KV-Prefix registers (`Q = 256, KV = 272`, 43.48 MiB L2-resident working set).

## Decide / Act

- Keep `T38_PREREG.redacted.json` (`524fca09…`) frozen and untouched.
- Pre-declare `T38_AMENDMENT1_STAGE3_FIT.json` before any Stage-3 launch.
- Stage `T39_FRONTIER_PREREG.json` and `hg_v3_gate_verified.json` under `sheaf-verify campaign` audit gates.
