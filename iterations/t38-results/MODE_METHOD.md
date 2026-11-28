# Iteration t38-results: mode and method

## Mode

T-38 results record: plain Transformer (arm P) against the HaDukeNN structure component (arm H) on the same 32k BPE tokenizer, bytes-matched, staged over five model sizes. Summary of what was run under `iterations/t38-design/T38_PREREG.redacted.json`. If this summary and that JSON disagree, the JSON governs.

## Method

- Data: FineWeb-Edu subset `fwedu32`, 540,570,430 unique train bytes. The metric is test bpb_corpus on identical test bytes (10,005,000). Selection used the validation split.
- Shared: RoPE, RMSNorm, SwiGLU, tied embeddings, seq 256, batch 48 (micro 16), AdamW wd 0.1, warmup 2%, cosine to 0.1×, `torch.compile`, 85 °C stop, one GPU, one run at a time.
- Rungs: XS (d 256, 4 layers), S (d 384, 6), M (d 576, 8; ff 1,536, 9 heads), M+ (d 640, 10), L (d 768, 12).
- Budgets: 1 epoch (540,570,430 B) and 4 epochs (2,162,281,720 B). Sweeps used 0.5 epoch (270,285,215 B). The smoke pair used 0.05 epoch (27,028,522 B).
- Learning rate: lr 1e-3 for every arm at every rung. The per-arm sweeps over {5e-4, 1e-3, 2e-3} at seed 13 (M, M+, L) all picked 1e-3 (lowest validation BPB; at L, P's 5e-4 and 1e-3 are within 0.001, so the tie rule gives 1e-3); the sweep rows are in `per_seed.csv`.
- Seeds: M-4ep uses 13, 21, 34, 55, 89. Every other cell uses 13, 21, 34. No early stopping.
- Horizon-matched P control: P only, M, seeds 13, 21, 34, budget 97,550,600,306,688,000 analytic FLOPs (24,925 steps), run after the ladder.

## Pre-registered rules used

- Headline, with delta = BPB_P − BPB_H: PASS is one-sided Welch p < 0.01 with H < P and delta ≥ 2 SE. KILL is the mirror. PARITY: not PASS and the one-sided 99% upper bound of BPB_H − BPB_P is below 0.005. Otherwise INCONCLUSIVE. Precedence PASS > KILL > PARITY.
- `stage3_stop`: skip Stage 3 iff the headline is KILL with BPB_H − BPB_P > 0.0117 and the Stage 2 4-epoch gaps are not decreasing in N (Spearman rho ≥ 0).
- Ladder: per-arm fit L(N) = E + A·N^(−alpha) on N_body, seed bootstrap (2,000 reps). PASS_scaling needs the 95% CI of alpha_H − alpha_P to exclude 0 in H's favour. Amendment 1 sets when L is dropped from that fit.
- Horizon-matched control: descriptive and secondary only. Welch of 3 P against 5 H (Stage 1), alpha 0.01, delta = mean_H − mean_P.

## Controls and gates

T-38 speaks to the structure component at equal bytes on a shared tokenizer. It claims no G, H or P gate. The analysis code and the trainer are not in this repository.
