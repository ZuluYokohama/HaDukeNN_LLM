# T-38 design: mode and method

Summary of `T38_PREREG.redacted.json`. If this summary and the JSON disagree, the JSON governs.

## Mode

Same-tokenizer, same-data, same-backbone comparison of arm P (plain, flat head) and arm H (structure component), bytes-matched, staged over five model sizes.

## Method

- Data: FineWeb-Edu subset `fwedu32`, 540,570,430 unique train bytes (117,579,261 bpe32k tokens). The metric is test bpb_corpus on identical test bytes (10,005,000). Selection uses the validation split.
- Shared: RoPE, RMSNorm, SwiGLU (ff ≈ 8/3·d), tied embeddings, seq 256, batch 48 (micro 16), AdamW wd 0.1, warmup 2%, cosine to 0.1×, `torch.compile`, 85 °C stop.
- Rungs: XS (d 256, 4 layers), S (d 384, 6), M (d 576, 8, ff 1,536, 9 heads; headline), M+ (d 640, 10), L (d 768, 12).
- Budgets: 1 epoch (540,570,430 B) and 4 epochs (2,162,281,720 B).
- LR: a per-arm sweep over {5e-4, 1e-3, 2e-3} at seed 13 and 0.5 epoch, for M, M+, and L. The pick is the lowest validation BPB, and ties under 0.001 go to 1e-3. XS and S use 1e-3.
- Seeds: headline M-4ep uses 13, 21, 34, 55, 89. Other cells use 13, 21, 34. No optional stopping.
- Stages: smoke (S, seed 13, 0.05 epoch, gate only) → Stage 1 (M sweep, then M-4ep × 5 seeds × 2 arms) → Stage 2 (M+ sweep; XS, S, M+ at 1 and 4 epochs; M at 1 epoch) → Stage 3 (L sweep; L at 1 and 4 epochs, gated by `stage3_stop`).

## Pre-registered rules

- Headline, with delta = BPB_P − BPB_H:
  - PASS: one-sided Welch p < 0.01 with H < P, and delta ≥ 2 SE.
  - KILL: the mirror of PASS.
  - PARITY: not PASS, and the one-sided 99% upper bound of BPB_H − BPB_P is below 0.005.
  - Otherwise INCONCLUSIVE. Precedence is PASS > KILL > PARITY.
- Ladder:
  - Each cell uses the same PASS/KILL rule, with Holm-Bonferroni over all rung × data cells at family α 0.01.
  - PASS_scaling: the 95% seed-bootstrap CI (2,000 reps) of α_H − α_P excludes 0 in H's favour.
- Power note in the prereg: σ 0.0016 gives MDE80 0.0039 BPB with 5 seeds and 0.0064 with 3 seeds.
- Secondary, descriptive only:
  - BPB at matched analytic FLOPs and at matched wall-seconds;
  - throughput and peak memory;
  - zero-shot lm-eval;
  - a rare-token probe;
  - cross-domain WikiText-103;
  - the paired-seed addendum.

## Estimates (E)

GPU hours: about 11.5 for the Stage 1 headline, 20.8 for Stage 2, 20.4 for Stage 3 including its sweep, and about 54 for the full ladder.

## Controls and gates

T-38 speaks to the structure component at equal bytes on a shared tokenizer. Until it reports, no structure claim is made. The analysis code and the trainer are not in this repository.
