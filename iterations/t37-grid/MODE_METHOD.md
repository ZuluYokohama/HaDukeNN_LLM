# Iteration t37-grid: mode and method

## Mode

Tokenizer scaling grid: fwedu-hnt40k against fwedu-struct32k, flat heads, bytes-matched, 2 sizes × 4 budgets.

## Method

- Arms: tokenizer `fwedu-hnt40k` or `fwedu-struct32k`; struct none, compose mean, resid lowrank, no spectral init, flat head. The structured path is off in both arms.
- Sizes: S is 6 layers, d 384, 6 heads, ff 1,024 (h40 25,981,824 params; s32 23,204,736). M is 8 layers, d 576, 6 heads, ff 1,024 (h40 47,822,400; s32 43,656,768).
- Body: decoder-only transformer, RoPE, RMSNorm, SwiGLU, tied embeddings; seq 256, batch 48 (micro 16), lr 1e-3, cosine to 0.1×, warmup 2%, AdamW wd 0.1, `torch.compile` on.
- Budgets: 0.25×, 0.5×, 1.0×, 1.5× of 348,328,219 train bytes. Seeds 13, 21, 34.
- Data: FineWeb-Edu subset `fwedu32`, 540,570,430 train bytes; test 10,005,000 bytes, validation 10,023,995 bytes.
- Eval: BPB_corpus = Σ nats / (ln 2 × split bytes) on the full test split.
- LR check: M-0.25× s32 at lr 5e-4 and 2e-3 (`t37-lr-*` rows in `per_seed.csv`).
- Hardware: one RTX 5070, one run at a time.

## Pre-registered rules

- Cell: PASS if gap ≥ 2 SE and one-sided Welch p < 0.01.
- Data axis: sample-complexity view if log2 k > 0 at every budget and the slope CI lies within [−0.25, 0.25] and contains 0. Bias-variance view if the slope CI is below 0 or the gap is ≤ 0 at 1.5×. Compounding view if the CI is above 0.
- Kill: gap ≤ 0 at M-1.5×.
- Engineering: compiled and eager S-1× gaps within 0.010.

## Controls and gates

These runs speak only to the tokenizer choice. They do not claim G1–G6, H, or P gates, and they say nothing about the structured head.
