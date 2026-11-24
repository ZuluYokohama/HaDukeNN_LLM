# Iteration g1g: mode and method

## Mode

hnt40k-struct-v2, FLOPs-matched to g1d.

## Method

- Arm: `hnt40k-struct-v2`; knobs: tokenizer hnt40k; struct spectral; compose slot; resid full; spectral init.
- Body: decoder-only transformer, 6 layers, d 384, 6 heads, SwiGLU 1,024, RMSNorm, RoPE base 10,000, tied embeddings; batch 48 × 256, lr 1e-3, cosine to 0.1×, warmup 2%, AdamW wd 0.1, bf16.
- Budget: budget_flops 1.0756e16. Seeds 13, 21, 34.
- Eval: full official WikiText-103 test and validation splits, BPB_corpus = Σ nats / (ln 2 × official split bytes).
- Safety: one GPU, 1 s temperature log, 85 °C stop, memory watchdog, no clock or power changes.

## Controls and gates

Contrasts are in `BRIEFING.md`. These runs speak to the G1-family follow-ups only. They do not claim G2–G6, H, or P gates.
