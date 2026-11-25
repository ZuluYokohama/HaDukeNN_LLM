# Iteration hg-e1-pack-random: mode and method

## Mode

E1 corpus-hypergraph packing control: seeded random article order, struct32k, bytes-matched to g1d.

## Method

- Arm: `hg-e1-pack-random-wt103`; tokenizer struct32k (V 32,768); struct none, flat tied head.
- Body: 6 layers, d 384, 6 heads, ff 1,024, RoPE, seq 1,024, batch 12 (micro 6), lr 1e-3, cosine to 0.1×, warmup 2%, AdamW wd 0.1.
- Budget: 348,328,219 train bytes (g1d's budget). Seeds 13, 21, 34.
- Primary metric: E1 held-out-article BPB (`xb_bpb`, protocol causal-v2, 569 held-out articles).
- Also reported: full WikiText-103 test BPB_corpus.
- Safety: one GPU, 1 s temperature log, 85 °C stop, memory watchdog.

## Controls and gates

This is the control for the E1 packing comparison (the hyper and hyper-null arms). It does not claim any G, H, or P gate.
