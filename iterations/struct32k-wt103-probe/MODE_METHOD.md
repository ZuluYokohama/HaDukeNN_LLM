# Iteration struct32k-wt103-probe: mode and method

## Mode

Probe: struct32k tokenizer, S size, flat head, structure off, full residual table, at a small data budget on WikiText-103. No paired comparison.

## Method

- Arm: plan `probe-tokenizer-struct32k-resid-full-bytes`; tokenizer struct32k (V 32,768); struct none, compose mean, resid full, no spectral init; eager (no `torch.compile`).
- Body: 6 layers, d 384, 6 heads, ff 1,024, RoPE, seq 256, batch 48 (micro 16), lr 1e-3, cosine to 0.1×, warmup 2%, AdamW wd 0.1.
- Budget: 87,087,822 train bytes. Seeds 13, 21, 34.
- Metric: full WikiText-103 test BPB_corpus.
- Safety: one GPU, 1 s temperature log, 85 °C stop, memory watchdog.

## Controls and gates

None. This probe does not claim any G, H or P gate.
