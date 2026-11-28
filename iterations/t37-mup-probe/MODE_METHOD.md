# Iteration t37-mup-probe: mode and method

## Mode

Single-factor probe: muP-scaled hidden learning rate against the T-37 baseline cell fwedu-struct32k, M size, 0.25× bytes.

## Method

- Arm: plan `t37-mup-s32-M-b025-bytes`; tokenizer fwedu-struct32k (V 32,768); struct none, flat tied head, compose mean, low-rank residual, no spectral init.
- Body: 8 layers, d 576, 6 heads, ff 1,024, RoPE, seq 256, batch 48 (micro 16), lr 1e-3, cosine to 0.1×, warmup 2%, AdamW wd 0.1, `torch.compile` on.
- muP: `mup` on with reference width d0 = 384 (hidden-matrix learning rate scaled by d0 / d).
- Budget: 87,082,055 train bytes (87,111,190 bytes seen). Seeds 13, 21, 34. Data: FineWeb-Edu subset `fwedu32`, full test split (10,005,000 bytes).
- Safety: one GPU, 1 s temperature log, 85 °C stop, memory watchdog.

## Controls and gates

The control is the T-37 grid cell `t37-sg-s32-M-b025` (same size, tokenizer, budget and seeds). This probe does not claim any G, H or P gate.
