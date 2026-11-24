# Iteration g1i: plain hnt40k (structured path off), bytes-matched to g1d, briefing

Times are US Central (CT), Mon Nov 23, 2026. Numbers are measured from the per-seed trainer records unless marked as an estimate. WikiText-103 test bits per byte (BPB_corpus), lower is better.

## Observe

| Seed | Test BPB | Valid BPB | Train bytes | Train FLOPs |
|---:|---:|---:|---:|---:|
| 13 | 1.210431 | 1.187898 | 348.5M | 9.3502e+15 |
| 21 | 1.215228 | 1.193431 | 348.5M | 9.3502e+15 |
| 34 | 1.208666 | 1.186437 | 348.4M | 9.3502e+15 |

Test BPB 1.211442 ± 0.003395 (n=3, 95% CI 1.203007 to 1.219876). Parameters 25,981,824 total, 10,621,824 non-embedding.

## Orient

- vs g1d: +0.017287 BPB (Welch two-sided p 0.0094). Equal train bytes (fair contrast for the plain arm).
- vs g1h: +0.013805 BPB (Welch two-sided p 0.018). Same arm, fewer bytes and FLOPs.

## Decide / Act

Recorded. This iteration does not by itself change a G-gate status. See `docs/index.html` for the campaign table.

## Caveats

n=3, one model size (6 layers, d 384), WikiText-103 only. Every run applied RoPE: the trainer's StructGPT path applies RoPE only, so `--pe heat` is a no-op there and no heat positional encoding was tested.
