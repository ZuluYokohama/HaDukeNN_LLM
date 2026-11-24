# Iteration g1d: struct32k, data-matched to G1c hnt bytes, briefing

Times are US Central (CT), Mon Nov 23, 2026. Numbers are measured from the per-seed trainer records unless marked as an estimate. WikiText-103 test bits per byte (BPB_corpus), lower is better.

## Observe

| Seed | Test BPB | Valid BPB | Train bytes | Train FLOPs |
|---:|---:|---:|---:|---:|
| 13 | 1.193249 | 1.169532 | 348.4M | 1.0756e+16 |
| 21 | 1.194859 | 1.171576 | 348.4M | 1.0756e+16 |
| 34 | 1.194357 | 1.172360 | 348.4M | 1.0756e+16 |

Test BPB 1.194155 ± 0.000824 (n=3, 95% CI 1.192108 to 1.196201). Parameters 23,204,736 total, 10,621,824 non-embedding.

## Orient

- Baseline for g1g, g1h, and g1i.

## Decide / Act

Recorded. This iteration does not by itself change a G-gate status. See `docs/index.html` for the campaign table.

## Caveats

n=3, one model size (6 layers, d 384), WikiText-103 only. Every run applied RoPE: the trainer's StructGPT path applies RoPE only, so `--pe heat` is a no-op there and no heat positional encoding was tested. Epochs are train bytes divided by the WikiText-103 train split size; 1.65 for g1g and 0.64 for g1d are the reported figures.
