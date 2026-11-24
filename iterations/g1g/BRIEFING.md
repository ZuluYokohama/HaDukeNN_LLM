# Iteration g1g: hnt40k-struct-v2, FLOPs-matched to g1d, briefing

Times are US Central (CT), Mon Nov 23, 2026. Numbers are measured from the per-seed trainer records unless marked as an estimate. WikiText-103 test bits per byte (BPB_corpus), lower is better.

## Observe

| Seed | Test BPB | Valid BPB | Train bytes | Train FLOPs |
|---:|---:|---:|---:|---:|
| 13 | 1.174805 | 1.150929 | 891.7M | 1.0756e+16 |
| 21 | 1.173698 | 1.151319 | 891.7M | 1.0756e+16 |
| 34 | 1.174444 | 1.150757 | 891.7M | 1.0756e+16 |

Test BPB 1.174315 ± 0.000564 (n=3, 95% CI 1.172913 to 1.175718). Parameters 39,943,040 total, 10,621,824 non-embedding.

## Orient

- vs g1d: -0.019839 BPB (Welch two-sided p 1.3e-05). Equal analytic train FLOPs only; g1g saw about 2.56x the train bytes and ran about 3x longer per seed.

## Decide / Act

Recorded. This iteration does not by itself change a G-gate status. See `docs/index.html` for the campaign table.

## Caveats

n=3, one model size (6 layers, d 384), WikiText-103 only. Every run applied RoPE: the trainer's StructGPT path applies RoPE only, so `--pe heat` is a no-op there and no heat positional encoding was tested. Epochs are train bytes divided by the WikiText-103 train split size; 1.65 for g1g and 0.64 for g1d are the reported figures.
