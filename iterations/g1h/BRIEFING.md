# Iteration g1h: plain hnt40k (structured path off), FLOPs-matched to g1d/g1g, briefing

Times are US Central (CT), Mon Nov 23, 2026. Numbers are measured from the per-seed trainer records unless marked as an estimate. WikiText-103 test bits per byte (BPB_corpus), lower is better.

## Observe

| Seed | Test BPB | Valid BPB | Train bytes | Train FLOPs |
|---:|---:|---:|---:|---:|
| 13 | 1.198146 | 1.174116 | 400.9M | 1.0758e+16 |
| 21 | 1.197111 | 1.175110 | 400.9M | 1.0758e+16 |
| 34 | 1.197653 | 1.174671 | 400.8M | 1.0758e+16 |

Test BPB 1.197637 ± 0.000517 (n=3, 95% CI 1.196352 to 1.198922). Parameters 25,981,824 total, 10,621,824 non-embedding.

## Orient

- vs g1g: +0.023321 BPB (Welch two-sided p 8.4e-07). Equal train FLOPs, same tokenizer; isolates the structured path.
- vs g1d: +0.003482 BPB (Welch two-sided p 0.006). Equal train FLOPs.

## Decide / Act

Recorded. This iteration does not by itself change a G-gate status. See `docs/index.html` for the campaign table.

## Caveats

n=3, one model size (6 layers, d 384), WikiText-103 only. Every run applied RoPE: the trainer's StructGPT path applies RoPE only, so `--pe heat` is a no-op there and no heat positional encoding was tested.
