# Iteration hg-e1-pack-random: random article order control for corpus-hypergraph packing, briefing

Times are US Central (CT), Wed Nov 25, 2026. Numbers come from the per-seed trainer records. Lower is better.

## Observe

| Run | Seeds | E1 held-out-article BPB (primary) | WikiText-103 test BPB_corpus |
|---|---|---:|---:|
| official (7:55–8:24 AM) | 13, 21, 34 | 1.093936 ± 0.003312 (1.093312, 1.090980, 1.097516) | 1.184595 ± 0.002326 |
| extra replicate (8:25–8:54 AM) | 13, 21, 34 | 1.095827 ± 0.006172 (1.099489, 1.088702, 1.099291) | 1.186588 ± 0.005957 |

± is the seed standard deviation, n = 3. Parameters: 23,204,736. Train FLOPs: 1.2317e16 per seed.

## Orient

- This is the control arm of the E1 packing experiment. It shows what ordinary token-window training sees: a seeded random article order. It has no pass or fail of its own. The E1 verdict comes from comparing the hyper-ordered and hyper-null arms against this control, and those results are not recorded here.
- The scheduler did not mark the plan complete after the official run, so it relaunched the same plan at 8:25 AM CT. That second run is kept as a labelled extra replicate. It is not pre-registered and not used for any verdict. It lands within 1 SE of the official run: +0.001891 BPB, against an SE of 0.004044 (descriptive only).

## Decide / Act

Recorded as a control. A guard now stops the remaining packing arms from relaunching this way. This iteration does not change a G-gate status.

## Caveats

n = 3, one model size (6 layers, d 384), struct32k tokenizer, WikiText-103 articles only, seq 1,024.
