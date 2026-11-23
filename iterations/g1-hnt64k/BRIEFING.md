# Iteration g1-hnt64k: units-only pilot (G1), briefing

Times are US Central (CT), Mon Nov 23, 2026. All numbers are measured; none are estimates.

## 1. Situation (OBSERVE)

Twelve runs: 4 vocabularies × seeds 13, 21, 34, on one RTX 5070 between 13:00:38 and 14:01:47 CT. Same transformer body for every arm. Byte-matched at 136 MB of WikiText-103 train text. Scored on the full official WikiText-103 test split (1,290,590 bytes) as BPB_corpus, which is total nats ÷ (ln 2 × official bytes). Lower is better.

| Vocab | V | Total params | Test BPB mean ± sd | 95% CI | Train FLOPs/byte |
|---|---:|---:|---:|---:|---:|
| gpt2 | 50,257 | 29,920,512 | 1.328219 ± 0.002166 | [1.322838, 1.333600] | 4.1140e7 |
| struct32k | 32,768 | 23,204,736 | 1.322354 ± 0.006949 | [1.305090, 1.339617] | 3.0882e7 |
| hnt64k (hypergraph span units) | 65,536 | 35,787,648 | 1.452254 ± 0.016348 | [1.411643, 1.492865] | 3.5706e7 |
| sw64k (superword BPE control) | 65,536 | 35,787,648 | 1.549741 ± 0.002788 | [1.542814, 1.556668] | 3.5156e7 |

Welch tests on test BPB (n = 3 per arm, two-sided):
- hnt64k vs struct32k: +0.129900, worse (t 12.67, p 0.0018).
- hnt64k vs gpt2: +0.124035, worse (t 13.03, p 0.0051).
- hnt64k vs sw64k: −0.097487, better (t −10.18, p 0.0078).
- struct32k vs gpt2: −0.005865 (one-sided p 0.139, not significant).

Per-seed numbers are in `per_seed.csv` and the full summary is in `results.json`.

## 2. Mission

Test gate G1: does a vocabulary of hypergraph span units, with no other architecture change, match the 32k tokenizer on BPB at lower training FLOPs per byte?

## 3. Execution (ORIENT)

- **Verdict: FAIL on both criteria.**
  - BPB: hnt64k scored 1.452254, above the threshold of struct32k mean + 1 sd = 1.329303.
  - FLOPs/byte: hnt64k's 3.5706e7 is above struct32k's 3.0882e7 and above sw64k's 3.5156e7.
- **Interpretation:**
  - hnt64k encodes the same text in 27% fewer units, but each unit costs 221.8 MFLOPs against 146.3 for struct32k. The 65,536-row tied output layer causes most of that.
  - At 136 MB (about a quarter epoch), the 15,279 span units are rarely seen, so their embeddings get little training.
  - Choosing spans by corpus savings does beat plain superword BPE merges at equal size, by 0.0975 BPB.
- **Threats to validity:**
  - n = 3 per arm.
  - One model size and one learning rate (1e-3), with no per-arm sweep.
  - The superword control was trained on a 40 MB slice of the training text.
  - Seed-34 runs (all arms) were slowed by disk I/O contention from a separate tokenization job. BPB is byte-matched and unaffected, and FLOPs/byte is analytic, but wall-clock and units/s from these runs must not be used for speed claims.
  - struct32k s13 and s21 throughput also varied (87,197 and 143,153 units/s) for no identified reason.
  - The spec caps the vocabulary at 64,000, and hnt64k has 65,536.

## 4. Sustainment (DECIDE)

The next step is iteration g1b-structured. It fixes the two failure causes with hypergraph structure rather than simply shrinking the vocabulary:
- compositional embeddings shared along derivation hyperedges, plus a spectral initialization, so rare units inherit trained signal;
- a cluster-then-unit output head over hypergraph spectral clusters, which cuts per-unit output cost;
- a pruned vocabulary at V = 40,000, which meets the vocabulary cap.

## 5. Command & Signal (ACT)

- Launched in this iteration: the G1 queue (12 runs), plus a one-second GPU telemetry log for its whole window.
- Safety limits held: GPU 0 only, idle check before each run, 85 °C abort (max reached 63 °C), no clock or power changes.
- Gates dashboard: G0 shows "partial" (vocab cap) and G1 shows "fail".
- Nothing was published. This bundle is a local record.
