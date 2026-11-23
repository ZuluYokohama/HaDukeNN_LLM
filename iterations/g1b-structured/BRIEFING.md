# Iteration g1b-structured: structured hypergraph vocabulary (G1b), briefing

Times are US Central (CT), Mon Nov 23, 2026. All numbers are measured unless marked analytic; none are estimates.

## 1. Situation (OBSERVE)

G1 failed (hnt64k flat test BPB 1.452254 vs threshold 1.329303; FLOPs/byte 3.5706e7 vs struct32k 3.0882e7). G1b ran six main runs on one RTX 5070, 14:30–14:50 CT, plus one variant run 14:56–15:04 CT. Same body and 1.36e8-byte budget as G1; full official WikiText-103 test (1,290,590 bytes), BPB_corpus.

| Arm | V | Test BPB mean ± sd | 95% CI | Train FLOPs/byte | FLOPs/unit |
|---|---:|---:|---:|---:|---:|
| struct32k (G1 baseline) | 32,768 | 1.322354 ± 0.006949 | [1.305090, 1.339617] | 3.0882e7 | 146,276,352 |
| gpt2 (G1 baseline) | 50,257 | 1.328219 ± 0.002166 | [1.322838, 1.333600] | 4.1140e7 | 186,571,008 |
| hnt40k-struct | 40,000 | 1.477304 ± 0.007156 | [1.459528, 1.495079] | 1.1929e7 | 72,437,352 |
| sw40k-struct (control) | 40,000 | 1.583940 ± 0.005420 | [1.570475, 1.597404] | 1.2438e7 | 72,456,788 |
| hnt40k-struct-v2 (n = 1, seed 13) | 40,000 | 1.442863 | — | 1.2060e7 | 73,230,469 |

Welch (two-sided, n = 3): hnt40k-struct vs struct32k +0.154950 (t 26.906, df 4.00, p 1.143e-05); vs gpt2 +0.149084 (p 2.988e-04); vs sw40k-struct −0.106636 (t −20.575, df 3.73, p 5.647e-05); vs G1 flat hnt64k +0.025049 (t 2.431, df 2.74, p 0.101). Per-seed numbers: `per_seed.csv`; full summary: `results.json`.

Graph-build cost (CPU only; GPU graph-build FLOPs 0): hnt40k units 322.3 CPU-s + spectral clusters 277.4 CPU-s = 599.7 CPU-s (412 Laplacian matvecs, analytic 1.316e11 FLOPs, negligible against 1.62e15 train FLOPs). sw40k control: 97.0 s BPE + 0.9 s members.

## 2. Mission

Beat G1's two failure causes with the corpus hypergraph itself: compositional embeddings shared along derivation hyperedges (rare units), and a cluster-then-unit head over hypergraph clusters (head FLOPs), at V ≤ 64,000. Pass = G1 rule against struct32k and an equal-size superword control with identical machinery.

## 3. Execution (ORIENT)

- **Verdict: FAIL.** BPB fails (1.477304 > 1.329303); FLOPs passes (1.1929e7 < 3.0882e7 and < 1.2438e7).
- **What worked:** the exact factorized head cut head FLOPs/unit from 92,160,000 (flat tied, V = 40,000) to 921,600 and total FLOPs/byte to 0.3863 × struct32k. Hypergraph span selection beat superword BPE at equal size and machinery by 0.1066 BPB (p 5.6e-05).
- **What did not:** mean composition plus a rank-32 residual did not recover span-unit BPB; hnt40k-struct is statistically indistinguishable from flat hnt64k and 0.155 worse than struct32k.
- **One principled variant (v2):** order- and role-aware composition (per-slot diagonal gates, separate input/output tables) plus a zero-init full residual. Seed 13: 1.442863 (−0.041699 vs v1 seed 13), still 0.1136 above threshold. Incomplete: control and seeds 21/34 not run (see Sustainment).
- **Diagnostics:** the sentence hypergraph Laplacian has 4 near-zero eigenvalues (≥ 4 components); spectral clusters have marginal entropy 7.1663 bits vs 7.5717 random.
- **Threats to validity:** n = 3 (v2 n = 1); one size, one lr, no per-arm sweep; ~quarter-epoch budget; superword control trained on 40 MB; v2 throughput (51,985 units/s) reflects an unoptimized gather and is not a speed claim.

## 4. Sustainment (DECIDE)

- At 15:05 CT the memory watchdog condition fired (commit free 1.22 GB < 4 GB; an unrelated browser held 16.0 GB commit). The queue, the in-flight sw40k-struct-v2 seed-13 run (partial, no result) and its telemetry logger were stopped. Commit later recovered (23.85 GB free at 15:21).
- From then until at least 15:26 CT GPU 0 reported 100% utilization, P0, 45–49 W, 21 MiB, with no attributable process, so the idle-check rule blocked relaunch. Remaining v2 work: sw40k-struct-v2 seeds 13/21/34 and hnt40k-struct-v2 seeds 21/34 (about 45 GPU-min at measured v2 speed); the queue skips finished runs.
- Decision needed: whether to finish v2 once GPU 0 is genuinely idle, or stop the units line (G1 and G1b both fail on BPB) and redirect GPU time to the architecture arms.

## 5. Command & Signal (ACT)

- Launched: v1 queue (6 runs), v1 and v2 smoke tests (150 steps each), v2 queue (1 run completed). Telemetry: 1 s nvidia-smi log for every GPU window, now written to the data drive with spare room rather than the low-space drive.
- Limits held: GPU 0 only, idle check before each launch, 85 °C abort (max 61 °C), no clock or power changes, protected paths untouched, FineWeb not restarted and its partial file kept.
- Gate record: G1b "fail" entry added to the gates file. The dashboard currently shows only G0/G1 because another worker was rewriting the dashboard server concurrently; I stopped editing it.
- Nothing published, committed or pushed. This bundle is a local record.
