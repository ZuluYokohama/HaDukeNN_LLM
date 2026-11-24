# Iteration g1b-structured: structured hypergraph vocabulary (G1b), briefing

Times are US Central (CT), Mon Nov 23, 2026. All numbers in sections 1–5 are measured unless marked analytic; none are estimates. Those sections are the record as written that day.

**Amendment, 2026-11-23.** The v2 remainder was later run as G1c. The n=1 seed-13 figure 1.442863 in section 1 is not the G1c result. G1b v1 stays a fail (test BPB 1.477304). The G1c pass is compute-matched for hnt40k-struct-v2 on WikiText-103, n=3: test bits per byte 1.269376 ± 0.004417 at 1.2061e7 train FLOPs per byte and 4.2006e15 train FLOPs. The sw40k-struct-v2 control is 1.322698 ± 0.006710. struct32k is 1.322354. The GPT-2 tokenizer arm is 1.328219. hnt versus that control is -0.0533 (p 7.1e-4). g1d, data-matched struct32k at 348.3M bytes (the same train bytes hnt saw), is 1.194155 ± 0.000824 at 1.0756e16 train FLOPs and beats hnt by 0.075 at equal data. G1c is a compute-efficiency win, not a per-byte modeling win. g1g, hnt40k-struct-v2 FLOPs-matched to g1d at 1.0756e16 train FLOPs, scored 1.174315 ± 0.000564 (seeds 1.174805, 1.173698, 1.174444). That is 0.019839 below g1d (Welch p 1.3e-5), That is a win at equal analytic train FLOPs only (the 6N + attention + head count). The structured model swaps the flat 40k output softmax for a factored cluster-then-unit head (200 clusters × 200 units). At d=384 the flat head is about 56.6% of plain-model FLOPs per token, so struct-v2 counts about 2.22× fewer FLOPs per token. At equal FLOPs, g1g therefore saw 891.7M train bytes (about 1.65 epochs, against g1d's 0.64), and it took about 1,488 s per seed against about 501–505 s for g1d and g1h, so it is about 3× slower on wall clock. g1h, plain hnt40k with the structured path off (struct none, mean compose, low-rank residual, no spectral init) at the same 1.0756e16 train FLOPs, scored 1.197637 ± 0.000517 (seeds 1.198146, 1.197111, 1.197653). That is 0.023321 above g1g (p 8.4e-7) and 0.003482 above g1d (p 0.006): at equal analytic FLOPs and the same tokenizer, the structured run scores lower, but it also saw about 2.2× the data. At equal data the order reverses. G1c, the same hnt40k-struct-v2 model at 348.3M train bytes, is 1.269376 ± 0.004417. That is 0.0579 worse than plain hnt40k at the same bytes (g1i, Welch p 8.8e-5) and 0.0752 worse than struct32k (g1d, p 8.2e-4). The mechanism resembles a class-factored softmax (Goodman 2001, arXiv:cs/0108006). This is not evidence that the structure layer models text better. Wall-clock-matched and 2× data-ladder controls are pending. g1i, the same plain hnt40k at g1d's 348.35M train bytes (9.350e15 train FLOPs), scored 1.211442 ± 0.003395 (seeds 1.210431, 1.215228, 1.208666). That is 0.017287 above g1d (p 0.0094), so at equal bytes plain hnt40k loses to struct32k. g1i and g1d use different tokenizers, so this is a tokenizer contrast, not a structure ablation. Caveats: n=3, one model size, WikiText-103 only. Parameter totals are verified against the per-seed trainer records (the trainer itself is not in this repository): hnt40k-struct-v2 39,943,040, struct32k 23,204,736, and plain hnt40k 25,981,824, each with 10,621,824 non-embedding. Every run applied RoPE. The trainer's structured model path applies RoPE only and ignores `--pe heat`, so no heat positional encoding has been tested.

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
- The 2026-11-23 amendment at the top supersedes this "v2 not finished" decision. G1b v1 remains a fail. G1c is the compute-matched v2 result. g1g, g1h, and g1i are recorded in `iterations/`.

## 5. Command & Signal (ACT)

- Launched: v1 queue (6 runs), v1 and v2 smoke tests (150 steps each), v2 queue (1 run completed). Telemetry: 1 s nvidia-smi log for every GPU window, now written to the data drive with spare room rather than the low-space drive.
- Limits held: GPU 0 only, idle check before each launch, 85 °C abort (max 61 °C), no clock or power changes, protected paths untouched, FineWeb not restarted and its partial file kept.
- Gate record: G1b "fail" entry added to the gates file. The dashboard currently shows only G0/G1 because another worker was rewriting the dashboard server concurrently; I stopped editing it.
- Nothing published, committed or pushed. This bundle is a local record.
