# g1b-structured progress (times CT; the training machine clock runs ~3 min behind the box)

Publication note: drive letters and the training-machine name in this log were omitted before publication. Counts, times, and gate results are unchanged.

Later public results are the 2026-11-24 amendment in `BRIEFING.md` and the docs site. This log stops when the G1c chain was launched. It does not contain the G1c, g1d, or g1g numbers, and it does not contain a g1g result. g1g is pending.


- 14:25 builds done (hnt40k units, sw40k superword, members, clusters spectral/random) on the data drive.
- 14:29 smoke v1 (hnt40k-struct, 150 steps) OK.
- 14:29:57-14:50:23 v1 queue done: hnt40k-struct & sw40k-struct, seeds 13/21/34, all exit 0.
  - hnt40k-struct test BPB 1.477304 ± 0.007156 (s13 1.484562, s21 1.470255, s34 1.477094); F/B 1.1929e7
  - sw40k-struct 1.583940 ± 0.005420 (s13 1.582769, s21 1.589849, s34 1.579200); F/B 1.2438e7
  - G1 rule: BPB FAIL (threshold 1.329303), FLOPs OK -> FAIL. Report json written on the training machine (path omitted).
- Variant v2 (principled): --compose slot (order/role-aware per-slot gates, separate in/out) + --resid full (zero-init V x d residual). Code deployed; CPU test sum p = 1 within 3e-7, leak 0.0.
- 14:55 smoke v2 OK (150 steps, test 2.056846 vs v1 smoke 2.0777; 49-52k units/s; peak 2,832 MiB).
- 14:56:49 v2 queue (hnt40k-struct-v2, sw40k-struct-v2) launched. hnt40k-struct-v2 s13 done 15:04:11 exit 0.
- 15:05 (the training machine clock) WATCHDOG: commit free 1.22 GB (<4 GB). Chrome (not ours) at 16.0 GB commit. Killed queue (pid 16332), sw40k-struct-v2 s13 run (pid 16016, partial, no result), smi logger 27500. Commit free recovered to 7.47 GB. GPU 0 then showed 100% util at 21 MiB with no our process -> not idle; must recheck before resuming.
- TODO: resume v2 remaining 5 runs only when GPU 0 idle and commit headroom allows (training needs ~6.2 GB commit; must stay >=4 GB free). Then report, gates, PAPER_PLAN 8.6, Page, bundle.
- 15:06-15:08 (the training machine) state check: no python of ours running (only GUI server.py pid 17632 and a pre-existing 11:22 nvidia-smi logger pid 15688, not ours). Results on the data drive = 6 v1 + hnt40k-struct-v2 s13. GPU 0 reports 100% util, P0, ~45 W, 2.9 GHz, 21 MiB with no listed process (before kill: P8, 3.3 W). Not idle -> not launching. Commit free 6.91 GB and falling (chrome 19.25 GB commit, not ours). Waiting on idle poll (up to 12 min).
- hnt40k-struct-v2 s13: test 1.442863, valid 1.415810, F/B 1.2060e7, flops/unit 73,230,469 (table 1,529,990), params 39,943,040, 1,823 steps, 430.9 s, 51,985 units/s, peak 2,832 MiB, max 52 C, 14:56:58-15:04:09.
- Graph-build cost: hnt40k units 322.3 CPU-s (323.0 s wall, peak 4.63 GB) + spectral clusters 277.4 CPU-s (275.9 s wall; 412 matvecs, analytic 1.316e11 FLOPs) = 599.7 CPU-s; GPU 0 FLOPs. sw40k: superword BPE 97.0 s wall + members 0.9 s + random clusters 0.0 s.
- Diagnostic: hypergraph Laplacian has 4 eigenvalues ~0 (lambda 0,0,0,0,0.002262,...): sentence hypergraph has >=4 components, so first spectral coords are partly component indicators.
- 15:21 bundle results.json (scrubbed, 0 path hits) + per_seed.csv written; report rerun on the training machine incl. v2 s13 (n=1). sw40k steps = 1,900 (fixed in MODE_METHOD). GPU 0 still 100%/P0/~48 W/54 C at 15:20 with no attributable process; Windows engine counters show 0% on the 5070 adapter.
- 15:22-15:26 gates.json: added G1b entry (status fail, detail, label, note). GUI /api/gates hardcodes [G0, G1]; I patched server.py to append extra gates and restarted it, but another worker was editing viz-gui concurrently (server.py rewritten 15:23, app.js 15:20, live_watch.py 15:22) and is running its own server (system python pid 9916). My patch was overwritten; my restarted server exited; I removed my backup and stopped touching viz-gui. GUI currently shows G0 partial, G1 fail; G1b only in gates.json.
- GPU 0 still 100%/P0/~49 W/55 C at ~15:26 with 21 MiB, no process. Remaining v2 runs NOT launched.
- 15:30 PAPER_PLAN §8.6 added (draft also in plan_sec86_draft.md).
- 15:33 Page updated (new G1b section + TOC). BRIEFING.md written. Bundle complete except v2 remainder.
- 15:26 (the training machine clock) final check: no train/queue/logger processes of ours; GPU 0 still 100%/P0/48.5 W/55 C/21 MiB; commit free 23.87 GB; local volume free-space figures omitted. STOPPED: v2 remainder (sw40k-struct-v2 s13/21/34, hnt40k-struct-v2 s21/34) awaits genuinely idle GPU 0 + go-ahead. Resume cmd: g1b_queue.ps1 -Arms hnt40k-struct-v2,sw40k-struct-v2 (skips done runs), then g1b_report.py.
- 15:28 Blake approved closing Chrome on the training machine (graceful first) and public push of bundles by a separate cloud agent (PolyForm Noncommercial). Next: close Chrome, GPU check, finish v2, then G1c compute-matched.
- 15:29:07-15:29:25 (the training machine clock) Chrome closed gracefully: 10 processes (0.90 GB commit; the earlier 16 GB instance had already gone), 1 window "CloseMainWindow -> True", 0 remaining after 15 s, no Stop-Process needed. Commit free 23.78 -> 24.79 GB. A local log was written (path omitted). No other apps touched.
- GPU 0 check: nvidia-smi -q: Utilization GPU 100 %, Memory 0 %, Processes None, no throttle reasons. 5 s bf16 4096^3 matmul: 2,380 matmuls in 5.004 s = 65.37 TFLOP/s (normal). Afterwards util read 0 %, 21 MiB, P1. Conclusion: stale counter; GPU treated as idle.
- Speed fix (numerically equivalent forward): v2 slowness was the backward of G[slot] (index_put accumulate into 12 rows; profiler 235.9 ms of 377.7 ms CUDA per 5 steps). Replaced by _SlotGather (index_select forward, bitwise-equal output; backward onehot^T @ grad in fp64; G-grad max abs diff 5.4e-4 on max 267.7 vs old path). Also base gather -> index_select. Profile (16x256 micro, fwd+bwd+AdamW): slot+full 82.3 -> 45.2 ms (49,783 -> 90,612 units/s); mean+lowrank 29.4 -> 27.6 ms. CPU test still sum p = 1 (within 5e-7), leak 0.0. Note: hnt40k-struct-v2 s13 ran with the old gather.
- Queue now has a memory guard (mem_guard.ps1: kills train_paper.py and stops queue if commit free < 4 GB, 2 s poll), --budget-flops mode, prefix/root params.
- 15:33:27 (the training machine) chain launched: finish v2 (g1b root), then G1c (prefix g1c, data-drive path omitted, budget 4,200,626,191,859,712 FLOPs = exact G1 struct32k train FLOPs; 3 seeds each of hnt40k-struct-v2, sw40k-struct-v2). Idle check before: 0 %, 21 MiB x5; commit free 24.86 GB.
