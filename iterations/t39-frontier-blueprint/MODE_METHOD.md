# T-39 blueprint: mode and method

Summary of `T38_AMENDMENT1_STAGE3_FIT.json`, `T39_FRONTIER_PREREG.json`, and `hg_v3_gate_verified.json`.

## Mode

Two-track architectural falsification verified by `sheaf-verify` v0.2.0:

1. **Part A (Spectral Vocabulary Bookend):** Same-backbone, equal-bytes and equal-memory comparison of compiled flat/chunked cross-entropy against `NC × 128` Tensor-Core-aligned `tied_root + K_neg = 3` spectral heads across V ∈ {32,768, 131,072, 262,144}.
2. **Part B (Asymmetric Sheaf KV-Prefix Memory):** Equal-bytes comparison of `kvreg-sheaf` (`Q = 256, KV = 272`, diagonal sheaf restriction maps over self-excluded `hg-v3` stalks) against `kvreg-null` (Chodrow configuration-model hypergraph preserving exact vertex degrees and hyperedge sizes) and `pack-random`.

## Verification commands

```bash
sheaf-verify campaign --mode cluster-geometry --vocab 32768 --nc 256 --cluster-size 128 --struct-rank 0 --json
sheaf-verify campaign --mode t37-grid --path iterations/t37-grid/results.json --json
sheaf-verify campaign --mode t38-prereg --path iterations/t38-design/T38_PREREG.redacted.json \
  --addendum iterations/t38-design/T38_PREREG_ADDENDUM_PAIRED.json --json
sheaf-verify campaign --mode hg-v3 --path iterations/t39-frontier-blueprint/hg_v3_gate_verified.json --json
sheaf-verify campaign --mode flop-scaling --json
```

## Controls and statistical discipline

- Primary verdicts remain locked to pre-registered Welch tests with Holm-Bonferroni familywise control at α 0.01.
- Paired-seed Student-t, Smyth Empirical-Bayes tier-pooled paired-seed contrasts, and Kelly SAVI e-processes are reported as secondary diagnostics via `sheaf_verify.savi`.
- Any causal suffix canary > 0.0 immediately marks a hypergraph run `VOID`.
