# Ready-to-Push PR & Post-T-38 Workstation Guide (`ZuluYokohama/HaDukeNN_LLM`)

## 1. Opening the GitHub Pull Request Atop Merged PR #9

Unpack `HaDukeNN_LLM-roadmap-complete.zip` into a clean directory separate from any active training folder, verify all 8 repository gates locally, and push a feature branch to [ZuluYokohama/HaDukeNN_LLM](https://github.com/ZuluYokohama/HaDukeNN_LLM):

```bash
git clone https://github.com/ZuluYokohama/HaDukeNN_LLM.git hadukenn-pr-worktree
cd hadukenn-pr-worktree
git checkout -b iter/t48-roadmap-and-learned-bridge

# Copy updated src/, tests/, iterations/, scripts/, .agents/, AGENTS.md, and COMMERCIAL_LICENSING.md
# from the unpacked HaDukeNN_LLM-roadmap-complete archive (keeping fixtures/passing untouched), then run:
PYTHONPATH=.:src python3 -m sheaf_verify.selftest
PYTHONPATH=.:src python3 -m sheaf_verify campaign \
  --mode t48-roadmap-complete \
  --path iterations/t48-roadmap-complete/T48_ROADMAP_COMPLETE_RESULTS.json

git add .
git commit -m "Add live T-38 Stage 1-3 scaling audit, learned hypergraph action bridge, and SCITT suite"
git push -u origin iter/t48-roadmap-and-learned-bridge
```

### Pre-Filled `.github/PULL_REQUEST_TEMPLATE.md` Fields
- **Gates touched**: Adds `t38_scaling_audit.py`, `learned_action_bridge.py`, `compliance.py`, `glassbox.py`, `cosimilar_trace.py`, and `--mode t48-roadmap-complete`; all 22 existing control-graph gates and `fixtures/passing` remain untouched.
- **Evidence files**: `iterations/t48-roadmap-complete/T48_ROADMAP_COMPLETE_RESULTS.json`, `ANTIGRAVITY_VERIFICATION_REPORT.json`.
- **Ledger head before**: `307711a4d841c2c8fa0bbd954d15b2a585012608528c9f53b9ea08ad717bd7c4`
- **Ledger head after**: `307711a4d841c2c8fa0bbd954d15b2a585012608528c9f53b9ea08ad717bd7c4` (unchanged)
- **Pre-registration / amendment status**: Frozen `T-38` Stage 1–2 verdicts and `25f13f2d` descriptive FLOP-control addendum preserved without modification.
- **Model weights / binary check**: `0` binary weight files included (`test_docs.py` verified).

## 2. Workstation Queue & Post-T-38 Reload Checklist

1. **Current Live Workstation State (Verified from Live Master Sheet)**:
   - Stage 2 (`XS, S, M, M+` at `1` and `4` epochs) is **100% complete** (`Spearman rho = -1.00`, `Holm 8/8` in favor of `P`).
   - `hold:after-T38` is **already active** on all 3 FLOP-control plans (`t38-flopmatch-P-M-s13/s21/s34`).
   - Stage 3 `L` (`d=768, L=12`) completed its `0.5`-epoch LR sweep and `L-1ep-P` (`1.26408` @ `45.36 TFLOPS`, `~30.5 min/seed`—running faster than the initial `20.4 h` estimate!). Note for the paper log: in the `L` `0.5`-epoch sweep on `P`, `lr = 0.0005` edged `lr = 0.001` on full-split `val_bpb` (`1.28417` vs `1.28442`) while `L-1ep-P` ran at `lr = 0.001`; verify whether `L` selected on `valid(partial)` or kept default `0.001`.
2. **Single Gap Reload After `L-4ep` + `t38-flopmatch-P-M` Finish**:
   - Restart the launcher service once in the idle gap after the 3 FLOP-control seeds finish.
   - Verify `kvreg-null-prefix-wt103` and `kvreg-sheaf-prefix-wt103` clear `hg-v3-multiview-build-wt103` (if `depends_on` still references the held build plan ID rather than `CACHE_SPECS`, remove that satisfied plan ID string in the gap).
   - Release `hg-e2-knnlm-wt103` and `hg-e1v3-prefix-causal-wt103` on GPU 1 while GPU 0 executes `kvreg-null/sheaf` and `r2` (`#47-#49`, plus `#45 muP` once re-queued).
