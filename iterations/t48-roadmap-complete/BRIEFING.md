# T-48 Complete Roadmap Execution: Live T-38 Scaling Audit & Learned Action Bridge

## 1. Scope & Verified Outcomes

This iteration completes the four-phase post-PR-#9 execution roadmap across both the empirical scaling track and the end-to-end neuro-symbolic runtime:

1. **Phase A — Live `T-38` Stage 1–3 Scaling Audit (`src/sheaf_verify/t38_scaling_audit.py`)**:
   - **Stage 1 Headline (`M`: `d=576, L=8, ff=1536`, `4` epochs, `5` seeds)**: Verified `KILL` (`mean_P = 1.209366` vs `mean_H = 1.228146`, `delta = +0.018780` BPB, `Welch SE = 0.001111`, `p_one_sided_H_worse = 2.31e-7`, `41.33` vs `31.44` TFLOPS).
   - **Stage 2 Complete Ladder (`XS, S, M, M+` across `1` and `4` epochs)**: All `8/8` cells reject in favor of flat dense `Arm P` under Holm-Bonferroni (`alpha = 0.01`), with 4-epoch gaps shrinking monotonically (`XS +0.032048` -> `S +0.024723` -> `M +0.018780` -> `M+ +0.015266`, `Spearman rho = -1.00`, keeping `stage3_stop` false).
   - **Scaling Exponent Parity (`N_body` in force)**: Across `XS..M+`, `alpha_H - alpha_P = -0.0311` (`95%` bootstrap CI `[-0.0609, +0.0030]`), confirming `PASS_scaling` fails (`0` inside CI) and the raw gap narrows solely from a multiplicative effective-parameter penalty (`k_eff = 0.65..0.75`).
   - **Stage 3 `L` (`d=768, L=12`) `muP` Width-Drift Diagnostic**: At `0.5` epochs, unscaled `Arm P` crossed over from `lr = 0.001` (`val 1.284416`) to `lr = 0.0005` (`val 1.284167`, `-0.000249` BPB) with a `1.561055` spike at `lr = 0.002`, while `L-1ep-P` reached `1.264082` mean test BPB at `45.36 TFLOPS`.
2. **Phase D — Learned Byte-Span + Sheaf-KV -> Hypergraph Action Bridge (`src/hadukenn_core/learned_action_bridge.py`)**:
   - Trains a causal `ByteBPETokenizer` (`2.2337` bytes/token) + cellular-sheaf incidence prefix encoder + `1/sqrt(d)` `muP`-scaled stalk-pointer policy (`d=96`, `2` recurrent loop steps) across `5` canonical seeds (`13, 21, 34, 55, 89`), quantized into a symmetric per-channel `INT8` NPU engine (`84,640` bytes arena `< 13.0 GiB` @ `14.5 W`).
   - Emits authentic [PR #9](https://github.com/ZuluYokohama/HaDukeNN_LLM/pull/9) `hadukenn.exec_trace.v1` JSON commands (`source`, `set`, `derive`, `fork`, `retract`, `query`, `assert`) directly into `IncrementalHypergraphKernel`.
   - Achieves **`85.00%` 5-seed OOD compositional accuracy** on held-out `7..8`-step multi-hop cyclic transfer + counterfactual fork + simultaneous main-branch revision + transitive retraction episodes versus **`34.44%`** on a matched-width flat neural baseline (**`+50.56` percentage points gain**), with **`0.076%` (`0.000758`) INT8 quantization disagreement**, **`100.0%` conservation-spoof refusal**, **`100.0%` deterministic semantic replay**, and verified [IETF SCITT RFC 9943](https://www.rfc-editor.org/info/rfc9943/) compliance bundles.

## 2. Reproducing & Verifying

```bash
PYTHONPATH=.:src python3 -m sheaf_verify campaign \
  --mode t48-roadmap-complete \
  --path iterations/t48-roadmap-complete/T48_ROADMAP_COMPLETE_RESULTS.json --json
```
