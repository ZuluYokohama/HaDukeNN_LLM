---
name: run-sheaf-bridge
description: Executes and verifies the 5-seed T-45 Inverted Oracle-to-Small-LLM Relational Sheaf Bridge benchmark (whitened SVD warm-start, sparse LLE + H0 MST sheaf Laplacian, sheaf context stalks vs Chodrow null, INT8 drift, and SCITT RFC 9943 bundle). Do not use when editing pure-stdlib sheaf_verify ledger internals unrelated to T-45.
---

# Run & Verify Oracle-to-Small-LLM Sheaf Bridge (`/run-sheaf-bridge`)

Use this skill to execute, extend, or verify the `T-45` relational topology distillation pipeline (`src/hadukenn_core/sheaf_bridge.py`).

## Workflow

1. Run the bundled benchmark & campaign verification script (`requires numpy`):
   ```bash
   python3 .agents/skills/run-sheaf-bridge/scripts/run_bridge_benchmark.py
   ```
2. Verify all four `T-45` campaign criteria pass via `sheaf-verify`:
   ```bash
   PYTHONPATH=src python3 -m sheaf_verify campaign --mode t45-sheaf-bridge \
     --path iterations/t45-oracle-sheaf-bridge/T45_SHEAF_BRIDGE_RESULTS.json
   ```
3. Confirm `max_int8_quant_drift_bpb <= 0.01`, `delta_sheaf_vs_rigid_coord_bpb < -0.02` (`p < 0.01`), `delta_stalk_vs_chodrow_null_bpb < -0.02` (`p < 0.01`), and `scitt_rfc9943_verified == true`.

## Safety

- Writes JSON outputs strictly inside `iterations/t45-oracle-sheaf-bridge/`; never writes binary `.pt`, `.bin`, or `.safetensors` weights into git.
- Uses only the public test seal vector (`sheaf-verify-public-test-vector`) for benchmark attestation; never store production signing keys inside the workspace.
