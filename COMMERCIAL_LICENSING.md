# Dual-Licensing & Commercial Deployment Posture

`HaDukeNN_LLM` (`sheaf-verify` and `hadukenn_core`) is distributed under a dual-license model aligned with `LICENSE` and `NOTICE`:

1. **Open Verification & Academic Research Tier (`LICENSE` — PolyForm Noncommercial License 1.0.0):**
   Public auditability, academic replication, noncommercial benchmarking, and independent verification of all gate ledgers (`P0..P6`, `G0..G6`, `H0..H7`, and `T-37..T-43`) remain open for inspection and execution without runtime dependencies (`dependencies = []` on `sheaf-verify`).

2. **Commercial Enterprise & Regulated Runtime Tier (`NOTICE`):**
   Per `NOTICE`, commercial production deployment—including embedding `sheaf-verify`, `LocalActionOntology`, `VerifiableGraphCheckpointer`, `IncrementalHypergraphKernel`, or `emit_regulatory_bundle` (`IETF SCITT RFC 9943`, `EU AI Act Article 12/14`, and `FINRA Rule 3110 WORM` compliance packages) inside for-profit products, client engagements, or internal enterprise operations—requires a separate commercial license from the copyright holder:
   - **Copyright Holder:** Blake Jones (`@ZuluYokohama` on [GitHub](https://github.com/ZuluYokohama/HaDukeNN_LLM)).
   - **Commercial Scope Options:**
     - *Tier A — Verifiable Agent Sidecar SDK:* Pure-Python stdlib `sheaf-verify` + `compliance.py` + `IncrementalHypergraphKernel` for deterministic state checkpointing, `ABSTAIN_TIE / RECOVERABLE_FAULT` contradiction gating, and regulatory receipt generation alongside any third-party or local LLM.
     - *Tier B — Sovereign Edge & Workstation Runtime:* `hadukenn_core` C11 AVX-512 / `RTX 5070` `Cluster-IS` surgery and `Snapdragon` INT8 NPU + CPU split adaptation for air-gapped deployments.
