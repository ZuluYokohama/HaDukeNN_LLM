# Iteration `t45-oracle-sheaf-bridge` — Inverted Oracle-to-Small-LLM Relational Sheaf Bridge (`v0.8.0`)

Transfers dimension-free relational topology from a high-dimensional frozen embedding oracle into a compact small weighted student model (`src/hadukenn_core/sheaf_bridge.py`) without modifying the dense output head (`0` extra vocabulary-head FLOPs) and with full [IETF SCITT RFC 9943](https://www.rfc-editor.org/info/rfc9943/) attestation.

## Architecture Layers Verified Across 5 Seeds (`13, 21, 34, 55, 89`)

1. **Whitened Spectral Warm-Start (`whitened_spectral_warmstart`)**:
   - Extracts isotropic top-`d_student` SVD coordinates from the centered oracle embedding space (`mean_isotropy_ratio >= 0.25`), eliminating dead Zipfian tail vectors at initialization (`0` online FLOPs).
2. **Dimension-Free LLE + Kruskal `H0` MST Sheaf Laplacian (`build_sheaf_bridge_operator`)**:
   - Computes affine Locally Linear Embedding barycentric reconstruction weights `W` (`sum_j W[i, j] = 1`) combined with the global `H0` Kruskal minimum-spanning-tree Laplacian `L_mst`, regularizing student embeddings via `(I - W)^T (I - W) + 0.15 * L_mst` rather than distorted raw Euclidean distances.
3. **Sheaf-Filtered Context Register Stalks (`Sheaf-Bridge + Stalks` vs `Chodrow Null`)**:
   - Downweights high-coboundary distractor neighbors and beats degree-matched [Chodrow (2020)](https://arxiv.org/abs/1902.09302) null registers (`p < 0.01`) while keeping symmetric per-row INT8 quantization drift `<= 0.01` BPB for `14.5 W` looped NPU deployment.

## Reproduction

```bash
PYTHONPATH=src python3 src/hadukenn_core/sheaf_bridge.py
PYTHONPATH=src python3 -m sheaf_verify campaign --mode t45-sheaf-bridge \
  --path iterations/t45-oracle-sheaf-bridge/T45_SHEAF_BRIDGE_RESULTS.json
```
