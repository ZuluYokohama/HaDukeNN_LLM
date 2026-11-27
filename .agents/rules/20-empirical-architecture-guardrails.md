---
trigger: always_on
description: Empirical neural-and-hypergraph design guardrails from T-30, T-37, E1, T-38, and T-41..T-45.
---

# Empirical Architecture Guardrails

- **Keep Input Span Geometry + Dense Flat Heads for `V = 32,768` GPU Models**:
  - `struct32k` byte-span tokenization passed `16/16` scaling cells (`-0.0337` BPB at `M-1.5x`) and scales positively with model capacity (`dlog2k = +0.1246`).
  - Do not propose `182x181` class-factored output bottlenecks on `32k` GPU pretraining (`T-38` proved a ~25% effective-capacity tax across scales `XS..M+` with `alpha_body ~ 0.233` and `-23.9%` Tensor-Core TFLOPS loss).
- **Use Dimension-Free Relational Sheaf Transfer (`sheaf_bridge.py`) for Small Models**:
  - When distilling a high-dimensional embedding oracle into a small weighted student, use whitened SVD warm-start + sparse LLE barycentric & Kruskal H0 MST sheaf Laplacian (`L_bridge`) + sheaf-filtered context stalks rather than unwhitened Euclidean coordinate MSE.
- **Require Degree-Matched Chodrow Null Controls on All Hypergraph Claims**:
  - Every corpus-hypergraph or sheaf-prefix experiment (`kvreg-sheaf`, `E2` kNN-LM, `Sheaf-Bridge` stalks) must beat both the standard baseline and a degree- and edge-size-preserving [Chodrow (2020)](https://arxiv.org/abs/1902.09302) configuration-model null (`hypernull`).
