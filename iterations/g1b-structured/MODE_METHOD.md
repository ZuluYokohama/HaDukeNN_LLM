# Iteration g1b-structured: mode and method

## Arms (3 seeds each: 13, 21, 34)

Baselines are reused from iteration g1-hnt64k (g1-gpt2, g1-struct32k; g1-hnt64k and g1-sw64k listed for reference). The flat-vocabulary code path is numerically unchanged, so they remain comparable.

| Arm | Vocabulary | V | Embedding | Output head | Clusters |
|---|---|---:|---|---|---|
| g1b-hnt40k-struct | struct32k tokens + 7,232 span units | 40,000 | compose-mean + rank-32 residual | cluster-then-unit, exact | hypergraph spectral |
| g1b-sw40k-struct (control) | superword BPE | 40,000 | same | same | random, capacity-balanced |
| g1b-hnt40k-struct-v2 (variant) | as hnt40k | 40,000 | compose-slot (order/role-aware) + full residual | same | hypergraph spectral |
| g1b-sw40k-struct-v2 (control) | as sw40k | 40,000 | same as hnt v2 | same | random, capacity-balanced |

## Vocabulary builds (WikiText-103 train only; valid/test only encoded)

- **hnt40k:** base = the project 32k byte-level BPE (32,768 ids). Candidates are n-grams of base tokens, n = 2..12, count ≥ 32. Ranked by savings = count × (n − 1) (gain × frequency); top 7,232 added, V = 40,000 (inside the 64,000 cap). Greedy longest-match canonical encoding, lossless.
  - Valid: 22.29% fewer units than the base tokenizer, 25.57% fewer than GPT-2; byte-fallback 0.0452%. Test: 25.79% fewer units than GPT-2.
  - Train stream: 89,036,259 units, 6.0713 bytes/unit; 39,123 of 40,000 units occur.
  - Derivation hyperedges ("members"): each unit's ordered base-token decomposition, 59,991 member entries, max 12 members.
- **sw40k (control):** HF tokenizers byte-level BPE with superword merges; pre-split only at newline and " . "; first 40 MB of train; min frequency 2; max token length 32; V = 40,000. Train stream 92,798,865 units. Members = greedy longest-match decomposition of each token's bytes into base-32k tokens: 129,098 entries (3.227 per unit), max 10.
- **Hypergraph clusters (hnt40k):** incidence H = units × sentence hyperedges over the train stream (a sentence ends at a unit whose bytes end in "." or contain a newline); 2,579,611 hyperedges, 79,841,636 nonzeros, mean hyperedge size 30.95. Zhou normalized hypergraph Laplacian applied through incidence matvecs (no clique expansion); 32 smallest nontrivial eigenvectors by Lanczos (412 matvecs; analytic cost 4 × nnz per matvec ≈ 1.316e11 FLOPs). Coordinates row-normalized; units with degree < 3 (1,112) get zero coordinates. Spherical k-means (15 iterations) into NC = 200 centroids, then capacity-balanced greedy assignment (most confident units first) into clusters of S = 200 slots. Cluster-marginal entropy 7.1663 bits (max cluster probability 0.03288).
- **Random clusters (sw40k):** uniform random capacity-balanced assignment, NC = 200, S = 200. Cluster-marginal entropy 7.5717 bits.

## Structured model

- **Body:** identical to G1 (6 layers, d = 384, 6 heads, SwiGLU 1,024, RMSNorm, RoPE, no biases; 10,621,824 non-embedding params).
- **Embedding v1 (compose-mean):** e(u) = mean over members m of E_base[m] + A[u]·B, A ∈ R^{V×32} zero-init, B ∈ R^{32×d}. Same table for input and output.
- **Embedding v2 (compose-slot + full residual):** e_in(u) = mean_i(G_in[i] ⊙ E_base[m_i]) + A[u]·B + D[u], e_out(u) = same with G_out. G_in, G_out ∈ R^{12×d} per-slot diagonal gates (init 1, no weight decay), D ∈ R^{V×d} zero-init. Rationale: derivation hyperedges are ordered and a span is read (input) and predicted (output) by different roles; the mean is order-invariant, and a rank-32 residual cannot let frequent units depart from their members.
- **Spectral init:** E_base = √0.5 · random N(0, 0.02) + √0.5 · (spectral coords of base units, per-dim standardized, random orthonormal projection to d, rescaled to std 0.02). hnt arms only.
- **Head (exact, normalized):** p(u|h) = softmax_c(h·C)[c(u)] × softmax over the S slots of cluster c(u) of (h·e_out). Verified on CPU: Σ_u p = 1 within 3e-7; causal leak 0.0.
- **FLOPs/unit (analytic, counted honestly):** 6 × body matmul params + 12·L·T·d attention + head 6·d·(NC + S) + per-microbatch table build 3·(composition + low-rank 2·V·r·d [+ V·d full residual], ×2 tables in v2) ÷ 4,096 units per microbatch.
- **Training:** AdamW (0.9/0.95), lr 1e-3, wd 0.1 on matrices only, clip 1.0, warmup 2% then cosine to 0.1×, 48 × 256 units/step, bf16 autocast; byte-matched at 1.36e8 train bytes (hnt40k 1,823 steps; sw40k 1,900 steps).
- **Eval:** full official WikiText-103 test and validation, non-overlapping 256-unit windows plus masked remainder; BPB_corpus = Σ nats / (ln 2 × official split bytes).
- **Graph-build cost:** CPU only; GPU graph-build FLOPs = 0. Reported in CPU-seconds separately.

## Gate rule (G1, applied to G1b)

PASS if the hnt arm's mean test BPB_corpus ≤ struct32k mean + 1 sd (1.329303) **and** its train FLOPs/byte is below both struct32k (3.0882e7) and the equal-size control with the same machinery. Statistics: Welch t-tests on test BPB_corpus, n = 3 per arm.
