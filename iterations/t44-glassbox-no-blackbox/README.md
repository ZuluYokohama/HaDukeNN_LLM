# Iteration `t44-glassbox-no-blackbox` — Zero-Black-Box ('Glass-Box') AI Suite (`v0.7.0`)

Eliminates opaque neural matrix parameters (`opaque_neural_parameters_in_glassbox = 0`) across sequence prediction, machine unlearning, relational rule induction, and sheaf conservation enforcement in pure Python 3.11+ standard library (`src/sheaf_verify/glassbox.py`).

## Subsystems Verified Across 5 Seeds (`13, 21, 34, 55, 89`)

1. **`CausalMDLSpanGraph` + `GlassBoxSequenceEngine` (`CTW-PY` + Hypergraph Schur-Complement Trace)**:
   - Induces reversible multi-byte spans strictly when Minimum Description Length bit-savings exceed dictionary overhead (`> 0` bits).
   - Combines variable-order Bayesian Context-Tree Weighting ([arXiv:2211.02676](https://arxiv.org/pdf/2211.02676)) and Hierarchical Pitman-Yor discounting ([Teh, 2006](https://aclanthology.org/P06-1124/)) with a 2-hop document-hyperedge Schur-complement trace, achieving **`0.629481` BPB** in **`44.54 ms`** versus **`3.263958` BPB** in **`1,108.94 ms`** for a matched Black-Box Neural AR-MLP (`-2.634478` BPB advantage, `paired p = 1.94e-6`, `SAVI E = 7.0175`, **`24.9x` speedup**).
2. **Exact `O(L)` Machine Unlearning (`unlearn_document`)**:
   - Decrements exact suffix sufficient statistics, provenance multisets, and hypergraph incidence in **`0.6985 ms`**, matching clean-room Merkle SHA-256 commitments with **`0.0` L1 probability drift** across all 5 seeds (compared against `0.005944` residual drift under gradient-ascent unlearning; see [Bourtoule et al., 2021](https://arxiv.org/abs/1912.03817)).
3. **`GlassBoxRuleAndSheafInducer` + `GlassBoxWorldKernel` (`Depth 5..8` OOD)**:
   - Uses Plotkin anti-unification to induce human-readable slot-role templates and exact rational Gauss-Jordan elimination over `fractions.Fraction` to discover linear conservation laws (`TankA + TankB + TankC + TankD == invariant_constant`) from `depth <= 3` trajectories.
   - Reaches **`100.0%` held-out `depth 5..8` OOD accuracy** (versus `3.2%` Black-Box MLP, `SAVI E = 12.1858`), **`100.0%` adversarial violation refusal** ([Hansen & Ghrist, 2019](https://arxiv.org/abs/1808.01513)), and **`100.0%` verified [IETF SCITT RFC 9943](https://www.rfc-editor.org/info/rfc9943/) bundles**.
4. **Reversible Provenance & Epistemic Abstention**:
   - `5/5` seeds passed reversible leave-one-document-out provenance verification and `EPISTEMIC_ABSTAIN` gating on unseen byte prefixes ([Rudin, 2019](https://arxiv.org/abs/1811.10154)).

## Reproduction

```bash
PYTHONPATH=src python3 src/hadukenn_core/glassbox_benchmark.py
PYTHONPATH=src python3 -m sheaf_verify campaign --mode t44-glassbox \
  --path iterations/t44-glassbox-no-blackbox/T44_GLASSBOX_RESULTS.json
```
