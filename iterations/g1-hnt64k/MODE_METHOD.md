# Iteration g1-hnt64k: mode and method

## Arms (3 seeds each: 13, 21, 34)

| Arm | Vocabulary | V | How it was built |
|---|---|---:|---|
| g1-gpt2 | GPT-2 byte-level BPE | 50,257 | stock |
| g1-struct32k | project 32k byte-level BPE | 32,768 | pre-existing project tokenizer, trained on WikiText-103 train |
| g1-hnt64k | GPT-2 tokens + 15,279 span units | 65,536 | see "Span vocabulary" |
| g1-sw64k | superword BPE (merges may cross spaces) | 65,536 | HF tokenizers BPE; pre-split only at newline and " . "; byte-level; trained on the first 40 MB of WikiText-103 train; min frequency 2; max token length 32 |

## Span vocabulary (hnt64k)

- **Source:** WikiText-103 train only. Validation and test are only encoded.
- **Candidates:** n-grams of GPT-2 tokens for n = 2..12, counted with a rolling 64-bit hash, keeping those with count ≥ 32.
- **Ranking:** savings = count × (n − 1). This is frequency × compression gain. The top 15,279 distinct n-grams became units.
- **Canonical path:** deterministic greedy longest match. Each byte string has one unit path, and encoding is lossless (decoded byte counts equal the official counts on every split). Scoring only this path gives a valid, conservative code length.
- **G0 numbers (valid split):**
  - 27.35% fewer units than GPT-2 (181,544 vs 249,887);
  - 0.048% of bytes coded by single-byte fallback units;
  - build cost 335.6 CPU-s (348.7 s wall), peak RAM 4.83 GB, zero GPU FLOPs.

## Model and training (identical body for all arms)

- **Architecture:** decoder-only transformer, 6 layers, d = 384, 6 heads, SwiGLU feed-forward (1,024), RMSNorm, RoPE (base 10,000), no biases. Embeddings are tied to the output layer. Residual projections use init std 0.02/√12. 10,621,824 non-embedding params.
- **Optimizer:** AdamW (β 0.9/0.95, eps 1e-8), weight decay 0.1 on matrices only, gradient clipping at 1.0.
- **Schedule:** learning rate 1e-3, linear warmup over 2% of steps, then cosine decay to 0.1×. Each run has its own full schedule.
- **Batching:** 48 sequences × 256 units per step (micro-batch 16), random windows from the training stream. bf16 autocast, no dropout.
- **Budget:** byte-matched at 1.36e8 train bytes. Steps = ⌈1.36e8 / (train bytes per unit × 12,288)⌉ (gpt2 2,441, struct32k 2,337, hnt64k 1,782, sw64k 1,756).
- **Eval:** non-overlapping 256-unit windows plus a masked remainder window over the full split. BPB_corpus = Σ nats / (ln 2 × official split bytes).
- **FLOPs:** per unit = 6 × (matmul params, including the tied output layer V × d) + 12·L·T·d attention. FLOPs/byte = per-unit FLOPs × units ÷ bytes seen.

## Gate rule (G1)

PASS if hnt64k's mean test BPB_corpus ≤ struct32k mean + 1 sd (same protocol, n = 3) **and** hnt64k's training FLOPs/byte is below both struct32k and the equal-size superword control. The statistics use Welch t-tests on test BPB_corpus.
