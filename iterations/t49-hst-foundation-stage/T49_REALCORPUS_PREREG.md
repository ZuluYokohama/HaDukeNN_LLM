# T-49 real-corpus pre-registration (WikiText-103 raw)

Campaign id: `T-49-REAL-WIKI103`. This is **not** the 48,000-byte synthetic T-49 toy campaign.

The binding document is [`T49_REALCORPUS_PREREG.json`](T49_REALCORPUS_PREREG.json). It was registered in `iterations/t49-hst-foundation-stage/prereg-store/` **before** official training. Do not treat any number from this run as a result until the official 5 seeds finish and are registered through the training gates.

## Design (summary)

- **Corpus:** WikiText-103 raw v1 from `Salesforce/wikitext` (`refs/convert/parquet`), reconstructed and hashed. Data lives under `~/hadukenn/data/wikitext-103-raw/` and is never committed. Hash mismatch aborts.
- **Arms (fair tokenizer comparison):** HST-v1 vs Byte-BPE, both `V = 8192`, **identical** Dense-SP causal LM (`d=512, L=8, H=8, d_ff=1536`, RoPE, RMSNorm, SwiGLU, tied embeddings, ~31.5M params), identical AdamW recipe. Dense SP is the shared recipe, not a third arm.
- **Equal bytes:** both arms consume the same locked UTF-8 training-byte budget from `wiki.train.raw`. Optimizer step counts may differ.
- **Metric:** exact target-token bits-per-byte on `wiki.test.raw`, scored once at the end of each arm. Validation is for logging / first-eval sanity only.
- **Test:** `paired_t` + `one_sided_less` on 5 official seeds `(13, 21, 34, 55, 89)`, α = 0.01. No imputation if a pair is incomplete.
- **Budget lock:** target 200,000,000 bytes per arm-seed, scaled down by the pre-registered wall-clock rule after a discarded seed-0 probe, then frozen.
- **Guards:** pause at 83 °C, resume ≤ 80 °C, stop at 85 °C; AC required.

The hypothesis is pre-registered as a test, not as a promised win. No claims beyond the data.
