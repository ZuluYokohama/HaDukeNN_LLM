# Recorded G0 metadata

`g0.json` is the feasibility build's G0 record. It is recorded metadata. It is not corpus bytes, not WikiText, not a training run, and not model weights.

The counts are 27.35% fewer units on the valid split (181,544 vs 249,887), 0.048% byte fallback, 0.093 CPU-hours, 4.83 GB peak, and `V = 65,536`. The written gate caps vocabulary at 64,000. 65,536 is 2^16 and is sometimes also called 64k. The harness flags that inconsistency: the other G0 checks pass, and `g0.vocab_cap` fails.

This directory is not a training store and it is not the synthetic study in `fixtures/passing/`.
