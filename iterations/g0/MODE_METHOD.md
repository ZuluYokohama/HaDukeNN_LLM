# Mode and method — g0

## Mode

Unit counting on the valid split. The comparison arm is GPT-2 BPE. The recorded vocabulary size is 65,536. This is not a training mode and not a bits-per-byte measurement.

## Method

Inputs are the five recorded measurements in `results/g0.json`: valid units, GPT-2 BPE valid units, byte-fallback percent, build CPU-hours, peak RAM, and vocabulary size. The file is one JSON object of recorded metadata. It is not a corpus and not a tokenizer merge table.

The unit reduction is `1 - 181544 / 249887`, reported as 27.35%. The other fields are the build's own measurements, copied as recorded. This branch does not re-count the text.

## Controls

GPT-2 BPE unit count on the same valid split: 249,887. No superword-BPE control and no struct32k arm belong to this iteration.

## Gates

This iteration may speak to G0 only.

| Check | Threshold on main | Recorded value | Line |
| --- | --- | --- | --- |
| Fewer units than GPT-2 BPE | ≥ 15% | 27.35% | holds |
| Byte fallback | ≤ 1% | 0.048% | holds |
| Vocabulary | ≤ 64,000 | 65,536 | fails |
| Build CPU | ≤ 2 h | 0.093 h | holds |
| Peak RAM | ≤ 16 GB | 4.83 GB | holds |

G0 as a whole is partial, because the vocabulary line fails. This iteration does not loosen `G0_VOCAB_CAP`. It does not speak to G1.

## Out of scope

Training loss, bits per byte, FLOPs per byte, spectral or compositional arms, kernels, scaling, motif heads, and any claim that a pilot beat struct32k.
