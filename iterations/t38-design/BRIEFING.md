# T-38 design: plain Transformer against the HaDukeNN structure component, same tokenizer, briefing

Times are US Central (CT), Wed Nov 25, 2026. **No T-38 result exists yet.** This folder holds the frozen pre-registration and its paired-seed addendum, written before any T-38 data. Everything marked (E) is an estimate.

## Observe

- T-37 (`iterations/t37-grid/`) showed a tokenizer effect: struct32k beats hnt40k in all 8 S/M contrasts on FineWeb-Edu, and hnt40k's per-byte penalty grows from S to M at 1× data (0.0346 → 0.0511 BPB). Those were flat-head runs, so they say nothing about the structured head.
- On FineWeb-Edu, struct32k and plain bpe32k were statistically tied in T-30 (p = 0.82, as recorded in the prereg).
- Nothing so far isolates the structure component on a shared, standard tokenizer. T-38 is built to do that.

## Orient

The question, as frozen: does the HaDukeNN structure component beat or match a tuned modern plain Transformer at equal training bytes, with the same tokenizer, data, and backbone, and does it change the scaling exponent?

- **Tokenizer:** plain 32k byte-level BPE (`fwedu-bpe32k`, V = 32,768) on both arms. That removes the tokenizer confound.
- **Arm P:** flat output head, structure off.
- **Arm H:** the same backbone plus the structure component only: spectral clusters (182 clusters × up to 181 units), compose mean, blocked factored head, rank-32 low-rank residual, members = identity, no spectral init. No roadmap-2 arm is included, because none had a recorded pass under its own frozen prereg at freeze time.

## Decide

Frozen at 6:39:34 AM CT (file time; the file's own `frozen_ct` field says 6:45 AM). Approved by Blake Jones at 6:34 AM CT: a staged ladder with 5 seeds at the headline cell.

| File | SHA-256 | Status |
|---|---|---|
| `T38_PREREG.json` (original, not in this repo) | `524fca093dc6c00e803b6f9998e69277f2804cc38e44ad71f94ffd0b88d63590` | Frozen, read-only, unedited |
| `T38_PREREG.redacted.json` (this folder) | `aec7341632806afee30df3ddc7b1ea40056a64029fb16832ebae030eaa996c02` | Public copy of the original. Five local path or drive strings are replaced with `<redacted: …>` text: `plan_doc`, `shared_args.root`, `shared_args.data_root`, the P2 entry in `prerequisites`, and `checkpoints.all_seeds_needs`. Every other byte is the same |
| `T38_PREREG_ADDENDUM_PAIRED.json` (this folder) | `34ffcbdc7fa2df0df2d8718ca5395b74dc873b727a87adb4b7d236486650bba1` | Frozen 9:23:50 AM CT, verbatim copy. Reported secondary only |

The addendum pre-declares a paired-seed analysis (d = BPB_H − BPB_P over shared seeds; mean, paired SE, one-sided paired t, 95% CI, Pearson r). It never decides a verdict. If paired and Welch disagree, both are reported side by side. It was approved by Blake Jones at 9:22 AM CT, and the generator confirmed that no T-38 data existed when it was written.

A separate Amendment 1 to the Stage 3 L-fit rule exists only as a **draft**. It is not approved and not in force, and it is not in this repository.

## Act

Status at the time of this commit:

- The struct cache for `fwedu-bpe32k-s` was built and verified in the GPU gap between 7:50 and 7:54 AM CT: 182 clusters, cluster sizes 98–181, all 32,768 units used exactly once. A CPU load check with both arms passed.
- A check of the 174 padding slots in the 182 × 181 cluster table found zero probability and zero gradient on pads. Gradients match a pad-free reference to within 2.4e-6.
- The S-size smoke pair (P and H, seed 13, 0.05 epoch) is queued behind the corpus-hypergraph packing runs. Stage 1 follows the smoke pair and the M learning-rate sweep.

## Caveats

- Wall-clock numbers are secondary. The plain arm is compiled but uses standard, unfused cross-entropy.
- The T-37 M model (ff 1,024, 6 heads) differs from the T-38 M model (ff 1,536, 9 heads). T-37 runs are not a T-38 pilot or baseline.
- At 4 epochs the top rungs see few tokens per parameter. The prereg's 1-epoch against 4-epoch design and its L-fit restriction are there to handle this.
