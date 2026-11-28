# T-38 results: plain Transformer against the HaDukeNN structure component, Stages 1 to 3, briefing

Times are US Central (CT). Stage 1 ran Nov 25, Stage 2 Nov 25 to 26, Stage 3 Nov 26 to 27, and the horizon-matched P control Nov 27, 2026. FineWeb-Edu subset `fwedu32`, plain 32k byte-level BPE on both arms, test bits per byte (BPB_corpus over 10,005,000 test bytes). Lower is better. The frozen pre-registration is in `iterations/t38-design/`. This folder holds the results. Numbers come from the per-seed trainer records, and the ladder fit is copied from the author's analysis output (the trainer and the analysis code are not in this repository).

**Summary.** The headline is **KILL**. At M and 4 epochs, H is 0.018780 BPB worse than P at equal bytes (5 seeds per arm). H is worse at every rung and data budget, and the gap shrinks with model size. The scaling-exponent fit does not favour H. At equal analytic FLOPs, P is still better by 0.004321. No claim that the structure component helps at this tokenizer, size range or data budget.

## Observe

**Stage 1 headline (M, 4 epochs = 2,162,281,720 train bytes, seeds 13, 21, 34, 55, 89).**

| | n | mean test BPB |
|---|---:|---:|
| P (plain, flat head) | 5 | 1.209366 |
| H (structure component) | 5 | 1.228146 |

H minus P is +0.018780. Welch SE 0.001111, df 7.21, one-sided p for H worse 2.31e-07. The pre-registered rule gives KILL (the mirror of PASS: one-sided Welch p < 0.01 with H worse, and a gap of at least 2 SE). The T-38 driver recorded the same verdict, SE and p at Stage 1 close, and they were recomputed here from `per_seed.csv`.

**Stage 2 and 3 cells, 4 epochs (3 seeds per cell except M).** Gap is mean H minus mean P. The Welch SE and two-sided p use all seeds in the cell.

| Rung | n P/H | P | H | Gap | Welch SE | Welch p (two-sided) |
|---|---:|---:|---:|---:|---:|---:|
| XS | 3/3 | 1.333279 | 1.365327 | +0.032048 | 0.000746 | 7.11e-05 |
| S | 3/3 | 1.260141 | 1.284864 | +0.024723 | 0.000793 | 6.70e-04 |
| M | 5/5 | 1.209366 | 1.228146 | +0.018780 | 0.001111 | 4.62e-07 |
| M+ | 3/3 | 1.194506 | 1.209772 | +0.015266 | 0.000623 | 1.87e-04 |
| L | 3/3 | 1.181805 | 1.195278 | +0.013473 | 0.000646 | 8.40e-04 |

**1 epoch (540,570,430 train bytes), 3 seeds per cell.**

| Rung | n P/H | P | H | Gap | Welch SE | Welch p (two-sided) |
|---|---:|---:|---:|---:|---:|---:|
| XS | 3/3 | 1.384750 | 1.423064 | +0.038313 | 0.001912 | 4.51e-05 |
| S | 3/3 | 1.321693 | 1.356107 | +0.034415 | 0.001863 | 8.69e-05 |
| M | 3/3 | 1.277560 | 1.304991 | +0.027430 | 0.000724 | 5.01e-04 |
| M+ | 3/3 | 1.266576 | 1.289548 | +0.022972 | 0.000586 | 6.33e-04 |
| L | 3/3 | 1.264082 | 1.282692 | +0.018610 | 0.001590 | 3.64e-04 |

Per-cell PASS/KILL labels and the Holm step over all rung × data cells are **not** computed in this record. The cells are numbers only.

**Size axis (what N means).** `N_body` is non-embedding parameters and is the same for both arms at each rung. The totals differ by the extra parameters of the structure component. At M, H has 51,906,496 total parameters against P's 50,734,656 (+2.3%).

| Rung | N_body | N_tot P | N_tot H |
|---|---:|---:|---:|
| XS | 3,213,568 | 11,602,176 | 12,705,536 |
| S | 10,621,824 | 23,204,736 | 24,335,488 |
| M | 31,860,288 | 50,734,656 | 51,906,496 |
| M+ | 49,575,040 | 70,546,560 | 71,732,096 |
| L | 84,953,856 | 110,119,680 | 111,332,608 |

## Orient

- **Equal bytes.** H loses to P in all 10 cells (5 rungs × 2 data budgets). At 4 epochs the gap falls with size: XS +0.0320, S +0.0247, M +0.0188, M+ +0.0153, L +0.0135. At 1 epoch it also falls monotonically (+0.0383 at XS to +0.0186 at L).
- **Stage 3 gate.** `stage3_stop` needs a KILL headline with a gap above 0.0117 and a Spearman rho of at least 0 between the 4-epoch gaps and N. The headline gap is above 0.0117, but rho over the five rungs is -1.00 (exact two-sided permutation p 2/5! = 0.0167), the same against N_body and N_tot. The gate did not fire and Stage 3 (L) ran.
- **Amendment 1 (L-fit restriction).** Neither arm met the rule, so L stays in the primary fit (drop_L = false). The 5-rung fit is the primary one.
- **Scaling fit (N_body, 5 rungs, 2,000-rep seed bootstrap).** alpha_H minus alpha_P is -0.0278, 95% CI [-0.0493, -0.0046]. Fits: P has E 1.0974, A 28.59, alpha 0.3201; H has E 1.0861, A 22.32, alpha 0.2923. The CI is below 0, so the fit does not show H scaling better. PASS_scaling needs the CI to exclude 0 in H's favour. The 4-rung preliminary (before L finished) was [-0.0599, +0.0002], which included 0. Two cautions: a three-parameter fit on five points is loose, and on the N_tot axis (sanity contrast only, P and H do not share N_tot) the sign flips and the CI includes 0. The conclusion depends on N_body being the right size axis, as the prereg and Amendment 1 specify.
- **Equal analytic FLOPs (secondary, descriptive).** H needs fewer analytic FLOPs per token than P at equal bytes, so Stage 1 H used 9.755e16 train FLOPs against P's 1.498e17. The horizon-matched control trains P alone at M, seeds 13, 21, 34, natively to H's 9.755e16 (24,925 steps, cosine to 0.1×; about 2.60 epochs). P scores 1.223825 (sd 0.000863) against H's 1.228146. H minus P is +0.004321, 95% CI [+0.001761, +0.006881], Welch SE 0.001035, df 5.74, two-sided p 0.00644, one-sided p for H worse 0.00322. So the equal-FLOPs deficit is about 23% of the equal-bytes deficit and is still significant at alpha 0.01 (two-sided). This control is secondary and does not change the headline. It compares 5 H seeds with 3 P seeds, and wall-clock cost is not part of it.

## Decide

Recorded as results. The T-38 verdict stands as written in the pre-registration: KILL at Stage 1. Nothing in this folder amends a rule. No G, H or P gate status changes.

## Act

Files in this folder:

- `per_seed.csv`: every T-38 run (87 runs: smoke, learning-rate sweeps, Stage 1, 2, 3 cells, and the horizon-matched P control) with test and validation BPB, parameters, train bytes and FLOPs.
- `results.json`: headline, cells, ladder and fit, Amendment 1 outcome, and the horizon-matched control.
- `T38_PREREG_AMENDMENT_1.json`: verbatim frozen copy, SHA-256 `87b570ced8ef6410636d7baf108af5924a8d1db72bcd839ee93fb4a48da5f160`. It supersedes only `ladder_rule.fit_restriction`.
- `T38_PREREG_ADDENDUM_FLOPMATCH.redacted.json`: public copy of the frozen horizon-matched addendum. Original SHA-256 `25f13f2df8f6d60d46da42b5ba5581405ae119afc75346c6046af2e72221bed4` (CRLF line endings). The copy differs from the original in two ways only: CRLF became LF, and the trainer script's file name in the `steps_rule` text was replaced by the word `trainer`. This copy's SHA-256 is `f7bdbc8df66fce04d9f894a29e052e146be06ddf89e43a9bf42aada0d737e3cc`.

The original `T38_PREREG.json` (`524fca09…`) and the paired addendum (`34ffcbdc…`) are unchanged and are in `iterations/t38-design/`.

## Caveats

- n = 3 per cell, and 5 at M-4ep. One tokenizer, one dataset (FineWeb-Edu subset), flat output head for P.
- Equal-bytes and equal-FLOPs are different comparisons. P reaches a lower BPB than H in both. H's lower per-token FLOPs and its throughput are not scored here. Wall-clock numbers are secondary and not included.
- The ladder fit is copied from the author's output, not recomputed in this repository. The cells, headline and equal-FLOPs Welch are recomputed from `per_seed.csv` and match the author's records.
- At 4 epochs the top rungs see few tokens per parameter. The prereg's 1-epoch and 4-epoch design and its L-fit restriction are there for that.
- The T-37 M model (ff 1,024, 6 heads) differs from the T-38 M model (ff 1,536, 9 heads), so T-37 rows are not a T-38 baseline.
- The mid-run FLOP-matched curves and the secondary readouts (lm-eval, rare-token probe, cross-domain) are not part of this record.
