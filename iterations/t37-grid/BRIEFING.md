# Iteration t37-grid: tokenizer scaling grid on FineWeb-Edu, briefing

Times are US Central (CT), Wed Nov 25, 2026. Numbers come from the per-seed trainer records and the grid fit output (run 7:52 AM CT, 2,000 seed-bootstrap reps). FineWeb-Edu subset `fwedu32` test bits per byte (BPB_corpus over 10,005,000 test bytes). Lower is better.

**Scope.** Both arms use plain flat output heads with the structured path off. The only difference between the arms is the tokenizer: `fwedu-hnt40k` (h40) against `fwedu-struct32k` (s32). Every gap below is a tokenizer effect. None of it is evidence about the structured head.

## Observe

16 cells: 2 model sizes × 4 byte budgets × 2 tokenizers, seeds 13, 21, 34 in every cell. Budgets are multiples of g1d's 348,328,219 train bytes. Gap = BPB(h40) − BPB(s32); positive means s32 is better.

| Size | Budget | h40 BPB | s32 BPB | Gap | SE | One-sided p | log2 k [95% CI] | Holm (α 0.01, 8 cells) |
|---|---|---:|---:|---:|---:|---:|---|---|
| S | 0.25× (87,082,055 B) | 1.734986 | 1.583445 | 0.151541 | 0.011636 | 2.10e-03 | 0.713 [0.648, 0.783] | yes |
| S | 0.5× (174,164,109 B) | 1.522494 | 1.441415 | 0.081079 | 0.011434 | 8.16e-03 | 0.472 [0.362, 0.577] | no |
| S | 1.0× (348,328,219 B) | 1.391735 | 1.357179 | 0.034557 | 0.002154 | 5.94e-05 | 0.291 [0.253, 0.335] | yes |
| S | 1.5× (522,492,328 B) | 1.334097 | 1.325217 | 0.008880 | 0.001476 | 6.08e-03 | 0.090 [0.066, 0.115] | no |
| M | 0.25× (87,082,055 B) | 1.743289 | 1.581185 | 0.162104 | 0.006171 | 1.32e-05 | 0.728 [0.675, 0.799] | yes |
| M | 0.5× (174,164,109 B) | 1.520579 | 1.420412 | 0.100168 | 0.009466 | 4.15e-03 | 0.545 [0.480, 0.644] | no |
| M | 1.0× (348,328,219 B) | 1.375478 | 1.324365 | 0.051113 | 0.001918 | 4.33e-04 | 0.412 [0.371, 0.452] | yes |
| M | 1.5× (522,492,328 B) | 1.323749 | 1.290067 | 0.033682 | 0.001615 | 5.11e-05 | 0.381 [0.338, 0.435] | yes |

k is the effective data multiplier: the factor by which h40 would need more bytes to reach s32's BPB at the same size, from h40's own loss-versus-log2(bytes) curve.

## Orient

- **Cell rule.** All 8 contrasts (16 cells) pass the pre-registered cell rule (gap ≥ 2 SE and one-sided Welch p < 0.01, n = 3 per arm). The kill check (gap ≤ 0 at M-1.5×) did not fire: M-1.5× gap is 0.0337 (p 5.1e-05).
- **Multiplicity.** Holm-Bonferroni at family α 0.01 over the 8 S/M contrasts keeps 5: M-0.25×, M-1.5×, S-1×, M-1×, S-0.25×. M-0.5× (p 0.00415) fails the 0.01/3 step, so S-1.5× (p 0.00608) and S-0.5× (p 0.00816) fail too. This Holm step is derived from the fit's one-sided p values.
- **Data axis.** s32's edge shrinks with more data at both sizes. The OLS slope of log2 k on log2 D is -0.232 at S (95% CI -0.270 to -0.194) and -0.137 at M (-0.176 to -0.104). Both CIs are below 0, so the pre-registered bias-variance view is supported and the sample-complexity view is not.
- **Size axis.** The edge is larger at M than at S: mean Δlog2 k (M − S) over the four budgets is +0.125 (95% CI +0.080 to +0.173). That rules out "it only helps small models" for this tokenizer contrast, at two sizes.
- **Learning rate at M.** At M-0.25× s32, lr 5e-4 (1.598370) and 2e-3 (1.646708) both score worse than lr 1e-3 (1.581185), so the grid's lr 1e-3 is kept for M. This was checked on one cell only.
- **Engineering check.** S-1× compiled gap 0.034557 against an eager-mode gap of 0.02625, inside the pre-registered 0.010 tolerance.

## Decide / Act

Recorded. The grid fixed the tokenizer for T-38: plain 32k BPE on both arms (see `iterations/t38-design/`). It does not change a G-gate status.

## Caveats

n = 3 per cell, two model sizes, one dataset (FineWeb-Edu subset), flat heads only. At M the largest budget is 0.97 passes over the 540,570,430 train bytes. The M model here (d 576, 8 layers, ff 1,024, 6 heads) is not the T-38 M model (ff 1,536, 9 heads), so these runs are not a T-38 pilot or baseline.
