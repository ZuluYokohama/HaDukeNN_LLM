# Iteration t37-mup-probe: muP-scaled hidden learning rate at T-37 M, 0.25× bytes, briefing

Times are US Central (CT), Fri Nov 27, 2026, 8:00 to 8:09 PM. FineWeb-Edu subset `fwedu32`, test BPB_corpus, lower is better. Numbers come from the per-seed trainer records. The baseline row is `t37-sg-s32-M-b025` in `iterations/t37-grid/`.

## Observe

| Arm | seeds 13 / 21 / 34 | mean | sd |
|---|---|---:|---:|
| muP (`mup` on, d0 = 384) | 1.542126 / 1.556173 / 1.543023 | 1.547107 | 0.007864 |
| baseline, same size and bytes (t37-grid) | 1.571415 / 1.584377 / 1.587762 | 1.581185 | 0.008628 |

muP minus baseline is -0.034077 BPB (per seed -0.029289, -0.028204, -0.044739). Unequal-variance Welch, n = 3 against 3: t -5.06, df 3.97, two-sided p 0.0074. Computed here from the six values, descriptive only.

## Orient

The arm is the T-37 fwedu-struct32k M model (8 layers, d 576, 6 heads, ff 1,024, flat head, struct none) at 87,111,190 train bytes (0.25× of g1d's budget), with the hidden-layer learning rate scaled by muP from a reference width of 384. In the recorded trainer arguments, only `mup` and `mup_d0` differ from the baseline run. The other differences are flags that did not exist when the baseline ran and are at their defaults or off.

## Decide / Act

Recorded. No pass/fail is claimed. Gates touched: none.

## Caveats

- The baseline ran on an earlier trainer version than this run (the r2 trainer, promoted 2026-11-27; see `iterations/code-hash-records/`). The r2 flags are off in this run, and an equivalence check on the flags-off path is recorded there, but a same-version baseline rerun has not been done.
- n = 3, one size, one data budget (0.25×), one learning rate (1e-3 base). Whether the gain holds at larger budgets is untested here.
- This is one tokenizer contrast cell from T-37, not a T-38 run.
