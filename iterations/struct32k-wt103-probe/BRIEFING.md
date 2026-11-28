# Iteration struct32k-wt103-probe: struct32k at S size, 87.1M bytes, WikiText-103, briefing

Times are US Central (CT), Fri Nov 27, 2026, 11:04 to 11:11 AM. WikiText-103 test BPB_corpus, lower is better. Numbers come from the per-seed trainer records.

## Observe

| seed | test BPB |
|---|---:|
| 13 | 1.424167 |
| 21 | 1.418576 |
| 34 | 1.421989 |
| mean ± sd | 1.421577 ± 0.002818 |

## Orient

A three-seed probe of the struct32k tokenizer (V 32,768) at S size (6 layers, d 384, 6 heads, ff 1,024; 23,204,736 parameters, 10,621,824 non-embedding), flat head, structure off, full residual table, eager mode, at 87,087,822 train bytes. **No comparison run was made.** Do not read this number against g1d, g1i, or any other row: they differ in data budget, residual setting, or compile mode.

## Decide / Act

Recorded as a number only. No pass/fail is claimed. Gates touched: none.

## Caveats

n = 3, one size, one budget, one pass of the scheduler. The plan is marked completed with three seeds and is not a finished protocol.
