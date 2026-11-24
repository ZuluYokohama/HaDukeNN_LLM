# Iterations

Index for HaDukeN //\\\\// HGNLLM Experiments. The package name stays `sheaf-verify`.

`main` holds the harness and this index. Each experiment lives on its own branch `iter/<id>`, cut from `main`. The branch adds:

| Path | Contents |
| --- | --- |
| `iterations/<id>/BRIEFING.md` | OODA (Observe, Orient, Decide, Act), then op-order sections Situation, Mission, Execution, Sustainment, Command & Signal |
| `iterations/<id>/MODE_METHOD.md` | Mode, method, controls, and the gates that iteration may speak to |
| `iterations/<id>/results/` | Records from that iteration |

`iterations/TEMPLATE/` is the blank form. It is not an iteration and it has no branch.

Do not put weights, a corpus, a tokenizer merge table, a credential, or a machine path on an iteration branch.

Keep this table and the iteration list in `docs/index.html` in step. The link for a row is `https://github.com/ZuluYokohama/HaDukeNN_LLM/tree/iter/<id>`.

| id | branch | Status on that branch |
| --- | --- | --- |
| g0 | [`iter/g0`](https://github.com/ZuluYokohama/HaDukeNN_LLM/tree/iter/g0) | Feasibility unit counts. G0 partial: V=65,536 is over the written cap of 64,000. |
| g1-hnt64k | [`iter/g1-hnt64k`](https://github.com/ZuluYokohama/HaDukeNN_LLM/tree/iter/g1-hnt64k) | Units-only pilot. G1 fail: test BPB 1.452254 is above struct32k + 1 sd (1.329303), and train FLOPs/byte 3.5706e7 is above struct32k and sw64k. |
| g1b-structured | [`iter/g1b-structured`](https://github.com/ZuluYokohama/HaDukeNN_LLM/tree/iter/g1b-structured) | Structured hnt40k. G1b fail on BPB (1.477304 > 1.329303). Train FLOPs/byte 1.1929e7 is about 61% below struct32k (3.0882e7) and below the equal-size control. |

G1c does not have an `iter/` branch here. g1d, g1g, g1h, and g1i have folders under `iterations/` on `main`. The same numbers are in `docs/index.html`, `docs/briefing.html`, and the README. WikiText-103 test bits per byte. The measured rows are n=3.

| id | Status |
| --- | --- |
| G1c | G1c pass, compute-matched. hnt40k-struct-v2 1.269376 ± 0.004417 at 1.2061e7 train FLOPs per byte and 4.2006e15 train FLOPs. sw40k-struct-v2 control 1.322698 ± 0.006710. struct32k 1.322354. GPT-2 tokenizer arm 1.328219. hnt versus that control is -0.0533 (p 7.1e-4). Compute-efficiency win, not a per-byte modeling win. |
| g1d | Data-matched struct32k at 348.3M bytes, the same train bytes hnt saw: 1.194155 ± 0.000824 at 1.0756e16 train FLOPs. Beats hnt by 0.075 at equal data. |
| g1g | g1g, hnt40k-struct-v2 FLOPs-matched to g1d at 1.0756e16 train FLOPs, scored 1.174315 ± 0.000564 (seeds 1.174805, 1.173698, 1.174444). That is 0.019839 below g1d (Welch p 1.3e-5), That is a win at equal analytic train FLOPs only (the 6N + attention + head count). The structured model swaps the flat 40k output softmax for a factored cluster-then-unit head (200 clusters × 200 units). At d=384 the flat head is about 56.6% of plain-model FLOPs per token, so struct-v2 counts about 2.22× fewer FLOPs per token. At equal FLOPs, g1g therefore saw 891.7M train bytes (about 1.65 epochs, against g1d's 0.64), and it took about 1,488 s per seed against about 501–505 s for g1d and g1h, so it is about 3× slower on wall clock. |
| g1h | g1h, plain hnt40k with the structured path off (struct none, mean compose, low-rank residual, no spectral init) at the same 1.0756e16 train FLOPs, scored 1.197637 ± 0.000517 (seeds 1.198146, 1.197111, 1.197653). That is 0.023321 above g1g (p 8.4e-7) and 0.003482 above g1d (p 0.006): at equal analytic FLOPs and the same tokenizer, the structured run scores lower, but it also saw about 2.2× the data. At equal data the order reverses. G1c, the same hnt40k-struct-v2 model at 348.3M train bytes, is 1.269376 ± 0.004417. That is 0.0579 worse than plain hnt40k at the same bytes (g1i, Welch p 8.8e-5) and 0.0752 worse than struct32k (g1d, p 8.2e-4). The mechanism resembles a class-factored softmax (Goodman 2001, arXiv:cs/0108006). This is not evidence that the structure layer models text better. Wall-clock-matched and 2× data-ladder controls are pending. |
| g1i | g1i, the same plain hnt40k at g1d's 348.35M train bytes (9.350e15 train FLOPs), scored 1.211442 ± 0.003395 (seeds 1.210431, 1.215228, 1.208666). That is 0.017287 above g1d (p 0.0094), so at equal bytes plain hnt40k loses to struct32k. g1i and g1d use different tokenizers, so this is a tokenizer contrast, not a structure ablation. |
| A3 | A3, the random-cluster control (hnt40k with random clusters, FLOPs-matched), is running. No result is recorded. |

Caveats: n=3, one model size, WikiText-103 only. Parameter totals are verified against the per-seed trainer records (the trainer itself is not in this repository): hnt40k-struct-v2 39,943,040, struct32k 23,204,736, and plain hnt40k 25,981,824, each with 10,621,824 non-embedding. Every run applied RoPE. The trainer's structured model path applies RoPE only and ignores `--pe heat`, so no heat positional encoding has been tested. G1 and G1b stay fails.
