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

G1c, g1d, and g1g do not have `iter/` branches here. The same numbers are in `docs/index.html`, `docs/briefing.html`, and the README. WikiText-103 test bits per byte. The measured rows are n=3.

| id | Status |
| --- | --- |
| G1c | G1c pass, compute-matched. hnt40k-struct-v2 1.269376 ± 0.004417 at 1.2061e7 train FLOPs per byte and 4.2006e15 train FLOPs. sw40k-struct-v2 control 1.322698 ± 0.006710. struct32k 1.322354. GPT-2 tokenizer arm 1.328219. hnt versus that control is -0.0533 (p 7.1e-4). Compute-efficiency win, not a per-byte modeling win. |
| g1d | Data-matched struct32k at 348.3M bytes, the same train bytes hnt saw: 1.194155 ± 0.000824 at 1.0756e16 train FLOPs. Beats hnt by 0.075 at equal data. |
| g1g | Pending. FLOPs-matched hnt at 1.0756e16 train FLOPs. Still running. No result is recorded. |

Caveats: n=3, one model size, WikiText-103 only, 39.9M versus 23.2M total parameters, same 10.62M non-embedding. G1 and G1b stay fails.
