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
