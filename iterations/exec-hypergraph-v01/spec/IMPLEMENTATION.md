# Executable-state prototype v0.1

## Status and purpose

This iteration builds the reference semantics and measurement harness needed before a learned graph-native model can be evaluated. It includes no model weights, training loop, language parser, spectral embedding, sheaf diffusion, or production inference server. Source text is preserved exactly but not interpreted automatically.

The candidate architecture keeps exact observations, overlapping byte spans, typed relations, operation execution, memory revision, and output evidence identifiable. The prototype implements the first executable interfaces. The byte-span graph and relational kernel are separate modules: the learned bridge from spans to typed actions is deliberately still absent and must be evaluated, not silently supplied in a capability claim.

## Command language

Every command is a JSON object with `action`. Scope defaults to `main`.

| Action | Required fields | Meaning |
| --- | --- | --- |
| `source` | `id`, `text` | Create an immutable UTF-8 source record |
| `set` | `name`, `value`; optional `unit`, `source` | Explicitly revise a formal binding |
| `derive` | `name`, `op`, `args`; optional `source` | Define a relation with named input ports |
| `retract` | `name` | Remove the current binding and invalidate dependent cached values |
| `fork` | `new_scope` | Snapshot the selected scope into an independent branch |
| `query` | `name` | Evaluate the requested binding and record the actual dependency computation |
| `assert` | `name`, `value`; optional `unit` | Compare an independently supplied expected quantity with evaluation |

`add`, `sub`, `mul`, and `div` use `left` and `right`. `neg` and `copy` use `value`. References resolve inside the selected scope. Forward references may be declared; evaluation refuses missing references and cycles. Replacing a definition changes its revision, rewires its dependencies, and invalidates affected cached conclusions. Forks snapshot definitions and rebind references to the new scope, so subsequent parent changes do not leak across branches.

Values are integers or signed rational strings and are computed with exact fractions. Floats and booleans are refused. Units are literal symbols. Addition/subtraction requires equal symbols; multiplication permits a dimensionless operand; division permits a dimensionless denominator or equal units, producing a dimensionless ratio. This is a small declared unit algebra, not a general units engine.

`set` is an authoritative revision in this formal input contract. It does not verify whether a natural-language source supports the value. `assert` can expose a mismatch but cannot establish source truth. A future learned proposer must label interpretations and supply evidence rather than inheriting this authoritative-input assumption.

## Provenance and replay

Receipts record the command, result, actual evaluation evidence, semantic-state commitments, configuration, sequence, and previous receipt hash. Cache reuse points to the computation that originally produced the cached value. Replay verifies the chain and re-executes commands, comparing their actual outputs and evidence. A rehashed fabricated answer therefore still fails semantic replay.

An independently retained final head can detect truncation. Without that head, a valid prefix remains a valid prefix. Hashes do not authenticate the author, establish an independent timestamp, or prove that the external world matched the formal input. Replay uses the same kernel implementation; the independent reference evaluator and literal expected answers provide a separate semantic check.

## Resource bounds

Finite limits apply to active nodes, uncached evaluation steps, scopes, revisions, source bytes, command size, and retained receipts. Their exact configured values are included in receipts. These are limits on specific data structures; they are not measured process-RAM or worst-case wall-time guarantees. Receipt exhaustion refuses the next operation before execution and cannot append another receipt. The caller must retain that terminal refusal.

The current implementation recomputes complete semantic-state hashes per command. This makes audit cost visible and is expected to dominate some small workloads. No efficiency superiority claim is licensed by the first timing table. A later matched-audit comparison should compare incremental state commitments with equally capable conventional stores.

## Byte-span boundary

The input module preserves exact bytes and constructs bounded overlapping span candidates. A span is connected to its ordered constituent bytes through typed incidence ports. The observed prefix is selected before computing source identity or candidates, preventing unseen suffix bytes from affecting prefix representations. Byte reconstruction is exact even for invalid UTF-8 and incomplete code points.

These candidates are deterministic lexical proposals. They do not establish semantic units, learned tokenization, entropy patching, a language-model loss improvement, or a graph-to-action interpretation capability.

## Planned learned-interface experiment

After the reference semantics are stable, the next experiment should compare a graph-conditioned action policy with a conventional controller using the same primitives, evidence, storage, verifier, examples, and search budget. First test novel compositions of known operations; then evaluate extraction from text separately. Freeze world/template families before tuning. Count parsing, graph construction, state updates, search, checking, and fallback. Ordinary program search and a competent tool-using model remain controls.

The ultimate operating-range target is tenfold greater necessary dependency depth at a declared verified-success threshold and fixed resource budget. This pilot does not measure that target: operation definitions and inputs are supplied, and every exact baseline has access to the same arithmetic.

## Existing program boundary

This iteration changes no G0-G6, H0-H7, or P0-P6 research verdict. It does not alter the frozen T-38 design, attach a new result to T-38, or reopen the killed packing experiment. The original synthetic verification fixture and ledger remain unchanged.
