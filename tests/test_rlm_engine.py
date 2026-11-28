"""Unit, concurrency, sandbox-security, and replay tests for ``hadukenn_core.rlm_engine`` (Feature 16)."""

from __future__ import annotations

import asyncio
import time

import pytest

from hadukenn_core.rlm_engine import (
    RLM_PROMPT_PROFILES,
    RLMContextREPL,
    RLMEngine,
    RLMOrchestrator,
    RLMPromptProfile,
    RLMReplEnvironment,
    RLMRunner,
    SandboxedPythonREPL,
    SandboxedRLMRepl,
    get_rlm_prompt_profile,
    oolong_pairs_stitch,
    rlm_subgraph_stitch,
    run_oolong_pairs_stitching,
    stitch_oolong_pairs,
)
from sheaf_verify.hashing import SheafError
from sheaf_verify.rlm_guards import RLMBudgetPolicy, RLMBudgetState


def test_public_aliases_and_prompt_profiles():
    assert RLMContextREPL is SandboxedPythonREPL
    assert RLMReplEnvironment is SandboxedPythonREPL
    assert SandboxedRLMRepl is SandboxedPythonREPL
    assert RLMEngine is RLMOrchestrator
    assert RLMRunner is RLMOrchestrator
    assert run_oolong_pairs_stitching is oolong_pairs_stitch
    assert rlm_subgraph_stitch is oolong_pairs_stitch
    assert stitch_oolong_pairs is oolong_pairs_stitch

    seen_prompts: set[str] = set()
    profiles: dict[str, RLMPromptProfile] = {}
    for fam in ("gpt5", "gemini", "qwen", "litert"):
        prof = get_rlm_prompt_profile(fam)
        assert prof is RLM_PROMPT_PROFILES[fam]
        assert isinstance(prof.system_prompt, str) and prof.system_prompt.strip()
        assert prof.prompt == prof.system_prompt
        assert prof["system_prompt"] == prof.system_prompt
        assert prof.get("prompt") == prof.system_prompt
        assert "prompt" in prof and "target_batch_chars" in prof
        assert "context" in prof.system_prompt.lower()
        assert prof.target_batch_chars > 0
        seen_prompts.add(prof.system_prompt)
        profiles[fam] = prof

    assert len(seen_prompts) == 4
    assert 150_000 <= profiles["qwen"].target_batch_chars <= 250_000
    assert profiles["gpt5"].target_batch_chars >= 300_000
    assert profiles["litert"].target_batch_chars < profiles["gpt5"].target_batch_chars

    assert get_rlm_prompt_profile("  Qwen3 ").family == "qwen"
    with pytest.raises(SheafError):
        get_rlm_prompt_profile("unknown_model_family")
    with pytest.raises(SheafError):
        get_rlm_prompt_profile(123)  # type: ignore[arg-type]


def test_sandboxed_repl_context_metadata_root_prompt_and_safe_stdlib():
    docs = ["alpha " * 300, "beta " * 150, "gamma " * 50]
    repl = SandboxedPythonREPL(docs, max_stdout_chars=1024)

    assert repl.variables is repl.namespace
    assert repl.env is repl.namespace
    assert repl.namespace["context"] == docs
    assert repl.namespace["context_lengths"] == [len(d) for d in docs]
    assert repl.namespace["context_total_length"] == sum(len(d) for d in docs)
    assert repl.namespace["context_type"] == "list"
    assert "__builtins__" not in repl.namespace

    root_prompt = repl.build_root_prompt("Count alpha occurrences", model_family="qwen")
    assert "alpha alpha alpha" not in root_prompt
    assert str(repl.namespace["context_total_length"]) in root_prompt
    assert "Count alpha occurrences" in root_prompt

    res = repl.execute(
        "import re, json, math, collections\n"
        "def count_alpha(items):\n"
        "    return [len(re.findall(r'alpha', d)) for d in items]\n"
        "counts = count_alpha(context)\n"
        "summary = collections.Counter({'alpha': sum(counts)})\n"
        "print(json.dumps({'counts': counts, 'sqrt': round(math.sqrt(sum(counts)), 2), 'total': summary['alpha']}))\n"
    )
    assert res["ok"] is True
    assert res["truncated"] is False
    assert '"total": 300' in res["stdout"]
    assert "__builtins__" not in repl.namespace
    assert "counts" in res["new_vars"]


@pytest.mark.parametrize(
    "unsafe_code",
    [
        "import os\nos.system('echo x')",
        "import subprocess",
        "from os import path",
        "from . import something",
        "from math import *",
        "open('/etc/passwd', 'r')",
        "__import__('os')",
        "eval('1 + 1')",
        "exec('x = 1')",
        "compile('1', '<x>', 'eval')",
        "getattr((), '__class__')",
        "setattr({}, 'x', 1)",
        "delattr({}, 'x')",
        "hasattr((), '__class__')",
        "vars()",
        "dir()",
        "globals()",
        "locals()",
        "().__class__.__base__.__subclasses__()",
        "(lambda: 1).__code__",
        "(x for x in range(2)).gi_frame",
        '"{0.__class__}".format(())',
        'tmpl = "{0._hidden}"; tmpl.format(())',
        "__builtins__",
    ],
)
def test_sandboxed_repl_blocks_escapes_and_preserves_repl_step_budget(unsafe_code: str):
    policy = RLMBudgetPolicy(max_repl_steps=3, min_batch_chars=10)
    state = RLMBudgetState()
    repl = SandboxedPythonREPL("ctx", budget_policy=policy, budget_state=state)

    with pytest.raises((PermissionError, ImportError, AttributeError, NameError, SheafError)):
        repl.execute(unsafe_code)
    assert state.repl_steps == 0

    for _ in range(3):
        assert repl.execute("x = 1")["ok"] is True
    with pytest.raises(SheafError, match="max_repl_steps"):
        repl.execute("x = 2")


def test_sandboxed_repl_bounded_stdout_truncation():
    repl = SandboxedPythonREPL({"k1": "abc", "k2": "defghi"}, max_stdout_chars=256)
    assert repl.namespace["context_lengths"] == [3, 6]
    assert repl.namespace["context_total_length"] == 9

    res = repl.execute("print('A' * 50000)")
    assert res["truncated"] is True
    assert len(res["stdout"]) <= 256
    assert "trunc" in res["stdout"].lower()
    assert res["total_stdout_chars"] >= 50000


def test_llm_query_sync_async_and_budget_enforcement():
    calls: list[str] = []

    def sync_backend(prompt: str) -> str:
        calls.append(prompt)
        return f"SYNC:{len(prompt)}"

    policy = RLMBudgetPolicy(
        max_depth=1,
        max_subcalls=2,
        max_total_tokens=12_000,
        min_batch_chars=100,
        thinking_reserve=256,
    )
    state = RLMBudgetState()
    repl = SandboxedPythonREPL("ctx", budget_policy=policy, budget_state=state, llm_backend=sync_backend)

    # Micro-batch rejected before invoking backend
    with pytest.raises(SheafError, match="min_batch_chars"):
        repl.llm_query("short", depth=1)
    # Depth exceeded rejected before invoking backend
    with pytest.raises(SheafError, match="max_depth"):
        repl.llm_query("x" * 150, depth=2)
    # Token cap rejected before invoking backend
    with pytest.raises(SheafError, match="max_total_tokens"):
        repl.llm_query("x" * 150, depth=1, tokens=20_000)
    assert len(calls) == 0

    assert repl.llm_query("a" * 120, depth=1) == "SYNC:120"
    assert repl.llm_query("b" * 140, depth=1) == "SYNC:140"
    with pytest.raises(SheafError, match="max_subcalls"):
        repl.llm_query("c" * 120, depth=1)
    assert len(calls) == 2

    # Async backend invoked synchronously outside AND inside a running event loop
    async def async_backend(prompt: str) -> str:
        await asyncio.sleep(0.01)
        return f"ASYNC:{prompt[:8]}"

    repl_async = SandboxedPythonREPL(
        "ctx",
        budget_policy=RLMBudgetPolicy(min_batch_chars=10),
        budget_state=RLMBudgetState(),
        llm_backend=async_backend,
    )
    assert repl_async.llm_query("hello world 123", depth=1) == "ASYNC:hello wo"

    async def call_from_running_loop() -> str:
        return repl_async.llm_query("inside loop 456", depth=1)

    assert asyncio.run(call_from_running_loop()) == "ASYNC:inside l"


def test_llm_query_batch_semaphore_concurrency_and_atomic_rollback():
    active = 0
    peak = 0
    invocations: list[str] = []

    async def mock_async_backend(prompt: str) -> str:
        nonlocal active, peak
        invocations.append(prompt)
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.025)
        active -= 1
        return f"OUT:{prompt[:10]}"

    policy = RLMBudgetPolicy(
        max_depth=1,
        max_subcalls=6,
        max_total_tokens=100_000,
        min_batch_chars=100,
    )
    state = RLMBudgetState()
    repl = SandboxedPythonREPL("ctx", budget_policy=policy, budget_state=state, llm_backend=mock_async_backend)

    # Second prompt violates min_batch_chars -> atomic rollback, 0 invocations
    bad_batch = ["valid_" + "x" * 120, "tiny"]
    with pytest.raises(SheafError, match="min_batch_chars"):
        asyncio.run(repl.llm_query_batch(bad_batch, max_concurrency=4, depth=1))
    assert state.subcalls == 0
    assert state.total_tokens == 0
    assert invocations == []

    # Valid batch runs concurrently under semaphore(4) and preserves order
    prompts = [f"item_{i}_" + "z" * 120 for i in range(5)]
    results = asyncio.run(repl.llm_query_batch(prompts, max_concurrency=4, depth=1))
    assert results == [f"OUT:{p[:10]}" for p in prompts]
    assert peak >= 2
    assert peak <= 4
    assert state.subcalls == 5

    # Sync backend also executes concurrently in llm_query_batch
    sync_active = 0
    sync_peak = 0

    def mock_sync_backend(prompt: str) -> str:
        nonlocal sync_active, sync_peak
        sync_active += 1
        sync_peak = max(sync_peak, sync_active)
        time.sleep(0.02)
        sync_active -= 1
        return f"SYNC:{prompt[:6]}"

    repl_sync = SandboxedPythonREPL("ctx", llm_backend=mock_sync_backend)
    sync_out = asyncio.run(repl_sync.llm_query_batch(["p1_xxx", "p2_yyy", "p3_zzz"], max_concurrency=3))
    assert sync_out == ["SYNC:p1_xxx", "SYNC:p2_yyy", "SYNC:p3_zzz"]
    assert sync_peak >= 2


def test_orchestrator_final_var_fenced_ignore_plan_reject_and_thinking_reserve():
    policy = RLMBudgetPolicy(max_depth=1, max_subcalls=8, max_total_tokens=50_000, min_batch_chars=10, thinking_reserve=1024)
    orch = RLMOrchestrator(budget_policy=policy, max_turns=4, model_family="gemini")

    # 1. Valid multi-turn REPL + FINAL_VAR
    turns_ok = [
        {"text": "```repl\nresult_buf = ['a', 'b']\nprint(len(result_buf))\n```", "max_output_tokens": 4096, "thinking_budget": 2048},
        {"text": "FINAL_VAR(result_buf)", "max_output_tokens": 4096, "thinking_budget": 512},
    ]
    idx = 0

    def model_ok(_msgs):
        nonlocal idx
        item = turns_ok[idx]
        idx += 1
        return item

    res_ok = orch.run("extract pairs", ["a", "b"], model_ok)
    assert res_ok["ok"] is True and res_ok["passed"] is True
    assert res_ok["answer"] == ["a", "b"]

    # 2. Thinking reserve violation raises SheafError
    with pytest.raises(SheafError, match="thinking_reserve"):
        orch.run(
            "fail reserve",
            "ctx",
            lambda _m: {"text": "FINAL(ok)", "max_output_tokens": 4096, "thinking_budget": 3500},
        )

    # 3. Fenced FINAL(fake) ignored and scratch variable NOT leaked on max_turns exit
    fenced_orch = RLMOrchestrator(max_turns=2)
    fenced_turns = ["```repl\nfake = 'bad'\nprint('FINAL(fake)')\n```", "Still working..."]
    f_idx = 0

    def model_fenced(_msgs):
        nonlocal f_idx
        out = fenced_turns[f_idx]
        f_idx += 1
        return out

    res_fenced = fenced_orch.run("find", "ctx", model_fenced)
    assert res_fenced["ok"] is False and res_fenced["passed"] is False
    assert res_fenced["answer"] is None

    # 4. Undefined/empty FINAL_VAR and plan text in FINAL(...) rejected
    for bad_seq in (
        ["```repl\nx = 'nonempty'\n```", "FINAL_VAR(missing_var)"],
        ["```repl\nempty_s = '   '\n```", "FINAL_VAR(empty_s)"],
        ["```repl\nempty_l = []\n```", "FINAL_VAR(empty_l)"],
        ["FINAL(I will now inspect the context first.)", "FINAL(Step 1: check context)"],
    ):
        pos = 0

        def model_bad(_msgs, _seq=bad_seq):
            nonlocal pos
            out = _seq[min(pos, len(_seq) - 1)]
            pos += 1
            return out

        res_bad = RLMOrchestrator(max_turns=2).run("q", "ctx", model_bad)
        assert res_bad["ok"] is False and res_bad["passed"] is False and res_bad["answer"] is None


def test_orchestrator_anti_repetition_recovers_best_verified_value():
    turns = [
        "```repl\nbest_pairs = [('n1', 'n2')]\nprint(best_pairs)\n```",
        "Checking best_pairs again.",
        "Checking best_pairs again.",
        "Checking best_pairs again.",
        "FINAL(wrong fallback free text)",
    ]
    idx = 0

    def model_fn(_msgs):
        nonlocal idx
        out = turns[min(idx, len(turns) - 1)]
        idx += 1
        return out

    orch = RLMOrchestrator(max_turns=6, max_consec_no_new_var=3)
    res = orch.run("pairs", "n1 n2", model_fn)
    assert res["ok"] is True and res["passed"] is True
    assert res["answer"] == [("n1", "n2")]
    assert res["termination"] == "anti_repetition"
    assert idx == 4


@pytest.mark.parametrize("n_items", [0, 1, 2, 11, 15, 19, 21])
def test_oolong_pairs_stitch_arbitrary_n_and_kernel_replay(n_items: int):
    items = [f"item_{i}:tag_{i % 3}" for i in range(n_items)]
    res = oolong_pairs_stitch(
        items,
        n_chunks=4,
        pair_predicate=lambda a, b: a.split(":")[-1] == b.split(":")[-1],
    )
    assert res["n_items"] == n_items
    assert res["covered_indices"] == list(range(n_items))
    assert len(set(res["covered_indices"])) == n_items
    assert res["total_pairs_evaluated"] == n_items * (n_items - 1) // 2
    expected_matches = [
        (i, j)
        for i in range(n_items)
        for j in range(i + 1, n_items)
        if items[i].split(":")[-1] == items[j].split(":")[-1]
    ]
    assert sorted(res["matched_pairs"]) == expected_matches


def test_oolong_pairs_stitch_exec_kernel_trace_and_replay():
    pytest.importorskip("numpy")
    from hadukenn_core.exec_bridge import IncrementalHypergraphKernel

    kern = IncrementalHypergraphKernel()
    items = [f"doc_{i}_group_{i % 2}" for i in range(11)]
    res = oolong_pairs_stitch(items, n_chunks=4, exec_kernel=kern)
    assert res["ok"] is True
    assert res["covered_indices"] == list(range(11))
    assert res["total_pairs_evaluated"] == 55
    trace = kern.export_trace()
    assert len(trace) >= 1
    replayed = IncrementalHypergraphKernel.replay(trace, expected_head=kern.head)
    assert replayed.state_digest() == kern.state_digest()
