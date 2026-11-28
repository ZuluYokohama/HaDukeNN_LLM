"""Pure-stdlib unit tests for ``sheaf_verify.rlm_guards`` (Feature 12)."""

from __future__ import annotations

import math

import pytest

from sheaf_verify.hashing import SheafError
from sheaf_verify.rlm_guards import (
    AntiRepetitionTracker,
    RLMBudgetPolicy,
    RLMBudgetState,
    SiipVerifyGate,
    parse_final_directive,
)


def test_siip_verify_gate_damping_and_dual_mode():
    gate = SiipVerifyGate()
    assert gate.mode == "formal"

    # Zero-norm guard
    assert gate.damp_direction([1.0, 0.0], [0.0, 0.0]) == [1.0, 0.0]
    assert gate.hesitation_damping_events == 0

    # Strict boundary: cos == -0.25 is NOT damped
    p_exact = [-0.25, math.sqrt(1.0 - 0.25**2)]
    out_exact = gate.damp_direction(p_exact, [1.0, 0.0])
    assert out_exact[0] == pytest.approx(-0.25)
    assert gate.hesitation_damping_events == 0

    # cos == -0.30 < -0.25 triggers 0.5x damping
    p_below = [-0.30, math.sqrt(1.0 - 0.30**2)]
    out_below = gate.damp_direction(p_below, [1.0, 0.0])
    assert out_below[0] == pytest.approx(-0.15)
    assert gate.hesitation_damping_events == 1

    # Extreme finite vector scaling in damp_direction (avoid overflow ValueError/NaN and underflow)
    extreme_gate = SiipVerifyGate()
    assert extreme_gate.damp_direction([1e200, 1e200], [-1e200, 1e200]) == [1e200, 1e200]
    assert extreme_gate.hesitation_damping_events == 0
    assert extreme_gate.damp_direction([1e200], [-1e200]) == [5e199]
    assert extreme_gate.hesitation_damping_events == 1
    assert extreme_gate.damp_direction([1e-200], [-1e-200]) == [5e-201]
    assert extreme_gate.hesitation_damping_events == 2

    # Formal mode requires strict f_cand < f_k
    assert gate.verify_commit(1.0, 0.75, [0.5, 0.5])["accepted"] is True
    assert gate.verify_commit(1.0, 1.0, [0.5, 0.5])["accepted"] is False
    assert gate.verify_commit(1.0, 1.2, [0.5, 0.5])["accepted"] is False

    # Speculative/YOLO/sandbox mode allows finite increases
    assert gate.set_mode("yolo") == "speculative"
    assert gate.verify_commit(1.0, 1.2, [0.5, 0.5])["accepted"] is True
    assert gate.set_mode("verify") == "formal"

    for bad_fn in (
        lambda: SiipVerifyGate(mode=123),  # type: ignore[arg-type]
        lambda: SiipVerifyGate(mode="unknown"),
        lambda: SiipVerifyGate(hesitation_cos_threshold=2.0),
        lambda: SiipVerifyGate(damping_factor=1.0),
        lambda: gate.damp_direction("ab", [1.0, 0.0]),  # type: ignore[arg-type]
        lambda: gate.damp_direction([1.0], [1.0, 0.0]),
        lambda: gate.verify_commit(1.0, float("nan"), [0.5]),
        lambda: gate.verify_commit(1.0, 0.5, "x"),  # type: ignore[arg-type]
        lambda: gate.verify_commit(1.0, 0.5, []),
        lambda: gate.verify_commit(1.0, 0.5, [float("inf")]),
    ):
        with pytest.raises(SheafError):
            bad_fn()


def test_rlm_budget_policy_and_state():
    policy = RLMBudgetPolicy(
        max_depth=1,
        max_subcalls=2,
        max_total_tokens=1_000,
        min_batch_chars=100,
        thinking_reserve=256,
    )
    state = RLMBudgetState()
    assert state.subcall_count == 0

    res = state.record_subcall(policy, depth=1, chars=500, tokens=400, max_output_tokens=2048, thinking_budget=512)
    assert res["ok"] is True
    assert state.subcalls == 1 and state.subcall_count == 1
    assert state.total_tokens == 400 and state.max_depth_seen == 1

    # Depth exceeded (state stays unchanged)
    with pytest.raises(SheafError):
        policy.check_subcall(state, depth=2, chars=500, tokens=100)
    # Micro-batch below floor
    with pytest.raises(SheafError):
        policy.check_subcall(state, depth=1, chars=50, tokens=100)
    # Token budget exceeded
    with pytest.raises(SheafError):
        policy.check_subcall(state, depth=1, chars=500, tokens=700)
    # Thinking reserve exhausted
    with pytest.raises(SheafError):
        policy.check_subcall(state, depth=1, chars=500, tokens=100, max_output_tokens=1024, thinking_budget=900)
    # Negative thinking/output tokens
    with pytest.raises(SheafError):
        policy.check_subcall(state, depth=1, chars=500, tokens=100, max_output_tokens=1024, thinking_budget=-1)
    assert state.subcalls == 1

    # Second valid call succeeds; third hits max_subcalls
    assert policy.check_subcall(state, depth=1, chars=500, tokens=100)["ok"] is True
    with pytest.raises(SheafError):
        policy.check_subcall(state, depth=1, chars=500, tokens=100)

    for bad_kw in (
        {"max_depth": -1},
        {"max_subcalls": 0},
        {"max_total_tokens": 0},
        {"min_batch_chars": -1},
        {"thinking_reserve": -1},
        {"max_repl_steps": 0},
    ):
        with pytest.raises(SheafError):
            RLMBudgetPolicy(**bad_kw)  # type: ignore[arg-type]

    with pytest.raises(SheafError):
        policy.check_subcall("bad", 1, 500, 100)  # type: ignore[arg-type]
    with pytest.raises(SheafError):
        policy.check_subcall(state, -1, 500, 100)

    step_policy = RLMBudgetPolicy(max_repl_steps=2)
    step_state = RLMBudgetState()
    assert step_policy.check_repl_step(step_state)["repl_steps"] == 1
    assert step_policy.check_repl_step(step_state)["repl_steps"] == 2
    with pytest.raises(SheafError, match="max_repl_steps"):
        step_policy.check_repl_step(step_state)
    assert step_state.repl_steps == 2
    with pytest.raises(SheafError):
        step_policy.check_repl_step("bad")  # type: ignore[arg-type]


def test_parse_final_directive_and_pitfall_guards():
    ok_direct = parse_final_directive("Done.\nFINAL(42 verified (nested) pairs)", {})
    assert ok_direct["found"] is True and ok_direct["value"] == "42 verified (nested) pairs"

    ok_var = parse_final_directive("Done.\nFINAL_VAR(ans)", {"ans": [("a", "b")]})
    assert ok_var["found"] is True and ok_var["value"] == [("a", "b")]

    ok_zero = parse_final_directive("FINAL_VAR(count)", {"count": 0})
    assert ok_zero["found"] is True and ok_zero["value"] == 0

    # Fenced code blocks (including tilde and unclosed trailing fence) are ignored
    fenced = "```repl\nFINAL(fake)\n```\n~~~\nFINAL_VAR(ans)\n~~~\nStill running\n```python\nFINAL(unclosed)"
    res_fenced = parse_final_directive(fenced, {"ans": "real"})
    assert res_fenced["found"] is False and res_fenced["value"] is None

    # CommonMark nested fences, info-string non-closers, 4-space non-closers, and inline code spans
    nested_bt = "````markdown\n```repl\nFINAL(inside_bt)\n```\n```not_closing\n    ````\n   ````\nFINAL(outside_bt)"
    assert parse_final_directive(nested_bt, {})["value"] == "outside_bt"

    nested_tl = "~~~~markdown\n~~~python\nFINAL(inside_tl)\n~~~\n~~~~\nFINAL(outside_tl)"
    assert parse_final_directive(nested_tl, {})["value"] == "outside_tl"

    for span_text, expected in (
        ("Do not emit ```FINAL(fake)``` here. FINAL(real)", "real"),
        ("```repl FINAL(fake) ```\nFINAL(real)", "real"),
        ("~~~repl FINAL(fake) ~~~\nFINAL(real)", "real"),
        ("~~~a~~~b\nFINAL(inside)\n~~~\nFINAL(real)", "real"),
        ("~~~py ~~~ FINAL(fake)\n~~~\nFINAL(real)", "real"),
        ("Use `FINAL(fake)` or `` `FINAL(fake2)` `` then FINAL(real)", "real"),
        ("Here is `code\nFINAL(fake)`\nFINAL(real)", "real"),
        ("Here is ``code\n`inner`\nFINAL(fake)``\nFINAL(real)", "real"),
        ("Stray ` backtick and ~~ tilde and ~~~unclosed\nFINAL(real)", "real"),
        ("Inline ~~~tilde\nFINAL(real)", "real"),
        ("Unclosed ` then FINAL(real)", "real"),
        ("Unclosed `\n\nFINAL(real)", "real"),
        ("Unclosed `\n   \nFINAL(real)", "real"),
        ('Unclosed `\n\nFINAL("a `b` c")', '"a `b` c"'),
        ("Here is `unclosed\n```python\nx = 1\n```\nFINAL(`42`)", "`42`"),
        ("Here is `unclosed\n~~~python\nx = 1\n~~~\nFINAL(`42`)", "`42`"),
        ('FINAL("a `b` c")', '"a `b` c"'),
        ("FINAL(`42`)", "`42`"),
        ("FINAL(unclosed\n`FINAL(fake)`\nFINAL(real)", "real"),
        ("FINAL(Step 1.5)", "Step 1.5"),
    ):
        parsed = parse_final_directive(span_text, {})
        assert parsed["found"] is True and parsed["value"] == expected

    # Unclosed tilde fence with tilde in info string hides everything after it
    assert parse_final_directive("~~~py ~~~ FINAL(fake)\nFINAL(real)", {})["found"] is False
    # CommonMark §6.1 paragraph-level backtick pairing: unclosed backtick pairs across lines until blank line
    assert parse_final_directive('Unclosed ` then FINAL("a `b` c")', {})["found"] is False
    assert parse_final_directive('Unclosed `\nFINAL("a `b` c")', {})["found"] is False

    # Unbalanced FINAL(
    assert parse_final_directive("FINAL(unclosed", {})["found"] is False

    # Missing, empty, and plan-like FINAL outputs rejected
    for text, ns in (
        ("FINAL_VAR(missing)", {}),
        ("FINAL_VAR(empty_str)", {"empty_str": "   "}),
        ("FINAL_VAR(empty_list)", {"empty_list": []}),
        ("FINAL_VAR(none_val)", {"none_val": None}),
        ("FINAL()", {}),
        ("FINAL(I will now write a Python script to inspect the context)", {}),
        ("FINAL(I'll now inspect the context)", {}),
        ("FINAL(Let me check the files first)", {}),
        ("FINAL(Step 1. Read the context)", {}),
        ("FINAL(My plan is to scan chunks)", {}),
    ):
        rejected = parse_final_directive(text, ns)
        assert rejected["found"] is False and rejected["accepted"] is False and rejected["value"] is None

    with pytest.raises(SheafError):
        parse_final_directive(123, {})  # type: ignore[arg-type]
    with pytest.raises(SheafError):
        parse_final_directive("FINAL(ok)", "not-a-mapping")  # type: ignore[arg-type]


def test_anti_repetition_tracker():
    tracker = AntiRepetitionTracker(max_consec_no_new_var=3)
    r1 = tracker.record_turn("step 1", {"buf"}, verified_value="v1")
    assert r1.should_stop is False and r1["should_stop"] is False
    assert r1.get("consec_no_new_var") == 0
    assert r1.get("missing", "fallback") == "fallback"
    assert "best_verified_value" in r1 and "missing" not in r1
    with pytest.raises(KeyError):
        _ = r1["missing"]

    assert tracker.record_turn("step 2", {"buf", "buf2"}).should_stop is False
    assert tracker.record_turn("repeat 1", {"buf2"}).should_stop is False
    assert tracker.record_turn("repeat 2", {"buf2"}).should_stop is False
    r_stop = tracker.record_turn("repeat 3", {"buf2"})
    assert r_stop.should_stop is True
    assert r_stop["best_verified_value"] == "v1"

    with pytest.raises(SheafError):
        AntiRepetitionTracker(max_consec_no_new_var=0)
    with pytest.raises(SheafError):
        tracker.record_turn(123, {"a"})  # type: ignore[arg-type]
    with pytest.raises(SheafError):
        tracker.record_turn("ok", "abc")
