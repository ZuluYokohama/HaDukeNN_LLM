"""Tests for sheaf_verify.action_gate (exact units, witness consensus, six-verdict transfer gate)."""

from __future__ import annotations

from fractions import Fraction

import pytest

from sheaf_verify.action_gate import (
    FORMAL_VERDICTS,
    ActionRuleSpec,
    ExactUnitLattice,
    LocalActionOntology,
    witness_consensus,
)
from sheaf_verify.hashing import SheafError


def _ont(**rule_kw):
    rule = {"action_name": "wire", "dimension": "currency", "max_amount": Fraction(5000)}
    rule.update(rule_kw)
    return LocalActionOntology("test.v1", [ActionRuleSpec(**rule)])


def _run(ont=None, **kw):
    args = {
        "action_name": "wire",
        "actor_role": "operator",
        "src_entity": "A",
        "dst_entity": "B",
        "src_witnesses": {"w1": ("1000", "USD")},
        "dst_initial": ("200", "USD"),
        "transfer_amount": ("100", "USD"),
    }
    args.update(kw)
    return (ont or _ont()).evaluate_transfer_action(**args)


def test_unit_lattice_exact_and_refusals():
    assert ExactUnitLattice.add("750", "mL", "1.25", "L") == (Fraction(2), "L")
    assert ExactUnitLattice.sub("2.5", "kUSD", "50000", "cents") == (Fraction(2000), "USD")
    assert ExactUnitLattice.normalize("1", "kWh") == (Fraction(3_600_000), "J", "energy")
    assert ExactUnitLattice.normalize(Fraction(1, 3), " L ")[0] == Fraction(1, 3)
    assert ExactUnitLattice.normalize(7, "t")[0] == Fraction(7000)
    # 0.1 + 0.2 is exact here, unlike binary floats
    assert ExactUnitLattice.add("0.1", "USD", "0.2", "USD")[0] == Fraction(3, 10)
    with pytest.raises(SheafError, match="dimensional mismatch"):
        ExactUnitLattice.add("1", "USD", "1", "L")
    with pytest.raises(SheafError, match="dimensional mismatch"):
        ExactUnitLattice.sub("1", "USD", "1", "L")
    with pytest.raises(SheafError, match="unknown unit"):
        ExactUnitLattice.normalize("1", "parsecs")
    for bad in ("ten", "nan", "inf", "1/0", ""):
        with pytest.raises(SheafError, match="malformed quantity"):
            ExactUnitLattice.normalize(bad, "USD")
    for bad in (True, 1.5, None):
        with pytest.raises(SheafError):
            ExactUnitLattice.normalize(bad, "USD")
    with pytest.raises(SheafError, match="unit must be a string"):
        ExactUnitLattice.normalize("1", None)


@pytest.mark.parametrize(
    ("values", "regime", "resolved", "faults"),
    [
        ({"a": 5}, "CONSISTENT", 5, ()),
        ({"a": 5, "b": 5, "c": 5}, "CONSISTENT", 5, ()),
        ({"a": 5, "b": 9}, "ABSTAIN_TIE", None, ()),
        ({"a": 5, "b": 5, "c": 9}, "RECOVERABLE_FAULT", 5, ("c",)),
        ({"a": 5, "b": 5, "c": 5, "d": 9}, "RECOVERABLE_FAULT", 5, ("d",)),
        ({"a": 5, "b": 5, "c": 9, "d": 9}, "ABSTAIN_TIE", None, ()),
        ({"a": 1, "b": 2, "c": 3}, "ABSTAIN_TIE", None, ()),
        ({"a": 5, "b": 5, "c": 7, "d": 9}, "ABSTAIN_TIE", None, ()),
    ],
)
def test_witness_consensus_rule(values, regime, resolved, faults):
    got = witness_consensus({k: Fraction(v) for k, v in values.items()})
    assert got == (regime, None if resolved is None else Fraction(resolved), faults)


def test_witness_consensus_is_exact_not_float():
    big = Fraction(100_000_000)
    # A float gate with an absolute tolerance treats these as equal; the exact gate does not.
    assert witness_consensus({"a": big, "b": big + Fraction(1, 10**7)})[0] == "ABSTAIN_TIE"
    # These two collapse to the same binary float; the exact gate still sees a disagreement.
    huge = Fraction(10**17)
    assert float(huge) == float(huge + 1)
    assert witness_consensus({"a": huge, "b": huge + 1})[0] == "ABSTAIN_TIE"
    with pytest.raises(SheafError):
        witness_consensus({})


def test_ontology_construction_guards_and_policy_hash():
    with pytest.raises(SheafError, match="ontology_id"):
        LocalActionOntology("  ", [])
    with pytest.raises(SheafError, match="duplicate"):
        LocalActionOntology("x", [ActionRuleSpec("a", "currency"), ActionRuleSpec("a", "mass")])
    with pytest.raises(SheafError, match="unknown dimension"):
        LocalActionOntology("x", [ActionRuleSpec("a", "parsecs")])
    a = _ont()
    b = _ont()
    c = _ont(max_amount=Fraction(6000))
    assert a.policy_sha256 == b.policy_sha256 != c.policy_sha256
    assert len(a.policy_sha256) == 64


def test_all_six_verdicts_reachable():
    seen = {
        _run().verdict,
        _run(claimed_post_src=("800", "USD")).verdict,
        _run(action_name="nope").verdict,
        _run(dst_entity="?").verdict,
        _run(src_witnesses={"w1": ("1000", "USD"), "w2": ("900", "USD")}).verdict,
        _run(src_witnesses={"w1": ("1000", "USD"), "w2": ("1", "kUSD"), "w3": ("50", "USD")}).verdict,
    }
    assert seen == set(FORMAL_VERDICTS)


def test_valid_transfer_recomputes_balances_across_units():
    v = _run(src_witnesses={"w1": ("1", "kUSD")}, transfer_amount=("2500", "cents"))
    assert v.verdict == "VALID" and v.approved and not v.human_escalation_required
    assert (v.resolved_initial_src, v.post_src_balance, v.post_dst_balance) == ("1000", "975", "225")
    d = v.as_dict()
    assert d["verdict"] == "VALID" and d["isolated_fault_sources"] == [] and d["canonical_unit"] == "USD"


def test_recoverable_fault_sets_aside_single_outlier():
    v = _run(src_witnesses={"w1": ("1000", "USD"), "w2": ("100000", "cents"), "w3": ("7", "USD")})
    assert v.verdict == "RECOVERABLE_FAULT" and v.approved
    assert v.isolated_fault_sources == ("w3",) and v.resolved_initial_src == "1000"


def test_poisoned_majority_is_not_detected():
    # Documented limitation: two agreeing witnesses outvote one honest one.
    v = _run(src_witnesses={"honest": ("1000", "USD"), "p1": ("5000", "USD"), "p2": ("5000", "USD")})
    assert v.verdict == "RECOVERABLE_FAULT" and v.isolated_fault_sources == ("honest",)


@pytest.mark.parametrize(
    ("kw", "reason"),
    [
        ({"actor_role": "guest"}, "not permitted"),
        ({"src_witnesses": {"w1": ("1000", "L")}}, "dimension"),
        ({"dst_initial": ("1", "kg")}, "dimension"),
        ({"transfer_amount": ("1", "s")}, "dimension"),
        ({"transfer_amount": ("ten", "USD")}, "malformed"),
        ({"src_witnesses": {"w1": ("nan", "USD")}}, "malformed"),
        ({"dst_initial": ("1", "parsecs")}, "unknown unit"),
        ({"transfer_amount": "100 USD"}, "pair"),
        ({"claimed_post_src": ("900", "kg")}, "dimension"),
        ({"claimed_post_dst": ("bad", "USD")}, "malformed"),
    ],
)
def test_bad_input_is_impossible_and_escalated_not_an_exception(kw, reason):
    v = _run(**kw)
    assert v.verdict == "IMPOSSIBLE" and not v.approved and v.human_escalation_required
    assert reason in v.reason


def test_bounds_and_balance():
    assert _run(transfer_amount=("6000", "USD"), src_witnesses={"w1": ("9000", "USD")}).verdict == "IMPOSSIBLE"
    assert _run(transfer_amount=("-1", "USD")).verdict == "IMPOSSIBLE"
    over = _run(transfer_amount=("1500", "USD"))
    assert over.verdict == "IMPOSSIBLE" and "insufficient" in over.reason and not over.human_escalation_required
    ok = _run(_ont(allow_negative_post_balance=True), transfer_amount=("1500", "USD"))
    assert ok.verdict == "VALID" and ok.post_src_balance == "-500"


@pytest.mark.parametrize(
    "kw",
    [
        {"src_entity": None},
        {"dst_entity": ""},
        {"src_entity": "?"},
        {"src_entity": "B"},
        {"src_witnesses": {}},
    ],
)
def test_translation_ambiguous(kw):
    v = _run(**kw)
    assert v.verdict == "TRANSLATION_AMBIGUOUS" and v.human_escalation_required


def test_claimed_post_state_checks():
    assert _run(claimed_post_src=("900", "USD"), claimed_post_dst=("300", "USD")).verdict == "VALID"
    assert _run(claimed_post_src=("0.9", "kUSD")).verdict == "VALID"
    broken = _run(claimed_post_src=("900", "USD"), claimed_post_dst=("400", "USD"))
    assert broken.verdict == "INVALID" and "conservation" in broken.reason
    # Conserved sum but wrong split
    split = _run(claimed_post_src=("850", "USD"), claimed_post_dst=("350", "USD"))
    assert split.verdict == "INVALID" and "disagree" in split.reason
    # Without conservation required, a wrong split is still caught by the recompute check
    loose = _run(_ont(require_conservation=False), claimed_post_src=("1", "USD"))
    assert loose.verdict == "INVALID" and "disagree" in loose.reason


def test_tie_abstains_and_escalates():
    v = _run(src_witnesses={"w1": ("1000", "USD"), "w2": ("1001", "USD")})
    assert v.verdict == "ABSTAIN_TIE" and v.human_escalation_required and v.resolved_initial_src is None
