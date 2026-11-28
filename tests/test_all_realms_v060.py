"""Unit, security, compliance, and CLI tests for v0.6.0 All-Realms Suite (T-43)."""

from __future__ import annotations

import copy
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from hadukenn_core.all_realms_suite import (
    QuantizedINT8NPULoopEngine,
    VerifiableGraphCheckpointer,
)
from sheaf_verify.campaign import verify_t43_all_realms
from sheaf_verify.cli import main as cli_main
from sheaf_verify.compliance import (
    ActionRuleSpec,
    ExactUnitLattice,
    LocalActionOntology,
    emit_regulatory_bundle,
    verify_regulatory_bundle,
)
from sheaf_verify.hashing import SheafError

ROOT = Path(__file__).resolve().parents[1]
T43_JSON = ROOT / "iterations" / "t43-all-realms-dominance" / "T43_ALL_REALMS_RESULTS.json"


def test_exact_unit_lattice_and_mismatch_refusal() -> None:
    tot_l, sym_l = ExactUnitLattice.add("750", "mL", "1.25", "L")
    assert tot_l == Fraction(2, 1) and sym_l == "L"

    diff_usd, sym_u = ExactUnitLattice.sub("2.5", "kUSD", "50000", "cents")
    assert diff_usd == Fraction(2000, 1) and sym_u == "USD"

    with pytest.raises(SheafError, match="dimensional mismatch"):
        ExactUnitLattice.add("10", "USD", "5", "L")

    with pytest.raises(SheafError, match="dimensional mismatch"):
        ExactUnitLattice.sub("10", "USD", "5", "L")

    with pytest.raises(SheafError, match="unknown unit"):
        ExactUnitLattice.normalize("10", "parsecs")


def test_local_action_ontology_all_six_verdicts_and_edge_branches() -> None:
    with pytest.raises(SheafError, match="ontology_id"):
        LocalActionOntology("   ", [])

    ont = LocalActionOntology(
        "test.ontology.v1",
        [ActionRuleSpec(action_name="wire_funds", dimension="currency", max_amount=Fraction(5000, 1))],
    )

    # Undeclared action verb -> IMPOSSIBLE
    v_undec = ont.evaluate_transfer_action(
        action_name="unknown_verb",
        actor_role="operator",
        src_entity="VaultA",
        dst_entity="VaultB",
        src_witnesses={"w1": ("1000", "USD")},
        dst_initial=("200", "USD"),
        transfer_amount=("100", "USD"),
    )
    assert v_undec.verdict == "IMPOSSIBLE" and v_undec.as_dict()["approved"] is False

    # Unauthorized actor role -> IMPOSSIBLE
    v_role = ont.evaluate_transfer_action(
        action_name="wire_funds",
        actor_role="guest",
        src_entity="VaultA",
        dst_entity="VaultB",
        src_witnesses={"w1": ("1000", "USD")},
        dst_initial=("200", "USD"),
        transfer_amount=("100", "USD"),
    )
    assert v_role.verdict == "IMPOSSIBLE"

    # Witness dimension mismatch -> IMPOSSIBLE
    v_dim = ont.evaluate_transfer_action(
        action_name="wire_funds",
        actor_role="operator",
        src_entity="VaultA",
        dst_entity="VaultB",
        src_witnesses={"w1": ("1000", "L")},
        dst_initial=("200", "USD"),
        transfer_amount=("100", "USD"),
    )
    assert v_dim.verdict == "IMPOSSIBLE"

    # Destination dimension mismatch -> IMPOSSIBLE
    v_dst_dim = ont.evaluate_transfer_action(
        action_name="wire_funds",
        actor_role="operator",
        src_entity="VaultA",
        dst_entity="VaultB",
        src_witnesses={"w1": ("1000", "USD")},
        dst_initial=("200", "L"),
        transfer_amount=("100", "USD"),
    )
    assert v_dst_dim.verdict == "IMPOSSIBLE"

    # Exceeds max_amount -> IMPOSSIBLE
    v_max = ont.evaluate_transfer_action(
        action_name="wire_funds",
        actor_role="operator",
        src_entity="VaultA",
        dst_entity="VaultB",
        src_witnesses={"w1": ("10000", "USD")},
        dst_initial=("200", "USD"),
        transfer_amount=("6000", "USD"),
    )
    assert v_max.verdict == "IMPOSSIBLE"

    # 1. VALID
    v_ok = ont.evaluate_transfer_action(
        action_name="wire_funds",
        actor_role="operator",
        src_entity="VaultA",
        dst_entity="VaultB",
        src_witnesses={"w1": ("1.5", "kUSD")},
        dst_initial=("200", "USD"),
        transfer_amount=("300", "USD"),
        claimed_post_src=("1200", "USD"),
        claimed_post_dst=("500", "USD"),
    )
    assert v_ok.verdict == "VALID" and v_ok.approved is True

    # 2a. INVALID (claimed post-balance violates conservation)
    v_inv = ont.evaluate_transfer_action(
        action_name="wire_funds",
        actor_role="operator",
        src_entity="VaultA",
        dst_entity="VaultB",
        src_witnesses={"w1": ("1000", "USD")},
        dst_initial=("200", "USD"),
        transfer_amount=("300", "USD"),
        claimed_post_src=("999", "USD"),
        claimed_post_dst=("500", "USD"),
    )
    assert v_inv.verdict == "INVALID" and v_inv.approved is False

    # 2b. INVALID (conserved total holds, but swapped/wrong split)
    v_split = ont.evaluate_transfer_action(
        action_name="wire_funds",
        actor_role="operator",
        src_entity="VaultA",
        dst_entity="VaultB",
        src_witnesses={"w1": ("1000", "USD")},
        dst_initial=("200", "USD"),
        transfer_amount=("300", "USD"),
        claimed_post_src=("600", "USD"),
        claimed_post_dst=("600", "USD"),
    )
    assert v_split.verdict == "INVALID"

    # 2c. INVALID (unknown unit in claimed post-balance)
    v_badu = ont.evaluate_transfer_action(
        action_name="wire_funds",
        actor_role="operator",
        src_entity="VaultA",
        dst_entity="VaultB",
        src_witnesses={"w1": ("1000", "USD")},
        dst_initial=("200", "USD"),
        transfer_amount=("300", "USD"),
        claimed_post_src=("700", "badunit"),
    )
    assert v_badu.verdict == "INVALID"

    # 3. IMPOSSIBLE (overdraft / negative post-balance)
    v_imp = ont.evaluate_transfer_action(
        action_name="wire_funds",
        actor_role="operator",
        src_entity="VaultA",
        dst_entity="VaultB",
        src_witnesses={"w1": ("100", "USD")},
        dst_initial=("200", "USD"),
        transfer_amount=("300", "USD"),
    )
    assert v_imp.verdict == "IMPOSSIBLE" and v_imp.approved is False

    # 4. TRANSLATION_AMBIGUOUS (unresolved destination pronoun)
    v_amb = ont.evaluate_transfer_action(
        action_name="wire_funds",
        actor_role="operator",
        src_entity="VaultA",
        dst_entity=None,
        src_witnesses={"w1": ("1000", "USD")},
        dst_initial=("200", "USD"),
        transfer_amount=("300", "USD"),
    )
    assert v_amb.verdict == "TRANSLATION_AMBIGUOUS" and v_amb.human_escalation_required is True

    # 5. ABSTAIN_TIE (1-vs-1 conflicting witnesses)
    v_tie = ont.evaluate_transfer_action(
        action_name="wire_funds",
        actor_role="operator",
        src_entity="VaultA",
        dst_entity="VaultB",
        src_witnesses={"w1": ("1000", "USD"), "w2": ("4000", "USD")},
        dst_initial=("200", "USD"),
        transfer_amount=("300", "USD"),
    )
    assert v_tie.verdict == "ABSTAIN_TIE" and v_tie.human_escalation_required is True

    # 6. RECOVERABLE_FAULT (2-vs-1 witness consensus isolates poisoned feed)
    v_rec = ont.evaluate_transfer_action(
        action_name="wire_funds",
        actor_role="operator",
        src_entity="VaultA",
        dst_entity="VaultB",
        src_witnesses={"w1": ("1000", "USD"), "w_bad": ("9999", "USD"), "w2": ("1", "kUSD")},
        dst_initial=("200", "USD"),
        transfer_amount=("300", "USD"),
    )
    assert v_rec.verdict == "RECOVERABLE_FAULT"
    assert v_rec.approved is True
    assert v_rec.isolated_fault_sources == ("w_bad",)
    assert v_rec.post_src_balance == "700" and v_rec.post_dst_balance == "500"


def test_regulatory_bundle_verifiable_checkpointer_and_npu() -> None:
    ckpt = VerifiableGraphCheckpointer("test-finra-euai")
    assert ckpt.put_superstep("main", {"a": (500, "USD"), "b": (500, "USD")})["status"] == "ok"
    assert ckpt.put_superstep("main", {"a": (350, "USD"), "b": (650, "USD")})["status"] == "ok"
    assert ckpt.put_superstep("main", {"a": (900, "USD"), "b": (650, "USD")})["status"] == "refused"
    assert ckpt.fork_thread("main", "alt")["status"] == "ok"
    assert ckpt.verify_integrity() is True

    key = b"test-compliance-secret"
    pol_sha = "a" * 64
    bundle = ckpt.export_compliance_bundle("main", policy_sha256=pol_sha, signing_key=key)
    ok_b, errs = verify_regulatory_bundle(bundle, signing_key=key)
    assert ok_b is True and not errs

    ok_wrong_key, errs_wk = verify_regulatory_bundle(bundle, signing_key=b"wrong-key")
    assert ok_wrong_key is False and errs_wk

    ok_bad_schema, _ = verify_regulatory_bundle({"bundle_schema": "bad"}, signing_key=key)
    assert ok_bad_schema is False

    ok_empty_ev, _ = verify_regulatory_bundle(
        {"bundle_schema": "sheaf-verify.regulatory-bundle.v1", "events": []}, signing_key=key
    )
    assert ok_empty_ev is False

    bad_bundle = copy.deepcopy(bundle)
    bad_bundle["events"][0]["step"] = 999
    ok_bad, errs_bad = verify_regulatory_bundle(bad_bundle, signing_key=key)
    assert ok_bad is False and errs_bad

    with pytest.raises(SheafError, match="retention_months >= 6"):
        emit_regulatory_bundle(
            system_id="s",
            trace_head_sha256="b" * 64,
            state_digest_sha256="c" * 64,
            policy_sha256=pol_sha,
            events=[{"step": 1}],
            created_at="2026-11-25T00:00:00Z",
            signing_key=key,
            retention_months=2,
        )

    with pytest.raises(SheafError, match="at least one recorded event"):
        emit_regulatory_bundle(
            system_id="s",
            trace_head_sha256="b" * 64,
            state_digest_sha256="c" * 64,
            policy_sha256=pol_sha,
            events=[],
            created_at="2026-11-25T00:00:00Z",
            signing_key=key,
        )

    rng = np.random.default_rng(7)
    w = rng.normal(0.0, 0.3, size=(32, 24)).astype(np.float32)
    x = rng.normal(0.0, 0.5, size=(16, 32)).astype(np.float32)
    npu = QuantizedINT8NPULoopEngine(w)
    y_npu = npu.forward_npu_int8(x)
    y_fp = x @ w
    rel_err = float(np.linalg.norm(y_npu - y_fp) / np.linalg.norm(y_fp))
    assert rel_err < 0.03


def test_verify_t43_all_realms_and_cli() -> None:
    doc = json.loads(T43_JSON.read_text(encoding="utf-8"))
    ok_flag, criteria, advisories = verify_t43_all_realms(doc)
    assert ok_flag is True
    assert len(criteria) == 5 and all(c.passed for c in criteria)
    assert any("DUAL_TRACK_DISCIPLINE" in a for a in advisories)

    broken = copy.deepcopy(doc)
    broken["realm_a_noisy_nl_compiler"]["silent_conflict_corruption_rate_hadukenn"] = 0.25
    ok_bad, _, _ = verify_t43_all_realms(broken)
    assert ok_bad is False

    rc = cli_main(["campaign", "--mode", "t43-all-realms", "--json", "--path", str(T43_JSON)])
    assert rc == 0


def test_local_action_ontology_exact_fraction_witness_consensus() -> None:
    ont = LocalActionOntology(
        "test.exact.consensus.v1",
        [ActionRuleSpec(action_name="wire_funds", dimension="currency")],
    )

    # Large integer difference that collapses under IEEE-754 float64
    v_big = ont.evaluate_transfer_action(
        action_name="wire_funds",
        actor_role="operator",
        src_entity="VaultA",
        dst_entity="VaultB",
        src_witnesses={"w1": ("100000000000000000", "USD"), "w2": ("100000000000000001", "USD")},
        dst_initial=("0", "USD"),
        transfer_amount=("10", "USD"),
    )
    assert v_big.verdict == "ABSTAIN_TIE"
    assert v_big.approved is False
    assert v_big.human_escalation_required is True

    # Tiny 1e-7 tie below float cohomology tolerance (tol=1e-6)
    v_tiny_tie = ont.evaluate_transfer_action(
        action_name="wire_funds",
        actor_role="operator",
        src_entity="VaultA",
        dst_entity="VaultB",
        src_witnesses={"w1": ("1000", "USD"), "w2": ("1000.0000001", "USD")},
        dst_initial=("0", "USD"),
        transfer_amount=("10", "USD"),
    )
    assert v_tiny_tie.verdict == "ABSTAIN_TIE"
    assert v_tiny_tie.approved is False
    assert v_tiny_tie.human_escalation_required is True

    # Tiny 1e-7 2-vs-1 outlier must isolate faulty witness
    v_tiny_fault = ont.evaluate_transfer_action(
        action_name="wire_funds",
        actor_role="operator",
        src_entity="VaultA",
        dst_entity="VaultB",
        src_witnesses={"w1": ("1000", "USD"), "w2": ("1000", "USD"), "w3": ("1000.0000001", "USD")},
        dst_initial=("200", "USD"),
        transfer_amount=("100", "USD"),
    )
    assert v_tiny_fault.verdict == "RECOVERABLE_FAULT"
    assert v_tiny_fault.approved is True
    assert v_tiny_fault.isolated_fault_sources == ("w3",)


def test_local_action_ontology_rejects_post_state_dimension_mismatch() -> None:
    ont = LocalActionOntology(
        "test.postdim.v1",
        [ActionRuleSpec(action_name="wire_funds", dimension="currency")],
    )

    v_both = ont.evaluate_transfer_action(
        action_name="wire_funds",
        actor_role="operator",
        src_entity="VaultA",
        dst_entity="VaultB",
        src_witnesses={"w1": ("1000", "USD")},
        dst_initial=("200", "USD"),
        transfer_amount=("100", "USD"),
        claimed_post_src=("900", "kg"),
        claimed_post_dst=("300", "L"),
    )
    assert v_both.verdict == "INVALID"
    assert v_both.approved is False
    assert "dimension" in v_both.reason

    v_src_only = ont.evaluate_transfer_action(
        action_name="wire_funds",
        actor_role="operator",
        src_entity="VaultA",
        dst_entity="VaultB",
        src_witnesses={"w1": ("1000", "USD")},
        dst_initial=("200", "USD"),
        transfer_amount=("100", "USD"),
        claimed_post_src=("900", "kg"),
    )
    assert v_src_only.verdict == "INVALID"
    assert v_src_only.approved is False

    v_dst_only = ont.evaluate_transfer_action(
        action_name="wire_funds",
        actor_role="operator",
        src_entity="VaultA",
        dst_entity="VaultB",
        src_witnesses={"w1": ("1000", "USD")},
        dst_initial=("200", "USD"),
        transfer_amount=("100", "USD"),
        claimed_post_dst=("300", "L"),
    )
    assert v_dst_only.verdict == "INVALID"
    assert v_dst_only.approved is False


def test_regulatory_bundle_rejects_tampered_metadata_and_predicates() -> None:
    from sheaf_verify.attestation import dsse_sign, dsse_verify

    key = b"test-compliance-secret"
    pol_sha = "a" * 64
    trace_sha = "b" * 64
    state_sha = "c" * 64
    events = [{"step": 0, "approved": True}, {"step": 1, "approved": True}]
    bundle = emit_regulatory_bundle(
        system_id="sys-alpha",
        trace_head_sha256=trace_sha,
        state_digest_sha256=state_sha,
        policy_sha256=pol_sha,
        events=events,
        created_at="2026-11-28T00:00:00Z",
        signing_key=key,
    )
    ok_b, errs_b = verify_regulatory_bundle(bundle, signing_key=key)
    assert ok_b is True and errs_b == []

    # Non-dict bundle
    for bad_obj in (None, [], "bad"):
        ok_n, _ = verify_regulatory_bundle(bad_obj, signing_key=key)  # type: ignore[arg-type]
        assert ok_n is False

    # Tamper top-level metadata fields
    for field, val in (
        ("system_id", "sys-evil"),
        ("created_at", "2099-01-01T00:00:00Z"),
        ("policy_sha256", "f" * 64),
    ):
        tampered = copy.deepcopy(bundle)
        tampered[field] = val
        ok_t, errs_t = verify_regulatory_bundle(tampered, signing_key=key)
        assert ok_t is False and errs_t

    # Re-sign DSSE with tampered predicate/subject fields
    base_stmt = dsse_verify(bundle["dsse_statement_envelope"], key=key)

    def _check_resigned(mutator):
        t = copy.deepcopy(bundle)
        stmt = copy.deepcopy(base_stmt)
        mutator(stmt)
        t["dsse_statement_envelope"] = dsse_sign(stmt, key=key)
        ok_m, errs_m = verify_regulatory_bundle(t, signing_key=key)
        assert ok_m is False and errs_m

    _check_resigned(lambda s: s.__setitem__("predicateType", "https://example.org/other"))
    _check_resigned(lambda s: s["predicate"].__setitem__("system_id", "sys-evil"))
    _check_resigned(lambda s: s["predicate"].__setitem__("created_at", "2099-01-01T00:00:00Z"))
    _check_resigned(lambda s: s["predicate"]["finra_rule_3110_worm"].__setitem__("policy_snapshot_sha256", "d" * 64))
    _check_resigned(lambda s: s["predicate"]["finra_rule_3110_worm"].__setitem__("worm_integrity_sha256", "e" * 64))
    _check_resigned(lambda s: s["predicate"]["eu_ai_act_article_12"].__setitem__("total_events_logged", 99))
    _check_resigned(lambda s: s["predicate"]["ietf_scitt_rfc9943"].__setitem__("receipt_tree_size", 99))
    _check_resigned(lambda s: s["subject"][0].__setitem__("name", "other-sys/events.json"))

    # Transplant SCITT receipt from a different bundle signed with the same key
    other = emit_regulatory_bundle(
        system_id="sys-alpha",
        trace_head_sha256=trace_sha,
        state_digest_sha256=state_sha,
        policy_sha256=pol_sha,
        events=[{"step": 999, "approved": True}],
        created_at="2026-11-28T01:00:00Z",
        signing_key=key,
    )
    transplanted = copy.deepcopy(bundle)
    transplanted["scitt_merkle_receipt"] = other["scitt_merkle_receipt"]
    ok_tr, errs_tr = verify_regulatory_bundle(transplanted, signing_key=key)
    assert ok_tr is False and errs_tr

    # 2-event SCITT receipt transplant sharing events[-1], len(events), created_at, signing_key
    # but differing in events[0] must fail full Merkle root recomputation
    other_2 = emit_regulatory_bundle(
        system_id="sys-other",
        trace_head_sha256=trace_sha,
        state_digest_sha256=state_sha,
        policy_sha256=pol_sha,
        events=[{"step": 999, "approved": True}, events[-1]],
        created_at="2026-11-28T00:00:00Z",
        signing_key=key,
    )
    transplanted_2 = copy.deepcopy(bundle)
    transplanted_2["scitt_merkle_receipt"] = other_2["scitt_merkle_receipt"]
    ok_tr2, errs_tr2 = verify_regulatory_bundle(transplanted_2, signing_key=key)
    assert ok_tr2 is False and any("merkle_root_sha256" in e for e in errs_tr2)

    # Non-canonical float("nan") in bundle["events"] must return (False, errors) cleanly
    nan_bundle = copy.deepcopy(bundle)
    nan_bundle["events"] = [{"action": float("nan")}]
    ok_nan, errs_nan = verify_regulatory_bundle(nan_bundle, signing_key=key)
    assert ok_nan is False and errs_nan


def test_compliance_exact_unit_lattice_and_ontology_rejects_malformed_inputs() -> None:
    for bad_val in (1.5, float("nan"), True, False, "abc", "1/0", None):
        with pytest.raises(SheafError):
            ExactUnitLattice.normalize(bad_val, "USD")  # type: ignore[arg-type]

    ont = LocalActionOntology(
        "test.malformed.v1",
        [ActionRuleSpec(action_name="wire_funds", dimension="currency")],
    )

    for bad_val in (1.5, float("nan"), True, False, "abc", "1/0"):
        v_post_src = ont.evaluate_transfer_action(
            action_name="wire_funds",
            actor_role="operator",
            src_entity="VaultA",
            dst_entity="VaultB",
            src_witnesses={"w1": ("1000", "USD")},
            dst_initial=("200", "USD"),
            transfer_amount=("100", "USD"),
            claimed_post_src=(bad_val, "USD"),  # type: ignore[arg-type]
        )
        assert v_post_src.verdict == "INVALID" and v_post_src.approved is False

        v_post_dst = ont.evaluate_transfer_action(
            action_name="wire_funds",
            actor_role="operator",
            src_entity="VaultA",
            dst_entity="VaultB",
            src_witnesses={"w1": ("1000", "USD")},
            dst_initial=("200", "USD"),
            transfer_amount=("100", "USD"),
            claimed_post_dst=(bad_val, "USD"),  # type: ignore[arg-type]
        )
        assert v_post_dst.verdict == "INVALID" and v_post_dst.approved is False

        v_wit = ont.evaluate_transfer_action(
            action_name="wire_funds",
            actor_role="operator",
            src_entity="VaultA",
            dst_entity="VaultB",
            src_witnesses={"w1": (bad_val, "USD")},  # type: ignore[dict-item]
            dst_initial=("200", "USD"),
            transfer_amount=("100", "USD"),
        )
        assert v_wit.verdict == "IMPOSSIBLE" and v_wit.approved is False
