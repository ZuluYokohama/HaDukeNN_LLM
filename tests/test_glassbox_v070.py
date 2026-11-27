"""Unit, unlearning, counterfactual, sheaf-nullspace, and CLI tests for v0.7.0 Glass-Box AI (T-44)."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

from sheaf_verify.campaign import verify_t44_glassbox
from sheaf_verify.cli import main as cli_main
from sheaf_verify.compliance import verify_regulatory_bundle
from sheaf_verify.glassbox import (
    CausalMDLSpanGraph,
    GlassBoxRuleAndSheafInducer,
    GlassBoxSequenceEngine,
    GlassBoxWorldKernel,
)
from sheaf_verify.hashing import SheafError

ROOT = Path(__file__).resolve().parents[1]
T44_JSON = ROOT / "iterations" / "t44-glassbox-no-blackbox" / "T44_GLASSBOX_RESULTS.json"


def test_causal_mdl_span_graph_roundtrip_and_prefix_determinism() -> None:
    sample = (b"Alpha holds 20 L. Beta holds 5 L. Alpha transfers 12 L to Beta.\n" * 6)
    sg = CausalMDLSpanGraph()
    v_size = sg.fit_causal_spans(sample, max_merges=32, min_pair_count=3)
    assert v_size > 256
    encoded = sg.encode_bytes(sample)
    assert len(encoded) < len(sample)
    assert sg.decode_tokens(encoded) == sample

    # Short input edge case
    sg_tiny = CausalMDLSpanGraph()
    assert sg_tiny.fit_causal_spans(b"x") == 256
    with pytest.raises(SheafError, match="Unknown span token ID"):
        sg.decode_tokens([999999])


def test_glassbox_sequence_engine_exact_unlearning_and_attribution() -> None:
    corpus_a = b"VaultA holds 30 L and PoolB holds 10 L. VaultA transfers 5 L to PoolB.\n" * 4
    corpus_b = b"PoolB holds 15 L and TankC holds 25 L. TankC returns 5 L to PoolB.\n" * 4
    canary = b"SECRET_CANARY_OVERRIDE = ZETA-7741-ALPHA.\n" * 3

    sg = CausalMDLSpanGraph()
    sg.fit_causal_spans(corpus_a + corpus_b, max_merges=24, min_pair_count=2)

    with pytest.raises(SheafError, match="max_order"):
        GlassBoxSequenceEngine(sg, max_order=0)
    with pytest.raises(SheafError, match="discount"):
        GlassBoxSequenceEngine(sg, discount=1.2)

    clean = GlassBoxSequenceEngine(sg, max_order=4)
    clean.ingest_document("doc-a", corpus_a)
    clean.ingest_document("doc-b", corpus_b)
    assert clean.document_ids == ("doc-a", "doc-b")

    with pytest.raises(SheafError, match="already ingested"):
        clean.ingest_document("doc-a", corpus_a)
    with pytest.raises(SheafError, match="Cannot unlearn unknown"):
        clean.unlearn_document("missing-doc")
    with pytest.raises(SheafError, match="eval_bytes must be non-empty"):
        clean.evaluate_bpb(b"")

    contaminated = GlassBoxSequenceEngine(sg, max_order=4)
    contaminated.ingest_document("doc-a", corpus_a)
    contaminated.ingest_document("canary", canary)
    contaminated.ingest_document("doc-b", corpus_b)
    assert contaminated.state_merkle_sha256() != clean.state_merkle_sha256()

    contaminated.unlearn_document("canary")
    assert contaminated.state_merkle_sha256() == clean.state_merkle_sha256()

    probe = sg.encode_bytes(b"SECRET_CANARY_OVERRIDE = ")
    p_clean, _, _ = clean.predict_distribution(probe)
    p_unlearned, _, _ = contaminated.predict_distribution(probe)
    assert sum(abs(x - y) for x, y in zip(p_clean, p_unlearned, strict=True)) == 0.0

    # Supported explanation + provenance + hyperedge witnesses
    expl = clean.explain_prediction(b"VaultA transfers 5 L to ")
    assert expl.epistemic_status == "SUPPORTED"
    assert len(expl.provenance_citations) >= 1
    assert len(expl.hyperedge_witnesses) >= 1
    assert abs(sum(expl.order_mixture_weights.values()) - 1.0) < 1e-4

    # Out-of-support unseen byte sequence triggers EPISTEMIC_ABSTAIN
    abstain_expl = clean.explain_prediction(b"\xfe\xff\xfd\xfb")
    assert abstain_expl.epistemic_status == "EPISTEMIC_ABSTAIN"


def test_rule_induction_q_nullspace_laws_counterfactual_branches_and_scitt() -> None:
    pairs = [
        ("A holds 20 L", {"op": "init", "entity": "A", "amount": "20", "unit": "L"}),
        ("B holds 5 L", {"op": "init", "entity": "B", "amount": "5", "unit": "L"}),
        ("A transfers 12 L to B", {"op": "transfer", "source": "A", "amount": "12", "unit": "L", "destination": "B"}),
        ("B returns 3 L to A", {"op": "transfer", "source": "B", "amount": "3", "unit": "L", "destination": "A"}),
        ("assert A equals 11 L", {"op": "assert_eq", "entity": "A", "amount": "11", "unit": "L"}),
        ("retract B", {"op": "retract", "entity": "B"}),
        ("query A", {"op": "query", "entity": "A"}),
    ]
    trajectories = [
        [
            {"A": Fraction(20), "B": Fraction(5)},
            {"A": Fraction(8), "B": Fraction(17)},
            {"A": Fraction(11), "B": Fraction(14)},
        ],
    ]
    inducer = GlassBoxRuleAndSheafInducer()
    rules = inducer.induce_rules(pairs)
    laws = inducer.discover_sheaf_invariants(trajectories)
    assert len(rules) >= 5
    assert len(laws) == 1
    assert "A + B == invariant_constant" in laws[0].formula_str()

    kernel = GlassBoxWorldKernel(inducer, canonical_unit="L")
    assert kernel.step("A holds 20 L").status == "ACCEPTED"
    assert kernel.step("B holds 5 L").status == "ACCEPTED"

    # Fork hypothetical counterfactual branch before main transfer
    kernel.fork_branch("hypo", from_branch="main")
    assert kernel.step("A transfers 8000 mL to B", branch_id="hypo").status == "ACCEPTED"
    assert kernel.branch_state("hypo") == {"A": Fraction(12), "B": Fraction(13)}

    # Main branch executes 12 L transfer + 3 L return -> A=11, B=14 (sum=25 conserved)
    assert kernel.step("A transfers 12 L to B").status == "ACCEPTED"
    assert kernel.step("B returns 3 L to A").status == "ACCEPTED"
    assert kernel.branch_state("main") == {"A": Fraction(11), "B": Fraction(14)}
    assert kernel.branch_state("hypo") == {"A": Fraction(12), "B": Fraction(13)}

    # Adversarial assertion A = 99 L is refused with exact coboundary residual 88!
    bad_assert = kernel.step("assert A equals 99 L")
    assert bad_assert.status == "REFUSED_INVARIANT_VIOLATION"
    assert bad_assert.coboundary_residual == "88"

    # Overdraft transfer making stalk negative is refused!
    bad_overdraft = kernel.step("A transfers 50 L to B")
    assert bad_overdraft.status == "REFUSED_INVARIANT_VIOLATION"

    # Dimensional mismatch (USD instead of L) is refused!
    bad_dim = kernel.step("A transfers 5 USD to B")
    assert bad_dim.status == "REFUSED_INVARIANT_VIOLATION"

    # Unknown syntax abstains safely
    abstain_rec = kernel.step("tell me a bedtime poem about stars")
    assert abstain_rec.status == "ABSTAIN_NO_RULE"

    # Retraction invalidates dependent queries rather than returning stale state
    assert kernel.step("retract B").status == "ACCEPTED"
    assert kernel.step("query B").status == "UNRESOLVED_DEPENDENCY"
    assert kernel.step("query A").status == "ACCEPTED"

    # Full semantic replay & SCITT RFC 9943 verification
    assert kernel.replay_and_verify("main") is True
    assert kernel.replay_and_verify("hypo") is True
    key = b"unit-test-scitt-key"
    bundle = kernel.export_scitt_compliance_bundle(branch_id="main", signing_key=key)
    ok_b, probs = verify_regulatory_bundle(bundle, signing_key=key)
    assert ok_b is True and not probs

    with pytest.raises(SheafError, match="Unknown source branch"):
        kernel.fork_branch("x", from_branch="nope")
    with pytest.raises(SheafError, match="already exists"):
        kernel.fork_branch("hypo")


def test_t44_campaign_json_and_cli_gate() -> None:
    assert T44_JSON.is_file(), "T44_GLASSBOX_RESULTS.json must be generated before running tests"
    import json

    doc = json.loads(T44_JSON.read_text(encoding="utf-8"))
    ok_flag, criteria, advisories = verify_t44_glassbox(doc)
    assert ok_flag is True, [c.as_dict() for c in criteria if not c.passed]
    assert len(criteria) == 5 and len(advisories) >= 1

    rc = cli_main(["campaign", "--mode", "t44-glassbox", "--path", str(T44_JSON), "--json"])
    assert rc == 0
