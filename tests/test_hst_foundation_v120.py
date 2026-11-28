"""Comprehensive unit, invariant, manifest tamper, width-scaling, and CLI tests for T-49 HST-v1 & HaDukeNN-LM."""

from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

from hadukenn_core.hst_foundation_lm import (
    LIVE_T38_FIVE_RUNG_1EP_GAPS,
    STAGED_FOUNDATION_LADDER,
    build_foundation_corpus,
    run_inline_receipt_gated_decode,
    run_mup_width_transfer_probe,
)
from sheaf_verify.campaign import verify_t48_roadmap_complete, verify_t49_hst_foundation
from sheaf_verify.cli import main as cli_main
from sheaf_verify.hst_standard import (
    FIRST_MERGE_ID,
    HST_SCHEMA_ID,
    RESERVED_CONTROL_TOKENS,
    UNIVERSAL_CONTROL_OFFSET,
    HSTTokenizer,
    verify_hst_manifest_integrity,
)

ROOT = Path(__file__).resolve().parents[1]


def test_hst_v1_tokenizer_modes_digit_atomicity_and_prefix_invariance():
    tr_docs, te_docs, _, _ = build_foundation_corpus(seed=13)
    tok = HSTTokenizer.train(
        tr_docs, vocab_size=360, stage_a_fraction=0.65, min_pair_count=2, min_superspan_pmi=1.1, max_span_bytes=16
    )
    assert tok.vocab_size == 360
    assert tok.control_map["<|hst:bos|>"] == UNIVERSAL_CONTROL_OFFSET
    assert tok.control_map["<|hst:receipt|>"] == FIRST_MERGE_ID - 1
    assert len(RESERVED_CONTROL_TOKENS) == 8
    assert tok.v_base_subword < tok.vocab_size

    parts, counts = tok.stage_b_subword_parts()
    assert len(parts) == len(counts) == tok.vocab_size - tok.v_base_subword
    assert all(all(p < tok.v_base_subword for p in tup) for tup in parts)

    # Lossless roundtrip across all 3 encoding modes + arbitrary binary octets
    sample_raw = (te_docs[0] + " Binary tail: \x00\x01\xfe\xff 98765").encode("utf-8", errors="surrogatepass")
    for mode in ("causal_local", "staged_bpe", "dp_min_tokens"):
        ids = tok.encode_bytes(sample_raw, mode=mode)
        assert tok.decode_bytes(ids) == sample_raw
    assert len(tok.encode_bytes(sample_raw, mode="dp_min_tokens")) <= len(
        tok.encode_bytes(sample_raw, mode="causal_local")
    )
    assert tok.encode_bytes(b"") == []
    assert tok.encode("") == []

    # Digit atomicity check
    digit_ids = tok.encode_bytes(b"2048", mode="causal_local")
    assert [tok.id_to_bytes[i] for i in digit_ids] == [b"2", b"0", b"4", b"8"]

    # Control token parsing + decode roundtrip
    tagged = "<|hst:bos|> reserve 25 <|hst:unit|> L <|hst:assert|> ok <|hst:eos|>"
    tagged_ids = tok.encode(tagged, mode="causal_local")
    assert tok.control_map["<|hst:bos|>"] in tagged_ids
    assert tok.control_map["<|hst:unit|>"] in tagged_ids
    assert tok.decode(tagged_ids) == tagged

    # StochasTok preserves exact decoded byte stream at all drop probabilities
    rng = random.Random(42)
    base_ids = tok.encode_bytes(te_docs[1].encode("utf-8"), mode="causal_local")
    assert tok.encode_stochastok(base_ids, 0.0, rng) == base_ids
    split_ids = tok.encode_stochastok(base_ids, 0.5, rng)
    assert len(split_ids) >= len(base_ids)
    assert tok.decode_bytes(split_ids) == te_docs[1].encode("utf-8")

    # Bounded causal prefix invariance
    ok_pref, rep = tok.verify_causal_prefix_invariance(
        te_docs[0].encode("utf-8"),
        [b"extra suffix one", b"999999", b"\nsecond line check"],
    )
    assert ok_pref is True and rep["violations"] == 0


def test_hst_v1_manifest_merkle_verification_and_tamper_rejection():
    manifest_path = ROOT / "iterations" / "t49-hst-foundation-stage" / "HST_V1_REFERENCE_MANIFEST.json"
    doc = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert doc["schema"] == HST_SCHEMA_ID
    ok_m, probs = verify_hst_manifest_integrity(doc)
    assert ok_m is True and probs == []

    reloaded = HSTTokenizer.from_manifest(doc)
    assert reloaded.merkle_root_sha256() == doc["merkle_root_sha256"]

    # Tamper Merkle root -> must fail
    tampered = dict(doc)
    tampered["merkle_root_sha256"] = "0" * 64
    ok_bad, bad_probs = verify_hst_manifest_integrity(tampered)
    assert ok_bad is False and any("Merkle root mismatch" in msg for msg in bad_probs)

    counted = json.loads(json.dumps(doc))
    changed = False
    for entry in counted["merges"]:
        if entry["stage"] == "B_superspan":
            entry["count"] = 1_000_000_000
            changed = True
            break
    assert changed
    ok_count, count_probs = verify_hst_manifest_integrity(counted)
    assert ok_count is False and any("Merkle root mismatch" in msg for msg in count_probs)

    # Invalid mode / invalid token ID / invalid vocab size checks
    with pytest.raises(ValueError, match="Unknown HST encode mode"):
        reloaded.encode_bytes(b"abc", mode="nonexistent")
    with pytest.raises(ValueError, match="Unknown HST token ID"):
        reloaded.decode_bytes([999_999])
    with pytest.raises(ValueError, match="vocab_size must be >="):
        HSTTokenizer.train(["short"], vocab_size=260)
    with pytest.raises(ValueError, match="stage_a_fraction"):
        HSTTokenizer.train(["short"], vocab_size=300, stage_a_fraction=0.995)


def test_t49_mup_probe_receipt_gate_and_cli_verification():
    artifact = ROOT / "iterations" / "t49-hst-foundation-stage" / "T49_HST_FOUNDATION_RESULTS.json"
    doc = json.loads(artifact.read_text(encoding="utf-8"))
    ok_flag, criteria, advisories = verify_t49_hst_foundation(doc)
    assert ok_flag is True
    assert all(c.passed for c in criteria)
    assert any("HST_FOUNDATION_STAGE_READY" in a for a in advisories)

    # Negative check: corrupt wins_all_5_seeds -> must fail T49_EQUAL_BYTES_FOUNDATION_LM_WIN
    broken = json.loads(json.dumps(doc))
    broken["pillar_2_equal_bytes_foundation_lm"]["wins_all_5_seeds"] = False
    ok_b, crit_b, _ = verify_t49_hst_foundation(broken)
    assert ok_b is False
    assert any(c.id == "T49_EQUAL_BYTES_FOUNDATION_LM_WIN" and not c.passed for c in crit_b)

    # Verify live 5-rung 1ep monotonic progression XS > S > M > M+ > L and staged ladder
    gaps = [LIVE_T38_FIVE_RUNG_1EP_GAPS[r]["delta_h_minus_p"] for r in ("XS", "S", "M", "M+", "L")]
    assert gaps == sorted(gaps, reverse=True)
    assert len(STAGED_FOUNDATION_LADDER) == 5

    # Fast inline receipt gate & muP smoke
    manifest_doc = json.loads(
        (ROOT / "iterations" / "t49-hst-foundation-stage" / "HST_V1_REFERENCE_MANIFEST.json").read_text(
            encoding="utf-8"
        )
    )
    tok = HSTTokenizer.from_manifest(manifest_doc)
    gate_res = run_inline_receipt_gated_decode(tok, seed=21)
    assert gate_res["spoof_intercepted_and_refused"] is True
    assert gate_res["semantic_replay_verified"] is True

    tr_docs, te_docs, _, _ = build_foundation_corpus(seed=13)
    probe = run_mup_width_transfer_probe(tok, "".join(tr_docs[:24]), "".join(te_docs[:8]), seed=13)
    assert probe["base_width_d"] == 32

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "sheaf_verify",
            "campaign",
            "--mode",
            "t49-hst-foundation",
            "--path",
            str(artifact),
            "--json",
        ],
        cwd=ROOT,
        env={"PYTHONPATH": f"{ROOT}:{ROOT / 'src'}"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    parsed = json.loads(proc.stdout)
    assert parsed["ok"] is True


def test_hst_manifest_rejects_tampered_byte_len():
    manifest_path = ROOT / "iterations" / "t49-hst-foundation-stage" / "HST_V1_REFERENCE_MANIFEST.json"
    doc = json.loads(manifest_path.read_text(encoding="utf-8"))

    tampered = json.loads(json.dumps(doc))
    tampered["merges"][0]["byte_len"] = 99
    ok_bad, probs = verify_hst_manifest_integrity(tampered)
    assert ok_bad is False
    assert any("byte_len mismatch" in msg for msg in probs)
    with pytest.raises(ValueError, match="byte_len mismatch"):
        HSTTokenizer.from_manifest(tampered)

    missing_blen = json.loads(json.dumps(doc))
    del missing_blen["merges"][0]["byte_len"]
    ok_miss, probs_miss = verify_hst_manifest_integrity(missing_blen)
    assert ok_miss is False
    assert any("byte_len mismatch" in msg for msg in probs_miss)
    with pytest.raises(ValueError, match="byte_len mismatch"):
        HSTTokenizer.from_manifest(missing_blen)

    for bad_blen in ("2", 2.9, True):
        bad_doc = json.loads(json.dumps(doc))
        bad_doc["merges"][0]["byte_len"] = bad_blen
        ok_b, probs_b = verify_hst_manifest_integrity(bad_doc)
        assert ok_b is False and any("byte_len mismatch" in msg for msg in probs_b)
        with pytest.raises(ValueError, match="byte_len mismatch"):
            HSTTokenizer.from_manifest(bad_doc)

    for bad_b64 in ("ZXI=!!!", 12345):
        bad_doc = json.loads(json.dumps(doc))
        bad_doc["merges"][0]["token_b64"] = bad_b64
        ok_b64, _ = verify_hst_manifest_integrity(bad_doc)
        assert ok_b64 is False
        with pytest.raises(ValueError):
            HSTTokenizer.from_manifest(bad_doc)


def test_t48_and_t49_campaign_handle_null_nested_sections(tmp_path: Path, capsys):
    t48_path = ROOT / "iterations" / "t48-roadmap-complete" / "T48_ROADMAP_COMPLETE_RESULTS.json"
    t48_doc = json.loads(t48_path.read_text(encoding="utf-8"))
    for key in ("phase_a_t38_live_scaling_audit", "phase_d_learned_neural_hypergraph_bridge"):
        bad = json.loads(json.dumps(t48_doc))
        bad[key] = None
        ok_flag, criteria, _ = verify_t48_roadmap_complete(bad)
        assert ok_flag is False
        assert any(not c.passed for c in criteria)

    bad_headline = json.loads(json.dumps(t48_doc))
    bad_headline["phase_a_t38_live_scaling_audit"]["headline_M_4ep"] = None
    ok_hl, _, _ = verify_t48_roadmap_complete(bad_headline)
    assert ok_hl is False

    t48_null_file = tmp_path / "t48_null.json"
    t48_null_file.write_text(json.dumps({**t48_doc, "phase_a_t38_live_scaling_audit": None}), encoding="utf-8")
    assert cli_main(["campaign", "--mode", "t48-roadmap-complete", "--path", str(t48_null_file), "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False

    t49_path = ROOT / "iterations" / "t49-hst-foundation-stage" / "T49_HST_FOUNDATION_RESULTS.json"
    t49_doc = json.loads(t49_path.read_text(encoding="utf-8"))
    for key in (
        "pillar_1_hst_v1_tokenization_standard",
        "pillar_2_equal_bytes_foundation_lm",
        "pillar_3_mup_zero_shot_width_transfer",
        "pillar_4_inline_receipt_gated_decoding",
    ):
        bad = json.loads(json.dumps(t49_doc))
        bad[key] = None
        ok_flag, criteria, _ = verify_t49_hst_foundation(bad)
        assert ok_flag is False
        assert any(not c.passed for c in criteria)

    t49_null_file = tmp_path / "t49_null.json"
    t49_null_file.write_text(
        json.dumps({**t49_doc, "pillar_1_hst_v1_tokenization_standard": None}), encoding="utf-8"
    )
    assert cli_main(["campaign", "--mode", "t49-hst-foundation", "--path", str(t49_null_file), "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False


def test_t48_and_t49_campaign_reject_top_level_non_dict(tmp_path: Path, capsys):
    for bad_root in (None, [], 42, "bad", True, False):
        ok48, crit48, adv48 = verify_t48_roadmap_complete(bad_root)  # type: ignore[arg-type]
        assert ok48 is False and adv48 == [] and len(crit48) == 1 and not crit48[0].passed
        ok49, crit49, adv49 = verify_t49_hst_foundation(bad_root)  # type: ignore[arg-type]
        assert ok49 is False and adv49 == [] and len(crit49) == 1 and not crit49[0].passed

    for idx, raw in enumerate(("null", "[]")):
        p = tmp_path / f"root_{idx}.json"
        p.write_text(raw, encoding="utf-8")
        for mode in ("t48-roadmap-complete", "t49-hst-foundation", "t49-hst-foundation-stage"):
            for flag in ("--path", "--in"):
                assert cli_main(["campaign", "--mode", mode, flag, str(p), "--json"]) == 1
                assert json.loads(capsys.readouterr().out)["ok"] is False


def test_t49_rejects_negative_p_value():
    artifact = ROOT / "iterations" / "t49-hst-foundation-stage" / "T49_HST_FOUNDATION_RESULTS.json"
    doc = json.loads(artifact.read_text(encoding="utf-8"))
    broken = json.loads(json.dumps(doc))
    broken["pillar_2_equal_bytes_foundation_lm"]["p_one_sided_hst_beats_bpe"] = -0.5
    ok_flag, criteria, _ = verify_t49_hst_foundation(broken)
    assert ok_flag is False
    assert any(c.id == "T49_EQUAL_BYTES_FOUNDATION_LM_WIN" and not c.passed for c in criteria)


def test_hst_manifest_rejects_negative_and_bool_child_ids():
    manifest_path = ROOT / "iterations" / "t49-hst-foundation-stage" / "HST_V1_REFERENCE_MANIFEST.json"
    doc = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key, bad_val in (
        ("left_id", -1),
        ("right_id", -1),
        ("left_id", True),
        ("right_id", False),
        ("left_id", 999_999),
    ):
        bad_doc = json.loads(json.dumps(doc))
        bad_doc["merges"][0][key] = bad_val
        ok_m, _ = verify_hst_manifest_integrity(bad_doc)
        assert ok_m is False
        with pytest.raises(ValueError):
            HSTTokenizer.from_manifest(bad_doc)


def test_hst_encode_with_empty_control_map():
    from sheaf_verify.hst_standard import HSTMergedRule

    id_to_bytes = [bytes([b]) for b in range(256)]
    rules: list[HSTMergedRule] = []
    for idx in range(8):
        tid = 256 + idx
        lid = ord("a") + idx
        rid = ord("b") + idx
        id_to_bytes.append(id_to_bytes[lid] + id_to_bytes[rid])
        rules.append(
            HSTMergedRule(
                token_id=tid,
                left_id=lid,
                right_id=rid,
                stage="A_lexical",
                pmi_bits=1.0,
                count=10,
                byte_len=2,
            )
        )
    tok_empty = HSTTokenizer(
        vocab_size=264,
        id_to_bytes=id_to_bytes,
        rules=rules,
        control_map={},
    )
    assert tok_empty.encode("hello ab 123") == tok_empty.encode_bytes(b"hello ab 123")
    manifest = tok_empty.export_manifest()
    reloaded = HSTTokenizer.from_manifest(manifest)
    assert reloaded.control_map == {}
    assert reloaded.encode("hello ab 123") == reloaded.encode_bytes(b"hello ab 123")


def test_cli_selftest_report_out_creates_parent_dir(tmp_path: Path, monkeypatch, capsys):
    import sheaf_verify.selftest as sv_selftest

    monkeypatch.setattr(sv_selftest, "run_full_verification", lambda *_a, **_kw: (True, {"ok": True, "stub": True}))
    out_path = tmp_path / "missing_parent" / "nested" / "selftest_report.json"
    assert not out_path.parent.exists()
    rc = cli_main(["selftest", "--report-out", str(out_path), "--json"])
    assert rc == 0
    assert out_path.is_file()
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved == {"ok": True, "stub": True}
    emitted = json.loads(capsys.readouterr().out)
    assert emitted == {"ok": True, "stub": True}


def test_hst_manifest_rejects_malformed_containers_and_vocab_size():
    manifest_path = ROOT / "iterations" / "t49-hst-foundation-stage" / "HST_V1_REFERENCE_MANIFEST.json"
    doc = json.loads(manifest_path.read_text(encoding="utf-8"))

    for bad_root in ("not_a_dict", ["list"], 123, None):
        with pytest.raises(ValueError):
            HSTTokenizer.from_manifest(bad_root)  # type: ignore[arg-type]

    for mutator in (
        lambda d: d.__setitem__("merges", "bad"),
        lambda d: d.__setitem__("merges", ["bad"]),
        lambda d: d.__setitem__("control_tokens", ["bad"]),
        lambda d: d.__setitem__("control_tokens", {"<|hst:bos|>": True}),
        lambda d: d.pop("vocab_size"),
        lambda d: d.__setitem__("vocab_size", True),
        lambda d: d.__setitem__("vocab_size", 200),
        lambda d: d["merges"][0].pop("stage"),
        lambda d: d["merges"][0].__setitem__("stage", "C_invalid"),
        lambda d: d["merges"][0].__setitem__("pmi_bits", "bad"),
        lambda d: d["merges"][0].__setitem__("count", True),
    ):
        bad_doc = json.loads(json.dumps(doc))
        mutator(bad_doc)
        with pytest.raises(ValueError):
            HSTTokenizer.from_manifest(bad_doc)


def test_t49_rejects_over_100_token_reduction_pct():
    artifact = ROOT / "iterations" / "t49-hst-foundation-stage" / "T49_HST_FOUNDATION_RESULTS.json"
    doc = json.loads(artifact.read_text(encoding="utf-8"))
    broken = json.loads(json.dumps(doc))
    broken["pillar_1_hst_v1_tokenization_standard"]["token_sequence_reduction_pct"] = 150.0
    ok_flag, criteria, _ = verify_t49_hst_foundation(broken)
    assert ok_flag is False
    assert any(c.id == "T49_HST_V1_STANDARD_AND_COMPRESSION" and not c.passed for c in criteria)


def test_hst_manifest_rejects_tampered_merge_count():
    manifest_path = ROOT / "iterations" / "t49-hst-foundation-stage" / "HST_V1_REFERENCE_MANIFEST.json"
    doc = json.loads(manifest_path.read_text(encoding="utf-8"))
    tampered = json.loads(json.dumps(doc))
    tampered["merges"][0]["count"] = int(tampered["merges"][0]["count"]) + 1
    ok_bad, probs = verify_hst_manifest_integrity(tampered)
    assert ok_bad is False
    assert any("Merkle root mismatch" in msg for msg in probs)
    with pytest.raises(ValueError, match="Merkle root mismatch"):
        HSTTokenizer.from_manifest(tampered)
