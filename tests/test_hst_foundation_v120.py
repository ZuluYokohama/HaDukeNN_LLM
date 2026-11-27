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
from sheaf_verify.campaign import verify_t49_hst_foundation
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
