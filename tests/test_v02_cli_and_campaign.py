"""Tests for v0.2.0 campaign verifiers and CLI subcommands on committed iteration records."""

from __future__ import annotations

import json
from pathlib import Path

from sheaf_verify.campaign import (
    audit_cluster_geometry,
    audit_t38_prereg,
    classify_domain_of_calibration,
    head_flop_scaling_table,
    verify_hg_v3_gate,
    verify_t37_grid,
)
from sheaf_verify.cli import main
from sheaf_verify.synthetic import TEST_KEY, build

ROOT = Path(__file__).resolve().parents[1]


def test_cluster_geometry_256x128_beats_182x181_and_128x256():
    opt = audit_cluster_geometry(vocab_size=32768, nc=256, cluster_size=128, d_model=576, k_neg=3, struct_rank=0)
    assert opt.recommended is True
    assert opt.dead_pad_slots == 0
    assert opt.tensor_core_128_aligned is True
    assert opt.passes_p1_size_gate is True
    assert opt.redundant_unit_table_params == 0
    assert abs(opt.head_mflops_per_tok_kneg - 2.654208) < 1e-5
    assert abs(opt.gather_mib_bf16 - 2304.0) < 1e-6

    pad_trap = audit_cluster_geometry(vocab_size=32768, nc=182, cluster_size=181, d_model=576, k_neg=3, struct_rank=32, resid="full")
    assert pad_trap.recommended is False
    assert pad_trap.dead_pad_slots == 174
    assert pad_trap.tensor_core_64_aligned is False
    assert pad_trap.redundant_unit_table_params == 32 * (32768 + 576) + 32768 * 576

    inv_trap = audit_cluster_geometry(vocab_size=32768, nc=128, cluster_size=256, d_model=576, k_neg=3, struct_rank=0)
    assert inv_trap.recommended is False
    assert inv_trap.passes_p1_size_gate is False


def test_domain_of_calibration_transitions():
    base = {"budget_kind": "bytes", "budget_value": 348328219.0, "n_seeds": 3, "seq_len": 256, "validated_scales": 1}
    prog = {
        "budget_kind": "bytes",
        "budget_value": 348328219.0,
        "n_seeds": 5,
        "prereg_frozen_before_data": True,
        "removes_tokenizer_confound": True,
        "seq_len": 256,
        "sm_arch": "sm_120",
        "validated_scales": 4,
    }
    assert classify_domain_of_calibration(base, prog).classification == "PROGRESSIVE"

    stag = {**base, "prereg_frozen_before_data": True, "sm_arch": "sm_120"}
    assert classify_domain_of_calibration(base, stag).classification == "STAGNANT"

    imp = {
        "budget_kind": "wallclock",
        "budget_value": 400000000.0,
        "prereg_frozen_before_data": False,
        "seq_len": 1024,
        "sm_arch": "sm_75",
    }
    res_imp = classify_domain_of_calibration(base, imp)
    assert res_imp.classification == "IMPROPER"
    assert len(res_imp.reasons) >= 4


def test_real_t37_t38_and_hgv3_campaign_records():
    t37_doc = json.loads((ROOT / "iterations" / "t37-grid" / "results.json").read_text(encoding="utf-8"))
    ok_37, _crit_37, summary_37 = verify_t37_grid(t37_doc)
    assert ok_37 is True
    assert len(summary_37["holm_survivors"]) == 5

    t38_doc = json.loads((ROOT / "iterations" / "t38-design" / "T38_PREREG.redacted.json").read_text(encoding="utf-8"))
    t38_add = json.loads((ROOT / "iterations" / "t38-design" / "T38_PREREG_ADDENDUM_PAIRED.json").read_text(encoding="utf-8"))
    ok_38, _crit_38, advisories_38 = audit_t38_prereg(t38_doc, t38_add)
    assert ok_38 is True
    adv_text = "\n".join(advisories_38)
    assert "GEOMETRY_PAD_TRAP" in adv_text
    assert "PARAM_REDUNDANCY" in adv_text
    assert "STAGE3_FALSE_DROP_BUG" in adv_text

    hg_doc = json.loads((ROOT / "iterations" / "t39-frontier-blueprint" / "hg_v3_gate_verified.json").read_text(encoding="utf-8"))
    ok_hg, _crit_hg = verify_hg_v3_gate(hg_doc)
    assert ok_hg is True

    table = head_flop_scaling_table()
    assert len(table) == 6
    assert table[0]["flat_head_flop_share_pct"] > 37.0
    assert table[2]["flat_logits_bf16_gib"] == 4.0


def test_v02_cli_commands_end_to_end(tmp_path: Path, capsys):
    store = build(tmp_path / "store")
    key_file = tmp_path / "seal.key"
    key_file.write_bytes(TEST_KEY)

    # 1. attest & verify-attestation
    att_out = tmp_path / "attestation.dsse.json"
    assert main(["attest", "--store", str(store.root), "--seal-key-file", str(key_file), "--out", str(att_out), "--json"]) == 0
    capsys.readouterr()
    assert att_out.is_file()
    assert main(["verify-attestation", "--seal-key-file", str(key_file), "--envelope", str(att_out), "--json"]) == 0
    att_verified = json.loads(capsys.readouterr().out)
    assert att_verified["ok"] is True

    # 2. receipt & verify-receipt
    rec_out = tmp_path / "gate_g4.receipt.json"
    assert main(["receipt", "--store", str(store.root), "--seal-key-file", str(key_file), "--seq", "13", "--out", str(rec_out), "--json"]) == 0
    capsys.readouterr()
    assert main(["verify-receipt", "--seal-key-file", str(key_file), "--receipt", str(rec_out), "--json"]) == 0
    rec_verified = json.loads(capsys.readouterr().out)
    assert rec_verified["ok"] is True

    # 3. audit-kernel
    kern = tmp_path / "kernel.py"
    kern.write_text("def clean_apply(x):\n    return x * 2\n", encoding="utf-8")
    assert main(["audit-kernel", "--path", str(kern), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True

    # 4. sheaf-check
    sheaf_json = tmp_path / "sheaf.json"
    sheaf_json.write_text(json.dumps({
        "anchors": {"v0": [1.0]},
        "assignment": {"v0": [1.0], "v1": [1.0]},
        "edges": [{"edge_dim": 1, "id": "e0", "source": "v0", "source_map": [[1.0]], "target": "v1", "target_map": [[1.0]]}],
        "vertex_dims": {"v0": 1, "v1": 1},
    }), encoding="utf-8")
    assert main(["sheaf-check", "--path", str(sheaf_json), "--json"]) == 0
    s_out = json.loads(capsys.readouterr().out)
    assert s_out["dim_h0"] == 1
    assert s_out["contradiction_regime"] == "CONSISTENT"

    # 5. campaign modes
    assert main(["campaign", "--mode", "cluster-geometry", "--nc", "256", "--cluster-size", "128", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["recommended"] is True

    assert main(["campaign", "--mode", "flop-scaling", "--json"]) == 0
    assert len(json.loads(capsys.readouterr().out)["rows"]) == 6

    assert main(["campaign", "--mode", "t37-grid", "--path", str(ROOT / "iterations/t37-grid/results.json"), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True

    assert main([
        "campaign",
        "--mode", "t38-prereg",
        "--path", str(ROOT / "iterations/t38-design/T38_PREREG.redacted.json"),
        "--addendum", str(ROOT / "iterations/t38-design/T38_PREREG_ADDENDUM_PAIRED.json"),
        "--json",
    ]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True

    assert main([
        "campaign",
        "--mode", "hg-v3",
        "--path", str(ROOT / "iterations/t39-frontier-blueprint/hg_v3_gate_verified.json"),
        "--json",
    ]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True

    # 6. mutation-score
    assert main(["mutation-score", "--json"]) == 0
    m_out = json.loads(capsys.readouterr().out)
    assert m_out["mutation_score"] == 1.0
    assert m_out["survived"] == 0
