"""Unit, gradient, cryptographic receipt, and CLI tests for `v0.4.0` Frontier-5070 (`T-41`)."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from hadukenn_core.rtx5070_frontier import (
    Blockwise8BitAdamW,
    NF4BlockQuantizer,
    factorize_pretrained_lm_head,
    plan_rtx5070_frontier_vram,
)
from hadukenn_core.transformer_lm import CausalTransformerLM, TransformerConfig
from sheaf_verify.anticheat import audit_kernel_source
from sheaf_verify.campaign import verify_t41_frontier5070
from sheaf_verify.cli import main as cli_main

ROOT = Path(__file__).resolve().parents[1]


def test_rtx5070_vram_planner_and_quantizers():
    plans = plan_rtx5070_frontier_vram(head_chunk_tokens=256)
    assert len(plans) == 6
    for p in plans:
        assert p.fits_5070_12gb is True
        assert p.frontier5070_total_peak_gib <= 12.0
        assert p.frontier5070_l2_hypergraph_mib <= 48.0
        assert p.vram_headroom_gib >= 3.0
    p_8b = next(p for p in plans if p.vocab_size == 262144)
    assert p_8b.naive_3d_gather_trap_gib == 16.0
    assert p_8b.naive_fits_5070 is False
    assert p_8b.frontier5070_head_buffer_gib < 1.1

    rng = np.random.default_rng(17)
    w = (rng.standard_normal((37, 53)) * 0.15).astype(np.float32)
    w[0, 0] = 0.0
    nf4 = NF4BlockQuantizer(block_size=64)
    qs = nf4.quantize(w)
    assert 4.0 <= float(qs["bits_per_param"]) <= 4.35
    w_rec = nf4.dequantize(qs)
    assert w_rec.shape == w.shape
    assert w_rec[0, 0] == 0.0
    assert float(np.sqrt(np.mean((w - w_rec) ** 2))) < 0.03

    params = {"W": (rng.standard_normal((32, 32)) * 0.1).astype(np.float32), "frozen_W": np.ones((16, 16), dtype=np.float32)}
    opt = Blockwise8BitAdamW(params, base_width_mup=32, d_model=32, block_size=128)
    fp32_moments = sum(v.nbytes * 2 for v in params.values())
    assert fp32_moments / opt.state_bytes() > 3.5
    grads = {"W": np.ones_like(params["W"]) * 0.05, "frozen_W": np.ones_like(params["frozen_W"]) * 0.5}
    opt.step(params, grads, lr=1e-2, frozen_names={"frozen_W"})
    assert np.all(params["frozen_W"] == 1.0)
    assert not np.allclose(params["W"], 0.0)

    emb = (rng.standard_normal((128, 16)) * 0.08).astype(np.float32)
    part, d_rt = factorize_pretrained_lm_head(emb, 16, 8, seed=13)
    assert part.verify_bijection()["bijection_ok"] is True
    assert d_rt.shape == (16, 16)
    try:
        factorize_pretrained_lm_head(emb, 15, 8)
        raise AssertionError("Expected ValueError on NC*S != V")
    except ValueError:
        pass


def test_rope_decoupled_sheaf_finite_diff_and_torch_anticheat():
    rng = np.random.default_rng(42)
    cfg = TransformerConfig(
        vocab_size=24,
        d_model=8,
        n_heads=2,
        n_layers=2,
        d_ff=16,
        window_len=6,
        kv_prefix_slots=3,
        d_stalk=16,
        mem_score_bias=-2.5,
    )
    mod = CausalTransformerLM(cfg, seed=7)
    xb = rng.integers(0, 24, size=(2, 6), dtype=np.int32)
    yb = rng.integers(0, 24, size=(2, 6), dtype=np.int32)
    ps = rng.standard_normal((2, 3, 16)).astype(np.float32)
    _, grads = mod.forward_backward(xb, yb, prefix_stalks=ps, training=True)
    h_enc = mod.encode_hidden(xb, prefix_stalks=ps)
    assert h_enc.shape == (12, 8)

    eps = 1e-3
    for pname in ("L0.Wq", "L0.W_mk", "L0.W_mv", "L1.alpha_kv"):
        p = mod.params[pname].reshape(-1)
        g = grads[pname].reshape(-1)
        orig = float(p[0])
        p[0] = orig + eps
        lp, _ = mod.forward_backward(xb, yb, prefix_stalks=ps, training=False)
        p[0] = orig - eps
        lm, _ = mod.forward_backward(xb, yb, prefix_stalks=ps, training=False)
        p[0] = orig
        assert abs((lp - lm) / (2 * eps) - float(g[0])) < 2e-3

    rep = audit_kernel_source((ROOT / "src" / "hadukenn_core" / "torch_bridge.py").read_text(encoding="utf-8"))
    assert rep.ok is True and rep.violations == ()


def test_t41_frontier5070_cli_attestation_and_negative_gates(tmp_path: Path):
    t41_dir = ROOT / "iterations" / "t41-frontier-5070"
    res_path = t41_dir / "T41_FRONTIER5070_RESULTS.json"
    att_path = t41_dir / "T41_DSSE_ATTESTATION.json"
    rec_path = t41_dir / "T41_SCITT_RECEIPT.json"
    key_path = ROOT / "fixtures" / "test_seal.key"
    assert res_path.is_file() and att_path.is_file() and rec_path.is_file()

    rc = cli_main(["campaign", "--mode", "t41-frontier5070", "--path", str(res_path), "--json"])
    assert rc == 0

    rc_att = cli_main(["verify-attestation", "--seal-key-file", str(key_path), "--envelope", str(att_path), "--json"])
    assert rc_att == 0
    rc_rec = cli_main(["verify-receipt", "--seal-key-file", str(key_path), "--receipt", str(rec_path), "--json"])
    assert rc_rec == 0

    doc = json.loads(res_path.read_text(encoding="utf-8"))
    bad_doc = copy.deepcopy(doc)
    bad_doc["part3_learned_stalk_lpft_g2"]["bipartite_self_inclusion_drift"] = 0.15
    ok_flag, criteria, _ = verify_t41_frontier5070(bad_doc)
    assert ok_flag is False
    assert any(c.id == "T41_G2_SHEAF_BEATS_CHODROW_NULL" and not c.passed for c in criteria)

    bad_path = tmp_path / "bad_t41.json"
    bad_path.write_text(json.dumps(bad_doc), encoding="utf-8")
    assert cli_main(["campaign", "--mode", "t41-frontier5070", "--path", str(bad_path), "--json"]) == 1

    proc = subprocess.run(
        [sys.executable, "-m", "sheaf_verify", "campaign", "--mode", "t41-frontier5070", "--path", str(res_path), "--json"],
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT / "src")},
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    parsed = json.loads(proc.stdout)
    assert parsed["ok"] is True
