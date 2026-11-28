"""Unit, statistical, CLI, and Antigravity handoff tests for v0.8.0 (T-45 Sheaf-Bridge & selftest)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hadukenn_core.sheaf_bridge import (
    build_sheaf_bridge_operator,
    compute_kruskal_mst_laplacian,
    compute_lle_barycentric_weights,
    quantize_dequantize_symmetric_int8,
    run_t45_sheaf_bridge_suite,
    whitened_spectral_warmstart,
)
from sheaf_verify.campaign import verify_t45_sheaf_bridge
from sheaf_verify.cli import main as cli_main
from sheaf_verify.selftest import audit_python_style_stdlib

ROOT = Path(__file__).resolve().parents[1]


def test_whitened_warmstart_and_lle_mst_invariants():
    rng = np.random.default_rng(13)
    oracle = rng.normal(0.0, 1.0, size=(32, 64))
    w_init, isotropy = whitened_spectral_warmstart(oracle, d_student=8)
    assert w_init.shape == (32, 8)
    assert 0.25 <= isotropy <= 1.0

    with pytest.raises(ValueError, match="d_student"):
        whitened_spectral_warmstart(oracle, d_student=1)

    w_lle = compute_lle_barycentric_weights(oracle, k=4)
    assert w_lle.shape == (32, 32)
    row_sums = np.sum(w_lle, axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-9)
    assert np.allclose(np.diag(w_lle), 0.0)

    l_mst, edges = compute_kruskal_mst_laplacian(oracle)
    assert len(edges) == 31
    assert np.allclose(l_mst, l_mst.T)
    assert np.allclose(np.sum(l_mst, axis=1), 0.0, atol=1e-9)

    l_bridge, stats = build_sheaf_bridge_operator(oracle, k=4)
    assert l_bridge.shape == (32, 32)
    assert np.allclose(l_bridge, l_bridge.T, atol=1e-9)
    assert stats["mst_edges"] == 31
    assert stats["lle_nnz"] == 32 * 4

    q_w = quantize_dequantize_symmetric_int8(w_init)
    assert q_w.shape == w_init.shape
    assert float(np.max(np.abs(q_w - w_init))) < 0.02


def test_t45_sheaf_bridge_suite_and_campaign_cli(tmp_path: Path):
    results, bundle = run_t45_sheaf_bridge_suite()
    ok_flag, criteria, advisories = verify_t45_sheaf_bridge(results)
    assert ok_flag is True, [c.as_dict() for c in criteria]
    assert len(criteria) == 4
    assert len(advisories) >= 1
    assert bundle["bundle_schema"] == "sheaf-verify.regulatory-bundle.v1"

    tampered = dict(results)
    tampered["dense_flat_head_preserved"] = False
    bad_ok, bad_criteria, _ = verify_t45_sheaf_bridge(tampered)
    assert bad_ok is False
    assert any(not c.passed and c.id == "T45_SCHEMA_AND_DENSE_HEAD" for c in bad_criteria)

    res_path = tmp_path / "t45.json"
    res_path.write_text(json.dumps(results), encoding="utf-8")
    rc = cli_main(["campaign", "--mode", "t45-sheaf-bridge", "--path", str(res_path), "--json"])
    assert rc == 0


def test_antigravity_handoff_layout_and_stdlib_style():
    required_paths = [
        ROOT / "AGENTS.md",
        ROOT / "ANTIGRAVITY_HANDOFF.md",
        ROOT / ".agents" / "rules" / "00-repository-invariants.md",
        ROOT / ".agents" / "rules" / "10-sheaf-verify-stdlib.md",
        ROOT / ".agents" / "rules" / "20-empirical-architecture-guardrails.md",
        ROOT / ".agents" / "skills" / "verify-all-gates" / "SKILL.md",
        ROOT / ".agents" / "skills" / "verify-all-gates" / "scripts" / "verify_all.py",
        ROOT / ".agents" / "skills" / "run-sheaf-bridge" / "SKILL.md",
        ROOT / ".agents" / "skills" / "run-sheaf-bridge" / "scripts" / "run_bridge_benchmark.py",
        ROOT / ".agents" / "skills" / "prepare-iteration-pr" / "SKILL.md",
        ROOT / ".agents" / "skills" / "prepare-iteration-pr" / "scripts" / "check_pr_hygiene.py",
        ROOT / ".agents" / "agents" / "gate-auditor" / "agent.md",
        ROOT / ".agents" / "agents" / "sheaf-bridge-builder" / "agent.md",
    ]
    for p in required_paths:
        assert p.is_file(), f"Missing Antigravity handoff file: {p}"

    for skill_md in (ROOT / ".agents" / "skills").glob("*/SKILL.md"):
        content = skill_md.read_text(encoding="utf-8")
        assert content.startswith("---\n"), skill_md
        assert "## Safety" in content, f"{skill_md} missing mandatory ## Safety section"
        assert "Do not use when" in content, f"{skill_md} missing boundary clause"

    issues = audit_python_style_stdlib(ROOT)
    assert issues == [], issues
