"""Unit, statistical audit, learned action bridge, and CLI tests for T-48 complete roadmap (v1.0.0)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from hadukenn_core.learned_action_bridge import (
    ENTITIES,
    FlatNeuralSequenceBaseline,
    LearnedSheafHypergraphBridge,
    QuantizedLoopedBridgeNPU,
    execute_predicted_episode,
    generate_relational_episodes,
    run_t48_roadmap_campaign,
)
from hadukenn_core.tokenization import ByteBPETokenizer
from sheaf_verify.campaign import verify_t48_roadmap_complete
from sheaf_verify.t38_scaling_audit import (
    RECORDED_T38_CELLS,
    bootstrap_scaling_exponent,
    compute_cell_contrast,
    holm_bonferroni_gate,
    run_t38_live_audit,
    spearman_rank_correlation,
)

ROOT = Path(__file__).resolve().parents[1]


def test_t38_live_audit_matches_recorded_verdicts() -> None:
    audit = run_t38_live_audit(n_boot=120)
    assert audit["scitt_verified"] is True
    assert audit["headline_M_4ep"]["verdict"] == "KILL"
    assert abs(audit["headline_M_4ep"]["delta_h_minus_p"] - 0.01877957) < 1e-6
    assert audit["holm_bonferroni_alpha_0p01"]["cells_killing_H"] == 8
    assert audit["spearman_rho_4ep_vs_N"] == -1.0
    assert audit["stage3_stop_fired"] is False
    assert audit["scaling_fit_n_body"]["pass_scaling_in_favor_of_H"] is False
    assert audit["stage3_L_diagnostic"]["mup_width_drift_detected"] is True
    assert audit["stage3_L_diagnostic"]["p_best_full_val_lr"] == "0.0005"


def test_spearman_ties_and_holm_stopping_branches() -> None:
    # Exact tie handling and constant-vector edge branch
    rho_tie = spearman_rank_correlation([1.0, 2.0, 3.0, 4.0], [0.032, 0.032, 0.019, 0.025])
    assert -1.0 <= rho_tie <= 1.0
    assert spearman_rank_correlation([1.0, 2.0, 3.0], [5.0, 5.0, 5.0]) == 0.0

    # Synthetic inconclusive cell stops Holm cascade
    weak_cell = dict(RECORDED_T38_CELLS["XS_1ep"])
    weak_cell["h_test"] = [x + 0.0001 for x in weak_cell["p_test"]]
    c_weak = compute_cell_contrast("WEAK", weak_cell)
    assert c_weak.verdict == "INCONCLUSIVE"
    holm = holm_bonferroni_gate([c_weak], alpha=0.01)
    assert holm["cells_killing_H"] == 0

    fit = bootstrap_scaling_exponent(n_boot=60, param_key="n_body")
    assert fit["ci95_low"] < 0.0


def test_learned_bridge_single_seed_and_campaign_verifier() -> None:
    import numpy as np

    rng = np.random.default_rng(21)
    tr = generate_relational_episodes(rng, n_episodes=24, ood_long_horizon=False)
    ood = generate_relational_episodes(rng, n_episodes=9, ood_long_horizon=True)
    tok = ByteBPETokenizer.train([s.text for e in tr for s in e.steps], vocab_size=280, min_pair_count=2)
    bridge = LearnedSheafHypergraphBridge(tok.vocab_size, d_model=64, seed=21)
    loss = bridge.train_policy(tok, tr, epochs=95, lr=0.045)
    assert loss < 0.25
    npu = QuantizedLoopedBridgeNPU(bridge)
    assert npu.arena_bytes() < 200_000
    ok_state, ok_spoof, ok_replay = execute_predicted_episode(npu, tok, ood[0])
    assert isinstance(ok_state, bool)
    assert ok_spoof is True
    assert ok_replay is True
    flat = FlatNeuralSequenceBaseline(tok.vocab_size, d_model=64, seed=21)
    f_acc = flat.fit_and_score(tok, tr, ood)
    assert 0.0 <= f_acc <= 1.0
    assert len(ENTITIES) == 4


def test_sealed_t48_artifact_and_cli_mode() -> None:
    artifact = ROOT / "iterations" / "t48-roadmap-complete" / "T48_ROADMAP_COMPLETE_RESULTS.json"
    assert artifact.is_file()
    doc = json.loads(artifact.read_text(encoding="utf-8"))
    ok_flag, criteria, advisories = verify_t48_roadmap_complete(doc)
    assert ok_flag is True
    assert all(c.passed for c in criteria)
    assert any("ROADMAP_COMPLETE_VERIFIED" in a for a in advisories)

    # Corrupt copy must fail verification cleanly
    bad_doc = dict(doc)
    bad_doc["scitt_verified"] = False
    bad_ok, _, _ = verify_t48_roadmap_complete(bad_doc)
    assert bad_ok is False

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "sheaf_verify",
            "campaign",
            "--mode",
            "t48-roadmap-complete",
            "--path",
            str(artifact),
            "--json",
        ],
        cwd=ROOT,
        env={"PYTHONPATH": f"{ROOT}:{ROOT / 'src'}"},
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    assert out["ok"] is True

    # Quick smoke of campaign runner on 1 seed
    smoke = run_t48_roadmap_campaign(seeds=(13,))
    assert smoke["scitt_verified"] is True


def test_t48_rejects_negative_disagreement() -> None:
    artifact = ROOT / "iterations" / "t48-roadmap-complete" / "T48_ROADMAP_COMPLETE_RESULTS.json"
    doc = json.loads(artifact.read_text(encoding="utf-8"))
    bad_doc = json.loads(json.dumps(doc))
    bad_doc["phase_d_learned_neural_hypergraph_bridge"]["int8_vs_fp32_action_disagreement_mean"] = -0.5
    ok_flag, criteria, _ = verify_t48_roadmap_complete(bad_doc)
    assert ok_flag is False
    assert any(c.id == "T48_LEARNED_BRIDGE_AND_LOOPED_INT8_NPU" and not c.passed for c in criteria)


def test_t48_rejects_ood_accuracy_above_one() -> None:
    artifact = ROOT / "iterations" / "t48-roadmap-complete" / "T48_ROADMAP_COMPLETE_RESULTS.json"
    doc = json.loads(artifact.read_text(encoding="utf-8"))
    bad_doc = json.loads(json.dumps(doc))
    bad_doc["phase_d_learned_neural_hypergraph_bridge"]["int8_looped_npu_ood_accuracy_mean"] = 1.5
    ok_flag, criteria, _ = verify_t48_roadmap_complete(bad_doc)
    assert ok_flag is False
    assert any(c.id == "T48_LEARNED_BRIDGE_AND_LOOPED_INT8_NPU" and not c.passed for c in criteria)
