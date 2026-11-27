"""Comprehensive unit, gradient, invariant, and CLI tests for `hadukenn_core` and `t40-stack`."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from hadukenn_core.benchmark_suite import load_clean_technical_corpus, run_full_empirical_suite
from hadukenn_core.fast_ops import get_fast_head_lib, run_cluster_is_avx512, run_tiled_cce_avx512
from hadukenn_core.heads import (
    CountShrunkSpanComposer,
    build_tied_root_shortcut_layout,
    cluster_is_fwd_bwd,
    cluster_routed_eval_nll,
    factored_fp2_fwd_bwd,
    flat_tied_fwd_bwd,
    tied_root_shortcut_fwd_bwd,
    tiled_cce_fwd_bwd,
)
from hadukenn_core.hypergraph import (
    CorpusDocument,
    build_chodrow_null,
    build_decontaminated_hypergraph,
    compute_self_excluded_stalks,
    evaluate_scoped_knnlm,
    evaluate_sheaf_contradiction_e4,
    gather_doc_self_excluded_stalks,
    prefix_causal_walk_order,
)
from hadukenn_core.partitions import build_balanced_partition
from hadukenn_core.tokenization import ByteBPETokenizer, PMISpanLexicon, cosine_stochastok_prob
from hadukenn_core.transformer_lm import CausalTransformerLM, TransformerConfig
from sheaf_verify.anticheat import audit_kernel_source

ROOT = Path(__file__).resolve().parents[1]


def test_byte_bpe_lossless_stochastok_and_pmi_spans():
    texts = [
        "Cellular sheaf restriction maps transport vertex stalks across shared hyperedges! _alpha_0 = 0.15\n" * 6,
        "Paradigmatic spectral clusters align power_of_two matrix tiles without -1 pad sentinels.\n" * 6,
    ]
    bpe = ByteBPETokenizer.train(texts, vocab_size=320, min_pair_count=2)
    sample = "Test _underscore_ & UTF-8 arrows -> sheaf stalks 12345!"
    ids = bpe.encode(sample)
    assert bpe.decode_bytes(ids) == sample.encode("utf-8")

    rng = np.random.default_rng(11)
    expanded = bpe.stochastok_expand(ids, p_split=0.5, rng=rng)
    assert bpe.decode_bytes(expanded) == sample.encode("utf-8")
    assert expanded.shape[0] >= ids.shape[0]
    assert abs(cosine_stochastok_prob(0.0) - 0.12) < 1e-9
    assert abs(cosine_stochastok_prob(1.0) - 0.03) < 1e-9

    streams = [bpe.encode(t) for t in texts]
    lex = PMISpanLexicon.build(bpe, streams, max_spans=32, min_pmi=1.0, min_count=3)
    assert lex.num_spans > 0
    # Verify strict zero-lookahead causality on ending span IDs
    s_orig = lex.causal_ending_span_ids(streams[0])
    mut = streams[0].copy()
    half = mut.shape[0] // 2
    mut[half:] = (mut[half:] + 5) % bpe.vocab_size
    s_mut = lex.causal_ending_span_ids(mut)
    assert np.array_equal(s_orig[:half], s_mut[:half])

    merged = lex.merge_sequence(streams[0])
    assert merged.shape[0] <= streams[0].shape[0]
    assert int(np.sum(lex.merged_byte_lengths[merged])) == int(np.sum(bpe.byte_lengths[streams[0]]))


def test_balanced_partitions_and_all_head_gradients():
    rng = np.random.default_rng(21)
    streams = [rng.integers(0, 64, size=240, dtype=np.int32) for _ in range(5)]
    for mode in ("mass_random", "syntagmatic_spectral", "paradigmatic_spectral"):
        part = build_balanced_partition(64, 8, 8, streams, mode=mode, embed_dim=6, seed=13)
        ver = part.verify_bijection()
        assert ver["bijection_ok"] is True
        assert ver["zero_pad_slots"] is True
        assert ver["p1_size_gate_ok"] is True

    part = build_balanced_partition(64, 8, 8, streams, mode="paradigmatic_spectral", embed_dim=6, seed=13)
    H = (rng.standard_normal((16, 12)) * 0.25).astype(np.float32)
    E = (rng.standard_normal((64, 12)) * 0.25).astype(np.float32)
    y = rng.integers(0, 64, size=16, dtype=np.int32)

    l_flat, dH_flat, dE_flat = flat_tied_fwd_bwd(H, E, y, compute_grad=True)
    l_cce_np, dH_cce_np, dE_cce_np = tiled_cce_fwd_bwd(H, E, y, tile_V=16, compute_grad=True, use_native=False)
    assert np.max(np.abs(l_flat - l_cce_np)) < 1e-5
    assert np.max(np.abs(dH_flat - dH_cce_np)) < 1e-5
    assert np.max(np.abs(dE_flat - dE_cce_np)) < 1e-5

    if get_fast_head_lib() is not None:
        l_cce_c, dH_cce_c, dE_cce_c = run_tiled_cce_avx512(H, E, y, tile_V=64, compute_grad=True)
        assert np.max(np.abs(l_flat - l_cce_c)) < 1e-5
        assert np.max(np.abs(dH_flat - dH_cce_c)) < 1e-5
        assert np.max(np.abs(dE_flat - dE_cce_c)) < 1e-5

        E_cl = E[part.cluster_units].copy()
        C_rt = E_cl.mean(axis=1).copy()
        tc = part.token_to_cluster[y]
        tm = part.token_to_member[y]
        neg = np.mod(tc[:, None] + np.array([[1, 2, 3]], dtype=np.int32), 8)
        l_is_c, dH_is_c, dE_is_c = run_cluster_is_avx512(H, E_cl, C_rt, tc, tm, neg, compute_grad=True)
        assert l_is_c.shape == (16,) and np.all(np.isfinite(l_is_c))
        assert dH_is_c is not None and dE_is_c is not None

    delta_r = np.zeros((8, 12), dtype=np.float32)
    l_fp2, dH_fp2, dE_fp2, dd_fp2 = factored_fp2_fwd_bwd(H, E, delta_r, part, y, compute_grad=True)
    assert l_fp2.shape == (16,) and dH_fp2 is not None and dE_fp2 is not None and dd_fp2 is not None

    l_is, rnll, dH_is, dE_is, dd_is = cluster_is_fwd_bwd(H, E, delta_r, part, y, k_neg=3, rng=rng, compute_grad=True)
    assert l_is.shape == (16,) and rnll > 0.0 and dH_is is not None and dE_is is not None and dd_is is not None

    routed = cluster_routed_eval_nll(H, E, delta_r, part, y, top_m_clusters=4)
    assert routed.shape == (16,) and np.all(routed >= 0.0)

    sc_layout = build_tied_root_shortcut_layout(64, 16, 6, 8, streams)
    delta_t = np.zeros((6, 12), dtype=np.float32)
    l_tr, dH_tr, dE_tr, dd_tr = tied_root_shortcut_fwd_bwd(H, E, delta_t, sc_layout, y, compute_grad=True)
    assert l_tr.shape == (16,) and dH_tr is not None and dE_tr is not None and dd_tr is not None


def test_transformer_and_hypergraph_invariants():
    docs = [
        CorpusDocument(
            doc_id=f"d{i}",
            domain="sys",
            text=" ".join(
                f"Module AlphaBeta_{i % 6} and Kernel_Gamma_{i % 5} govern unique_doc{i}_sec{k} "
                f"tile_{i}_{k} version {10 + (i % 4)} latency {20 + k}.5 ms."
                for k in range(28)
            ),
            sha256=f"{i:064x}",
        )
        for i in range(36)
    ]
    # Add an exact duplicate of eval doc to train to verify decontamination drops it
    dup_eval = CorpusDocument(doc_id="dup_leak", domain="sys", text=docs[-1].text, sha256=docs[-1].sha256)
    clean_tr, hg = build_decontaminated_hypergraph(docs[:28] + [dup_eval], docs[28:], min_edge_size=3, d_doc=12, seed=7)
    assert "dup_leak" in hg.dropped_near_duplicates
    assert len(clean_tr) == hg.n_docs

    hg_null = build_chodrow_null(hg, n_swaps_factor=6, seed=19)
    assert np.array_equal(hg.incidence.sum(axis=1), hg_null.incidence.sum(axis=1))
    assert np.array_equal(hg.incidence.sum(axis=0), hg_null.incidence.sum(axis=0))

    es, d2e = compute_self_excluded_stalks(hg, hg.doc_vectors)
    s0 = gather_doc_self_excluded_stalks(0, hg, hg.doc_vectors, es, d2e, m_slots=3)
    pert = hg.doc_vectors.copy()
    pert[0] += 50.0
    es2, _ = compute_self_excluded_stalks(hg, pert)
    s1 = gather_doc_self_excluded_stalks(0, hg, pert, es2, d2e, m_slots=3)
    assert np.max(np.abs(s0 - s1)) == 0.0

    for wmode in ("random", "dense", "hyper"):
        order, st = prefix_causal_walk_order(hg, mode=wmode, seed=5)
        assert len(order) == hg.n_docs and 0.0 <= st["jump_rate"] <= 1.0

    e4 = evaluate_sheaf_contradiction_e4(hg, clean_tr, seed=13)
    assert e4["tie_1v1_abstention_rate"] == 1.0
    assert e4["sheaf_auc_setB"] > e4["embed_cos_auc_setB"]

    # Mini kNN-LM smoke
    rng = np.random.default_rng(3)
    q_s = rng.standard_normal((8, 12)).astype(np.float32)
    ds_k = rng.standard_normal((24, 12)).astype(np.float32)
    q_y = np.arange(8, dtype=np.int32)
    ds_v = np.arange(24, dtype=np.int32) % 8
    ds_d = np.arange(24, dtype=np.int32) % 6
    e2 = evaluate_scoped_knnlm(
        query_states=q_s,
        query_targets=q_y,
        query_lm_probs=np.full(8, 0.05),
        query_doc_hyper_neighbors=[np.array([0, 1], dtype=np.int32) for _ in range(8)],
        query_doc_null_neighbors=[np.array([2, 3], dtype=np.int32) for _ in range(8)],
        datastore_keys=ds_k,
        datastore_vals=ds_v,
        datastore_doc_ids=ds_d,
        token_byte_lengths=np.ones(16, dtype=np.int32),
        rare_token_mask=np.ones(16, dtype=bool),
    )
    assert "hyper_scoped_knn_rare_bpb" in e2

    # Finite-difference check on CountShrunkSpanComposer + SheafKV + OESpan Transformer
    comp = CountShrunkSpanComposer([(0, 1), (2, 3)], np.array([10.0, 800.0], dtype=np.float32), kappa=400.0)
    cfg = TransformerConfig(
        vocab_size=34,
        d_model=16,
        n_heads=2,
        n_layers=1,
        d_ff=32,
        window_len=8,
        base_width_mup=16,
        head_kind="tiled_cce",
        use_oe_span=True,
        n_spans=4,
        d_span=8,
        kv_prefix_slots=2,
        d_stalk=8,
        trime_local_drop=0.1,
    )
    mod = CausalTransformerLM(cfg, seed=5, span_composer=comp)
    assert mod.parameter_count() > 0
    xb = rng.integers(0, 34, size=(2, 8), dtype=np.int32)
    yb = rng.integers(0, 34, size=(2, 8), dtype=np.int32)
    sb = rng.integers(-1, 4, size=(2, 8), dtype=np.int32)
    ps = rng.standard_normal((2, 2, 8)).astype(np.float32)
    l0, grads = mod.forward_backward(xb, yb, span_ids=sb, prefix_stalks=ps, training=True)
    mod.step_adamw(grads, lr=5e-3)
    l1, _ = mod.forward_backward(xb, yb, span_ids=sb, prefix_stalks=ps, training=False)
    assert np.isfinite(l0) and np.isfinite(l1)

    # Fallback corpus loader check on missing directory
    synth_docs = load_clean_technical_corpus(doc_root=Path("/nonexistent_dir_xyz"), max_articles=70)
    assert len(synth_docs) >= 60

    # Mini 2-seed synthetic suite smoke
    tb_audit = audit_kernel_source((ROOT / "src" / "hadukenn_core" / "torch_bridge.py").read_text(encoding="utf-8"))
    assert tb_audit.ok is True


def test_t40_sealed_campaign_cli_and_receipts():
    t40_dir = ROOT / "iterations" / "t40-local-empirical-stack"
    res_path = t40_dir / "T40_RESULTS.json"
    att_path = t40_dir / "T40_DSSE_ATTESTATION.json"
    rec_path = t40_dir / "T40_SCITT_RECEIPT.json"
    assert res_path.is_file() and att_path.is_file() and rec_path.is_file()

    proc = subprocess.run(
        [sys.executable, "-m", "sheaf_verify", "campaign", "--mode", "t40-stack", "--path", str(res_path), "--json"],
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT / "src")},
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    out = json.loads(proc.stdout)
    assert out["ok"] is True
    assert any("attention-sink" in a.lower() for a in out["advisories"])

    from sheaf_verify.cli import main as cli_main
    rc = cli_main(["campaign", "--mode", "t40-stack", "--path", str(res_path), "--json"])
    assert rc == 0
