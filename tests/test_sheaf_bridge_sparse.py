"""Tests for M3 chunked sparse CSR Sheaf-Bridge (F10) and ARM64/CPU torch_bridge (F11)."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

import hadukenn_core.torch_bridge as torch_bridge
from hadukenn_core.sheaf_bridge import (
    SparseCSRMatrix,
    _convert_csr_dtype,
    build_sheaf_bridge_csr,
    build_sheaf_bridge_operator,
    compute_kruskal_mst_laplacian,
    compute_lle_barycentric_weights,
    compute_lle_barycentric_weights_csr,
    compute_mst_laplacian_csr,
    quantize_dequantize_symmetric_int8,
    whitened_spectral_warmstart,
)
from sheaf_verify.anticheat import audit_kernel_source

ROOT = Path(__file__).resolve().parents[1]


def _count_connected_components_from_laplacian_csr(l_mst: SparseCSRMatrix) -> int:
    n_verts = l_mst.shape[0]
    visited = np.zeros(n_verts, dtype=bool)
    comps = 0
    for start in range(n_verts):
        if visited[start]:
            continue
        comps += 1
        stack = [start]
        visited[start] = True
        while stack:
            node = stack.pop()
            p0 = int(l_mst.indptr[node])
            p1 = int(l_mst.indptr[node + 1])
            for nbr in l_mst.indices[p0:p1]:
                n_idx = int(nbr)
                if n_idx != node and not visited[n_idx]:
                    visited[n_idx] = True
                    stack.append(n_idx)
    return comps


def test_dense_apis_preserved_and_small_v_csr_parity():
    rng = np.random.default_rng(17)
    oracle_iso = rng.normal(0.0, 1.0, size=(32, 64)).astype(np.float32)

    w_warm, isotropy = whitened_spectral_warmstart(oracle_iso, d_student=8)
    assert w_warm.shape == (32, 8)
    assert 0.25 <= isotropy <= 1.0
    q_w = quantize_dequantize_symmetric_int8(w_warm)
    assert q_w.shape == w_warm.shape

    w_dense = compute_lle_barycentric_weights(oracle_iso, k=4)
    w_csr = compute_lle_barycentric_weights_csr(oracle_iso, k=4, chunk_size=16)
    assert w_csr.shape == (32, 32)
    assert w_csr.nnz == 32 * 4
    assert np.allclose(w_csr.to_dense(), w_dense, atol=1e-5)

    l_mst_dense, edges = compute_kruskal_mst_laplacian(oracle_iso)
    l_mst_csr = compute_mst_laplacian_csr(oracle_iso, k_cand=16, chunk_size=16)
    assert len(edges) == 31
    assert l_mst_csr.nnz == 3 * 32 - 2
    assert np.allclose(l_mst_csr.to_dense(), l_mst_dense, atol=1e-6)

    l_dense, stats = build_sheaf_bridge_operator(oracle_iso, k=4, mst_weight=0.15)
    assert stats["mst_edges"] == 31

    res_f32 = build_sheaf_bridge_csr(oracle_iso, k=4, mst_weight=0.15, chunk_size=16, dtype="float32")
    op_f32 = res_f32["operator"]
    assert np.allclose(op_f32.to_dense(), l_dense, atol=1e-4)

    x_1d = rng.normal(0.0, 1.0, size=(32,)).astype(np.float32)
    x_2d = rng.normal(0.0, 1.0, size=(32, 8)).astype(np.float32)
    assert np.allclose(op_f32.matvec(x_1d), l_dense @ x_1d, atol=1e-4)
    assert np.allclose(op_f32 @ x_2d, l_dense @ x_2d, atol=1e-4)

    res_f16 = build_sheaf_bridge_csr(oracle_iso, k=4, mst_weight=0.15, chunk_size=16, dtype="float16")
    assert np.allclose(res_f16["operator"].matvec(x_2d), l_dense @ x_2d, atol=2e-2)

    centers = rng.normal(0.0, 1.0, size=(12, 256)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    oracle_clust = centers[np.arange(192) % 12] + rng.normal(0.0, 0.15, size=(192, 256)).astype(np.float32)
    l_clust_dense, _ = build_sheaf_bridge_operator(oracle_clust, k=8, mst_weight=0.15)
    res_clust = build_sheaf_bridge_csr(oracle_clust, k=8, mst_weight=0.15, chunk_size=64, dtype="float32")
    x_c = rng.normal(0.0, 1.0, size=(192, 6)).astype(np.float32)
    assert np.allclose(res_clust["operator"].matvec(x_c), l_clust_dense @ x_c, atol=2e-4)


def test_mst_csr_disconnected_clusters_connectivity():
    rng = np.random.default_rng(29)

    for vocab_size, n_clusters, k_cand in ((128, 8, 8), (2560, 8, 16)):
        centers = rng.normal(0.0, 1.0, size=(n_clusters, 32)).astype(np.float32)
        centers /= np.linalg.norm(centers, axis=1, keepdims=True)
        labels = np.arange(vocab_size) % n_clusters
        emb = centers[labels] + rng.normal(0.0, 0.03, size=(vocab_size, 32)).astype(np.float32)

        l_mst = compute_mst_laplacian_csr(emb, k_cand=k_cand, chunk_size=256)
        assert l_mst.shape == (vocab_size, vocab_size)
        assert l_mst.nnz == 3 * vocab_size - 2
        assert _count_connected_components_from_laplacian_csr(l_mst) == 1

        ones = np.ones(vocab_size, dtype=np.float32)
        assert float(np.max(np.abs(l_mst.matvec(ones)))) < 1e-5
        x_vec = rng.normal(0.0, 1.0, size=(vocab_size,)).astype(np.float32)
        assert np.allclose(l_mst.matvec(x_vec), l_mst.rmatvec(x_vec), atol=1e-5)


def test_sheaf_bridge_csr_v32768_budget_and_peak_ram():
    rng = np.random.default_rng(42)
    vocab_size = 32768
    d_emb = 64
    n_clusters = 64
    centers = rng.normal(0.0, 1.0, size=(n_clusters, d_emb)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    emb = centers[np.arange(vocab_size) % n_clusters] + rng.normal(0.0, 0.12, size=(vocab_size, d_emb)).astype(
        np.float32
    )

    res = build_sheaf_bridge_csr(emb, k=8, reg=1e-3, mst_weight=0.15, chunk_size=1024, dtype="float16")
    assert res["vocab_size"] == vocab_size
    assert res["lle_nnz"] == vocab_size * 8
    assert res["mst_edges"] == vocab_size - 1
    assert res["dense_flat_head_preserved"] is True
    assert res["csr_bytes"] <= 8 * 1024 * 1024
    assert res["peak_ram_bytes"] < 64 * 1024 * 1024
    assert res["within_budget"] is True

    op = res["operator"]
    ones = np.ones((vocab_size, 8), dtype=np.float32)
    out_ones = op.matvec(ones)
    assert out_ones.shape == (vocab_size, 8)
    assert np.all(np.isfinite(out_ones))
    assert float(np.max(np.abs(out_ones))) < 1e-2

    with pytest.raises(MemoryError):
        op.to_dense()


def test_sparse_csr_input_validation_dtypes_and_empty_rows():
    rng = np.random.default_rng(7)
    emb = rng.normal(0.0, 1.0, size=(24, 16)).astype(np.float32)

    with pytest.raises(ValueError):
        compute_lle_barycentric_weights_csr(emb, k=0)
    with pytest.raises(ValueError):
        compute_lle_barycentric_weights_csr(emb, k=24)
    with pytest.raises(ValueError):
        compute_lle_barycentric_weights_csr(emb, k=4, chunk_size=0)
    with pytest.raises(ValueError):
        compute_mst_laplacian_csr(emb, k_cand=0)
    with pytest.raises(ValueError):
        build_sheaf_bridge_csr(emb, k=4, dtype="bfloat16")
    with pytest.raises(ValueError):
        build_sheaf_bridge_csr(emb[:, 0], k=4)

    res_i8 = build_sheaf_bridge_csr(emb, k=4, chunk_size=8, dtype="int8")
    assert res_i8["w_lle"].data.dtype == np.int8
    assert res_i8["w_lle"].scales is not None
    assert res_i8["l_mst"].data.dtype in (np.float16, np.float32)
    assert res_i8["l_mst"].scales is None
    l_mst_iso_dense = res_i8["l_mst"].to_dense()
    assert float(np.max(np.abs(l_mst_iso_dense - l_mst_iso_dense.T))) == 0.0
    assert float(np.max(np.abs(res_i8["l_mst"].matvec(np.ones(24, dtype=np.float32))))) == 0.0
    assert res_i8["csr_bytes"] <= 8 * 1024 * 1024
    assert res_i8["csr_bytes"] == res_i8["w_lle"].nbytes + res_i8["l_mst"].nbytes
    x_test = rng.normal(0.0, 1.0, size=(24, 4)).astype(np.float32)
    y_i8 = res_i8["operator"].matvec(x_test)
    y_f32 = build_sheaf_bridge_csr(emb, k=4, chunk_size=8, dtype="float32")["operator"].matvec(x_test)
    assert y_i8.shape == (24, 4)
    assert np.allclose(y_i8, y_f32, atol=0.15)
    assert res_i8["operator"].to_dense().shape == (24, 24)

    # All-identical hub (V=2560, degree-1702 star hub) must keep L_mst symmetric & zero row-sum under dtype="int8"
    res_hub_i8 = build_sheaf_bridge_csr(np.ones((2560, 16), dtype=np.float32), k=8, chunk_size=256, dtype="int8")
    assert res_hub_i8["w_lle"].data.dtype == np.int8
    assert res_hub_i8["l_mst"].data.dtype in (np.float16, np.float32)
    assert res_hub_i8["l_mst"].scales is None
    assert res_hub_i8["csr_bytes"] <= 8 * 1024 * 1024
    l_mst_hub_dense = res_hub_i8["l_mst"].to_dense()
    assert float(np.max(np.abs(l_mst_hub_dense - l_mst_hub_dense.T))) == 0.0
    assert float(np.max(np.abs(res_hub_i8["l_mst"].matvec(np.ones(2560, dtype=np.float32))))) == 0.0

    # Explicit _convert_csr_dtype: preserve_large_int_diag=True falls back to float32 >2048; False still quantizes to int8
    big_diag = SparseCSRMatrix(
        shape=(2, 2),
        indptr=np.array([0, 2, 4], dtype=np.int32),
        indices=np.array([0, 1, 0, 1], dtype=np.int32),
        data=np.array([2500.0, -1.0, -1.0, 1.0], dtype=np.float32),
    )
    assert _convert_csr_dtype(big_diag, "int8", preserve_large_int_diag=True).data.dtype == np.float32
    explicit_i8 = _convert_csr_dtype(big_diag, "int8", preserve_large_int_diag=False)
    assert explicit_i8.data.dtype == np.int8 and explicit_i8.scales is not None

    with pytest.raises(ValueError):
        res_i8["w_lle"].matvec(np.ones((10,), dtype=np.float32))
    with pytest.raises(ValueError):
        res_i8["w_lle"].rmatvec(np.ones((10,), dtype=np.float32))

    # Empty-row CSR matvec/rmatvec correctness
    indptr = np.array([0, 2, 2, 3, 3], dtype=np.int32)
    indices = np.array([0, 3, 1], dtype=np.int32)
    data = np.array([1.5, -2.0, 3.0], dtype=np.float16)
    csr = SparseCSRMatrix(shape=(4, 4), indptr=indptr, indices=indices, data=data)
    dense = csr.to_dense()
    vec = np.array([2.0, -1.0, 0.5, 4.0], dtype=np.float32)
    mat = np.stack([vec, vec * 0.5], axis=1)
    assert np.allclose(csr.matvec(vec), dense @ vec, atol=1e-5)
    assert np.allclose(csr.rmatvec(vec), dense.T @ vec, atol=1e-5)
    assert np.allclose(csr.matvec(mat), dense @ mat, atol=1e-5)
    assert np.allclose(csr.rmatvec(mat), dense.T @ mat, atol=1e-5)


def test_torch_bridge_anticheat_hygiene_and_import_fallback():
    tb_path = ROOT / "src" / "hadukenn_core" / "torch_bridge.py"
    sb_path = ROOT / "src" / "hadukenn_core" / "sheaf_bridge.py"
    tb_src = tb_path.read_text(encoding="utf-8")
    sb_src = sb_path.read_text(encoding="utf-8")

    rep = audit_kernel_source(tb_src)
    assert rep.ok is True
    assert rep.violations == ()

    for text in (tb_src, sb_src):
        assert "torch.cuda" not in text
        assert 'device="cuda"' not in text
        for drive in ("C:" + "\\", "D:" + "\\", "E:" + "\\", "H:" + "\\"):
            assert drive not in text

    assert isinstance(torch_bridge.TORCH_AVAILABLE, bool)
    assert len(torch_bridge.NF4_VALUES) == 16

    had_torch = "torch" in sys.modules
    orig_torch = sys.modules.get("torch")
    try:
        sys.modules["torch"] = None  # type: ignore[assignment]
        reloaded = importlib.reload(torch_bridge)
        assert reloaded.TORCH_AVAILABLE is False
    finally:
        if had_torch:
            sys.modules["torch"] = orig_torch
        else:
            sys.modules.pop("torch", None)
        importlib.reload(torch_bridge)


def test_torch_bridge_cpu_mixed_dtypes_when_available():
    if not torch_bridge.TORCH_AVAILABLE:
        return

    import torch

    torch.manual_seed(11)
    nc, s_sz, d_model = 4, 8, 16
    vocab_size = nc * s_sz
    cluster_units = torch.arange(vocab_size, dtype=torch.long).view(nc, s_sz)
    targets = torch.tensor([[0, 5, 12, 31], [3, 9, 17, 25]], dtype=torch.int32)
    span_ids = torch.tensor([[-1, 0, 2, -1], [1, -1, 3, 0]], dtype=torch.int32)

    for dt in (torch.float32, torch.bfloat16, torch.float16):
        # 1. Mixed-dtype path: float32 modules + bf16/fp16 activations without .to(dt)
        head = torch_bridge.ParadigmaticClusterISHeadTorch(cluster_units, d_model=d_model, k_neg=2, chunk_tokens=4)
        base_w = torch.randn(vocab_size, d_model, dtype=dt, requires_grad=True)
        hid = torch.randn(2, 4, d_model, dtype=dt, requires_grad=True)
        lm_loss, root_nll = head(hid, targets, base_w)
        assert torch.isfinite(lm_loss) and torch.isfinite(root_nll)
        (lm_loss + root_nll).backward()
        assert hid.grad is not None and torch.isfinite(hid.grad).all()
        assert base_w.grad is not None and torch.isfinite(base_w.grad).all()
        assert head.delta_root.grad is not None and torch.isfinite(head.delta_root.grad).all()

        lin = torch_bridge.NF4FrozenLinearTorch(torch.randn(12, d_model, dtype=torch.float32), block_size=8, lora_rank=4)
        x_lin = torch.randn(3, d_model, dtype=dt, requires_grad=True)
        y_lin = lin(x_lin)
        assert y_lin.shape == (3, 12) and y_lin.dtype == dt and torch.isfinite(y_lin).all()
        y_lin.float().sum().backward()
        assert x_lin.grad is not None and torch.isfinite(x_lin.grad).all()

        span_mod = torch_bridge.OESpanEmbeddingTorch(n_spans=4, d_span=6, d_model=d_model)
        base_emb = torch.randn(2, 4, d_model, dtype=dt, requires_grad=True)
        out_emb = span_mod(base_emb, span_ids)
        assert out_emb.shape == (2, 4, d_model) and out_emb.dtype == dt and torch.isfinite(out_emb).all()
        out_emb.float().sum().backward()
        assert base_emb.grad is not None and torch.isfinite(base_emb.grad).all()

        skv = torch_bridge.AsymmetricSheafKVPrefixTorch(d_model=d_model, d_stalk=8, n_heads=2)
        q_ctx = torch.randn(2, d_model, dtype=dt, requires_grad=True)
        stalks = torch.randn(2, 3, 8, dtype=dt, requires_grad=True)
        k_mem, v_mem = skv(q_ctx, stalks)
        assert k_mem.dtype == dt and v_mem.dtype == dt
        q_u = torch.randn(2, 2, 4, 8, dtype=dt)
        q_r = torch.randn(2, 2, 4, 8, dtype=dt)
        k_r = torch.randn(2, 2, 4, 8, dtype=dt)
        scores = skv.compute_hybrid_scores(q_u, q_r, k_mem, k_r)
        assert scores.shape == (2, 2, 4, 7) and scores.dtype == dt and torch.isfinite(scores).all()
        (scores.float().sum() + v_mem.float().sum()).backward()
        assert q_ctx.grad is not None and torch.isfinite(q_ctx.grad).all()

        # 2. Homogeneous path: module explicitly cast via .to(dt)
        head_cast = torch_bridge.ParadigmaticClusterISHeadTorch(cluster_units, d_model=d_model, k_neg=2).to(dt)
        lm2, rn2 = head_cast(hid.detach(), targets, base_w.detach())
        assert torch.isfinite(lm2) and torch.isfinite(rn2)
        skv_cast = torch_bridge.AsymmetricSheafKVPrefixTorch(d_model=d_model, d_stalk=8, n_heads=2).to(dt)
        km2, vm2 = skv_cast(q_ctx.detach(), stalks.detach())
        assert km2.dtype == dt and vm2.dtype == dt

    # 3. Boundary-input guards: n_clusters == 1, k_neg >= n_clusters, odd/non-positive block_size, n_tok == 0
    with pytest.raises(ValueError):
        torch_bridge.ParadigmaticClusterISHeadTorch(torch.arange(8).view(1, 8), d_model=d_model, k_neg=1)
    with pytest.raises(ValueError):
        torch_bridge.ParadigmaticClusterISHeadTorch(torch.arange(16).view(2, 8), d_model=d_model, k_neg=2)
    with pytest.raises(ValueError):
        torch_bridge.ParadigmaticClusterISHeadTorch(torch.arange(16).view(2, 8), d_model=d_model, k_neg=0)
    with pytest.raises(ValueError):
        torch_bridge.NF4FrozenLinearTorch(torch.randn(4, 8, dtype=torch.float32), block_size=3)
    with pytest.raises(ValueError):
        torch_bridge.NF4FrozenLinearTorch(torch.randn(4, 8, dtype=torch.float32), block_size=0)

    head_empty = torch_bridge.ParadigmaticClusterISHeadTorch(cluster_units, d_model=d_model, k_neg=2)
    base_empty = torch.randn(vocab_size, d_model, dtype=torch.float32, requires_grad=True)
    hid_empty = torch.randn(2, 0, d_model, dtype=torch.float32, requires_grad=True)
    targets_empty = torch.empty(2, 0, dtype=torch.int32)
    lm_empty, rn_empty = head_empty(hid_empty, targets_empty, base_empty)
    assert torch.isfinite(lm_empty) and lm_empty.item() == 0.0 and lm_empty.requires_grad
    assert torch.isfinite(rn_empty) and rn_empty.item() == 0.0 and rn_empty.requires_grad
    (lm_empty + rn_empty).backward()
    assert base_empty.grad is not None and torch.isfinite(base_empty.grad).all()
    assert head_empty.delta_root.grad is not None and torch.isfinite(head_empty.delta_root.grad).all()
    assert hid_empty.grad is not None and hid_empty.grad.shape == (2, 0, d_model)

    head_empty2 = torch_bridge.ParadigmaticClusterISHeadTorch(cluster_units, d_model=d_model, k_neg=2)
    lm_only, _ = head_empty2(hid_empty.detach(), targets_empty, base_empty.detach())
    assert torch.isfinite(lm_only) and lm_only.item() == 0.0 and lm_only.requires_grad
    lm_only.backward()
    assert head_empty2.delta_root.grad is not None and torch.isfinite(head_empty2.delta_root.grad).all()
