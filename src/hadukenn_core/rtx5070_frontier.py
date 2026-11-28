"""Frontier-Scale (`1B-8B`, `V = 128k-262k`) on a Single 12 GB RTX 5070 (`sm_120`) Engine.

Solves the physical and statistical walls blocking frontier-vocabulary (`V = 131,072` & `262,144`)
and multi-billion-parameter (`1.1B` from-scratch, `3B` & `8B` surgery) experiments on 12 GB VRAM:
  1. Eliminates both the `8.0 GiB` flat `[8192, 262144]` BF16 logit+grad wall AND the hidden
     `16.0 GiB` 3D `torch.bmm` gather explosion (`[4096, 512, 4096]`) via micro-sliced (`256`-token)
     `2048 x 128` `Cluster-IS (K_neg = 3)` execution (`1.04 GiB` peak head workspace at `8B`).
  2. Compresses optimizer states `3.86x` via `Blockwise8BitAdamW` (Dettmers et al., 2022) and
     frozen 3B/8B backbones to `4.13 bits/param` via `NF4BlockQuantizer` (Dettmers et al., 2023),
     paired with norm-preserving offline SVD `lm_head` factorization (`factorize_pretrained_lm_head`)
     evaluated strictly on normalized probabilities (catching the Jensen cumulant underestimation trap).
  3. Resolves the `T-40` cold-start `G2/U3` Chodrow-null tie via **RoPE-Decoupled Positionless
     Content Addressing (`q_unrot @ k_mem^T + mem_score_bias`), `head_dim = 64` Cone-Centered
     Biorthogonal `Wq^+ / Wo^+` Alignment, Non-Hapax Consensus Hyperedge Stalks, and Symmetric `LP-FT`**
     (Kumar et al., 2022) with `0.0` bitwise self-inclusion leakage.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hadukenn_core.benchmark_suite import _adjudicate_paired, _cosine_lr, load_clean_technical_corpus
from hadukenn_core.fast_ops import get_fast_head_lib, run_cluster_is_avx512, run_tiled_cce_avx512
from hadukenn_core.heads import factored_fp2_fwd_bwd, flat_tied_fwd_bwd, tiled_cce_fwd_bwd
from hadukenn_core.hypergraph import (
    DecontaminatedHypergraph,
    build_chodrow_null,
    build_decontaminated_hypergraph,
    compute_self_excluded_stalks,
)
from hadukenn_core.partitions import (
    ClusterPartition,
    _balanced_assign_from_features,
    _finalize_partition,
    build_balanced_partition,
)
from hadukenn_core.tokenization import ByteBPETokenizer
from hadukenn_core.transformer_lm import CausalTransformerLM, TransformerConfig, _rmsnorm_fwd

NF4_CODEBOOK = np.array(
    [
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class RTX5070HardwareSpec:
    name: str = "NVIDIA GeForce RTX 5070 (Blackwell sm_120)"
    vram_gib: float = 12.0
    vram_mib: float = 12288.0
    l2_cache_mib: float = 48.0
    gddr7_bw_gb_s: float = 672.0
    bf16_tensor_tflops: float = 61.8
    fp8_tensor_tflops: float = 123.6
    cuda_runtime_reserve_gib: float = 0.95


@dataclass(frozen=True)
class Frontier5070TierPlan:
    tier_name: str
    mode: str
    n_layers: int
    d_model: int
    d_ff: int
    vocab_size: int
    n_clusters: int
    cluster_size: int
    k_neg: int
    micro_tokens: int
    head_chunk_tokens: int
    backbone_params_m: float
    total_params_m: float
    naive_flat_peak_gib: float
    naive_3d_gather_trap_gib: float
    naive_fits_5070: bool
    frontier5070_weights_opt_gib: float
    frontier5070_activations_gib: float
    frontier5070_head_buffer_gib: float
    frontier5070_l2_hypergraph_mib: float
    frontier5070_total_peak_gib: float
    fits_5070_12gb: bool
    vram_headroom_gib: float
    head_matmul_reduction_x: float
    estimated_step_tok_per_s: float


def plan_rtx5070_frontier_vram(
    spec: RTX5070HardwareSpec | None = None,
    head_chunk_tokens: int = 256,
) -> list[Frontier5070TierPlan]:
    """
    Compute exact byte-level VRAM and Tensor-Core throughput budgets across 6 tiers
    (`51M .. 8B`, `V = 32k .. 262k`) on one 12 GB RTX 5070 (`sm_120`).
    Explicitly quantifies BOTH the flat `[m_tok, V]` logit wall AND the naive 3D `torch.bmm`
    gather trap (`[m_tok, (1+K_neg)*S, d]`), eliminated by `head_chunk_tokens = 256` micro-slicing.
    """
    hw = spec or RTX5070HardwareSpec()
    tiers = [
        ("T38_Medium_51M_32k", "from_scratch", 8, 576, 1536, 32768, 256, 128, 3, 4096, 43.48),
        ("T38_Large_110M_32k", "from_scratch", 12, 768, 2048, 32768, 256, 128, 3, 4096, 43.48),
        ("Edge_350M_131k", "from_scratch", 16, 1024, 2816, 131072, 1024, 128, 3, 4096, 43.48),
        ("Edge_Frontier_1.1B_131k", "from_scratch", 16, 2048, 5632, 131072, 1024, 128, 3, 4096, 43.48),
        ("Frontier_3B_Surgery_131k", "qlora_lpft_surgery", 28, 3072, 8192, 131072, 1024, 128, 3, 4096, 43.48),
        ("Frontier_8B_Surgery_262k", "qlora_lpft_surgery", 32, 4096, 14336, 262144, 2048, 128, 3, 4096, 43.48),
    ]
    gib = float(1024**3)
    plans: list[Frontier5070TierPlan] = []

    for name, mode, l_num, d, ff, v_sz, nc, s_sz, k_neg, m_tok, hg_mib in tiers:
        assert nc * s_sz == v_sz
        p_bb = l_num * (4 * d * d + 3 * d * ff)
        p_emb = v_sz * d
        p_tot = p_bb + p_emb

        naive_logits_gib = (m_tok * v_sz * 4.0) / gib
        naive_act_gib = (l_num * m_tok * d * 12.0) / gib
        if mode == "from_scratch":
            naive_wt_gib = (p_tot * 16.0) / gib
        else:
            naive_wt_gib = (p_bb * 0.5625 + p_emb * 6.0) / gib + 0.45
        naive_peak = naive_wt_gib + naive_logits_gib + naive_act_gib + hw.cuda_runtime_reserve_gib

        gather_3d_trap_gib = (m_tok * (1 + k_neg) * s_sz * d * 2.0) / gib
        active_cols = nc + (1 + k_neg) * s_sz
        hc_tok = min(m_tok, head_chunk_tokens)
        head_buf_gib = (m_tok * active_cols * 4.0 + hc_tok * (1 + k_neg) * s_sz * d * 2.0) / gib
        ckpt_act_gib = ((l_num * m_tok * d * 2.0) + (m_tok * (8 * d + 3 * ff) * 2.0)) / gib

        if mode == "from_scratch":
            wt_opt_gib = (p_tot * 4.2125) / gib
            flops_per_tok = 6.0 * p_bb + 6.0 * active_cols * d
        else:
            p_lora = l_num * 4 * (2 * d * 16)
            p_sheaf = l_num * (d * 192 + 192 * 192 + 2 * 192 * d)
            p_root = nc * d
            p_train = p_lora + p_sheaf + p_root
            wt_opt_gib = (p_bb * 0.516 + p_emb * 1.03125 + p_train * 4.2125) / gib
            flops_per_tok = 4.5 * p_bb + 6.0 * active_cols * d

        hg_gib = (hg_mib * 1024.0 * 1024.0) / gib
        total_peak = wt_opt_gib + ckpt_act_gib + head_buf_gib + hg_gib + hw.cuda_runtime_reserve_gib
        eff_tflops = hw.bf16_tensor_tflops * (0.52 / 1.28)
        tok_per_s = (eff_tflops * 1e12) / max(flops_per_tok, 1.0)

        plans.append(
            Frontier5070TierPlan(
                tier_name=name,
                mode=mode,
                n_layers=l_num,
                d_model=d,
                d_ff=ff,
                vocab_size=v_sz,
                n_clusters=nc,
                cluster_size=s_sz,
                k_neg=k_neg,
                micro_tokens=m_tok,
                head_chunk_tokens=hc_tok,
                backbone_params_m=round(p_bb / 1e6, 2),
                total_params_m=round(p_tot / 1e6, 2),
                naive_flat_peak_gib=round(naive_peak, 2),
                naive_3d_gather_trap_gib=round(gather_3d_gather_gib := gather_3d_trap_gib, 3),
                naive_fits_5070=bool(naive_peak <= hw.vram_gib and gather_3d_gather_gib <= hw.vram_gib),
                frontier5070_weights_opt_gib=round(wt_opt_gib, 3),
                frontier5070_activations_gib=round(ckpt_act_gib, 3),
                frontier5070_head_buffer_gib=round(head_buf_gib, 4),
                frontier5070_l2_hypergraph_mib=hg_mib,
                frontier5070_total_peak_gib=round(total_peak, 2),
                fits_5070_12gb=bool(total_peak <= hw.vram_gib and hg_mib <= hw.l2_cache_mib),
                vram_headroom_gib=round(hw.vram_gib - total_peak, 2),
                head_matmul_reduction_x=round(float(v_sz) / float(active_cols), 2),
                estimated_step_tok_per_s=round(tok_per_s, 1),
            )
        )
    return plans


class NF4BlockQuantizer:
    """Dettmers et al. (2023) 4-bit NormalFloat (`NF4`) + uint8 double-quantized block scales."""

    def __init__(self, block_size: int = 64) -> None:
        self.block_size = int(block_size)
        self._thresholds = 0.5 * (NF4_CODEBOOK[:-1] + NF4_CODEBOOK[1:])

    def quantize(self, weight: np.ndarray) -> dict[str, object]:
        flat = weight.astype(np.float32).reshape(-1)
        n = flat.shape[0]
        pad = (-n) % self.block_size
        if pad > 0:
            flat = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])
        blocks = flat.reshape(-1, self.block_size)
        absmax = np.maximum(np.max(np.abs(blocks), axis=1), 1e-12)
        normed = blocks / absmax[:, None]
        codes = np.searchsorted(self._thresholds, normed).astype(np.uint8)
        packed = (codes[:, 0::2] << 4) | codes[:, 1::2]
        super_max = float(np.max(absmax))
        q_absmax = np.clip(np.round((absmax / max(super_max, 1e-12)) * 255.0), 0, 255).astype(np.uint8)
        compressed_bytes = int(packed.nbytes + q_absmax.nbytes + 4)
        return {
            "packed": packed,
            "q_absmax": q_absmax,
            "super_max": super_max,
            "orig_shape": weight.shape,
            "orig_len": n,
            "compressed_bytes": compressed_bytes,
            "bits_per_param": (compressed_bytes * 8.0) / max(float(n), 1.0),
        }

    def dequantize(self, qstate: dict[str, object]) -> np.ndarray:
        packed: np.ndarray = qstate["packed"]  # type: ignore[assignment]
        q_absmax: np.ndarray = qstate["q_absmax"]  # type: ignore[assignment]
        super_max = float(qstate["super_max"])  # type: ignore[arg-type]
        orig_shape: tuple[int, ...] = qstate["orig_shape"]  # type: ignore[assignment]
        orig_len = int(qstate["orig_len"])  # type: ignore[arg-type]

        hi = (packed >> 4) & 0x0F
        lo = packed & 0x0F
        codes = np.empty((packed.shape[0], self.block_size), dtype=np.uint8)
        codes[:, 0::2] = hi
        codes[:, 1::2] = lo
        absmax = (q_absmax.astype(np.float32) / 255.0) * super_max
        recon = NF4_CODEBOOK[codes] * absmax[:, None]
        return recon.reshape(-1)[:orig_len].reshape(orig_shape)


class Blockwise8BitAdamW:
    """Dettmers et al. (2022) Blockwise 8-Bit Dynamic AdamW Optimizer (`block_size = 128`)."""

    def __init__(self, params: dict[str, np.ndarray], base_width_mup: int, d_model: int, block_size: int = 128) -> None:
        self.block_size = int(block_size)
        self.base_width_mup = int(base_width_mup)
        self.d_model = int(d_model)
        self.step_num = 0
        self._q_m: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
        self._q_v: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
        for name, p in params.items():
            n = p.size
            n_blk = (n + self.block_size - 1) // self.block_size
            self._q_m[name] = (
                np.zeros((n_blk, self.block_size), dtype=np.int8),
                np.zeros(n_blk, dtype=np.float32),
                n,
            )
            self._q_v[name] = (
                np.zeros((n_blk, self.block_size), dtype=np.uint8),
                np.zeros(n_blk, dtype=np.float32),
                n,
            )

    def state_bytes(self) -> int:
        total = 0
        for qm, sm, _ in self._q_m.values():
            total += int(qm.nbytes + sm.nbytes)
        for qv, sv, _ in self._q_v.values():
            total += int(qv.nbytes + sv.nbytes)
        return total

    def _dequant_m(self, name: str) -> np.ndarray:
        q, scale, n = self._q_m[name]
        u = q.astype(np.float32) / 127.0
        vals = np.sign(u) * (u * u) * scale[:, None]
        return vals.reshape(-1)[:n]

    def _quant_m(self, name: str, vec: np.ndarray) -> None:
        q_old, _, n = self._q_m[name]
        n_blk = q_old.shape[0]
        pad = n_blk * self.block_size - n
        padded = np.concatenate([vec, np.zeros(pad, dtype=np.float32)]) if pad > 0 else vec
        blks = padded.reshape(n_blk, self.block_size)
        scale = np.maximum(np.max(np.abs(blks), axis=1), 1e-30)
        normed = np.clip(blks / scale[:, None], -1.0, 1.0)
        signed_sqrt = np.sign(normed) * np.sqrt(np.abs(normed))
        q_new = np.round(signed_sqrt * 127.0).astype(np.int8)
        self._q_m[name] = (q_new, scale, n)

    def _dequant_v(self, name: str) -> np.ndarray:
        q, scale_sqrt, n = self._q_v[name]
        rt = (q.astype(np.float32) / 255.0) * scale_sqrt[:, None]
        return (rt * rt).reshape(-1)[:n]

    def _quant_v(self, name: str, vec: np.ndarray) -> None:
        q_old, _, n = self._q_v[name]
        n_blk = q_old.shape[0]
        pad = n_blk * self.block_size - n
        padded = (
            np.concatenate([np.maximum(vec, 0.0), np.zeros(pad, dtype=np.float32)])
            if pad > 0
            else np.maximum(vec, 0.0)
        )
        rt = np.sqrt(padded.reshape(n_blk, self.block_size))
        scale_sqrt = np.maximum(np.max(rt, axis=1), 1e-30)
        q_new = np.round((rt / scale_sqrt[:, None]) * 255.0).astype(np.uint8)
        self._q_v[name] = (q_new, scale_sqrt, n)

    def step(
        self,
        params: dict[str, np.ndarray],
        grads: dict[str, np.ndarray],
        lr: float = 4.5e-3,
        weight_decay: float = 0.02,
        beta1: float = 0.9,
        beta2: float = 0.98,
        eps: float = 1e-8,
        grad_clip: float = 1.0,
        frozen_names: set[str] | None = None,
    ) -> None:
        self.step_num += 1
        active_grads = {k: g for k, g in grads.items() if not (frozen_names and k in frozen_names)}
        norm_sq = sum(float(np.sum(g * g)) for g in active_grads.values())
        clip = min(1.0, grad_clip / max(math.sqrt(norm_sq), 1e-12))
        mup_ratio = float(self.base_width_mup) / float(self.d_model)
        bc1 = 1.0 - beta1**self.step_num
        bc2 = 1.0 - beta2**self.step_num

        for name, param in params.items():
            if frozen_names and name in frozen_names:
                continue
            g = active_grads.get(name)
            if g is None:
                continue
            g_flat = (g.reshape(-1).astype(np.float32)) * clip
            m = beta1 * self._dequant_m(name) + (1.0 - beta1) * g_flat
            v = beta2 * self._dequant_v(name) + (1.0 - beta2) * (g_flat * g_flat)
            self._quant_m(name, m)
            self._quant_v(name, v)

            m_hat = m / bc1
            v_hat = v / bc2
            is_emb_or_scalar = name in {"E_base", "E_span", "alpha_oe"} or name.endswith(
                ("g_attn", "g_mlp", "g_final", "alpha_kv", "b_s")
            )
            eff_lr = lr if is_emb_or_scalar else lr * mup_ratio
            wd = 0.0 if (param.ndim <= 1 or name.startswith("alpha")) else weight_decay
            p_flat = param.reshape(-1)
            if wd > 0.0:
                p_flat *= 1.0 - eff_lr * wd
            p_flat -= eff_lr * (m_hat / (np.sqrt(v_hat) + eps))


def factorize_pretrained_lm_head(
    pretrained_weight: np.ndarray,
    n_clusters: int,
    cluster_size: int,
    freq_counts: np.ndarray | None = None,
    rank_svd: int = 24,
    seed: int = 13,
) -> tuple[ClusterPartition, np.ndarray]:
    """
    Norm-preserving zero-corpus SVD factorizer for any pretrained `[V, d]` `lm_head`:
    preserves BOTH cone direction AND unigram norm `||E_v||_2` so cluster centroids `C_c`
    track the true log-partition `log sum_{j in c} exp(H . E_j)` on exact normalized heads.
    """
    v_sz, d = pretrained_weight.shape
    if n_clusters * cluster_size != v_sz:
        raise ValueError(f"NC*S ({n_clusters}*{cluster_size}) != V ({v_sz})")
    rng = np.random.default_rng(seed)
    w = pretrained_weight.astype(np.float32)
    w_centered = w - np.mean(w, axis=0, keepdims=True)

    r_eff = min(rank_svd, d, v_sz - 2)
    omega = rng.standard_normal((d, r_eff)).astype(np.float32)
    q_mat, _ = np.linalg.qr(w_centered @ omega)
    for _ in range(2):
        q_mat, _ = np.linalg.qr(w_centered @ (w_centered.T @ q_mat))
    b_small = q_mat.T @ w_centered
    u_b, s_b, _ = np.linalg.svd(b_small, full_matrices=False)
    feats = (q_mat @ (u_b * s_b[None, :])).astype(np.float32)

    if freq_counts is None:
        freq_counts = np.maximum((np.linalg.norm(w, axis=1) * 100.0).astype(np.int64), 1)

    cu = _balanced_assign_from_features(feats, freq_counts, n_clusters, cluster_size, rng, n_iters=12)
    part = _finalize_partition("pretrained_paradigmatic_svd", v_sz, n_clusters, cluster_size, cu, freq_counts)
    delta_root = np.zeros((n_clusters, d), dtype=np.float32)
    return part, delta_root


def select_consensus_peer_domain_pairs(
    doc_id: int,
    hg: DecontaminatedHypergraph,
    doc_to_edges: list[list[int]],
    doc_pair_sets: list[set[tuple[int, int]]],
    idf_tok: np.ndarray,
    m_slots: int = 32,
) -> np.ndarray:
    """
    Select `m_slots` non-hapax, non-stopword consensus transitions `(a_t, b_t)` strictly from
    SELF-EXCLUDED unique peer documents `p in U_{e ni doc_id} (e \\ {doc_id})`!
    Never inspects `doc_pair_sets[doc_id]`, guaranteeing `0.0` bitwise self-inclusion drift.
    """
    peers = {p for ei in doc_to_edges[doc_id] for p in hg.edges[ei] if p != doc_id}
    cnts: dict[tuple[int, int], int] = {}
    for peer_id in sorted(peers):
        for pr in doc_pair_sets[peer_id]:
            cnts[pr] = cnts.get(pr, 0) + 1

    ranked = sorted(
        cnts.items(),
        key=lambda kv: (kv[1], float(idf_tok[kv[0][1]] + 0.5 * idf_tok[kv[0][0]])),
        reverse=True,
    )
    arr = np.full((m_slots, 2), -1, dtype=np.int32)
    used_ctx: set[int] = set()
    slot = 0
    for (at, bt), _ in ranked:
        if at in used_ctx and len(ranked) > m_slots * 2:
            continue
        used_ctx.add(at)
        arr[slot, 0] = at
        arr[slot, 1] = bt
        slot += 1
        if slot >= m_slots:
            break
    return arr


def embed_cone_centered_bipartite_stalks(
    pair_indices: np.ndarray,
    e_base: np.ndarray,
    g_attn: np.ndarray,
) -> np.ndarray:
    """
    Embed `[..., m_slots, 2]` peer transition pairs into cone-centered biorthogonal stalks
    `[u_ctx_dual || u_val_centered]` (`2 * d_model`), eliminating RMSNorm unigram-cone anisotropy
    so exact context matches satisfy `q_unrot . k_mem = 1.0` while off-diagonal tokens stay near `0.0`.
    """
    d_mod = e_base.shape[1]
    enorm, _, _ = _rmsnorm_fwd(e_base, g_attn)
    ec_ctx = enorm - np.mean(enorm, axis=0, keepdims=True)
    ec_ctx_dual = (ec_ctx / np.maximum(np.sum(ec_ctx * ec_ctx, axis=1, keepdims=True), 1e-4)).astype(np.float32)

    ec_val = e_base - np.mean(e_base, axis=0, keepdims=True)
    ec_val_u = (ec_val / np.maximum(np.sqrt(np.mean(ec_val * ec_val, axis=1, keepdims=True)), 1e-6)).astype(np.float32)

    sh = pair_indices.shape[:-1]
    fp = pair_indices.reshape(-1, 2)
    valid = (fp[:, 0] >= 0) & (fp[:, 1] >= 0)
    sa = np.maximum(fp[:, 0], 0)
    sb = np.maximum(fp[:, 1], 0)
    out = np.zeros((fp.shape[0], 2 * d_mod), dtype=np.float32)
    out[valid, :d_mod] = ec_ctx_dual[sa[valid]]
    out[valid, d_mod:] = ec_val_u[sb[valid]]
    return out.reshape(*sh, 2 * d_mod)


def initialize_biorthogonal_sheaf_bridge(
    target_model: CausalTransformerLM,
    source_ckpt: CausalTransformerLM,
    nf4_quantizer: NF4BlockQuantizer | None = None,
) -> set[str]:
    """
    Transplant `source_ckpt` into `target_model` (NF4-quantizing linear backbone blocks) and
    initialize `AsymmetricSheafKVPrefix` in the biorthogonal `Wq^+` / `Wo^+` subspace.
    """
    d = source_ckpt.cfg.d_model
    backbone_keys: set[str] = set()
    for pname, pval in source_ckpt.params.items():
        if nf4_quantizer is not None and pname.startswith("L") and pname.endswith(("Wq", "Wk", "Wv", "Wo", "W1", "W2", "W3")):
            target_model.params[pname] = nf4_quantizer.dequantize(nf4_quantizer.quantize(pval))
        else:
            target_model.params[pname] = pval.copy()
        backbone_keys.add(pname)

    if target_model.cfg.kv_prefix_slots > 0:
        assert target_model.cfg.d_stalk == 2 * d
        dh = d // target_model.cfg.n_heads
        tau_k = 16.0 * math.sqrt(float(dh)) * float(target_model.cfg.n_heads)
        for l_i in range(target_model.cfg.n_layers):
            p = f"L{l_i}."
            wq = target_model.params[p + "Wq"]
            wo = target_model.params[p + "Wo"]
            wq_dual = np.linalg.solve(wq.T @ wq + 1e-4 * np.eye(d, dtype=np.float32), wq.T).T
            wo_pinv = np.linalg.solve(wo @ wo.T + 1e-2 * np.eye(d, dtype=np.float32), wo).T

            w_mk = np.zeros((2 * d, d), dtype=np.float32)
            w_mv = np.zeros((2 * d, d), dtype=np.float32)
            if l_i == target_model.cfg.n_layers - 1:
                w_mk[:d, :] = tau_k * wq_dual
                w_mv[d:, :] = 1.15 * wo_pinv
                target_model.params[p + "alpha_kv"] = np.full((target_model.cfg.n_heads,), 0.65, dtype=np.float32)
            else:
                target_model.params[p + "alpha_kv"] = np.zeros((target_model.cfg.n_heads,), dtype=np.float32)

            target_model.params[p + "W_mk"] = w_mk
            target_model.params[p + "W_mv"] = w_mv
            target_model.params[p + "W_sq"] = np.zeros((d, 2 * d), dtype=np.float32)
            target_model.params[p + "W_se"] = 0.05 * np.eye(2 * d, dtype=np.float32)
            target_model.params[p + "b_s"] = np.full((2 * d,), 4.0, dtype=np.float32)

    return backbone_keys


def run_t41_frontier5070_suite(
    output_dir: Path | None = None,
    seeds: tuple[int, ...] = (13, 21, 34, 55, 89),
) -> dict[str, object]:
    """Execute the 4-part `T-41` Frontier-5070 Empirical & Silicon Verification Suite."""
    t0_all = time.perf_counter()

    # PART 1: Exact 6-tier RTX 5070 12 GB VRAM & L2 Cache Plans (`51M..8B`, `V=32k..262k`)
    vram_plans = [asdict(p) for p in plan_rtx5070_frontier_vram(head_chunk_tokens=256)]
    assert all(bool(p["fits_5070_12gb"]) for p in vram_plans), "A Frontier-5070 tier exceeded 12 GB!"

    all_docs = load_clean_technical_corpus(max_articles=300)
    train_raw = [d for i, d in enumerate(all_docs) if i % 10 not in (8, 9)]
    eval_raw = [d for i, d in enumerate(all_docs) if i % 10 in (8, 9)]
    clean_train_docs, hg_clean = build_decontaminated_hypergraph(
        train_raw, eval_raw, min_edge_size=3, max_edge_size=8, max_doc_degree=4, d_doc=32, seed=13
    )
    hg_null = build_chodrow_null(hg_clean, n_swaps_factor=12, seed=29)

    # PART 2: Blockwise 8-Bit Dynamic AdamW + NF4 + Exact-Normalized Pretrained SVD `lm_head` Surgery (5 Seeds)
    vocab_p2 = 512
    w_len = 24
    bsz = 12
    d_p2 = 48
    ff_p2 = 96
    bpe_p2 = ByteBPETokenizer.train([d.text for d in clean_train_docs[:120]], vocab_size=vocab_p2, min_pair_count=4)
    streams_p2 = [bpe_p2.encode(d.text) for d in clean_train_docs]
    test_ids_p2 = bpe_p2.encode("\n\n".join(d.text for d in eval_raw[len(eval_raw) // 2 :]))

    flat_p2 = np.concatenate(streams_p2)[:40000]
    tok_counts_p2 = np.zeros(vocab_p2, dtype=np.int64)
    np.add.at(tok_counts_p2, flat_p2, 1)
    nw_p2 = (flat_p2.size - 1) // w_len
    xw_p2 = flat_p2[: nw_p2 * w_len].reshape(nw_p2, w_len)
    yw_p2 = flat_p2[1 : nw_p2 * w_len + 1].reshape(nw_p2, w_len)

    nw_ev_p2 = (test_ids_p2.size - 1) // w_len
    x_ev_p2 = test_ids_p2[: nw_ev_p2 * w_len].reshape(nw_ev_p2, w_len)
    y_ev_p2 = test_ids_p2[1 : nw_ev_p2 * w_len + 1].reshape(nw_ev_p2, w_len)
    ev_bytes_p2 = float(np.sum(bpe_p2.byte_lengths[y_ev_p2]))

    bpb_fp32_adamw: list[float] = []
    bpb_8bit_adamw: list[float] = []
    bpb_fp2_step0_svd: list[float] = []
    bpb_fp2_step0_rand: list[float] = []
    bpb_nf4_cluster_is_svd: list[float] = []
    bpb_nf4_cluster_is_rand: list[float] = []
    nf4_bits_list: list[float] = []
    opt_compression_ratios: list[float] = []

    nf4_q = NF4BlockQuantizer(block_size=64)
    n_steps_p2 = 68

    for seed in seeds:
        sched_rng = np.random.default_rng(seed + 5000)
        sched = [sched_rng.integers(0, nw_p2, size=bsz) for _ in range(n_steps_p2)]

        cfg_base = TransformerConfig(
            vocab_size=vocab_p2, d_model=d_p2, n_heads=4, n_layers=2, d_ff=ff_p2, window_len=w_len, head_kind="flat"
        )
        mod_fp32 = CausalTransformerLM(cfg_base, seed=seed)
        fp32_state_bytes = sum(int(v.nbytes) * 2 for v in mod_fp32.params.values())

        mod_8bit = CausalTransformerLM(cfg_base, seed=seed)
        opt_8b = Blockwise8BitAdamW(mod_8bit.params, base_width_mup=d_p2, d_model=d_p2, block_size=128)
        opt_compression_ratios.append(fp32_state_bytes / max(float(opt_8b.state_bytes()), 1.0))

        for s_i, idx_b in enumerate(sched):
            lr = _cosine_lr(s_i, n_steps_p2, peak_lr=4.5e-3)
            _, g32 = mod_fp32.forward_backward(xw_p2[idx_b], yw_p2[idx_b], training=True)
            mod_fp32.step_adamw(g32, lr=lr)
            _, g8 = mod_8bit.forward_backward(xw_p2[idx_b], yw_p2[idx_b], training=True)
            opt_8b.step(mod_8bit.params, g8, lr=lr)

        nll32, _ = mod_fp32.forward_backward(x_ev_p2, y_ev_p2, training=False)
        nll8, _ = mod_8bit.forward_backward(x_ev_p2, y_ev_p2, training=False)
        bpb_fp32_adamw.append(float((nll32 * y_ev_p2.size) / (ev_bytes_p2 * math.log(2.0))))
        bpb_8bit_adamw.append(float((nll8 * y_ev_p2.size) / (ev_bytes_p2 * math.log(2.0))))

        part_svd, _ = factorize_pretrained_lm_head(
            mod_fp32.params["E_base"], 32, 16, freq_counts=np.maximum(tok_counts_p2, 1), seed=seed
        )
        part_rnd = build_balanced_partition(
            vocab_p2, 32, 16, [np.arange(vocab_p2, dtype=np.int32)], mode="mass_random", seed=seed
        )

        # Evaluate strictly on normalized class-factored probabilities (`sum_v P(v|h) == 1.0`)!
        h_ev = mod_fp32.encode_hidden(x_ev_p2)
        y_ev_flat = y_ev_p2.reshape(-1)
        zero_rt = np.zeros((32, d_p2), dtype=np.float32)

        for p_obj, step0_log, bpb_log in [
            (part_svd, bpb_fp2_step0_svd, bpb_nf4_cluster_is_svd),
            (part_rnd, bpb_fp2_step0_rand, bpb_nf4_cluster_is_rand),
        ]:
            l_s0, _, _, _ = factored_fp2_fwd_bwd(h_ev, mod_fp32.params["E_base"], zero_rt, p_obj, y_ev_flat, compute_grad=False)
            step0_log.append(float(np.sum(l_s0) / (ev_bytes_p2 * math.log(2.0))))

            cfg_surg = TransformerConfig(
                vocab_size=vocab_p2,
                d_model=d_p2,
                n_heads=4,
                n_layers=2,
                d_ff=ff_p2,
                window_len=w_len,
                head_kind="cluster_is",
                k_neg=3,
            )
            mod_surg = CausalTransformerLM(cfg_surg, seed=seed, partition=p_obj)
            frozen_bb = initialize_biorthogonal_sheaf_bridge(mod_surg, mod_fp32, nf4_quantizer=nf4_q)
            if len(nf4_bits_list) < len(seeds):
                qs_sample = nf4_q.quantize(mod_fp32.params["L0.W1"])
                nf4_bits_list.append(float(qs_sample["bits_per_param"]))

            opt_s = Blockwise8BitAdamW(mod_surg.params, base_width_mup=d_p2, d_model=d_p2, block_size=128)
            frz_only_bb = frozen_bb - {"E_base"}
            for s_i in range(24):
                idx_b = sched[(s_i + 20) % len(sched)]
                _, gs = mod_surg.forward_backward(xw_p2[idx_b], yw_p2[idx_b], training=True)
                opt_s.step(mod_surg.params, gs, lr=4.5e-4, grad_clip=0.5, frozen_names=frz_only_bb)

            h_surg = mod_surg.encode_hidden(x_ev_p2)
            l_surg, _, _, _ = factored_fp2_fwd_bwd(
                h_surg, mod_surg.params["E_base"], mod_surg.params["delta_root"], p_obj, y_ev_flat, compute_grad=False
            )
            bpb_log.append(float(np.sum(l_surg) / (ev_bytes_p2 * math.log(2.0))))

    # PART 3: Frontier `head_dim = 64` RoPE-Decoupled Biorthogonal Sheaf `KV-Prefix` + Symmetric `LP-FT` (5 Seeds)
    vocab_g2 = 1536
    d_g2 = 64
    ff_g2 = 128
    m_slots = 32
    bpe_g2 = ByteBPETokenizer.train([d.text for d in clean_train_docs[:150]], vocab_size=vocab_g2, min_pair_count=3)
    streams_g2 = [bpe_g2.encode(d.text) for d in clean_train_docs]
    flat_g2 = np.concatenate(streams_g2)[:40000]
    nw_g2 = (flat_g2.size - 1) // w_len
    xw_g2 = flat_g2[: nw_g2 * w_len].reshape(nw_g2, w_len)
    yw_g2 = flat_g2[1 : nw_g2 * w_len + 1].reshape(nw_g2, w_len)

    doc_freq = np.zeros(vocab_g2, dtype=np.int32)
    for seq in streams_g2:
        p_arr = np.zeros(vocab_g2, dtype=bool)
        p_arr[seq] = True
        doc_freq += p_arr
    domain_mask = (doc_freq >= 3) & (doc_freq <= int(0.25 * len(streams_g2)))
    ctx_nonstop_mask = doc_freq <= int(0.45 * len(streams_g2))
    idf_tok = np.where(domain_mask, np.log(1.0 + len(streams_g2) / np.maximum(doc_freq.astype(np.float32), 1.0)), 0.0)

    _, de_cl = compute_self_excluded_stalks(hg_clean, hg_clean.doc_vectors)
    _, de_nu = compute_self_excluded_stalks(hg_null, hg_null.doc_vectors)
    valid_docs = [d_i for d_i in range(hg_clean.n_docs) if len(de_cl[d_i]) >= 1 and streams_g2[d_i].size >= 2 * w_len + 2]

    doc_pair_sets: list[set[tuple[int, int]]] = []
    for seq in streams_g2:
        a, b = seq[:-1], seq[1:]
        keep = ctx_nonstop_mask[a] & domain_mask[b]
        doc_pair_sets.append(set(zip(a[keep].tolist(), b[keep].tolist())))

    # Bitwise self-inclusion canary test!
    canary_doc = valid_docs[0]
    p_before = select_consensus_peer_domain_pairs(canary_doc, hg_clean, de_cl, doc_pair_sets, idf_tok, m_slots=m_slots)
    mut_sets = list(doc_pair_sets)
    mut_sets[canary_doc] = {(0, 1), (2, 3), (4, 5)}
    p_after = select_consensus_peer_domain_pairs(canary_doc, hg_clean, de_cl, mut_sets, idf_tok, m_slots=m_slots)
    bipartite_self_leak_drift = float(np.max(np.abs(p_before - p_after)))
    assert bipartite_self_leak_drift == 0.0, "Self-inclusion leak detected!"

    xs_g2, ys_g2, pcl_list, pnu_list = [], [], [], []
    hits_clean, hits_null, total_dom_targets = 0, 0, 0
    for d_i in valid_docs:
        qw = streams_g2[d_i][w_len : 2 * w_len + 1]
        xs_g2.append(qw[:w_len])
        ys_g2.append(qw[1 : w_len + 1])
        q_dom = [(int(qw[k]), int(qw[k + 1])) for k in range(w_len) if ctx_nonstop_mask[qw[k]] and domain_mask[qw[k + 1]]]
        total_dom_targets += len(q_dom)
        arr_cl = select_consensus_peer_domain_pairs(d_i, hg_clean, de_cl, doc_pair_sets, idf_tok, m_slots=m_slots)
        arr_nu = select_consensus_peer_domain_pairs(d_i, hg_null, de_nu, doc_pair_sets, idf_tok, m_slots=m_slots)
        pcl_list.append(arr_cl)
        pnu_list.append(arr_nu)
        set_cl = {(int(arr_cl[s, 0]), int(arr_cl[s, 1])) for s in range(m_slots) if arr_cl[s, 0] >= 0}
        set_nu = {(int(arr_nu[s, 0]), int(arr_nu[s, 1])) for s in range(m_slots) if arr_nu[s, 0] >= 0}
        hits_clean += sum(1 for pr in q_dom if pr in set_cl)
        hits_null += sum(1 for pr in q_dom if pr in set_nu)

    xs_a = np.asarray(xs_g2, dtype=np.int32)
    ys_a = np.asarray(ys_g2, dtype=np.int32)
    pcl_a = np.asarray(pcl_list, dtype=np.int32)
    pnu_a = np.asarray(pnu_list, dtype=np.int32)
    sp = int(0.35 * len(valid_docs))
    gx_tr, gy_tr, gx_ev, gy_ev = xs_a[:sp], ys_a[:sp], xs_a[sp:], ys_a[sp:]

    dom_bpb_isolated: list[float] = []
    dom_bpb_chodrow_lpft: list[float] = []
    dom_bpb_sheaf_lpft: list[float] = []
    all_bpb_isolated: list[float] = []
    all_bpb_chodrow_lpft: list[float] = []
    all_bpb_sheaf_lpft: list[float] = []

    for seed in seeds:
        sr = np.random.default_rng(seed + 5000)
        sched_g2 = [sr.integers(0, nw_g2, size=bsz) for _ in range(56)]
        base_g2 = CausalTransformerLM(
            TransformerConfig(
                vocab_size=vocab_g2, d_model=d_g2, n_heads=1, n_layers=1, d_ff=ff_g2, window_len=w_len, head_kind="flat"
            ),
            seed=seed,
        )
        base_g2.params["g_final"][:] = 2.5
        for s_i, ib in enumerate(sched_g2):
            _, g = base_g2.forward_backward(xw_g2[ib], yw_g2[ib], training=True)
            base_g2.step_adamw(g, lr=_cosine_lr(s_i, 56, 5.0e-3))

        eb_g2 = base_g2.params["E_base"]
        ga_g2 = base_g2.params["L0.g_attn"]
        st_cl_tr = embed_cone_centered_bipartite_stalks(pcl_a[:sp], eb_g2, ga_g2)
        st_cl_ev = embed_cone_centered_bipartite_stalks(pcl_a[sp:], eb_g2, ga_g2)
        st_nu_tr = embed_cone_centered_bipartite_stalks(pnu_a[:sp], eb_g2, ga_g2)
        st_nu_ev = embed_cone_centered_bipartite_stalks(pnu_a[sp:], eb_g2, ga_g2)

        gr = np.random.default_rng(seed + 7000)
        g_sched = [gr.integers(0, gx_tr.shape[0], size=bsz) for _ in range(32)]
        for (all_log, dom_log), tr_s, ev_s in [
            ((all_bpb_isolated, dom_bpb_isolated), None, None),
            ((all_bpb_chodrow_lpft, dom_bpb_chodrow_lpft), st_nu_tr, st_nu_ev),
            ((all_bpb_sheaf_lpft, dom_bpb_sheaf_lpft), st_cl_tr, st_cl_ev),
        ]:
            cfg_g = TransformerConfig(
                vocab_size=vocab_g2,
                d_model=d_g2,
                n_heads=1,
                n_layers=1,
                d_ff=ff_g2,
                window_len=w_len,
                base_width_mup=d_g2,
                head_kind="flat",
                kv_prefix_slots=m_slots if tr_s is not None else 0,
                d_stalk=2 * d_g2,
                mem_score_bias=-10.5,
            )
            mg = CausalTransformerLM(cfg_g, seed=seed)
            bb_k = initialize_biorthogonal_sheaf_bridge(mg, base_g2, nf4_quantizer=nf4_q)
            opt_g = Blockwise8BitAdamW(mg.params, base_width_mup=d_g2, d_model=d_g2, block_size=128)
            for s_i, ib in enumerate(g_sched):
                ps = tr_s[ib] if tr_s is not None else None
                if s_i < 24:
                    if tr_s is not None:
                        _, gg = mg.forward_backward(gx_tr[ib], gy_tr[ib], prefix_stalks=ps, training=True)
                        opt_g.step(mg.params, gg, lr=1.5e-3, grad_clip=0.5, frozen_names=bb_k)
                else:
                    _, gg = mg.forward_backward(gx_tr[ib], gy_tr[ib], prefix_stalks=ps, training=True)
                    opt_g.step(mg.params, gg, lr=2.5e-4, grad_clip=0.5)

            hf = mg.encode_hidden(gx_ev, prefix_stalks=ev_s)
            yf = gy_ev.reshape(-1)
            losses, _, _ = flat_tied_fwd_bwd(hf, mg.full_embedding_table(), yf, compute_grad=False)
            all_b = float(np.sum(bpe_g2.byte_lengths[yf])) * math.log(2.0)
            dm = domain_mask[yf]
            dom_b = float(np.sum(bpe_g2.byte_lengths[yf[dm]])) * math.log(2.0)
            all_log.append(float(np.sum(losses) / all_b))
            dom_log.append(float(np.sum(losses[dm]) / dom_b))

    # PART 4: True Gemma-Class `V = 262,144` (`2048 x 128`) Native C11 AVX-512 Silicon Benchmark
    hw_262k: dict[str, object] = {}
    if get_fast_head_lib() is not None:
        rng_f = np.random.default_rng(2026)
        n_tok_f = 256
        d_f = 576
        v_f = 262144
        nc_f = 2048
        s_f = 128
        k_f = 3

        H_f = (rng_f.standard_normal((n_tok_f, d_f)) * 0.08).astype(np.float32)
        E_f = (rng_f.standard_normal((v_f, d_f)) * 0.08).astype(np.float32)
        E_cl_f = E_f.reshape(nc_f, s_f, d_f)
        C_rt_f = E_cl_f.mean(axis=1).copy()
        tc_f = rng_f.integers(0, nc_f, size=n_tok_f, dtype=np.int32)
        tm_f = rng_f.integers(0, s_f, size=n_tok_f, dtype=np.int32)
        neg_f = rng_f.integers(0, nc_f, size=(n_tok_f, k_f), dtype=np.int32)
        y_f = tc_f * s_f + tm_f

        _ = run_cluster_is_avx512(H_f, E_cl_f, C_rt_f, tc_f, tm_f, neg_f, compute_grad=True)
        t_is0 = time.perf_counter()
        loss_is_262k, _, _ = run_cluster_is_avx512(H_f, E_cl_f, C_rt_f, tc_f, tm_f, neg_f, compute_grad=True)
        ms_is_262k = (time.perf_counter() - t_is0) * 1000.0

        t_cce0 = time.perf_counter()
        loss_cce_sub, _, _ = run_tiled_cce_avx512(H_f[:32], E_f, y_f[:32], tile_V=512, compute_grad=True)
        ms_cce_sub = (time.perf_counter() - t_cce0) * 1000.0
        ms_cce_262k_equiv = ms_cce_sub * (float(n_tok_f) / 32.0)

        loss_fp64_sub, _, _ = tiled_cce_fwd_bwd(
            H_f[:8], E_f, y_f[:8], tile_V=512, compute_grad=False, use_native=False
        )
        fp64_err_262k = float(np.max(np.abs(loss_cce_sub[:8] - loss_fp64_sub)))

        t_svd0 = time.perf_counter()
        part_16k, _ = factorize_pretrained_lm_head(E_f[:16384, :256], 128, 128, rank_svd=16, seed=13)
        svd_16k_ms = (time.perf_counter() - t_svd0) * 1000.0

        hw_262k = {
            "vocab_size": v_f,
            "cluster_shape": f"{nc_f}x{s_f}",
            "d_model": d_f,
            "N_tokens": n_tok_f,
            "avx512_cluster_is_ms": round(ms_is_262k, 3),
            "avx512_tiled_cce_equiv_ms": round(ms_cce_262k_equiv, 3),
            "native_c_speedup_x": round(ms_cce_262k_equiv / max(ms_is_262k, 1e-6), 2),
            "dense_8k_bf16_logit_plus_grad_gib": round((8192 * v_f * 4.0) / (1024.0**3), 3),
            "cluster_is_8k_bf16_logit_plus_grad_gib": round((8192 * (nc_f + 4 * s_f) * 4.0) / (1024.0**3), 4),
            "head_logit_memory_reduction_x": round(float(v_f) / float(nc_f + 4 * s_f), 2),
            "cce_262k_vs_fp64_max_err": round(fp64_err_262k, 8),
            "cluster_is_mean_nll": round(float(np.mean(loss_is_262k)), 4),
            "offline_svd_16k_bijection_ok": part_16k.verify_bijection()["bijection_ok"],
            "offline_svd_16k_ms": round(svd_16k_ms, 1),
        }

    elapsed = time.perf_counter() - t0_all
    report: dict[str, object] = {
        "schema": "sheaf-verify.frontier-5070.v1",
        "version": "0.4.0",
        "seeds": list(seeds),
        "elapsed_seconds": round(elapsed, 2),
        "rtx5070_vram_plans": vram_plans,
        "part2_quantized_surgery": {
            "mean_nf4_bits_per_param": round(float(np.mean(nf4_bits_list)), 4),
            "adamw_8bit_moment_compression_x": round(float(np.mean(opt_compression_ratios)), 3),
            "per_seed_bpb": {
                "fp32_adamw": [round(x, 6) for x in bpb_fp32_adamw],
                "blockwise_8bit_adamw": [round(x, 6) for x in bpb_8bit_adamw],
                "fp2_step0_random": [round(x, 6) for x in bpb_fp2_step0_rand],
                "fp2_step0_pretrained_svd": [round(x, 6) for x in bpb_fp2_step0_svd],
                "nf4_cluster_is_random": [round(x, 6) for x in bpb_nf4_cluster_is_rand],
                "nf4_cluster_is_pretrained_svd": [round(x, 6) for x in bpb_nf4_cluster_is_svd],
            },
            "adamw_8bit_vs_fp32": _adjudicate_paired(bpb_fp32_adamw, bpb_8bit_adamw),
            "fp2_step0_svd_vs_random": _adjudicate_paired(bpb_fp2_step0_rand, bpb_fp2_step0_svd),
            "nf4_svd_vs_random_cluster_is": _adjudicate_paired(bpb_nf4_cluster_is_rand, bpb_nf4_cluster_is_svd),
        },
        "part3_learned_stalk_lpft_g2": {
            "bipartite_self_inclusion_drift": bipartite_self_leak_drift,
            "rope_decoupled_content_addressing": True,
            "head_dim": d_g2,
            "peer_consensus_recall_clean_pct": round(100.0 * hits_clean / max(total_dom_targets, 1), 2),
            "peer_consensus_recall_chodrow_null_pct": round(100.0 * hits_null / max(total_dom_targets, 1), 2),
            "per_seed_domain_bpb": {
                "window_isolated_nf4": [round(x, 6) for x in dom_bpb_isolated],
                "kvreg_lpft_chodrow_null": [round(x, 6) for x in dom_bpb_chodrow_lpft],
                "kvreg_lpft_diagonal_sheaf": [round(x, 6) for x in dom_bpb_sheaf_lpft],
            },
            "per_seed_all_token_bpb": {
                "window_isolated_nf4": [round(x, 6) for x in all_bpb_isolated],
                "kvreg_lpft_chodrow_null": [round(x, 6) for x in all_bpb_chodrow_lpft],
                "kvreg_lpft_diagonal_sheaf": [round(x, 6) for x in all_bpb_sheaf_lpft],
            },
            "sheaf_lpft_vs_chodrow_null_domain": _adjudicate_paired(dom_bpb_chodrow_lpft, dom_bpb_sheaf_lpft),
            "sheaf_lpft_vs_window_isolated_domain": _adjudicate_paired(dom_bpb_isolated, dom_bpb_sheaf_lpft),
            "sheaf_lpft_vs_window_isolated_all": _adjudicate_paired(all_bpb_isolated, all_bpb_sheaf_lpft),
        },
        "part4_frontier_262k_avx512": hw_262k,
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "T41_FRONTIER5070_RESULTS.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report


RTX_5070_SPECS: dict[str, Any] = {
    "gpu": "NVIDIA GeForce RTX 5070 (Blackwell GB205)",
    "sms": 48,
    "cuda_cores": 6144,
    "tensor_cores_5th_gen": 192,
    "rt_cores_4th_gen": 48,
    "boost_clock_ghz": 2.512,
    "ai_tops_fp4_sparse": 988.0,
    "fp8_dense_tflops": 247.0,
    "fp32_int32_tflops": 30.87,
    "register_file_total_kb": 12288,
    "l1_shared_sram_total_kb": 6144,
    "l2_cache_total_kb": 49152,
    "vram_gddr7_gb": 12.0,
    "vram_bw_gbs": 672.0,
    "tgp_watts": 250.0,
}

_CU_SOURCE_PATH = Path(__file__).resolve().parent / "csrc" / "c0_rtx5070_sheaf_tuple.cu"
CUDA_RTX5070_KERNEL_SOURCE: str = _CU_SOURCE_PATH.read_text(encoding="utf-8")
C0_RTX5070_CUDA_SOURCE: str = CUDA_RTX5070_KERNEL_SOURCE
_CUDA_LIB_CACHE: dict[tuple[str, str, str], ctypes.CDLL | None] = {}


def export_cuda_kernel_source(dest_path: str | Path | None = None) -> str:
    """Return (and optionally write) the single-source RTX 5070 sm_120 CUDA C++ kernel."""
    if dest_path is not None:
        out = Path(dest_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(CUDA_RTX5070_KERNEL_SOURCE, encoding="utf-8")
    return CUDA_RTX5070_KERNEL_SOURCE


export_c0_rtx5070_cuda_kernel = export_cuda_kernel_source


def _qmul_f32(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    pw, px, py, pz = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
    qw, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    ow = pw * qw - px * qx - py * qy - pz * qz
    ox = pw * qx + px * qw + py * qz - pz * qy
    oy = pw * qy - px * qz + py * qw + pz * qx
    oz = pw * qz + px * qy - py * qx + pz * qw
    return np.stack([ow, ox, oy, oz], axis=-1).astype(np.float32)


def _validate_warp_inputs(
    c_H_in: np.ndarray | Sequence[Sequence[float]],
    gamma_sheaf: float,
) -> tuple[np.ndarray, float]:
    c = np.asarray(c_H_in, dtype=np.float32)
    if c.ndim != 2 or c.shape[1] != 4:
        raise ValueError(f"c_H_in must have shape (N, 4), got {c.shape}")
    if not np.all(np.isfinite(c)):
        raise ValueError("c_H_in must contain only finite values")
    gamma_f = float(gamma_sheaf)
    if not math.isfinite(gamma_f):
        raise ValueError("gamma_sheaf must be finite")
    return c, gamma_f


def c0_rtx5070_warp_julia_and_tuple_cpu(
    c_H_in: np.ndarray | Sequence[Sequence[float]],
    gamma_sheaf: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Emulate ``c0_rtx5070_warp_julia_and_tuple_kernel`` in strict ``np.float32`` with 32-lane warp resets."""
    c, gamma_f = _validate_warp_inputs(c_H_in, gamma_sheaf)
    n = int(c.shape[0])
    if n == 0:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

    gamma = np.float32(gamma_f)
    q = c.copy()
    lanes = np.arange(n, dtype=np.int32) & 31
    for offset in (1, 2, 4, 8, 16):
        prev = np.zeros_like(q)
        if offset < n:
            prev[offset:] = q[:-offset]
        mask = lanes >= offset
        updated = _qmul_f32(prev, q)
        q = np.where(mask[:, None], updated, q)

    c0 = c[..., 0]
    c1 = c[..., 1]
    c2 = c[..., 2]
    c3 = c[..., 3]
    two = np.float32(2.0)
    quarter = np.float32(0.25)
    one = np.float32(1.0)
    three = np.float32(3.0)

    for _ in range(4):
        sq = _qmul_f32(q, q)
        qx = q[..., 1]
        qy = q[..., 2]
        qz = q[..., 3]
        lx = two * (qy * c3 - qz * c2)
        ly = two * (qz * c1 - qx * c3)
        lz = two * (qx * c2 - qy * c1)

        nw = sq[..., 0] + c0
        nx = sq[..., 1] + c1 + gamma * lx
        ny = sq[..., 2] + c2 + gamma * ly
        nz = sq[..., 3] + c3 + gamma * lz

        msq = quarter * (nw * nw + nx * nx + ny * ny + nz * nz)
        gain = (three + msq) / (one + three * msq)
        q = np.stack([nw * gain, nx * gain, ny * gain, nz * gain], axis=-1).astype(np.float32)

    p5 = np.sqrt(q[..., 2] ** 2 + q[..., 3] ** 2).astype(np.float32)
    return q, p5


emulate_warp_julia_and_tuple_cpu = c0_rtx5070_warp_julia_and_tuple_cpu
emulate_c0_rtx5070_warp_julia_kernel = c0_rtx5070_warp_julia_and_tuple_cpu
cpu_warp_julia_tuple_parity = c0_rtx5070_warp_julia_and_tuple_cpu
run_sm120_warp_julia_cpu_ref = c0_rtx5070_warp_julia_and_tuple_cpu


def _scalar_lane_warp_julia_ref(c: np.ndarray, gamma_f: float) -> tuple[np.ndarray, np.ndarray]:
    n = int(c.shape[0])
    q_ref = np.empty((n, 4), dtype=np.float32)
    p5_ref = np.empty((n,), dtype=np.float32)
    f32 = np.float32
    gamma = f32(gamma_f)

    for w_start in range(0, n, 32):
        w_end = min(w_start + 32, n)
        w_len = w_end - w_start
        warp_q = [tuple(f32(v) for v in c[w_start + idx]) for idx in range(w_len)]
        for offset in (1, 2, 4, 8, 16):
            next_q = list(warp_q)
            for lane in range(offset, w_len):
                pw, px, py, pz = warp_q[lane - offset]
                qw, qx, qy, qz = warp_q[lane]
                next_q[lane] = (
                    f32(pw * qw - px * qx - py * qy - pz * qz),
                    f32(pw * qx + px * qw + py * qz - pz * qy),
                    f32(pw * qy - px * qz + py * qw + pz * qx),
                    f32(pw * qz + px * qy - py * qx + pz * qw),
                )
            warp_q = next_q

        for lane in range(w_len):
            qw, qx, qy, qz = warp_q[lane]
            cw, cx, cy, cz = (f32(v) for v in c[w_start + lane])
            for _ in range(4):
                sqw = f32(qw * qw - qx * qx - qy * qy - qz * qz)
                sqx = f32(qw * qx + qx * qw + qy * qz - qz * qy)
                sqy = f32(qw * qy - qx * qz + qy * qw + qz * qx)
                sqz = f32(qw * qz + qx * qy - qy * qx + qz * qw)

                lx = f32(f32(2.0) * f32(qy * cz - qz * cy))
                ly = f32(f32(2.0) * f32(qz * cx - qx * cz))
                lz = f32(f32(2.0) * f32(qx * cy - qy * cx))

                nw = f32(sqw + cw)
                nx = f32(sqx + cx + gamma * lx)
                ny = f32(sqy + cy + gamma * ly)
                nz = f32(sqz + cz + gamma * lz)

                msq = f32(f32(0.25) * f32(nw * nw + nx * nx + ny * ny + nz * nz))
                gain = f32(f32(f32(3.0) + msq) / f32(f32(1.0) + f32(3.0) * msq))
                qw = f32(nw * gain)
                qx = f32(nx * gain)
                qy = f32(ny * gain)
                qz = f32(nz * gain)

            q_ref[w_start + lane] = (qw, qx, qy, qz)
            p5_ref[w_start + lane] = f32(math.sqrt(float(qy * qy + qz * qz)))

    return q_ref, p5_ref


def verify_c0_rtx5070_cpu_parity(
    c_H_in: np.ndarray | Sequence[Sequence[float]],
    gamma_sheaf: float = 0.25,
    *,
    reference_q: np.ndarray | None = None,
    reference_p5: np.ndarray | None = None,
    atol: float = 1e-5,
) -> dict[str, Any]:
    """Verify 32-lane float32 warp-Julia CPU parity against provided or scalar-lane reference."""
    c_arr, gamma_f = _validate_warp_inputs(c_H_in, gamma_sheaf)
    q_out, p5_out = c0_rtx5070_warp_julia_and_tuple_cpu(c_arr, gamma_f)
    n = int(c_arr.shape[0])

    if reference_q is not None and reference_p5 is not None:
        ref_q = np.asarray(reference_q, dtype=np.float32)
        ref_p5 = np.asarray(reference_p5, dtype=np.float32)
    else:
        ref_q, ref_p5 = _scalar_lane_warp_julia_ref(c_arr, gamma_f)

    if n == 0:
        max_q_err = 0.0
        max_p5_err = 0.0
    else:
        max_q_err = float(np.max(np.abs(q_out - ref_q)))
        max_p5_err = float(np.max(np.abs(p5_out - ref_p5)))

    ok = bool(
        np.all(np.isfinite(q_out))
        and np.all(np.isfinite(p5_out))
        and max_q_err <= atol
        and max_p5_err <= atol
    )
    return {
        "ok": ok,
        "passed": ok,
        "q_out": q_out,
        "p5_jk_rotor_out": p5_out,
        "max_q_err": max_q_err,
        "max_p5_err": max_p5_err,
        "n_elements": n,
        "n_warps": (n + 31) // 32,
        "backend": "cpu_parity",
    }


verify_sm120_warp_julia_cpu_parity = verify_c0_rtx5070_cpu_parity


def discover_nvcc() -> str | None:
    """Locate an ``nvcc`` compiler binary if available on the host."""
    env_nvcc = os.environ.get("NVCC", "").strip()
    if env_nvcc:
        found_env = shutil.which(env_nvcc) or (env_nvcc if Path(env_nvcc).is_file() else None)
        if found_env:
            return found_env
    which_nvcc = shutil.which("nvcc")
    if which_nvcc:
        return which_nvcc
    default_cuda = Path("/usr/local/cuda/bin/nvcc")
    if default_cuda.is_file():
        return str(default_cuda)
    return None


def compile_c0_rtx5070_cuda_kernel(
    arch: str = "sm_120",
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    """Lazily compile ``c0_rtx5070_sheaf_tuple.cu`` into ``$HOME/.cache/hadukenn_core`` when ``nvcc`` exists."""
    nvcc_bin = discover_nvcc()
    if nvcc_bin is None:
        return {
            "available": False,
            "compiled": False,
            "reason": "nvcc_not_found",
            "arch": arch,
            "source_path": str(_CU_SOURCE_PATH),
            "library_path": None,
        }

    target_dir = cache_dir or (Path.home() / ".cache" / "hadukenn_core")
    target_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(f"{CUDA_RTX5070_KERNEL_SOURCE}:{arch}:{nvcc_bin}".encode()).hexdigest()[:16]
    so_path = target_dir / f"c0_rtx5070_{digest}_{arch}.so"
    if so_path.is_file():
        return {
            "available": True,
            "compiled": True,
            "reason": "cached",
            "arch": arch,
            "source_path": str(_CU_SOURCE_PATH),
            "library_path": str(so_path),
        }

    tmp_so = target_dir / f".tmp_c0_rtx5070_{digest}_{os.getpid()}.so"
    cmd = [
        nvcc_bin,
        "-O3",
        f"-arch={arch}",
        "-Xcompiler",
        "-fPIC",
        "-shared",
        str(_CU_SOURCE_PATH),
        "-o",
        str(tmp_so),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        if tmp_so.exists():
            tmp_so.unlink()
        return {
            "available": True,
            "compiled": False,
            "reason": f"nvcc_exec_error: {exc}",
            "arch": arch,
            "source_path": str(_CU_SOURCE_PATH),
            "library_path": None,
        }

    if proc.returncode != 0 or not tmp_so.is_file():
        if tmp_so.exists():
            tmp_so.unlink()
        return {
            "available": True,
            "compiled": False,
            "reason": f"nvcc_failed_{proc.returncode}",
            "arch": arch,
            "source_path": str(_CU_SOURCE_PATH),
            "library_path": None,
        }

    os.replace(tmp_so, so_path)
    return {
        "available": True,
        "compiled": True,
        "reason": "compiled",
        "arch": arch,
        "source_path": str(_CU_SOURCE_PATH),
        "library_path": str(so_path),
    }


def run_c0_rtx5070_cuda_if_available(
    c_H_in: np.ndarray | Sequence[Sequence[float]],
    gamma_sheaf: float = 0.25,
    *,
    arch: str = "sm_120",
) -> tuple[np.ndarray, np.ndarray, str]:
    """Run ``c0_rtx5070_warp_julia_host`` on GPU when available, else return CPU warp parity."""
    c_arr, gamma_f = _validate_warp_inputs(c_H_in, gamma_sheaf)
    if (
        os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        or os.environ.get("NVIDIA_VISIBLE_DEVICES") == "void"
        or not Path("/dev/nvidiactl").exists()
        or discover_nvcc() is None
    ):
        q_cpu, p5_cpu = c0_rtx5070_warp_julia_and_tuple_cpu(c_arr, gamma_f)
        return q_cpu, p5_cpu, "cpu_parity"

    comp = compile_c0_rtx5070_cuda_kernel(arch=arch)
    lib_path = comp.get("library_path")
    if not comp.get("compiled") or not isinstance(lib_path, str):
        q_cpu, p5_cpu = c0_rtx5070_warp_julia_and_tuple_cpu(c_arr, gamma_f)
        return q_cpu, p5_cpu, "cpu_parity"

    cache_key = (lib_path, arch, str(_CU_SOURCE_PATH))
    lib = _CUDA_LIB_CACHE.get(cache_key)
    if lib is None and cache_key not in _CUDA_LIB_CACHE:
        try:
            loaded = ctypes.CDLL(lib_path)
            host_fn = loaded.c0_rtx5070_warp_julia_host
            host_fn.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_float,
                ctypes.c_int,
            ]
            host_fn.restype = ctypes.c_int
            lib = loaded
        except OSError:
            lib = None
        _CUDA_LIB_CACHE[cache_key] = lib

    if lib is None:
        q_cpu, p5_cpu = c0_rtx5070_warp_julia_and_tuple_cpu(c_arr, gamma_f)
        return q_cpu, p5_cpu, "cpu_parity"

    n = int(c_arr.shape[0])
    c_contig = np.ascontiguousarray(c_arr, dtype=np.float32)
    q_out = np.empty((n, 4), dtype=np.float32)
    p5_out = np.empty((n,), dtype=np.float32)
    rc = int(
        lib.c0_rtx5070_warp_julia_host(
            c_contig.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            q_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            p5_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_float(gamma_f),
            ctypes.c_int(n),
        )
    )
    if rc == 0:
        return q_out, p5_out, f"cuda_{arch}"

    q_cpu, p5_cpu = c0_rtx5070_warp_julia_and_tuple_cpu(c_arr, gamma_f)
    return q_cpu, p5_cpu, "cpu_parity"


if __name__ == "__main__":  # pragma: no cover
    out = Path(__file__).resolve().parents[2] / "iterations" / "t41-frontier-5070"
    res = run_t41_frontier5070_suite(output_dir=out)
    print(json.dumps(res, indent=2))
