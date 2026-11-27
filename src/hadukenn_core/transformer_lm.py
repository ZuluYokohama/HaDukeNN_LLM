"""Asymmetric Diagonal-Sheaf KV-Prefix Registers (`Q = W, KV = W + M_prefix`) and Causal RoPE-SwiGLU Transformer LM.

Decouples positionless hypergraph sheaf content addressing (`q_unrot @ k_mem^T + mem_score_bias`)
from within-window relative positional attention (`RoPE(t)q @ (RoPE(s)k_seq)^T`) so prefix memory
keys never suffer position-dependent RoPE sign-flipping across sequence offsets `t = 0 .. W - 1`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from hadukenn_core.heads import (
    CountShrunkSpanComposer,
    TiedRootShortcutLayout,
    cluster_is_fwd_bwd,
    factored_fp2_fwd_bwd,
    fast_scatter_add_2d,
    flat_tied_fwd_bwd,
    tied_root_shortcut_fwd_bwd,
    tiled_cce_fwd_bwd,
)
from hadukenn_core.partitions import ClusterPartition


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _silu(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sig = _sigmoid(x)
    return x * sig, sig


def _build_rope_cache(seq_len: int, head_dim: int, base: float = 10000.0) -> tuple[np.ndarray, np.ndarray]:
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (np.arange(0, half, dtype=np.float32) / float(half)))
    pos = np.arange(seq_len, dtype=np.float32)
    freqs = np.outer(pos, inv_freq)
    return np.cos(freqs), np.sin(freqs)


def _apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    d_half = x.shape[-1] // 2
    x1 = x[..., :d_half]
    x2 = x[..., d_half:]
    c = cos[None, None, :, :]
    s = sin[None, None, :, :]
    return np.concatenate([x1 * c - x2 * s, x1 * s + x2 * c], axis=-1)


def _apply_rope_bwd(dy: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    d_half = dy.shape[-1] // 2
    dy1 = dy[..., :d_half]
    dy2 = dy[..., d_half:]
    c = cos[None, None, :, :]
    s = sin[None, None, :, :]
    return np.concatenate([dy1 * c + dy2 * s, -dy1 * s + dy2 * c], axis=-1)


def _rmsnorm_fwd(x: np.ndarray, g: np.ndarray, eps: float = 1e-5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ms = np.mean(x * x, axis=-1, keepdims=True)
    inv_rms = 1.0 / np.sqrt(ms + eps)
    x_hat = x * inv_rms
    return x_hat * g, x_hat, inv_rms


def _rmsnorm_bwd(
    dy: np.ndarray,
    x_hat: np.ndarray,
    inv_rms: np.ndarray,
    g: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    dg = np.sum(dy * x_hat, axis=tuple(range(dy.ndim - 1)))
    dx_hat = dy * g
    mean_proj = np.mean(dx_hat * x_hat, axis=-1, keepdims=True)
    dx = inv_rms * (dx_hat - x_hat * mean_proj)
    return dx, dg


@dataclass
class TransformerConfig:
    vocab_size: int = 1024
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 128
    window_len: int = 32
    base_width_mup: int = 64
    head_kind: str = "flat"  # 'flat', 'tiled_cce', 'fp2', 'cluster_is', 'tied_root'
    k_neg: int = 3
    use_oe_span: bool = False
    n_spans: int = 0
    d_span: int = 32
    kv_prefix_slots: int = 0
    d_stalk: int = 32
    trime_local_drop: float = 0.0
    mem_score_bias: float = 0.0


class CausalTransformerLM:
    """Analytically backpropagated Causal RoPE + RMSNorm + SwiGLU Transformer LM using pure BLAS `@`."""

    def __init__(
        self,
        cfg: TransformerConfig,
        seed: int = 13,
        partition: ClusterPartition | None = None,
        shortcut_layout: TiedRootShortcutLayout | None = None,
        span_composer: CountShrunkSpanComposer | None = None,
    ) -> None:
        self.cfg = cfg
        self.partition = partition
        self.shortcut_layout = shortcut_layout
        self.span_composer = span_composer
        self.rng = np.random.default_rng(seed)
        d = cfg.d_model
        h = cfg.n_heads
        dh = d // h
        assert d == h * dh, "d_model must be divisible by n_heads"
        self.cos, self.sin = _build_rope_cache(cfg.window_len, dh)

        idx = np.arange(cfg.window_len)
        self.causal_mask = idx[:, None] >= idx[None, :]

        std_emb = 0.04
        std_hid = 1.0 / math.sqrt(float(d))
        std_ff = 1.0 / math.sqrt(float(cfg.d_ff))

        self.params: dict[str, np.ndarray] = {}
        v_base = cfg.vocab_size - (len(span_composer.span_parts) if span_composer is not None else 0)
        self.v_base = v_base
        self.params["E_base"] = (self.rng.standard_normal((v_base, d)) * std_emb).astype(np.float32)

        if span_composer is not None and len(span_composer.span_parts) > 0:
            self.params["delta_span"] = np.zeros((len(span_composer.span_parts), d), dtype=np.float32)

        if cfg.head_kind in {"fp2", "cluster_is"}:
            if partition is None:
                raise ValueError(f"head_kind={cfg.head_kind} requires ClusterPartition")
            self.params["delta_root"] = np.zeros((partition.n_clusters, d), dtype=np.float32)
        elif cfg.head_kind == "tied_root":
            if shortcut_layout is None:
                raise ValueError("head_kind='tied_root' requires TiedRootShortcutLayout")
            self.params["delta_tail"] = np.zeros((shortcut_layout.n_tail_clusters, d), dtype=np.float32)

        if cfg.use_oe_span and cfg.n_spans > 0:
            self.params["E_span"] = (self.rng.standard_normal((cfg.n_spans, cfg.d_span)) * std_emb).astype(np.float32)
            self.params["W_oe"] = (self.rng.standard_normal((cfg.d_span, d)) * (1.0 / math.sqrt(cfg.d_span))).astype(
                np.float32
            )
            self.params["alpha_oe"] = np.full((1,), 0.15, dtype=np.float32)

        for l_i in range(cfg.n_layers):
            p = f"L{l_i}."
            self.params[p + "g_attn"] = np.ones((d,), dtype=np.float32)
            self.params[p + "Wq"] = (self.rng.standard_normal((d, d)) * std_hid).astype(np.float32)
            self.params[p + "Wk"] = (self.rng.standard_normal((d, d)) * std_hid).astype(np.float32)
            self.params[p + "Wv"] = (self.rng.standard_normal((d, d)) * std_hid).astype(np.float32)
            self.params[p + "Wo"] = (
                self.rng.standard_normal((d, d)) * (std_hid / math.sqrt(2.0 * cfg.n_layers))
            ).astype(np.float32)
            self.params[p + "g_mlp"] = np.ones((d,), dtype=np.float32)
            self.params[p + "W1"] = (self.rng.standard_normal((d, cfg.d_ff)) * std_hid).astype(np.float32)
            self.params[p + "W3"] = (self.rng.standard_normal((d, cfg.d_ff)) * std_hid).astype(np.float32)
            self.params[p + "W2"] = (
                self.rng.standard_normal((cfg.d_ff, d)) * (std_ff / math.sqrt(2.0 * cfg.n_layers))
            ).astype(np.float32)

            if cfg.kv_prefix_slots > 0:
                ds = cfg.d_stalk
                self.params[p + "W_sq"] = (self.rng.standard_normal((d, ds)) * std_hid).astype(np.float32)
                self.params[p + "W_se"] = (self.rng.standard_normal((ds, ds)) * (1.0 / math.sqrt(ds))).astype(
                    np.float32
                )
                self.params[p + "b_s"] = np.full((ds,), 0.5, dtype=np.float32)
                self.params[p + "W_mk"] = (self.rng.standard_normal((ds, d)) * (1.0 / math.sqrt(ds))).astype(
                    np.float32
                )
                self.params[p + "W_mv"] = (self.rng.standard_normal((ds, d)) * (1.0 / math.sqrt(ds))).astype(
                    np.float32
                )
                self.params[p + "alpha_kv"] = np.full((h,), 0.15, dtype=np.float32)

        self.params["g_final"] = np.ones((d,), dtype=np.float32)
        self._m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self._v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self._step = 0
        self.last_residual_share: float = 0.0

    def parameter_count(self) -> int:
        return sum(int(v.size) for v in self.params.values())

    def full_embedding_table(self) -> np.ndarray:
        if self.span_composer is not None and "delta_span" in self.params:
            u_full, share = self.span_composer.compose(self.params["E_base"], self.params["delta_span"])
            self.last_residual_share = share
            return u_full
        return self.params["E_base"]

    def encode_hidden(
        self,
        x_ids: np.ndarray,
        span_ids: np.ndarray | None = None,
        prefix_stalks: np.ndarray | None = None,
        prior_win_ctx: np.ndarray | None = None,
    ) -> np.ndarray:
        """Run forward pass through final RMSNorm and return flattened `[B * W, d_model]` states."""
        cfg = self.cfg
        bsz, w_len = x_ids.shape
        d = cfg.d_model
        h_num = cfg.n_heads
        dh = d // h_num
        m_pre = cfg.kv_prefix_slots if prefix_stalks is not None else 0
        cos_w = self.cos[:w_len]
        sin_w = self.sin[:w_len]

        U_table = self.full_embedding_table()
        x = U_table[x_ids].copy()
        if cfg.use_oe_span and cfg.n_spans > 0 and span_ids is not None:
            valid = span_ids >= 0
            safe_s = np.maximum(span_ids, 0)
            s_emb = self.params["E_span"][safe_s] * valid[:, :, None].astype(np.float32)
            x = x + float(np.tanh(self.params["alpha_oe"][0])) * (s_emb @ self.params["W_oe"])

        q_ctx = x[:, 0, :].copy() if prior_win_ctx is None else prior_win_ctx
        scale = np.float32(1.0 / math.sqrt(float(dh)))
        causal_m = self.causal_mask[:w_len, :w_len]

        for l_i in range(cfg.n_layers):
            p = f"L{l_i}."
            x_norm, _, _ = _rmsnorm_fwd(x, self.params[p + "g_attn"])
            q = (x_norm @ self.params[p + "Wq"]).reshape(bsz, w_len, h_num, dh).transpose(0, 2, 1, 3)
            k_seq = (x_norm @ self.params[p + "Wk"]).reshape(bsz, w_len, h_num, dh).transpose(0, 2, 1, 3)
            v_seq = (x_norm @ self.params[p + "Wv"]).reshape(bsz, w_len, h_num, dh).transpose(0, 2, 1, 3)
            q_rot = _apply_rope(q, cos_w, sin_w)
            k_rot = _apply_rope(k_seq, cos_w, sin_w)

            if m_pre > 0 and prefix_stalks is not None:
                gate_pre = (
                    (q_ctx @ self.params[p + "W_sq"])[:, None, :]
                    + (prefix_stalks @ self.params[p + "W_se"])
                    + self.params[p + "b_s"][None, None, :]
                )
                transported = _sigmoid(gate_pre).astype(np.float32) * prefix_stalks
                k_mem = (transported @ self.params[p + "W_mk"]).reshape(bsz, m_pre, h_num, dh).transpose(0, 2, 1, 3)
                v_mem_raw = (transported @ self.params[p + "W_mv"]).reshape(bsz, m_pre, h_num, dh).transpose(0, 2, 1, 3)
                v_mem = v_mem_raw * np.tanh(self.params[p + "alpha_kv"])[None, :, None, None]
                scores_mem = (q @ k_mem.swapaxes(-1, -2)) * scale + np.float32(cfg.mem_score_bias)
                scores_seq = (q_rot @ k_rot.swapaxes(-1, -2)) * scale
                scores = np.concatenate([scores_mem, scores_seq], axis=-1)
                v_all = np.concatenate([v_mem, v_seq], axis=2)
                full_mask = np.ones((bsz, 1, w_len, m_pre + w_len), dtype=bool)
                full_mask[:, :, :, m_pre:] = causal_m[None, None, :, :]
            else:
                scores = (q_rot @ k_rot.swapaxes(-1, -2)) * scale
                v_all = v_seq
                full_mask = np.broadcast_to(causal_m[None, None, :, :], (bsz, 1, w_len, w_len))

            scores_masked = np.where(full_mask, scores, -1e9)
            mx_s = np.max(scores_masked, axis=-1, keepdims=True)
            ex_s = np.where(full_mask, np.exp(scores_masked - mx_s), 0.0)
            attn_probs = (ex_s / np.maximum(np.sum(ex_s, axis=-1, keepdims=True), 1e-12)).astype(np.float32)
            ctx_flat = (attn_probs @ v_all).transpose(0, 2, 1, 3).reshape(bsz, w_len, d)
            x = x + ctx_flat @ self.params[p + "Wo"]
            x_m_norm, _, _ = _rmsnorm_fwd(x, self.params[p + "g_mlp"])
            act1, _ = _silu(x_m_norm @ self.params[p + "W1"])
            x = x + (act1 * (x_m_norm @ self.params[p + "W3"])) @ self.params[p + "W2"]

        h_final, _, _ = _rmsnorm_fwd(x, self.params["g_final"])
        return h_final.reshape(bsz * w_len, d)

    def forward_backward(
        self,
        x_ids: np.ndarray,
        y_ids: np.ndarray,
        span_ids: np.ndarray | None = None,
        prefix_stalks: np.ndarray | None = None,
        prior_win_ctx: np.ndarray | None = None,
        training: bool = True,
    ) -> tuple[float, dict[str, np.ndarray]]:
        cfg = self.cfg
        bsz, w_len = x_ids.shape
        d = cfg.d_model
        h_num = cfg.n_heads
        dh = d // h_num
        m_pre = cfg.kv_prefix_slots if prefix_stalks is not None else 0
        cos_w = self.cos[:w_len]
        sin_w = self.sin[:w_len]

        U_table = self.full_embedding_table()
        x = U_table[x_ids].copy()

        oe_cache: tuple[np.ndarray, np.ndarray, np.ndarray, float] | None = None
        if cfg.use_oe_span and cfg.n_spans > 0 and span_ids is not None:
            valid = span_ids >= 0
            safe_s = np.maximum(span_ids, 0)
            s_emb = self.params["E_span"][safe_s] * valid[:, :, None].astype(np.float32)
            proj = s_emb @ self.params["W_oe"]
            t_alpha = float(np.tanh(self.params["alpha_oe"][0]))
            x = x + t_alpha * proj
            oe_cache = (valid, safe_s, s_emb, t_alpha)

        q_ctx = x[:, 0, :].copy() if prior_win_ctx is None else prior_win_ctx
        layer_caches: list[dict[str, object]] = []
        scale = np.float32(1.0 / math.sqrt(float(dh)))

        for l_i in range(cfg.n_layers):
            p = f"L{l_i}."
            x_norm, x_hat_a, inv_rms_a = _rmsnorm_fwd(x, self.params[p + "g_attn"])
            q = (x_norm @ self.params[p + "Wq"]).reshape(bsz, w_len, h_num, dh).transpose(0, 2, 1, 3)
            k_seq = (x_norm @ self.params[p + "Wk"]).reshape(bsz, w_len, h_num, dh).transpose(0, 2, 1, 3)
            v_seq = (x_norm @ self.params[p + "Wv"]).reshape(bsz, w_len, h_num, dh).transpose(0, 2, 1, 3)

            q_rot = _apply_rope(q, cos_w, sin_w)
            k_rot = _apply_rope(k_seq, cos_w, sin_w)

            sheaf_cache: dict[str, np.ndarray] | None = None
            if m_pre > 0 and prefix_stalks is not None:
                q_proj = q_ctx @ self.params[p + "W_sq"]
                e_proj = prefix_stalks @ self.params[p + "W_se"]
                gate_pre = q_proj[:, None, :] + e_proj + self.params[p + "b_s"][None, None, :]
                f_gate = _sigmoid(gate_pre).astype(np.float32)
                transported = f_gate * prefix_stalks

                k_mem = (transported @ self.params[p + "W_mk"]).reshape(bsz, m_pre, h_num, dh).transpose(0, 2, 1, 3)
                v_mem_raw = (transported @ self.params[p + "W_mv"]).reshape(bsz, m_pre, h_num, dh).transpose(0, 2, 1, 3)
                t_kv = np.tanh(self.params[p + "alpha_kv"])
                v_mem = v_mem_raw * t_kv[None, :, None, None]

                scores_mem = (q @ k_mem.swapaxes(-1, -2)) * scale + np.float32(cfg.mem_score_bias)
                scores_seq = (q_rot @ k_rot.swapaxes(-1, -2)) * scale
                scores = np.concatenate([scores_mem, scores_seq], axis=-1)
                v_all = np.concatenate([v_mem, v_seq], axis=2)
                sheaf_cache = {
                    "f_gate": f_gate,
                    "transported": transported,
                    "v_mem_raw": v_mem_raw,
                    "t_kv": t_kv,
                    "k_mem": k_mem,
                    "q_unrot": q,
                }
            else:
                scores = (q_rot @ k_rot.swapaxes(-1, -2)) * scale
                v_all = v_seq

            causal_m = self.causal_mask[:w_len, :w_len]
            if m_pre > 0:
                full_mask = np.ones((bsz, 1, w_len, m_pre + w_len), dtype=bool)
                full_mask[:, :, :, m_pre:] = causal_m[None, None, :, :]
                if training and cfg.trime_local_drop > 0.0 and w_len > 8:
                    dist = np.arange(w_len)[:, None] - np.arange(w_len)[None, :]
                    far_local = dist > 8
                    keep_draw = self.rng.random((bsz, 1, w_len, w_len)) >= cfg.trime_local_drop
                    full_mask[:, :, :, m_pre:] &= np.where(far_local[None, None, :, :], keep_draw, True)
            else:
                full_mask = np.broadcast_to(causal_m[None, None, :, :], (bsz, 1, w_len, w_len))

            scores_masked = np.where(full_mask, scores, -1e9)
            mx_s = np.max(scores_masked, axis=-1, keepdims=True)
            ex_s = np.where(full_mask, np.exp(scores_masked - mx_s), 0.0)
            attn_probs = (ex_s / np.maximum(np.sum(ex_s, axis=-1, keepdims=True), 1e-12)).astype(np.float32)

            ctx = attn_probs @ v_all
            ctx_flat = ctx.transpose(0, 2, 1, 3).reshape(bsz, w_len, d)
            x = x + ctx_flat @ self.params[p + "Wo"]

            x_m_norm, x_hat_m, inv_rms_m = _rmsnorm_fwd(x, self.params[p + "g_mlp"])
            u1 = x_m_norm @ self.params[p + "W1"]
            u3 = x_m_norm @ self.params[p + "W3"]
            act1, sig1 = _silu(u1)
            hidden_ff = act1 * u3
            x = x + hidden_ff @ self.params[p + "W2"]

            layer_caches.append(
                {
                    "x_norm": x_norm,
                    "x_hat_a": x_hat_a,
                    "inv_rms_a": inv_rms_a,
                    "q_rot": q_rot,
                    "k_rot": k_rot,
                    "v_all": v_all,
                    "attn_probs": attn_probs,
                    "ctx_flat": ctx_flat,
                    "sheaf_cache": sheaf_cache,
                    "x_m_norm": x_m_norm,
                    "x_hat_m": x_hat_m,
                    "inv_rms_m": inv_rms_m,
                    "u1": u1,
                    "u3": u3,
                    "act1": act1,
                    "sig1": sig1,
                    "hidden_ff": hidden_ff,
                }
            )

        h_final, x_hat_f, inv_rms_f = _rmsnorm_fwd(x, self.params["g_final"])
        H_flat = h_final.reshape(bsz * w_len, d)
        targets_flat = y_ids.reshape(bsz * w_len)

        grads: dict[str, np.ndarray] = {}
        d_delta_root: np.ndarray | None = None
        d_delta_tail: np.ndarray | None = None

        if not training:
            if cfg.head_kind == "fp2":
                assert self.partition is not None
                loss_vec, _, _, _ = factored_fp2_fwd_bwd(
                    H_flat, U_table, self.params["delta_root"], self.partition, targets_flat, compute_grad=False
                )
            elif cfg.head_kind == "tied_root":
                assert self.shortcut_layout is not None
                loss_vec, _, _, _ = tied_root_shortcut_fwd_bwd(
                    H_flat, U_table, self.params["delta_tail"], self.shortcut_layout, targets_flat, compute_grad=False
                )
            elif cfg.vocab_size <= 4096:
                loss_vec, _, _ = flat_tied_fwd_bwd(H_flat, U_table, targets_flat, compute_grad=False)
            else:
                loss_vec, _, _ = tiled_cce_fwd_bwd(H_flat, U_table, targets_flat, compute_grad=False)
            return float(np.mean(loss_vec)), grads

        if cfg.head_kind == "flat":
            loss_vec, dH_flat, dU_out = flat_tied_fwd_bwd(H_flat, U_table, targets_flat, compute_grad=True)
        elif cfg.head_kind == "tiled_cce":
            loss_vec, dH_flat, dU_out = tiled_cce_fwd_bwd(H_flat, U_table, targets_flat, compute_grad=True)
        elif cfg.head_kind == "fp2":
            assert self.partition is not None
            loss_vec, dH_flat, dU_out, d_delta_root = factored_fp2_fwd_bwd(
                H_flat, U_table, self.params["delta_root"], self.partition, targets_flat, compute_grad=True
            )
        elif cfg.head_kind == "cluster_is":
            assert self.partition is not None
            loss_vec, _, dH_flat, dU_out, d_delta_root = cluster_is_fwd_bwd(
                H_flat,
                U_table,
                self.params["delta_root"],
                self.partition,
                targets_flat,
                k_neg=cfg.k_neg,
                rng=self.rng,
                compute_grad=True,
            )
        elif cfg.head_kind == "tied_root":
            assert self.shortcut_layout is not None
            loss_vec, dH_flat, dU_out, d_delta_tail = tied_root_shortcut_fwd_bwd(
                H_flat, U_table, self.params["delta_tail"], self.shortcut_layout, targets_flat, compute_grad=True
            )
        else:
            raise ValueError(f"Unsupported head_kind={cfg.head_kind!r}")

        assert dH_flat is not None and dU_out is not None
        if d_delta_root is not None:
            grads["delta_root"] = d_delta_root
        if d_delta_tail is not None:
            grads["delta_tail"] = d_delta_tail

        dx, dg_f = _rmsnorm_bwd(dH_flat.reshape(bsz, w_len, d), x_hat_f, inv_rms_f, self.params["g_final"])
        grads["g_final"] = dg_f

        dq_ctx_accum = np.zeros((bsz, d), dtype=np.float32)

        for l_i in range(cfg.n_layers - 1, -1, -1):
            p = f"L{l_i}."
            lc = layer_caches[l_i]
            x_m_norm = lc["x_m_norm"]  # type: ignore[assignment]
            hidden_ff = lc["hidden_ff"]  # type: ignore[assignment]
            u1 = lc["u1"]  # type: ignore[assignment]
            u3 = lc["u3"]  # type: ignore[assignment]
            act1 = lc["act1"]  # type: ignore[assignment]
            sig1 = lc["sig1"]  # type: ignore[assignment]

            grads[p + "W2"] = hidden_ff.reshape(-1, cfg.d_ff).T @ dx.reshape(-1, d)
            d_hidden = dx @ self.params[p + "W2"].T
            du3 = d_hidden * act1
            d_act1 = d_hidden * u3
            du1 = d_act1 * (sig1 * (1.0 + u1 * (1.0 - sig1)))
            grads[p + "W1"] = x_m_norm.reshape(-1, d).T @ du1.reshape(-1, cfg.d_ff)
            grads[p + "W3"] = x_m_norm.reshape(-1, d).T @ du3.reshape(-1, cfg.d_ff)
            dx_m_norm = du1 @ self.params[p + "W1"].T + du3 @ self.params[p + "W3"].T
            dx_mlp_in, dg_m = _rmsnorm_bwd(
                dx_m_norm,
                lc["x_hat_m"],  # type: ignore[arg-type]
                lc["inv_rms_m"],  # type: ignore[arg-type]
                self.params[p + "g_mlp"],
            )
            grads[p + "g_mlp"] = dg_m
            dx = dx + dx_mlp_in

            ctx_flat = lc["ctx_flat"]  # type: ignore[assignment]
            attn_probs = lc["attn_probs"]  # type: ignore[assignment]
            q_rot = lc["q_rot"]  # type: ignore[assignment]
            k_rot = lc["k_rot"]  # type: ignore[assignment]
            v_all = lc["v_all"]  # type: ignore[assignment]
            x_norm = lc["x_norm"]  # type: ignore[assignment]

            grads[p + "Wo"] = ctx_flat.reshape(-1, d).T @ dx.reshape(-1, d)
            d_ctx_flat = dx @ self.params[p + "Wo"].T
            d_ctx = d_ctx_flat.reshape(bsz, w_len, h_num, dh).transpose(0, 2, 1, 3)

            d_attn = d_ctx @ v_all.swapaxes(-1, -2)
            dv_all = attn_probs.swapaxes(-1, -2) @ d_ctx

            sum_da = np.sum(d_attn * attn_probs, axis=-1, keepdims=True)
            d_scores = attn_probs * (d_attn - sum_da) * scale

            if m_pre > 0 and prefix_stalks is not None and lc["sheaf_cache"] is not None:
                sc = lc["sheaf_cache"]  # type: ignore[assignment]
                d_scores_mem = d_scores[:, :, :, :m_pre]
                d_scores_seq = d_scores[:, :, :, m_pre:]
                dq_unrot = d_scores_mem @ sc["k_mem"]
                dk_mem = (d_scores_mem.swapaxes(-1, -2) @ sc["q_unrot"]).transpose(0, 2, 1, 3).reshape(bsz, m_pre, d)
                dq_rot = d_scores_seq @ k_rot
                dk_rot = d_scores_seq.swapaxes(-1, -2) @ q_rot
                dv_mem = dv_all[:, :, :m_pre, :]
                dv_seq = dv_all[:, :, m_pre:, :]

                t_kv = sc["t_kv"]
                v_mem_raw = sc["v_mem_raw"]
                d_t_kv = np.sum(dv_mem * v_mem_raw, axis=(0, 2, 3))
                grads[p + "alpha_kv"] = d_t_kv * (1.0 - t_kv * t_kv)

                dv_mem_raw = (dv_mem * t_kv[None, :, None, None]).transpose(0, 2, 1, 3).reshape(bsz, m_pre, d)
                transported = sc["transported"]
                grads[p + "W_mk"] = transported.reshape(-1, cfg.d_stalk).T @ dk_mem.reshape(-1, d)
                grads[p + "W_mv"] = transported.reshape(-1, cfg.d_stalk).T @ dv_mem_raw.reshape(-1, d)

                d_trans = dk_mem @ self.params[p + "W_mk"].T + dv_mem_raw @ self.params[p + "W_mv"].T
                f_gate = sc["f_gate"]
                d_gate_pre = d_trans * prefix_stalks * f_gate * (1.0 - f_gate)
                grads[p + "b_s"] = np.sum(d_gate_pre, axis=(0, 1))
                grads[p + "W_se"] = prefix_stalks.reshape(-1, cfg.d_stalk).T @ d_gate_pre.reshape(-1, cfg.d_stalk)
                d_q_proj = np.sum(d_gate_pre, axis=1)
                grads[p + "W_sq"] = q_ctx.T @ d_q_proj
                if prior_win_ctx is None:
                    dq_ctx_accum += d_q_proj @ self.params[p + "W_sq"].T
                dq = (_apply_rope_bwd(dq_rot, cos_w, sin_w) + dq_unrot).transpose(0, 2, 1, 3).reshape(bsz, w_len, d)
            else:
                dq_rot = d_scores @ k_rot
                dk_rot = d_scores.swapaxes(-1, -2) @ q_rot
                dv_seq = dv_all
                dq = _apply_rope_bwd(dq_rot, cos_w, sin_w).transpose(0, 2, 1, 3).reshape(bsz, w_len, d)

            dk = _apply_rope_bwd(dk_rot, cos_w, sin_w).transpose(0, 2, 1, 3).reshape(bsz, w_len, d)
            dv = dv_seq.transpose(0, 2, 1, 3).reshape(bsz, w_len, d)

            grads[p + "Wq"] = x_norm.reshape(-1, d).T @ dq.reshape(-1, d)
            grads[p + "Wk"] = x_norm.reshape(-1, d).T @ dk.reshape(-1, d)
            grads[p + "Wv"] = x_norm.reshape(-1, d).T @ dv.reshape(-1, d)

            dx_norm = dq @ self.params[p + "Wq"].T + dk @ self.params[p + "Wk"].T + dv @ self.params[p + "Wv"].T
            dx_a_in, dg_a = _rmsnorm_bwd(
                dx_norm,
                lc["x_hat_a"],  # type: ignore[arg-type]
                lc["inv_rms_a"],  # type: ignore[arg-type]
                self.params[p + "g_attn"],
            )
            grads[p + "g_attn"] = dg_a
            dx = dx + dx_a_in

        if prior_win_ctx is None and m_pre > 0:
            dx[:, 0, :] += dq_ctx_accum

        if oe_cache is not None:
            valid, safe_s, s_emb, t_alpha = oe_cache
            proj = s_emb @ self.params["W_oe"]
            d_t_alpha = float(np.sum(dx * proj))
            grads["alpha_oe"] = np.array([d_t_alpha * (1.0 - t_alpha * t_alpha)], dtype=np.float32)
            d_proj = dx * t_alpha
            grads["W_oe"] = s_emb.reshape(-1, cfg.d_span).T @ d_proj.reshape(-1, d)
            d_s_emb = (d_proj @ self.params["W_oe"].T) * valid[:, :, None].astype(np.float32)
            dE_span = np.zeros_like(self.params["E_span"])
            fast_scatter_add_2d(dE_span, safe_s.reshape(-1), d_s_emb.reshape(-1, cfg.d_span))
            grads["E_span"] = dE_span

        dU_total = dU_out.copy()
        fast_scatter_add_2d(dU_total, x_ids.reshape(-1), dx.reshape(-1, d))
        if self.span_composer is not None and "delta_span" in self.params:
            dE_base, d_delta_span = self.span_composer.backward(dU_total, self.v_base)
            grads["E_base"] = dE_base
            grads["delta_span"] = d_delta_span
        else:
            grads["E_base"] = dU_total

        return float(np.mean(loss_vec)), grads

    def step_adamw(
        self,
        grads: dict[str, np.ndarray],
        lr: float = 2e-3,
        weight_decay: float = 0.02,
        beta1: float = 0.9,
        beta2: float = 0.98,
        eps: float = 1e-8,
        grad_clip: float = 1.0,
    ) -> None:
        self._step += 1
        norm_sq = sum(float(np.sum(g * g)) for g in grads.values())
        gnorm = math.sqrt(norm_sq)
        clip_scale = min(1.0, grad_clip / max(gnorm, 1e-12))

        mup_ratio = float(self.cfg.base_width_mup) / float(self.cfg.d_model)
        bc1 = 1.0 - beta1**self._step
        bc2 = 1.0 - beta2**self._step

        for name, param in self.params.items():
            g = grads.get(name)
            if g is None:
                continue
            g_scaled = g * clip_scale
            m = self._m[name]
            v = self._v[name]
            m[:] = beta1 * m + (1.0 - beta1) * g_scaled
            v[:] = beta2 * v + (1.0 - beta2) * (g_scaled * g_scaled)
            m_hat = m / bc1
            v_hat = v / bc2

            is_emb_or_scalar = name in {"E_base", "E_span", "alpha_oe"} or name.endswith(
                ("g_attn", "g_mlp", "g_final", "alpha_kv", "b_s")
            )
            eff_lr = lr if is_emb_or_scalar else lr * mup_ratio
            wd = 0.0 if (param.ndim <= 1 or name.startswith("alpha")) else weight_decay
            if wd > 0.0:
                param *= 1.0 - eff_lr * wd
            param -= eff_lr * (m_hat / (np.sqrt(v_hat) + eps))
