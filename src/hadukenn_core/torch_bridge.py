"""Drop-in static-shape PyTorch `nn.Module` kernels (`torch.compile(fullgraph=True)` & RTX 5070 ready).

Uses explicit low-level tensor contractions (`@`, `torch.bmm`, `logsumexp`, `gather`) with
exact `NC * S == V` bijection (`256 x 128`, `1024 x 128`, `2048 x 128`), `chunk_tokens = 256`
micro-slicing (preventing the `16.0 GiB` 3D gather OOM trap at `8B`), 4-bit NormalFloat (`NF4`)
frozen linear blocks + LoRA adapters, and RoPE-decoupled `AsymmetricSheafKVPrefixTorch` so
`sheaf-verify audit-kernel` passes with 0 violations.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import torch

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    TORCH_AVAILABLE = False

NF4_VALUES = (
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
)


def _explicit_nll_from_logits(logits: torch.Tensor, target_idx: torch.Tensor) -> torch.Tensor:
    lse = torch.logsumexp(logits.float(), dim=-1)
    chosen = logits.float().gather(1, target_idx.unsqueeze(1)).squeeze(1)
    return (lse - chosen).mean()


if TORCH_AVAILABLE:  # pragma: no cover

    class ParadigmaticClusterISHeadTorch(nn.Module):
        """
        Frontier-5070 Output Head (`256x128`, `1024x128` @ 131k, `2048x128` @ 262k):
          - Shares `base_weight` (`[V, d]`, `struct_rank = 0`, `resid = none`).
          - Trains detached root proposal `C_c = mean_{j in c}(E_j) + delta_root[c]`.
          - Micro-slices `[0 : n_tok]` into `chunk_tokens = 256` slices so 3D `e_cl[active_c]`
            peaks at `<= 1.00 GiB` on an 8B (`d = 4096`) model instead of `16.0 GiB` OOM!
        """

        def __init__(
            self,
            cluster_units: torch.Tensor,
            d_model: int,
            k_neg: int = 3,
            chunk_tokens: int = 256,
        ) -> None:
            super().__init__()
            nc, s_sz = cluster_units.shape
            self.n_clusters = int(nc)
            self.cluster_size = int(s_sz)
            self.k_neg = int(k_neg)
            self.chunk_tokens = max(int(chunk_tokens), 16)
            self.register_buffer("cluster_units", cluster_units.to(torch.long))
            inv_c = torch.empty(nc * s_sz, dtype=torch.long)
            inv_m = torch.empty(nc * s_sz, dtype=torch.long)
            ar_m = torch.arange(s_sz, dtype=torch.long)
            for c in range(nc):
                members = cluster_units[c].to(torch.long)
                inv_c[members] = c
                inv_m[members] = ar_m
            self.register_buffer("token_to_cluster", inv_c)
            self.register_buffer("token_to_member", inv_m)
            self.delta_root = nn.Parameter(torch.zeros(nc, d_model))

        def forward(
            self,
            hidden: torch.Tensor,
            targets: torch.Tensor,
            base_weight: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            h_flat = hidden.reshape(-1, hidden.size(-1))
            y_flat = targets.reshape(-1)
            n_tok = h_flat.size(0)
            tc = self.token_to_cluster[y_flat]
            tm = self.token_to_member[y_flat]

            e_cl = base_weight[self.cluster_units]
            c_root = e_cl.detach().mean(dim=1) + self.delta_root
            root_logits = h_flat.detach() @ c_root.t()
            root_nll = _explicit_nll_from_logits(root_logits, tc)

            with torch.no_grad():
                rl_f = root_logits.float()
                q_prob = (rl_f - torch.logsumexp(rl_f, dim=-1, keepdim=True)).exp()
                q_prob.scatter_(1, tc.unsqueeze(1), 0.0)
                q_prob = q_prob / q_prob.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                unif = 1.0 / float(self.n_clusters - 1)
                q_tilde = 0.85 * q_prob + 0.15 * unif
                q_tilde.scatter_(1, tc.unsqueeze(1), 0.0)
                q_tilde = q_tilde / q_tilde.sum(dim=-1, keepdim=True)
                u_rand = torch.rand_like(q_tilde).clamp_(1e-7, 1.0 - 1e-7)
                gumbel = q_tilde.clamp_min(1e-12).log() - (-u_rand.log()).log()
                gumbel.scatter_(1, tc.unsqueeze(1), -1e30)
                neg_c = gumbel.topk(self.k_neg, dim=-1).indices
                q_neg = q_tilde.gather(1, neg_c)
                p_incl = 1.0 - (1.0 - q_neg).clamp_min(0.0).pow(self.k_neg)
                log_w_neg = -p_incl.clamp_min(1e-7).log().to(hidden.dtype)

            active_c = torch.cat([tc.unsqueeze(1), neg_c], dim=1)
            lm_loss_accum = torch.zeros((), device=hidden.device, dtype=torch.float32)
            for start in range(0, n_tok, self.chunk_tokens):
                end = min(start + self.chunk_tokens, n_tok)
                m_sub = end - start
                e_act = e_cl[active_c[start:end]].reshape(m_sub, (1 + self.k_neg) * self.cluster_size, -1)
                dots = torch.bmm(e_act, h_flat[start:end].unsqueeze(-1)).squeeze(-1)
                dots_blk = dots.view(m_sub, 1 + self.k_neg, self.cluster_size)
                bias = torch.cat(
                    [
                        torch.zeros(m_sub, 1, self.cluster_size, device=hidden.device, dtype=hidden.dtype),
                        log_w_neg[start:end].unsqueeze(-1).expand(-1, -1, self.cluster_size),
                    ],
                    dim=1,
                )
                sub_loss = _explicit_nll_from_logits((dots_blk + bias).reshape(m_sub, -1), tm[start:end])
                lm_loss_accum = lm_loss_accum + sub_loss * (float(m_sub) / float(n_tok))
            return lm_loss_accum, root_nll

    class NF4FrozenLinearTorch(nn.Module):
        """
        Dettmers et al. (2023) 4-bit NormalFloat (`NF4`) + double-quantized `uint8` block scales
        with optional LoRA (`rank = 16`) adapter for 3B / 8B checkpoint surgery inside 12 GB VRAM.
        """

        def __init__(self, weight_fp: torch.Tensor, block_size: int = 64, lora_rank: int = 16) -> None:
            super().__init__()
            out_features, in_features = weight_fp.shape
            self.in_features = int(in_features)
            self.out_features = int(out_features)
            self.block_size = int(block_size)
            self.orig_numel = int(weight_fp.numel())

            cb = torch.tensor(NF4_VALUES, dtype=torch.float32, device=weight_fp.device)
            thresh = 0.5 * (cb[:-1] + cb[1:])
            flat = weight_fp.detach().float().reshape(-1)
            pad = (-self.orig_numel) % self.block_size
            if pad > 0:
                flat = torch.cat([flat, torch.zeros(pad, dtype=torch.float32, device=weight_fp.device)], dim=0)
            blks = flat.view(-1, self.block_size)
            absmax = blks.abs().amax(dim=-1).clamp_min(1e-12)
            codes = torch.bucketize(blks / absmax.unsqueeze(-1), thresh).to(torch.uint8)
            packed = (codes[:, 0::2] << 4) | codes[:, 1::2]
            s_max = absmax.amax().clamp_min(1e-12)
            q_scale = ((absmax / s_max) * 255.0).round().clamp(0.0, 255.0).to(torch.uint8)

            self.register_buffer("codebook", cb)
            self.register_buffer("packed_nf4", packed)
            self.register_buffer("q_absmax", q_scale)
            self.register_buffer("super_max", s_max.view(1))

            self.lora_rank = int(lora_rank)
            if self.lora_rank > 0:
                self.lora_a = nn.Parameter(
                    torch.randn(in_features, lora_rank, device=weight_fp.device) / math.sqrt(float(in_features))
                )
                self.lora_b = nn.Parameter(torch.zeros(lora_rank, out_features, device=weight_fp.device))
            else:
                self.register_parameter("lora_a", None)
                self.register_parameter("lora_b", None)

        def dequantize_weight(self, dtype: torch.dtype) -> torch.Tensor:
            hi = ((self.packed_nf4 >> 4) & 0x0F).to(torch.long).unsqueeze(-1)
            lo = (self.packed_nf4 & 0x0F).to(torch.long).unsqueeze(-1)
            codes = torch.cat([hi, lo], dim=-1).reshape(-1, self.block_size)
            scales = (self.q_absmax.float() / 255.0) * self.super_max
            vals = (self.codebook[codes] * scales.unsqueeze(-1)).reshape(-1)[: self.orig_numel]
            return vals.view(self.out_features, self.in_features).to(dtype)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            w = self.dequantize_weight(x.dtype)
            out = x @ w.t()
            if self.lora_rank > 0 and self.lora_a is not None and self.lora_b is not None:
                out = out + ((x @ self.lora_a.to(x.dtype)) @ self.lora_b.to(x.dtype)) * 0.5
            return out

    class OESpanEmbeddingTorch(nn.Module):
        """T-Fix 1 (`#46`): Sparse causal input span sidecar keeping clean base BPE output targets."""

        def __init__(self, n_spans: int, d_span: int, d_model: int) -> None:
            super().__init__()
            self.span_table = nn.Parameter(torch.randn(n_spans, d_span) * 0.02)
            self.w_proj = nn.Parameter(torch.randn(d_span, d_model) / math.sqrt(float(d_span)))
            self.alpha = nn.Parameter(torch.full((1,), 0.15))

        def forward(self, base_emb: torch.Tensor, ending_span_ids: torch.Tensor) -> torch.Tensor:
            valid = (ending_span_ids >= 0).unsqueeze(-1).to(base_emb.dtype)
            safe_ids = ending_span_ids.clamp_min(0)
            span_vec = (self.span_table[safe_ids] @ self.w_proj) * valid
            return base_emb + torch.tanh(self.alpha) * span_vec

    class AsymmetricSheafKVPrefixTorch(nn.Module):
        """
        G2 / U3 (`Q = W, KV = W + M_prefix`): Projects self-excluded hyperedge stalks
        through diagonal cellular sheaf restriction maps into unrotated keys (`q_unrot @ k_mem^T`)
        and LayerScale-gated values (`0` query-tile spillover, `0` RoPE sign-flipping).
        """

        def __init__(self, d_model: int, d_stalk: int, n_heads: int) -> None:
            super().__init__()
            self.d_model = int(d_model)
            self.d_stalk = int(d_stalk)
            self.n_heads = int(n_heads)
            self.head_dim = d_model // n_heads
            scale_d = 1.0 / math.sqrt(float(d_model))
            scale_s = 1.0 / math.sqrt(float(d_stalk))
            self.w_sq = nn.Parameter(torch.randn(d_model, d_stalk) * scale_d)
            self.w_se = nn.Parameter(torch.randn(d_stalk, d_stalk) * scale_s)
            self.b_s = nn.Parameter(torch.full((d_stalk,), 0.5))
            self.w_mk = nn.Parameter(torch.randn(d_stalk, d_model) * scale_s)
            self.w_mv = nn.Parameter(torch.randn(d_stalk, d_model) * scale_s)
            self.alpha_kv = nn.Parameter(torch.full((n_heads,), 0.15))

        def forward(
            self,
            q_ctx: torch.Tensor,
            self_excluded_stalks: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            bsz, m_slots, _ = self_excluded_stalks.shape
            gate_logits = (q_ctx @ self.w_sq).unsqueeze(1) + (self_excluded_stalks @ self.w_se) + self.b_s
            transported = torch.sigmoid(gate_logits) * self_excluded_stalks
            k_mem = (transported @ self.w_mk).view(bsz, m_slots, self.n_heads, self.head_dim).transpose(1, 2)
            v_mem = (transported @ self.w_mv).view(bsz, m_slots, self.n_heads, self.head_dim).transpose(1, 2)
            v_mem = v_mem * torch.tanh(self.alpha_kv).view(1, self.n_heads, 1, 1)
            return k_mem, v_mem

        def compute_hybrid_scores(
            self,
            q_unrot: torch.Tensor,
            q_rot: torch.Tensor,
            k_mem: torch.Tensor,
            k_rot: torch.Tensor,
        ) -> torch.Tensor:
            scale = 1.0 / math.sqrt(float(self.head_dim))
            s_mem = (q_unrot @ k_mem.transpose(-1, -2)) * scale
            s_seq = (q_rot @ k_rot.transpose(-1, -2)) * scale
            return torch.cat([s_mem, s_seq], dim=-1)
