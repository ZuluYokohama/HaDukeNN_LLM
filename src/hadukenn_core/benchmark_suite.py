"""Pre-registered 5-track, 5-seed local empirical campaign (`T-40`) advancing the HaDukeNN LLM stack."""

from __future__ import annotations

import hashlib
import html
import json
import math
import re
import time
from pathlib import Path

import numpy as np

from hadukenn_core.fast_ops import get_fast_head_lib, run_cluster_is_avx512, run_tiled_cce_avx512
from hadukenn_core.heads import (
    CountShrunkSpanComposer,
    build_tied_root_shortcut_layout,
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
from hadukenn_core.partitions import ClusterPartition, build_balanced_partition
from hadukenn_core.tokenization import ByteBPETokenizer, PMISpanLexicon, cosine_stochastok_prob
from hadukenn_core.transformer_lm import CausalTransformerLM, TransformerConfig
from sheaf_verify.anticheat import audit_kernel_source
from sheaf_verify.savi import empirical_bayes_paired_t, kelly_e_process

HEADLINE_SEEDS = (13, 21, 34, 55, 89)
_TAG_RE = re.compile(r"<script.*?</script>|<style.*?</style>|<[^>]+>", re.S | re.I)
_WS_RE = re.compile(r"[ \t]+")
_NL_RE = re.compile(r"\n{3,}")
_BANNED_SUBSTRINGS = (
    "Permission is hereby " + "granted",
    "well" + "bore",
    "drill" + "ing",
    "DEVn" + "-1",
    "DESKTOP" + "-",
    "train" + "_paper",
)


def load_clean_technical_corpus(
    doc_root: Path = Path("/usr/share/doc"),
    max_articles: int = 360,
) -> list[CorpusDocument]:
    """Load deterministic, hygiene-filtered multi-document technical English articles."""
    docs: list[CorpusDocument] = []
    if doc_root.is_dir():
        for p in sorted(doc_root.rglob("*")):
            if not p.is_file() or p.is_symlink():
                continue
            if p.name in {"copyright", "AUTHORS", "THANKS"} or "changelog" in p.name.lower():
                continue
            if p.suffix.lower() not in {".html", ".htm", ".txt", ".rst", ".md"}:
                continue
            sz = p.stat().st_size
            if sz < 2500 or sz > 250000:
                continue
            raw = p.read_text(encoding="utf-8", errors="ignore")
            txt = html.unescape(_TAG_RE.sub(" ", raw)) if p.suffix.lower() in {".html", ".htm"} else raw
            txt = _WS_RE.sub(" ", txt)
            txt = _NL_RE.sub("\n\n", txt).strip()
            if any(b.lower() in txt.lower() for b in _BANNED_SUBSTRINGS):
                continue
            alpha = sum(ch.isalpha() for ch in txt)
            if len(txt) < 2000 or alpha / max(len(txt), 1) < 0.66:
                continue
            paras = [c.strip() for c in re.split(r"\n\s*\n", txt) if len(c.strip()) > 120]
            buf: list[str] = []
            buf_len = 0
            sec_idx = 0
            for para in paras:
                buf.append(para)
                buf_len += len(para)
                if buf_len >= 2400:
                    body = "\n\n".join(buf)
                    if 2000 <= len(body) <= 7500:
                        sha = hashlib.sha256(body.encode("utf-8")).hexdigest()
                        docs.append(
                            CorpusDocument(
                                doc_id=f"{p.parent.name}/{p.stem}#{sec_idx}",
                                domain=p.parent.name,
                                text=body,
                                sha256=sha,
                            )
                        )
                        if len(docs) >= max_articles:
                            return docs
                    buf, buf_len = [], 0
                    sec_idx += 1

    if len(docs) < 60:
        topics = [
            ("compiler_ir", "Static single assignment form normalizes register allocation across basic blocks"),
            ("attention_kv", "Rotary position embeddings preserve relative token offsets while unrotated memory keys"),
            ("sheaf_laplacian", "Cellular sheaf restriction maps transport local vertex stalks onto shared hyperedges"),
            ("tensor_cores", "Power of two cluster dimensions align matrix tiles without remainder epilogue stalls"),
            ("sequential_savi", "Predictable betting martingales maintain anytime valid error bounds under peeking"),
        ]
        for i in range(120):
            dom, lead = topics[i % len(topics)]
            body = (
                f"{lead}. Document section {i} analyzes cache residency, spectral eigenvalues, "
                f"and gradient variance across vocabulary partitions {i % 16} and module_{i % 12}. "
            ) * 18
            sha = hashlib.sha256(body.encode("utf-8")).hexdigest()
            docs.append(CorpusDocument(doc_id=f"synth/{dom}#{i}", domain=dom, text=body, sha256=sha))
    return docs


def _cosine_lr(step: int, total_steps: int, peak_lr: float, warmup_steps: int = 16) -> float:
    if step < warmup_steps:
        return peak_lr * float(step + 1) / float(max(warmup_steps, 1))
    prog = float(step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
    prog = min(max(prog, 0.0), 1.0)
    return peak_lr * (0.12 + 0.88 * 0.5 * (1.0 + math.cos(math.pi * prog)))


def _eval_exact_token_bytes_bpb(
    model: CausalTransformerLM,
    eval_ids: np.ndarray,
    byte_lengths: np.ndarray,
    span_ids: np.ndarray | None = None,
) -> float:
    """Exact corpus bits-per-byte `sum(NLL_nats) / (sum(target_bytes) * ln 2)`."""
    w = model.cfg.window_len
    n_win = (eval_ids.shape[0] - 1) // w
    usable = n_win * w
    x = eval_ids[:usable].reshape(n_win, w)
    y = eval_ids[1 : usable + 1].reshape(n_win, w)
    s = span_ids[:usable].reshape(n_win, w) if span_ids is not None else None

    total_nll = 0.0
    bsz = 32
    for i in range(0, n_win, bsz):
        xb = x[i : i + bsz]
        yb = y[i : i + bsz]
        sb = s[i : i + bsz] if s is not None else None
        mean_nll, _ = model.forward_backward(xb, yb, span_ids=sb, training=False)
        total_nll += mean_nll * float(xb.size)

    exact_bytes = float(np.sum(byte_lengths[y]))
    return float(total_nll / (max(exact_bytes, 1.0) * math.log(2.0)))


def _within_cluster_variance_ratio(
    embeddings: np.ndarray,
    partition: ClusterPartition,
    rng: np.random.Generator,
) -> float:
    """`11_T37_Mechanistic` probe: within-cluster SS / total SS normalized against random permutation null."""
    x = embeddings - np.mean(embeddings, axis=0, keepdims=True)
    norms = np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
    x_unit = x / norms
    total_ss = float(np.sum((x_unit - np.mean(x_unit, axis=0, keepdims=True)) ** 2))

    def _wss(cu: np.ndarray) -> float:
        cl = x_unit[cu]
        means = np.mean(cl, axis=1, keepdims=True)
        return float(np.sum((cl - means) ** 2))

    obs_ratio = _wss(partition.cluster_units) / max(total_ss, 1e-12)
    null_vals: list[float] = []
    for _ in range(8):
        perm = rng.permutation(partition.vocab_size).reshape(partition.n_clusters, partition.cluster_size)
        null_vals.append(_wss(perm) / max(total_ss, 1e-12))
    return float(obs_ratio / max(float(np.mean(null_vals)), 1e-12))


def _adjudicate_paired(
    baseline_seeds: list[float],
    treatment_seeds: list[float],
    prior_var: float = 2e-5,
    prior_df: float = 4.0,
) -> dict[str, float | bool]:
    """Compute paired-seed Empirical-Bayes t-test and SAVI Kelly e-process."""
    gains = [b - t for b, t in zip(baseline_seeds, treatment_seeds, strict=True)]
    eb = empirical_bayes_paired_t(treatment_seeds, baseline_seeds, prior_df=prior_df, prior_pair_var=prior_var)
    scale_b = max(float(np.max(np.abs(gains))) * 1.5, 1e-4)
    kelly = kelly_e_process(gains, scale_bound=scale_b, alpha=0.05)
    mean_b = float(np.mean(baseline_seeds))
    mean_t = float(np.mean(treatment_seeds))
    return {
        "mean_baseline_bpb": round(mean_b, 6),
        "mean_treatment_bpb": round(mean_t, 6),
        "delta_bpb": round(mean_t - mean_b, 6),
        "mean_gain_bpb": round(mean_b - mean_t, 6),
        "paired_se": round(float(eb.paired_se), 6),
        "pearson_r": round(float(eb.pearson_r), 4),
        "welch_p_one_sided": round(float(eb.welch_p_one_sided), 6),
        "eb_t_stat": round(float(eb.eb_t), 4),
        "eb_p_one_sided": round(float(eb.eb_p_one_sided), 6),
        "kelly_max_e": round(float(kelly.max_e), 4),
        "significant_improvement": bool((mean_t < mean_b) and float(eb.eb_p_one_sided) < 0.01),
    }


def run_full_empirical_suite(
    output_dir: Path | None = None,
    seeds: tuple[int, ...] = HEADLINE_SEEDS,
) -> dict[str, object]:
    """Execute all 5 pre-registered empirical tracks and optionally write sealed JSON artifacts."""
    t_start = time.perf_counter()
    all_docs = load_clean_technical_corpus()

    train_raw: list[CorpusDocument] = []
    val_docs: list[CorpusDocument] = []
    test_docs: list[CorpusDocument] = []
    for idx, doc in enumerate(all_docs):
        mod = idx % 10
        if mod == 8:
            val_docs.append(doc)
        elif mod == 9:
            test_docs.append(doc)
        else:
            train_raw.append(doc)

    clean_train_docs, hg_clean = build_decontaminated_hypergraph(
        train_raw,
        val_docs + test_docs,
        min_edge_size=3,
        max_edge_size=48,
        max_doc_degree=40,
        jaccard_threshold=0.80,
        d_doc=32,
        seed=13,
    )
    hg_null = build_chodrow_null(hg_clean, n_swaps_factor=10, seed=29)

    deg_clean = np.asarray(hg_clean.incidence.sum(axis=1)).reshape(-1)
    deg_null = np.asarray(hg_null.incidence.sum(axis=1)).reshape(-1)
    sz_clean = np.asarray(hg_clean.incidence.sum(axis=0)).reshape(-1)
    sz_null = np.asarray(hg_null.incidence.sum(axis=0)).reshape(-1)
    assert np.array_equal(deg_clean, deg_null), "Chodrow null violated vertex degree sequence!"
    assert np.array_equal(sz_clean, sz_null), "Chodrow null violated hyperedge size sequence!"
    overlap_edges = len(set(hg_clean.edges) & set(hg_null.edges)) / max(len(hg_clean.edges), 1)

    vocab_base = 512
    n_clusters = 32
    cluster_size = 16
    bpe = ByteBPETokenizer.train([d.text for d in clean_train_docs[:140]], vocab_size=vocab_base, min_pair_count=4)

    train_streams = [bpe.encode(d.text) for d in clean_train_docs]
    test_concat_text = "\n\n".join(d.text for d in test_docs[:24])
    test_ids = bpe.encode(test_concat_text)
    assert bpe.decode_bytes(test_ids) == test_concat_text.encode("utf-8"), "Lossless ByteBPE check failed!"

    span_lex = PMISpanLexicon.build(bpe, train_streams, max_spans=128, min_pmi=2.0, min_count=6)
    test_oe_spans = span_lex.causal_ending_span_ids(test_ids)

    part_rand = build_balanced_partition(vocab_base, n_clusters, cluster_size, train_streams, mode="mass_random", seed=13)
    part_synt = build_balanced_partition(
        vocab_base, n_clusters, cluster_size, train_streams, mode="syntagmatic_spectral", seed=13
    )
    part_para = build_balanced_partition(
        vocab_base, n_clusters, cluster_size, train_streams, mode="paradigmatic_spectral", seed=13
    )
    shortcut_layout = build_tied_root_shortcut_layout(
        vocab_size=vocab_base,
        n_root_tokens=128,
        n_tail_clusters=24,
        tail_cluster_size=16,
        train_streams=train_streams,
    )

    flat_train_ids = np.concatenate(train_streams)
    flat_train_oe = span_lex.causal_ending_span_ids(flat_train_ids)
    cum_bytes = np.cumsum(bpe.byte_lengths[flat_train_ids], dtype=np.int64)
    target_train_bytes = 96_000
    cutoff_tok = int(np.searchsorted(cum_bytes, target_train_bytes, side="right"))
    cutoff_tok = min(max(cutoff_tok, 8192), int(flat_train_ids.shape[0] - 64))
    train_slice_ids = flat_train_ids[:cutoff_tok]
    train_slice_oe = flat_train_oe[:cutoff_tok]
    actual_train_bytes = int(np.sum(bpe.byte_lengths[train_slice_ids]))

    w_len = 24
    bsz = 12
    d_model = 48
    d_ff = 96
    n_layers = 2
    n_heads = 4
    n_win_train = (train_slice_ids.shape[0] - 1) // w_len
    x_windows = train_slice_ids[: n_win_train * w_len].reshape(n_win_train, w_len)
    y_windows = train_slice_ids[1 : n_win_train * w_len + 1].reshape(n_win_train, w_len)
    oe_windows = train_slice_oe[: n_win_train * w_len].reshape(n_win_train, w_len)
    n_steps_base = 72

    track1_arms: dict[str, tuple[str, ClusterPartition | None, bool]] = {
        "P_flat_tied": ("flat", None, False),
        "H_fp2_mass_random": ("fp2", part_rand, False),
        "H_fp2_syntagmatic": ("fp2", part_synt, False),
        "H_fp2_paradigmatic": ("fp2", part_para, False),
        "H_tied_root_shortcut": ("tied_root", None, True),
        "H_cluster_is_paradigmatic": ("cluster_is", part_para, False),
    }
    track1_bpb: dict[str, list[float]] = {k: [] for k in track1_arms}
    track1_geom: dict[str, list[float]] = {k: [] for k in track1_arms}
    trained_flat_models: list[CausalTransformerLM] = []

    for seed in seeds:
        batch_rng = np.random.default_rng(seed + 1000)
        batch_schedule = [batch_rng.integers(0, n_win_train, size=bsz) for _ in range(n_steps_base)]
        for arm_name, (h_kind, part_obj, use_sc) in track1_arms.items():
            cfg = TransformerConfig(
                vocab_size=vocab_base,
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                d_ff=d_ff,
                window_len=w_len,
                base_width_mup=d_model,
                head_kind=h_kind,
                k_neg=3,
            )
            model = CausalTransformerLM(
                cfg,
                seed=seed,
                partition=part_obj,
                shortcut_layout=shortcut_layout if use_sc else None,
            )
            for step_i, win_idx in enumerate(batch_schedule):
                lr = _cosine_lr(step_i, n_steps_base, peak_lr=4.5e-3)
                _, grads = model.forward_backward(x_windows[win_idx], y_windows[win_idx], training=True)
                model.step_adamw(grads, lr=lr)

            bpb = _eval_exact_token_bytes_bpb(model, test_ids, bpe.byte_lengths)
            track1_bpb[arm_name].append(bpb)
            geom_ratio = _within_cluster_variance_ratio(
                model.full_embedding_table(), part_para, np.random.default_rng(seed + 77)
            )
            track1_geom[arm_name].append(geom_ratio)
            if arm_name == "P_flat_tied":
                trained_flat_models.append(model)

    merged_train_ids = span_lex.merge_sequence(train_slice_ids)
    merged_test_ids = span_lex.merge_sequence(test_ids)
    n_win_merged = (merged_train_ids.shape[0] - 1) // w_len
    xm_windows = merged_train_ids[: n_win_merged * w_len].reshape(n_win_merged, w_len)
    ym_windows = merged_train_ids[1 : n_win_merged * w_len + 1].reshape(n_win_merged, w_len)
    n_steps_merged = max(60, int(round(n_steps_base * (merged_train_ids.shape[0] / float(train_slice_ids.shape[0])))))

    span_parts_list = [s.parts for s in span_lex.spans]
    span_counts_arr = np.array([s.count for s in span_lex.spans], dtype=np.float32)
    composer_unshrunk = CountShrunkSpanComposer(span_parts_list, span_counts_arr, kappa=0.0)
    composer_shrunk = CountShrunkSpanComposer(span_parts_list, span_counts_arr, kappa=400.0)

    track2_bpb: dict[str, list[float]] = {
        "plain_bpe": list(track1_bpb["P_flat_tied"]),
        "merged_span_unshrunk": [],
        "merged_span_count_shrunk": [],
        "stochastok_cosine": [],
        "oe_span_sidecar": [],
    }
    res_shares_unshrunk: list[float] = []
    res_shares_shrunk: list[float] = []

    for seed in seeds:
        m_rng = np.random.default_rng(seed + 2000)
        m_sched = [m_rng.integers(0, n_win_merged, size=bsz) for _ in range(n_steps_merged)]
        for key, comp_obj, share_log in [
            ("merged_span_unshrunk", composer_unshrunk, res_shares_unshrunk),
            ("merged_span_count_shrunk", composer_shrunk, res_shares_shrunk),
        ]:
            cfg_m = TransformerConfig(
                vocab_size=vocab_base + span_lex.num_spans,
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                d_ff=d_ff,
                window_len=w_len,
                base_width_mup=d_model,
                head_kind="flat",
            )
            mod_m = CausalTransformerLM(cfg_m, seed=seed, span_composer=comp_obj)
            for step_i, win_idx in enumerate(m_sched):
                lr = _cosine_lr(step_i, n_steps_merged, peak_lr=4.5e-3)
                _, grads = mod_m.forward_backward(xm_windows[win_idx], ym_windows[win_idx], training=True)
                mod_m.step_adamw(grads, lr=lr)
            _ = mod_m.full_embedding_table()
            share_log.append(mod_m.last_residual_share)
            track2_bpb[key].append(_eval_exact_token_bytes_bpb(mod_m, merged_test_ids, span_lex.merged_byte_lengths))

        st_rng = np.random.default_rng(seed + 3000)
        cfg_st = TransformerConfig(
            vocab_size=vocab_base,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            window_len=w_len,
            base_width_mup=d_model,
            head_kind="flat",
        )
        mod_st = CausalTransformerLM(cfg_st, seed=seed)
        for phase_i, p_val in enumerate([cosine_stochastok_prob(0.15), cosine_stochastok_prob(0.75)]):
            exp_ids = bpe.stochastok_expand(train_slice_ids, p_split=p_val, rng=st_rng)
            nw_e = (exp_ids.shape[0] - 1) // w_len
            xe = exp_ids[: nw_e * w_len].reshape(nw_e, w_len)
            ye = exp_ids[1 : nw_e * w_len + 1].reshape(nw_e, w_len)
            half_steps = n_steps_base // 2
            for s_local in range(half_steps):
                g_step = phase_i * half_steps + s_local
                win_idx = st_rng.integers(0, nw_e, size=bsz)
                lr = _cosine_lr(g_step, n_steps_base, peak_lr=4.5e-3)
                _, grads = mod_st.forward_backward(xe[win_idx], ye[win_idx], training=True)
                mod_st.step_adamw(grads, lr=lr)
        track2_bpb["stochastok_cosine"].append(_eval_exact_token_bytes_bpb(mod_st, test_ids, bpe.byte_lengths))

        oe_rng = np.random.default_rng(seed + 1000)
        oe_sched = [oe_rng.integers(0, n_win_train, size=bsz) for _ in range(n_steps_base)]
        cfg_oe = TransformerConfig(
            vocab_size=vocab_base,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            window_len=w_len,
            base_width_mup=d_model,
            head_kind="flat",
            use_oe_span=True,
            n_spans=max(span_lex.num_spans, 1),
            d_span=24,
        )
        mod_oe = CausalTransformerLM(cfg_oe, seed=seed)
        for step_i, win_idx in enumerate(oe_sched):
            lr = _cosine_lr(step_i, n_steps_base, peak_lr=4.5e-3)
            _, grads = mod_oe.forward_backward(
                x_windows[win_idx],
                y_windows[win_idx],
                span_ids=oe_windows[win_idx],
                training=True,
            )
            mod_oe.step_adamw(grads, lr=lr)
        track2_bpb["oe_span_sidecar"].append(
            _eval_exact_token_bytes_bpb(mod_oe, test_ids, bpe.byte_lengths, span_ids=test_oe_spans)
        )

    edge_sums_clean, doc_to_edges_clean = compute_self_excluded_stalks(hg_clean, hg_clean.doc_vectors)
    stalk_orig = gather_doc_self_excluded_stalks(0, hg_clean, hg_clean.doc_vectors, edge_sums_clean, doc_to_edges_clean)
    perturbed_bank = hg_clean.doc_vectors.copy()
    perturbed_bank[0] += 99.0
    edge_sums_pert, _ = compute_self_excluded_stalks(hg_clean, perturbed_bank)
    stalk_pert = gather_doc_self_excluded_stalks(0, hg_clean, perturbed_bank, edge_sums_pert, doc_to_edges_clean)
    self_inclusion_drift = float(np.max(np.abs(stalk_orig - stalk_pert)))

    canary_mod = trained_flat_models[0]
    x_can = test_ids[:w_len].reshape(1, w_len).copy()
    x_can_corrupt = x_can.copy()
    x_can_corrupt[0, w_len // 2 :] = (x_can_corrupt[0, w_len // 2 :] + 17) % vocab_base

    def _prefix_hidden(mod: CausalTransformerLM, seq_in: np.ndarray) -> np.ndarray:
        u_tab = mod.full_embedding_table()
        h = u_tab[seq_in].copy()
        dh = mod.cfg.d_model // mod.cfg.n_heads
        sc = 1.0 / math.sqrt(float(dh))
        cm = mod.causal_mask[:w_len, :w_len]
        for l_i in range(mod.cfg.n_layers):
            p = f"L{l_i}."
            ms = np.mean(h * h, axis=-1, keepdims=True)
            xn = h / np.sqrt(ms + 1e-5) * mod.params[p + "g_attn"]
            q = (xn @ mod.params[p + "Wq"]).reshape(1, w_len, mod.cfg.n_heads, dh).transpose(0, 2, 1, 3)
            k = (xn @ mod.params[p + "Wk"]).reshape(1, w_len, mod.cfg.n_heads, dh).transpose(0, 2, 1, 3)
            v = (xn @ mod.params[p + "Wv"]).reshape(1, w_len, mod.cfg.n_heads, dh).transpose(0, 2, 1, 3)
            att = np.where(cm[None, None, :, :], np.einsum("bhid,bhjd->bhij", q, k) * sc, -1e9)
            att = np.exp(att - np.max(att, axis=-1, keepdims=True)) * cm[None, None, :, :]
            att /= np.sum(att, axis=-1, keepdims=True)
            ct = np.einsum("bhij,bhjd->bhid", att, v).transpose(0, 2, 1, 3).reshape(1, w_len, mod.cfg.d_model)
            h = h + ct @ mod.params[p + "Wo"]
            ms2 = np.mean(h * h, axis=-1, keepdims=True)
            xm = h / np.sqrt(ms2 + 1e-5) * mod.params[p + "g_mlp"]
            u1 = xm @ mod.params[p + "W1"]
            u3 = xm @ mod.params[p + "W3"]
            sig = 1.0 / (1.0 + np.exp(-np.clip(u1, -30.0, 30.0)))
            h = h + (u1 * sig * u3) @ mod.params[p + "W2"]
        return h[0, : w_len // 2]

    future_canary_drift = float(
        np.max(np.abs(_prefix_hidden(canary_mod, x_can) - _prefix_hidden(canary_mod, x_can_corrupt)))
    )

    _, walk_random = prefix_causal_walk_order(hg_clean, mode="random", seed=13)
    _, walk_dense = prefix_causal_walk_order(hg_clean, mode="dense", seed=13)
    _, walk_null = prefix_causal_walk_order(hg_null, mode="hyper", seed=13)
    _, walk_hyper = prefix_causal_walk_order(hg_clean, mode="hyper", seed=13)

    tok_counts = np.zeros(vocab_base, dtype=np.int64)
    np.add.at(tok_counts, flat_train_ids, 1)
    freq_order = np.argsort(tok_counts)
    cum_mass = np.cumsum(tok_counts[freq_order]) / max(float(tok_counts.sum()), 1.0)
    rare_mask = np.zeros(vocab_base, dtype=bool)
    rare_mask[freq_order[cum_mass <= 0.15]] = True
    if int(np.sum(rare_mask)) < 32:
        rare_mask[freq_order[:64]] = True

    ds_doc_list: list[int] = []
    ds_x_list: list[np.ndarray] = []
    ds_y_list: list[np.ndarray] = []
    q_doc_list: list[int] = []
    q_x_list: list[np.ndarray] = []
    q_y_list: list[np.ndarray] = []

    _, doc_edges_null = compute_self_excluded_stalks(hg_null, hg_null.doc_vectors)
    for d_i in range(min(hg_clean.n_docs, 110)):
        seq = train_streams[d_i]
        if seq.shape[0] < 3 * w_len + 2:
            continue
        for w_off in (0, w_len):
            ds_x_list.append(seq[w_off : w_off + w_len])
            ds_y_list.append(seq[w_off + 1 : w_off + w_len + 1])
            ds_doc_list.extend([d_i] * w_len)
        if len(q_x_list) < 36 and len(doc_to_edges_clean[d_i]) >= 2:
            q_off = 2 * w_len
            q_x_list.append(seq[q_off : q_off + w_len])
            q_y_list.append(seq[q_off + 1 : q_off + w_len + 1])
            q_doc_list.extend([d_i] * w_len)

    ds_x_arr = np.asarray(ds_x_list, dtype=np.int32)
    ds_y_arr = np.asarray(ds_y_list, dtype=np.int32).reshape(-1)
    ds_doc_arr = np.asarray(ds_doc_list, dtype=np.int32)
    q_x_arr = np.asarray(q_x_list, dtype=np.int32)
    q_y_arr = np.asarray(q_y_list, dtype=np.int32).reshape(-1)

    q_hyper_nbs: list[np.ndarray] = []
    q_null_nbs: list[np.ndarray] = []
    for d_q in q_doc_list:
        h_nbs: set[int] = set()
        for e_i in doc_to_edges_clean[d_q]:
            h_nbs.update(hg_clean.edges[e_i])
        h_nbs.discard(d_q)
        n_nbs: set[int] = set()
        for e_i in doc_edges_null[d_q]:
            n_nbs.update(hg_null.edges[e_i])
        n_nbs.discard(d_q)
        q_hyper_nbs.append(np.asarray(sorted(h_nbs), dtype=np.int32))
        q_null_nbs.append(np.asarray(sorted(n_nbs), dtype=np.int32))

    def _final_hidden_and_probs(
        mod: CausalTransformerLM,
        x_in: np.ndarray,
        y_in: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        u_tab = mod.full_embedding_table()
        bsz_l, wl = x_in.shape
        h = u_tab[x_in].copy()
        dh = mod.cfg.d_model // mod.cfg.n_heads
        sc = 1.0 / math.sqrt(float(dh))
        cm = mod.causal_mask[:wl, :wl]
        cos_l, sin_l = mod.cos[:wl], mod.sin[:wl]
        from hadukenn_core.transformer_lm import _apply_rope, _rmsnorm_fwd

        for l_i in range(mod.cfg.n_layers):
            p = f"L{l_i}."
            xn, _, _ = _rmsnorm_fwd(h, mod.params[p + "g_attn"])
            q = _apply_rope(
                (xn @ mod.params[p + "Wq"]).reshape(bsz_l, wl, mod.cfg.n_heads, dh).transpose(0, 2, 1, 3),
                cos_l,
                sin_l,
            )
            k = _apply_rope(
                (xn @ mod.params[p + "Wk"]).reshape(bsz_l, wl, mod.cfg.n_heads, dh).transpose(0, 2, 1, 3),
                cos_l,
                sin_l,
            )
            v = (xn @ mod.params[p + "Wv"]).reshape(bsz_l, wl, mod.cfg.n_heads, dh).transpose(0, 2, 1, 3)
            att = np.where(cm[None, None, :, :], np.einsum("bhid,bhjd->bhij", q, k) * sc, -1e9)
            att = np.exp(att - np.max(att, axis=-1, keepdims=True)) * cm[None, None, :, :]
            att /= np.sum(att, axis=-1, keepdims=True)
            ct = np.einsum("bhij,bhjd->bhid", att, v).transpose(0, 2, 1, 3).reshape(bsz_l, wl, mod.cfg.d_model)
            h = h + ct @ mod.params[p + "Wo"]
            xm, _, _ = _rmsnorm_fwd(h, mod.params[p + "g_mlp"])
            u1 = xm @ mod.params[p + "W1"]
            u3 = xm @ mod.params[p + "W3"]
            sig = 1.0 / (1.0 + np.exp(-np.clip(u1, -30.0, 30.0)))
            h = h + (u1 * sig * u3) @ mod.params[p + "W2"]
        hf, _, _ = _rmsnorm_fwd(h, mod.params["g_final"])
        hf_flat = hf.reshape(-1, mod.cfg.d_model)
        logits = hf_flat @ u_tab.T
        mx = np.max(logits, axis=1, keepdims=True)
        ex = np.exp(logits - mx)
        probs = ex[np.arange(hf_flat.shape[0]), y_in] / np.sum(ex, axis=1)
        return hf_flat, probs

    e2_lm_seeds: list[float] = []
    e2_unscoped_seeds: list[float] = []
    e2_null_seeds: list[float] = []
    e2_hyper_seeds: list[float] = []
    for mod in trained_flat_models:
        ds_h, _ = _final_hidden_and_probs(mod, ds_x_arr, ds_y_arr)
        q_h, q_p = _final_hidden_and_probs(mod, q_x_arr, q_y_arr)
        e2_res = evaluate_scoped_knnlm(
            query_states=q_h,
            query_targets=q_y_arr,
            query_lm_probs=q_p,
            query_doc_hyper_neighbors=q_hyper_nbs,
            query_doc_null_neighbors=q_null_nbs,
            datastore_keys=ds_h,
            datastore_vals=ds_y_arr,
            datastore_doc_ids=ds_doc_arr,
            token_byte_lengths=bpe.byte_lengths,
            rare_token_mask=rare_mask,
        )
        e2_lm_seeds.append(e2_res["lm_rare_bpb"])
        e2_unscoped_seeds.append(e2_res["unscoped_knn_rare_bpb"])
        e2_null_seeds.append(e2_res["chodrow_null_knn_rare_bpb"])
        e2_hyper_seeds.append(e2_res["hyper_scoped_knn_rare_bpb"])

    e4_sheaf_B: list[float] = []
    e4_cos_B: list[float] = []
    e4_lex_B: list[float] = []
    e4_sheaf_A: list[float] = []
    e4_tie_rates: list[float] = []
    e4_loop_rates: list[float] = []
    for seed in seeds:
        e4 = evaluate_sheaf_contradiction_e4(hg_clean, clean_train_docs, seed=seed)
        e4_sheaf_A.append(e4["sheaf_auc_setA"])
        e4_sheaf_B.append(e4["sheaf_auc_setB"])
        e4_cos_B.append(e4["embed_cos_auc_setB"])
        e4_lex_B.append(e4["lexical_auc_setB"])
        e4_tie_rates.append(e4["tie_1v1_abstention_rate"])
        e4_loop_rates.append(e4["loop_2v1_recovery_rate"])

    d_stalk = 24
    m_slots = 4
    proj_rng = np.random.default_rng(99)
    tok_proj = (proj_rng.standard_normal((vocab_base, d_stalk)) / math.sqrt(float(d_stalk))).astype(np.float32)
    doc_lex_bank = np.zeros((hg_clean.n_docs, d_stalk), dtype=np.float32)
    for d_i, seq in enumerate(train_streams):
        tf = np.zeros(vocab_base, dtype=np.float32)
        if seq.size > 0:
            np.add.at(tf, seq, 1.0)
            tf = np.log1p(tf) / max(float(np.linalg.norm(np.log1p(tf))), 1e-8)
        doc_lex_bank[d_i] = tf @ tok_proj

    es_clean_lex, de_clean_lex = compute_self_excluded_stalks(hg_clean, doc_lex_bank)
    es_null_lex, de_null_lex = compute_self_excluded_stalks(hg_null, doc_lex_bank)
    sim_lex = doc_lex_bank @ doc_lex_bank.T
    np.fill_diagonal(sim_lex, -1e9)

    cw_x: list[np.ndarray] = []
    cw_y: list[np.ndarray] = []
    cw_stalk_sheaf: list[np.ndarray] = []
    cw_stalk_null: list[np.ndarray] = []
    cw_stalk_dense: list[np.ndarray] = []

    for d_i, seq in enumerate(train_streams):
        n_tok = int(seq.shape[0])
        if n_tok < 2 * w_len + 2:
            continue
        w0_tf = np.zeros(vocab_base, dtype=np.float32)
        np.add.at(w0_tf, seq[:w_len], 1.0)
        w0_vec = (np.log1p(w0_tf) / max(float(np.linalg.norm(np.log1p(w0_tf))), 1e-8)) @ tok_proj

        sh_nb = gather_doc_self_excluded_stalks(
            d_i, hg_clean, doc_lex_bank, es_clean_lex, de_clean_lex, m_slots=m_slots - 1
        )
        nu_nb = gather_doc_self_excluded_stalks(
            d_i, hg_null, doc_lex_bank, es_null_lex, de_null_lex, m_slots=m_slots - 1
        )
        top_dense = np.argsort(-sim_lex[d_i])[: m_slots - 1]
        de_nb = doc_lex_bank[top_dense]

        null_prior = doc_lex_bank[(d_i + 37) % hg_clean.n_docs]
        for w_idx in range(1, min(n_tok // w_len, 4)):
            off = w_idx * w_len
            if off + w_len + 1 > n_tok:
                break
            cw_x.append(seq[off : off + w_len])
            cw_y.append(seq[off + 1 : off + w_len + 1])
            cw_stalk_sheaf.append(np.vstack([w0_vec[None, :], sh_nb]))
            cw_stalk_null.append(np.vstack([null_prior[None, :], nu_nb]))
            cw_stalk_dense.append(np.vstack([w0_vec[None, :], de_nb]))

    cw_x_arr = np.asarray(cw_x, dtype=np.int32)
    cw_y_arr = np.asarray(cw_y, dtype=np.int32)
    cw_sh_arr = np.asarray(cw_stalk_sheaf, dtype=np.float32)
    cw_nu_arr = np.asarray(cw_stalk_null, dtype=np.float32)
    cw_de_arr = np.asarray(cw_stalk_dense, dtype=np.float32)

    n_cw = cw_x_arr.shape[0]
    split_cw = int(0.80 * n_cw)
    tr_cw_x, ev_cw_x = cw_x_arr[:split_cw], cw_x_arr[split_cw:]
    tr_cw_y, ev_cw_y = cw_y_arr[:split_cw], cw_y_arr[split_cw:]
    tr_sh, ev_sh = cw_sh_arr[:split_cw], cw_sh_arr[split_cw:]
    tr_nu, ev_nu = cw_nu_arr[:split_cw], cw_nu_arr[split_cw:]
    tr_de, ev_de = cw_de_arr[:split_cw], cw_de_arr[split_cw:]

    track4_bpb: dict[str, list[float]] = {
        "window_isolated": [],
        "kvreg_chodrow_null": [],
        "kvreg_dense_cosine": [],
        "kvreg_diagonal_sheaf": [],
    }
    n_steps_cw = 72
    ev_cw_bytes = float(np.sum(bpe.byte_lengths[ev_cw_y]))

    for seed in seeds:
        cw_rng = np.random.default_rng(seed + 4000)
        cw_sched = [cw_rng.integers(0, split_cw, size=bsz) for _ in range(n_steps_cw)]
        for arm_key, tr_st, ev_st, trime_p in [
            ("window_isolated", None, None, 0.0),
            ("kvreg_chodrow_null", tr_nu, ev_nu, 0.0),
            ("kvreg_dense_cosine", tr_de, ev_de, 0.0),
            ("kvreg_diagonal_sheaf", tr_sh, ev_sh, 0.0),
        ]:
            cfg_kv = TransformerConfig(
                vocab_size=vocab_base,
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                d_ff=d_ff,
                window_len=w_len,
                base_width_mup=d_model,
                head_kind="flat",
                kv_prefix_slots=m_slots if tr_st is not None else 0,
                d_stalk=d_stalk,
                trime_local_drop=trime_p,
            )
            mod_kv = CausalTransformerLM(cfg_kv, seed=seed)
            for step_i, idx_b in enumerate(cw_sched):
                lr = _cosine_lr(step_i, n_steps_cw, peak_lr=4.5e-3)
                ps = tr_st[idx_b] if tr_st is not None else None
                _, grads = mod_kv.forward_backward(tr_cw_x[idx_b], tr_cw_y[idx_b], prefix_stalks=ps, training=True)
                mod_kv.step_adamw(grads, lr=lr)

            mean_ev_nll, _ = mod_kv.forward_backward(
                ev_cw_x,
                ev_cw_y,
                prefix_stalks=ev_st,
                training=False,
            )
            bpb_cw = float((mean_ev_nll * float(ev_cw_y.size)) / (max(ev_cw_bytes, 1.0) * math.log(2.0)))
            track4_bpb[arm_key].append(bpb_cw)

    rng_hw = np.random.default_rng(2026)
    hw_configs = [
        ("T38_Medium_32k", 512, 576, 32768, 256, 128, 3),
        ("LargeVocab_64k", 512, 576, 65536, 512, 128, 3),
    ]
    track5_hw: list[dict[str, float | str | int]] = []
    for label, n_b, d_b, v_b, nc_b, s_b, k_b in hw_configs:
        H_b = (rng_hw.standard_normal((n_b, d_b)) * 0.1).astype(np.float32)
        E_b = (rng_hw.standard_normal((v_b, d_b)) * 0.1).astype(np.float32)
        E_cl_b = E_b.reshape(nc_b, s_b, d_b).copy()
        C_rt_b = E_cl_b.mean(axis=1).copy()
        tc_b = rng_hw.integers(0, nc_b, size=n_b, dtype=np.int32)
        tm_b = rng_hw.integers(0, s_b, size=n_b, dtype=np.int32)
        neg_b = rng_hw.integers(0, nc_b, size=(n_b, k_b), dtype=np.int32)
        y_b = tc_b * s_b + tm_b

        _ = run_tiled_cce_avx512(H_b, E_b, y_b, tile_V=512, compute_grad=True)
        _ = run_cluster_is_avx512(H_b, E_cl_b, C_rt_b, tc_b, tm_b, neg_b, compute_grad=True)

        t0 = time.perf_counter()
        for _ in range(3):
            loss_cce, _, _ = run_tiled_cce_avx512(H_b, E_b, y_b, tile_V=512, compute_grad=True)
        ms_cce = ((time.perf_counter() - t0) / 3.0) * 1000.0

        t0 = time.perf_counter()
        for _ in range(3):
            loss_is, _, _ = run_cluster_is_avx512(H_b, E_cl_b, C_rt_b, tc_b, tm_b, neg_b, compute_grad=True)
        ms_is = ((time.perf_counter() - t0) / 3.0) * 1000.0

        loss_ref64, _, _ = tiled_cce_fwd_bwd(
            H_b[:32], E_b, y_b[:32], tile_V=512, compute_grad=False, use_native=False
        )
        cce_max_err = float(np.max(np.abs(loss_cce[:32] - loss_ref64)))

        dense_logit_mib = (n_b * v_b * 4) / (1024.0 * 1024.0)
        is_logit_mib = (n_b * (nc_b + (1 + k_b) * s_b) * 4) / (1024.0 * 1024.0)
        cce_buf_mib = (512 * 4 * 2) / (1024.0 * 1024.0)

        track5_hw.append(
            {
                "config": label,
                "N_tokens": n_b,
                "d_model": d_b,
                "vocab_size": v_b,
                "cluster_shape": f"{nc_b}x{s_b}",
                "avx512_tiled_cce_ms": round(ms_cce, 3),
                "avx512_cluster_is_ms": round(ms_is, 3),
                "native_c_speedup_x": round(ms_cce / max(ms_is, 1e-6), 2),
                "cluster_is_tok_per_s": round((n_b * 1000.0) / max(ms_is, 1e-6), 1),
                "tiled_cce_tok_per_s": round((n_b * 1000.0) / max(ms_cce, 1e-6), 1),
                "dense_logit_buffer_mib": round(dense_logit_mib, 3),
                "cluster_is_buffer_mib": round(is_logit_mib, 3),
                "tiled_cce_scratch_mib": round(cce_buf_mib, 4),
                "cce_vs_fp64_max_err": round(cce_max_err, 8),
            }
        )

    tb_path = Path(__file__).resolve().parent / "torch_bridge.py"
    tb_audit = audit_kernel_source(tb_path.read_text(encoding="utf-8"))

    elapsed_s = time.perf_counter() - t_start

    results: dict[str, object] = {
        "schema": "sheaf-verify.local-stack-bench.v1",
        "version": "0.3.0",
        "seeds": list(seeds),
        "elapsed_seconds": round(elapsed_s, 2),
        "corpus_summary": {
            "raw_train_candidates": len(train_raw),
            "clean_train_docs": hg_clean.n_docs,
            "val_docs": len(val_docs),
            "test_docs": len(test_docs),
            "dropped_near_duplicates": len(hg_clean.dropped_near_duplicates),
            "fixed_train_bytes_per_seed": actual_train_bytes,
            "vocab_base": vocab_base,
            "pmi_spans_extracted": span_lex.num_spans,
            "native_avx512_loaded": get_fast_head_lib() is not None,
            "pytorch_bridge_anticheat_ok": tb_audit.ok,
            "pytorch_bridge_violations": list(tb_audit.violations),
        },
        "partitions": {
            "mass_random": part_rand.verify_bijection(),
            "syntagmatic_spectral": part_synt.verify_bijection(),
            "paradigmatic_spectral": part_para.verify_bijection(),
        },
        "track1_output_heads": {
            "per_seed_bpb": {k: [round(x, 6) for x in v] for k, v in track1_bpb.items()},
            "paradigmatic_cluster_var_ratio": {
                k: round(float(np.mean(v)), 6) for k, v in track1_geom.items()
            },
            "comparisons": {
                "fp2_para_vs_fp2_synt": _adjudicate_paired(
                    track1_bpb["H_fp2_syntagmatic"], track1_bpb["H_fp2_paradigmatic"]
                ),
                "fp2_para_vs_fp2_rand": _adjudicate_paired(
                    track1_bpb["H_fp2_mass_random"], track1_bpb["H_fp2_paradigmatic"]
                ),
                "cluster_is_vs_fp2_synt": _adjudicate_paired(
                    track1_bpb["H_fp2_syntagmatic"], track1_bpb["H_cluster_is_paradigmatic"]
                ),
                "cluster_is_vs_fp2_para": _adjudicate_paired(
                    track1_bpb["H_fp2_paradigmatic"], track1_bpb["H_cluster_is_paradigmatic"]
                ),
                "cluster_is_vs_flat_tied": _adjudicate_paired(
                    track1_bpb["P_flat_tied"], track1_bpb["H_cluster_is_paradigmatic"]
                ),
                "tied_root_vs_fp2_synt": _adjudicate_paired(
                    track1_bpb["H_fp2_syntagmatic"], track1_bpb["H_tied_root_shortcut"]
                ),
            },
        },
        "track2_tokenization_sidecars": {
            "n_steps_base_equal_bytes": n_steps_base,
            "n_steps_merged_equal_bytes": n_steps_merged,
            "step_starvation_pct": round(100.0 * (1.0 - n_steps_merged / float(n_steps_base)), 2),
            "mean_residual_share_unshrunk": round(float(np.mean(res_shares_unshrunk)), 6),
            "mean_residual_share_count_shrunk": round(float(np.mean(res_shares_shrunk)), 6),
            "per_seed_bpb": {k: [round(x, 6) for x in v] for k, v in track2_bpb.items()},
            "comparisons": {
                "plain_bpe_vs_merged_unshrunk": _adjudicate_paired(
                    track2_bpb["merged_span_unshrunk"], track2_bpb["plain_bpe"]
                ),
                "count_shrunk_vs_merged_unshrunk": _adjudicate_paired(
                    track2_bpb["merged_span_unshrunk"], track2_bpb["merged_span_count_shrunk"]
                ),
                "stochastok_vs_plain_bpe": _adjudicate_paired(
                    track2_bpb["plain_bpe"], track2_bpb["stochastok_cosine"]
                ),
                "oe_span_vs_plain_bpe": _adjudicate_paired(
                    track2_bpb["plain_bpe"], track2_bpb["oe_span_sidecar"]
                ),
                "oe_span_vs_merged_unshrunk": _adjudicate_paired(
                    track2_bpb["merged_span_unshrunk"], track2_bpb["oe_span_sidecar"]
                ),
            },
        },
        "track3_corpus_hypergraph": {
            "n_edges": len(hg_clean.edges),
            "edge_kind_counts": {
                k: sum(1 for ek in hg_clean.edge_kinds if ek == k)
                for k in ("entity", "span", "topic")
            },
            "chodrow_degree_preserved": True,
            "chodrow_size_preserved": True,
            "chodrow_edge_overlap_frac": round(overlap_edges, 6),
            "self_inclusion_stalk_drift": self_inclusion_drift,
            "future_window_canary_drift": future_canary_drift,
            "e1_walk_packing": {
                "random": {k: round(v, 5) for k, v in walk_random.items()},
                "dense": {k: round(v, 5) for k, v in walk_dense.items()},
                "chodrow_null": {k: round(v, 5) for k, v in walk_null.items()},
                "hyper_clean": {k: round(v, 5) for k, v in walk_hyper.items()},
            },
            "e2_scoped_knnlm": {
                "per_seed_rare_bpb": {
                    "lm_only": [round(x, 6) for x in e2_lm_seeds],
                    "unscoped_knn": [round(x, 6) for x in e2_unscoped_seeds],
                    "chodrow_null_knn": [round(x, 6) for x in e2_null_seeds],
                    "hyper_scoped_knn": [round(x, 6) for x in e2_hyper_seeds],
                },
                "hyper_vs_unscoped": _adjudicate_paired(e2_unscoped_seeds, e2_hyper_seeds),
                "hyper_vs_chodrow_null": _adjudicate_paired(e2_null_seeds, e2_hyper_seeds),
                "hyper_vs_lm_only": _adjudicate_paired(e2_lm_seeds, e2_hyper_seeds),
            },
            "e4_sheaf_contradiction": {
                "mean_sheaf_auc_setA": round(float(np.mean(e4_sheaf_A)), 6),
                "mean_sheaf_auc_setB": round(float(np.mean(e4_sheaf_B)), 6),
                "mean_embed_cos_auc_setB": round(float(np.mean(e4_cos_B)), 6),
                "mean_lexical_auc_setB": round(float(np.mean(e4_lex_B)), 6),
                "sheaf_vs_embed_cos_setB_gain": round(float(np.mean(e4_sheaf_B) - np.mean(e4_cos_B)), 6),
                "tie_1v1_abstention_rate": round(float(np.mean(e4_tie_rates)), 6),
                "loop_2v1_recovery_rate": round(float(np.mean(e4_loop_rates)), 6),
            },
        },
        "track4_sheaf_kv_prefix": {
            "per_seed_bpb": {k: [round(x, 6) for x in v] for k, v in track4_bpb.items()},
            "comparisons": {
                "sheaf_vs_window_isolated": _adjudicate_paired(
                    track4_bpb["window_isolated"], track4_bpb["kvreg_diagonal_sheaf"]
                ),
                "sheaf_vs_chodrow_null": _adjudicate_paired(
                    track4_bpb["kvreg_chodrow_null"], track4_bpb["kvreg_diagonal_sheaf"]
                ),
                "sheaf_vs_dense_cosine": _adjudicate_paired(
                    track4_bpb["kvreg_dense_cosine"], track4_bpb["kvreg_diagonal_sheaf"]
                ),
                "chodrow_null_vs_window_isolated": _adjudicate_paired(
                    track4_bpb["window_isolated"], track4_bpb["kvreg_chodrow_null"]
                ),
            },
        },
        "track5_avx512_silicon": track5_hw,
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_file = output_dir / "T40_RESULTS.json"
        out_file.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return results


if __name__ == "__main__":  # pragma: no cover
    out = Path(__file__).resolve().parents[2] / "iterations" / "t40-local-empirical-stack"
    summary = run_full_empirical_suite(output_dir=out)
    print(json.dumps(summary, indent=2))
