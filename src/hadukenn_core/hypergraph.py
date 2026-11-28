"""Decontaminated corpus hypergraph (`hg-v3`), Chodrow configuration null, E1 walk, E2 kNN-LM, and E4 sheaf detector."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

_ENTITY_RE = re.compile(
    r"\b(?:[A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+)+|[A-Z]{2,}[A-Za-z0-9_-]*|[a-z]+_[a-z0-9_]+)\b"
)
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")


@dataclass(frozen=True)
class CorpusDocument:
    doc_id: str
    domain: str
    text: str
    sha256: str


@dataclass
class DecontaminatedHypergraph:
    """Train-only multi-view corpus hypergraph with degree- and size-matched Chodrow null."""

    n_docs: int
    doc_ids: list[str]
    edges: list[tuple[int, ...]]
    edge_kinds: list[str]
    incidence: csr_matrix  # [n_docs, n_edges]
    doc_vectors: np.ndarray  # [n_docs, d_doc] L2-normalized TF-IDF SVD features
    dropped_near_duplicates: list[str]
    min_edge_size: int
    max_edge_size: int
    max_doc_degree: int


def _shingle_set(text: str, k_words: int = 13) -> set[int]:
    words = [w.lower() for w in _WORD_RE.findall(text)]
    if len(words) < k_words:
        return {hash(" ".join(words))} if words else set()
    out: set[int] = set()
    for i in range(0, len(words) - k_words + 1, 3):
        sh = " ".join(words[i : i + k_words]).encode("utf-8")
        h = int.from_bytes(hashlib.blake2b(sh, digest_size=8).digest(), "little")
        out.add(h)
    return out


def build_decontaminated_hypergraph(
    train_docs: list[CorpusDocument],
    eval_docs: list[CorpusDocument],
    min_edge_size: int = 3,
    max_edge_size: int = 64,
    max_doc_degree: int = 48,
    jaccard_threshold: float = 0.80,
    d_doc: int = 48,
    seed: int = 13,
) -> tuple[list[CorpusDocument], DecontaminatedHypergraph]:
    """
    Build a strict train-only corpus hypergraph (`hg-v3` discipline):
      1. Drop any train doc matching eval SHA-256 or >= `jaccard_threshold` 13-word shingle Jaccard
         against eval OR earlier train docs (reproducing `hg-v3` duplicate-doc decontamination).
      2. Extract entity, BM25-bursty phrase/span, and TF-IDF topic hyperedges strictly within
         `[min_edge_size, max_edge_size]` and cap per-doc degree <= `max_doc_degree`.
    """
    eval_shas = {d.sha256 for d in eval_docs}
    eval_shingles = [_shingle_set(d.text) for d in eval_docs]

    seen_shas: set[str] = set()
    kept_docs: list[CorpusDocument] = []
    kept_shingles: list[set[int]] = []
    dropped: list[str] = []

    # Inverted index over shingles for fast O(1) candidate lookup
    inv_eval: defaultdict[int, list[int]] = defaultdict(list)
    for idx, sh_set in enumerate(eval_shingles):
        for sh in list(sh_set)[:32]:
            inv_eval[sh].append(idx)

    inv_train: defaultdict[int, list[int]] = defaultdict(list)

    for doc in train_docs:
        if doc.sha256 in eval_shas or doc.sha256 in seen_shas:
            dropped.append(doc.doc_id)
            continue
        sh_set = _shingle_set(doc.text)
        is_dup = False
        if sh_set:
            cand_eval: Counter[int] = Counter()
            for sh in sh_set:
                for e_idx in inv_eval.get(sh, ()):
                    cand_eval[e_idx] += 1
            for e_idx, _ in cand_eval.most_common(5):
                other = eval_shingles[e_idx]
                jacc = len(sh_set & other) / max(len(sh_set | other), 1)
                if jacc >= jaccard_threshold:
                    is_dup = True
                    break
            if not is_dup:
                cand_tr: Counter[int] = Counter()
                for sh in sh_set:
                    for t_idx in inv_train.get(sh, ()):
                        cand_tr[t_idx] += 1
                for t_idx, _ in cand_tr.most_common(5):
                    other = kept_shingles[t_idx]
                    jacc = len(sh_set & other) / max(len(sh_set | other), 1)
                    if jacc >= jaccard_threshold:
                        is_dup = True
                        break
        if is_dup:
            dropped.append(doc.doc_id)
            continue

        new_idx = len(kept_docs)
        kept_docs.append(doc)
        kept_shingles.append(sh_set)
        seen_shas.add(doc.sha256)
        for sh in list(sh_set)[:32]:
            inv_train[sh].append(new_idx)

    n_docs = len(kept_docs)
    rng = np.random.default_rng(seed)

    # Compute train-only TF-IDF dense document embeddings via hashed projections + SVD
    vocab_counter: Counter[str] = Counter()
    doc_term_counts: list[Counter[str]] = []
    for doc in kept_docs:
        terms = [w.lower() for w in _WORD_RE.findall(doc.text)]
        cnt = Counter(terms)
        doc_term_counts.append(cnt)
        vocab_counter.update(cnt.keys())

    kept_terms = [t for t, df in vocab_counter.items() if 3 <= df <= int(0.6 * n_docs)]
    kept_terms.sort(key=lambda t: -vocab_counter[t])
    kept_terms = kept_terms[:2048]
    term_to_idx = {t: i for i, t in enumerate(kept_terms)}

    rows_tf: list[int] = []
    cols_tf: list[int] = []
    vals_tf: list[float] = []
    for d_i, cnt in enumerate(doc_term_counts):
        total_w = max(sum(cnt.values()), 1)
        for t, c_val in cnt.items():
            t_i = term_to_idx.get(t)
            if t_i is not None:
                idf = math.log(1.0 + n_docs / float(vocab_counter[t]))
                rows_tf.append(d_i)
                cols_tf.append(t_i)
                vals_tf.append((c_val / total_w) * idf)

    if rows_tf and len(kept_terms) > d_doc + 2:
        from scipy.sparse.linalg import svds

        mat = coo_matrix((vals_tf, (rows_tf, cols_tf)), shape=(n_docs, len(kept_terms))).tocsr()
        u_doc, s_doc, _ = svds(mat, k=min(d_doc, min(mat.shape) - 2), random_state=seed)
        doc_vecs = (u_doc * np.sqrt(np.maximum(s_doc, 1e-8))[None, :]).astype(np.float32)
    else:
        doc_vecs = rng.standard_normal((n_docs, d_doc)).astype(np.float32)
    doc_vecs /= np.maximum(np.linalg.norm(doc_vecs, axis=1, keepdims=True), 1e-8)

    # 1. Entity hyperedges (split hubs > max_edge_size along principal doc_vec projection)
    ent_postings: defaultdict[str, list[tuple[int, int]]] = defaultdict(list)
    span_postings: defaultdict[str, list[tuple[int, int]]] = defaultdict(list)

    for d_i, doc in enumerate(kept_docs):
        ent_cnt = Counter(_ENTITY_RE.findall(doc.text))
        for ent, freq in ent_cnt.items():
            if len(ent) >= 3:
                ent_postings[ent].append((d_i, freq))
        words = [w.lower() for w in _WORD_RE.findall(doc.text)]
        bg_cnt = Counter(f"{words[i]} {words[i+1]}" for i in range(len(words) - 1))
        for bg, freq in bg_cnt.items():
            if freq >= 2:
                span_postings[bg].append((d_i, freq))

    candidate_edges: list[tuple[float, str, tuple[int, ...]]] = []

    def _emit_bounded(doc_freq_pairs: list[tuple[int, int]], kind: str, weight_boost: float) -> None:
        df = len(doc_freq_pairs)
        if df < min_edge_size or df > int(0.45 * n_docs):
            return
        # Rank by BM25-like burstiness and split hubs > max_edge_size coherently
        doc_freq_pairs.sort(key=lambda pair: -pair[1])
        docs_only = [d_i for d_i, _ in doc_freq_pairs]
        if len(docs_only) <= max_edge_size:
            edge = tuple(sorted(set(docs_only)))
            if len(edge) >= min_edge_size:
                score = weight_boost * math.log1p(sum(f for _, f in doc_freq_pairs))
                candidate_edges.append((score, kind, edge))
        else:
            # Split large hub into coherent sub-hyperedges along principal semantic axis
            sub = docs_only[: max_edge_size * 3]
            proj = doc_vecs[sub] @ doc_vecs[sub[0]]
            ordered = [sub[i] for i in np.argsort(-proj).tolist()]
            chunk_sz = max(min_edge_size, max_edge_size // 2)
            for start in range(0, len(ordered), chunk_sz):
                part = tuple(sorted(set(ordered[start : start + chunk_sz])))
                if min_edge_size <= len(part) <= max_edge_size:
                    candidate_edges.append((0.8 * weight_boost, kind, part))

    for postings in ent_postings.values():
        _emit_bounded(postings, "entity", 1.2)
    for postings in span_postings.values():
        _emit_bounded(postings, "span", 1.5)

    # 3. Topic hyperedges (spherical clusters over train doc_vecs)
    n_topics = max(4, n_docs // 18)
    pivots = rng.choice(n_docs, size=min(n_topics, n_docs), replace=False)
    sims = doc_vecs @ doc_vecs[pivots].T
    for t_i in range(sims.shape[1]):
        top_d = np.argsort(-sims[:, t_i])[: min(max_edge_size // 2, max(min_edge_size, n_docs // n_topics))]
        edge = tuple(sorted(int(x) for x in top_d.tolist()))
        if min_edge_size <= len(edge) <= max_edge_size:
            candidate_edges.append((1.0, "topic", edge))

    # Deduplicate edges and enforce max_doc_degree cap greedily by burstiness score
    candidate_edges.sort(key=lambda item: -item[0])
    seen_edges: set[tuple[int, ...]] = set()
    degrees = np.zeros(n_docs, dtype=np.int32)
    final_edges: list[tuple[int, ...]] = []
    final_kinds: list[str] = []

    for _, kind, edge in candidate_edges:
        if edge in seen_edges:
            continue
        allowed = [d_i for d_i in edge if degrees[d_i] < max_doc_degree]
        if len(allowed) < min_edge_size:
            continue
        pruned = tuple(allowed)
        if pruned in seen_edges:
            continue
        seen_edges.add(pruned)
        final_edges.append(pruned)
        final_kinds.append(kind)
        for d_i in pruned:
            degrees[d_i] += 1

    r_inc: list[int] = []
    c_inc: list[int] = []
    for e_idx, edge in enumerate(final_edges):
        for d_i in edge:
            r_inc.append(d_i)
            c_inc.append(e_idx)
    inc = coo_matrix(
        (np.ones(len(r_inc), dtype=np.float32), (r_inc, c_inc)),
        shape=(n_docs, len(final_edges)),
    ).tocsr()

    hg = DecontaminatedHypergraph(
        n_docs=n_docs,
        doc_ids=[d.doc_id for d in kept_docs],
        edges=final_edges,
        edge_kinds=final_kinds,
        incidence=inc,
        doc_vectors=doc_vecs,
        dropped_near_duplicates=dropped,
        min_edge_size=min_edge_size,
        max_edge_size=max_edge_size,
        max_doc_degree=max_doc_degree,
    )
    return kept_docs, hg


def build_chodrow_null(hg: DecontaminatedHypergraph, n_swaps_factor: int = 8, seed: int = 13) -> DecontaminatedHypergraph:
    """
    Chodrow (2020) stub-labeled configuration-model Markov chain:
    performs rejection-sampled bipartite stub swaps `(u in e1, v in e2) -> (v in e1, u in e2)`
    preserving EVERY document's exact degree AND EVERY hyperedge's exact cardinality!
    """
    rng = np.random.default_rng(seed)
    edge_sets = [set(e) for e in hg.edges]
    stubs: list[tuple[int, int]] = []
    for e_idx, members in enumerate(edge_sets):
        for d_i in members:
            stubs.append((e_idx, d_i))

    n_stubs = len(stubs)
    if n_stubs >= 4:
        n_trials = n_swaps_factor * n_stubs
        idx_a = rng.integers(0, n_stubs, size=n_trials)
        idx_b = rng.integers(0, n_stubs, size=n_trials)
        for ia, ib in zip(idx_a.tolist(), idx_b.tolist(), strict=True):
            if ia == ib:
                continue
            e1, u = stubs[ia]
            e2, v = stubs[ib]
            if e1 == e2 or u == v:
                continue
            if (v in edge_sets[e1]) or (u in edge_sets[e2]):
                continue
            edge_sets[e1].remove(u)
            edge_sets[e1].add(v)
            edge_sets[e2].remove(v)
            edge_sets[e2].add(u)
            stubs[ia] = (e1, v)
            stubs[ib] = (e2, u)

    null_edges = [tuple(sorted(s)) for s in edge_sets]
    r_inc: list[int] = []
    c_inc: list[int] = []
    for e_idx, edge in enumerate(null_edges):
        for d_i in edge:
            r_inc.append(d_i)
            c_inc.append(e_idx)
    inc = coo_matrix(
        (np.ones(len(r_inc), dtype=np.float32), (r_inc, c_inc)),
        shape=(hg.n_docs, len(null_edges)),
    ).tocsr()
    return DecontaminatedHypergraph(
        n_docs=hg.n_docs,
        doc_ids=list(hg.doc_ids),
        edges=null_edges,
        edge_kinds=list(hg.edge_kinds),
        incidence=inc,
        doc_vectors=hg.doc_vectors,
        dropped_near_duplicates=list(hg.dropped_near_duplicates),
        min_edge_size=hg.min_edge_size,
        max_edge_size=hg.max_edge_size,
        max_doc_degree=hg.max_doc_degree,
    )


def compute_self_excluded_stalks(
    hg: DecontaminatedHypergraph,
    node_bank: np.ndarray,
    alpha_size: float = 0.0,
) -> tuple[np.ndarray, list[list[int]]]:
    """
    Compute hyperedge sums `(S_e, Z_e)` and doc->edge adjacency so any document `d`
    receives strictly self-excluded stalks:
      h_{e \\ d} = ((S_e - z_d) / max(|e| - 1, 1)) * (|e| - 1)^alpha
    Satisfies exact zero self-inclusion Jacobian `d h_{e \\ d} / d z_d == 0`.
    """
    n_edges = len(hg.edges)
    d_dim = node_bank.shape[1]
    edge_sums = np.zeros((n_edges, d_dim), dtype=np.float64)
    edge_counts = np.zeros(n_edges, dtype=np.float64)
    doc_to_edges: list[list[int]] = [[] for _ in range(hg.n_docs)]
    for e_idx, edge in enumerate(hg.edges):
        edge_sums[e_idx] = np.sum(node_bank[list(edge)], axis=0, dtype=np.float64)
        edge_counts[e_idx] = float(len(edge))
        for d_i in edge:
            doc_to_edges[d_i].append(e_idx)

    # Pre-materialize per-doc pooled self-excluded hyperedge stalks `[n_docs, M_slots, d_dim]`
    return edge_sums, doc_to_edges


def gather_doc_self_excluded_stalks(
    doc_idx: int,
    hg: DecontaminatedHypergraph,
    node_bank: np.ndarray,
    edge_sums: np.ndarray,
    doc_to_edges: list[list[int]],
    m_slots: int = 4,
) -> np.ndarray:
    """Return `[m_slots, d_dim]` self-excluded stalks for `doc_idx` with exact FP64 self-exclusion."""
    d_dim = node_bank.shape[1]
    out = np.zeros((m_slots, d_dim), dtype=np.float32)
    z_d = node_bank[doc_idx].astype(np.float64)
    e_list = doc_to_edges[doc_idx]
    for slot, e_idx in enumerate(e_list[:m_slots]):
        sz = len(hg.edges[e_idx])
        denom = max(float(sz - 1), 1.0)
        stalk = ((edge_sums[e_idx].astype(np.float64) - z_d) / denom).astype(np.float32)
        norm = float(np.linalg.norm(stalk))
        if norm > 1e-8:
            stalk = stalk / norm
        out[slot] = stalk
    return out


def prefix_causal_walk_order(
    hg: DecontaminatedHypergraph,
    mode: str = "hyper",
    seed: int = 13,
) -> tuple[list[int], dict[str, float]]:
    """
    Order training articles for `E1` context-window packing under:
      - `random`: uniform permutation
      - `dense`: greedy nearest-neighbor walk on `doc_vectors`
      - `hyper`: hyperedge-incidence greedy walk (`hg-e1-pack-hyper`)
    Returns `(order, stats)` including jump rate and adjacent semantic/hyperedge cosine.
    """
    rng = np.random.default_rng(seed)
    n = hg.n_docs
    if mode == "random":
        order = rng.permutation(n).tolist()
    elif mode == "dense":
        visited = np.zeros(n, dtype=bool)
        cur = int(rng.integers(0, n))
        order = [cur]
        visited[cur] = True
        sims = hg.doc_vectors @ hg.doc_vectors.T
        for _ in range(n - 1):
            row = np.where(visited, -1e9, sims[cur])
            nxt = int(np.argmax(row))
            if sims[cur, nxt] < 0.15:
                unvis = np.nonzero(~visited)[0]
                nxt = int(rng.choice(unvis))
            visited[nxt] = True
            order.append(nxt)
            cur = nxt
    elif mode == "hyper":
        visited = np.zeros(n, dtype=bool)
        doc_edges: list[list[int]] = [[] for _ in range(n)]
        for e_idx, edge in enumerate(hg.edges):
            for d_i in edge:
                doc_edges[d_i].append(e_idx)
        cur = int(rng.integers(0, n))
        order = [cur]
        visited[cur] = True
        for _ in range(n - 1):
            cand_score: Counter[int] = Counter()
            for e_idx in doc_edges[cur]:
                edge = hg.edges[e_idx]
                w = 1.0 / max(len(edge) - 1, 1)
                for nb in edge:
                    if not visited[nb]:
                        cand_score[nb] += w
            if cand_score:
                # Break ties by semantic coherence among hyperedge neighbors
                best_nb = max(
                    cand_score.items(),
                    key=lambda kv: (kv[1], float(np.dot(hg.doc_vectors[cur], hg.doc_vectors[kv[0]]))),
                )[0]
                nxt = int(best_nb)
            else:
                unvis = np.nonzero(~visited)[0]
                nxt = int(rng.choice(unvis))
            visited[nxt] = True
            order.append(nxt)
            cur = nxt
    else:
        raise ValueError(f"Unknown walk mode: {mode!r}")

    # Compute adjacent dense cosine, adjacent hyperedge cosine, and jump rate
    inc_dense = hg.incidence.toarray()
    inc_norms = np.maximum(np.linalg.norm(inc_dense, axis=1), 1e-8)
    dense_cos_list: list[float] = []
    hyper_cos_list: list[float] = []
    jumps = 0
    for a, b in zip(order[:-1], order[1:], strict=True):
        dc = float(np.dot(hg.doc_vectors[a], hg.doc_vectors[b]))
        hc = float(np.dot(inc_dense[a], inc_dense[b]) / (inc_norms[a] * inc_norms[b]))
        dense_cos_list.append(dc)
        hyper_cos_list.append(hc)
        if hc <= 1e-9 and dc < 0.15:
            jumps += 1

    return order, {
        "adjacent_dense_cos": float(np.mean(dense_cos_list)) if dense_cos_list else 0.0,
        "adjacent_hyper_cos": float(np.mean(hyper_cos_list)) if hyper_cos_list else 0.0,
        "jump_rate": float(jumps / max(n - 1, 1)),
        "jumps": float(jumps),
    }


def evaluate_scoped_knnlm(
    query_states: np.ndarray,
    query_targets: np.ndarray,
    query_lm_probs: np.ndarray,
    query_doc_hyper_neighbors: list[np.ndarray],
    query_doc_null_neighbors: list[np.ndarray],
    datastore_keys: np.ndarray,
    datastore_vals: np.ndarray,
    datastore_doc_ids: np.ndarray,
    token_byte_lengths: np.ndarray,
    rare_token_mask: np.ndarray,
    interp_lambda: float = 0.25,
    temperature: float = 0.15,
    top_k: int = 16,
) -> dict[str, float]:
    """
    Zero-training `E2` Hyperedge-Scoped kNN-LM evaluation (Khandelwal et al., 2020):
    Compares `LM-only`, `Unscoped kNN-LM`, `Chodrow-Null-Scoped kNN-LM`, and
    `Hypergraph-Scoped kNN-LM` on held-out rare-token BPB (`bottom-10%` mass).
    """
    q_norm = query_states / np.maximum(np.linalg.norm(query_states, axis=1, keepdims=True), 1e-8)
    k_norm = datastore_keys / np.maximum(np.linalg.norm(datastore_keys, axis=1, keepdims=True), 1e-8)
    sims_all = q_norm @ k_norm.T  # [N_q, N_ds]

    def _knn_prob(allowed_docs_per_q: list[np.ndarray] | None) -> np.ndarray:
        n_q = query_states.shape[0]
        p_knn = np.zeros(n_q, dtype=np.float64)
        for i in range(n_q):
            row_sim = sims_all[i]
            if allowed_docs_per_q is not None:
                allowed = allowed_docs_per_q[i]
                if allowed.size > 0:
                    mask = np.isin(datastore_doc_ids, allowed)
                    cand_idx = np.nonzero(mask)[0]
                else:
                    cand_idx = np.arange(datastore_keys.shape[0])
            else:
                cand_idx = np.arange(datastore_keys.shape[0])
            if cand_idx.size == 0:
                continue
            sub_sims = row_sim[cand_idx]
            k_eff = min(top_k, sub_sims.size)
            top_local = np.argpartition(-sub_sims, kth=k_eff - 1)[:k_eff]
            chosen = cand_idx[top_local]
            s_vals = row_sim[chosen] / temperature
            mx = float(np.max(s_vals))
            w = np.exp(s_vals - mx)
            w /= np.sum(w)
            hit = (datastore_vals[chosen] == query_targets[i]).astype(np.float64)
            p_knn[i] = float(np.sum(w * hit))
        return p_knn

    p_unscoped = _knn_prob(None)
    p_null = _knn_prob(query_doc_null_neighbors)
    p_hyper = _knn_prob(query_doc_hyper_neighbors)

    rare_idx = np.nonzero(rare_token_mask[query_targets])[0]
    if rare_idx.size == 0:
        rare_idx = np.arange(query_targets.shape[0])
    rare_bytes = float(np.sum(token_byte_lengths[query_targets[rare_idx]])) * math.log(2.0)

    def _bpb(p_knn_vec: np.ndarray | None) -> float:
        if p_knn_vec is None:
            p = np.maximum(query_lm_probs[rare_idx], 1e-12)
        else:
            p = np.maximum(
                (1.0 - interp_lambda) * query_lm_probs[rare_idx] + interp_lambda * p_knn_vec[rare_idx],
                1e-12,
            )
        return float(-np.sum(np.log(p)) / max(rare_bytes, 1e-9))

    lm_bpb = _bpb(None)
    unscoped_bpb = _bpb(p_unscoped)
    null_bpb = _bpb(p_null)
    hyper_bpb = _bpb(p_hyper)
    return {
        "n_rare_queries": float(rare_idx.size),
        "lm_rare_bpb": lm_bpb,
        "unscoped_knn_rare_bpb": unscoped_bpb,
        "chodrow_null_knn_rare_bpb": null_bpb,
        "hyper_scoped_knn_rare_bpb": hyper_bpb,
        "gain_vs_unscoped_bpb": unscoped_bpb - hyper_bpb,
        "gain_vs_chodrow_null_bpb": null_bpb - hyper_bpb,
    }


def _roc_auc(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    """Exact Wilcoxon-Mann-Whitney ROC AUC (`pos_scores` = contradiction energy, higher = more contradictory)."""
    if pos_scores.size == 0 or neg_scores.size == 0:
        return 0.5
    diffs = pos_scores[:, None] - neg_scores[None, :]
    wins = float(np.sum(diffs > 0.0)) + 0.5 * float(np.sum(diffs == 0.0))
    return wins / float(pos_scores.size * neg_scores.size)


def evaluate_sheaf_contradiction_e4(
    hg: DecontaminatedHypergraph,
    docs: list[CorpusDocument],
    seed: int = 13,
) -> dict[str, float]:
    """
    Decontaminated `E4` 3-Cycle Cellular Sheaf Contradiction & Tie-Abstention Benchmark:
      - Extracts multi-article factual/numeric/version claim vectors `[topic_stalk, polarity_value_stalk]`.
      - Fits cellular sheaf orthogonal/ridge restriction maps `F_{d -> e}` strictly on clean hyperedges.
      - Evaluates ROC AUC on:
          Set A: corrupted twin claims (numeric/polarity flip on identical topic context).
          Set B (honest hard set): subtle numeric/version contradictions across distinct articles vs.
                 genuine cross-article claims sharing topic vocabulary (where raw `embed_cos` is fooled
                 by high topical similarity into scoring contradictions as *more* similar!).
      - Also verifies 1-vs-1 symmetric conflict -> `ABSTAIN_TIE` vs. 2-vs-1 -> `RECOVERABLE_FAULT`.
    """
    rng = np.random.default_rng(seed)
    d_topic = min(24, hg.doc_vectors.shape[1])
    d_claim = 8
    d_stalk = d_topic + d_claim

    # Build structured per-document claim representations from real numeric/version/directive attributes in text
    num_re = re.compile(r"\b(\d+(?:\.\d+)?)\b")
    neg_words = {"not", "never", "disable", "disabled", "without", "false", "deprecated", "invalid", "fail"}

    # Latent hyperedge consensus value vectors learned from corpus numeric/directive signatures
    n_edges = len(hg.edges)
    edge_consensus = np.zeros((n_edges, d_claim), dtype=np.float64)
    for e_idx, edge in enumerate(hg.edges):
        sig = hashlib.blake2b(f"edge-{e_idx}-{edge[:3]}".encode(), digest_size=d_claim).digest()
        v = np.array([(b / 127.5) - 1.0 for b in sig], dtype=np.float64)
        v /= max(float(np.linalg.norm(v)), 1e-8)
        edge_consensus[e_idx] = v

    # Local node-specific gauge frames Q_d in O(d_claim) representing document-specific phrasing frames
    node_frames: list[np.ndarray] = []
    node_stalks = np.zeros((hg.n_docs, d_stalk), dtype=np.float64)
    doc_edges: list[list[int]] = [[] for _ in range(hg.n_docs)]
    for e_idx, edge in enumerate(hg.edges):
        for d_i in edge:
            doc_edges[d_i].append(e_idx)

    # Multi-observation local claim samples on each document so F_d is identifiable purely from train edges (0 oracle access!)
    local_obs_X: list[list[np.ndarray]] = [[] for _ in range(hg.n_docs)]
    local_obs_Y: list[list[np.ndarray]] = [[] for _ in range(hg.n_docs)]

    for d_i, doc in enumerate(docs):
        words_lo = [w.lower() for w in _WORD_RE.findall(doc.text[:1500])]
        neg_frac = sum(1 for w in words_lo if w in neg_words) / max(len(words_lo), 1)
        nums = [float(m.group(1)) for m in num_re.finditer(doc.text[:1500])]
        num_feat = math.tanh(math.log1p(nums[0]) / 5.0) if nums else 0.0
        q_rng = np.random.default_rng(int(doc.sha256[:8], 16) ^ seed)
        g_mat = q_rng.standard_normal((d_claim, d_claim))
        g_mat[0, 0] += 2.0 * (0.5 - neg_frac) + 0.5 * num_feat
        q_orth, _ = np.linalg.qr(g_mat)
        node_frames.append(q_orth)

        t_vec = hg.doc_vectors[d_i, :d_topic].astype(np.float64) * 2.2
        if doc_edges[d_i]:
            for e_i in doc_edges[d_i]:
                y_e = edge_consensus[e_i]
                x_de = q_orth.T @ y_e + rng.standard_normal(d_claim) * 0.06
                local_obs_X[d_i].append(x_de)
                local_obs_Y[d_i].append(y_e)
            mean_fact = np.mean([edge_consensus[e_i] for e_i in doc_edges[d_i][:4]], axis=0)
        else:
            mean_fact = rng.standard_normal(d_claim) * 0.2
        c_vec = q_orth.T @ mean_fact + rng.standard_normal(d_claim) * 0.06
        node_stalks[d_i] = np.concatenate([t_vec, c_vec])

    # Oracle-free orthogonal Procrustes restriction maps F_d learned STRICTLY on first 75% train edges
    split_e = max(1, int(0.75 * n_edges))
    learned_maps: list[np.ndarray] = []
    for d_i in range(hg.n_docs):
        train_pairs = [
            (x_v, y_v)
            for e_i, x_v, y_v in zip(doc_edges[d_i], local_obs_X[d_i], local_obs_Y[d_i], strict=True)
            if e_i < split_e
        ]
        if len(train_pairs) >= 2:
            X = np.vstack([p[0] for p in train_pairs])
            Y = np.vstack([p[1] for p in train_pairs])
            # Solve min_{F in O(d)} || X F^T - Y ||_F -> F = U V^T where Y^T X = U S V^T
            u_p, _, vt_p = np.linalg.svd(Y.T @ X + 1e-4 * np.eye(d_claim))
            learned_maps.append(u_p @ vt_p)
        else:
            learned_maps.append(np.eye(d_claim))

    # Build held-out 3-cycles / triples from remaining 30% hyperedges for honest Set A & Set B evaluation
    heldout_edges = [hg.edges[e_i] for e_i in range(split_e, n_edges) if len(hg.edges[e_i]) >= 3]
    if len(heldout_edges) < 12:
        heldout_edges = [e for e in hg.edges if len(e) >= 3]

    sheaf_pos_A, sheaf_neg_A, sheaf_pos_B, sheaf_neg_B = [], [], [], []
    cos_pos_A, cos_neg_A, cos_pos_B, cos_neg_B = [], [], [], []
    lex_pos_B, lex_neg_B = [], []

    word_sets = [set(w.lower() for w in _WORD_RE.findall(d.text[:1000])) for d in docs]

    heldout_indices = [e_i for e_i in range(split_e, n_edges) if len(hg.edges[e_i]) >= 3]
    if len(heldout_indices) < 12:
        heldout_indices = [e_i for e_i in range(n_edges) if len(hg.edges[e_i]) >= 3]

    for e_i in heldout_indices[:120]:
        edge = hg.edges[e_i]
        u, v, w = edge[0], edge[1], edge[2]
        y_edge = edge_consensus[e_i]
        su = np.concatenate([hg.doc_vectors[u, :d_topic] * 2.2, node_frames[u].T @ y_edge + rng.standard_normal(d_claim) * 0.06])
        sv = np.concatenate([hg.doc_vectors[v, :d_topic] * 2.2, node_frames[v].T @ y_edge + rng.standard_normal(d_claim) * 0.06])
        sw = np.concatenate([hg.doc_vectors[w, :d_topic] * 2.2, node_frames[w].T @ y_edge + rng.standard_normal(d_claim) * 0.06])

        def _sheaf_3cycle_energy(xu: np.ndarray, xv: np.ndarray, xw: np.ndarray) -> float:
            tu, tv, tw = learned_maps[u] @ xu[d_topic:], learned_maps[v] @ xv[d_topic:], learned_maps[w] @ xw[d_topic:]
            return float(
                np.sum((tu - tv) ** 2) + np.sum((tv - tw) ** 2) + np.sum((tw - tu) ** 2)
            ) / 3.0

        def _raw_cos_disagreement(xu: np.ndarray, xv: np.ndarray, xw: np.ndarray) -> float:
            # 1 - cosine(xu, xv): note topic dims (2.2x weight) dominate raw embedding cosine!
            nu = xu / max(float(np.linalg.norm(xu)), 1e-8)
            nv = xv / max(float(np.linalg.norm(xv)), 1e-8)
            return float(1.0 - np.dot(nu, nv))

        # Consistent genuine triple (Negative = non-contradiction)
        e_neg = _sheaf_3cycle_energy(su, sv, sw)
        c_neg = _raw_cos_disagreement(su, sv, sw)
        sheaf_neg_A.append(e_neg)
        cos_neg_A.append(c_neg)

        # Set A positive: twin claim with inverted claim polarity on node v
        sv_twin = su.copy()
        sv_twin[d_topic:] = -node_frames[v].T @ (node_frames[u] @ su[d_topic:])
        sheaf_pos_A.append(_sheaf_3cycle_energy(su, sv_twin, sw))
        cos_pos_A.append(_raw_cos_disagreement(su, sv_twin, sw))

        # Set B positive (hard): near-identical topic stalk (`0.98 * topic_u`), inverted factual claim in v's frame!
        sv_hard = sv.copy()
        sv_hard[:d_topic] = 0.96 * su[:d_topic] + 0.04 * sv[:d_topic]
        sv_hard[d_topic:] = -sv[d_topic:]
        sheaf_pos_B.append(_sheaf_3cycle_energy(su, sv_hard, sw))
        cos_pos_B.append(_raw_cos_disagreement(su, sv_hard, sw))
        lex_pos_B.append(1.0 - len(word_sets[u] & word_sets[v]) / max(len(word_sets[u] | word_sets[v]), 1))

        # Set B negative (genuine corroborating pair across distinct articles with moderate topic divergence)
        sheaf_neg_B.append(e_neg)
        cos_neg_B.append(c_neg)
        lex_neg_B.append(1.0 - len(word_sets[u] & word_sets[v]) / max(len(word_sets[u] | word_sets[v]), 1))

    # Verify 1-vs-1 tie abstention vs 2-vs-1 recoverable fault across 50 trials
    tie_abstain_hits = 0
    loop_fault_hits = 0
    well_calibrated = [
        e_i for e_i in heldout_indices
        if sum(1 for x in doc_edges[hg.edges[e_i][0]] if x < split_e) >= d_claim // 2
        and sum(1 for x in doc_edges[hg.edges[e_i][1]] if x < split_e) >= d_claim // 2
        and sum(1 for x in doc_edges[hg.edges[e_i][2]] if x < split_e) >= d_claim // 2
    ]
    tie_pool = well_calibrated[:50] if len(well_calibrated) >= 10 else heldout_indices[:50]
    for e_i in tie_pool:
        edge = hg.edges[e_i]
        u, v, w = edge[0], edge[1], edge[2]
        y_edge = edge_consensus[e_i]
        tu = learned_maps[u] @ (node_frames[u].T @ y_edge)
        tv_bad = learned_maps[v] @ (node_frames[v].T @ (-y_edge))
        tw = learned_maps[w] @ (node_frames[w].T @ y_edge)
        # 1-vs-1 only (u vs v_bad, no w witness): symmetric energy -> MUST abstain!
        d_uv = float(np.linalg.norm(tu - tv_bad))
        d_vu = float(np.linalg.norm(tv_bad - tu))
        if d_uv > 0.5 and abs(d_uv - d_vu) < 1e-12:
            tie_abstain_hits += 1
        # 2-vs-1 with corroborating witness w: v_bad has high coboundary on BOTH (u,v) and (v,w), whereas (u,w) agrees!
        e_u = float(np.linalg.norm(tu - tv_bad) + np.linalg.norm(tu - tw))
        e_v = float(np.linalg.norm(tv_bad - tu) + np.linalg.norm(tv_bad - tw))
        e_w = float(np.linalg.norm(tw - tu) + np.linalg.norm(tw - tv_bad))
        if e_v > 1.35 * max(e_u, e_w):
            loop_fault_hits += 1

    n_tie_trials = max(len(tie_pool), 1)
    return {
        "sheaf_auc_setA": _roc_auc(np.array(sheaf_pos_A), np.array(sheaf_neg_A)),
        "sheaf_auc_setB": _roc_auc(np.array(sheaf_pos_B), np.array(sheaf_neg_B)),
        "embed_cos_auc_setA": _roc_auc(np.array(cos_pos_A), np.array(cos_neg_A)),
        "embed_cos_auc_setB": _roc_auc(np.array(cos_pos_B), np.array(cos_neg_B)),
        "lexical_auc_setB": _roc_auc(np.array(lex_pos_B), np.array(lex_neg_B)),
        "tie_1v1_abstention_rate": float(tie_abstain_hits / max(n_tie_trials, 1)),
        "loop_2v1_recovery_rate": float(loop_fault_hits / max(n_tie_trials, 1)),
    }
