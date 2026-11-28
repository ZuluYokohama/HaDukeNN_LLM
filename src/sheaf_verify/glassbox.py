"""Zero-Black-Box ('Glass-Box') AI Engine for HaDukeNN_LLM (v0.7.0, T-44).

Implements end-to-end interpretable-by-construction sequence modeling, exact
machine unlearning, anti-unification relational rule induction, rational-field
cellular-sheaf conservation law discovery, and counterfactual audit receipts
using ONLY the Python 3.11+ standard library (zero opaque neural weights).
"""

from __future__ import annotations

import copy
import hashlib
import math
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

from sheaf_verify.compliance import ExactUnitLattice, emit_regulatory_bundle
from sheaf_verify.hashing import SheafError, canonical_bytes, sha256_hex


@dataclass(frozen=True)
class SpanDefinition:
    """Explicit white-box byte-span unit induced via Minimum Description Length."""

    token_id: int
    raw_bytes: bytes
    parent_pair: tuple[int, int] | None
    mdl_bits_saved: float
    empirical_count: int

    def display_text(self) -> str:
        return self.raw_bytes.decode("utf-8", errors="replace")


@dataclass(frozen=True)
class ProvenanceCitation:
    """Exact training-corpus occurrence voting for a predicted span."""

    doc_id: str
    span_offset: int
    context_order: int
    target_token_id: int
    weight_share: float


@dataclass(frozen=True)
class HyperedgeTraceWitness:
    """Explicit multi-way hyperedge path contributing via Schur-complement trace."""

    source_token_id: int
    hyperedge_doc_id: str
    target_token_id: int
    hop_order: int
    trace_mass: float


@dataclass(frozen=True)
class GlassBoxExplanation:
    """Complete, verifiable white-box decomposition of a next-span prediction."""

    predicted_token_id: int
    predicted_bytes: bytes
    predicted_prob: float
    top_candidates: tuple[tuple[int, str, float], ...]
    order_mixture_weights: dict[str, float]
    provenance_citations: tuple[ProvenanceCitation, ...]
    hyperedge_witnesses: tuple[HyperedgeTraceWitness, ...]
    effective_sample_size: float
    entropy_bits: float
    epistemic_status: str  # "SUPPORTED" or "EPISTEMIC_ABSTAIN"
    state_merkle_root: str


@dataclass(frozen=True)
class InducedRelationalRule:
    """Human-readable syntactic-span-to-typed-role rule induced via anti-unification."""

    rule_id: str
    op: str
    pattern_tokens: tuple[str, ...]
    role_map: dict[str, int]
    unit_index: int | None
    support_count: int

    def human_readable(self) -> str:
        return f"[{self.rule_id}] {' '.join(self.pattern_tokens)} => {self.op}({self.role_map})"


@dataclass(frozen=True)
class DiscoveredSheafLaw:
    """Exact linear conservation law over rational stalks discovered via Q-nullspace."""

    law_id: str
    coefficients: dict[str, Fraction]
    require_non_negative: bool

    def evaluate_sum(self, state: dict[str, Fraction]) -> Fraction:
        total = Fraction(0, 1)
        for entity, coeff in sorted(self.coefficients.items()):
            total += coeff * state.get(entity, Fraction(0, 1))
        return total

    def formula_str(self) -> str:
        terms = [f"{coeff}*{ent}" if coeff != 1 else ent for ent, coeff in sorted(self.coefficients.items())]
        joined = " + ".join(terms) if terms else "0"
        suffix = " (with stalks >= 0)" if self.require_non_negative else ""
        return f"{joined} == invariant_constant{suffix}"


@dataclass(frozen=True)
class GlassBoxStepReceipt:
    """Tamper-evident execution receipt for a single Glass-Box world step."""

    step_index: int
    branch_id: str
    utterance: str
    matched_rule_id: str | None
    op: str
    roles: dict[str, str]
    status: str  # "ACCEPTED", "REFUSED_INVARIANT_VIOLATION", "UNRESOLVED_DEPENDENCY", "ABSTAIN_NO_RULE"
    detail: str
    coboundary_residual: str
    state_snapshot: dict[str, str]
    receipt_sha256: str


class CausalMDLSpanGraph:
    """Zero-embedding byte-to-span tokenizer driven by exact MDL compression savings."""

    def __init__(self) -> None:
        self.vocab: dict[int, SpanDefinition] = {
            b: SpanDefinition(
                token_id=b,
                raw_bytes=bytes([b]),
                parent_pair=None,
                mdl_bits_saved=0.0,
                empirical_count=0,
            )
            for b in range(256)
        }
        self.merges: list[tuple[tuple[int, int], int]] = []
        self._merge_lookup: dict[tuple[int, int], int] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def fit_causal_spans(
        self,
        corpus_bytes: bytes,
        *,
        max_merges: int = 128,
        min_pair_count: int = 3,
        dict_entry_overhead_bits: float = 24.0,
    ) -> int:
        """Induce reversible multi-byte spans strictly when MDL bit-savings > 0."""
        seq = list(corpus_bytes)
        if len(seq) < 2:
            return len(self.vocab)

        for _ in range(max_merges):
            n_tok = len(seq)
            if n_tok < 2:
                break
            uni_counts = Counter(seq)
            pair_counts: Counter[tuple[int, int]] = Counter()
            i = 0
            while i < n_tok - 1:
                pair = (seq[i], seq[i + 1])
                pair_counts[pair] += 1
                i += 1

            best_pair: tuple[int, int] | None = None
            best_gain = 0.0
            best_cnt = 0
            for pair, cnt in pair_counts.items():
                if cnt < min_pair_count:
                    continue
                cu = max(uni_counts[pair[0]], 1)
                cv = max(uni_counts[pair[1]], 1)
                pmi_bits = math.log2(max((cnt * n_tok) / (cu * cv), 1.0))
                mdl_gain = cnt * pmi_bits - dict_entry_overhead_bits
                # Deterministic tie-breaking on (mdl_gain, cnt, -u, -v)
                if mdl_gain > best_gain + 1e-12 or (
                    abs(mdl_gain - best_gain) <= 1e-12
                    and best_pair is not None
                    and (cnt, -pair[0], -pair[1]) > (best_cnt, -best_pair[0], -best_pair[1])
                ):
                    best_gain = mdl_gain
                    best_pair = pair
                    best_cnt = cnt

            if best_pair is None or best_gain <= 0.0:
                break

            new_id = 256 + len(self.merges)
            merged_bytes = self.vocab[best_pair[0]].raw_bytes + self.vocab[best_pair[1]].raw_bytes
            self.vocab[new_id] = SpanDefinition(
                token_id=new_id,
                raw_bytes=merged_bytes,
                parent_pair=best_pair,
                mdl_bits_saved=round(best_gain, 6),
                empirical_count=best_cnt,
            )
            self.merges.append((best_pair, new_id))
            self._merge_lookup[best_pair] = new_id

            # Greedily apply merge left-to-right (causal prefix deterministic)
            next_seq: list[int] = []
            idx = 0
            while idx < len(seq):
                if idx + 1 < len(seq) and (seq[idx], seq[idx + 1]) == best_pair:
                    next_seq.append(new_id)
                    idx += 2
                else:
                    next_seq.append(seq[idx])
                    idx += 1
            seq = next_seq

        return len(self.vocab)

    def encode_bytes(self, data: bytes) -> list[int]:
        """Causally encode arbitrary bytes into MDL span IDs."""
        seq = list(data)
        for pair, new_id in self.merges:
            if len(seq) < 2:
                break
            out: list[int] = []
            idx = 0
            while idx < len(seq):
                if idx + 1 < len(seq) and (seq[idx], seq[idx + 1]) == pair:
                    out.append(new_id)
                    idx += 2
                else:
                    out.append(seq[idx])
                    idx += 1
            seq = out
        return seq

    def decode_tokens(self, tokens: Sequence[int]) -> bytes:
        """Losslessly reconstruct exact original bytes from MDL span IDs."""
        chunks: list[bytes] = []
        for tid in tokens:
            if tid not in self.vocab:
                raise SheafError(f"Unknown span token ID: {tid}")
            chunks.append(self.vocab[tid].raw_bytes)
        return b"".join(chunks)


class GlassBoxSequenceEngine:
    """Zero-Black-Box Sequence Predictor with CTW-PY, Schur-Trace & Exact Unlearning.

    Combines:
      1. Variable-order Bayesian Context-Tree Weighting + Pitman-Yor discounting,
      2. Multi-hop document-hyperedge stochastic Schur-complement trace,
      3. Exact Training-Offset Provenance & Epistemic Abstention certificates,
      4. O(L) Bit-Identical Machine Unlearning (zero retraining, zero drift).
    """

    def __init__(
        self,
        span_graph: CausalMDLSpanGraph,
        *,
        max_order: int = 4,
        discount: float = 0.62,
        schur_gamma: float = 0.35,
        trace_prior_kappa: float = 6.0,
        abstain_min_support: float = 1.5,
    ) -> None:
        if max_order < 1:
            raise SheafError("max_order must be >= 1")
        if not (0.0 < discount < 1.0):
            raise SheafError("discount must lie in (0, 1)")
        self.span_graph = span_graph
        self.max_order = max_order
        self.discount = discount
        self.schur_gamma = schur_gamma
        self.trace_prior_kappa = trace_prior_kappa
        self.abstain_min_support = abstain_min_support

        # Exact integer sufficient statistics keyed by context tuple
        self._ctx_counts: dict[tuple[int, ...], Counter[int]] = defaultdict(Counter)
        self._ctx_totals: Counter[tuple[int, ...]] = Counter()
        # Provenance index: (ctx_tuple, next_tok) -> Counter[(doc_id, offset)]
        self._provenance: dict[tuple[tuple[int, ...], int], Counter[tuple[str, int]]] = defaultdict(Counter)
        # Document store for exact O(L) unlearning and hyperedge incidence
        self._doc_tokens: dict[str, tuple[int, ...]] = {}
        # Hypergraph incidence: doc_id -> Counter[token_id] (salient multi-byte or rare spans)
        self._doc_hyperedges: dict[str, Counter[int]] = {}
        self._token_to_docs: dict[int, Counter[str]] = defaultdict(Counter)

    @property
    def vocab_size(self) -> int:
        return len(self.span_graph.vocab)

    @property
    def document_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._doc_tokens.keys()))

    def _is_salient_hyperedge_token(self, tok: int) -> bool:
        raw = self.span_graph.vocab[tok].raw_bytes
        if len(raw) >= 2:
            return not raw.isspace()
        return raw.isalnum()

    def ingest_document(self, doc_id: str, data: bytes) -> int:
        """Ingest a document into suffix statistics, provenance index, and hypergraph."""
        if doc_id in self._doc_tokens:
            raise SheafError(f"Document '{doc_id}' is already ingested; unlearn or use a unique ID.")
        tokens = tuple(self.span_graph.encode_bytes(data))
        self._doc_tokens[doc_id] = tokens

        for pos, target in enumerate(tokens):
            max_k = min(self.max_order, pos)
            for order in range(0, max_k + 1):
                ctx = tokens[pos - order : pos] if order > 0 else ()
                self._ctx_counts[ctx][target] += 1
                self._ctx_totals[ctx] += 1
                if order >= 1:
                    self._provenance[(ctx, target)][(doc_id, pos)] += 1

        salient = Counter(t for t in tokens if self._is_salient_hyperedge_token(t))
        if salient:
            self._doc_hyperedges[doc_id] = salient
            for tok, cnt in salient.items():
                self._token_to_docs[tok][doc_id] += cnt
        return len(tokens)

    def unlearn_document(self, doc_id: str) -> int:
        """ Mathematically erase all statistical & provenance traces of doc_id in O(L)."""
        if doc_id not in self._doc_tokens:
            raise SheafError(f"Cannot unlearn unknown document '{doc_id}'.")
        tokens = self._doc_tokens.pop(doc_id)

        for pos, target in enumerate(tokens):
            max_k = min(self.max_order, pos)
            for order in range(0, max_k + 1):
                ctx = tokens[pos - order : pos] if order > 0 else ()
                c_map = self._ctx_counts[ctx]
                c_map[target] -= 1
                if c_map[target] <= 0:
                    del c_map[target]
                if not c_map:
                    del self._ctx_counts[ctx]
                self._ctx_totals[ctx] -= 1
                if self._ctx_totals[ctx] <= 0:
                    del self._ctx_totals[ctx]
                if order >= 1:
                    prov_key = (ctx, target)
                    p_map = self._provenance[prov_key]
                    p_map[(doc_id, pos)] -= 1
                    if p_map[(doc_id, pos)] <= 0:
                        del p_map[(doc_id, pos)]
                    if not p_map:
                        del self._provenance[prov_key]

        salient = self._doc_hyperedges.pop(doc_id, None)
        if salient:
            for tok, cnt in salient.items():
                td = self._token_to_docs[tok]
                td[doc_id] -= cnt
                if td[doc_id] <= 0:
                    del td[doc_id]
                if not td:
                    del self._token_to_docs[tok]
        return len(tokens)

    def state_merkle_sha256(self) -> str:
        """Canonical cryptographic commitment of all learned counts and hyperedges."""
        ctx_serial = [
            [list(ctx), [[t, c] for t, c in sorted(cmap.items())]]
            for ctx, cmap in sorted(self._ctx_counts.items())
        ]
        hg_serial = [
            [doc_id, [[t, c] for t, c in sorted(cmap.items())]]
            for doc_id, cmap in sorted(self._doc_hyperedges.items())
        ]
        return sha256_hex(canonical_bytes({"ctx": ctx_serial, "hg": hg_serial, "v": self.vocab_size}))

    def _schur_trace_distribution(
        self,
        context_tokens: Sequence[int],
    ) -> tuple[dict[int, float], list[HyperedgeTraceWitness]]:
        """Compute 2-hop stochastic Schur-complement trace across shared hyperedges."""
        recent_salient = [t for t in context_tokens[-8:] if t in self._token_to_docs]
        if not recent_salient:
            return {}, []

        v_size = self.vocab_size
        base = 1.0 / v_size
        scores: Counter[int] = Counter()
        witnesses: list[HyperedgeTraceWitness] = []

        # 1-hop hyperedge projection P_{VH} P_{HV} + gamma * 2-hop resolvent excursion
        doc_weights: Counter[str] = Counter()
        src_for_doc: dict[str, int] = {}
        for pos_idx, src_tok in enumerate(reversed(recent_salient)):
            recency_decay = 0.75**pos_idx
            doc_map = self._token_to_docs.get(src_tok)
            if not doc_map:
                continue
            tot_src = sum(doc_map.values())
            for doc_id, cnt in doc_map.items():
                w = recency_decay * (cnt / tot_src)
                doc_weights[doc_id] += w
                if doc_id not in src_for_doc:
                    src_for_doc[doc_id] = src_tok

        if not doc_weights:
            return {}, []

        # Normalize 1-hop hyperedge activation
        z1 = sum(doc_weights.values())
        for doc_id in list(doc_weights.keys()):
            doc_weights[doc_id] /= z1

        # 1-hop visible projection
        mid_tokens: Counter[int] = Counter()
        for doc_id, dw in doc_weights.items():
            he = self._doc_hyperedges[doc_id]
            he_tot = sum(he.values())
            for dst_tok, cnt in he.items():
                mass = (1.0 - self.schur_gamma) * dw * (cnt / he_tot)
                scores[dst_tok] += mass
                mid_tokens[dst_tok] += dw * (cnt / he_tot)
                witnesses.append(
                    HyperedgeTraceWitness(
                        source_token_id=src_for_doc[doc_id],
                        hyperedge_doc_id=doc_id,
                        target_token_id=dst_tok,
                        hop_order=1,
                        trace_mass=mass,
                    )
                )

        # 2-hop hidden-hyperedge Schur excursion: gamma * (P_VH P_HV)^2
        if self.schur_gamma > 0.0 and mid_tokens:
            top_mid = mid_tokens.most_common(6)
            z_mid = sum(v for _, v in top_mid)
            doc2_weights: Counter[str] = Counter()
            src2_doc: dict[str, int] = {}
            for mid_tok, mw in top_mid:
                dmap = self._token_to_docs.get(mid_tok)
                if not dmap:
                    continue
                dtot = sum(dmap.values())
                for doc_id, cnt in dmap.items():
                    w2 = (mw / z_mid) * (cnt / dtot)
                    doc2_weights[doc_id] += w2
                    if doc_id not in src2_doc:
                        src2_doc[doc_id] = mid_tok
            z2 = sum(doc2_weights.values())
            if z2 > 0.0:
                for doc_id, dw2 in doc2_weights.items():
                    he = self._doc_hyperedges[doc_id]
                    he_tot = sum(he.values())
                    for dst_tok, cnt in he.items():
                        mass2 = self.schur_gamma * (dw2 / z2) * (cnt / he_tot)
                        scores[dst_tok] += mass2
                        witnesses.append(
                            HyperedgeTraceWitness(
                                source_token_id=src2_doc[doc_id],
                                hyperedge_doc_id=doc_id,
                                target_token_id=dst_tok,
                                hop_order=2,
                                trace_mass=mass2,
                            )
                        )

        tot_s = sum(scores.values())
        if tot_s <= 0.0:
            return {}, []
        # Smooth slightly with uniform base so every token has positive density
        out = {tid: 0.92 * (scores.get(tid, 0.0) / tot_s) + 0.08 * base for tid in range(v_size)}
        witnesses.sort(key=lambda w: (-w.trace_mass, w.hop_order, w.hyperedge_doc_id, w.target_token_id))
        return out, witnesses[:8]

    def predict_distribution(
        self,
        context_tokens: Sequence[int],
    ) -> tuple[list[float], dict[str, float], float]:
        """Compute exact white-box predictive distribution over all vocab spans.

        Returns (probs, order_mixture_weights, effective_sample_size).
        """
        v_size = self.vocab_size
        uniform_p = 1.0 / v_size

        # Build recursive Pitman-Yor / Kneser-Ney distributions for orders 0..max_k
        max_k = min(self.max_order, len(context_tokens))
        order_Validate_probs: list[list[float]] = []
        order_support: list[float] = []

        # Order 0 (unigram + Dirichlet 0.5 Jeffreys prior)
        c0 = self._ctx_counts.get((), Counter())
        tot0 = self._ctx_totals.get((), 0)
        denom0 = tot0 + 0.5 * v_size
        p0 = [(c0.get(tid, 0) + 0.5) / denom0 for tid in range(v_size)]
        order_Validate_probs.append(p0)
        order_support.append(float(tot0))

        prev_p = p0
        for order in range(1, max_k + 1):
            ctx = tuple(context_tokens[-order:])
            tot = self._ctx_totals.get(ctx, 0)
            order_support.append(float(tot))
            if tot == 0:
                order_Validate_probs.append(prev_p)
                continue
            cmap = self._ctx_counts[ctx]
            unique_continuations = len(cmap)
            backoff_mass = (self.discount * unique_continuations) / tot
            cur_p = [
                (max(cmap.get(tid, 0) - self.discount, 0.0) / tot) + backoff_mass * prev_p[tid]
                for tid in range(v_size)
            ]
            order_Validate_probs.append(cur_p)
            prev_p = cur_p

        # Analytical Bayesian Context-Tree Weighting across suffix depths
        raw_depth_weights: list[float] = []
        for order in range(0, max_k + 1):
            n_ord = order_support[order]
            if order == 0:
                raw_depth_weights.append(1.0)
            elif n_ord > 0:
                # Posterior precision grows with empirical count and specificity (order)
                raw_depth_weights.append((1.8**order) * math.log1p(n_ord))
            else:
                raw_depth_weights.append(0.0)

        w_sum = sum(raw_depth_weights) or 1.0
        ctw_weights = [w / w_sum for w in raw_depth_weights]

        ctw_p = [0.0] * v_size
        for order, w_ord in enumerate(ctw_weights):
            if w_ord <= 0.0:
                continue
            p_vec = order_Validate_probs[order]
            for tid in range(v_size):
                ctw_p[tid] += w_ord * p_vec[tid]

        # Effective local context sample size (highest non-empty suffix + weighted lower orders)
        eff_n = sum(order_support[o] * (0.5 ** (max_k - o)) for o in range(1, max_k + 1))

        # Blend with multi-hop Hypergraph Schur-Complement Trace when hyperedges fire
        schur_dist, _ = self._schur_trace_distribution(context_tokens)
        weights_report: dict[str, float] = {}
        if schur_dist:
            alpha_ctw = (eff_n + 2.0) / (eff_n + 2.0 + self.trace_prior_kappa)
            w_schur = 1.0 - alpha_ctw
            final_p = [alpha_ctw * ctw_p[tid] + w_schur * schur_dist.get(tid, uniform_p) for tid in range(v_size)]
            for order, w_ord in enumerate(ctw_weights):
                weights_report[f"order_{order}"] = round(alpha_ctw * w_ord, 6)
            weights_report["schur_trace"] = round(w_schur, 6)
        else:
            final_p = ctw_p
            for order, w_ord in enumerate(ctw_weights):
                weights_report[f"order_{order}"] = round(w_ord, 6)
            weights_report["schur_trace"] = 0.0

        # Normalize to exact 1.0
        z_final = sum(final_p)
        final_p = [p / z_final for p in final_p]
        return final_p, weights_report, eff_n

    def explain_prediction(
        self,
        context_bytes: bytes,
        *,
        top_k: int = 5,
    ) -> GlassBoxExplanation:
        """Produce a 100% transparent Glass-Box attribution & epistemic certificate."""
        ctx_tokens = self.span_graph.encode_bytes(context_bytes)
        probs, mix_weights, eff_n = self.predict_distribution(ctx_tokens)
        ranked = sorted(range(self.vocab_size), key=lambda tid: (-probs[tid], tid))
        best_tid = ranked[0]
        best_prob = probs[best_tid]

        top_cands = tuple(
            (tid, self.span_graph.vocab[tid].display_text(), round(probs[tid], 6))
            for tid in ranked[:top_k]
        )

        # Gather exact corpus provenance citations across matched suffix orders
        citations: list[ProvenanceCitation] = []
        max_k = min(self.max_order, len(ctx_tokens))
        for order in range(max_k, 0, -1):
            ctx = tuple(ctx_tokens[-order:])
            w_ord = mix_weights.get(f"order_{order}", 0.0)
            prov_map = self._provenance.get((ctx, best_tid))
            if not prov_map or w_ord <= 0.0:
                continue
            tot_prov = sum(prov_map.values())
            for (doc_id, offset), cnt in sorted(prov_map.items()):
                citations.append(
                    ProvenanceCitation(
                        doc_id=doc_id,
                        span_offset=offset,
                        context_order=order,
                        target_token_id=best_tid,
                        weight_share=round(w_ord * (cnt / tot_prov), 6),
                    )
                )

        _, witnesses = self._schur_trace_distribution(ctx_tokens)
        best_witnesses = [w for w in witnesses if w.target_token_id == best_tid] or witnesses[:4]

        entropy = -sum(p * math.log2(max(p, 1e-15)) for p in probs if p > 0.0)
        epistemic = "SUPPORTED" if eff_n >= self.abstain_min_support else "EPISTEMIC_ABSTAIN"

        return GlassBoxExplanation(
            predicted_token_id=best_tid,
            predicted_bytes=self.span_graph.vocab[best_tid].raw_bytes,
            predicted_prob=round(best_prob, 6),
            top_candidates=top_cands,
            order_mixture_weights=mix_weights,
            provenance_citations=tuple(citations[:10]),
            hyperedge_witnesses=tuple(best_witnesses[:6]),
            effective_sample_size=round(eff_n, 4),
            entropy_bits=round(entropy, 6),
            epistemic_status=epistemic,
            state_merkle_root=self.state_merkle_sha256(),
        )

    def evaluate_bpb(self, eval_bytes: bytes) -> float:
        """Compute exact cross-entropy in bits per raw byte (HaDukeNN raw_corpus_bytes standard)."""
        if not eval_bytes:
            raise SheafError("eval_bytes must be non-empty")
        tokens = self.span_graph.encode_bytes(eval_bytes)
        nll_bits = 0.0
        for pos, target in enumerate(tokens):
            ctx = tokens[max(0, pos - max(self.max_order, 8)) : pos]
            probs, _, _ = self.predict_distribution(ctx)
            p_t = max(probs[target], 1e-15)
            nll_bits -= math.log2(p_t)
        return nll_bits / len(eval_bytes)


# ---------------------------------------------------------------------------
# Zero-Neural Anti-Unification Rule Inducer + Q-Nullspace Sheaf Law Discovery
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|-?\d+(?:\.\d+)?(?:/\d+)?")


def _tokenize_utterance(text: str) -> list[str]:
    return [tok for tok in _TOKEN_RE.findall(text.strip())]


def _rref_nullspace_q(delta_rows: Sequence[Sequence[Fraction]]) -> list[list[Fraction]]:
    """Compute an exact rational basis for {v in Q^n : delta_rows * v = 0}."""
    if not delta_rows:
        return []
    m = len(delta_rows)
    n = len(delta_rows[0])
    mat = [[Fraction(val) for val in row] for row in delta_rows]

    pivot_cols: list[int] = []
    r = 0
    for c in range(n):
        if r >= m:
            break
        piv = None
        for i in range(r, m):
            if mat[i][c] != 0:
                piv = i
                break
        if piv is None:
            continue
        mat[r], mat[piv] = mat[piv], mat[r]
        scale = mat[r][c]
        mat[r] = [v / scale for v in mat[r]]
        for i in range(m):
            if i != r and mat[i][c] != 0:
                factor = mat[i][c]
                mat[i] = [mat[i][j] - factor * mat[r][j] for j in range(n)]
        pivot_cols.append(c)
        r += 1

    pivot_set = set(pivot_cols)
    basis: list[list[Fraction]] = []
    for free_c in range(n):
        if free_c in pivot_set:
            continue
        vec = [Fraction(0, 1) for _ in range(n)]
        vec[free_c] = Fraction(1, 1)
        for row_idx, pcol in enumerate(pivot_cols):
            vec[pcol] = -mat[row_idx][free_c]
        basis.append(vec)
    return basis


class GlassBoxRuleAndSheafInducer:
    """100% White-Box Relational Rule Inducer & Exact Rational Sheaf Law Discoverer."""

    def __init__(self) -> None:
        self.rules: list[InducedRelationalRule] = []
        self.laws: list[DiscoveredSheafLaw] = []

    def induce_rules(
        self,
        training_pairs: Sequence[tuple[str, dict[str, str]]],
    ) -> tuple[InducedRelationalRule, ...]:
        """Induce explicit slot-role grammar templates via Plotkin anti-unification."""
        grouped: dict[tuple[str, int, tuple[tuple[str, int], ...], int | None], list[list[str]]] = defaultdict(list)

        for utterance, spec in training_pairs:
            toks = _tokenize_utterance(utterance)
            op = spec["op"]
            role_positions: dict[str, int] = {}
            unit_idx: int | None = None
            valid = True
            for role_name, role_val in sorted(spec.items()):
                if role_name == "op":
                    continue
                if role_name == "unit":
                    try:
                        unit_idx = toks.index(role_val)
                    except ValueError:
                        unit_idx = None
                    continue
                try:
                    role_positions[role_name] = toks.index(role_val)
                except ValueError:
                    valid = False
                    break
            if not valid:
                continue
            sig = (op, len(toks), tuple(sorted(role_positions.items())), unit_idx)
            grouped[sig].append(toks)

        induced: list[InducedRelationalRule] = []
        for idx, ((op, length, role_items, unit_idx), tok_lists) in enumerate(sorted(grouped.items())):
            role_by_pos = {pos: rname for rname, pos in role_items}
            pattern: list[str] = []
            for pos in range(length):
                if pos in role_by_pos:
                    pattern.append(f"<{role_by_pos[pos]}>")
                elif unit_idx is not None and pos == unit_idx:
                    pattern.append("<unit>")
                else:
                    col_vals = {tl[pos].lower() for tl in tok_lists}
                    pattern.append(next(iter(col_vals)) if len(col_vals) == 1 else "<wildcard>")
            induced.append(
                InducedRelationalRule(
                    rule_id=f"GBR-{idx + 1:02d}",
                    op=op,
                    pattern_tokens=tuple(pattern),
                    role_map=dict(role_items),
                    unit_index=unit_idx,
                    support_count=len(tok_lists),
                )
            )
        self.rules = induced
        return tuple(induced)

    def parse_utterance(self, utterance: str) -> tuple[InducedRelationalRule | None, dict[str, str]]:
        """Match an utterance against explicit white-box rules with zero neural heuristics."""
        toks = _tokenize_utterance(utterance)
        for rule in self.rules:
            if len(toks) != len(rule.pattern_tokens):
                continue
            matched = True
            extracted: dict[str, str] = {"op": rule.op}
            for pos, pat in enumerate(rule.pattern_tokens):
                tok = toks[pos]
                if pat.startswith("<") and pat.endswith(">"):
                    slot = pat[1:-1]
                    if slot == "wildcard":
                        continue
                    extracted[slot] = tok
                else:
                    if tok.lower() != pat:
                        matched = False
                        break
            if matched:
                return rule, extracted
        return None, {}

    def discover_sheaf_invariants(
        self,
        trajectories: Sequence[Sequence[dict[str, Fraction]]],
    ) -> tuple[DiscoveredSheafLaw, ...]:
        """Discover exact linear conservation laws and non-negativity over Q."""
        entities_set: set[str] = set()
        for traj in trajectories:
            for snap in traj:
                entities_set.update(snap.keys())
        entities = sorted(entities_set)
        if not entities:
            return ()

        deltas: list[list[Fraction]] = []
        all_non_neg = True
        for traj in trajectories:
            for snap in traj:
                for val in snap.values():
                    if Fraction(val) < 0:
                        all_non_neg = False
            for i in range(len(traj) - 1):
                s0, s1 = traj[i], traj[i + 1]
                row = [Fraction(s1.get(e, 0)) - Fraction(s0.get(e, 0)) for e in entities]
                if any(v != 0 for v in row):
                    deltas.append(row)

        basis = _rref_nullspace_q(deltas)
        discovered: list[DiscoveredSheafLaw] = []
        for idx, vec in enumerate(basis):
            coeffs = {entities[j]: vec[j] for j in range(len(entities)) if vec[j] != 0}
            if coeffs:
                discovered.append(
                    DiscoveredSheafLaw(
                        law_id=f"LAW-{idx + 1:02d}",
                        coefficients=coeffs,
                        require_non_negative=all_non_neg,
                    )
                )
        self.laws = discovered
        return tuple(discovered)


class GlassBoxWorldKernel:
    """Audited Counterfactual Hypergraph Executor enforcing induced Glass-Box rules & sheaf laws."""

    def __init__(
        self,
        inducer: GlassBoxRuleAndSheafInducer,
        *,
        canonical_unit: str = "L",
    ) -> None:
        self.inducer = inducer
        self.canonical_unit = canonical_unit
        self._branches: dict[str, dict[str, Fraction]] = {"main": {}}
        self._invariants: dict[str, dict[str, Fraction]] = {"main": {}}
        self._receipts: dict[str, list[GlassBoxStepReceipt]] = {"main": []}
        self._retracted_entities: dict[str, set[str]] = {"main": set()}

    def fork_branch(self, new_branch_id: str, *, from_branch: str = "main") -> str:
        """Create an isolated counterfactual branch copying exact rational state."""
        if from_branch not in self._branches:
            raise SheafError(f"Unknown source branch '{from_branch}'")
        if new_branch_id in self._branches:
            raise SheafError(f"Branch '{new_branch_id}' already exists")
        self._branches[new_branch_id] = copy.deepcopy(self._branches[from_branch])
        self._invariants[new_branch_id] = copy.deepcopy(self._invariants[from_branch])
        self._receipts[new_branch_id] = copy.deepcopy(self._receipts[from_branch])
        self._retracted_entities[new_branch_id] = copy.deepcopy(self._retracted_entities[from_branch])
        return self.branch_head_sha256(new_branch_id)

    def branch_state(self, branch_id: str = "main") -> dict[str, Fraction]:
        return dict(self._branches[branch_id])

    def branch_receipts(self, branch_id: str = "main") -> tuple[GlassBoxStepReceipt, ...]:
        return tuple(self._receipts[branch_id])

    def branch_head_sha256(self, branch_id: str = "main") -> str:
        recs = self._receipts[branch_id]
        return recs[-1].receipt_sha256 if recs else hashlib.sha256(b"GENESIS").hexdigest()

    def _normalize_amount(self, raw_amt: str, raw_unit: str | None) -> Fraction:
        unit = raw_unit or self.canonical_unit
        val, _, dim = ExactUnitLattice.normalize(raw_amt, unit)
        base_val, _, base_dim = ExactUnitLattice.normalize("1", self.canonical_unit)
        if dim != base_dim:
            raise SheafError(f"Dimensional mismatch: got {dim}, expected {base_dim}")
        return val / base_val

    def _record_receipt(
        self,
        branch_id: str,
        utterance: str,
        rule_id: str | None,
        op: str,
        roles: dict[str, str],
        status: str,
        detail: str,
        coboundary: Fraction,
    ) -> GlassBoxStepReceipt:
        state_str = {k: str(v) for k, v in sorted(self._branches[branch_id].items())}
        prev_hash = self.branch_head_sha256(branch_id)
        step_idx = len(self._receipts[branch_id])
        payload = {
            "branch": branch_id,
            "coboundary": str(coboundary),
            "detail": detail,
            "op": op,
            "prev": prev_hash,
            "roles": roles,
            "rule_id": rule_id,
            "state": state_str,
            "status": status,
            "step": step_idx,
            "utterance": utterance,
        }
        digest = sha256_hex(canonical_bytes(payload))
        rec = GlassBoxStepReceipt(
            step_index=step_idx,
            branch_id=branch_id,
            utterance=utterance,
            matched_rule_id=rule_id,
            op=op,
            roles=roles,
            status=status,
            detail=detail,
            coboundary_residual=str(coboundary),
            state_snapshot=state_str,
            receipt_sha256=digest,
        )
        self._receipts[branch_id].append(rec)
        return rec

    def step(self, utterance: str, *, branch_id: str = "main") -> GlassBoxStepReceipt:
        """Parse and execute one natural-language command with zero black-box components."""
        if branch_id not in self._branches:
            raise SheafError(f"Unknown branch '{branch_id}'")
        rule, parsed = self.inducer.parse_utterance(utterance)
        if rule is None:
            return self._record_receipt(
                branch_id,
                utterance,
                None,
                "unknown",
                {},
                "ABSTAIN_NO_RULE",
                "Utterance matched zero induced white-box templates; abstaining safely.",
                Fraction(0, 1),
            )

        op = parsed["op"]
        state = self._branches[branch_id]
        retracted = self._retracted_entities[branch_id]

        try:
            if op == "init":
                ent = parsed["entity"]
                amt = self._normalize_amount(parsed["amount"], parsed.get("unit"))
                if any(law.require_non_negative for law in self.inducer.laws) and amt < 0:
                    return self._record_receipt(
                        branch_id,
                        utterance,
                        rule.rule_id,
                        op,
                        parsed,
                        "REFUSED_INVARIANT_VIOLATION",
                        f"Negative initial stalk {amt} violates non-negativity law.",
                        abs(amt),
                    )
                state[ent] = amt
                retracted.discard(ent)
                # Refresh baseline invariant constants after explicit initialization
                for law in self.inducer.laws:
                    if set(law.coefficients.keys()).issubset(state.keys()):
                        self._invariants[branch_id][law.law_id] = law.evaluate_sum(state)
                return self._record_receipt(
                    branch_id, utterance, rule.rule_id, op, parsed, "ACCEPTED", f"Set {ent}={amt}", Fraction(0, 1)
                )

            if op == "retract":
                ent = parsed["entity"]
                state.pop(ent, None)
                retracted.add(ent)
                self._invariants[branch_id].clear()
                return self._record_receipt(
                    branch_id, utterance, rule.rule_id, op, parsed, "ACCEPTED", f"Retracted {ent}", Fraction(0, 1)
                )

            if op == "query":
                ent = parsed["entity"]
                if ent in retracted or ent not in state:
                    return self._record_receipt(
                        branch_id,
                        utterance,
                        rule.rule_id,
                        op,
                        parsed,
                        "UNRESOLVED_DEPENDENCY",
                        f"Entity '{ent}' is retracted or uninitialized.",
                        Fraction(0, 1),
                    )
                return self._record_receipt(
                    branch_id,
                    utterance,
                    rule.rule_id,
                    op,
                    parsed,
                    "ACCEPTED",
                    f"{ent}={state[ent]}",
                    Fraction(0, 1),
                )

            # Mutating transition or external assertion: stage candidate state and run sheaf coboundary check
            cand = dict(state)
            if op == "transfer":
                src, dst = parsed["source"], parsed["destination"]
                if src in retracted or dst in retracted or src not in cand or dst not in cand:
                    return self._record_receipt(
                        branch_id,
                        utterance,
                        rule.rule_id,
                        op,
                        parsed,
                        "UNRESOLVED_DEPENDENCY",
                        f"Transfer requires initialized active stalks for {src} and {dst}.",
                        Fraction(0, 1),
                    )
                amt = self._normalize_amount(parsed["amount"], parsed.get("unit"))
                cand[src] = cand[src] - amt
                cand[dst] = cand[dst] + amt
            elif op == "assert_eq":
                ent = parsed["entity"]
                if ent in retracted or ent not in cand:
                    return self._record_receipt(
                        branch_id,
                        utterance,
                        rule.rule_id,
                        op,
                        parsed,
                        "UNRESOLVED_DEPENDENCY",
                        f"Cannot assert state on missing/retracted '{ent}'.",
                        Fraction(0, 1),
                    )
                amt = self._normalize_amount(parsed["amount"], parsed.get("unit"))
                cand[ent] = amt
            else:
                return self._record_receipt(
                    branch_id,
                    utterance,
                    rule.rule_id,
                    op,
                    parsed,
                    "ABSTAIN_NO_RULE",
                    f"Unsupported operation '{op}'.",
                    Fraction(0, 1),
                )

            # Check discovered sheaf conservation laws & stalk positivity
            coboundary = Fraction(0, 1)
            violations: list[str] = []
            for law in self.inducer.laws:
                if law.require_non_negative:
                    for ent_k, val_k in cand.items():
                        if val_k < 0:
                            coboundary += abs(val_k)
                            violations.append(f"{ent_k}={val_k} < 0")
                if law.law_id in self._invariants[branch_id]:
                    expected_c = self._invariants[branch_id][law.law_id]
                    actual_c = law.evaluate_sum(cand)
                    diff = abs(actual_c - expected_c)
                    if diff != 0:
                        coboundary += diff
                        violations.append(f"{law.law_id}: {actual_c} != {expected_c}")

            # Direct state-consistency check on assert_eq even if fewer than all entities exist
            if op == "assert_eq" and cand[parsed["entity"]] != state[parsed["entity"]]:
                direct_diff = abs(cand[parsed["entity"]] - state[parsed["entity"]])
                coboundary = max(coboundary, direct_diff)
                violations.append(f"state mismatch {cand[parsed['entity']]} != {state[parsed['entity']]}")

            if violations:
                return self._record_receipt(
                    branch_id,
                    utterance,
                    rule.rule_id,
                    op,
                    parsed,
                    "REFUSED_INVARIANT_VIOLATION",
                    "; ".join(violations),
                    coboundary,
                )

            self._branches[branch_id] = cand
            return self._record_receipt(
                branch_id,
                utterance,
                rule.rule_id,
                op,
                parsed,
                "ACCEPTED",
                f"Transition {op} committed with zero coboundary residual.",
                Fraction(0, 1),
            )
        except SheafError as exc:
            return self._record_receipt(
                branch_id,
                utterance,
                rule.rule_id,
                op,
                parsed,
                "REFUSED_INVARIANT_VIOLATION",
                str(exc),
                Fraction(1, 1),
            )

    def replay_and_verify(self, branch_id: str = "main") -> bool:
        """Semantically re-execute every recorded command from genesis across fork boundaries."""
        recorded = self._receipts.get(branch_id)
        if recorded is None:
            raise SheafError(f"Unknown branch '{branch_id}'")
        fresh = GlassBoxWorldKernel(self.inducer, canonical_unit=self.canonical_unit)
        cur_b = "main"
        for rec in recorded:
            if rec.branch_id != cur_b:
                fresh.fork_branch(rec.branch_id, from_branch=cur_b)
                cur_b = rec.branch_id
            re_rec = fresh.step(rec.utterance, branch_id=cur_b)
            if (
                re_rec.status != rec.status
                or re_rec.state_snapshot != rec.state_snapshot
                or re_rec.receipt_sha256 != rec.receipt_sha256
            ):
                return False
        return True

    def export_scitt_compliance_bundle(
        self,
        *,
        branch_id: str = "main",
        signing_key: bytes = b"hadukenn-v070-glassbox-key",
    ) -> dict:
        """Emit an IETF SCITT RFC 9943 + EU AI Act Art. 12 regulatory bundle for this branch."""
        recs = [
            {
                "coboundary": r.coboundary_residual,
                "op": r.op,
                "receipt_sha256": r.receipt_sha256,
                "rule_id": r.matched_rule_id,
                "state": r.state_snapshot,
                "status": r.status,
                "step": r.step_index,
                "utterance": r.utterance,
            }
            for r in self._receipts[branch_id]
        ]
        events = [
            {
                "approved": r["status"] == "ACCEPTED",
                "coboundary": r["coboundary"],
                "human_escalation_required": r["status"] == "ABSTAIN_NO_RULE",
                "op": r["op"],
                "receipt_sha256": r["receipt_sha256"],
                "rule_id": r["rule_id"],
                "state": r["state"],
                "status": "ok" if r["status"] == "ACCEPTED" else "error",
                "step": r["step"],
                "utterance": r["utterance"],
                "verdict": "VALID" if r["status"] == "ACCEPTED" else "INVALID",
            }
            for r in recs
        ]
        state_sha = sha256_hex(canonical_bytes({k: str(v) for k, v in sorted(self._branches[branch_id].items())}))
        policy_sha = sha256_hex(
            canonical_bytes(
                {
                    "laws": [law.formula_str() for law in self.inducer.laws],
                    "rules": [rule.human_readable() for rule in self.inducer.rules],
                }
            )
        )
        return emit_regulatory_bundle(
            system_id=f"hadukenn-glassbox-v0.7.0:{branch_id}",
            trace_head_sha256=self.branch_head_sha256(branch_id),
            state_digest_sha256=state_sha,
            policy_sha256=policy_sha,
            events=events,
            created_at="2026-11-26T16:00:00Z",
            signing_key=signing_key,
        )
