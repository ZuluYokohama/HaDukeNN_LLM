"""Byte-level BPE, PMI-curated span sidecars (OE-Span), and scheduled StochasTok splitting."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

# Exhaustive byte-level pre-tokenizer: partitions every byte of any arbitrary UTF-8 stream with zero loss.
_BYTE_PRE_RE = re.compile(rb"[A-Za-z]+|[0-9]+|_+| +|[\r\n\t]+|.")
_PUNCT_ONLY_RE = re.compile(r"^[\s\W_]+$", re.ASCII)


@dataclass(frozen=True)
class SpanSpec:
    """PMI-curated multi-token span over base BPE token IDs."""

    span_id: int
    parts: tuple[int, ...]
    pmi: float
    count: int
    byte_len: int


class ByteBPETokenizer:
    """Exact byte-level BPE tokenizer with byte-accurate BPB accounting."""

    def __init__(
        self,
        vocab_size: int,
        merges: list[tuple[int, int]],
        token_bytes: list[bytes],
    ) -> None:
        if vocab_size < 256 or len(token_bytes) != vocab_size:
            raise ValueError(f"Invalid vocab_size={vocab_size} vs len(token_bytes)={len(token_bytes)}")
        self.vocab_size = vocab_size
        self.merges = list(merges)
        self.merge_rank: dict[tuple[int, int], tuple[int, int]] = {
            pair: (rank, 256 + rank) for rank, pair in enumerate(self.merges)
        }
        self.token_bytes = list(token_bytes)
        self.byte_lengths = np.array([len(b) for b in self.token_bytes], dtype=np.int32)
        self._piece_cache: dict[bytes, list[int]] = {}
        self._split_parent: dict[int, tuple[int, int]] = {
            256 + rank: pair for rank, pair in enumerate(self.merges)
        }

    @classmethod
    def train(cls, texts: list[str], vocab_size: int = 1024, min_pair_count: int = 4) -> ByteBPETokenizer:
        """Train an exact byte-level BPE vocabulary over pre-tokenized byte chunks."""
        if vocab_size < 256:
            raise ValueError("ByteBPE vocab_size must be >= 256.")
        piece_counts: Counter[bytes] = Counter()
        for txt in texts:
            raw_txt = txt.encode("utf-8")
            for m in _BYTE_PRE_RE.finditer(raw_txt):
                piece = m.group(0)
                if piece:
                    piece_counts[piece] += 1

        words: list[list[int]] = [list(raw) for raw in piece_counts]
        freqs: list[int] = [piece_counts[raw] for raw in piece_counts]
        token_bytes: list[bytes] = [bytes([b]) for b in range(256)]
        merges: list[tuple[int, int]] = []

        target_merges = vocab_size - 256
        for _ in range(target_merges):
            pair_counts: Counter[tuple[int, int]] = Counter()
            for seq, f in zip(words, freqs, strict=True):
                if len(seq) < 2:
                    continue
                for i in range(len(seq) - 1):
                    pair_counts[(seq[i], seq[i + 1])] += f
            if not pair_counts:
                break
            best_pair, best_cnt = pair_counts.most_common(1)[0]
            if best_cnt < min_pair_count:
                break
            new_id = len(token_bytes)
            merges.append(best_pair)
            token_bytes.append(token_bytes[best_pair[0]] + token_bytes[best_pair[1]])
            a, b = best_pair
            for idx, seq in enumerate(words):
                if len(seq) < 2:
                    continue
                out: list[int] = []
                i = 0
                n = len(seq)
                while i < n:
                    if i + 1 < n and seq[i] == a and seq[i + 1] == b:
                        out.append(new_id)
                        i += 2
                    else:
                        out.append(seq[i])
                        i += 1
                words[idx] = out

        pad_idx = 0
        while len(token_bytes) < vocab_size:
            a = 65 + (pad_idx // 26) % 26
            b = 97 + pad_idx % 26
            pair = (a, b)
            if pair not in {p for p in merges}:
                merges.append(pair)
                token_bytes.append(token_bytes[a] + token_bytes[b])
            else:
                alt = (48 + (pad_idx % 10), 97 + ((pad_idx // 10) % 26))
                merges.append(alt)
                token_bytes.append(token_bytes[alt[0]] + token_bytes[alt[1]])
            pad_idx += 1

        return cls(vocab_size=vocab_size, merges=merges, token_bytes=token_bytes)

    def _encode_piece(self, raw: bytes) -> list[int]:
        cached = self._piece_cache.get(raw)
        if cached is not None:
            return cached
        seq = list(raw)
        while len(seq) >= 2:
            best_rank = 1_000_000_000
            best_pair: tuple[int, int] | None = None
            best_new = -1
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                info = self.merge_rank.get(pair)
                if info is not None and info[0] < best_rank:
                    best_rank, best_new = info
                    best_pair = pair
            if best_pair is None:
                break
            a, b = best_pair
            out: list[int] = []
            i = 0
            while i < len(seq):
                if i + 1 < len(seq) and seq[i] == a and seq[i + 1] == b:
                    out.append(best_new)
                    i += 2
                else:
                    out.append(seq[i])
                    i += 1
            seq = out
        self._piece_cache[raw] = seq
        return seq

    def encode(self, text: str) -> np.ndarray:
        """Encode UTF-8 string to int32 base token IDs with 100% lossless byte reconstruction."""
        ids: list[int] = []
        raw_txt = text.encode("utf-8")
        for m in _BYTE_PRE_RE.finditer(raw_txt):
            piece = m.group(0)
            if piece:
                ids.extend(self._encode_piece(piece))
        return np.asarray(ids, dtype=np.int32)

    def decode_bytes(self, ids: np.ndarray | list[int]) -> bytes:
        return b"".join(self.token_bytes[int(t)] for t in ids)

    def stochastok_expand(self, ids: np.ndarray, p_split: float, rng: np.random.Generator) -> np.ndarray:
        """
        Sims et al. (ICLR 2026) StochasTok post-tokenization online splitting:
        randomly split merged tokens into their exact constituent BPE pair (t1, t2)
        with probability `p_split`, preserving the exact vocabulary and byte stream.
        """
        if p_split <= 0.0 or ids.size == 0:
            return ids
        draws = rng.random(ids.shape[0])
        out: list[int] = []
        for tok, u in zip(ids.tolist(), draws.tolist(), strict=True):
            parent = self._split_parent.get(tok)
            if parent is not None and u < p_split:
                out.append(parent[0])
                out.append(parent[1])
            else:
                out.append(tok)
        return np.asarray(out, dtype=np.int32)


def cosine_stochastok_prob(progress: float, p_start: float = 0.12, p_end: float = 0.03) -> float:
    """Cosine-annealed StochasTok split schedule from T-Fix 2 (`p: 0.12 -> 0.03`)."""
    t = min(max(float(progress), 0.0), 1.0)
    return float(p_end + 0.5 * (p_start - p_end) * (1.0 + math.cos(math.pi * t)))


class PMISpanLexicon:
    """
    PMI-curated multi-token span table supporting:
      1. `OESpanSidecar` (T-Fix 1): causal input span index s_{<=t} keeping base BPE output targets.
      2. `MergedSpanStream` (hnt-style baseline): greedy multi-token compression into V_base + K_span.
    """

    def __init__(self, base_tokenizer: ByteBPETokenizer, spans: list[SpanSpec]) -> None:
        self.base = base_tokenizer
        self.spans = list(spans)
        self.num_spans = len(self.spans)
        self.pair_to_span: dict[tuple[int, int], int] = {
            (s.parts[0], s.parts[1]): s.span_id for s in self.spans if len(s.parts) == 2
        }
        self.tri_to_span: dict[tuple[int, int, int], int] = {
            (s.parts[0], s.parts[1], s.parts[2]): s.span_id for s in self.spans if len(s.parts) == 3
        }
        all_lens = list(self.base.byte_lengths) + [s.byte_len for s in self.spans]
        self.merged_byte_lengths = np.asarray(all_lens, dtype=np.int32)

    @classmethod
    def build(
        cls,
        base_tokenizer: ByteBPETokenizer,
        train_streams: list[np.ndarray],
        max_spans: int = 256,
        min_pmi: float = 2.0,
        min_count: int = 8,
    ) -> PMISpanLexicon:
        """Extract high-PMI 2- and 3-token spans with punctuation-glue hygiene."""
        uni: Counter[int] = Counter()
        bi: Counter[tuple[int, int]] = Counter()
        tri: Counter[tuple[int, int, int]] = Counter()
        total = 0
        for seq in train_streams:
            lst = seq.tolist()
            n = len(lst)
            total += n
            uni.update(lst)
            for i in range(n - 1):
                bi[(lst[i], lst[i + 1])] += 1
            for i in range(n - 2):
                tri[(lst[i], lst[i + 1], lst[i + 2])] += 1

        if total == 0:
            return cls(base_tokenizer, [])

        candidates: list[tuple[float, tuple[int, ...], float, int, int]] = []
        for (a, b), cnt in bi.items():
            if cnt < min_count:
                continue
            raw_str = (base_tokenizer.token_bytes[a] + base_tokenizer.token_bytes[b]).decode("utf-8", errors="ignore")
            if _PUNCT_ONLY_RE.match(raw_str) or sum(ch.isalnum() for ch in raw_str) < 2:
                continue
            pmi = math.log((cnt * total) / max(uni[a] * uni[b], 1))
            if pmi >= min_pmi:
                score = cnt * pmi
                blen = int(base_tokenizer.byte_lengths[a] + base_tokenizer.byte_lengths[b])
                candidates.append((score, (a, b), pmi, cnt, blen))

        for (a, b, c), cnt in tri.items():
            if cnt < min_count:
                continue
            raw_str = (
                base_tokenizer.token_bytes[a]
                + base_tokenizer.token_bytes[b]
                + base_tokenizer.token_bytes[c]
            ).decode("utf-8", errors="ignore")
            if _PUNCT_ONLY_RE.match(raw_str) or sum(ch.isalnum() for ch in raw_str) < 3:
                continue
            denom = max(uni[a] * uni[b] * uni[c], 1)
            pmi = 0.5 * math.log((cnt * (total**2)) / denom)
            if pmi >= min_pmi:
                score = 1.15 * cnt * pmi
                blen = int(
                    base_tokenizer.byte_lengths[a]
                    + base_tokenizer.byte_lengths[b]
                    + base_tokenizer.byte_lengths[c]
                )
                candidates.append((score, (a, b, c), pmi, cnt, blen))

        candidates.sort(key=lambda item: -item[0])
        spans: list[SpanSpec] = []
        for idx, (_, parts, pmi, cnt, blen) in enumerate(candidates[:max_spans]):
            spans.append(SpanSpec(span_id=idx, parts=parts, pmi=pmi, count=cnt, byte_len=blen))
        return cls(base_tokenizer, spans)

    def causal_ending_span_ids(self, base_ids: np.ndarray) -> np.ndarray:
        """
        Return `span_idx[t] in {-1, 0..K_span-1}` using ONLY `(base_ids[t-2], base_ids[t-1], base_ids[t])`.
        Provably zero future lookahead (`partial span_idx[t] / partial base_ids[>t] == 0`).
        """
        n = int(base_ids.shape[0])
        out = np.full(n, -1, dtype=np.int32)
        lst = base_ids.tolist()
        for t in range(1, n):
            if t >= 2:
                s3 = self.tri_to_span.get((lst[t - 2], lst[t - 1], lst[t]))
                if s3 is not None:
                    out[t] = s3
                    continue
            s2 = self.pair_to_span.get((lst[t - 1], lst[t]))
            if s2 is not None:
                out[t] = s2
        return out

    def merge_sequence(self, base_ids: np.ndarray) -> np.ndarray:
        """Greedy left-to-right multi-token merging (`hnt`-style compressed output vocabulary)."""
        lst = base_ids.tolist()
        v_base = self.base.vocab_size
        out: list[int] = []
        i = 0
        n = len(lst)
        while i < n:
            if i + 2 < n:
                s3 = self.tri_to_span.get((lst[i], lst[i + 1], lst[i + 2]))
                if s3 is not None:
                    out.append(v_base + s3)
                    i += 3
                    continue
            if i + 1 < n:
                s2 = self.pair_to_span.get((lst[i], lst[i + 1]))
                if s2 is not None:
                    out.append(v_base + s2)
                    i += 2
                    continue
            out.append(lst[i])
            i += 1
        return np.asarray(out, dtype=np.int32)
