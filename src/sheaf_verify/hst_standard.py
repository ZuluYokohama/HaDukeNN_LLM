"""Pure-stdlib reference implementation of the HaDukeNN Span-Trace v1 (HST-v1) Open Tokenization Standard.

HST-v1 unifies four structural properties absent from standalone BPE, SentencePiece, or SuperBPE:
  1. Universal fixed prefix layout:
       - IDs 0..255: Exact 1-to-1 raw byte atoms (0x00..0xFF, zero <unk> fallback).
       - IDs 256..263: Universal fixed control & verification primitives (<|hst:bos|>..<|hst:receipt|>)
         so control token IDs never shift across 32k, 49k, 65k, or 131k vocabulary tiers.
  2. Digit-atomic two-stage curriculum induction:
       - Stage A (IDs 264 .. v_base-1): Intra-lexeme byte-level BPE with single-digit isolation.
       - Stage B (IDs v_base .. V-1): PMI-gated cross-boundary super-spans bridging spaces/units
         solely when PMI(u, v) >= tau_pmi, directly compatible with CountShrunkSpanComposer.
  3. Bounded causal prefix invariance (lookahead <= max_span_bytes) + DP minimal-token encoding
     + merge-DAG StochasTok regularization (scheduled p: 0.12 -> 0.00).
  4. Canonical SHA-256 Merkle vocabulary manifest (hadukenn.hst.manifest.v1).
"""

from __future__ import annotations

import base64
import hashlib
import math
import random
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from sheaf_verify.hashing import canonical_bytes

HST_SCHEMA_ID = "hadukenn.hst.manifest.v1"

RESERVED_CONTROL_TOKENS: tuple[str, ...] = (
    "<|hst:bos|>",
    "<|hst:eos|>",
    "<|hst:span_open|>",
    "<|hst:span_close|>",
    "<|hst:fork|>",
    "<|hst:assert|>",
    "<|hst:unit|>",
    "<|hst:receipt|>",
)

UNIVERSAL_CONTROL_OFFSET = 256
FIRST_MERGE_ID = UNIVERSAL_CONTROL_OFFSET + len(RESERVED_CONTROL_TOKENS)  # 264

# Digit-atomic exhaustive byte pre-tokenizer: every digit is an isolated 1-byte atom in Stage A;
# letters (with optional leading space), spaces, newlines, underscores, and symbols partition 100% of bytes.
_STAGE_A_PRE_RE = re.compile(rb" ?[A-Za-z]+|[0-9]|_+|\r?\n| +|.")


@dataclass(frozen=True)
class HSTMergedRule:
    """Single binary merge rule in Stage A (lexical) or Stage B (cross-boundary PMI super-span)."""

    token_id: int
    left_id: int
    right_id: int
    stage: str  # "A_lexical" or "B_superspan"
    pmi_bits: float
    count: int
    byte_len: int


class HSTTokenizer:
    """Reference pure-stdlib implementation of the HST-v1 Open Tokenization Standard."""

    def __init__(
        self,
        vocab_size: int,
        id_to_bytes: list[bytes],
        rules: list[HSTMergedRule],
        control_map: dict[str, int],
        max_span_bytes: int = 16,
    ) -> None:
        if vocab_size < FIRST_MERGE_ID or len(id_to_bytes) != vocab_size:
            raise ValueError(f"Invalid HST vocab_size={vocab_size} vs len(id_to_bytes)={len(id_to_bytes)}")
        if any(id_to_bytes[i] != bytes([i]) for i in range(256)):
            raise ValueError("HST-v1 requires IDs 0..255 to be exact identity single-byte atoms.")

        self.vocab_size = vocab_size
        self.id_to_bytes = list(id_to_bytes)
        self.rules = list(rules)
        self.control_map = dict(control_map)
        self.id_to_control = {v: k for k, v in self.control_map.items()}
        self.control_ids = set(self.id_to_control.keys())

        non_ctrl_lens = [len(b) for tid, b in enumerate(self.id_to_bytes) if tid not in self.control_ids]
        self.max_span_bytes = max(int(max_span_bytes), max(non_ctrl_lens, default=1))

        self._stage_a_rank: dict[tuple[int, int], tuple[int, int]] = {}
        self._stage_b_rank: dict[tuple[int, int], tuple[int, int]] = {}
        self._split_children: dict[int, tuple[int, int]] = {}
        self._rule_by_id: dict[int, HSTMergedRule] = {}
        for rank, rule in enumerate(self.rules):
            pair = (rule.left_id, rule.right_id)
            self._split_children[rule.token_id] = pair
            self._rule_by_id[rule.token_id] = rule
            if rule.stage == "A_lexical":
                self._stage_a_rank[pair] = (rank, rule.token_id)
            else:
                self._stage_b_rank[pair] = (rank, rule.token_id)

        # First-byte index over non-control byte-span tokens for fast bounded-horizon encoding
        self._by_first_byte: dict[int, list[tuple[bytes, int]]] = {b: [] for b in range(256)}
        for tid, raw in enumerate(self.id_to_bytes):
            if tid in self.control_ids or not raw:
                continue
            self._by_first_byte[raw[0]].append((raw, tid))
        for b in range(256):
            self._by_first_byte[b].sort(key=lambda item: (-len(item[0]), item[1]))

    @classmethod
    def universal_control_map(cls) -> dict[str, int]:
        return {name: UNIVERSAL_CONTROL_OFFSET + idx for idx, name in enumerate(RESERVED_CONTROL_TOKENS)}

    @property
    def v_base_subword(self) -> int:
        """Vocabulary cutoff separating base atoms + controls + Stage A subwords from Stage B super-spans."""
        for rule in self.rules:
            if rule.stage == "B_superspan":
                return rule.token_id
        return self.vocab_size

    def stage_b_subword_parts(self) -> tuple[list[tuple[int, ...]], list[int]]:
        """Decompose each Stage B super-span into its constituent Stage A/base token IDs (< v_base_subword)."""
        v_base = self.v_base_subword
        parts_list: list[tuple[int, ...]] = []
        counts_list: list[int] = []
        for tid in range(v_base, self.vocab_size):
            rule = self._rule_by_id[tid]
            counts_list.append(rule.count)
            leaves: list[int] = []
            stack = [rule.right_id, rule.left_id]
            while stack:
                cur = stack.pop()
                if cur < v_base:
                    leaves.append(cur)
                else:
                    ch = self._split_children[cur]
                    stack.append(ch[1])
                    stack.append(ch[0])
            parts_list.append(tuple(leaves))
        return parts_list, counts_list

    @classmethod
    def train(
        cls,
        texts: list[str],
        vocab_size: int = 1024,
        stage_a_fraction: float = 0.68,
        min_pair_count: int = 2,
        min_superspan_pmi: float = 1.15,
        max_span_bytes: int = 16,
    ) -> HSTTokenizer:
        """Induce an HST-v1 vocabulary via Stage A lexical BPE + Stage B PMI-gated super-spans."""
        if vocab_size < FIRST_MERGE_ID + 4:
            raise ValueError(f"HST-v1 vocab_size must be >= {FIRST_MERGE_ID + 4}, got {vocab_size}")
        if not (0.25 <= stage_a_fraction <= 0.98):
            raise ValueError("stage_a_fraction must be in [0.25, 0.98].")

        ctrl_map = cls.universal_control_map()
        id_to_bytes: list[bytes] = [bytes([b]) for b in range(256)]
        for name in RESERVED_CONTROL_TOKENS:
            id_to_bytes.append(name.encode("utf-8"))
        assert len(id_to_bytes) == FIRST_MERGE_ID

        total_merges_budget = vocab_size - FIRST_MERGE_ID
        stage_a_budget = max(1, int(round(total_merges_budget * stage_a_fraction)))

        # Strip any control tags before learning byte-span statistics
        ctrl_re = re.compile("|".join(re.escape(k) for k in RESERVED_CONTROL_TOKENS))
        raw_docs = [ctrl_re.sub("", t).encode("utf-8") for t in texts if t]
        raw_docs = [d for d in raw_docs if d]

        piece_counts: Counter[bytes] = Counter()
        for doc in raw_docs:
            for m in _STAGE_A_PRE_RE.finditer(doc):
                piece = m.group(0)
                if piece:
                    piece_counts[piece] += 1

        words: list[list[int]] = [list(p) for p in piece_counts]
        freqs: list[int] = [piece_counts[p] for p in piece_counts]
        rules: list[HSTMergedRule] = []

        # Stage A: Intra-lexeme byte BPE (digits remain isolated because _STAGE_A_PRE_RE emits 1-byte digits)
        for _ in range(stage_a_budget):
            pair_counts: Counter[tuple[int, int]] = Counter()
            total_tokens = 0
            uni_counts: Counter[int] = Counter()
            for seq, freq in zip(words, freqs, strict=True):
                total_tokens += len(seq) * freq
                for tok in seq:
                    uni_counts[tok] += freq
                for i in range(len(seq) - 1):
                    if len(id_to_bytes[seq[i]]) + len(id_to_bytes[seq[i + 1]]) <= max_span_bytes:
                        pair_counts[(seq[i], seq[i + 1])] += freq
            if not pair_counts:
                break
            best_pair, best_cnt = max(pair_counts.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))
            if best_cnt < min_pair_count:
                break
            u, v = best_pair
            pmi = math.log2(max(1e-12, (best_cnt * max(1, total_tokens)) / max(1, uni_counts[u] * uni_counts[v])))
            new_id = len(id_to_bytes)
            merged_raw = id_to_bytes[u] + id_to_bytes[v]
            id_to_bytes.append(merged_raw)
            rules.append(
                HSTMergedRule(
                    token_id=new_id,
                    left_id=u,
                    right_id=v,
                    stage="A_lexical",
                    pmi_bits=round(pmi, 6),
                    count=int(best_cnt),
                    byte_len=len(merged_raw),
                )
            )
            for idx, seq in enumerate(words):
                if len(seq) < 2:
                    continue
                out: list[int] = []
                i = 0
                n = len(seq)
                while i < n:
                    if i + 1 < n and seq[i] == u and seq[i + 1] == v:
                        out.append(new_id)
                        i += 2
                    else:
                        out.append(seq[i])
                        i += 1
                words[idx] = out

        piece_to_ids = {raw_p: seq for raw_p, seq in zip(piece_counts, words, strict=True)}
        doc_streams: list[list[int]] = []
        for doc in raw_docs:
            stream: list[int] = []
            for m in _STAGE_A_PRE_RE.finditer(doc):
                stream.extend(piece_to_ids[m.group(0)])
            if stream:
                doc_streams.append(stream)

        # Stage B: PMI-gated cross-boundary super-spans
        while len(id_to_bytes) < vocab_size:
            pair_counts_b: Counter[tuple[int, int]] = Counter()
            uni_counts_b: Counter[int] = Counter()
            total_b = 0
            for stream in doc_streams:
                total_b += len(stream)
                for tok in stream:
                    uni_counts_b[tok] += 1
                for i in range(len(stream) - 1):
                    u, v = stream[i], stream[i + 1]
                    bu, bv = id_to_bytes[u], id_to_bytes[v]
                    if len(bu) + len(bv) > max_span_bytes:
                        continue
                    if bu.isdigit() and bv.isdigit():
                        continue
                    if b"\n" in bu or b"\n" in bv or b"\r" in bu or b"\r" in bv:
                        continue
                    pair_counts_b[(u, v)] += 1
            if not pair_counts_b:
                break

            best_candidate: tuple[float, int, tuple[int, int], float] | None = None
            for (u, v), cnt in pair_counts_b.items():
                if cnt < min_pair_count:
                    continue
                pmi = math.log2(max(1e-12, (cnt * max(1, total_b)) / max(1, uni_counts_b[u] * uni_counts_b[v])))
                if pmi < min_superspan_pmi:
                    continue
                gain_score = float(cnt) * (1.0 + 0.35 * pmi)
                cand = (gain_score, cnt, (u, v), pmi)
                if best_candidate is None or (cand[0], cand[1], -cand[2][0], -cand[2][1]) > (
                    best_candidate[0],
                    best_candidate[1],
                    -best_candidate[2][0],
                    -best_candidate[2][1],
                ):
                    best_candidate = cand

            if best_candidate is None:
                break

            _, best_cnt, (u, v), best_pmi = best_candidate
            new_id = len(id_to_bytes)
            merged_raw = id_to_bytes[u] + id_to_bytes[v]
            id_to_bytes.append(merged_raw)
            rules.append(
                HSTMergedRule(
                    token_id=new_id,
                    left_id=u,
                    right_id=v,
                    stage="B_superspan",
                    pmi_bits=round(best_pmi, 6),
                    count=int(best_cnt),
                    byte_len=len(merged_raw),
                )
            )
            for s_idx, stream in enumerate(doc_streams):
                if len(stream) < 2:
                    continue
                out_s: list[int] = []
                i = 0
                n = len(stream)
                while i < n:
                    if i + 1 < n and stream[i] == u and stream[i + 1] == v:
                        out_s.append(new_id)
                        i += 2
                    else:
                        out_s.append(stream[i])
                        i += 1
                doc_streams[s_idx] = out_s

        return cls(
            vocab_size=len(id_to_bytes),
            id_to_bytes=id_to_bytes,
            rules=rules,
            control_map=ctrl_map,
            max_span_bytes=max_span_bytes,
        )

    def _apply_rank_merges(self, seq: list[int], rank_map: dict[tuple[int, int], tuple[int, int]]) -> list[int]:
        if len(seq) < 2 or not rank_map:
            return seq
        cur = list(seq)
        while len(cur >= [0] if False else cur) >= 2:
            best_rank = 1_000_000_000
            best_pair: tuple[int, int] | None = None
            best_new = -1
            for i in range(len(cur) - 1):
                info = rank_map.get((cur[i], cur[i + 1]))
                if info is not None and info[0] < best_rank:
                    best_rank, best_new = info
                    best_pair = (cur[i], cur[i + 1])
            if best_pair is None:
                break
            u, v = best_pair
            nxt: list[int] = []
            i = 0
            n = len(cur)
            while i < n:
                if i + 1 < n and cur[i] == u and cur[i + 1] == v:
                    nxt.append(best_new)
                    i += 2
                else:
                    nxt.append(cur[i])
                    i += 1
            cur = nxt
        return cur

    def encode_bytes(self, raw: bytes, mode: str = "causal_local") -> list[int]:
        """Encode arbitrary bytes into HST-v1 token IDs (excluding control-tag parsing)."""
        if not raw:
            return []
        if mode == "causal_local":
            out: list[int] = []
            i = 0
            n = len(raw)
            while i < n:
                b0 = raw[i]
                matched_id = b0
                matched_len = 1
                for cand_bytes, tid in self._by_first_byte[b0]:
                    clen = len(cand_bytes)
                    if i + clen <= n and raw[i : i + clen] == cand_bytes:
                        matched_id = tid
                        matched_len = clen
                        break
                out.append(matched_id)
                i += matched_len
            return out

        if mode == "staged_bpe":
            stream: list[int] = []
            for m in _STAGE_A_PRE_RE.finditer(raw):
                stream.extend(self._apply_rank_merges(list(m.group(0)), self._stage_a_rank))
            return self._apply_rank_merges(stream, self._stage_b_rank)

        if mode == "dp_min_tokens":
            n = len(raw)
            inf = n + 10
            dp_cost = [inf] * (n + 1)
            dp_next_len = [1] * n
            dp_tok = [0] * n
            dp_cost[n] = 0
            for i in range(n - 1, -1, -1):
                best_c = dp_cost[i + 1] + 1
                best_l = 1
                best_t = raw[i]
                for cand_bytes, tid in self._by_first_byte[raw[i]]:
                    clen = len(cand_bytes)
                    if i + clen <= n and raw[i : i + clen] == cand_bytes:
                        cand_c = 1 + dp_cost[i + clen]
                        if cand_c < best_c or (cand_c == best_c and clen > best_l):
                            best_c = cand_c
                            best_l = clen
                            best_t = tid
                dp_cost[i] = best_c
                dp_next_len[i] = best_l
                dp_tok[i] = best_t
            out_dp: list[int] = []
            pos = 0
            while pos < n:
                out_dp.append(dp_tok[pos])
                pos += dp_next_len[pos]
            return out_dp

        raise ValueError(f"Unknown HST encode mode: {mode!r}")

    def encode(self, text: str, mode: str = "causal_local") -> list[int]:
        """Encode UTF-8 text (parsing <|hst:...|> control tags into universal IDs 256..263)."""
        if not text:
            return []
        ctrl_pattern = "|".join(re.escape(k) for k in self.control_map)
        out: list[int] = []
        last = 0
        for m in re.finditer(ctrl_pattern, text):
            if m.start() > last:
                out.extend(self.encode_bytes(text[last : m.start()].encode("utf-8"), mode=mode))
            out.append(self.control_map[m.group(0)])
            last = m.end()
        if last < len(text):
            out.extend(self.encode_bytes(text[last:].encode("utf-8"), mode=mode))
        return out

    def decode_bytes(self, ids: list[int]) -> bytes:
        """Losslessly reconstruct exact raw bytes."""
        chunks: list[bytes] = []
        for tid in ids:
            if 0 <= tid < self.vocab_size:
                chunks.append(self.id_to_bytes[tid])
            else:
                raise ValueError(f"Unknown HST token ID: {tid}")
        return b"".join(chunks)

    def decode(self, ids: list[int]) -> str:
        return self.decode_bytes(ids).decode("utf-8", errors="replace")

    def encode_stochastok(self, ids: list[int], drop_prob: float, rng: random.Random) -> list[int]:
        """Stochastically decompose merged tokens along the HST merge DAG (preserving exact byte sequence)."""
        if drop_prob <= 0.0:
            return list(ids)
        out: list[int] = []
        stack = list(reversed(ids))
        while stack:
            tid = stack.pop()
            children = self._split_children.get(tid)
            if children is not None and rng.random() < drop_prob:
                stack.append(children[1])
                stack.append(children[0])
            else:
                out.append(tid)
        return out

    def token_byte_lengths(self) -> list[int]:
        return [len(b) for b in self.id_to_bytes]

    def verify_causal_prefix_invariance(self, prefix: bytes, suffixes: list[bytes]) -> tuple[bool, dict[str, Any]]:
        """Verify that appending unseen suffixes never alters committed prefix tokens
        ending at or before len(prefix) - max_span_bytes under causal_local encoding.
        """
        base_ids = self.encode_bytes(prefix, mode="causal_local")
        safe_cutoff = max(0, len(prefix) - self.max_span_bytes)

        base_spans: list[tuple[int, int, int]] = []
        pos = 0
        for tid in base_ids:
            end = pos + len(self.id_to_bytes[tid])
            if end <= safe_cutoff:
                base_spans.append((pos, end, tid))
            pos = end

        violations = 0
        for suf in suffixes:
            ext_ids = self.encode_bytes(prefix + suf, mode="causal_local")
            ext_spans: list[tuple[int, int, int]] = []
            epos = 0
            for tid in ext_ids:
                eend = epos + len(self.id_to_bytes[tid])
                if eend <= safe_cutoff:
                    ext_spans.append((epos, eend, tid))
                else:
                    break
                epos = eend
            if ext_spans != base_spans:
                violations += 1

        return violations == 0, {
            "checked_suffixes": len(suffixes),
            "committed_prefix_tokens": len(base_spans),
            "max_span_bytes": self.max_span_bytes,
            "prefix_bytes": len(prefix),
            "safe_cutoff_bytes": safe_cutoff,
            "violations": violations,
        }

    def merkle_root_sha256(self) -> str:
        """Compute canonical SHA-256 Merkle commitment over all HST-v1 vocabulary leaves and rules."""
        leaf_hashes: list[str] = []
        for tid, raw in enumerate(self.id_to_bytes):
            tag = "ctrl_tok" if tid in self.control_ids else "byte_tok"
            leaf_hashes.append(hashlib.sha256(f"{tag}:{tid}:".encode() + raw).hexdigest())
        for rule in self.rules:
            rbytes = f"rule:{rule.token_id}:{rule.left_id}:{rule.right_id}:{rule.stage}".encode()
            leaf_hashes.append(hashlib.sha256(rbytes).hexdigest())

        level = leaf_hashes
        while len(level) > 1:
            nxt: list[str] = []
            for i in range(0, len(level), 2):
                left = level[i]
                right = level[i + 1] if i + 1 < len(level) else left
                nxt.append(hashlib.sha256(f"{left}:{right}".encode()).hexdigest())
            level = nxt
        return level[0] if level else hashlib.sha256(b"empty").hexdigest()

    def export_manifest(self) -> dict[str, Any]:
        """Serialize portable HST-v1 wire manifest with canonical Merkle root."""
        stage_a_count = sum(1 for r in self.rules if r.stage == "A_lexical")
        stage_b_count = sum(1 for r in self.rules if r.stage == "B_superspan")
        return {
            "schema": HST_SCHEMA_ID,
            "version": "1.0.0",
            "vocab_size": self.vocab_size,
            "n_byte_tokens": self.vocab_size - len(self.control_map),
            "max_span_bytes": self.max_span_bytes,
            "digit_atomic_policy": True,
            "stage_a_merges": stage_a_count,
            "stage_b_superspans": stage_b_count,
            "control_tokens": dict(sorted(self.control_map.items(), key=lambda kv: kv[1])),
            "merges": [
                {
                    "byte_len": r.byte_len,
                    "count": r.count,
                    "left_id": r.left_id,
                    "pmi_bits": r.pmi_bits,
                    "right_id": r.right_id,
                    "stage": r.stage,
                    "token_b64": base64.b64encode(self.id_to_bytes[r.token_id]).decode("ascii"),
                    "token_id": r.token_id,
                }
                for r in self.rules
            ],
            "merkle_root_sha256": self.merkle_root_sha256(),
        }

    @classmethod
    def from_manifest(cls, doc: dict[str, Any]) -> HSTTokenizer:
        """Reconstruct and cryptographically verify an HSTTokenizer from its wire manifest."""
        if doc.get("schema") != HST_SCHEMA_ID:
            raise ValueError(f"Unsupported HST manifest schema: {doc.get('schema')!r}")
        ctrl_map = {str(k): int(v) for k, v in doc.get("control_tokens", {}).items()}
        id_to_bytes: list[bytes] = [bytes([b]) for b in range(256)]
        for name, cid in sorted(ctrl_map.items(), key=lambda kv: kv[1]):
            if cid != len(id_to_bytes):
                raise ValueError(f"Control token {name!r} at non-universal slot {cid}, expected {len(id_to_bytes)}")
            id_to_bytes.append(name.encode("utf-8"))

        rules: list[HSTMergedRule] = []
        for entry in doc.get("merges", []):
            tid = int(entry["token_id"])
            lid = int(entry["left_id"])
            rid = int(entry["right_id"])
            if tid != len(id_to_bytes):
                raise ValueError(f"Non-contiguous merge token_id={tid}, expected {len(id_to_bytes)}")
            raw = id_to_bytes[lid] + id_to_bytes[rid]
            expected_raw = base64.b64decode(entry["token_b64"])
            if raw != expected_raw:
                raise ValueError(f"Token byte mismatch at token_id={tid}")
            id_to_bytes.append(raw)
            rules.append(
                HSTMergedRule(
                    token_id=tid,
                    left_id=lid,
                    right_id=rid,
                    stage=str(entry["stage"]),
                    pmi_bits=float(entry["pmi_bits"]),
                    count=int(entry["count"]),
                    byte_len=int(entry["byte_len"]),
                )
            )
        tok = cls(
            vocab_size=int(doc["vocab_size"]),
            id_to_bytes=id_to_bytes,
            rules=rules,
            control_map=ctrl_map,
            max_span_bytes=int(doc.get("max_span_bytes", 16)),
        )
        computed_root = tok.merkle_root_sha256()
        if computed_root != doc.get("merkle_root_sha256"):
            raise ValueError(
                f"HST-v1 Merkle root mismatch: manifest={doc.get('merkle_root_sha256')} vs computed={computed_root}"
            )
        return tok


def verify_hst_manifest_integrity(doc: dict[str, Any]) -> tuple[bool, list[str]]:
    """Pure-stdlib validator checking HST-v1 manifest structure, Merkle commitment, and lossless invariants."""
    problems: list[str] = []
    try:
        tok = HSTTokenizer.from_manifest(doc)
    except Exception as exc:  # noqa: BLE001
        return False, [str(exc)]

    probe = (
        b"Transfer 25.50 L from Vessel_A to Vessel_B; total == 25.50 L\n"
        b"UTF-8 bytes: \xf0\x9f\xa7\xaa \xe2\x88\xab f(x)dx = 1042\x00\xff"
    )
    ids = tok.encode_bytes(probe, mode="causal_local")
    if tok.decode_bytes(ids) != probe:
        problems.append("Lossless byte roundtrip failed on binary+UTF8 probe.")

    num_ids = tok.encode_bytes(b"1042", mode="causal_local")
    if [tok.id_to_bytes[i] for i in num_ids] != [b"1", b"0", b"4", b"2"]:
        problems.append("Digit atomicity violated on probe '1042'.")

    ok_pref, pref_rep = tok.verify_causal_prefix_invariance(
        b"Causal prefix stability check across multi-word super-spans in HST-v1 standard ",
        [b"alpha.", b"beta_12345!", b"\nbranch_2"],
    )
    if not ok_pref:
        problems.append(f"Causal prefix invariance failed: {pref_rep}")

    _ = canonical_bytes({"merkle_root": tok.merkle_root_sha256(), "vocab": tok.vocab_size})
    return len(problems) == 0, problems
