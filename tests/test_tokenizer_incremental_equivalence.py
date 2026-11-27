"""The incremental-count tokenizer trainers must match the full-recount reference byte for byte.

``HSTTokenizer.train`` and ``ByteBPETokenizer.train`` are the pre-registered T-49 arms. The fast
paths only change how pair counts are maintained, so merges, rules and vocabularies must be identical
to ``train_reference`` on every corpus, including tie-heavy and degenerate ones.
"""

from __future__ import annotations

import random

import pytest

from hadukenn_core.tokenization import ByteBPETokenizer
from sheaf_verify.hst_standard import HSTTokenizer


def _synthetic(seed: int, n_lines: int) -> list[str]:
    rng = random.Random(seed)
    words = [
        "the", "of", "and", "Vessel_A", "Vessel_B", "transfer", "litre", "aaaa", "abab", "ba",
        "sheaf", "gate", "naïve", "café", "∫", "🧪", "<|hst:bos|>", "x", "y", "zz",
    ]  # fmt: skip
    lines = []
    for _ in range(n_lines):
        parts = []
        for _ in range(rng.randint(0, 24)):
            r = rng.random()
            if r < 0.15:
                parts.append(str(rng.randint(0, 99999)))
            elif r < 0.2:
                parts.append(rng.choice([",", ".", ";", "==", "  ", "\t", "@-@"]))
            else:
                parts.append(rng.choice(words))
        lines.append(" ".join(parts))
    return lines


CORPORA = {
    "synthetic_a": _synthetic(1, 300),
    "synthetic_b": _synthetic(7, 120),
    "ties": ["ab cd ef gh", "cd ab gh ef", "ef gh ab cd"] * 5,
    "runs": ["aaaaaaa aaaa aa a", "abababab abab", "1111 2222 1111"] * 4,
    "multiline": ["line one\nline two\r\nline three " * 3, "", "\n\n", "tail 42 tail 42"],
    "tiny": ["short"],
}


def _hst_signature(tok: HSTTokenizer) -> tuple:
    return (tok.vocab_size, tuple(tok.id_to_bytes), tuple(tok.rules), tok.merkle_root_sha256())


@pytest.mark.parametrize("name", sorted(CORPORA))
@pytest.mark.parametrize(
    ("vocab_size", "stage_a_fraction", "min_pair_count", "min_pmi", "max_span"),
    [(1024, 0.68, 2, 1.15, 16), (400, 0.5, 2, 0.0, 8), (700, 0.9, 3, 2.5, 16)],
)
def test_hst_incremental_matches_reference(name, vocab_size, stage_a_fraction, min_pair_count, min_pmi, max_span):
    texts = CORPORA[name]
    kwargs = dict(
        vocab_size=vocab_size,
        stage_a_fraction=stage_a_fraction,
        min_pair_count=min_pair_count,
        min_superspan_pmi=min_pmi,
        max_span_bytes=max_span,
    )
    assert _hst_signature(HSTTokenizer.train(texts, **kwargs)) == _hst_signature(
        HSTTokenizer.train_reference(texts, **kwargs)
    )


@pytest.mark.parametrize("name", sorted(CORPORA))
@pytest.mark.parametrize(("vocab_size", "min_pair_count"), [(1024, 4), (320, 2), (600, 1)])
def test_byte_bpe_incremental_matches_reference(name, vocab_size, min_pair_count):
    texts = CORPORA[name]
    fast = ByteBPETokenizer.train(texts, vocab_size=vocab_size, min_pair_count=min_pair_count)
    ref = ByteBPETokenizer.train_reference(texts, vocab_size=vocab_size, min_pair_count=min_pair_count)
    assert fast.merges == ref.merges
    assert fast.token_bytes == ref.token_bytes
    assert fast.vocab_size == ref.vocab_size
