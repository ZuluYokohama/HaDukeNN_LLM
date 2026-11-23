"""Declared review thresholds.

These numbers are part of the harness contract. They are not estimated from
a result file, and a study cannot loosen them by writing a smaller value
into its metrics.
"""

MIN_SEEDS = 5
MAX_FLOP_RELATIVE_TOLERANCE = 0.05
ALPHA = 0.05
MIN_BOOTSTRAP_DRAWS = 1000
MAX_BOOTSTRAP_DRAWS = 100_000
PRIMARY_ARM = "hypergraph"
CONTROL_ARMS = ("superword_bpe", "randomized_hypergraph")
REQUIRED_ARMS = (PRIMARY_ARM, *CONTROL_ARMS)
MAX_INLINE_BYTES = 512 * 1024 * 1024
MAX_TREE_BYTES = 32 * 1024 * 1024
