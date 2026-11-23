"""Per-gate thresholds taken from the architecture spec and the paper plan.

These numbers are part of the harness contract. They are not estimated from
a result file, and a study cannot loosen them by writing a smaller value
into its metrics. Pilots and the headline do not share one seed minimum:
the spec uses 3 seeds for pilots and 5 for the headline.
"""

PILOT_SEEDS = 3
HEADLINE_SEEDS = 5
G0_MIN_UNIT_REDUCTION = 0.15
G0_MAX_BYTE_FALLBACK_PERCENT = 1
G0_VOCAB_CAP = 64000
G0_RECORDED_VOCAB = 65536
G0_MAX_BUILD_CPU_HOURS = 2
G0_MAX_BUILD_RAM_GB = 16
G1_SD_MARGIN = 1
G3_MAX_REL_ERR = 0.001
G3_MAX_SHEAFHEAT_FRACTION = 0.15
G4_MIN_BASELINE_SD = 2
P4_SEEDS_15M = 3
P4_SEEDS_30M = 3
P4_SEEDS_60M = 2
P4_SEEDS_120M = 1
P4_SEED_SCHEDULE = {
    15_000_000: P4_SEEDS_15M,
    30_000_000: P4_SEEDS_30M,
    60_000_000: P4_SEEDS_60M,
    120_000_000: P4_SEEDS_120M,
}
MAX_FLOP_RELATIVE_TOLERANCE = 0.05
ALPHA = 0.05
MIN_BOOTSTRAP_DRAWS = 1000
MAX_BOOTSTRAP_DRAWS = 100_000
PRIMARY_ARM = "hypergraph"
CONTROL_ARMS = ("superword_bpe", "randomized_hypergraph")
REQUIRED_ARMS = (PRIMARY_ARM, *CONTROL_ARMS)
MAX_INLINE_BYTES = 512 * 1024 * 1024
MAX_TREE_BYTES = 32 * 1024 * 1024
