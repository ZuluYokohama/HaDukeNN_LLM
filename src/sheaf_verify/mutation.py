"""Scientific, topological, statistical, and cryptographic mutation adequacy scoring.

Executing lines of verification code does not by itself prove that the gates
catch subtle scientific or provenance faults. ``run_mutation_suite`` injects a
curated catalog of single-fault mutants into passing synthetic stores and
computes the formal Mutation Score ``MS = K / (G - E)`` where ``K`` is killed
mutants, ``G`` is generated mutants, and ``E`` is equivalent mutants.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from sheaf_verify.checks import incidence_sha256
from sheaf_verify.hashing import canonical_bytes
from sheaf_verify.ledger import Ledger
from sheaf_verify.runner import run
from sheaf_verify.synthetic import TEST_KEY, build
from sheaf_verify.verify import verify_store


@dataclass(frozen=True)
class MutantOutcome:
    id: str
    category: str
    description: str
    killed: bool
    detail: str


@dataclass(frozen=True)
class MutationReport:
    generated: int
    equivalent: int
    killed: int
    survived: int
    mutation_score: float
    outcomes: tuple[MutantOutcome, ...]


def _store_rejects(store, *, key: bytes = TEST_KEY) -> tuple[bool, str]:
    vrep = verify_store(store, key=key)
    if not vrep.ok:
        return True, f"integrity failure: {vrep.problems[0]}"
    failed_gates = [row["gate_id"] for row in vrep.gate_rows if row["status"] != "pass"]
    if failed_gates:
        return True, f"gate failure: {', '.join(failed_gates)}"
    open_obj = [item["id"] for item in vrep.checklist if not item["answered"]]
    if open_obj:
        return True, f"open objection: {open_obj[0]}"
    return False, "SURVIVED (store verified and all gates passed)"


def _mut_heldout_leak(parts: dict) -> None:
    parts["hypergraph"]["document_ids"] = ["doc-train", "doc-test"]


def _mut_omit_graph_flops(parts: dict) -> None:
    flops = parts["metrics"]["arms"]["hypergraph"]["flops"]
    flops["graph_build"] = 0
    flops["total"] = flops["train"]


def _mut_unequal_train_bytes(parts: dict) -> None:
    parts["metrics"]["arms"]["superword_bpe"]["train_bytes"] = 4


def _mut_token_denominator(parts: dict) -> None:
    parts["metrics"]["arms"]["hypergraph"]["eval"]["denominator"] = "tokens"


def _mut_forged_pvalue(parts: dict) -> None:
    parts["metrics"]["reported"] = {
        "hypergraph_vs_randomized_hypergraph": {"df": 8.0, "g": -2.0, "high": -0.2, "low": -0.8, "p": 0.0001, "t": -9.0},
        "hypergraph_vs_superword_bpe": {"df": 8.0, "g": -2.0, "high": -0.2, "low": -0.8, "p": 0.0001, "t": -9.0},
    }


def _mut_future_causal_leak(parts: dict) -> None:
    parts["trace"]["perturbed_outputs"][0] = "leaked-past-token"


def _mut_open_sheaf_cocycle(parts: dict) -> None:
    parts["hypergraph"]["sheaf"]["operations"][1]["morphism"] = {"a": "drifted"}


def _mut_broken_2complex_boundary(parts: dict) -> None:
    parts["hypergraph"]["complex"] = {"edges": [[0, 1], [1, 2]], "faces": [[0, 1, 2]], "num_vertices": 3}


def _mut_identical_random_null(parts: dict) -> None:
    parts["random"]["hyperedges"] = parts["hypergraph"]["hyperedges"]
    parts["metrics"]["arms"]["randomized_hypergraph"]["incidence_sha256"] = incidence_sha256(parts["random"])


def _mut_cuda_missing_host_ref(parts: dict) -> None:
    parts["kernel_meta"] = {"backend": "cuda", "entry": "apply"}


PART_MUTANTS: tuple[tuple[str, str, str, Callable[[dict], None]], ...] = (
    ("MUT-01-LEAK", "decontamination", "Inject held-out test document into train hypergraph", _mut_heldout_leak),
    ("MUT-02-GRAPH-FLOPS", "accounting", "Zero out hypergraph construction FLOPs in all-in budget", _mut_omit_graph_flops),
    ("MUT-03-BYTE-MISMATCH", "accounting", "Perturb control arm train_bytes away from train corpus size", _mut_unequal_train_bytes),
    ("MUT-04-TOKEN-DENOM", "metrics", "Replace raw_corpus_bytes BPB denominator with token count", _mut_token_denominator),
    ("MUT-05-FORGED-STATS", "statistics", "Submit fabricated trainer p-value and t-statistic", _mut_forged_pvalue),
    ("MUT-06-CAUSAL-LEAK", "causality", "Allow future suffix perturbation to alter past output", _mut_future_causal_leak),
    ("MUT-07-SHEAF-COCYCLE", "topology", "Break sheaf restriction morphism path commutativity", _mut_open_sheaf_cocycle),
    ("MUT-08-COMPLEX-BOUNDARY", "topology", "Break 2-complex boundary identity B1*B2 != 0", _mut_broken_2complex_boundary),
    ("MUT-09-NULL-GRAPH-CLONE", "controls", "Copy study hypergraph incidence into randomized null control", _mut_identical_random_null),
    ("MUT-10-CUDA-NO-HOST-REF", "reproducibility", "Register CUDA device kernel without host reference hash", _mut_cuda_missing_host_ref),
)


def run_mutation_suite(work_root: Path) -> MutationReport:
    """Execute all 12 deterministic scientific and cryptographic mutants and score adequacy."""
    work_root = Path(work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    outcomes: list[MutantOutcome] = []

    for mid, cat, desc, mutator in PART_MUTANTS:
        store = build(work_root / mid.lower(), mutate=mutator)
        killed, detail = _store_rejects(store)
        outcomes.append(MutantOutcome(mid, cat, desc, killed, detail))

    # MUT-11: Post-hoc claim registered after metrics
    store_late = build(work_root / "mut-11-late-claim")
    claim_inputs = store_late.binding()["claim"][1]["inputs"]
    store_late.register(
        kind="claim",
        role="claim",
        registered_at="2026-03-01T00:00:00Z",
        payload=canonical_bytes({
            "alpha": 0.05,
            "description": "Post-hoc claim registered after metrics already existed.",
            "direction": "less",
            "format": "sheaf-verify.claim.v1",
            "hypothesis_id": "post-hoc-harking",
            "primary_contrast": "superword_bpe",
            "primary_metric": "bpb",
        }),
        inputs=claim_inputs,
    )
    run(store_late, key=TEST_KEY)
    killed_11, detail_11 = _store_rejects(store_late)
    outcomes.append(MutantOutcome(
        "MUT-11-POSTHOC-CLAIM",
        "preregistration",
        "Register updated hypothesis claim timestamped after metrics",
        killed_11,
        detail_11,
    ))

    # MUT-12: Ledger tamper (flip fail/pass or alter criterion text without re-sealing)
    store_tamper = build(work_root / "mut-12-ledger-tamper")
    lpath = Ledger(store_tamper.root).path
    lines = lpath.read_bytes().splitlines()
    first_rec = json.loads(lines[0].decode("utf-8"))
    first_rec["criteria"][0]["detail"] = "tampered criterion detail"
    lines[0] = canonical_bytes(first_rec)
    lpath.write_bytes(b"\n".join(lines) + b"\n")
    killed_12, detail_12 = _store_rejects(store_tamper)
    outcomes.append(MutantOutcome(
        "MUT-12-LEDGER-TAMPER",
        "cryptography",
        "Modify a sealed ledger line without invalidating record_sha256/HMAC",
        killed_12,
        detail_12,
    ))

    generated = len(outcomes)
    equivalent = 0
    killed_count = sum(1 for item in outcomes if item.killed)
    survived = generated - equivalent - killed_count
    score = killed_count / (generated - equivalent) if (generated - equivalent) > 0 else 1.0
    return MutationReport(
        generated=generated,
        equivalent=equivalent,
        killed=killed_count,
        survived=survived,
        mutation_score=score,
        outcomes=tuple(outcomes),
    )
