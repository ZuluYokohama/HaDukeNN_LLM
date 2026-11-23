"""Adversarial-review checklist.

An objection is answered when the gate it names has a latest passing record,
or, for the two ledger objections, when verification found no problem of
that kind. An open objection does not by itself make ``verify`` fail. The
bundle command treats an open objection as a failed export.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Objection:
    id: str
    gate_id: str | None
    kind: str
    text: str


OBJECTIONS: tuple[Objection, ...] = (
    Objection("OBJ-TEST-IN-GRAPH", "H0", "gate", "Train hyperedges cite validation or test documents."),
    Objection("OBJ-SPLIT-OVERLAP", "G0", "gate", "Train, validation, and test document ids overlap or the byte hashes are not distinct."),
    Objection("OBJ-CHRONOLOGY", "G0", "gate", "The holdout is not strictly after the training split."),
    Objection("OBJ-FUTURE-LEAK", "H2", "gate", "Perturbing a future position changes an earlier output."),
    Objection("OBJ-CAUSAL-VACUOUS", "H2", "gate", "The causality probe does not change the future positions it claims to perturb."),
    Objection("OBJ-LOOP", "H2", "gate", "Attached sheaf restriction maps do not agree on converging paths."),
    Objection("OBJ-BOUNDARY", "H0", "gate", "An attached 2-complex fails the boundary identity."),
    Objection("OBJ-UNEQUAL-VOCAB", "H1", "gate", "The arms do not share one unit inventory size."),
    Objection("OBJ-NO-BPE-CONTROL", "H1", "gate", "There is no equal-size superword-BPE control."),
    Objection("OBJ-NO-RANDOM-GRAPH", "H3", "gate", "There is no randomized hypergraph of the same order and cardinality histogram."),
    Objection("OBJ-FLOPS-OMIT-GRAPH", "G1", "gate", "Reported FLOPs omit graph construction."),
    Objection("OBJ-BYTE-MISMATCH", "G1", "gate", "The arms were not trained on the same number of bytes."),
    Objection("OBJ-BPB-TOKEN-DENOM", "H6", "gate", "Bits per byte uses a token denominator instead of raw corpus bytes."),
    Objection("OBJ-SCORED-BYTES", "H6", "gate", "The arms were scored on different numbers of bytes."),
    Objection("OBJ-FEW-SEEDS", "H4", "gate", "The study uses fewer than the minimum number of seeds."),
    Objection("OBJ-STATS", "H7", "gate", "The reported test, interval, or effect size does not match a recomputation."),
    Objection("OBJ-CONTROLS-DISAGREE", "H7", "gate", "A superiority claim is carried by only one of the two controls."),
    Objection("OBJ-ENV", "H5", "gate", "Python, CUDA, driver, GPU, or platform is missing."),
    Objection("OBJ-CODE", "H5", "gate", "The code snapshot does not match its listing or git tree id."),
    Objection("OBJ-CHECKPOINT-IN-GIT", "H5", "gate", "Checkpoint bytes were stored in the harness instead of by hash."),
    Objection("OBJ-KERNEL-HOST", "H5", "gate", "A device kernel has no distinct host-reference payload."),
    Objection("OBJ-FIGURE", "P4", "gate", "The figure is not bound to the metrics payload."),
    Objection("OBJ-PREREG", "P0", "gate", "The claim is not a frozen preregistration."),
    Objection("OBJ-CLAIM-AFTER", "P1", "gate", "The claim was registered after the metrics or quotes their hash."),
    Objection("OBJ-LEDGER", None, "ledger", "The ledger was edited, reordered, truncated, or signed with the wrong key."),
    Objection("OBJ-PROVENANCE", None, "provenance", "A manifest hash, payload hash, or result link does not recompute."),
)


def evaluate_checklist(gate_status: dict[str, str], problems: list[str]) -> list[dict]:
    ledger_ok = not any(item.startswith("ledger") for item in problems)
    provenance_ok = not any(
        item.startswith("provenance") or item.startswith("manifest") or item.startswith("replay") or item.startswith("result")
        for item in problems
    )
    rows = []
    for objection in OBJECTIONS:
        if objection.kind == "ledger":
            answered = ledger_ok
            detail = "ledger verification found no problem" if answered else "ledger verification reported a problem"
        elif objection.kind == "provenance":
            answered = provenance_ok
            detail = "manifests, payloads, and result links recompute" if answered else "provenance verification reported a problem"
        else:
            status = gate_status.get(objection.gate_id or "", "missing")
            answered = status == "pass"
            detail = f"{objection.gate_id} is {status}"
        rows.append({
            "answered": answered,
            "detail": detail,
            "gate_id": objection.gate_id,
            "id": objection.id,
            "text": objection.text,
        })
    return rows
