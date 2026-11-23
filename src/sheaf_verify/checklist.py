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
    Objection("OBJ-G0-UNITS", "G0", "gate", "Canonical units are not at least 15% fewer than GPT-2 BPE on the valid split."),
    Objection("OBJ-G0-FALLBACK", "G0", "gate", "Byte fallback is above 1%."),
    Objection("OBJ-G0-VOCAB", "G0", "gate", "Vocabulary size exceeds the written cap of 64,000. V=65,536 is the flagged inconsistency."),
    Objection("OBJ-G0-BUILD", "G0", "gate", "The build exceeds 2 CPU-hours or 16 GB peak RAM."),
    Objection("OBJ-G1-BPB", "G1", "gate", "hnt64k test BPB_corpus is above struct32k plus one standard deviation."),
    Objection("OBJ-G1-TRAIN-FLOPS", "G1", "gate", "hnt64k train FLOPs per byte are not lower than struct32k and sw64k."),
    Objection("OBJ-PILOT-SEEDS", "G1", "gate", "A pilot arm has fewer than 3 seeds."),
    Objection("OBJ-TEST-IN-GRAPH", "H0", "gate", "Train hyperedges cite validation or test documents."),
    Objection("OBJ-SPLIT-OVERLAP", "H0", "gate", "Train, validation, and test document ids overlap or the byte hashes are not distinct."),
    Objection("OBJ-CHRONOLOGY", "H0", "gate", "The holdout is not strictly after the training split."),
    Objection("OBJ-FUTURE-LEAK", "H4", "gate", "Perturbing a future position changes an earlier output."),
    Objection("OBJ-CAUSAL-VACUOUS", "H4", "gate", "The causality probe does not change the future positions it claims to perturb."),
    Objection("OBJ-LOOP", "H4", "gate", "Attached sheaf restriction maps do not agree on converging paths."),
    Objection("OBJ-BOUNDARY", "H0", "gate", "An attached 2-complex fails the boundary identity."),
    Objection("OBJ-UNEQUAL-VOCAB", "G1", "gate", "The arms do not share one unit inventory size, or sw64k does not match hnt64k."),
    Objection("OBJ-NO-BPE-CONTROL", "G1", "gate", "The equal-size sw64k superword-BPE control is missing."),
    Objection("OBJ-NO-RANDOM-GRAPH", "G4", "gate", "There is no randomized hypergraph of the same order and cardinality histogram."),
    Objection("OBJ-FLOPS-OMIT-GRAPH", "G1", "gate", "Reported FLOPs omit graph construction."),
    Objection("OBJ-BYTE-MISMATCH", "G1", "gate", "The arms were not trained on the same number of bytes."),
    Objection("OBJ-BPB-TOKEN-DENOM", "G1", "gate", "Bits per byte uses a token denominator instead of raw corpus bytes."),
    Objection("OBJ-SCORED-BYTES", "G1", "gate", "The arms were scored on different numbers of bytes."),
    Objection("OBJ-FEW-SEEDS", "G4", "gate", "A headline arm has fewer than 5 seeds."),
    Objection("OBJ-STATS", "G4", "gate", "The reported test, interval, or effect size does not match a recomputation."),
    Objection("OBJ-CONTROLS-DISAGREE", "G4", "gate", "A superiority claim is carried by only one of the two controls."),
    Objection("OBJ-ENV", "P0", "gate", "Python, CUDA, driver, GPU, or platform is missing."),
    Objection("OBJ-CODE", "P0", "gate", "The code snapshot does not match its listing or git tree id."),
    Objection("OBJ-CHECKPOINT-IN-GIT", "P0", "gate", "Checkpoint bytes were stored in the harness instead of by hash."),
    Objection("OBJ-KERNEL-HOST", "G3", "gate", "A device kernel has no distinct host-reference payload."),
    Objection("OBJ-FIGURE", "H7", "gate", "The figure is not bound to the metrics payload."),
    Objection("OBJ-PREREG", "P0", "gate", "The claim is not a frozen preregistration."),
    Objection("OBJ-CLAIM-AFTER", "P0", "gate", "The claim was registered after the metrics or quotes their hash."),
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
