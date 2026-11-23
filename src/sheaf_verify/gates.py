"""Declarative control graph.

A gate passes only when every criterion it records has passed, including the
upstream criteria the runner adds. Those upstream criteria cite the ledger
record that passed against the same artifact hashes. Changing an input and
forgetting to re-run a downstream gate makes verification fail.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from sheaf_verify.checks import (
    bad,
    boundary_criterion,
    bpb_criteria,
    causality_criteria,
    claim_criteria,
    code_tree_criteria,
    corpus_criteria,
    document_ids,
    environment_criteria,
    flop_and_byte_criteria,
    kernel_criteria,
    leakage_criteria,
    loop_criterion,
    ok,
    random_control_criteria,
    seed_criteria,
    stats_criteria,
    vocab_criteria,
)
from sheaf_verify.context import Context
from sheaf_verify.hashing import SheafError, as_int, parse_json_bytes
from sheaf_verify.policy import REQUIRED_ARMS

Evaluator = Callable[[Context], list]


@dataclass(frozen=True)
class Gate:
    id: str
    title: str
    binds: tuple[str, ...]
    depends_on: tuple[str, ...]
    evaluate: Evaluator


def _ids(manifest: dict) -> set[str] | None:
    return document_ids(manifest)


def _order(manifest: dict):
    metadata = manifest.get("metadata")
    if not isinstance(metadata, dict):
        return None
    return metadata.get("order")


def evaluate_p0(ctx: Context) -> list:
    return claim_criteria(ctx.payload("claim"))


def evaluate_g0(ctx: Context) -> list:
    rows = {}
    for split in ("train", "valid", "test"):
        role = f"corpus.{split}"
        manifest = ctx.manifest(role)
        identifiers = _ids(manifest)
        if identifiers is None:
            return [bad("corpus.documents", f"{role} is missing unique document ids")]
        rows[split] = {
            "payload_sha256": manifest["payload_sha256"],
            "size_bytes": manifest["size_bytes"],
            "document_ids": identifiers,
            "order": _order(manifest),
        }
    return corpus_criteria(rows)


def evaluate_h4(ctx: Context) -> list:
    criteria = seed_criteria(ctx.parse("seeds"), ctx.parse("metrics"))
    if ctx.manifest_sha("seeds") in ctx.manifest("metrics").get("inputs", []):
        criteria.append(ok("seeds.provenance", "metrics cite the seeds manifest"))
    else:
        criteria.append(bad("seeds.provenance", "metrics do not cite the seeds manifest"))
    return criteria


def evaluate_h5(ctx: Context) -> list:
    listing = ctx.parse("code")
    if not isinstance(listing, list):
        return [bad("code.listing", "code listing is not a list")]
    digest = ctx.manifest("code")["payload_sha256"]
    files = []
    for entry in listing:
        files.append((entry["path"], entry["mode"], ctx.snapshot_bytes(digest, entry["path"])))
    criteria = code_tree_criteria(ctx.manifest("code"), listing, files)
    criteria.extend(environment_criteria(ctx.parse("environment")))
    config = ctx.parse("config")
    if isinstance(config, dict) and config:
        criteria.append(ok("config.object", "config is a non-empty JSON object"))
    else:
        criteria.append(bad("config.object", "config must be a non-empty JSON object"))
    kernel = ctx.manifest("kernel")
    input_payloads = set()
    for manifest_sha in kernel.get("inputs", []):
        input_payloads.add(ctx.store.read_manifest(manifest_sha)["payload_sha256"])
    criteria.extend(kernel_criteria(
        kernel.get("metadata"),
        kernel["payload_sha256"],
        kernel["size_bytes"],
        input_payloads,
    ))
    required = {
        ctx.manifest_sha("code"),
        ctx.manifest_sha("config"),
        ctx.manifest_sha("seeds"),
        ctx.manifest_sha("kernel"),
    }
    for arm in REQUIRED_ARMS:
        checkpoint = ctx.manifest(f"checkpoint.{arm}")
        if checkpoint.get("storage") != "external" or checkpoint.get("kind") != "checkpoint":
            criteria.append(bad(f"checkpoint.{arm}.external", "checkpoint bytes must stay outside the store"))
        elif ctx.store.payload_path(checkpoint["payload_sha256"]).exists():
            criteria.append(bad(f"checkpoint.{arm}.external", "checkpoint hash collides with a stored payload"))
        else:
            criteria.append(ok(f"checkpoint.{arm}.external", "checkpoint is recorded by hash, size, and locator"))
        if required <= set(checkpoint.get("inputs", [])):
            criteria.append(ok(f"checkpoint.{arm}.inputs", "checkpoint cites code, config, seeds, and kernel"))
        else:
            criteria.append(bad(f"checkpoint.{arm}.inputs", "checkpoint does not cite code, config, seeds, and kernel"))
        log = ctx.manifest(f"log.{arm}")
        if log.get("size_bytes", 0) > 0 and ctx.manifest_sha(f"checkpoint.{arm}") in log.get("inputs", []):
            criteria.append(ok(f"log.{arm}", "log is non-empty and cites its checkpoint"))
        else:
            criteria.append(bad(f"log.{arm}", "log is empty or does not cite its checkpoint"))
    return criteria


def evaluate_g1(ctx: Context) -> list:
    train_size = as_int(ctx.manifest("corpus.train")["size_bytes"], "train size")
    criteria = flop_and_byte_criteria(ctx.parse("metrics"), train_size)
    inputs = set(ctx.manifest("metrics").get("inputs", []))
    missing = [role for role in ("corpus.train", "corpus.valid", "corpus.test") if ctx.manifest_sha(role) not in inputs]
    if missing:
        criteria.append(bad("flops.corpus_inputs", "metrics do not cite " + ", ".join(missing)))
    else:
        criteria.append(ok("flops.corpus_inputs", "metrics cite the three corpus manifests"))
    return criteria


def evaluate_h0(ctx: Context) -> list:
    train_ids = _ids(ctx.manifest("corpus.train"))
    valid_ids = _ids(ctx.manifest("corpus.valid"))
    test_ids = _ids(ctx.manifest("corpus.test"))
    if train_ids is None or valid_ids is None or test_ids is None:
        return [bad("hypergraph.documents", "corpus document ids are missing")]
    criteria = []
    for role, label in (("hypergraph", "hypergraph"), ("hypergraph.random", "hypergraph.random")):
        document = ctx.parse(role)
        manifest = ctx.manifest(role)
        criteria.extend(leakage_criteria(
            document,
            train_ids=train_ids,
            valid_ids=valid_ids,
            test_ids=test_ids,
            inputs=manifest.get("inputs", []),
            train_manifest=ctx.manifest_sha("corpus.train"),
            valid_manifest=ctx.manifest_sha("corpus.valid"),
            test_manifest=ctx.manifest_sha("corpus.test"),
            label=label,
        ))
        complex_doc = document.get("complex") if isinstance(document, dict) else None
        criteria.append(boundary_criterion(complex_doc, label))
    return criteria


def evaluate_h1(ctx: Context) -> list:
    held = {ctx.manifest_sha("corpus.valid"), ctx.manifest_sha("corpus.test")}
    return vocab_criteria(
        ctx.parse("vocabulary"),
        ctx.parse("metrics"),
        ctx.manifest("vocabulary").get("inputs", []),
        ctx.manifest_sha("corpus.train"),
        held,
    )


def evaluate_h6(ctx: Context) -> list:
    test_size = as_int(ctx.manifest("corpus.test")["size_bytes"], "test size")
    return bpb_criteria(ctx.parse("metrics"), test_size)


def evaluate_h2(ctx: Context) -> list:
    hypergraph = ctx.parse("hypergraph")
    criteria = causality_criteria(ctx.parse("trace.causality"), hypergraph)
    if ctx.manifest_sha("hypergraph") in ctx.manifest("trace.causality").get("inputs", []):
        criteria.append(ok("causality.provenance", "the trace cites the study hypergraph"))
    else:
        criteria.append(bad("causality.provenance", "the trace does not cite the study hypergraph"))
    sheaf = hypergraph.get("sheaf") if isinstance(hypergraph, dict) else None
    criteria.append(loop_criterion(sheaf))
    return criteria


def evaluate_h3(ctx: Context) -> list:
    return random_control_criteria(ctx.parse("hypergraph"), ctx.parse("hypergraph.random"), ctx.parse("metrics"))


def evaluate_h7(ctx: Context) -> list:
    try:
        claim = parse_json_bytes(ctx.payload("claim"), "claim")
    except SheafError:
        claim = None
    if not isinstance(claim, dict):
        claim = None
    return stats_criteria(ctx.parse("metrics"), ctx.parse("seeds"), claim)


def evaluate_p1(ctx: Context) -> list:
    claim_time = ctx.manifest("claim")["registered_at"]
    metrics_time = ctx.manifest("metrics")["registered_at"]
    criteria = []
    if isinstance(claim_time, str) and isinstance(metrics_time, str) and claim_time < metrics_time:
        criteria.append(ok("stage.claim_before_metrics", "the claim was registered before the metrics"))
    else:
        criteria.append(bad("stage.claim_before_metrics", "the claim was not registered before the metrics"))
    digest = ctx.manifest("metrics")["payload_sha256"]
    if digest.encode("ascii") in ctx.payload("claim"):
        criteria.append(bad("stage.claim_blind", "the claim payload contains the metrics hash"))
    else:
        criteria.append(ok("stage.claim_blind", "the claim payload does not contain the metrics hash"))
    return criteria


def evaluate_p2(ctx: Context) -> list:
    kinds = {arm: ctx.parse("metrics")["arms"][arm]["kind"] for arm in REQUIRED_ARMS}
    if tuple(kinds) == REQUIRED_ARMS and all(kinds[arm] == arm for arm in REQUIRED_ARMS):
        return [ok("stage.controls", "the stage sees the study graph, superword-BPE, and the randomized graph")]
    return [bad("stage.controls", "the control set is incomplete")]


def evaluate_p3(ctx: Context) -> list:
    external = []
    for arm in REQUIRED_ARMS:
        checkpoint = ctx.manifest(f"checkpoint.{arm}")
        external.append(checkpoint.get("storage") == "external" and checkpoint.get("kind") == "checkpoint")
    if all(external):
        return [ok("stage.checkpoints", "all three checkpoints are external")]
    return [bad("stage.checkpoints", "a checkpoint is not external")]


def evaluate_p4(ctx: Context) -> list:
    payload = ctx.payload("figure")
    metadata = ctx.manifest("figure").get("metadata")
    criteria = []
    if not isinstance(metadata, dict):
        return [bad("figure.metadata", "figure metadata is missing")]
    expected = ctx.manifest("metrics")["payload_sha256"]
    if metadata.get("plots") == "bpb" and metadata.get("metrics_payload_sha256") == expected:
        criteria.append(ok("figure.binding", "figure metadata cites the metrics payload"))
    else:
        criteria.append(bad("figure.binding", "figure metadata does not cite the metrics payload"))
    if ctx.manifest_sha("metrics") in ctx.manifest("figure").get("inputs", []):
        criteria.append(ok("figure.inputs", "figure provenance cites the metrics manifest"))
    else:
        criteria.append(bad("figure.inputs", "figure provenance does not cite the metrics manifest"))
    stripped = payload.lstrip()
    if stripped.startswith(b"<svg") or stripped.startswith(b"<?xml"):
        criteria.append(ok("figure.markup", "figure payload is XML or SVG"))
    else:
        criteria.append(bad("figure.markup", "figure payload does not start with an SVG or XML marker"))
    return criteria


GATES: tuple[Gate, ...] = (
    Gate("P0", "Claim freeze", ("claim",), (), evaluate_p0),
    Gate("G0", "Corpus identity and disjoint splits", ("corpus.train", "corpus.valid", "corpus.test"), (), evaluate_g0),
    Gate("H4", "Seeds", ("seeds", "metrics"), (), evaluate_h4),
    Gate(
        "H5",
        "Code, kernel, environment, config, checkpoints, and logs",
        (
            "code",
            "environment",
            "config",
            "kernel",
            "metrics",
            "checkpoint.hypergraph",
            "checkpoint.superword_bpe",
            "checkpoint.randomized_hypergraph",
            "log.hypergraph",
            "log.superword_bpe",
            "log.randomized_hypergraph",
        ),
        (),
        evaluate_h5,
    ),
    Gate("G1", "Matched FLOPs and train bytes", ("corpus.train", "metrics"), ("G0",), evaluate_g1),
    Gate(
        "H0",
        "Train-only hypergraph",
        ("hypergraph", "hypergraph.random", "corpus.train", "corpus.valid", "corpus.test"),
        ("G0",),
        evaluate_h0,
    ),
    Gate(
        "H1",
        "Equal-size vocabulary and superword-BPE",
        ("corpus.train", "corpus.valid", "corpus.test", "vocabulary", "metrics"),
        ("G0",),
        evaluate_h1,
    ),
    Gate("H6", "Raw-byte BPB and identical scored bytes", ("corpus.test", "metrics"), ("G0",), evaluate_h6),
    Gate("H2", "Causality", ("hypergraph", "trace.causality"), ("H0",), evaluate_h2),
    Gate("H3", "Randomized hypergraph control", ("hypergraph", "hypergraph.random", "metrics"), ("H0",), evaluate_h3),
    Gate("H7", "Statistics", ("metrics", "seeds", "claim"), ("H4", "H6"), evaluate_h7),
    Gate("P1", "Data protocol", ("claim", "metrics"), ("P0", "G0"), evaluate_p1),
    Gate(
        "P2",
        "Units and controls",
        ("hypergraph", "vocabulary", "trace.causality", "hypergraph.random", "metrics"),
        ("H0", "H1", "H2", "H3"),
        evaluate_p2,
    ),
    Gate(
        "P3",
        "Matched training",
        ("metrics", "seeds", "code", "environment", "config", "checkpoint.hypergraph", "checkpoint.superword_bpe", "checkpoint.randomized_hypergraph"),
        ("G1", "H4", "H5"),
        evaluate_p3,
    ),
    Gate(
        "P4",
        "Evaluation, statistics, and figure",
        ("metrics", "seeds", "claim", "figure"),
        ("H6", "H7", "P0", "P1", "P2", "P3"),
        evaluate_p4,
    ),
)


def topological(gates: tuple[Gate, ...] = GATES) -> tuple[Gate, ...]:
    pending = list(gates)
    done: list[Gate] = []
    finished: set[str] = set()
    while pending:
        ready = [gate for gate in pending if set(gate.depends_on) <= finished]
        if not ready:
            raise SheafError("gate graph has a cycle")
        chosen = ready[0]
        pending.remove(chosen)
        done.append(chosen)
        finished.add(chosen.id)
    return tuple(done)


if tuple(gate.id for gate in topological()) != tuple(gate.id for gate in GATES):
    raise SheafError("GATES is not in topological order")
