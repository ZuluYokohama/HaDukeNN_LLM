"""Declarative control graph.

Gate ids and pass conditions follow the architecture spec's stages H0–H7
and gates G0–G6, plus the paper plan's phases P0–P6. A gate passes only
when every criterion it records has passed, including the upstream criteria
the runner adds. Those upstream criteria cite the ledger record that passed
against the same artifact hashes.
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
from sheaf_verify.hal import (
    evaluate_hal0,
    evaluate_hal1,
    evaluate_hal2,
    evaluate_hal3,
    evaluate_hal4,
    evaluate_hal5,
    evaluate_hal6,
)
from sheaf_verify.hashing import SheafError, as_int, parse_json_bytes
from sheaf_verify.policy import HEADLINE_SEEDS, REQUIRED_ARMS
from sheaf_verify.sheaf_cohomology import cellular_sheaf_criteria
from sheaf_verify.spec_gates import (
    g0_criteria,
    g1_criteria,
    g2_criteria,
    g3_criteria,
    g4_criteria,
    g5_criteria,
    g6_criteria,
    h1_criteria,
    h7_criteria,
    p0_fairness,
    p1_criteria,
    p2_criteria,
    p3_criteria,
    p4_criteria,
    p5_criteria,
    p6_criteria,
    stage_criteria,
)

Evaluator = Callable[[Context], list]


@dataclass(frozen=True)
class Gate:
    id: str
    title: str
    binds: tuple[str, ...]
    depends_on: tuple[str, ...]
    evaluate: Evaluator


def _metrics(ctx: Context) -> dict:
    document = ctx.parse("metrics")
    if not isinstance(document, dict):
        return {}
    return document


def _spec(ctx: Context) -> dict:
    spec = _metrics(ctx).get("spec")
    return spec if isinstance(spec, dict) else {}


def _claim(ctx: Context):
    try:
        claim = parse_json_bytes(ctx.payload("claim"), "claim")
    except SheafError:
        return None
    return claim if isinstance(claim, dict) else None


def _ids(manifest: dict) -> set[str] | None:
    return document_ids(manifest)


def _order(manifest: dict):
    metadata = manifest.get("metadata")
    if not isinstance(metadata, dict):
        return None
    return metadata.get("order")


def evaluate_p0(ctx: Context) -> list:
    criteria = claim_criteria(ctx.payload("claim"))
    claim_time = ctx.manifest("claim")["registered_at"]
    metrics_time = ctx.manifest("metrics")["registered_at"]
    if isinstance(claim_time, str) and isinstance(metrics_time, str) and claim_time < metrics_time:
        criteria.append(ok("stage.claim_before_metrics", "the claim was registered before the metrics"))
    else:
        criteria.append(bad("stage.claim_before_metrics", "the claim was not registered before the metrics"))
    digest = ctx.manifest("metrics")["payload_sha256"]
    if digest.encode("ascii") in ctx.payload("claim"):
        criteria.append(bad("stage.claim_blind", "the claim payload contains the metrics hash"))
    else:
        criteria.append(ok("stage.claim_blind", "the claim payload does not contain the metrics hash"))
    criteria.extend(p0_fairness(_spec(ctx).get("p0")))
    criteria.extend(_reproducibility(ctx))
    return criteria


def evaluate_g0(ctx: Context) -> list:
    return g0_criteria(_spec(ctx).get("g0"))


def evaluate_h0(ctx: Context) -> list:
    criteria = stage_criteria(_metrics(ctx), "H0")
    rows = {}
    for split in ("train", "valid", "test"):
        role = f"corpus.{split}"
        manifest = ctx.manifest(role)
        identifiers = _ids(manifest)
        if identifiers is None:
            return criteria + [bad("corpus.documents", f"{role} is missing unique document ids")]
        rows[split] = {
            "payload_sha256": manifest["payload_sha256"],
            "size_bytes": manifest["size_bytes"],
            "document_ids": identifiers,
            "order": _order(manifest),
        }
    criteria.extend(corpus_criteria(rows))
    train_ids = rows["train"]["document_ids"]
    valid_ids = rows["valid"]["document_ids"]
    test_ids = rows["test"]["document_ids"]
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


def evaluate_g1(ctx: Context) -> list:
    metrics = _metrics(ctx)
    train_size = as_int(ctx.manifest("corpus.train")["size_bytes"], "train size")
    test_size = as_int(ctx.manifest("corpus.test")["size_bytes"], "test size")
    criteria = g1_criteria(_spec(ctx).get("g1"), train_size, test_size)
    criteria.extend(flop_and_byte_criteria(metrics, train_size))
    criteria.extend(bpb_criteria(metrics, test_size))
    inputs = set(ctx.manifest("metrics").get("inputs", []))
    missing = [role for role in ("corpus.train", "corpus.valid", "corpus.test") if ctx.manifest_sha(role) not in inputs]
    if missing:
        criteria.append(bad("flops.corpus_inputs", "metrics do not cite " + ", ".join(missing)))
    else:
        criteria.append(ok("flops.corpus_inputs", "metrics cite the three corpus manifests"))
    held = {ctx.manifest_sha("corpus.valid"), ctx.manifest_sha("corpus.test")}
    criteria.extend(vocab_criteria(
        ctx.parse("vocabulary"),
        metrics,
        ctx.manifest("vocabulary").get("inputs", []),
        ctx.manifest_sha("corpus.train"),
        held,
    ))
    return criteria


def evaluate_h1(ctx: Context) -> list:
    criteria = stage_criteria(_metrics(ctx), "H1")
    criteria.extend(h1_criteria(_spec(ctx).get("g1")))
    return criteria


def evaluate_g2(ctx: Context) -> list:
    return g2_criteria(_spec(ctx).get("g2"))


def evaluate_h2(ctx: Context) -> list:
    return stage_criteria(_metrics(ctx), "H2")


def evaluate_g3(ctx: Context) -> list:
    criteria = g3_criteria(_spec(ctx).get("g3"))
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
    return criteria


def evaluate_h3(ctx: Context) -> list:
    return stage_criteria(_metrics(ctx), "H3")


def evaluate_p1(ctx: Context) -> list:
    return p1_criteria(_spec(ctx).get("p1"))


def evaluate_p2(ctx: Context) -> list:
    headline = _spec(ctx).get("g4", {}).get("headline", {})
    seeds = headline.get("bpb_per_seed") if isinstance(headline, dict) else None
    count = len(seeds) if isinstance(seeds, list) else 0
    return p2_criteria(_spec(ctx).get("p2"), count)


def evaluate_p3(ctx: Context) -> list:
    return p3_criteria(_spec(ctx).get("p3"))


def evaluate_p5(ctx: Context) -> list:
    return p5_criteria(_spec(ctx).get("p5"))


def evaluate_g4(ctx: Context) -> list:
    metrics = _metrics(ctx)
    criteria = g4_criteria(_spec(ctx).get("g4"))
    criteria.extend(seed_criteria(ctx.parse("seeds"), metrics, minimum=HEADLINE_SEEDS))
    if ctx.manifest_sha("seeds") in ctx.manifest("metrics").get("inputs", []):
        criteria.append(ok("seeds.provenance", "metrics cite the seeds manifest"))
    else:
        criteria.append(bad("seeds.provenance", "metrics do not cite the seeds manifest"))
    criteria.extend(random_control_criteria(ctx.parse("hypergraph"), ctx.parse("hypergraph.random"), metrics))
    criteria.extend(stats_criteria(metrics, ctx.parse("seeds"), _claim(ctx)))
    return criteria


def evaluate_h4(ctx: Context) -> list:
    criteria = stage_criteria(_metrics(ctx), "H4")
    hypergraph = ctx.parse("hypergraph")
    criteria.extend(causality_criteria(ctx.parse("trace.causality"), hypergraph))
    if ctx.manifest_sha("hypergraph") in ctx.manifest("trace.causality").get("inputs", []):
        criteria.append(ok("causality.provenance", "the trace cites the study hypergraph"))
    else:
        criteria.append(bad("causality.provenance", "the trace does not cite the study hypergraph"))
    sheaf = hypergraph.get("sheaf") if isinstance(hypergraph, dict) else None
    criteria.append(loop_criterion(sheaf))
    if isinstance(hypergraph, dict) and "cellular_sheaf" in hypergraph:
        criteria.extend(cellular_sheaf_criteria(hypergraph.get("cellular_sheaf")))
    return criteria


def evaluate_g5(ctx: Context) -> list:
    return g5_criteria(_spec(ctx).get("g5"))


def evaluate_h5(ctx: Context) -> list:
    criteria = stage_criteria(_metrics(ctx), "H5")
    criteria.extend(p3_criteria(_spec(ctx).get("p3")))
    return criteria


def evaluate_p4(ctx: Context) -> list:
    return p4_criteria(_spec(ctx).get("g5"))


def evaluate_g6(ctx: Context) -> list:
    return g6_criteria(_spec(ctx).get("g6"))


def evaluate_h6(ctx: Context) -> list:
    return stage_criteria(_metrics(ctx), "H6")


def evaluate_h7(ctx: Context) -> list:
    criteria = stage_criteria(_metrics(ctx), "H7")
    criteria.extend(h7_criteria(_spec(ctx).get("h7")))
    criteria.extend(_figure(ctx))
    return criteria


def evaluate_p6(ctx: Context) -> list:
    return p6_criteria(_spec(ctx).get("p6"))


def _reproducibility(ctx: Context) -> list:
    listing = ctx.parse("code")
    criteria = []
    if not isinstance(listing, list):
        criteria.append(bad("code.listing", "code listing is not a list"))
    else:
        digest = ctx.manifest("code")["payload_sha256"]
        files = [(entry["path"], entry["mode"], ctx.snapshot_bytes(digest, entry["path"])) for entry in listing]
        criteria.extend(code_tree_criteria(ctx.manifest("code"), listing, files))
    criteria.extend(environment_criteria(ctx.parse("environment")))
    config = ctx.parse("config")
    if isinstance(config, dict) and config:
        criteria.append(ok("config.object", "config is a non-empty JSON object"))
    else:
        criteria.append(bad("config.object", "config must be a non-empty JSON object"))
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


def _figure(ctx: Context) -> list:
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


_REPRO_ROLES = (
    "claim",
    "metrics",
    "code",
    "environment",
    "config",
    "seeds",
    "kernel",
    "checkpoint.hypergraph",
    "checkpoint.superword_bpe",
    "checkpoint.randomized_hypergraph",
    "log.hypergraph",
    "log.superword_bpe",
    "log.randomized_hypergraph",
)

GATES: tuple[Gate, ...] = (
    Gate("P0", "Fairness", _REPRO_ROLES, (), evaluate_p0),
    Gate("G0", "Unit reduction on WT103 valid", ("metrics",), (), evaluate_g0),
    Gate(
        "H0",
        "Corpus stats",
        ("metrics", "hypergraph", "hypergraph.random", "corpus.train", "corpus.valid", "corpus.test"),
        ("G0",),
        evaluate_h0,
    ),
    Gate(
        "G1",
        "Units-only pilot",
        ("metrics", "corpus.train", "corpus.test", "vocabulary", "corpus.valid"),
        ("G0",),
        evaluate_g1,
    ),
    Gate("H1", "Units-only pilot stage", ("metrics",), ("G1",), evaluate_h1),
    Gate("G2", "Spectral and compositional embeddings", ("metrics",), ("G1",), evaluate_g2),
    Gate("H2", "Spectral and compositional stage", ("metrics",), ("G2",), evaluate_h2),
    Gate("G3", "Kernel microbench", ("metrics", "kernel"), ("G0",), evaluate_g3),
    Gate("H3", "Kernel stage", ("metrics",), ("G3",), evaluate_h3),
    Gate("P1", "Finish ablations", ("metrics",), ("P0",), evaluate_p1),
    Gate("P2", "Strong baseline", ("metrics",), ("P1",), evaluate_p2),
    Gate("P3", "Data", ("metrics",), ("P0",), evaluate_p3),
    Gate("P5", "Speed", ("metrics",), ("G3",), evaluate_p5),
    Gate(
        "G4",
        "Full HNT at matched all-in FLOPs",
        ("metrics", "seeds", "claim", "hypergraph", "hypergraph.random"),
        ("G2", "G3"),
        evaluate_g4,
    ),
    Gate(
        "H4",
        "Ablations and headline",
        ("metrics", "hypergraph", "trace.causality"),
        ("G4",),
        evaluate_h4,
    ),
    Gate("G5", "Scaling slope", ("metrics",), ("G4",), evaluate_g5),
    Gate("H5", "Scaling stage", ("metrics",), ("G5",), evaluate_h5),
    Gate("P4", "Scaling ladder", ("metrics",), ("G5",), evaluate_p4),
    Gate("G6", "Motif head and corpus memory", ("metrics",), ("G5",), evaluate_g6),
    Gate("H6", "Motif-head stage", ("metrics",), ("G6",), evaluate_h6),
    Gate("H7", "lm-eval and write-up", ("metrics", "figure"), ("H6",), evaluate_h7),
    Gate("P6", "Evals and replication", ("metrics",), ("H7",), evaluate_p6),
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

PROFILE_TRAINING = "training"
PROFILE_HAL = "hal"
PROFILES = frozenset({PROFILE_TRAINING, PROFILE_HAL})

HAL_GATES: tuple[Gate, ...] = (
    Gate("HAL0", "HAL profile and claim", ("claim",), (), evaluate_hal0),
    Gate("HAL1", "Environment live-probed", ("environment",), ("HAL0",), evaluate_hal1),
    Gate("HAL2", "Code tree hashed", ("code",), ("HAL0",), evaluate_hal2),
    Gate("HAL3", "Kernel host-reference binding", ("kernel",), ("HAL2",), evaluate_hal3),
    Gate("HAL4", "Numeric parity vs float64 reference", ("log.parity", "kernel"), ("HAL3",), evaluate_hal4),
    Gate("HAL5", "Performance from raw timings", ("claim", "log.timings"), ("HAL0",), evaluate_hal5),
    Gate("HAL6", "Thermal and power guards", ("claim", "log.guards", "environment"), ("HAL1",), evaluate_hal6),
)


def all_gates() -> tuple[Gate, ...]:
    return GATES + HAL_GATES


def normalize_profile(value: str | None) -> str:
    if value is None or value == "":
        return PROFILE_TRAINING
    if value not in PROFILES:
        raise SheafError(f"unknown study profile: {value}")
    return value


def gates_for_profile(profile: str | None) -> tuple[Gate, ...]:
    if normalize_profile(profile) == PROFILE_HAL:
        return HAL_GATES
    return GATES


def profile_from_records(records: list[dict]) -> str:
    found = set()
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("gate_id"), str):
            continue
        if "profile" in record:
            found.add(normalize_profile(record.get("profile")))
    if not found:
        return PROFILE_TRAINING
    if len(found) > 1:
        raise SheafError("ledger records disagree on study profile")
    return found.pop()


if tuple(gate.id for gate in topological(HAL_GATES)) != tuple(gate.id for gate in HAL_GATES):
    raise SheafError("HAL_GATES is not in topological order")
