"""Methodology predicates.

Each function returns criteria and does not decide whether a gate passes.
A gate passes only when every criterion it records has passed, including
the upstream-hash criteria added by the runner.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable

from sheaf_verify.git_tree import git_tree_sha1
from sheaf_verify.hashing import (
    GIT_SHA1_RE,
    SHA256_RE,
    SheafError,
    as_int,
    as_real,
    canonical_bytes,
    fmt_float,
    is_sha256,
    sha256_hex,
)
from sheaf_verify.policy import (
    ALPHA,
    CONTROL_ARMS,
    MAX_BOOTSTRAP_DRAWS,
    MAX_FLOP_RELATIVE_TOLERANCE,
    MIN_BOOTSTRAP_DRAWS,
    MIN_SEEDS,
    PRIMARY_ARM,
    REQUIRED_ARMS,
)
from sheaf_verify.stats import (
    bootstrap_mean_difference,
    hedges_g,
    paired_t,
    welch,
)

HYPERGRAPH_FORMAT = "sheaf-verify.hypergraph.v1"
VOCAB_FORMAT = "sheaf-verify.vocabulary.v1"
SEEDS_FORMAT = "sheaf-verify.seeds.v1"
METRICS_FORMAT = "sheaf-verify.metrics.v1"
TRACE_FORMAT = "sheaf-verify.causality.v1"
ENV_FORMAT = "sheaf-verify.environment.v1"
CLAIM_FORMAT = "sheaf-verify.claim.v1"
KERNEL_BACKENDS = frozenset({"cpu", "cuda", "opencl", "c"})
BLANK = frozenset({"", "unknown", "none", "n/a", "na"})


class Criterion:
    def __init__(self, identifier: str, passed: bool, detail: str):
        self.id = identifier
        self.passed = passed
        self.detail = detail

    def as_dict(self) -> dict:
        return {"detail": self.detail, "id": self.id, "passed": self.passed}


def ok(identifier: str, detail: str) -> Criterion:
    return Criterion(identifier, True, detail)


def bad(identifier: str, detail: str) -> Criterion:
    return Criterion(identifier, False, detail)


def document_ids(manifest: dict) -> set[str] | None:
    metadata = manifest.get("metadata")
    if not isinstance(metadata, dict):
        return None
    raw = metadata.get("document_ids")
    if not isinstance(raw, list) or not raw:
        return None
    if any(not isinstance(item, str) or not item for item in raw):
        return None
    if len(set(raw)) != len(raw):
        return None
    return set(raw)


def incidence_sha256(document: dict) -> str:
    edges = []
    for edge in document["hyperedges"]:
        edges.append({
            "documents": list(edge["documents"]),
            "id": edge["id"],
            "positions": list(edge["positions"]),
            "vertices": list(edge["vertices"]),
        })
    edges.sort(key=lambda item: item["id"])
    return sha256_hex(canonical_bytes(edges))


def cardinality_histogram(document: dict) -> dict[str, int]:
    counts: dict[str, int] = {}
    for edge in document["hyperedges"]:
        key = str(len(edge["vertices"]))
        counts[key] = counts.get(key, 0) + 1
    return {key: counts[key] for key in sorted(counts, key=int)}


def hypergraph_errors(document: dict) -> list[str]:
    if not isinstance(document, dict):
        return ["hypergraph is not an object"]
    if document.get("format") != HYPERGRAPH_FORMAT:
        return ["hypergraph format is not sheaf-verify.hypergraph.v1"]
    errors = []
    documents = document.get("document_ids")
    if not isinstance(documents, list) or not documents or any(not isinstance(item, str) or not item for item in documents):
        errors.append("document_ids must be a non-empty list of strings")
        documents = []
    vertices = document.get("vertices")
    edges = document.get("hyperedges")
    if not isinstance(vertices, list) or not vertices:
        errors.append("vertices must be a non-empty list")
        vertices = []
    if not isinstance(edges, list) or not edges:
        errors.append("hyperedges must be a non-empty list")
        edges = []
    positions = {}
    seen_vertices = set()
    for vertex in vertices:
        if not isinstance(vertex, dict):
            errors.append("a vertex is not an object")
            continue
        ident = vertex.get("id")
        position = vertex.get("position")
        doc = vertex.get("document")
        if not isinstance(ident, str) or not ident or ident in seen_vertices:
            errors.append("vertex ids must be unique non-empty strings")
        else:
            seen_vertices.add(ident)
        if isinstance(position, bool) or not isinstance(position, int) or position < 0:
            errors.append(f"vertex {ident} has a bad position")
        elif isinstance(ident, str):
            positions[ident] = position
        if doc not in documents:
            errors.append(f"vertex {ident} is not in the hypergraph document list")
    seen_edges = set()
    for edge in edges:
        if not isinstance(edge, dict):
            errors.append("a hyperedge is not an object")
            continue
        ident = edge.get("id")
        members = edge.get("vertices")
        edge_positions = edge.get("positions")
        edge_docs = edge.get("documents")
        if not isinstance(ident, str) or not ident or ident in seen_edges:
            errors.append("hyperedge ids must be unique non-empty strings")
        else:
            seen_edges.add(ident)
        if not isinstance(members, list) or len(members) < 2 or any(item not in positions for item in members):
            errors.append(f"hyperedge {ident} does not list at least two known vertices")
            continue
        expected = [positions[item] for item in members]
        if edge_positions != expected:
            errors.append(f"hyperedge {ident} positions do not match its vertices")
        if not isinstance(edge_docs, list) or any(item not in documents for item in edge_docs):
            errors.append(f"hyperedge {ident} names a document outside the hypergraph")
    return errors


def corpus_criteria(rows: dict[str, dict]) -> list[Criterion]:
    """``rows`` maps split name to payload hash, size, manifest hash, and document ids."""
    criteria = []
    hashes = [rows[split]["payload_sha256"] for split in ("train", "valid", "test")]
    if len(set(hashes)) == 3:
        criteria.append(ok("corpus.distinct", "train, valid, and test payload hashes differ"))
    else:
        criteria.append(bad("corpus.distinct", "train, valid, and test payload hashes are not all different"))
    overlap = False
    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        shared = rows[left]["document_ids"] & rows[right]["document_ids"]
        if shared:
            overlap = True
            criteria.append(bad(f"corpus.overlap.{left}_{right}", "shared document ids: " + ", ".join(sorted(shared))))
    if not overlap:
        criteria.append(ok("corpus.overlap", "train, valid, and test document ids are pairwise disjoint"))
    for split in ("train", "valid", "test"):
        size = rows[split]["size_bytes"]
        if isinstance(size, int) and not isinstance(size, bool) and size > 0:
            criteria.append(ok(f"corpus.{split}.bytes", f"{size} bytes"))
        else:
            criteria.append(bad(f"corpus.{split}.bytes", "split is empty"))
    orders = [rows[split].get("order") for split in ("train", "valid", "test")]
    if all(isinstance(order, int) and not isinstance(order, bool) for order in orders) and orders[0] < orders[1] < orders[2]:
        criteria.append(ok("corpus.chronological", "train, valid, and test orders are strictly increasing"))
    else:
        criteria.append(bad("corpus.chronological", "split orders must be integers with train < valid < test"))
    return criteria


def leakage_criteria(
    document: dict,
    *,
    train_ids: set[str],
    valid_ids: set[str],
    test_ids: set[str],
    inputs: list[str],
    train_manifest: str,
    valid_manifest: str,
    test_manifest: str,
    label: str,
) -> list[Criterion]:
    criteria = []
    errors = hypergraph_errors(document)
    if errors:
        return [bad(f"{label}.structure", "; ".join(errors))]
    criteria.append(ok(f"{label}.structure", "vertices and hyperedges are internally consistent"))
    if document.get("built_from_split") != "train":
        criteria.append(bad(f"{label}.split", "built_from_split is not train"))
    else:
        criteria.append(ok(f"{label}.split", "built_from_split is train"))
    docs = set(document["document_ids"])
    if docs <= train_ids and not (docs & valid_ids) and not (docs & test_ids):
        criteria.append(ok(f"{label}.documents", "hypergraph documents are a subset of the train split"))
    else:
        criteria.append(bad(f"{label}.documents", "hypergraph documents are not confined to the train split"))
    input_set = set(inputs)
    if train_manifest in input_set and valid_manifest not in input_set and test_manifest not in input_set:
        criteria.append(ok(f"{label}.inputs", "provenance inputs include the train corpus and exclude valid and test"))
    else:
        criteria.append(bad(f"{label}.inputs", "provenance inputs do not show a train-only build"))
    return criteria


def vocab_criteria(document: dict, metrics: dict, inputs: list[str], train_manifest: str, held_out: Iterable[str]) -> list[Criterion]:
    criteria = []
    if not isinstance(document, dict) or document.get("format") != VOCAB_FORMAT:
        return [bad("vocabulary.format", "vocabulary format is not sheaf-verify.vocabulary.v1")]
    units = document.get("units")
    if not isinstance(units, list) or not units or any(not isinstance(item, str) or not item for item in units):
        return [bad("vocabulary.units", "units must be a non-empty list of strings")]
    if len(set(units)) != len(units):
        criteria.append(bad("vocabulary.units", "unit strings are not unique"))
    else:
        criteria.append(ok("vocabulary.units", f"{len(units)} unique units"))
    arms = _arms(metrics)
    if arms is None:
        criteria.append(bad("vocabulary.arms", "metrics arms are not the required trio"))
        return criteria
    mismatched = [name for name, arm in arms.items() if arm.get("vocab_size") != len(units)]
    if mismatched:
        criteria.append(bad("vocabulary.equal_size", "vocab size differs from the unit inventory for " + ", ".join(mismatched)))
    else:
        criteria.append(ok("vocabulary.equal_size", f"every arm uses vocab size {len(units)}"))
    if "superword_bpe" not in arms or arms["superword_bpe"].get("kind") != "superword_bpe":
        criteria.append(bad("vocabulary.superword_bpe", "equal-size superword-BPE arm is missing"))
    else:
        criteria.append(ok("vocabulary.superword_bpe", "superword-BPE control is present at the unit inventory size"))
    held = set(held_out)
    if train_manifest in inputs and not (set(inputs) & held):
        criteria.append(ok("vocabulary.inputs", "vocabulary provenance cites the train corpus only"))
    else:
        criteria.append(bad("vocabulary.inputs", "vocabulary provenance is not train-only"))
    return criteria


def flop_and_byte_criteria(metrics: dict, train_size: int) -> list[Criterion]:
    criteria = []
    arms = _arms(metrics)
    if arms is None:
        return [bad("flops.arms", "metrics arms are not the required trio")]
    declared = metrics.get("flop_relative_tolerance", MAX_FLOP_RELATIVE_TOLERANCE)
    if isinstance(declared, bool) or not isinstance(declared, (int, float)) or not 0 <= float(declared) <= MAX_FLOP_RELATIVE_TOLERANCE:
        criteria.append(bad("flops.tolerance", "flop tolerance must be between 0 and the harness cap 0.05"))
        tolerance = MAX_FLOP_RELATIVE_TOLERANCE
    else:
        tolerance = float(declared)
        criteria.append(ok("flops.tolerance", f"relative tolerance {fmt_float(tolerance)}"))
    totals = []
    bytes_ok = True
    for name, arm in arms.items():
        flops = arm.get("flops")
        if not isinstance(flops, dict):
            criteria.append(bad(f"flops.identity.{name}", "flops object is missing"))
            continue
        try:
            train = as_int(flops.get("train"), "train flops")
            graph = as_int(flops.get("graph_build"), "graph_build flops")
            total = as_int(flops.get("total"), "total flops")
        except SheafError as error:
            criteria.append(bad(f"flops.identity.{name}", str(error)))
            continue
        if train < 0 or graph < 0 or total < 0 or train + graph != total:
            criteria.append(bad(f"flops.identity.{name}", "total must equal train plus graph_build, all non-negative"))
        else:
            criteria.append(ok(f"flops.identity.{name}", f"total {total} includes graph_build {graph}"))
        kind = arm.get("kind")
        if kind in ("hypergraph", "randomized_hypergraph"):
            if graph <= 0:
                criteria.append(bad(f"flops.graph.{name}", "graph-build cost is not included"))
            else:
                criteria.append(ok(f"flops.graph.{name}", "graph-build cost is a positive part of the total"))
        elif kind == "superword_bpe":
            if graph != 0:
                criteria.append(bad(f"flops.graph.{name}", "superword-BPE control must record graph_build as 0"))
            else:
                criteria.append(ok(f"flops.graph.{name}", "superword-BPE graph_build is explicitly 0"))
        totals.append(total)
        if arm.get("train_bytes") != train_size:
            bytes_ok = False
            criteria.append(bad(f"bytes.{name}", f"train_bytes {arm.get('train_bytes')} does not equal the train corpus size {train_size}"))
    if bytes_ok and arms:
        criteria.append(ok("bytes.match", f"every arm records train_bytes {train_size}"))
    if len(totals) == len(arms) and totals:
        span = max(totals) - min(totals)
        width = 0 if max(totals) == 0 else span / max(totals)
        if width <= tolerance:
            criteria.append(ok("flops.match", f"relative span {fmt_float(width)}"))
        else:
            criteria.append(bad("flops.match", f"relative span {fmt_float(width)} exceeds {fmt_float(tolerance)}"))
    return criteria


def causality_criteria(trace: dict, hypergraph: dict) -> list[Criterion]:
    errors = hypergraph_errors(hypergraph)
    if errors:
        return [bad("causality.graph", "; ".join(errors))]
    if not isinstance(trace, dict) or trace.get("format") != TRACE_FORMAT:
        return [bad("causality.format", "trace format is not sheaf-verify.causality.v1")]
    clean = trace.get("clean_outputs")
    perturbed = trace.get("perturbed_outputs")
    positions = trace.get("perturbation_positions")
    consulted = trace.get("edges_consulted")
    horizon = trace.get("horizon")
    if not isinstance(clean, list) or not isinstance(perturbed, list) or len(clean) != len(perturbed) or len(clean) < 2:
        return [bad("causality.length", "clean and perturbed outputs must be equal-length lists of size at least 2")]
    if isinstance(horizon, bool) or not isinstance(horizon, int) or not 0 <= horizon < len(clean) - 1:
        return [bad("causality.horizon", "horizon must leave at least one future position")]
    if not isinstance(positions, list) or not positions or any(isinstance(item, bool) or not isinstance(item, int) for item in positions):
        return [bad("causality.positions", "perturbation_positions must be a non-empty list of indexes")]
    criteria = []
    if any(item <= horizon or item >= len(clean) for item in positions):
        criteria.append(bad("causality.future", "a perturbation index is not strictly after the horizon"))
    else:
        criteria.append(ok("causality.future", "perturbations are confined to positions after the horizon"))
    past_ok = all(clean[index] == perturbed[index] for index in range(horizon + 1))
    if past_ok:
        criteria.append(ok("causality.past", "outputs at and before the horizon are unchanged"))
    else:
        criteria.append(bad("causality.past", "perturbing the future changed an output at or before the horizon"))
    probed = True
    for index in positions:
        if 0 <= index < len(clean) and clean[index] == perturbed[index]:
            probed = False
    if probed and positions:
        criteria.append(ok("causality.probe", "every perturbed future position changed its output"))
    else:
        criteria.append(bad("causality.probe", "the future perturbation did not change the probed outputs"))
    edges = {edge["id"]: edge for edge in hypergraph["hyperedges"]}
    if not isinstance(consulted, list) or len(consulted) != len(clean):
        criteria.append(bad("causality.edges", "edges_consulted must have one entry per output position"))
        return criteria
    structural = True
    seen = set()
    for entry in consulted:
        if not isinstance(entry, dict):
            structural = False
            break
        predict_at = entry.get("predict_at")
        edge_ids = entry.get("edge_ids")
        if isinstance(predict_at, bool) or not isinstance(predict_at, int) or predict_at in seen or not 0 <= predict_at < len(clean):
            structural = False
            break
        seen.add(predict_at)
        if not isinstance(edge_ids, list):
            structural = False
            break
        for edge_id in edge_ids:
            edge = edges.get(edge_id)
            if edge is None or max(edge["positions"]) > predict_at:
                structural = False
                break
    if structural and seen == set(range(len(clean))):
        criteria.append(ok("causality.edges", "every consulted hyperedge ends at or before the position that uses it"))
    else:
        criteria.append(bad("causality.edges", "a consulted hyperedge touches a position after the prediction, or the trace is incomplete"))
    return criteria


def random_control_criteria(real: dict, random_graph: dict, metrics: dict) -> list[Criterion]:
    real_errors = hypergraph_errors(real)
    random_errors = hypergraph_errors(random_graph)
    if real_errors or random_errors:
        return [bad("random.structure", "; ".join(real_errors + random_errors))]
    criteria = []
    if real.get("randomized") is False and random_graph.get("randomized") is True:
        criteria.append(ok("random.flag", "the control graph is marked randomized and the study graph is not"))
    else:
        criteria.append(bad("random.flag", "randomized flags are not set on exactly the control graph"))
    real_hash = incidence_sha256(real)
    random_hash = incidence_sha256(random_graph)
    if real_hash != random_hash:
        criteria.append(ok("random.incidence", "incidence hashes differ"))
    else:
        criteria.append(bad("random.incidence", "randomized hypergraph has the same incidence hash as the study graph"))
    if len(real["vertices"]) == len(random_graph["vertices"]) and len(real["hyperedges"]) == len(random_graph["hyperedges"]):
        criteria.append(ok("random.order", "vertex count and hyperedge count match"))
    else:
        criteria.append(bad("random.order", "vertex count or hyperedge count differs"))
    if cardinality_histogram(real) == cardinality_histogram(random_graph):
        criteria.append(ok("random.histogram", "hyperedge cardinality histograms match"))
    else:
        criteria.append(bad("random.histogram", "hyperedge cardinality histograms differ"))
    arms = _arms(metrics)
    arm = None if arms is None else arms.get("randomized_hypergraph")
    if not isinstance(arm, dict) or arm.get("kind") != "randomized_hypergraph":
        criteria.append(bad("random.arm", "metrics have no randomized_hypergraph arm"))
        return criteria
    seed = arm.get("randomization_seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or arm.get("randomization_seed") != random_graph.get("randomization_seed"):
        criteria.append(bad("random.seed", "randomization seed is missing or disagrees with the control artifact"))
    else:
        criteria.append(ok("random.seed", f"randomization seed {seed}"))
    if arm.get("incidence_sha256") != random_hash:
        criteria.append(bad("random.metrics_hash", "metrics incidence hash does not match the control artifact"))
    else:
        criteria.append(ok("random.metrics_hash", "metrics cite the control artifact's incidence hash"))
    if arm.get("vertices") != len(random_graph["vertices"]) or arm.get("hyperedge_count") != len(random_graph["hyperedges"]):
        criteria.append(bad("random.metrics_order", "metrics order does not match the control artifact"))
    else:
        criteria.append(ok("random.metrics_order", "metrics order matches the control artifact"))
    if arm.get("edge_cardinality_histogram") != cardinality_histogram(random_graph):
        criteria.append(bad("random.metrics_histogram", "metrics histogram does not match the control artifact"))
    else:
        criteria.append(ok("random.metrics_histogram", "metrics histogram matches the control artifact"))
    return criteria


def seed_criteria(seeds: dict, metrics: dict) -> list[Criterion]:
    if not isinstance(seeds, dict) or seeds.get("format") != SEEDS_FORMAT:
        return [bad("seeds.format", "seeds format is not sheaf-verify.seeds.v1")]
    design = seeds.get("design")
    arms = seeds.get("arms")
    if design not in ("independent", "paired") or not isinstance(arms, dict):
        return [bad("seeds.design", "design must be independent or paired, with an arms object")]
    criteria = []
    if metrics.get("seed_design") != design:
        criteria.append(bad("seeds.design_match", "metrics seed_design does not match the seeds artifact"))
    else:
        criteria.append(ok("seeds.design_match", f"seed design is {design}"))
    if set(arms) != set(REQUIRED_ARMS):
        criteria.append(bad("seeds.arms", "seed arms are not exactly the required trio"))
        return criteria
    lists = {}
    count_ok = True
    for name in REQUIRED_ARMS:
        values = arms[name]
        if not isinstance(values, list) or any(isinstance(item, bool) or not isinstance(item, int) for item in values):
            criteria.append(bad(f"seeds.values.{name}", "seeds must be integers"))
            count_ok = False
            continue
        if len(values) != len(set(values)):
            criteria.append(bad(f"seeds.unique.{name}", "seeds repeat inside the arm"))
        else:
            criteria.append(ok(f"seeds.unique.{name}", f"{len(values)} distinct seeds"))
        if len(values) < MIN_SEEDS:
            count_ok = False
            criteria.append(bad(f"seeds.count.{name}", f"{len(values)} seeds is below the minimum {MIN_SEEDS}"))
        else:
            criteria.append(ok(f"seeds.count.{name}", f"{len(values)} seeds"))
        lists[name] = values
        metric_arm = (_arms(metrics) or {}).get(name, {})
        per_seed = metric_arm.get("bpb_per_seed")
        if not isinstance(per_seed, list) or len(per_seed) != len(values):
            criteria.append(bad(f"seeds.length.{name}", "bpb_per_seed length does not match the seed list"))
    if design == "independent" and len(lists) == 3:
        shared = set(lists[PRIMARY_ARM]) & set(lists["superword_bpe"]) | set(lists[PRIMARY_ARM]) & set(lists["randomized_hypergraph"]) | set(lists["superword_bpe"]) & set(lists["randomized_hypergraph"])
        if shared:
            criteria.append(bad("seeds.independence", "independent arms share seed values"))
        else:
            criteria.append(ok("seeds.independence", "seed sets are pairwise disjoint"))
    if design == "paired" and len(lists) == 3:
        if lists[PRIMARY_ARM] == lists["superword_bpe"] == lists["randomized_hypergraph"]:
            criteria.append(ok("seeds.paired", "paired arms use the same seed list"))
        else:
            criteria.append(bad("seeds.paired", "paired arms do not use the same seed list"))
    if count_ok and len(lists) == 3:
        criteria.append(ok("seeds.minimum", f"every arm has at least {MIN_SEEDS} seeds"))
    return criteria


def environment_criteria(document: dict) -> list[Criterion]:
    if not isinstance(document, dict) or document.get("format") != ENV_FORMAT:
        return [bad("environment.format", "environment format is not sheaf-verify.environment.v1")]
    criteria = []
    for field in ("python", "cuda", "driver", "gpu", "platform"):
        value = document.get(field)
        if isinstance(value, str) and value.strip() and value.strip().casefold() not in BLANK:
            criteria.append(ok(f"environment.{field}", value.strip()))
        else:
            criteria.append(bad(f"environment.{field}", f"{field} is missing or unknown"))
    return criteria


def bpb_criteria(metrics: dict, test_size: int) -> list[Criterion]:
    arms = _arms(metrics)
    if arms is None:
        return [bad("bpb.arms", "metrics arms are not the required trio")]
    criteria = []
    scored = []
    for name, arm in arms.items():
        evaluation = arm.get("eval")
        if not isinstance(evaluation, dict):
            criteria.append(bad(f"bpb.eval.{name}", "eval object is missing"))
            continue
        if evaluation.get("denominator") != "raw_corpus_bytes":
            criteria.append(bad(f"bpb.denominator.{name}", "denominator is not raw_corpus_bytes"))
        else:
            criteria.append(ok(f"bpb.denominator.{name}", "denominator is raw_corpus_bytes"))
        try:
            raw = as_int(evaluation.get("raw_bytes"), "raw_bytes")
            scored_bytes = as_int(evaluation.get("scored_bytes"), "scored_bytes")
            total_bits = as_real(evaluation.get("total_bits"), "total_bits")
            per_seed = [as_real(item, "bpb") for item in arm["bpb_per_seed"]]
        except (SheafError, KeyError, TypeError) as error:
            criteria.append(bad(f"bpb.values.{name}", str(error)))
            continue
        if raw != test_size or scored_bytes != raw or raw <= 0:
            criteria.append(bad(f"bpb.bytes.{name}", "raw_bytes and scored_bytes must both equal the test corpus size"))
        else:
            criteria.append(ok(f"bpb.bytes.{name}", f"scored {scored_bytes} raw test bytes"))
        headline = total_bits / raw if raw else math.nan
        center = sum(per_seed) / len(per_seed)
        if math.isfinite(headline) and abs(center - headline) <= 1e-9:
            criteria.append(ok(f"bpb.identity.{name}", f"mean seed BPB {fmt_float(center)} equals total_bits/raw_bytes"))
        else:
            criteria.append(bad(f"bpb.identity.{name}", "mean seed BPB does not equal total_bits/raw_bytes"))
        scored.append(scored_bytes)
    if len(scored) == len(arms) and len(set(scored)) == 1:
        criteria.append(ok("bpb.scored_equal", f"scored-byte count {scored[0]} is identical across arms"))
    elif scored:
        criteria.append(bad("bpb.scored_equal", "scored-byte counts differ across arms"))
    return criteria


def stats_criteria(metrics: dict, seeds: dict, claim=None) -> list[Criterion]:
    arms = _arms(metrics)
    if arms is None:
        return [bad("stats.arms", "metrics arms are not the required trio")]
    try:
        draws = as_int(metrics.get("bootstrap_draws"), "bootstrap_draws")
        seed = as_int(metrics.get("bootstrap_seed"), "bootstrap_seed")
    except SheafError as error:
        return [bad("stats.bootstrap_settings", str(error))]
    criteria = []
    if MIN_BOOTSTRAP_DRAWS <= draws <= MAX_BOOTSTRAP_DRAWS:
        criteria.append(ok("stats.bootstrap_settings", f"{draws} draws, seed {seed}"))
    else:
        criteria.append(bad("stats.bootstrap_settings", "bootstrap_draws is outside the allowed range"))
        return criteria
    design = seeds.get("design") if isinstance(seeds, dict) else None
    paired = design == "paired"
    if design not in ("independent", "paired"):
        return criteria + [bad("stats.design", "seed design is not available")]
    computed = {}
    for control in CONTROL_ARMS:
        try:
            primary = [as_real(item, "bpb") for item in arms[PRIMARY_ARM]["bpb_per_seed"]]
            other = [as_real(item, "bpb") for item in arms[control]["bpb_per_seed"]]
            test = paired_t(primary, other) if paired else welch(primary, other)
            low, high = bootstrap_mean_difference(primary, other, seed=seed, draws=draws, paired=paired)
            effect = hedges_g(primary, other)
        except (SheafError, KeyError, TypeError) as error:
            criteria.append(bad(f"stats.{control}", str(error)))
            continue
        computed[control] = {"t": test.t, "df": test.df, "p": test.p, "low": low, "high": high, "g": effect}
        criteria.append(ok(
            f"stats.{control}",
            "t=" + fmt_float(test.t)
            + " df=" + fmt_float(test.df)
            + " p=" + fmt_float(test.p)
            + " low=" + fmt_float(low)
            + " high=" + fmt_float(high)
            + " g=" + fmt_float(effect),
        ))
    reported = metrics.get("reported", None)
    if reported is None:
        criteria.append(ok("stats.reported", "no trainer-supplied p-values; the ledger values are the harness computation"))
    else:
        criteria.extend(_reported(reported, computed))
    criteria.extend(_superiority(metrics.get("superiority"), computed))
    criteria.append(_control_agreement(metrics.get("superiority"), computed))
    if isinstance(claim, dict) and claim.get("format") == CLAIM_FORMAT:
        against = None
        if isinstance(metrics.get("superiority"), dict):
            against = metrics["superiority"].get("against")
        if against == claim.get("primary_contrast"):
            criteria.append(ok("stats.claim_contrast", "metrics name the preregistered contrast"))
        else:
            criteria.append(bad("stats.claim_contrast", "metrics superiority.against does not match the preregistered contrast"))
    return criteria


def code_tree_criteria(manifest: dict, listing, files: list[tuple[str, str, bytes]]) -> list[Criterion]:
    metadata = manifest.get("metadata")
    if not isinstance(metadata, dict):
        return [bad("code.metadata", "code tree metadata is missing")]
    if not isinstance(listing, list) or not listing:
        return [bad("code.listing", "code listing is empty")]
    criteria = []
    digest = manifest.get("payload_sha256")
    if metadata.get("content_sha256") == digest and metadata.get("file_count") == len(listing):
        criteria.append(ok("code.identity", f"{len(listing)} files, content {digest}"))
    else:
        criteria.append(bad("code.identity", "content hash or file count does not match the listing"))
    rebuilt = []
    listing_ok = True
    for entry in listing:
        if not isinstance(entry, dict):
            listing_ok = False
            break
        path = entry.get("path")
        mode = entry.get("mode")
        recorded = entry.get("sha256")
        size = entry.get("size")
        match = next(((rel, file_mode, data) for rel, file_mode, data in files if rel == path), None)
        if match is None or match[1] != mode or sha256_hex(match[2]) != recorded or len(match[2]) != size:
            listing_ok = False
            break
        rebuilt.append(match)
    if listing_ok and len(rebuilt) == len(files):
        criteria.append(ok("code.bytes", "snapshotted file bytes match the listing"))
    else:
        criteria.append(bad("code.bytes", "snapshotted file bytes do not match the listing"))
        return criteria
    try:
        tree = git_tree_sha1(rebuilt)
    except SheafError as error:
        criteria.append(bad("code.git_tree", str(error)))
        return criteria
    recorded_tree = metadata.get("git_tree_sha1")
    if isinstance(recorded_tree, str) and GIT_SHA1_RE.fullmatch(recorded_tree) and recorded_tree == tree:
        criteria.append(ok("code.git_tree", recorded_tree))
    else:
        criteria.append(bad("code.git_tree", "recomputed git tree hash does not match the manifest"))
    return criteria


def _arms(metrics: dict) -> dict | None:
    if not isinstance(metrics, dict) or metrics.get("format") != METRICS_FORMAT:
        return None
    arms = metrics.get("arms")
    if not isinstance(arms, dict) or set(arms) != set(REQUIRED_ARMS):
        return None
    for name in REQUIRED_ARMS:
        arm = arms[name]
        if not isinstance(arm, dict) or arm.get("kind") != name:
            return None
    return arms


def _reported(reported, computed: dict) -> list[Criterion]:
    if not isinstance(reported, dict):
        return [bad("stats.reported", "reported block is not an object")]
    expected = {f"{PRIMARY_ARM}_vs_{control}" for control in CONTROL_ARMS}
    if set(reported) != expected:
        return [bad("stats.reported", "reported comparisons are not exactly the two control contrasts")]
    criteria = []
    fields = ("t", "df", "p", "low", "high", "g")
    for control in CONTROL_ARMS:
        key = f"{PRIMARY_ARM}_vs_{control}"
        block = reported[key]
        actual = computed.get(control)
        if actual is None:
            criteria.append(bad(f"stats.reported.{key}", "comparison was not computed"))
            continue
        if not isinstance(block, dict) or set(block) != set(fields):
            criteria.append(bad(f"stats.reported.{key}", "reported object fields are not t, df, p, low, high, g"))
            continue
        mismatches = []
        for field in fields:
            try:
                value = as_real(block[field], field)
            except SheafError:
                mismatches.append(field)
                continue
            if not _close(value, actual[{"t": "t", "df": "df", "p": "p", "low": "low", "high": "high", "g": "g"}[field]]):
                mismatches.append(field)
        if mismatches:
            criteria.append(bad(f"stats.reported.{key}", "disagrees on " + ", ".join(mismatches)))
        else:
            criteria.append(ok(f"stats.reported.{key}", "trainer-reported statistics match the recomputation"))
    return criteria


def _superiority(claim, computed: dict) -> list[Criterion]:
    if not isinstance(claim, dict):
        return [bad("stats.superiority", "superiority object is missing")]
    claimed = claim.get("claimed")
    arm = claim.get("arm")
    against = claim.get("against")
    metric = claim.get("metric")
    if claimed is False:
        return [ok("stats.superiority", "no superiority claim")]
    if claimed is not True or arm != PRIMARY_ARM or against not in CONTROL_ARMS or metric != "bpb":
        return [bad("stats.superiority", "superiority claim must name hypergraph against one control on bpb")]
    stats_row = computed.get(against)
    if stats_row is None:
        return [bad("stats.superiority", "the claimed contrast was not computed")]
    supported = stats_row["p"] < ALPHA and stats_row["high"] < 0 and stats_row["g"] < 0
    if supported:
        return [ok("stats.superiority", f"claim against {against} has p below {ALPHA}, a negative interval, and a negative effect")]
    return [bad(
        "stats.superiority",
        "claim is not supported: p=" + fmt_float(stats_row["p"])
        + " high=" + fmt_float(stats_row["high"])
        + " g=" + fmt_float(stats_row["g"]),
    )]


def _supports(row: dict) -> bool:
    return row["p"] < ALPHA and row["high"] < 0 and row["g"] < 0


def _control_agreement(claim, computed: dict) -> Criterion:
    if any(control not in computed for control in CONTROL_ARMS):
        return bad("stats.control_agreement", "both control contrasts were not computed")
    flags = [_supports(computed[control]) for control in CONTROL_ARMS]
    claimed = isinstance(claim, dict) and claim.get("claimed") is True
    if len(set(flags)) > 1 and claimed:
        return bad("stats.control_agreement", "superiority is claimed while the two controls disagree")
    if len(set(flags)) > 1:
        return ok("stats.control_agreement", "controls disagree and no superiority is claimed")
    return ok("stats.control_agreement", "the two control contrasts agree")


def claim_criteria(payload: bytes) -> list[Criterion]:
    if b"\x00" in payload:
        return [bad("claim.text", "claim contains NUL")]
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        return [bad("claim.text", "claim is not UTF-8")]
    if SHA256_RE.search(text):
        return [bad("claim.frozen", "claim contains a 64-character hex digest")]
    try:
        document = json.loads(text)
    except json.JSONDecodeError:
        document = None
    if isinstance(document, dict) and document.get("format") == CLAIM_FORMAT:
        return _structured_claim(document)
    if len(text.strip()) < 20:
        return [bad("claim.text", "claim text is shorter than 20 characters")]
    return [ok("claim.text", "plain-text claim is frozen as payload bytes")]


def _structured_claim(document: dict) -> list[Criterion]:
    criteria = []
    hypothesis = document.get("hypothesis_id")
    description = document.get("description")
    if isinstance(hypothesis, str) and hypothesis.strip() and isinstance(description, str) and len(description.strip()) >= 20:
        criteria.append(ok("claim.identity", hypothesis.strip()))
    else:
        criteria.append(bad("claim.identity", "hypothesis_id and a description of at least 20 characters are required"))
    if document.get("primary_metric") == "bpb" and document.get("direction") == "less":
        criteria.append(ok("claim.direction", "lower bits per raw byte is the preregistered direction"))
    else:
        criteria.append(bad("claim.direction", "preregistered metric must be bpb in the less direction"))
    alpha = document.get("alpha")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or abs(float(alpha) - ALPHA) > 1e-12:
        criteria.append(bad("claim.alpha", "claim alpha must equal the harness alpha"))
    else:
        criteria.append(ok("claim.alpha", "claim alpha matches the harness"))
    if document.get("primary_contrast") not in CONTROL_ARMS:
        criteria.append(bad("claim.contrast", "primary contrast must be one of the control arms"))
    else:
        criteria.append(ok("claim.contrast", str(document["primary_contrast"])))
    return criteria


def kernel_criteria(metadata, payload_sha: str, payload_size: int, input_payloads: set[str]) -> list[Criterion]:
    if not isinstance(metadata, dict):
        return [bad("kernel.metadata", "kernel metadata is missing")]
    backend = metadata.get("backend")
    entry = metadata.get("entry")
    criteria = []
    if backend not in KERNEL_BACKENDS:
        criteria.append(bad("kernel.backend", "backend must be cpu, cuda, opencl, or c"))
    else:
        criteria.append(ok("kernel.backend", str(backend)))
    if isinstance(entry, str) and entry.strip():
        criteria.append(ok("kernel.entry", entry.strip()))
    else:
        criteria.append(bad("kernel.entry", "kernel entry name is missing"))
    if isinstance(payload_size, bool) or not isinstance(payload_size, int) or payload_size <= 0:
        criteria.append(bad("kernel.bytes", "kernel payload is empty"))
    else:
        criteria.append(ok("kernel.bytes", f"{payload_size} bytes"))
    if backend in {"cuda", "opencl", "c"}:
        reference = metadata.get("host_reference_sha256")
        if not is_sha256(reference) or reference == payload_sha:
            criteria.append(bad("kernel.host_reference", "device kernel must name a distinct host-reference payload hash"))
        elif reference not in input_payloads:
            criteria.append(bad("kernel.host_reference", "host-reference hash is not the payload of an input manifest"))
        else:
            criteria.append(ok("kernel.host_reference", str(reference)))
    else:
        criteria.append(ok("kernel.host_reference", "cpu backend is its own reference"))
    return criteria


def boundary_criterion(complex_doc, label: str) -> Criterion:
    identifier = f"{label}.boundary"
    if complex_doc is None:
        return ok(identifier, "no 2-complex attached")
    passed, detail = _boundary_identity(complex_doc)
    return ok(identifier, detail) if passed else bad(identifier, detail)


def _boundary_identity(complex_doc) -> tuple[bool, str]:
    if not isinstance(complex_doc, dict):
        return False, "complex is not an object"
    try:
        count = as_int(complex_doc.get("num_vertices"), "num_vertices")
    except SheafError as error:
        return False, str(error)
    edges = complex_doc.get("edges")
    faces = complex_doc.get("faces")
    if count < 0 or not isinstance(edges, list) or not isinstance(faces, list):
        return False, "complex fields are missing"
    parsed = []
    lookup: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for index, edge in enumerate(edges):
        if not isinstance(edge, list) or len(edge) != 2:
            return False, f"edge {index} is not a pair"
        try:
            left = as_int(edge[0], "edge endpoint")
            right = as_int(edge[1], "edge endpoint")
        except SheafError as error:
            return False, str(error)
        if left == right or not (0 <= left < count and 0 <= right < count):
            return False, f"edge {index} is outside the vertex range"
        parsed.append((left, right))
        lookup.setdefault((left, right), []).append((index, 1))
        lookup.setdefault((right, left), []).append((index, -1))
    if not faces:
        return True, "max |B1 B2| = 0"
    residual = 0
    for face_index, face in enumerate(faces):
        if not isinstance(face, list) or len(face) < 3:
            return False, f"face {face_index} has fewer than 3 vertices"
        try:
            vertices = [as_int(vertex, "face vertex") for vertex in face]
        except SheafError as error:
            return False, str(error)
        if any(not 0 <= vertex < count for vertex in vertices):
            return False, f"face {face_index} leaves the vertex range"
        steps = []
        for index, start in enumerate(vertices):
            nxt = vertices[(index + 1) % len(vertices)]
            matches = lookup.get((start, nxt))
            if not matches:
                return False, f"face {face_index} is missing edge {start}->{nxt}"
            if len(matches) != 1:
                return False, f"face {face_index} has an ambiguous edge"
            steps.append(matches[0])
        accumulated: dict[int, int] = {}
        for edge_index, sign in steps:
            left, right = parsed[edge_index]
            accumulated[left] = accumulated.get(left, 0) - sign
            accumulated[right] = accumulated.get(right, 0) + sign
        face_residual = max((abs(value) for value in accumulated.values()), default=0)
        residual = max(residual, face_residual)
    if residual != 0:
        return False, f"max |B1 B2| = {residual}"
    return True, "max |B1 B2| = 0"


def loop_criterion(sheaf) -> Criterion:
    if sheaf is None:
        return ok("sheaf.loop_closure", "no sheaf operations attached")
    passed, detail = _loop_closure(sheaf)
    return ok("sheaf.loop_closure", detail) if passed else bad("sheaf.loop_closure", detail)


def _loop_closure(sheaf) -> tuple[bool, str]:
    if not isinstance(sheaf, dict):
        return False, "sheaf is not an object"
    states = sheaf.get("states")
    operations = sheaf.get("operations")
    loops = sheaf.get("loops")
    if not isinstance(states, list) or not states or any(not isinstance(item, str) or not item for item in states):
        return False, "states must be a non-empty list of strings"
    if not isinstance(operations, list) or not operations:
        return False, "operations must be a non-empty list"
    state_set = set(states)
    edges = {}
    for operation in operations:
        if not isinstance(operation, dict):
            return False, "an operation is not an object"
        ident = operation.get("id")
        source = operation.get("source")
        target = operation.get("target")
        morphism = operation.get("morphism")
        if not isinstance(ident, str) or not ident or ident in edges:
            return False, "operation ids must be unique strings"
        if source not in state_set or target not in state_set:
            return False, f"operation {ident} leaves the state set"
        if not isinstance(morphism, dict) or any(
            not isinstance(key, str) or not isinstance(value, str) for key, value in morphism.items()
        ):
            return False, f"operation {ident} morphism is not a string map"
        edges[ident] = morphism
    if not isinstance(loops, list) or not loops:
        return False, "loops must be a non-empty list when a sheaf is attached"
    for loop in loops:
        if not isinstance(loop, dict):
            return False, "a loop is not an object"
        left = _compose(edges, loop.get("path_a"))
        right = _compose(edges, loop.get("path_b"))
        if left is None or right is None:
            return False, "a loop path names an unknown operation"
        keys = set(left) | set(right)
        if any(left.get(key) != right.get(key) for key in keys):
            return False, "converging paths do not compose to the same morphism"
    return True, f"{len(loops)} loop closures hold"


def _compose(edges: dict, path) -> dict | None:
    if not isinstance(path, list) or not path or any(item not in edges for item in path):
        return None
    current = dict(edges[path[0]])
    for ident in path[1:]:
        nxt = edges[ident]
        current = {key: nxt.get(value, value) for key, value in current.items()}
    return current


def _close(left: float, right: float) -> bool:
    if math.isinf(left) or math.isinf(right):
        return left == right
    scale = max(abs(left), abs(right), 1.0)
    return abs(left - right) <= 1e-8 * scale
