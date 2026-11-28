"""Reproducible engineering pilot; never a trained-model capability benchmark."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import random
import statistics
import time
import tracemalloc
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from .baselines import CachedDAG, Reference, semantic_result
from .cases import benchmark_workloads, generate_suite, intervention_pairs
from .core import Kernel

ROOT = Path(__file__).resolve().parent.parent
ARMS = {"reference": Reference, "cached_dag": CachedDAG, "hypergraph_kernel": Kernel}


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def code_identity() -> dict:
    paths = sorted((ROOT / "hadukenn_exec").glob("*.py"))
    paths += sorted((ROOT / "tests").glob("*.py"))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")


def run_case(factory, case: dict) -> tuple[object, list[dict]]:
    engine = factory()
    return engine, [engine.execute(command) for command in case["commands"]]


def _expected(case: dict) -> dict:
    return {entry["index"]: entry["result"] for entry in case["expected"]}


def evaluate(output_dir: Path, *, benchmark: bool = True, repeats: int = 3) -> dict:
    """Run frozen deterministic inputs, independent oracles, replay and interventions."""
    spec_path = ROOT / "spec" / "PREREGISTRATION.json"
    prereg_bytes = spec_path.read_bytes()
    # These values are fixed in the accompanying preregistration, not tuned to results.
    seeds = (11, 29, 47, 71, 101)
    depths = (4, 16, 64, 128)
    cases = generate_suite(seeds=seeds, depths=depths)
    pairs = intervention_pairs()
    manifest = {
        "format": "hadukenn.exec.pilot-manifest.v1",
        "started_utc": datetime.now(UTC).isoformat(),
        "preregistration_sha256": hashlib.sha256(prereg_bytes).hexdigest(),
        "case_manifest_sha256": digest(cases),
        "intervention_manifest_sha256": digest(pairs),
        "code_sha256": code_identity(),
        "spec_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT / "spec").glob("*")) if p.is_file()},
        "seeds": list(seeds),
        "depths": list(depths),
        "timing_repeats": repeats,
        "environment": {"python": platform.python_version(), "system": platform.system(), "machine": platform.machine()},
        "execution_location": "session_cpu_container",
        "neural_model_trained": False,
        "gpu_used": False,
        "capability_claim": None,
    }
    write_json(output_dir / "run_manifest.json", manifest)
    failures = []
    counts = {name: Counter() for name in ARMS}
    replay_count = 0
    replay_failures = []
    case_rows = []
    for case in cases:
        expected = _expected(case)
        for name, factory in ARMS.items():
            engine, outcomes = run_case(factory, case)
            correct = 0
            for index, outcome in enumerate(outcomes):
                counts[name]["commands"] += 1
                if index in expected:
                    good = semantic_result(outcome) == expected[index]
                    counts[name]["checked"] += 1
                    counts[name]["correct"] += good
                    correct += good
                    if not good:
                        failures.append({"case": case["id"], "arm": name, "index": index, "expected": expected[index], "observed": semantic_result(outcome)})
            counts[name]["cases"] += 1
            counts[name]["queries"] += sum(c["action"] == "query" for c in case["commands"])
            case_rows.append({"case": case["id"], "family": case["family"], "seed": case["seed"], "depth": case["depth"], "arm": name, "checked": len(expected), "correct": correct})
            if name == "hypergraph_kernel":
                try:
                    replayed = Kernel.replay(engine.export_trace(), expected_head=engine.head)
                    if replayed.state_digest() != engine.state_digest():
                        raise ValueError("final state differs")
                    replay_count += 1
                except (ValueError, TypeError, KeyError) as exc:
                    replay_failures.append({"case": case["id"], "error": str(exc)})
    intervention_rows = []
    for pair in pairs:
        for name, factory in ARMS.items():
            sides = []
            gold_ok = True
            for side in ["base", "changed"]:
                _, outcomes = run_case(factory, pair[side])
                expected = _expected(pair[side])
                gold_ok &= all(semantic_result(outcomes[i]) == gold for i, gold in expected.items())
                final_query = next(i for i in range(len(outcomes) - 1, -1, -1) if pair[side]["commands"][i]["action"] == "query")
                sides.append(semantic_result(outcomes[final_query]))
            relationship_ok = (sides[0] == sides[1]) if pair["relation"] == "equal" else (sides[0] != sides[1])
            intervention_rows.append({"id": pair["id"], "kind": pair["kind"], "arm": name, "relation": pair["relation"], "gold_ok": gold_ok, "relationship_ok": relationship_ok, "passed": gold_ok and relationship_ok, "base": sides[0], "changed": sides[1]})
    write_json(output_dir / "correctness_cases.json", case_rows)
    write_json(output_dir / "interventions.json", intervention_rows)
    timing_rows = []
    memory_rows = []
    if benchmark:
        for seed in seeds:
            for depth in depths:
                workloads = benchmark_workloads(seed, depth, queries=40)
                for workload in workloads:
                    for repeat in range(repeats):
                        order = list(ARMS)
                        random.Random(seed * 100000 + depth * 100 + repeat).shuffle(order)
                        for name in order:
                            cpu0, wall0 = time.process_time_ns(), time.perf_counter_ns()
                            _, outcomes = run_case(ARMS[name], workload)
                            wall_ns, cpu_ns = time.perf_counter_ns() - wall0, time.process_time_ns() - cpu0
                            matches = all(semantic_result(outcomes[i]) == gold for i, gold in _expected(workload).items())
                            timing_rows.append({"case": workload["id"], "family": workload["family"], "seed": seed, "depth": depth, "repeat": repeat, "arm": name, "commands": len(workload["commands"]), "wall_ns": wall_ns, "cpu_ns": cpu_ns, "correct": matches})
                if seed == seeds[0]:
                    # A separate instrumented pass; never mixed into latency estimates.
                    workload = workloads[0]
                    for name, factory in ARMS.items():
                        tracemalloc.start()
                        engine, outcomes = run_case(factory, workload)
                        current, peak = tracemalloc.get_traced_memory()
                        tracemalloc.stop()
                        memory_rows.append({"case": workload["id"], "depth": depth, "arm": name, "traced_current_bytes": current, "traced_peak_bytes": peak, "metric": "Python allocator, not process RSS"})
                        del engine, outcomes
    if timing_rows:
        with (output_dir / "timing_raw.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(timing_rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(timing_rows)
    write_json(output_dir / "memory_profile.json", memory_rows)
    aggregates = []
    for name in ARMS:
        for depth in depths:
            selected = [r for r in timing_rows if r["arm"] == name and r["depth"] == depth]
            # Replicates are first averaged within workload/seed. Seeds are synthetic workload replicates, not training seeds.
            blocks = {}
            for r in selected:
                blocks.setdefault((r["case"], r["seed"]), []).append(r["wall_ns"] / 1e6)
            values = [statistics.mean(v) for v in blocks.values()]
            if values:
                aggregates.append({"arm": name, "depth": depth, "workload_blocks": len(values), "mean_ms": statistics.mean(values), "se_ms": statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None, "median_ms": statistics.median(values), "unit": "complete workload including state construction and all commands"})
    passed = not failures and not replay_failures and all(row["passed"] for row in intervention_rows) and all(row["correct"] for row in timing_rows)
    summary = {
        "format": "hadukenn.exec.pilot-summary.v1",
        "status": "pass" if passed else "fail",
        "scope": "engineering correctness and descriptive CPU cost only",
        "case_count": len(cases),
        "by_arm": {name: dict(values) for name, values in counts.items()},
        "failures": failures,
        "replay": {"passed": replay_count, "failed": replay_failures},
        "interventions": {"paired_cases": len(pairs), "arm_evaluations": len(intervention_rows), "passed": sum(r["passed"] for r in intervention_rows)},
        "timing_rows": len(timing_rows),
        "timing_failures": [row for row in timing_rows if not row["correct"]],
        "timing_aggregates": aggregates,
        "memory_profiles": memory_rows,
        "manifest_sha256": digest(manifest),
        "limitations": [
            "No neural model, learned tokenizer, learned policy, or natural-language extraction is present.",
            "Synthetic programs use supplied exact operations; success does not establish learned generalization.",
            "The kernel maintains revision/provenance and full deterministic receipts; ordinary baselines do not supply equivalent audit records. Timing is diagnostic, not a matched-audit superiority comparison.",
            "Timing repetitions and synthetic seeds are not independent training runs; no capability p-value is reported.",
            "Python allocator peaks exclude native allocations and do not measure process RSS or VRAM.",
            "Trace hashes are integrity checks, not signatures or independent execution attestations.",
            "A graph runtime does not establish hypergraph tokenizer gains or elimination of neural opacity.",
        ],
        "tenfold_capability_established": False,
        "learned_stage_evidence_ready": False,
    }
    write_json(output_dir / "summary.json", summary)
    write_report(output_dir / "REPORT.md", summary, manifest)
    return summary


def write_report(path: Path, summary: dict, manifest: dict) -> None:
    lines = ["# HaDukeNN executable hypergraph — engineering pilot", "", f"**Status: {summary['status'].upper()}.** This status applies to the engineering checks below.", "", "This is an executable symbolic reference environment for the proposed redesign. It is not a trained HaDukeNN model or a demonstration of a tenfold capability gain.", "", "## Recorded scope", "", f"- Scenarios: {summary['case_count']}", f"- Exact replay: {summary['replay']['passed']} passed; {len(summary['replay']['failed'])} failed", f"- Intervention evaluations: {summary['interventions']['passed']}/{summary['interventions']['arm_evaluations']}", f"- Preregistration SHA-256: `{manifest['preregistration_sha256']}`", "- Environment: session CPU container; no GPU run or user-workstation measurement", "", "## Exact expected outcomes", "", "| Arm | Cases | Checked commands | Correct | Queries |", "| --- | ---: | ---: | ---: | ---: |"]
    for name, counts in summary["by_arm"].items():
        lines.append(f"| {name} | {counts['cases']} | {counts['checked']} | {counts['correct']} | {counts['queries']} |")
    lines += ["", "## Descriptive end-to-end workload timings", "", "Costs include engine construction and all supplied commands. Input generator time is excluded identically. Timing blocks first average the repeats for one workload/seed. SE is across workload blocks, not training runs. The kernel emits richer audit records than the ordinary baselines, so this table cannot establish an architectural speedup.", "", "| Arm | Depth | Mean ms | SE ms | Workload blocks |", "| --- | ---: | ---: | ---: | ---: |"]
    for row in summary["timing_aggregates"]:
        se = "—" if row["se_ms"] is None else f"{row['se_ms']:.4f}"
        lines.append(f"| {row['arm']} | {row['depth']} | {row['mean_ms']:.4f} | {se} | {row['workload_blocks']} |")
    lines += ["", "## Claim boundaries", ""] + [f"- {item}" for item in summary["limitations"]]
    lines += ["", "## Decision", "", "A passing run qualifies the supplied symbolic semantics and reproducibility on these inputs. It does not authorize a capability claim, a tokenizer replacement, or a large training campaign. The next scientific comparison must match audit obligations and add a learned action-selection task with held-out compositions and a strong conventional controller.", "", "The existing T-38 design and campaign ledger are outside this iteration.", ""]
    path.write_text("\n".join(lines))


def replay_file(path: Path, expected_head: str | None = None) -> dict:
    trace = json.loads(path.read_text())
    kernel = Kernel.replay(trace, expected_head=expected_head)
    return {"status": "pass", "receipts": len(trace), "state_digest": kernel.state_digest()}


def cli_exit(summary: dict) -> int:
    print(json.dumps({k: summary[k] for k in ["status", "case_count", "by_arm", "replay", "interventions", "tenfold_capability_established"]}, indent=2))
    return 0 if summary["status"] == "pass" else 1
