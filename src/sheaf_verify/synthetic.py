"""Deterministic synthetic study used by the tests and the committed fixture.

The numbers are small on purpose. They exercise every gate without training
a model or storing a weight file.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from sheaf_verify.checks import cardinality_histogram, incidence_sha256
from sheaf_verify.hashing import canonical_bytes, sha256_hex
from sheaf_verify.runner import SequenceClock, run
from sheaf_verify.spec_gates import STAGES
from sheaf_verify.store import Store

TEST_KEY = b"sheaf-verify-public-test-vector"
TRAIN = b"train-bytes-0001"
VALID = b"valid-bytes-0001"
TEST = b"test-bytes-00001"
COMPLEX = {
    "edges": [[0, 1], [1, 2], [0, 2]],
    "faces": [[0, 1, 2]],
    "num_vertices": 3,
}


def default_parts() -> dict:
    hypergraph = _hypergraph(randomized=False, edges=[
        ("e0", ["v0", "v1"], [0, 1]),
        ("e1", ["v1", "v2"], [1, 2]),
    ])
    hypergraph["sheaf"] = {
        "loops": [{"path_a": ["op0"], "path_b": ["op1"]}],
        "operations": [
            {"id": "op0", "morphism": {"a": "b"}, "source": "s0", "target": "s1"},
            {"id": "op1", "morphism": {"a": "b"}, "source": "s0", "target": "s1"},
        ],
        "states": ["s0", "s1"],
    }
    random_graph = _hypergraph(randomized=True, edges=[
        ("r0", ["v0", "v2"], [0, 2]),
        ("r1", ["v0", "v1"], [0, 1]),
    ])
    random_graph["randomization_seed"] = 9001
    samples = {
        "hypergraph": [1.75, 2.0, 2.25, 1.5, 2.5],
        "randomized_hypergraph": [2.0, 2.25, 2.5, 2.25, 2.25],
        "superword_bpe": [2.25, 2.5, 2.75, 2.5, 2.5],
    }
    return {
        "claim": {
            "alpha": 0.05,
            "description": "Synthetic study of bits per raw byte under a frozen claim.",
            "direction": "less",
            "format": "sheaf-verify.claim.v1",
            "hypothesis_id": "fixture-bpb",
            "primary_contrast": "superword_bpe",
            "primary_metric": "bpb",
        },
        "code_files": {
            "model.py": b"class Model:\n    pass\n",
            "pkg/layer.py": b"def layer(tokens):\n    return tokens\n",
        },
        "config": {"format": "sheaf-verify.config.v1", "study": "synthetic"},
        "corpora": {
            "test": (TEST, {"document_ids": ["doc-test"], "order": 2, "split": "test"}),
            "train": (TRAIN, {"document_ids": ["doc-train"], "order": 0, "split": "train"}),
            "valid": (VALID, {"document_ids": ["doc-valid"], "order": 1, "split": "valid"}),
        },
        "environment": {
            "cuda": "not-used",
            "driver": "not-used",
            "gpu": "cpu-only",
            "platform": "linux",
            "python": "3.12.3",
        },
        "figure": b'<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg"><title>synthetic</title></svg>',
        "hypergraph": hypergraph,
        "kernel": b"def apply(tokens):\n    return tokens\n",
        "kernel_meta": {"backend": "cpu", "entry": "apply"},
        "logs": {arm: f"synthetic log {arm}\n".encode("ascii") for arm in samples},
        "metrics": _metrics(samples, random_graph),
        "random": random_graph,
        "seeds": {
            "arms": {
                "hypergraph": [101, 102, 103, 104, 105],
                "randomized_hypergraph": [301, 302, 303, 304, 305],
                "superword_bpe": [201, 202, 203, 204, 205],
            },
            "design": "independent",
            "format": "sheaf-verify.seeds.v1",
        },
        "trace": {
            "clean_outputs": ["c0", "c1", "c2", "c3"],
            "edges_consulted": [
                {"edge_ids": [], "predict_at": 0},
                {"edge_ids": ["e0"], "predict_at": 1},
                {"edge_ids": ["e0", "e1"], "predict_at": 2},
                {"edge_ids": ["e0", "e1"], "predict_at": 3},
            ],
            "format": "sheaf-verify.causality.v1",
            "horizon": 2,
            "perturbation_positions": [3],
            "perturbed_outputs": ["c0", "c1", "c2", "c3x"],
        },
        "vocabulary": {
            "format": "sheaf-verify.vocabulary.v1",
            "units": ["aa", "bb", "cc", "dd"],
        },
    }


def build(root: Path, *, mutate=None, key: bytes | None = None, clock=None) -> Store:
    parts = default_parts()
    if mutate is not None:
        mutate(parts)
    return assemble(root, parts, TEST_KEY if key is None else key, SequenceClock() if clock is None else clock)


def assemble(root: Path, parts: dict, key: bytes, clock) -> Store:
    root = Path(root)
    if root.exists():
        raise FileExistsError(root)
    store = Store(root)
    store.init()
    corpora = {}
    for split in ("train", "valid", "test"):
        payload, metadata = parts["corpora"][split]
        digest, _manifest = store.register(
            kind="raw_data",
            role=f"corpus.{split}",
            registered_at=clock.now(),
            payload=payload,
            metadata=metadata,
        )
        corpora[split] = digest
    vocabulary, _ = store.register(
        kind="vocabulary",
        role="vocabulary",
        registered_at=clock.now(),
        payload=canonical_bytes(parts["vocabulary"]),
        inputs=[corpora["train"]],
    )
    hypergraph, _ = store.register(
        kind="hypergraph",
        role="hypergraph",
        registered_at=clock.now(),
        payload=canonical_bytes(parts["hypergraph"]),
        inputs=[corpora["train"]],
    )
    random_graph, _ = store.register(
        kind="hypergraph",
        role="hypergraph.random",
        registered_at=clock.now(),
        payload=canonical_bytes(parts["random"]),
        inputs=[corpora["train"]],
    )
    with tempfile.TemporaryDirectory() as temporary:
        code_root = Path(temporary) / "code"
        for relative, data in parts["code_files"].items():
            target = code_root.joinpath(*relative.split("/"))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        code, _ = store.register_tree(role="code", path=code_root, registered_at=clock.now())
    kernel_inputs = list(parts.get("kernel_inputs", []))
    if "host_reference" in parts:
        _host, host_manifest = store.register(
            kind="kernel",
            role="kernel.host",
            registered_at=clock.now(),
            payload=parts["host_reference"],
        )
        kernel_inputs.append(_host)
        parts["kernel_meta"]["host_reference_sha256"] = host_manifest["payload_sha256"]
    kernel, _ = store.register(
        kind="kernel",
        role="kernel",
        registered_at=clock.now(),
        payload=parts["kernel"],
        metadata=parts["kernel_meta"],
        inputs=kernel_inputs,
    )
    env = parts["environment"]
    environment, _ = store.register_environment(
        role="environment",
        registered_at=clock.now(),
        python=env["python"],
        cuda=env["cuda"],
        driver=env["driver"],
        gpu=env["gpu"],
        platform=env["platform"],
    )
    config, _ = store.register(
        kind="config",
        role="config",
        registered_at=clock.now(),
        payload=canonical_bytes(parts["config"]),
    )
    seeds, _ = store.register(
        kind="seeds",
        role="seeds",
        registered_at=clock.now(),
        payload=canonical_bytes(parts["seeds"]),
    )
    store.register(
        kind="claim",
        role="claim",
        registered_at=clock.now(),
        payload=canonical_bytes(parts["claim"]) if isinstance(parts["claim"], dict) else parts["claim"],
    )
    training = [code, config, seeds, kernel]
    checkpoints = {}
    for arm, payload in parts["logs"].items():
        raw = _checkpoint_bytes(arm)
        checkpoint, _ = store.register(
            kind="checkpoint",
            role=f"checkpoint.{arm}",
            registered_at=clock.now(),
            storage="external",
            payload_sha256=sha256_hex(raw),
            size_bytes=len(raw),
            locator=f"external://trainer/checkpoints/{arm}",
            inputs=list(training),
        )
        checkpoints[arm] = checkpoint
        store.register(
            kind="log",
            role=f"log.{arm}",
            registered_at=clock.now(),
            payload=payload,
            inputs=[checkpoint],
        )
    store.register(
        kind="trace",
        role="trace.causality",
        registered_at=clock.now(),
        payload=canonical_bytes(parts["trace"]),
        inputs=[hypergraph],
    )
    metrics, metrics_manifest = store.register(
        kind="metrics",
        role="metrics",
        registered_at=clock.now(),
        payload=canonical_bytes(parts["metrics"]),
        inputs=[corpora["train"], corpora["valid"], corpora["test"], seeds, vocabulary, hypergraph, random_graph],
    )
    store.register(
        kind="figure",
        role="figure",
        registered_at=clock.now(),
        payload=parts["figure"],
        inputs=[metrics],
        metadata={"metrics_payload_sha256": metrics_manifest["payload_sha256"], "plots": "bpb"},
    )
    del environment
    run(store, key=key, clock=clock)
    lock = root / "ledger" / ".lock"
    if lock.exists() and lock.stat().st_size == 0:
        lock.unlink()
    return store


def _hypergraph(*, randomized: bool, edges: list[tuple]) -> dict:
    document = {
        "built_from_split": "train",
        "complex": COMPLEX,
        "document_ids": ["doc-train"],
        "format": "sheaf-verify.hypergraph.v1",
        "hyperedges": [
            {"documents": ["doc-train"], "id": ident, "positions": positions, "vertices": vertices}
            for ident, vertices, positions in edges
        ],
        "randomized": randomized,
        "vertices": [
            {"document": "doc-train", "id": "v0", "position": 0},
            {"document": "doc-train", "id": "v1", "position": 1},
            {"document": "doc-train", "id": "v2", "position": 2},
        ],
    }
    return document


def _metrics(samples: dict[str, list[float]], random_graph: dict) -> dict:
    arms = {}
    for name, values in samples.items():
        mean = sum(values) / len(values)
        graph = 0 if name == "superword_bpe" else 8000
        train = 108000 if name == "superword_bpe" else 100000
        arm = {
            "bpb_per_seed": values,
            "eval": {
                "denominator": "raw_corpus_bytes",
                "raw_bytes": len(TEST),
                "scored_bytes": len(TEST),
                "total_bits": mean * len(TEST),
            },
            "flops": {"graph_build": graph, "total": train + graph, "train": train},
            "kind": name,
            "train_bytes": len(TRAIN),
            "vocab_size": 4,
        }
        if name == "randomized_hypergraph":
            arm["edge_cardinality_histogram"] = cardinality_histogram(random_graph)
            arm["hyperedge_count"] = len(random_graph["hyperedges"])
            arm["incidence_sha256"] = incidence_sha256(random_graph)
            arm["randomization_seed"] = random_graph["randomization_seed"]
            arm["vertices"] = len(random_graph["vertices"])
        arms[name] = arm
    return {
        "arms": arms,
        "bootstrap_draws": 2000,
        "bootstrap_seed": 17,
        "flop_relative_tolerance": 0.05,
        "format": "sheaf-verify.metrics.v1",
        "seed_design": "independent",
        "spec": _spec(),
        "superiority": {
            "against": "superword_bpe",
            "arm": "hypergraph",
            "claimed": False,
            "metric": "bpb",
        },
    }


def _pilot_arm(bpb: list[float], flops: float, vocab: int, graph: int) -> dict:
    return {
        "bpb_per_seed": bpb,
        "eval": {
            "denominator": "raw_corpus_bytes",
            "raw_bytes": len(TEST),
            "scored_bytes": len(TEST),
        },
        "graph_build_flops": graph,
        "train_bytes": len(TRAIN),
        "train_flops_per_byte": flops,
        "vocab_size": vocab,
    }


def _spec() -> dict:
    return {
        "format": "sheaf-verify.spec.v1",
        "g0": {
            "build_cpu_hours": 0.5,
            "byte_fallback_percent": 0.1,
            "comparison": "gpt2_bpe",
            "fewer_units_percent": 20.0,
            "format": "sheaf-verify.g0.v1",
            "gpt2_bpe_valid_units": 1000,
            "note": "Synthetic gate inputs. Not a corpus and not a training run.",
            "not_data": True,
            "peak_ram_gb": 8.0,
            "record_class": "synthetic",
            "split": "valid",
            "valid_units": 800,
            "vocab_size": 64000,
        },
        "g1": {
            "arms": {
                "gpt2_bpe": _pilot_arm([1.5, 1.6, 1.7], 16.0, 50257, 0),
                "hnt64k": _pilot_arm([1.0, 1.1, 1.2], 10.0, 64000, 8000),
                "struct32k": _pilot_arm([1.3, 1.4, 1.5], 14.0, 32768, 0),
                "sw64k": _pilot_arm([1.25, 1.35, 1.45], 12.0, 64000, 0),
            },
        },
        "g2": {
            "comparisons": [
                {
                    "bpb_per_seed": [1.0, 1.1, 0.9],
                    "control": "shuffled_degree",
                    "control_bpb_per_seed": [2.0, 2.1, 1.9],
                    "name": "a2_spectral",
                },
                {
                    "bpb_per_seed": [1.0, 1.05, 0.95],
                    "control": "random_dag",
                    "control_bpb_per_seed": [2.0, 2.05, 1.95],
                    "name": "a3_compositional",
                },
            ],
        },
        "g3": {"dtype": "bf16", "rel_err": 0.0001, "sheafheat_step_fraction": 0.10},
        "g4": {
            "baseline": {
                "all_in_flops": 108000,
                "bpb_per_seed": [1.90, 2.00, 2.10, 1.95, 2.05],
                "graph_build_flops": 0,
                "name": "tuned_modern",
                "train_flops": 108000,
            },
            "components": [
                {
                    "bpb_per_seed": [1.0, 1.1, 0.9],
                    "control": "random_hypergraph",
                    "control_bpb_per_seed": [1.8, 1.9, 1.7],
                    "name": "attention_bias",
                },
                {
                    "bpb_per_seed": [1.0, 1.05, 0.95],
                    "control": "random_hypergraph",
                    "control_bpb_per_seed": [2.0, 2.05, 1.95],
                    "name": "compositional",
                },
                {
                    "bpb_per_seed": [1.0, 1.1, 0.9],
                    "control": "random_hypergraph",
                    "control_bpb_per_seed": [1.8, 1.9, 1.7],
                    "name": "sheaf_heat",
                },
                {
                    "bpb_per_seed": [1.0, 1.1, 0.9],
                    "control": "random_hypergraph",
                    "control_bpb_per_seed": [2.0, 2.1, 1.9],
                    "name": "spectral",
                },
            ],
            "headline": {
                "all_in_flops": 108000,
                "bpb_per_seed": [1.40, 1.50, 1.45, 1.55, 1.35],
                "graph_build_flops": 8000,
                "name": "hnt",
                "train_flops": 100000,
            },
        },
        "g5": {
            "sizes": [
                {"gain": 0.01, "params": 15000000, "seeds": 3},
                {"gain": 0.02, "params": 30000000, "seeds": 3},
                {"gain": 0.03, "params": 60000000, "seeds": 2},
                {"gain": 0.04, "params": 120000000, "seeds": 1},
            ],
        },
        "g6": {
            "a7_all_in_flops": 100000,
            "a7_bpb": 1.40,
            "all_in_flops": 100000,
            "arm": "motif_head",
            "bpb": 1.20,
        },
        "h7": {
            "headline": False,
            "note": "Secondary. Not the bits-per-byte headline.",
            "tasks": ["lambada", "hellaswag"],
        },
        "p0": {
            "baseline_wall_interpolated": False,
            "bytes_per_s": 1000.0,
            "flops": 108000,
            "mfu": 0.2,
            "schedule_complete": True,
        },
        "p1": {
            "arms": [
                "enhanced_wallmatch",
                "hybrid",
                "random_rewire",
                "s4d",
                "shape_matched",
                "sheaf_reg",
                "spectral_init",
            ],
        },
        "p2": {
            "headline_seeds": 5,
            "modern_baseline": True,
            "pe_controls": ["sinusoidal", "rope", "alibi", "nope", "heat", "heat_rope"],
        },
        "p3": {
            "fineweb_edu_val_bpb": 1.4,
            "not_data": True,
            "record_class": "synthetic",
            "wt103_full_epoch": True,
        },
        "p5": {"cuda_graph_all_arms": True, "time_to_target_bpb_s": 100.0},
        "p6": {
            "replication_note": "Synthetic second-machine note. Not a real rerun.",
            "replication_recorded": True,
        },
        "stages": {stage: {"gate": gate, "work": work} for stage, (work, gate) in STAGES.items()},
    }


def _checkpoint_bytes(arm: str) -> bytes:
    label = f"checkpoint-{arm}".encode("ascii")
    if len(label) > 4096:
        raise RuntimeError(arm)
    return label + b"\0" * (4096 - len(label))


def write_fixture(root: Path) -> None:
    build(root)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    write_fixture(args.out)
