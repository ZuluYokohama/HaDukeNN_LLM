"""AGENTS.md section 3 empirical exclusion and Chodrow-null audit policies.

Enforces:
- ``validate_output_head_policy``: rejects ``182x181`` / ``fp2`` / ``cluster_is`` / ``goodman_factored``
  heads on ``vocab_size == 32,768`` GPU pretraining (T-38 KILL); allows ``cluster_is`` for
  ``V >= 128,000`` GPU and ``182x181`` ``goodman_factored`` strictly on ``edge_npu`` with
  ``batch_size == 1`` and ``dtype == "int8"`` (T-43).
- ``validate_hypergraph_claim``: requires degree- and cardinality-matched ``build_chodrow_null``
  controls (Chodrow 2020) and strict ``hyper_bpb < null_bpb``.
- ``validate_e1_ordering_policy``: rejects passive ``E1`` article-order shuffling.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from hadukenn_core.hypergraph import DecontaminatedHypergraph
from hadukenn_core.partitions import ClusterPartition
from sheaf_verify.hashing import SheafError

_FACTORED_HEADS = frozenset({"182x181", "fp2", "factored_fp2", "cluster_is", "goodman_factored"})
_DENSE_HEADS = frozenset({"dense", "flat", "dense_flat", "tiled_cce"})
_ALLOWED_E1_MODES = frozenset({"hyper", "dense", "random", "active_sheaf"})
_FORBIDDEN_E1_MODES = frozenset({"passive_e1_shuffle", "e1_shuffle", "passive_shuffle", "shuffle", ""})


def validate_output_head_policy(
    vocab_size: int,
    n_clusters: int | None = None,
    cluster_size: int | None = None,
    head_kind: str | None = None,
    hardware_target: str = "gpu_pretrain",
    batch_size: int = 64,
    dtype: str = "bf16",
    *,
    hardware: str | None = None,
) -> dict[str, Any]:
    """Enforce empirical output-head regime rules (AGENTS.md section 3, T-38 / T-41 / T-43)."""
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, (int, np.integer)) or int(vocab_size) < 1:
        raise SheafError(f"vocab_size must be a positive integer, got {vocab_size!r}")
    if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or int(batch_size) < 1:
        raise SheafError(f"batch_size must be a positive integer, got {batch_size!r}")
    if n_clusters is not None:
        if isinstance(n_clusters, bool) or not isinstance(n_clusters, (int, np.integer)) or int(n_clusters) < 1:
            raise SheafError(f"n_clusters must be a positive integer, got {n_clusters!r}")
        n_clusters = int(n_clusters)
    if cluster_size is not None:
        if isinstance(cluster_size, bool) or not isinstance(cluster_size, (int, np.integer)) or int(cluster_size) < 1:
            raise SheafError(f"cluster_size must be a positive integer, got {cluster_size!r}")
        cluster_size = int(cluster_size)

    v_sz = int(vocab_size)
    b_sz = int(batch_size)
    hw = (hardware if hardware is not None else hardware_target).strip().lower()
    hk = head_kind.strip().lower() if head_kind is not None else None
    dt = dtype.strip().lower()

    is_182x181 = n_clusters == 182 and cluster_size == 181
    is_factored = is_182x181 or (hk in _FACTORED_HEADS)

    if hw in {"gpu_pretrain", "gpu", "cuda"}:
        if is_factored:
            if v_sz < 128_000:
                raise SheafError(
                    f"AGENTS.md empirical exclusion (T-38 KILL): head_kind={hk!r} "
                    f"({n_clusters}x{cluster_size}) rejected for V={v_sz} on {hw}"
                )
            if n_clusters is not None and cluster_size is not None and int(n_clusters) * int(cluster_size) < v_sz:
                raise SheafError(
                    f"Factored partition capacity {n_clusters}*{cluster_size} < vocab_size {v_sz}"
                )
            rec = hk or "cluster_is"
            return {
                "ok": True,
                "passed": True,
                "recommended_head": rec,
                "head": rec,
                "head_kind": rec,
                "regime": "large_vocab_gpu_cluster_is",
            }
        if hk is None or hk in _DENSE_HEADS:
            rec = hk or "tiled_cce"
            return {
                "ok": True,
                "passed": True,
                "recommended_head": rec,
                "head": rec,
                "head_kind": rec,
                "regime": "dense_gpu_pretrain",
            }
        raise SheafError(f"Unsupported head_kind {head_kind!r} for {hw}")

    if hw == "edge_npu":
        if is_factored:
            if b_sz != 1 or dt != "int8":
                raise SheafError(
                    f"edge_npu factored head requires batch_size=1 and dtype='int8', "
                    f"got batch_size={b_sz}, dtype={dt!r}"
                )
            if n_clusters is not None and cluster_size is not None and int(n_clusters) * int(cluster_size) < v_sz:
                raise SheafError(
                    f"Factored partition capacity {n_clusters}*{cluster_size} < vocab_size {v_sz}"
                )
            rec = hk or "goodman_factored"
            return {
                "ok": True,
                "passed": True,
                "recommended_head": rec,
                "head": rec,
                "head_kind": rec,
                "regime": "edge_npu_int8_looped",
            }
        if hk is None or hk in _DENSE_HEADS:
            rec = hk or "dense_flat"
            return {
                "ok": True,
                "passed": True,
                "recommended_head": rec,
                "head": rec,
                "head_kind": rec,
                "regime": "edge_npu_dense",
            }
        raise SheafError(f"Unsupported head_kind {head_kind!r} for edge_npu")

    if hw == "cpu":
        if is_factored and v_sz < 128_000 and b_sz > 1:
            raise SheafError(f"Factored head {hk!r} rejected on CPU with batch_size={b_sz} and V={v_sz}")
        if hk is not None and hk not in (_DENSE_HEADS | _FACTORED_HEADS):
            raise SheafError(f"Unsupported head_kind {head_kind!r} for cpu")
        rec = hk or "dense_flat"
        return {
            "ok": True,
            "passed": True,
            "recommended_head": rec,
            "head": rec,
            "head_kind": rec,
            "regime": "cpu_reference",
        }

    raise SheafError(f"Unknown hardware_target: {hw!r}")


select_output_head_regime = validate_output_head_policy
check_output_head_policy = validate_output_head_policy


def validate_hypergraph_claim(
    hypergraph: DecontaminatedHypergraph | None = None,
    chodrow_null: DecontaminatedHypergraph | None = None,
    hyper_bpb: float | None = None,
    null_bpb: float | None = None,
    *,
    hg: DecontaminatedHypergraph | None = None,
    null_hg: DecontaminatedHypergraph | None = None,
    metric: float | None = None,
) -> dict[str, Any]:
    """Require degree/cardinality-matched Chodrow (2020) null control and strict hyper_bpb < null_bpb."""
    hg_obj = hg if hg is not None else hypergraph
    null_obj = null_hg if null_hg is not None else chodrow_null
    h_bpb = hyper_bpb if hyper_bpb is not None else metric

    if hg_obj is None:
        raise SheafError("validate_hypergraph_claim requires a DecontaminatedHypergraph")
    if null_obj is None:
        raise SheafError("Hypergraph claim rejected: missing required Chodrow (2020) null control")
    if hg_obj is null_obj:
        raise SheafError("Hypergraph claim rejected: hypergraph cannot serve as its own Chodrow null control")
    if not hasattr(hg_obj, "incidence") or not hasattr(null_obj, "incidence"):
        raise SheafError("hypergraph and chodrow_null must expose incidence matrices")
    if int(hg_obj.n_docs) != int(null_obj.n_docs):
        raise SheafError(f"Chodrow null n_docs {null_obj.n_docs} != hypergraph n_docs {hg_obj.n_docs}")

    deg_orig = np.asarray(hg_obj.incidence.sum(axis=1)).ravel()
    deg_null = np.asarray(null_obj.incidence.sum(axis=1)).ravel()
    if not np.array_equal(deg_orig, deg_null):
        raise SheafError("Chodrow null document degrees do not match original hypergraph")

    card_orig = sorted(int(x) for x in np.asarray(hg_obj.incidence.sum(axis=0)).ravel())
    card_null = sorted(int(x) for x in np.asarray(null_obj.incidence.sum(axis=0)).ravel())
    if card_orig != card_null:
        raise SheafError("Chodrow null edge cardinalities do not match original hypergraph")

    delta_bpb: float | None = None
    if h_bpb is not None:
        h_val = float(h_bpb)
        if not math.isfinite(h_val):
            raise SheafError(f"hyper_bpb must be finite, got {h_bpb!r}")
        if null_bpb is None:
            raise SheafError("null_bpb is required when hyper_bpb is supplied")
        n_val = float(null_bpb)
        if not math.isfinite(n_val):
            raise SheafError(f"null_bpb must be finite, got {null_bpb!r}")
        if h_val >= n_val:
            raise SheafError(
                f"Hypergraph claim rejected: hyper_bpb={h_val:.6f} does not beat null_bpb={n_val:.6f}"
            )
        delta_bpb = n_val - h_val

    return {
        "ok": True,
        "passed": True,
        "n_docs": int(hg_obj.n_docs),
        "n_edges": int(len(hg_obj.edges)),
        "delta_bpb": delta_bpb,
    }


audit_chodrow_null_control = validate_hypergraph_claim


def validate_e1_ordering_policy(
    mode: str | None = None,
    e1_passive_shuffle: bool = False,
    *,
    ordering_mode: str | None = None,
) -> dict[str, Any]:
    """Reject passive E1 article-order shuffling (AGENTS.md section 3)."""
    if e1_passive_shuffle:
        raise SheafError("Passive E1 article-order shuffling is empirically excluded (AGENTS.md section 3)")
    raw_mode = ordering_mode if ordering_mode is not None else mode
    if not isinstance(raw_mode, str):
        raise SheafError("E1 ordering mode must be a non-empty string")
    norm = raw_mode.strip().lower()
    if norm in _FORBIDDEN_E1_MODES or norm not in _ALLOWED_E1_MODES:
        raise SheafError(f"E1 ordering mode {raw_mode!r} is rejected by AGENTS.md section 3")
    return {
        "ok": True,
        "passed": True,
        "mode": norm,
    }


check_e1_ordering_policy = validate_e1_ordering_policy


def validate_partition_policy(
    partition: ClusterPartition,
    *,
    head_kind: str = "cluster_is",
    hardware_target: str = "gpu_pretrain",
    batch_size: int = 64,
    dtype: str = "bf16",
) -> dict[str, Any]:
    """Validate both bijection integrity of ``ClusterPartition`` and output-head regime policy."""
    bio = partition.verify_bijection()
    if not bio["bijection_ok"]:
        raise SheafError("ClusterPartition failed exact bijection check")
    res = validate_output_head_policy(
        vocab_size=partition.vocab_size,
        n_clusters=partition.n_clusters,
        cluster_size=partition.cluster_size,
        head_kind=head_kind,
        hardware_target=hardware_target,
        batch_size=batch_size,
        dtype=dtype,
    )
    res["bijection_ok"] = True
    return res


__all__ = [
    "audit_chodrow_null_control",
    "check_e1_ordering_policy",
    "check_output_head_policy",
    "select_output_head_regime",
    "validate_e1_ordering_policy",
    "validate_hypergraph_claim",
    "validate_output_head_policy",
    "validate_partition_policy",
]
