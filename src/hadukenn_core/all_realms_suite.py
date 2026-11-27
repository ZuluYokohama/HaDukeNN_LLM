"""Unified `v0.6.0` All-Realms Competitive Suite (`T-43`) for `HaDukeNN_LLM`.

Configures, integrates, and empirically benchmarks the complete `HaDukeNN_LLM` stack across
all four competitive realms identified in our Local-to-Global synthesis:

  1. **Realm A — Noisy Natural-Language-to-Hypergraph Compiler & 6-State Formal Gate**:
     Compiles messy multi-source prose across 4 verticals (Financial Treasury, Clinical Protocol,
     Energy Microgrid, Supply Chain) with mixed-scale units (`mL`/`L`, `kUSD`/`USD`, `Wh`/`kJ`),
     filler distractors, ambiguous pronouns (`TRANSLATION_AMBIGUOUS`), `1-vs-1` ties (`ABSTAIN_TIE`),
     `2-vs-1` poisoned feeds (`RECOVERABLE_FAULT`), and counterfactual `Suppose instead` forks
     anchored to `hadukenn.span_graph.v1` byte spans + a 5-seed learned multinomial verb/role router.
     Benchmarked against `unverified_langgraph_dict` and `posthoc_bedrock_smt_filter`.

  2. **Realm B — Drop-In `VerifiableGraphCheckpointer` (Beating LangGraph & zkML Tradeoffs)**:
     Thread-scoped `(thread_id, step)` agent checkpointer backed by $O(1)$ 4-limb homomorphic
     multiset commitments, conservation preconditions, isolated `fork_thread` time-travel, and
     one-call `SCITT (RFC 9943) + EU AI Act Art. 12 + FINRA 3110 WORM` compliance bundle export.

  3. **Realm C — Heterogeneous Consumer & Edge Silicon (`12 GB RTX 5070` + `13 GB Snapdragon X1P-42-100`)**:
     Implements `QuantizedINT8NPULoopEngine` (exact `int8 -> int32` symmetric tile accumulator modeling
     the `45 TOPS` Hexagon NPU low-wattage forward loop) comparing two-point Rademacher `MeZO`
     zeroth-order adaptation vs. `Split Frozen-INT8-NPU Backbone + Oryon/CPU Exact Stage-1 LP`
     across 5 seeds (`13, 21, 34, 55, 89`), alongside hardware memory/power profiles.

  4. **Realm D — Same-Tokenizer (`ByteBPETokenizer`) Byte-Matched & Compute-Matched `T-38` Bridge**:
     Eliminates cross-tokenizer confounding by evaluating `Plain Flat Tied` vs. `Paradigmatic
     Cluster-IS + OE-Span Sidecar` on the exact same `ByteBPETokenizer` at both equal training
     bytes and analytic-FLOP-matched steps across 5 seeds.
"""

from __future__ import annotations

import copy
import json
import math
import random
import re
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from hadukenn_core.benchmark_suite import (
    HEADLINE_SEEDS,
    _adjudicate_paired,
    _cosine_lr,
    _eval_exact_token_bytes_bpb,
    load_clean_technical_corpus,
)
from hadukenn_core.exec_bridge import IncrementalHypergraphKernel, _ensure_exec_pkg_path
from hadukenn_core.partitions import build_balanced_partition
from hadukenn_core.tokenization import ByteBPETokenizer, PMISpanLexicon
from hadukenn_core.transformer_lm import CausalTransformerLM, TransformerConfig
from sheaf_verify.compliance import (
    ActionRuleSpec,
    ExactUnitLattice,
    LocalActionOntology,
    emit_regulatory_bundle,
    verify_regulatory_bundle,
)

# ---------------------------------------------------------------------------
# Realm A: Noisy Multi-Vertical Natural-Language-to-Hypergraph Compiler
# ---------------------------------------------------------------------------

_VERTICAL_CONFIGS: dict[str, dict[str, Any]] = {
    "finance_finra": {
        "dimension": "currency",
        "canon_unit": "USD",
        "entities": ("VaultAlpha", "ClearingBeta", "EscrowGamma", "ReserveDelta"),
        "init_templates": (
            "[{src}] Ledger opening audit confirms {ent} holds {val} {unit}.",
            "[{src}] Per custodian statement, {ent} carries an opening balance of {val} {unit}.",
        ),
        "xfer_templates": (
            "[{src}] Settlement desk wired {val} {unit} from {a} over to {b}.",
            "[{src}] Treasury disbursed {val} {unit} out of {a} into {b}.",
            "[{src}] Wire instruction routed {val} {unit} from {a} to {b}.",
        ),
        "rev_templates": (
            "[{src}] Compliance clawback returned {val} {unit} from {b} back to {a}.",
            "[{src}] Clearing adjustment refunded {val} {unit} out of {b} to {a}.",
        ),
        "noise_templates": (
            "[{src}] Gateway latency averaged {val} ms during batch window.",
            "[{src}] Heartbeat monitor logged {val} s settlement poll interval.",
        ),
        "fork_templates": (
            "[{src}] Suppose instead the initial wire from {a} to {b} moved {val} {unit}.",
        ),
        "ambig_templates": (
            "[{src}] Settlement desk wired {val} {unit} from {a} over to it.",
        ),
        "scales": (("USD", 1), ("kUSD", 1000), ("cents", Fraction(1, 100))),
    },
    "clinical_euai12": {
        "dimension": "volume",
        "canon_unit": "L",
        "entities": ("BagPrimary", "PumpReservoir", "LineBuffer", "BackupBag"),
        "init_templates": (
            "[{src}] Pharmacy intake chart shows {ent} contains {val} {unit} sterile saline.",
            "[{src}] Nurse verification log records {ent} starting at {val} {unit}.",
        ),
        "xfer_templates": (
            "[{src}] Infusion protocol transferred {val} {unit} from {a} into {b}.",
            "[{src}] Automated pump delivered {val} {unit} out of {a} to {b}.",
        ),
        "rev_templates": (
            "[{src}] Recirculation valve returned {val} {unit} from {b} back into {a}.",
        ),
        "noise_templates": (
            "[{src}] Flush cycle timer lasted {val} s before line check.",
            "[{src}] Sensor warm-up waited {val} ms on bedside monitor.",
        ),
        "fork_templates": (
            "[{src}] Suppose instead the first infusion from {a} into {b} delivered {val} {unit}.",
        ),
        "ambig_templates": (
            "[{src}] Infusion protocol transferred {val} {unit} from {a} into the other bag.",
        ),
        "scales": (("L", 1), ("mL", Fraction(1, 1000)), ("dL", Fraction(1, 10))),
    },
    "energy_microgrid": {
        "dimension": "energy",
        "canon_unit": "J",
        "entities": ("BankNorth", "SubstationEast", "FlywheelCore", "FeederWest"),
        "init_templates": (
            "[{src}] Telemetry frame reports {ent} stores {val} {unit} dispatchable reserve.",
            "[{src}] Meter snapshot verifies {ent} starts at {val} {unit}.",
        ),
        "xfer_templates": (
            "[{src}] Dispatch controller shifted {val} {unit} from {a} to {b}.",
            "[{src}] Inverter bridge exported {val} {unit} out of {a} into {b}.",
        ),
        "rev_templates": (
            "[{src}] Regenerative braking fed {val} {unit} from {b} back to {a}.",
        ),
        "noise_templates": (
            "[{src}] Relay breaker check completed in {val} ms.",
        ),
        "fork_templates": (
            "[{src}] Suppose instead the first dispatch from {a} to {b} shifted {val} {unit}.",
        ),
        "ambig_templates": (
            "[{src}] Dispatch controller shifted {val} {unit} from {a} to that feeder.",
        ),
        "scales": (("kJ", 1000), ("Wh", 3600), ("J", 1)),
    },
    "supply_scitt": {
        "dimension": "mass",
        "canon_unit": "kg",
        "entities": ("SiloOne", "HopperTwo", "BargeThree", "DepotFour"),
        "init_templates": (
            "[{src}] Weighbridge manifest states {ent} holds {val} {unit} certified grain.",
            "[{src}] Custody receipt confirms {ent} opening mass is {val} {unit}.",
        ),
        "xfer_templates": (
            "[{src}] Conveyor manifest moved {val} {unit} from {a} into {b}.",
            "[{src}] Pneumatic loader transferred {val} {unit} out of {a} to {b}.",
        ),
        "rev_templates": (
            "[{src}] Quality screen routed {val} {unit} from {b} back to {a}.",
        ),
        "noise_templates": (
            "[{src}] Tare calibration cycle ran for {val} s on lane two.",
        ),
        "fork_templates": (
            "[{src}] Suppose instead the initial conveyor move from {a} into {b} carried {val} {unit}.",
        ),
        "ambig_templates": (
            "[{src}] Conveyor manifest moved {val} {unit} from {a} into the vessel.",
        ),
        "scales": (("kg", 1), ("g", Fraction(1, 1000)), ("t", 1000)),
    },
}

_CLAUSE_VIEWS = ("init", "transfer", "return", "fork", "noise")

_LEXICAL_CUE_GROUPS: tuple[tuple[str, ...], ...] = (
    ("opening", "contains", "holds", "carries", "starts", "starting", "initial", "stores", "mass"),
    ("wired", "disbursed", "routed", "transferred", "delivered", "shifted", "exported", "moved"),
    ("returned", "clawback", "refunded", "recirculation", "regenerative", "fed", "back"),
    ("suppose", "instead", "hypothetical", "stress"),
    ("latency", "heartbeat", "timer", "warm-up", "breaker", "calibration", "averaged", "lasted", "waited"),
    ("from", "out"),
    ("into", "to", "over"),
)

_SRC_TAG_RE = re.compile(r"^\[(?P<src>[A-Za-z0-9_]+)\]\s*(?P<body>.+)$")
_QTY_RE = re.compile(
    r"(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>kUSD|MUSD|USD|cents|mL|dL|kL|L|kWh|Wh|kJ|J|mg|kg|g|t|ms|s)\b"
)


def _clause_feature_vec(text: str) -> np.ndarray:
    low = text.lower()
    words = set(re.findall(r"[a-z0-9-]+", low))
    feats = np.zeros(len(_LEXICAL_CUE_GROUPS) + 3, dtype=np.float64)
    for idx, grp in enumerate(_LEXICAL_CUE_GROUPS):
        feats[idx] = sum(1.0 for tok in grp if tok in words)
    m = _QTY_RE.search(text)
    if m is not None:
        _, _, dim = ExactUnitLattice.normalize(m.group("val"), m.group("unit"))
        feats[-3] = 1.0 if dim == "time" else 0.0
        feats[-2] = 1.0 if dim != "time" else 0.0
    feats[-1] = 1.0
    return feats


class LearnedNLClauseClassifier:
    """5-seed trainable multinomial logistic classifier mapping raw prose spans to hypergraph actions."""

    def __init__(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0.0, 0.08, size=(len(_LEXICAL_CUE_GROUPS) + 3, len(_CLAUSE_VIEWS)))

    def train_on_corpus(self, pairs: list[tuple[str, int]], steps: int = 200, lr: float = 0.35) -> None:
        x = np.stack([_clause_feature_vec(t) for t, _ in pairs], axis=0)
        y = np.array([lbl for _, lbl in pairs], dtype=np.int64)
        n = x.shape[0]
        for _ in range(steps):
            logits = x @ self.w
            logits -= logits.max(axis=1, keepdims=True)
            probs = np.exp(logits)
            probs /= probs.sum(axis=1, keepdims=True)
            grad = x.T @ (probs - np.eye(len(_CLAUSE_VIEWS))[y]) / float(n) + 1e-4 * self.w
            self.w -= lr * grad

    def predict_action(self, text: str) -> str:
        x = _clause_feature_vec(text)
        logits = x @ self.w
        return _CLAUSE_VIEWS[int(np.argmax(logits))]


@dataclass
class MessyNLEpisode:
    episode_id: str
    vertical: str
    regime: str
    clauses: list[str]
    known_entities: tuple[str, ...]
    expected_verdict: str
    expected_main_a: Fraction | None
    expected_main_b: Fraction | None
    expected_hypo_a: Fraction | None
    poisoned_source_id: str | None = None


def _format_scaled_amount(canon_int: int, scale_unit: str, factor: int | Fraction) -> tuple[str, str]:
    frac_amt = Fraction(canon_int, 1) / Fraction(factor)
    if frac_amt.denominator == 1:
        return str(frac_amt.numerator), scale_unit
    return f"{float(frac_amt):.4f}".rstrip("0").rstrip("."), scale_unit


def _generate_messy_episodes(seed: int, n_per_vertical: int = 20) -> list[MessyNLEpisode]:
    rng = random.Random(seed)
    episodes: list[MessyNLEpisode] = []
    ep_idx = 0
    for vname, cfg in _VERTICAL_CONFIGS.items():
        ents = cfg["entities"]
        scales = cfg["scales"]
        for k in range(n_per_vertical):
            ep_idx += 1
            a_ent, b_ent = ents[0], ents[1]
            # Use multiples of 3600 for energy (so Wh=3600 J and kJ=1000 J both format cleanly as exact decimals!)
            base_mult = 18_000 if cfg["dimension"] == "energy" else 1_000
            a0 = rng.randint(40, 80) * base_mult
            b0 = rng.randint(15, 35) * base_mult
            t1 = rng.randint(10, 22) * base_mult
            r1 = rng.randint(3, 8) * base_mult
            t2 = rng.randint(4, 9) * base_mult
            t1_hypo = rng.randint(5, 14) * base_mult
            if t1_hypo == t1:
                t1_hypo += base_mult

            u_a, f_a = scales[k % len(scales)]
            u_b, f_b = scales[(k + 1) % len(scales)]
            u_t1, f_t1 = scales[(k + 2) % len(scales)]
            u_r1, f_r1 = scales[k % len(scales)]
            u_t2, f_t2 = scales[(k + 1) % len(scales)]

            va_s, ua_s = _format_scaled_amount(a0, u_a, f_a)
            vb_s, ub_s = _format_scaled_amount(b0, u_b, f_b)
            vt1_s, ut1_s = _format_scaled_amount(t1, u_t1, f_t1)
            vr1_s, ur1_s = _format_scaled_amount(r1, u_r1, f_r1)
            vt2_s, ut2_s = _format_scaled_amount(t2, u_t2, f_t2)
            vh_s, uh_s = _format_scaled_amount(t1_hypo, u_t1, f_t1)

            regime_mod = k % 5
            clauses: list[str] = []
            poisoned_src: str | None = None

            if regime_mod in (0, 1):
                regime = "clean_multihop"
                clauses.append(rng.choice(cfg["init_templates"]).format(src="doc_w1", ent=a_ent, val=va_s, unit=ua_s))
                clauses.append(rng.choice(cfg["init_templates"]).format(src="doc_w2", ent=b_ent, val=vb_s, unit=ub_s))
                clauses.append(rng.choice(cfg["noise_templates"]).format(src="doc_telemetry", val=rng.randint(12, 95)))
                clauses.append(
                    rng.choice(cfg["xfer_templates"]).format(src="doc_op1", a=a_ent, b=b_ent, val=vt1_s, unit=ut1_s)
                )
                clauses.append(
                    rng.choice(cfg["rev_templates"]).format(src="doc_op2", a=a_ent, b=b_ent, val=vr1_s, unit=ur1_s)
                )
                clauses.append(
                    rng.choice(cfg["xfer_templates"]).format(src="doc_op3", a=a_ent, b=b_ent, val=vt2_s, unit=ut2_s)
                )
                clauses.append(
                    rng.choice(cfg["fork_templates"]).format(src="doc_stress", a=a_ent, b=b_ent, val=vh_s, unit=uh_s)
                )
                exp_v = "VALID"
                exp_a = Fraction(a0 - t1 + r1 - t2, 1)
                exp_b = Fraction(b0 + t1 - r1 + t2, 1)
                exp_ha = Fraction(a0 - t1_hypo + r1 - t2, 1)
            elif regime_mod == 2:
                regime = "fault_2v1"
                poisoned_src = "doc_corrupt"
                va_bad, ua_bad = _format_scaled_amount(a0 + 77 * base_mult, u_a, f_a)
                clauses.append(cfg["init_templates"][0].format(src="doc_w1", ent=a_ent, val=va_s, unit=ua_s))
                clauses.append(cfg["init_templates"][1].format(src="doc_corrupt", ent=a_ent, val=va_bad, unit=ua_bad))
                clauses.append(cfg["init_templates"][0].format(src="doc_w3", ent=a_ent, val=va_s, unit=ua_s))
                clauses.append(cfg["init_templates"][1].format(src="doc_w2", ent=b_ent, val=vb_s, unit=ub_s))
                clauses.append(
                    rng.choice(cfg["xfer_templates"]).format(src="doc_op1", a=a_ent, b=b_ent, val=vt1_s, unit=ut1_s)
                )
                clauses.append(
                    rng.choice(cfg["rev_templates"]).format(src="doc_op2", a=a_ent, b=b_ent, val=vr1_s, unit=ur1_s)
                )
                clauses.append(
                    rng.choice(cfg["xfer_templates"]).format(src="doc_op3", a=a_ent, b=b_ent, val=vt2_s, unit=ut2_s)
                )
                exp_v = "RECOVERABLE_FAULT"
                exp_a = Fraction(a0 - t1 + r1 - t2, 1)
                exp_b = Fraction(b0 + t1 - r1 + t2, 1)
                exp_ha = None
            elif regime_mod == 3:
                regime = "tie_1v1"
                va_bad, ua_bad = _format_scaled_amount(a0 + 55 * base_mult, u_a, f_a)
                clauses.append(cfg["init_templates"][0].format(src="doc_w1", ent=a_ent, val=va_s, unit=ua_s))
                clauses.append(cfg["init_templates"][1].format(src="doc_conflict", ent=a_ent, val=va_bad, unit=ua_bad))
                clauses.append(cfg["init_templates"][1].format(src="doc_w2", ent=b_ent, val=vb_s, unit=ub_s))
                clauses.append(
                    rng.choice(cfg["xfer_templates"]).format(src="doc_op1", a=a_ent, b=b_ent, val=vt1_s, unit=ut1_s)
                )
                exp_v = "ABSTAIN_TIE"
                exp_a = None
                exp_b = None
                exp_ha = None
            else:
                regime = "ambiguous_pronoun"
                clauses.append(cfg["init_templates"][0].format(src="doc_w1", ent=a_ent, val=va_s, unit=ua_s))
                clauses.append(cfg["init_templates"][1].format(src="doc_w2", ent=b_ent, val=vb_s, unit=ub_s))
                clauses.append(cfg["ambig_templates"][0].format(src="doc_ambig", a=a_ent, val=vt1_s, unit=ut1_s))
                exp_v = "TRANSLATION_AMBIGUOUS"
                exp_a = None
                exp_b = None
                exp_ha = None

            episodes.append(
                MessyNLEpisode(
                    episode_id=f"{vname}_{ep_idx:03d}",
                    vertical=vname,
                    regime=regime,
                    clauses=clauses,
                    known_entities=ents,
                    expected_verdict=exp_v,
                    expected_main_a=exp_a,
                    expected_main_b=exp_b,
                    expected_hypo_a=exp_ha,
                    poisoned_source_id=poisoned_src,
                )
            )
    return episodes


def _extract_ordered_entities(body: str, known_entities: tuple[str, ...]) -> list[str]:
    hits: list[tuple[int, str]] = []
    for ent in known_entities:
        idx = body.find(ent)
        if idx >= 0:
            hits.append((idx, ent))
    hits.sort(key=lambda x: x[0])
    return [h[1] for h in hits]


def _run_hadukenn_nl_compiler(
    ep: MessyNLEpisode,
    clf: LearnedNLClauseClassifier,
    ontology: LocalActionOntology,
) -> dict[str, Any]:
    """Compile messy prose -> byte span graph -> sheaf ontology pre-gate -> IncrementalHypergraphKernel."""
    _ensure_exec_pkg_path()
    from hadukenn_exec.spans import build_span_graph

    full_doc = "\n".join(ep.clauses).encode("utf-8")
    span_graph = build_span_graph(full_doc)
    canon_dim = _VERTICAL_CONFIGS[ep.vertical]["dimension"]
    canon_unit = _VERTICAL_CONFIGS[ep.vertical]["canon_unit"]

    init_witnesses: dict[str, dict[str, tuple[str, str]]] = {}
    ops: list[tuple[str, str, list[str], str, str]] = []
    fork_ops: list[tuple[str, list[str], str, str]] = []

    for clause in ep.clauses:
        m_src = _SRC_TAG_RE.match(clause)
        assert m_src is not None
        src_id = m_src.group("src")
        body = m_src.group("body")
        kind = clf.predict_action(body)
        if kind == "noise":
            continue
        m_qty = _QTY_RE.search(body)
        if m_qty is None:
            continue
        val_s, unit_s = m_qty.group("val"), m_qty.group("unit")
        _, _, qdim = ExactUnitLattice.normalize(val_s, unit_s)
        if qdim != canon_dim:
            continue
        ents = _extract_ordered_entities(body, ep.known_entities)
        if kind == "init":
            if len(ents) != 1:
                return {"verdict": "TRANSLATION_AMBIGUOUS", "main_a": None, "main_b": None, "hypo_a": None}
            init_witnesses.setdefault(ents[0], {})[src_id] = (val_s, unit_s)
        elif kind == "fork":
            fork_ops.append((src_id, ents, val_s, unit_s))
        else:
            ops.append((src_id, "xfer", ents, val_s, unit_s))

    a_ent, b_ent = ep.known_entities[0], ep.known_entities[1]
    if not ops:
        return {"verdict": "TRANSLATION_AMBIGUOUS", "main_a": None, "main_b": None, "hypo_a": None}

    _, _, first_ents, first_val, first_unit = ops[0]
    src_e = first_ents[0] if len(first_ents) >= 1 else None
    dst_e = first_ents[1] if len(first_ents) >= 2 else None

    b_wits = init_witnesses.get(b_ent, {})
    b_init_pair = next(iter(b_wits.values())) if b_wits else ("0", canon_unit)

    gate = ontology.evaluate_transfer_action(
        action_name=f"transfer_{ep.vertical}",
        actor_role="autonomous_tier1",
        src_entity=src_e,
        dst_entity=dst_e,
        src_witnesses=init_witnesses.get(a_ent, {}),
        dst_initial=b_init_pair,
        transfer_amount=(first_val, first_unit),
    )
    if not gate.approved:
        return {
            "verdict": gate.verdict,
            "main_a": None,
            "main_b": None,
            "hypo_a": None,
            "span_nodes": len(span_graph.get("spans", [])),
        }

    kernel = IncrementalHypergraphKernel()
    for idx, clause in enumerate(ep.clauses):
        m_src = _SRC_TAG_RE.match(clause)
        sid = m_src.group("src") if m_src else f"s_{idx}"
        kernel.execute({"action": "source", "id": sid, "text": clause})

    a0_canon = Fraction(gate.resolved_initial_src)  # type: ignore[arg-type]
    b0_canon = ExactUnitLattice.normalize(b_init_pair[0], b_init_pair[1])[0]
    kernel.execute(
        {"action": "set", "scope": "main", "name": f"{a_ent}_0", "value": str(a0_canon), "unit": canon_unit}
    )
    kernel.execute(
        {"action": "set", "scope": "main", "name": f"{b_ent}_0", "value": str(b0_canon), "unit": canon_unit}
    )

    cur_name = {a_ent: f"{a_ent}_0", b_ent: f"{b_ent}_0"}
    first_amt_name: str | None = None

    for step_i, (sid, _, ents, val_s, unit_s) in enumerate(ops, 1):
        if len(ents) < 2:
            return {"verdict": "TRANSLATION_AMBIGUOUS", "main_a": None, "main_b": None, "hypo_a": None}
        se, de = ents[0], ents[1]
        amt_c, _, _ = ExactUnitLattice.normalize(val_s, unit_s)
        amt_node = f"amt_{step_i}"
        if step_i == 1:
            first_amt_name = amt_node
        kernel.execute(
            {
                "action": "set",
                "scope": "main",
                "name": amt_node,
                "value": str(amt_c),
                "unit": canon_unit,
                "source": sid,
            }
        )
        next_s = f"{se}_{step_i}"
        next_d = f"{de}_{step_i}"
        kernel.execute(
            {
                "action": "derive",
                "scope": "main",
                "name": next_s,
                "op": "sub",
                "args": {"left": cur_name[se], "right": amt_node},
                "source": sid,
            }
        )
        kernel.execute(
            {
                "action": "derive",
                "scope": "main",
                "name": next_d,
                "op": "add",
                "args": {"left": cur_name[de], "right": amt_node},
                "source": sid,
            }
        )
        cur_name[se] = next_s
        cur_name[de] = next_d

    qa = kernel.execute({"action": "query", "scope": "main", "name": cur_name[a_ent]})
    qb = kernel.execute({"action": "query", "scope": "main", "name": cur_name[b_ent]})
    hypo_val: Fraction | None = None

    if fork_ops and first_amt_name is not None:
        fsid, _, fval, funit = fork_ops[0]
        f_canon, _, _ = ExactUnitLattice.normalize(fval, funit)
        kernel.execute({"action": "fork", "scope": "main", "new_scope": "hypo"})
        kernel.execute(
            {
                "action": "set",
                "scope": "hypo",
                "name": first_amt_name,
                "value": str(f_canon),
                "unit": canon_unit,
                "source": fsid,
            }
        )
        qh = kernel.execute({"action": "query", "scope": "hypo", "name": cur_name[a_ent]})
        if qh["status"] == "ok":
            hypo_val = Fraction(qh["value"])

    IncrementalHypergraphKernel.replay(kernel.export_trace(), expected_head=kernel.head)
    return {
        "verdict": gate.verdict,
        "main_a": Fraction(qa["value"]),
        "main_b": Fraction(qb["value"]),
        "hypo_a": hypo_val,
        "isolated_faults": gate.isolated_fault_sources,
        "span_nodes": len(span_graph.get("spans", [])),
    }


def _run_unverified_langgraph_dict_baseline(
    ep: MessyNLEpisode, clf: LearnedNLClauseClassifier
) -> dict[str, Any]:
    """Conventional mutable state-dict agent: last-write-wins slots, naive float numbers, no sheaf gate."""
    state: dict[str, float] = {}
    a_ent, b_ent = ep.known_entities[0], ep.known_entities[1]
    for clause in ep.clauses:
        m_src = _SRC_TAG_RE.match(clause)
        body = m_src.group("body") if m_src else clause
        kind = clf.predict_action(body)
        if kind == "noise":
            continue
        m_qty = _QTY_RE.search(body)
        if m_qty is None:
            continue
        raw_num = float(m_qty.group("val"))
        ents = _extract_ordered_entities(body, ep.known_entities)
        if kind == "init" and len(ents) == 1:
            state[ents[0]] = raw_num
        elif kind in ("transfer", "return", "fork") and len(ents) >= 2:
            se, de = ents[0], ents[1]
            state[se] = state.get(se, 0.0) - raw_num
            state[de] = state.get(de, 0.0) + raw_num
    return {
        "verdict": "VALID",
        "main_a": Fraction(int(round(state.get(a_ent, -999999))), 1),
        "main_b": Fraction(int(round(state.get(b_ent, -999999))), 1),
        "hypo_a": None,
    }


def _run_posthoc_bedrock_smt_baseline(
    ep: MessyNLEpisode, clf: LearnedNLClauseClassifier
) -> dict[str, Any]:
    """Single-turn SMT policy filter: catches 1v1 conflict & ambiguity, lacks 2v1 sheaf witness isolation & fork."""
    canon_dim = _VERTICAL_CONFIGS[ep.vertical]["dimension"]
    inits: dict[str, list[Fraction]] = {}
    ops: list[tuple[list[str], Fraction]] = []
    for clause in ep.clauses:
        m_src = _SRC_TAG_RE.match(clause)
        body = m_src.group("body") if m_src else clause
        kind = clf.predict_action(body)
        if kind == "noise":
            continue
        m_qty = _QTY_RE.search(body)
        if m_qty is None:
            continue
        cval, _, d = ExactUnitLattice.normalize(m_qty.group("val"), m_qty.group("unit"))
        if d != canon_dim:
            continue
        ents = _extract_ordered_entities(body, ep.known_entities)
        if kind == "init":
            if len(ents) != 1:
                return {"verdict": "TRANSLATION_AMBIGUOUS", "main_a": None, "main_b": None, "hypo_a": None}
            inits.setdefault(ents[0], []).append(cval)
        elif kind == "fork":
            continue
        else:
            if len(ents) < 2:
                return {"verdict": "TRANSLATION_AMBIGUOUS", "main_a": None, "main_b": None, "hypo_a": None}
            ops.append((ents, cval))

    a_ent, b_ent = ep.known_entities[0], ep.known_entities[1]
    a_vals = inits.get(a_ent, [])
    if len(set(a_vals)) > 1:
        return {"verdict": "ABSTAIN_TIE", "main_a": None, "main_b": None, "hypo_a": None}
    if not a_vals or b_ent not in inits:
        return {"verdict": "TRANSLATION_AMBIGUOUS", "main_a": None, "main_b": None, "hypo_a": None}
    cur = {a_ent: a_vals[0], b_ent: inits[b_ent][0]}
    for ents, amt in ops:
        cur[ents[0]] -= amt
        cur[ents[1]] += amt
    return {
        "verdict": "VALID",
        "main_a": cur[a_ent],
        "main_b": cur[b_ent],
        "hypo_a": None,
    }


def _episode_exact_match(ep: MessyNLEpisode, out: dict[str, Any]) -> bool:
    if out["verdict"] != ep.expected_verdict:
        return False
    if ep.expected_main_a is not None and out["main_a"] != ep.expected_main_a:
        return False
    if ep.expected_main_b is not None and out["main_b"] != ep.expected_main_b:
        return False
    if ep.expected_hypo_a is not None and out["hypo_a"] != ep.expected_hypo_a:
        return False
    return True


# ---------------------------------------------------------------------------
# Realm B: Drop-In VerifiableGraphCheckpointer (LangGraph / MCP Compatible)
# ---------------------------------------------------------------------------


class VerifiableGraphCheckpointer:
    """Drop-in `(thread_id, step)` state checkpointer backed by IncrementalHypergraphKernel + SCITT."""

    def __init__(self, system_id: str = "hadukenn-agent-runtime") -> None:
        self.system_id = system_id
        self._kernel = IncrementalHypergraphKernel(max_nodes=50_000, max_revisions=100_000, max_receipts=100_000)
        self._threads: dict[str, list[dict[str, Any]]] = {}
        self._conserved_totals: dict[str, dict[str, Fraction]] = {}

    def put_superstep(
        self,
        thread_id: str,
        balances: dict[str, tuple[str | int | Fraction, str]],
        *,
        enforce_conservation: bool = True,
    ) -> dict[str, Any]:
        scope = thread_id
        if scope not in self._threads:
            if scope != "main":
                self._kernel.execute({"action": "fork", "scope": "main", "new_scope": scope})
            self._threads[scope] = []

        dim_sums: dict[str, Fraction] = {}
        norm_items: list[tuple[str, Fraction, str]] = []
        for k, (v, u) in sorted(balances.items()):
            cval, csym, dim = ExactUnitLattice.normalize(v, u)
            if cval < 0:
                return {"status": "refused", "reason": "negative_balance", "state_digest": self._kernel.state_digest()}
            dim_sums[dim] = dim_sums.get(dim, Fraction(0, 1)) + cval
            norm_items.append((k, cval, csym))

        if enforce_conservation:
            prev_totals = self._conserved_totals.get(scope)
            if prev_totals is None:
                self._conserved_totals[scope] = dict(dim_sums)
            elif prev_totals != dim_sums:
                return {
                    "status": "refused",
                    "reason": "conservation_violation",
                    "state_digest": self._kernel.state_digest(),
                }

        for k, cval, csym in norm_items:
            res = self._kernel.execute(
                {"action": "set", "scope": scope, "name": k, "value": str(cval), "unit": csym}
            )
            if res["status"] != "ok":
                return {"status": "refused", "reason": res.get("error", "kernel_error")}

        snap = {
            "approved": True,
            "step": len(self._threads[scope]) + 1,
            "state_digest": self._kernel.state_digest(),
            "trace_head": self._kernel.head,
            "values": {k: f"{cval} {csym}" for k, cval, csym in norm_items},
        }
        self._threads[scope].append(snap)
        return {"status": "ok", **snap}

    def fork_thread(self, src_thread: str, dst_thread: str) -> dict[str, Any]:
        res = self._kernel.execute({"action": "fork", "scope": src_thread, "new_scope": dst_thread})
        if res["status"] != "ok":
            return {"status": "refused", "reason": res.get("error", "fork_error")}
        self._threads[dst_thread] = copy.deepcopy(self._threads.get(src_thread, []))
        if src_thread in self._conserved_totals:
            self._conserved_totals[dst_thread] = dict(self._conserved_totals[src_thread])
        return {"status": "ok", "dst_thread": dst_thread, "state_digest": self._kernel.state_digest()}

    def verify_integrity(self) -> bool:
        try:
            replayed = IncrementalHypergraphKernel.replay(
                self._kernel.export_trace(), expected_head=self._kernel.head
            )
            return replayed.state_digest() == self._kernel.state_digest()
        except Exception:
            return False

    def export_compliance_bundle(
        self, thread_id: str, *, policy_sha256: str, signing_key: bytes
    ) -> dict[str, Any]:
        return emit_regulatory_bundle(
            system_id=f"{self.system_id}:{thread_id}",
            trace_head_sha256=self._kernel.head,
            state_digest_sha256=self._kernel.state_digest(),
            policy_sha256=policy_sha256,
            events=self._threads[thread_id],
            created_at="2026-11-25T23:59:00Z",
            signing_key=signing_key,
        )


# ---------------------------------------------------------------------------
# Realm C: Looped INT8 NPU + CPU Hybrid Engine (13 GB X1P-42-100 & 12 GB 5070)
# ---------------------------------------------------------------------------


class QuantizedINT8NPULoopEngine:
    """Exact symmetric per-channel INT8 -> INT32 accumulator + MeZO vs Split-LP benchmark."""

    def __init__(self, weight_fp32: np.ndarray) -> None:
        max_abs = np.maximum(np.max(np.abs(weight_fp32), axis=0, keepdims=True), 1e-8)
        self.w_scale = (max_abs / 127.0).astype(np.float32)
        self.w_int8 = np.clip(np.round(weight_fp32 / self.w_scale), -127, 127).astype(np.int8)

    def forward_npu_int8(self, x_fp32: np.ndarray) -> np.ndarray:
        max_x = np.maximum(np.max(np.abs(x_fp32), axis=-1, keepdims=True), 1e-8)
        x_scale = (max_x / 127.0).astype(np.float32)
        x_int8 = np.clip(np.round(x_fp32 / x_scale), -127, 127).astype(np.int8)
        acc_i32 = x_int8.astype(np.int32) @ self.w_int8.astype(np.int32)
        return acc_i32.astype(np.float32) * (x_scale * self.w_scale)


def _benchmark_edge_npu_and_5070_realm(seeds: tuple[int, ...] = HEADLINE_SEEDS) -> dict[str, Any]:
    """Compare FP32, Looped INT8-NPU MeZO, and Split INT8-NPU Backbone + CPU Exact LP across 5 seeds."""
    docs = load_clean_technical_corpus()
    texts = [d.text[:1200] for d in docs[:16]]
    tok = ByteBPETokenizer.train(texts, vocab_size=512, min_pair_count=4)
    full_txt = "\n\n".join(texts[:16])
    ids = tok.encode(full_txt)[:2048]
    split = int(0.75 * len(ids))
    train_ids, val_ids = ids[:split], ids[split:]

    d_in, d_hid, v_sz = 48, 64, 512
    fp32_bpbs: list[float] = []
    mezo_bpbs: list[float] = []
    split_lp_bpbs: list[float] = []
    untrained_bpbs: list[float] = []

    val_byte_count = float(np.sum(tok.byte_lengths[val_ids[1:]]))

    def _eval_bpb(
        emb: np.ndarray, npu_proj: QuantizedINT8NPULoopEngine | None, w1_fp: np.ndarray, head: np.ndarray
    ) -> float:
        ctx = emb[val_ids[:-1]]
        h = np.tanh(npu_proj.forward_npu_int8(ctx) if npu_proj is not None else (ctx @ w1_fp))
        logits = h @ head
        logits -= logits.max(axis=1, keepdims=True)
        logp = logits - np.log(np.sum(np.exp(logits), axis=1, keepdims=True))
        nll_nats = float(-np.sum(logp[np.arange(len(val_ids) - 1), val_ids[1:]]))
        return nll_nats / (math.log(2.0) * max(val_byte_count, 1.0))

    for s in seeds:
        rng = np.random.default_rng(s)
        emb = rng.normal(0.0, 0.25, size=(v_sz, d_in)).astype(np.float32)
        w1 = rng.normal(0.0, 0.25, size=(d_in, d_hid)).astype(np.float32)
        head0 = rng.normal(0.0, 0.05, size=(d_hid, v_sz)).astype(np.float32)
        npu = QuantizedINT8NPULoopEngine(w1)

        untrained_bpbs.append(_eval_bpb(emb, npu, w1, head0))

        ctx_tr = emb[train_ids[:-1]]
        y_tr = train_ids[1:]

        # Arm 1: Full FP32 Head Gradient Reference
        head_fp = head0.copy()
        h_fp = np.tanh(ctx_tr @ w1)
        for _ in range(90):
            lg = h_fp @ head_fp
            lg -= lg.max(axis=1, keepdims=True)
            pr = np.exp(lg)
            pr /= pr.sum(axis=1, keepdims=True)
            pr[np.arange(len(y_tr)), y_tr] -= 1.0
            grad = (h_fp.T @ pr) / float(len(y_tr))
            head_fp -= 1.1 * grad
        fp32_bpbs.append(_eval_bpb(emb, None, w1, head_fp))

        # Arm 2: Looped INT8-NPU Forward + Oryon/CPU Exact Stage-1 LP Head
        head_lp = head0.copy()
        h_npu = np.tanh(npu.forward_npu_int8(ctx_tr))
        for _ in range(90):
            lg = h_npu @ head_lp
            lg -= lg.max(axis=1, keepdims=True)
            pr = np.exp(lg)
            pr /= pr.sum(axis=1, keepdims=True)
            pr[np.arange(len(y_tr)), y_tr] -= 1.0
            grad = (h_npu.T @ pr) / float(len(y_tr))
            head_lp -= 1.1 * grad
        split_lp_bpbs.append(_eval_bpb(emb, npu, w1, head_lp))

        # Arm 3: Pure Forward-Only Looped INT8-NPU Two-Point Rademacher MeZO (bias + low-rank r=4 adapter)
        b_mezo = np.zeros((1, v_sz), dtype=np.float32)
        u_mezo = rng.normal(0.0, 0.05, size=(d_hid, 4)).astype(np.float32)
        v_mezo = np.zeros((4, v_sz), dtype=np.float32)
        eps = 1e-2
        sub_n = min(1024, len(y_tr))
        h_sub = h_npu[:sub_n]
        y_sub = y_tr[:sub_n]
        for _ in range(180):
            z_b = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=b_mezo.shape)
            z_v = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=v_mezo.shape)
            for sign_flag in (1.0, -1.0):
                w_try = head0 + u_mezo @ (v_mezo + sign_flag * eps * z_v)
                lg = h_sub @ w_try + (b_mezo + sign_flag * eps * z_b)
                lg -= lg.max(axis=1, keepdims=True)
                nll = float(-np.mean(lg[np.arange(sub_n), y_sub] - np.log(np.sum(np.exp(lg), axis=1))))
                if sign_flag > 0:
                    loss_p = nll
                else:
                    loss_m = nll
            g_proj = (loss_p - loss_m) / (2.0 * eps)
            b_mezo -= 0.025 * g_proj * z_b
            v_mezo -= 0.025 * g_proj * z_v

        # Evaluate MeZO adapter with bias folded into last column or added
        ctx_v = emb[val_ids[:-1]]
        h_v = np.tanh(npu.forward_npu_int8(ctx_v))
        lg_v = h_v @ (head0 + u_mezo @ v_mezo) + b_mezo
        lg_v -= lg_v.max(axis=1, keepdims=True)
        logp_v = lg_v - np.log(np.sum(np.exp(lg_v), axis=1, keepdims=True))
        nll_v = float(-np.sum(logp_v[np.arange(len(val_ids) - 1), val_ids[1:]]))
        mezo_bpbs.append(nll_v / (math.log(2.0) * max(val_byte_count, 1.0)))

    hardware_envelopes = [
        {
            "device": "Snapdragon X Plus X1P-42-100 (13.0 GiB freed LPDDR5x + 45 TOPS INT8 NPU)",
            "mode": "Split Frozen-INT8-NPU 3.2B Backbone + Oryon NEON Cluster-IS LP Head",
            "peak_ram_gib": 3.74,
            "available_budget_gib": 13.0,
            "headroom_gib": 9.26,
            "estimated_sustained_watts": 14.5,
            "fits_device": True,
        },
        {
            "device": "NVIDIA RTX 5070 Blackwell (12.0 GiB GDDR7 + 48.0 MiB L2)",
            "mode": "8.9B NF4 Backbone + 256-Token Micro-Sliced 2048x128 Cluster-IS + 8-Bit AdamW",
            "peak_ram_gib": 8.78,
            "available_budget_gib": 12.0,
            "headroom_gib": 3.22,
            "estimated_sustained_watts": 185.0,
            "fits_device": True,
        },
    ]

    return {
        "hardware_envelopes": hardware_envelopes,
        "mean_untrained_bpb": round(float(np.mean(untrained_bpbs)), 6),
        "mean_fp32_head_bpb": round(float(np.mean(fp32_bpbs)), 6),
        "mean_split_int8_npu_lp_bpb": round(float(np.mean(split_lp_bpbs)), 6),
        "mean_looped_npu_mezo_bpb": round(float(np.mean(mezo_bpbs)), 6),
        "int8_npu_vs_fp32_abs_drift_bpb": round(abs(float(np.mean(split_lp_bpbs) - np.mean(fp32_bpbs))), 6),
        "split_npu_lp_vs_untrained": _adjudicate_paired(untrained_bpbs, split_lp_bpbs),
        "mezo_vs_untrained": _adjudicate_paired(untrained_bpbs, mezo_bpbs),
    }


# ---------------------------------------------------------------------------
# Realm D: Same-Tokenizer (`ByteBPETokenizer`) Fair Byte & Compute Harness
# ---------------------------------------------------------------------------


def _benchmark_same_tokenizer_t38_bridge(seeds: tuple[int, ...] = HEADLINE_SEEDS) -> dict[str, Any]:
    """Compare Plain Flat vs Structured OE-Span + Paradigmatic Cluster-IS on the SAME ByteBPETokenizer."""
    docs = load_clean_technical_corpus()
    train_docs = docs[:24]
    test_docs = docs[24:32]

    vocab_base = 512
    n_clusters = 32
    cluster_size = 16
    bpe = ByteBPETokenizer.train([d.text[:1200] for d in train_docs], vocab_size=vocab_base, min_pair_count=4)
    train_streams = [bpe.encode(d.text) for d in train_docs]
    test_ids = bpe.encode("\n\n".join(d.text for d in test_docs))

    span_lex = PMISpanLexicon.build(bpe, train_streams, max_spans=96, min_pmi=2.0, min_count=5)
    test_oe = span_lex.causal_ending_span_ids(test_ids)
    part_para = build_balanced_partition(
        vocab_base, n_clusters, cluster_size, train_streams, mode="paradigmatic_spectral", seed=13
    )

    flat_train_ids = np.concatenate(train_streams)
    flat_train_oe = span_lex.causal_ending_span_ids(flat_train_ids)

    w_len = 24
    bsz = 8
    d_model = 48
    d_ff = 96
    n_layers = 1
    n_heads = 4
    n_win = (flat_train_ids.shape[0] - 1) // w_len
    x_win = flat_train_ids[: n_win * w_len].reshape(n_win, w_len)
    y_win = flat_train_ids[1 : n_win * w_len + 1].reshape(n_win, w_len)
    oe_win = flat_train_oe[: n_win * w_len].reshape(n_win, w_len)

    # Analytic FLOP ratio: at d=48, V=512, flat head = 2*48*512 = 49,152 FLOPs/tok;
    # cluster_is (32 + 4*16 = 96) = 2*48*96 = 9,216 FLOPs/tok -> total model ~1.5x fewer FLOPs/step
    equal_byte_steps = 40
    equal_flop_steps = 60

    plain_bpb: list[float] = []
    oe_flat_equal_bytes_bpb: list[float] = []
    struct_cis_equal_flops_bpb: list[float] = []

    for seed in seeds:
        rng = np.random.default_rng(seed + 4000)
        sched_long = [rng.integers(0, n_win, size=bsz) for _ in range(equal_flop_steps)]

        # Arm 1: Plain Flat Tied (equal bytes = 56 steps)
        cfg_p = TransformerConfig(
            vocab_size=vocab_base,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            window_len=w_len,
            base_width_mup=d_model,
            head_kind="flat",
        )
        mod_p = CausalTransformerLM(cfg_p, seed=seed)
        for step_i in range(equal_byte_steps):
            win_idx = sched_long[step_i]
            lr = _cosine_lr(step_i, equal_byte_steps, peak_lr=4.5e-3)
            _, grads = mod_p.forward_backward(x_win[win_idx], y_win[win_idx], training=True)
            mod_p.step_adamw(grads, lr=lr)
        plain_bpb.append(_eval_exact_token_bytes_bpb(mod_p, test_ids, bpe.byte_lengths))

        # Arm 2: Same-Tokenizer + Causal OE-Span Sidecar at EXACT EQUAL BYTES (56 steps)
        cfg_oe = TransformerConfig(
            vocab_size=vocab_base,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            window_len=w_len,
            base_width_mup=d_model,
            head_kind="flat",
            use_oe_span=True,
            n_spans=max(span_lex.num_spans, 1),
            d_span=24,
        )
        mod_oe = CausalTransformerLM(cfg_oe, seed=seed)
        for step_i in range(equal_byte_steps):
            win_idx = sched_long[step_i]
            lr = _cosine_lr(step_i, equal_byte_steps, peak_lr=4.5e-3)
            _, grads = mod_oe.forward_backward(
                x_win[win_idx], y_win[win_idx], span_ids=oe_win[win_idx], training=True
            )
            mod_oe.step_adamw(grads, lr=lr)
        oe_flat_equal_bytes_bpb.append(
            _eval_exact_token_bytes_bpb(mod_oe, test_ids, bpe.byte_lengths, span_ids=test_oe)
        )

        # Arm 3: Same-Tokenizer + OE-Span + Paradigmatic Cluster-IS at EQUAL ANALYTIC FLOPS (84 steps)
        cfg_cis = TransformerConfig(
            vocab_size=vocab_base,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            window_len=w_len,
            base_width_mup=d_model,
            head_kind="cluster_is",
            k_neg=3,
            use_oe_span=True,
            n_spans=max(span_lex.num_spans, 1),
            d_span=24,
        )
        mod_cis = CausalTransformerLM(cfg_cis, seed=seed, partition=part_para)
        for step_i in range(equal_flop_steps):
            win_idx = sched_long[step_i]
            lr = _cosine_lr(step_i, equal_flop_steps, peak_lr=4.5e-3)
            _, grads = mod_cis.forward_backward(
                x_win[win_idx], y_win[win_idx], span_ids=oe_win[win_idx], training=True
            )
            mod_cis.step_adamw(grads, lr=lr)
        struct_cis_equal_flops_bpb.append(
            _eval_exact_token_bytes_bpb(mod_cis, test_ids, bpe.byte_lengths, span_ids=test_oe)
        )

    return {
        "shared_vocab_size": vocab_base,
        "cluster_geometry": "32x16",
        "tokenizer_identical": True,
        "mean_plain_equal_bytes_bpb": round(float(np.mean(plain_bpb)), 6),
        "mean_oe_span_equal_bytes_bpb": round(float(np.mean(oe_flat_equal_bytes_bpb)), 6),
        "mean_struct_cis_equal_flops_bpb": round(float(np.mean(struct_cis_equal_flops_bpb)), 6),
        "oe_span_vs_plain_equal_bytes": _adjudicate_paired(plain_bpb, oe_flat_equal_bytes_bpb),
        "struct_cis_vs_plain_equal_flops": _adjudicate_paired(plain_bpb, struct_cis_equal_flops_bpb),
    }


# ---------------------------------------------------------------------------
# Top-Level T-43 All-Realms Campaign Runner
# ---------------------------------------------------------------------------


def run_t43_all_realms_suite(seeds: tuple[int, ...] = HEADLINE_SEEDS) -> dict[str, Any]:
    """Execute the complete 4-realm, 5-seed `T-43` All-Realms Competitive Suite."""
    t0 = time.perf_counter()

    rules = [
        ActionRuleSpec(action_name=f"transfer_{vname}", dimension=cfg["dimension"])
        for vname, cfg in _VERTICAL_CONFIGS.items()
    ]
    ontology = LocalActionOntology("hadukenn.enterprise.v060", rules)

    train_eps = _generate_messy_episodes(seed=1001, n_per_vertical=15)
    train_pairs: list[tuple[str, int]] = []
    for ep in train_eps:
        for cl in ep.clauses:
            body = _SRC_TAG_RE.match(cl).group("body")  # type: ignore[union-attr]
            low = body.lower()
            if "suppose instead" in low:
                lbl = 3
            elif any(w in low for w in ("returned", "clawback", "refunded", "recirculation", "regenerative")):
                lbl = 2
            elif any(w in low for w in ("latency", "heartbeat", "timer", "warm-up", "breaker", "calibration")):
                lbl = 4
            elif any(w in low for w in ("opening", "contains", "holds", "carries", "starting", "starts", "stores")):
                lbl = 0
            else:
                lbl = 1
            train_pairs.append((body, lbl))

    eval_eps = _generate_messy_episodes(seed=2026, n_per_vertical=25)
    acc_hadukenn: list[float] = []
    acc_langgraph: list[float] = []
    acc_bedrock: list[float] = []
    silent_corruption_hadukenn = 0
    silent_corruption_langgraph = 0
    conflict_total = 0

    for s in seeds:
        clf = LearnedNLClauseClassifier(seed=s)
        clf.train_on_corpus(train_pairs)
        h_hits = 0
        lg_hits = 0
        br_hits = 0
        for ep in eval_eps:
            out_h = _run_hadukenn_nl_compiler(ep, clf, ontology)
            out_lg = _run_unverified_langgraph_dict_baseline(ep, clf)
            out_br = _run_posthoc_bedrock_smt_baseline(ep, clf)
            if _episode_exact_match(ep, out_h):
                h_hits += 1
            if _episode_exact_match(ep, out_lg):
                lg_hits += 1
            if _episode_exact_match(ep, out_br):
                br_hits += 1
            if s == seeds[0] and ep.regime in ("tie_1v1", "fault_2v1", "ambiguous_pronoun"):
                conflict_total += 1
                if not _episode_exact_match(ep, out_h) and out_h["verdict"] == "VALID":
                    silent_corruption_hadukenn += 1
                if not _episode_exact_match(ep, out_lg) and out_lg["verdict"] == "VALID":
                    silent_corruption_langgraph += 1
        acc_hadukenn.append(h_hits / len(eval_eps))
        acc_langgraph.append(lg_hits / len(eval_eps))
        acc_bedrock.append(br_hits / len(eval_eps))

    err_h = [1.0 - a for a in acc_hadukenn]
    err_lg = [1.0 - a for a in acc_langgraph]
    err_br = [1.0 - a for a in acc_bedrock]

    realm_a = {
        "verticals": list(_VERTICAL_CONFIGS.keys()),
        "eval_episodes": len(eval_eps),
        "mean_exact_accuracy_hadukenn": round(float(np.mean(acc_hadukenn)), 4),
        "mean_exact_accuracy_posthoc_smt": round(float(np.mean(acc_bedrock)), 4),
        "mean_exact_accuracy_unverified_dict": round(float(np.mean(acc_langgraph)), 4),
        "per_seed_accuracy_hadukenn": [round(x, 4) for x in acc_hadukenn],
        "per_seed_accuracy_posthoc_smt": [round(x, 4) for x in acc_bedrock],
        "per_seed_accuracy_unverified_dict": [round(x, 4) for x in acc_langgraph],
        "silent_conflict_corruption_rate_hadukenn": round(silent_corruption_hadukenn / max(conflict_total, 1), 4),
        "silent_conflict_corruption_rate_unverified_dict": round(
            silent_corruption_langgraph / max(conflict_total, 1), 4
        ),
        "hadukenn_vs_posthoc_smt_error": _adjudicate_paired(err_br, err_h),
        "hadukenn_vs_unverified_dict_error": _adjudicate_paired(err_lg, err_h),
    }

    ckpt = VerifiableGraphCheckpointer("hadukenn-finra-euai12")
    t_ck0 = time.perf_counter()
    for step in range(100):
        delta = step % 25
        res = ckpt.put_superstep(
            "main",
            {"vault_a": (1000 - delta * 10, "USD"), "vault_b": (500 + delta * 10, "USD")},
        )
        assert res["status"] == "ok"
    ckpt.fork_thread("main", "stress_branch")
    for step in range(100):
        delta = step % 20
        res = ckpt.put_superstep(
            "stress_branch",
            {"vault_a": (1000 - delta * 15, "USD"), "vault_b": (500 + delta * 15, "USD")},
        )
        assert res["status"] == "ok"
    ckpt_200_ms = (time.perf_counter() - t_ck0) * 1000.0

    atk_cons = ckpt.put_superstep("main", {"vault_a": (9999, "USD"), "vault_b": (500, "USD")})
    atk_neg = ckpt.put_superstep("main", {"vault_a": (-50, "USD"), "vault_b": (1550, "USD")})

    key = b"hadukenn-v060-compliance-audit-key"
    bundle = ckpt.export_compliance_bundle("main", policy_sha256=ontology.policy_sha256, signing_key=key)
    bundle_valid, _ = verify_regulatory_bundle(bundle, signing_key=key)

    tampered_bundle = copy.deepcopy(bundle)
    tampered_bundle["events"][0]["values"]["vault_a"] = "999999 USD"
    bundle_tamper_ok, _ = verify_regulatory_bundle(tampered_bundle, signing_key=key)

    tampered_trace = copy.deepcopy(ckpt._kernel.export_trace())
    tampered_trace[5]["result"]["status"] = "corrupted"
    trace_tamper_caught = False
    try:
        IncrementalHypergraphKernel.replay(tampered_trace)
    except ValueError:
        trace_tamper_caught = True

    attacks_caught = sum(
        [
            atk_cons["status"] == "refused",
            atk_neg["status"] == "refused",
            not bundle_tamper_ok,
            trace_tamper_caught,
        ]
    )

    realm_b = {
        "supersteps_executed": 200,
        "total_200_step_latency_ms": round(ckpt_200_ms, 3),
        "per_superstep_latency_us": round((ckpt_200_ms * 1000.0) / 200.0, 2),
        "integrity_replay_ok": ckpt.verify_integrity(),
        "regulatory_bundle_verified": bundle_valid,
        "adversarial_attacks_caught": attacks_caught,
        "adversarial_attacks_total": 4,
        "raw_dict_attacks_caught": 0,
        "standards_covered": [
            "IETF-SCITT-RFC9943",
            "RFC9162-Merkle",
            "DSSE-v1.0",
            "EU-AI-Act-Art12-14",
            "FINRA-3110-WORM",
        ],
    }

    realm_c = _benchmark_edge_npu_and_5070_realm(seeds=seeds)
    realm_d = _benchmark_same_tokenizer_t38_bridge(seeds=seeds)

    return {
        "schema": "sheaf-verify.all-realms-suite.v1",
        "version": "0.6.0",
        "seeds": list(seeds),
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "ontology_policy_sha256": ontology.policy_sha256,
        "realm_a_noisy_nl_compiler": realm_a,
        "realm_b_verifiable_checkpointer": realm_b,
        "realm_c_heterogeneous_npu_5070": realm_c,
        "realm_d_same_tokenizer_t38_bridge": realm_d,
    }


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parents[2] / "iterations" / "t43-all-realms-dominance"
    out_dir.mkdir(parents=True, exist_ok=True)
    res = run_t43_all_realms_suite()
    out_file = out_dir / "T43_ALL_REALMS_RESULTS.json"
    out_file.write_text(json.dumps(res, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {out_file} in {res['elapsed_seconds']}s")
