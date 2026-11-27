"""Standards-native formal action ontology, exact unit lattice, and regulatory compliance bundles.

Pure Python >= 3.11 standard library (`dependencies = []`) subsystem extending `sheaf-verify` to:
  1. `ExactUnitLattice`: Lossless `Fraction` dimensional scaling across volume, currency, mass, time,
     energy, and dimensionless ratios (`bps`, `%`, `1`) with cross-dimension mismatch refusal.
  2. `LocalActionOntology`: Six-verdict deterministic neuro-symbolic action gate (`VALID`, `INVALID`,
     `IMPOSSIBLE`, `TRANSLATION_AMBIGUOUS`, `ABSTAIN_TIE`, `RECOVERABLE_FAULT`) combining linear/interval
     rational preconditions and conservation laws with cellular sheaf multi-witness cohomology.
  3. `emit_regulatory_bundle` & `verify_regulatory_bundle`: Generates and verifies unified compliance
     packages covering IETF SCITT (`RFC 9943`), `RFC 9162` Merkle inclusion, `DSSE v1.0`, EU AI Act
     Article 12/14 lifetime event logging & human-oversight triggers, and FINRA Rule 3110 WORM records.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from sheaf_verify.attestation import (
    build_ledger_receipt,
    dsse_sign,
    dsse_verify,
    intoto_statement_v1,
    validate_strict_json_bytes,
    verify_ledger_receipt,
)
from sheaf_verify.hashing import SheafError, canonical_bytes
from sheaf_verify.sheaf_cohomology import CellularSheaf, SheafEdge, diagnose_contradictions

# Canonical base units per dimension:
#   volume -> L, currency -> USD, mass -> kg, time -> s, energy -> J, ratio -> 1
_UNIT_TABLE: dict[str, tuple[str, Fraction]] = {
    "ul": ("volume", Fraction(1, 1_000_000)),
    "ml": ("volume", Fraction(1, 1_000)),
    "cl": ("volume", Fraction(1, 100)),
    "dl": ("volume", Fraction(1, 10)),
    "l": ("volume", Fraction(1, 1)),
    "litre": ("volume", Fraction(1, 1)),
    "litres": ("volume", Fraction(1, 1)),
    "liter": ("volume", Fraction(1, 1)),
    "liters": ("volume", Fraction(1, 1)),
    "kl": ("volume", Fraction(1_000, 1)),
    "cent": ("currency", Fraction(1, 100)),
    "cents": ("currency", Fraction(1, 100)),
    "usd": ("currency", Fraction(1, 1)),
    "dollar": ("currency", Fraction(1, 1)),
    "dollars": ("currency", Fraction(1, 1)),
    "kusd": ("currency", Fraction(1_000, 1)),
    "musd": ("currency", Fraction(1_000_000, 1)),
    "mg": ("mass", Fraction(1, 1_000_000)),
    "g": ("mass", Fraction(1, 1_000)),
    "kg": ("mass", Fraction(1, 1)),
    "t": ("mass", Fraction(1_000, 1)),
    "ms": ("time", Fraction(1, 1_000)),
    "s": ("time", Fraction(1, 1)),
    "sec": ("time", Fraction(1, 1)),
    "seconds": ("time", Fraction(1, 1)),
    "min": ("time", Fraction(60, 1)),
    "minutes": ("time", Fraction(60, 1)),
    "hr": ("time", Fraction(3_600, 1)),
    "hours": ("time", Fraction(3_600, 1)),
    "j": ("energy", Fraction(1, 1)),
    "kj": ("energy", Fraction(1_000, 1)),
    "wh": ("energy", Fraction(3_600, 1)),
    "kwh": ("energy", Fraction(3_600_000, 1)),
    "bps": ("ratio", Fraction(1, 10_000)),
    "pct": ("ratio", Fraction(1, 100)),
    "%": ("ratio", Fraction(1, 100)),
    "1": ("ratio", Fraction(1, 1)),
    "units": ("ratio", Fraction(1, 1)),
}

_CANONICAL_SYMBOL: dict[str, str] = {
    "volume": "L",
    "currency": "USD",
    "mass": "kg",
    "time": "s",
    "energy": "J",
    "ratio": "1",
}

FORMAL_VERDICTS = (
    "VALID",
    "INVALID",
    "IMPOSSIBLE",
    "TRANSLATION_AMBIGUOUS",
    "ABSTAIN_TIE",
    "RECOVERABLE_FAULT",
)


class ExactUnitLattice:
    """Exact rational dimensional unit normalizer and arithmetic verifier."""

    @staticmethod
    def normalize(value: str | int | Fraction, unit: str) -> tuple[Fraction, str, str]:
        """Return `(canonical_value, canonical_unit_symbol, dimension)` in exact `Fraction` form."""
        key = unit.strip().lower()
        if key not in _UNIT_TABLE:
            raise SheafError(f"unknown unit {unit!r} in ExactUnitLattice")
        dim, scale = _UNIT_TABLE[key]
        frac = value if isinstance(value, Fraction) else Fraction(str(value).strip())
        return frac * scale, _CANONICAL_SYMBOL[dim], dim

    @staticmethod
    def add(
        v1: str | int | Fraction, u1: str, v2: str | int | Fraction, u2: str
    ) -> tuple[Fraction, str]:
        c1, sym1, d1 = ExactUnitLattice.normalize(v1, u1)
        c2, _, d2 = ExactUnitLattice.normalize(v2, u2)
        if d1 != d2:
            raise SheafError(f"dimensional mismatch: cannot combine {d1} ({u1}) with {d2} ({u2})")
        return c1 + c2, sym1

    @staticmethod
    def sub(
        v1: str | int | Fraction, u1: str, v2: str | int | Fraction, u2: str
    ) -> tuple[Fraction, str]:
        c1, sym1, d1 = ExactUnitLattice.normalize(v1, u1)
        c2, _, d2 = ExactUnitLattice.normalize(v2, u2)
        if d1 != d2:
            raise SheafError(f"dimensional mismatch: cannot subtract {d2} ({u2}) from {d1} ({u1})")
        return c1 - c2, sym1


@dataclass(frozen=True)
class ActionRuleSpec:
    """Formal precondition & conservation contract for an enterprise or scientific action."""

    action_name: str
    dimension: str
    min_amount: Fraction = Fraction(0, 1)
    max_amount: Fraction | None = None
    allow_negative_post_balance: bool = False
    require_conservation: bool = True
    authorized_roles: tuple[str, ...] = ("operator", "auditor", "autonomous_tier1")


@dataclass(frozen=True)
class ActionVerdict:
    """Six-state formal + topological verdict with human-oversight escalation flag."""

    verdict: str
    approved: bool
    human_escalation_required: bool
    canonical_unit: str
    resolved_initial_src: str | None
    post_src_balance: str | None
    post_dst_balance: str | None
    isolated_fault_sources: tuple[str, ...]
    reason: str
    policy_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "approved": self.approved,
            "canonical_unit": self.canonical_unit,
            "human_escalation_required": self.human_escalation_required,
            "isolated_fault_sources": list(self.isolated_fault_sources),
            "policy_sha256": self.policy_sha256,
            "post_dst_balance": self.post_dst_balance,
            "post_src_balance": self.post_src_balance,
            "reason": self.reason,
            "resolved_initial_src": self.resolved_initial_src,
            "verdict": self.verdict,
        }


class LocalActionOntology:
    """Deterministic local counterpart to cloud SMT guardrails and heavyweight enterprise ontologies."""

    def __init__(self, ontology_id: str, rules: list[ActionRuleSpec]) -> None:
        if not ontology_id.strip():
            raise SheafError("ontology_id must be non-empty")
        self.ontology_id = ontology_id.strip()
        self._rules: dict[str, ActionRuleSpec] = {r.action_name: r for r in rules}
        spec_doc = {
            "ontology_id": self.ontology_id,
            "rules": [
                {
                    "action_name": r.action_name,
                    "allow_negative_post_balance": r.allow_negative_post_balance,
                    "authorized_roles": list(r.authorized_roles),
                    "dimension": r.dimension,
                    "max_amount": str(r.max_amount) if r.max_amount is not None else None,
                    "min_amount": str(r.min_amount),
                    "require_conservation": r.require_conservation,
                }
                for r in sorted(rules, key=lambda x: x.action_name)
            ],
        }
        self.policy_sha256 = hashlib.sha256(canonical_bytes(spec_doc)).hexdigest()

    def evaluate_transfer_action(
        self,
        *,
        action_name: str,
        actor_role: str,
        src_entity: str | None,
        dst_entity: str | None,
        src_witnesses: dict[str, tuple[str | int | Fraction, str]],
        dst_initial: tuple[str | int | Fraction, str],
        transfer_amount: tuple[str | int | Fraction, str],
        claimed_post_src: tuple[str | int | Fraction, str] | None = None,
        claimed_post_dst: tuple[str | int | Fraction, str] | None = None,
    ) -> ActionVerdict:
        """Evaluate an action proposal across ambiguity, sheaf consensus, bounds, and conservation."""
        if action_name not in self._rules:
            return ActionVerdict(
                verdict="IMPOSSIBLE",
                approved=False,
                human_escalation_required=True,
                canonical_unit="?",
                resolved_initial_src=None,
                post_src_balance=None,
                post_dst_balance=None,
                isolated_fault_sources=(),
                reason=f"undeclared action verb {action_name!r}",
                policy_sha256=self.policy_sha256,
            )
        rule = self._rules[action_name]
        canon_sym = _CANONICAL_SYMBOL[rule.dimension]

        # 1. Check translation ambiguity (missing entity binding, self-loop, or empty witnesses)
        if (
            not src_entity
            or not dst_entity
            or src_entity == "?"
            or dst_entity == "?"
            or src_entity == dst_entity
            or not src_witnesses
        ):
            return ActionVerdict(
                verdict="TRANSLATION_AMBIGUOUS",
                approved=False,
                human_escalation_required=True,
                canonical_unit=canon_sym,
                resolved_initial_src=None,
                post_src_balance=None,
                post_dst_balance=None,
                isolated_fault_sources=(),
                reason="unresolved entity pronoun, missing witness, or degenerate self-transfer",
                policy_sha256=self.policy_sha256,
            )

        if actor_role not in rule.authorized_roles:
            return ActionVerdict(
                verdict="IMPOSSIBLE",
                approved=False,
                human_escalation_required=True,
                canonical_unit=canon_sym,
                resolved_initial_src=None,
                post_src_balance=None,
                post_dst_balance=None,
                isolated_fault_sources=(),
                reason=f"actor role {actor_role!r} not permitted for {action_name!r}",
                policy_sha256=self.policy_sha256,
            )

        # 2. Normalize units in exact rational arithmetic
        try:
            norm_witnesses: dict[str, Fraction] = {}
            for wid, (wval, wunit) in sorted(src_witnesses.items()):
                cval, _, wdim = ExactUnitLattice.normalize(wval, wunit)
                if wdim != rule.dimension:
                    raise SheafError(f"witness {wid} has dimension {wdim}, expected {rule.dimension}")
                norm_witnesses[wid] = cval
            dst_val, _, dst_dim = ExactUnitLattice.normalize(dst_initial[0], dst_initial[1])
            amt_val, _, amt_dim = ExactUnitLattice.normalize(transfer_amount[0], transfer_amount[1])
            if dst_dim != rule.dimension or amt_dim != rule.dimension:
                raise SheafError("destination or transfer dimension mismatch")
        except SheafError as exc:
            return ActionVerdict(
                verdict="IMPOSSIBLE",
                approved=False,
                human_escalation_required=True,
                canonical_unit=canon_sym,
                resolved_initial_src=None,
                post_src_balance=None,
                post_dst_balance=None,
                isolated_fault_sources=(),
                reason=str(exc),
                policy_sha256=self.policy_sha256,
            )

        # 3. Cellular Sheaf multi-witness cohomology gate over source witnesses
        fault_sources: tuple[str, ...] = ()
        if len(norm_witnesses) == 1:
            resolved_src = next(iter(norm_witnesses.values()))
        else:
            w_ids = list(norm_witnesses.keys())
            vertex_dims = {w: 1 for w in w_ids}
            edges = [
                SheafEdge(
                    id=f"e_{i}_{j}",
                    source=w_ids[i],
                    target=w_ids[j],
                    edge_dim=1,
                    source_map=[[1.0]],
                    target_map=[[1.0]],
                )
                for i in range(len(w_ids))
                for j in range(i + 1, len(w_ids))
            ]
            sheaf = CellularSheaf(vertex_dims=vertex_dims, edges=edges)
            w_report = diagnose_contradictions(
                sheaf,
                assignment={w: [float(norm_witnesses[w])] for w in w_ids},
                tol=1e-6,
            )
            if w_report.regime == "ABSTAIN_TIE":
                return ActionVerdict(
                    verdict="ABSTAIN_TIE",
                    approved=False,
                    human_escalation_required=True,
                    canonical_unit=canon_sym,
                    resolved_initial_src=None,
                    post_src_balance=None,
                    post_dst_balance=None,
                    isolated_fault_sources=(),
                    reason="1-vs-1 or symmetric witness tie detected via cellular sheaf coboundary; abstaining",
                    policy_sha256=self.policy_sha256,
                )
            fault_sources = tuple(w_report.faulty_vertices)
            clean_vals = [v for k, v in norm_witnesses.items() if k not in fault_sources]
            resolved_src = clean_vals[0]

        # 4. Precondition & bound checks (`IMPOSSIBLE` if domain invariants forbid transition)
        if amt_val < rule.min_amount or (rule.max_amount is not None and amt_val > rule.max_amount):
            return ActionVerdict(
                verdict="IMPOSSIBLE",
                approved=False,
                human_escalation_required=False,
                canonical_unit=canon_sym,
                resolved_initial_src=str(resolved_src),
                post_src_balance=None,
                post_dst_balance=None,
                isolated_fault_sources=fault_sources,
                reason=f"transfer amount {amt_val} {canon_sym} violates policy bounds",
                policy_sha256=self.policy_sha256,
            )

        post_src = resolved_src - amt_val
        post_dst = dst_val + amt_val
        if not rule.allow_negative_post_balance and post_src < 0:
            return ActionVerdict(
                verdict="IMPOSSIBLE",
                approved=False,
                human_escalation_required=False,
                canonical_unit=canon_sym,
                resolved_initial_src=str(resolved_src),
                post_src_balance=str(post_src),
                post_dst_balance=str(post_dst),
                isolated_fault_sources=fault_sources,
                reason=f"insufficient conserved balance: post_src={post_src} {canon_sym} < 0",
                policy_sha256=self.policy_sha256,
            )

        # 5. Claimed output conservation & exact equality check (`INVALID` if proposed post-state drifts)
        if claimed_post_src is not None or claimed_post_dst is not None:
            try:
                c_src = (
                    ExactUnitLattice.normalize(claimed_post_src[0], claimed_post_src[1])[0]
                    if claimed_post_src is not None
                    else post_src
                )
                c_dst = (
                    ExactUnitLattice.normalize(claimed_post_dst[0], claimed_post_dst[1])[0]
                    if claimed_post_dst is not None
                    else post_dst
                )
            except SheafError as exc:
                return ActionVerdict(
                    verdict="INVALID",
                    approved=False,
                    human_escalation_required=False,
                    canonical_unit=canon_sym,
                    resolved_initial_src=str(resolved_src),
                    post_src_balance=str(post_src),
                    post_dst_balance=str(post_dst),
                    isolated_fault_sources=fault_sources,
                    reason=str(exc),
                    policy_sha256=self.policy_sha256,
                )
            if rule.require_conservation and (c_src + c_dst) != (resolved_src + dst_val):
                return ActionVerdict(
                    verdict="INVALID",
                    approved=False,
                    human_escalation_required=False,
                    canonical_unit=canon_sym,
                    resolved_initial_src=str(resolved_src),
                    post_src_balance=str(post_src),
                    post_dst_balance=str(post_dst),
                    isolated_fault_sources=fault_sources,
                    reason="conservation law violated: claimed post-sum differs from initial sum",
                    policy_sha256=self.policy_sha256,
                )
            if c_src != post_src or c_dst != post_dst:
                return ActionVerdict(
                    verdict="INVALID",
                    approved=False,
                    human_escalation_required=False,
                    canonical_unit=canon_sym,
                    resolved_initial_src=str(resolved_src),
                    post_src_balance=str(post_src),
                    post_dst_balance=str(post_dst),
                    isolated_fault_sources=fault_sources,
                    reason=f"claimed post-balances ({c_src}, {c_dst}) disagree with ({post_src}, {post_dst})",
                    policy_sha256=self.policy_sha256,
                )

        verdict_code = "RECOVERABLE_FAULT" if fault_sources else "VALID"
        return ActionVerdict(
            verdict=verdict_code,
            approved=True,
            human_escalation_required=False,
            canonical_unit=canon_sym,
            resolved_initial_src=str(resolved_src),
            post_src_balance=str(post_src),
            post_dst_balance=str(post_dst),
            isolated_fault_sources=fault_sources,
            reason=(
                f"isolated faulty source(s) {list(fault_sources)} and verified conserved transition"
                if fault_sources
                else "all bounds, sheaf witness checks, and conservation laws satisfied"
            ),
            policy_sha256=self.policy_sha256,
        )


def emit_regulatory_bundle(
    *,
    system_id: str,
    trace_head_sha256: str,
    state_digest_sha256: str,
    policy_sha256: str,
    events: list[dict[str, Any]],
    created_at: str,
    signing_key: bytes,
    retention_months: int = 72,
) -> dict[str, Any]:
    """Emit a unified SCITT (RFC 9943) + EU AI Act Art. 12/14 + FINRA 3110 WORM compliance bundle."""
    if retention_months < 6:
        raise SheafError("EU AI Act Article 19/26(6) floor requires retention_months >= 6")
    if not events:
        raise SheafError("compliance bundle requires at least one recorded event")

    escalations = [
        idx
        for idx, ev in enumerate(events)
        if ev.get("human_escalation_required") is True
        or ev.get("verdict") in {"ABSTAIN_TIE", "TRANSLATION_AMBIGUOUS"}
    ]
    refusals = [
        idx for idx, ev in enumerate(events) if ev.get("approved") is False or ev.get("status") == "error"
    ]

    events_bytes = canonical_bytes(events)
    events_sha = hashlib.sha256(events_bytes).hexdigest()

    # Build a 1-leaf or multi-event RFC 9162 Merkle ledger receipt
    ledger_like_records = []
    prev = None
    for idx, ev in enumerate(events):
        rec_body = {
            "created_at": created_at,
            "event_sha256": hashlib.sha256(canonical_bytes(ev)).hexdigest(),
            "format": "sheaf-verify.ledger.v1",
            "gate_id": f"EVENT_{idx}",
            "prev_sha256": prev,
            "seq": idx,
            "status": "ESCALATED" if idx in escalations else ("REFUSED" if idx in refusals else "PASSED"),
        }
        rec_sha = hashlib.sha256(canonical_bytes(rec_body)).hexdigest()
        rec_body["record_sha256"] = rec_sha
        prev = rec_sha
        ledger_like_records.append(rec_body)

    scitt_receipt = build_ledger_receipt(
        ledger_like_records, seq=len(ledger_like_records) - 1, key=signing_key
    )
    statement = intoto_statement_v1(
        subjects=[
            {"name": f"{system_id}/events.json", "digest": {"sha256": events_sha}},
            {"name": f"{system_id}/trace_head", "digest": {"sha256": trace_head_sha256}},
            {"name": f"{system_id}/state_digest", "digest": {"sha256": state_digest_sha256}},
        ],
        predicate_type="https://hadukenn.org/compliance/scitt-euai12-finra3110/v1",
        predicate={
            "created_at": created_at,
            "eu_ai_act_article_12": {
                "automatic_lifetime_logging": True,
                "escalated_event_indices": escalations,
                "human_oversight_article_14_ready": True,
                "minimum_retention_months": retention_months,
                "refused_event_indices": refusals,
                "total_events_logged": len(events),
            },
            "finra_rule_3110_worm": {
                "books_and_records_reconstructible": True,
                "policy_snapshot_sha256": policy_sha256,
                "retention_years_floor": retention_months // 12,
                "supervisory_escalation_count": len(escalations),
                "worm_integrity_sha256": events_sha,
            },
            "ietf_scitt_rfc9943": {
                "merkle_tree_alg": "RFC9162-SHA256",
                "receipt_tree_size": len(ledger_like_records),
                "signed_statement_alg": "DSSE-v1-HMAC-SHA256",
            },
            "system_id": system_id,
        },
    )
    dsse_env = dsse_sign(statement, key=signing_key)
    bundle = {
        "bundle_schema": "sheaf-verify.regulatory-bundle.v1",
        "created_at": created_at,
        "dsse_statement_envelope": dsse_env,
        "events": events,
        "events_sha256": events_sha,
        "policy_sha256": policy_sha256,
        "scitt_merkle_receipt": scitt_receipt,
        "state_digest_sha256": state_digest_sha256,
        "system_id": system_id,
        "trace_head_sha256": trace_head_sha256,
    }
    validate_strict_json_bytes(canonical_bytes(bundle))
    return bundle


def verify_regulatory_bundle(bundle: dict[str, Any], *, signing_key: bytes) -> tuple[bool, list[str]]:
    """Cryptographically and structurally verify a regulatory compliance bundle."""
    errors: list[str] = []
    if bundle.get("bundle_schema") != "sheaf-verify.regulatory-bundle.v1":
        errors.append("unexpected bundle_schema")
        return False, errors

    events = bundle.get("events")
    if not isinstance(events, list) or not events:
        errors.append("events must be a non-empty list")
        return False, errors

    recomputed_events_sha = hashlib.sha256(canonical_bytes(events)).hexdigest()
    if recomputed_events_sha != bundle.get("events_sha256"):
        errors.append("events_sha256 mismatch (tampered event log)")

    try:
        stmt = dsse_verify(bundle["dsse_statement_envelope"], key=signing_key)
        subj_map = {s["name"].split("/")[-1]: s["digest"]["sha256"] for s in stmt.get("subject", [])}
        if subj_map.get("events.json") != recomputed_events_sha:
            errors.append("DSSE subject events.json digest mismatch")
        if subj_map.get("trace_head") != bundle.get("trace_head_sha256"):
            errors.append("DSSE subject trace_head digest mismatch")
        if subj_map.get("state_digest") != bundle.get("state_digest_sha256"):
            errors.append("DSSE subject state_digest mismatch")
    except Exception as exc:
        errors.append(f"DSSE verification failed: {exc}")

    ok_rcpt, rcpt_errs = verify_ledger_receipt(bundle.get("scitt_merkle_receipt", {}), key=signing_key)
    if not ok_rcpt:
        errors.extend(rcpt_errs)

    return len(errors) == 0, errors
