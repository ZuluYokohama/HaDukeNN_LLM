"""Exact unit normalisation and a deterministic transfer-action gate.

Pure standard library. Two pieces:

``ExactUnitLattice``
    Normalises a ``(value, unit)`` pair to the canonical unit of its dimension in exact
    ``Fraction`` arithmetic (volume -> L, currency -> USD, mass -> kg, time -> s,
    energy -> J, ratio -> 1) and refuses to combine different dimensions.

``LocalActionOntology``
    Checks a proposed transfer of a conserved quantity from one entity to another against a
    declared rule and returns one of six verdicts::

        VALID                  every check passed
        RECOVERABLE_FAULT      passed after setting aside exactly one disagreeing source witness
        INVALID                the claimed post-state breaks conservation or disagrees with the
                               recomputed post-state
        IMPOSSIBLE             undeclared action, unauthorised role, malformed or mismatched
                               units, out-of-bounds amount, or a negative balance the rule forbids
        TRANSLATION_AMBIGUOUS  missing or degenerate entity binding, or no source witness
        ABSTAIN_TIE            source witnesses disagree and no single outlier explains it

Witness consensus is computed exactly. Treating every pair of witnesses as an edge of a complete
graph, a disagreement is recoverable only when one witness is the endpoint of every disagreeing
edge and of at least two of them, i.e. exactly one witness differs from at least two witnesses
that agree with each other. Every other disagreement (1-vs-1, 2-vs-2, three distinct values, ...)
is a tie. This is a majority rule: it assumes the agreeing witnesses are independent and honest,
and it cannot detect a poisoned majority.

Adapted from the v0.6.0 drop (``compliance.py``). The regulatory bundle emitter from that file is
intentionally not included here.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from sheaf_verify.hashing import SheafError, canonical_bytes

Quantity = tuple["str | int | Fraction", str]

# unit -> (dimension, scale to the canonical unit of that dimension)
_UNIT_TABLE: dict[str, tuple[str, Fraction]] = {
    "ul": ("volume", Fraction(1, 1_000_000)),
    "ml": ("volume", Fraction(1, 1_000)),
    "cl": ("volume", Fraction(1, 100)),
    "dl": ("volume", Fraction(1, 10)),
    "l": ("volume", Fraction(1)),
    "litre": ("volume", Fraction(1)),
    "litres": ("volume", Fraction(1)),
    "liter": ("volume", Fraction(1)),
    "liters": ("volume", Fraction(1)),
    "kl": ("volume", Fraction(1_000)),
    "cent": ("currency", Fraction(1, 100)),
    "cents": ("currency", Fraction(1, 100)),
    "usd": ("currency", Fraction(1)),
    "dollar": ("currency", Fraction(1)),
    "dollars": ("currency", Fraction(1)),
    "kusd": ("currency", Fraction(1_000)),
    "musd": ("currency", Fraction(1_000_000)),
    "mg": ("mass", Fraction(1, 1_000_000)),
    "g": ("mass", Fraction(1, 1_000)),
    "kg": ("mass", Fraction(1)),
    "t": ("mass", Fraction(1_000)),
    "ms": ("time", Fraction(1, 1_000)),
    "s": ("time", Fraction(1)),
    "sec": ("time", Fraction(1)),
    "seconds": ("time", Fraction(1)),
    "min": ("time", Fraction(60)),
    "minutes": ("time", Fraction(60)),
    "hr": ("time", Fraction(3_600)),
    "hours": ("time", Fraction(3_600)),
    "j": ("energy", Fraction(1)),
    "kj": ("energy", Fraction(1_000)),
    "wh": ("energy", Fraction(3_600)),
    "kwh": ("energy", Fraction(3_600_000)),
    "bps": ("ratio", Fraction(1, 10_000)),
    "pct": ("ratio", Fraction(1, 100)),
    "%": ("ratio", Fraction(1, 100)),
    "1": ("ratio", Fraction(1)),
    "units": ("ratio", Fraction(1)),
}

CANONICAL_SYMBOL: dict[str, str] = {
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


def _to_fraction(value: Any) -> Fraction:
    if isinstance(value, bool):
        raise SheafError("boolean is not a quantity")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, str):
        try:
            return Fraction(value.strip())
        except (ValueError, ZeroDivisionError) as error:
            raise SheafError(f"malformed quantity {value!r}") from error
    raise SheafError(f"quantity must be str, int or Fraction, not {type(value).__name__}")


class ExactUnitLattice:
    """Exact rational unit normaliser."""

    @staticmethod
    def normalize(value: str | int | Fraction, unit: str) -> tuple[Fraction, str, str]:
        """Return ``(canonical_value, canonical_unit_symbol, dimension)``."""
        if not isinstance(unit, str):
            raise SheafError("unit must be a string")
        key = unit.strip().lower()
        if key not in _UNIT_TABLE:
            raise SheafError(f"unknown unit {unit!r}")
        dim, scale = _UNIT_TABLE[key]
        return _to_fraction(value) * scale, CANONICAL_SYMBOL[dim], dim

    @staticmethod
    def add(v1: str | int | Fraction, u1: str, v2: str | int | Fraction, u2: str) -> tuple[Fraction, str]:
        c1, sym, d1 = ExactUnitLattice.normalize(v1, u1)
        c2, _, d2 = ExactUnitLattice.normalize(v2, u2)
        if d1 != d2:
            raise SheafError(f"dimensional mismatch: cannot combine {d1} ({u1}) with {d2} ({u2})")
        return c1 + c2, sym

    @staticmethod
    def sub(v1: str | int | Fraction, u1: str, v2: str | int | Fraction, u2: str) -> tuple[Fraction, str]:
        c1, sym, d1 = ExactUnitLattice.normalize(v1, u1)
        c2, _, d2 = ExactUnitLattice.normalize(v2, u2)
        if d1 != d2:
            raise SheafError(f"dimensional mismatch: cannot subtract {d2} ({u2}) from {d1} ({u1})")
        return c1 - c2, sym


def _normalize_in(quantity: Any, dimension: str, label: str) -> Fraction:
    if not isinstance(quantity, tuple) or len(quantity) != 2:
        raise SheafError(f"{label} must be a (value, unit) pair")
    value, dim_seen = ExactUnitLattice.normalize(quantity[0], quantity[1])[0::2]
    if dim_seen != dimension:
        raise SheafError(f"{label} has dimension {dim_seen}, expected {dimension}")
    return value


def witness_consensus(values: dict[str, Fraction]) -> tuple[str, Fraction | None, tuple[str, ...]]:
    """Return ``(regime, resolved_value, faulty_witnesses)`` for exact witness values.

    ``regime`` is ``CONSISTENT``, ``RECOVERABLE_FAULT`` or ``ABSTAIN_TIE``.
    """
    if not values:
        raise SheafError("witness consensus needs at least one witness")
    groups: dict[Fraction, list[str]] = {}
    for wid in sorted(values):
        groups.setdefault(values[wid], []).append(wid)
    if len(groups) == 1:
        return "CONSISTENT", next(iter(groups)), ()
    if len(groups) == 2:
        (va, ga), (vb, gb) = sorted(groups.items(), key=lambda item: len(item[1]))
        if len(ga) == 1 and len(gb) >= 2:
            return "RECOVERABLE_FAULT", vb, (ga[0],)
    return "ABSTAIN_TIE", None, ()


@dataclass(frozen=True)
class ActionRuleSpec:
    """Preconditions and conservation contract for one action, in canonical units."""

    action_name: str
    dimension: str
    min_amount: Fraction = Fraction(0)
    max_amount: Fraction | None = None
    allow_negative_post_balance: bool = False
    require_conservation: bool = True
    authorized_roles: tuple[str, ...] = ("operator", "auditor", "autonomous_tier1")


@dataclass(frozen=True)
class ActionVerdict:
    """One of ``FORMAL_VERDICTS`` plus the recomputed balances and a human-review flag."""

    verdict: str
    approved: bool
    human_escalation_required: bool
    canonical_unit: str
    reason: str
    policy_sha256: str
    resolved_initial_src: str | None = None
    post_src_balance: str | None = None
    post_dst_balance: str | None = None
    isolated_fault_sources: tuple[str, ...] = field(default_factory=tuple)

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
    """Deterministic gate for transfers of a conserved quantity between two entities."""

    def __init__(self, ontology_id: str, rules: list[ActionRuleSpec]) -> None:
        if not isinstance(ontology_id, str) or not ontology_id.strip():
            raise SheafError("ontology_id must be non-empty")
        names = [r.action_name for r in rules]
        if len(set(names)) != len(names):
            raise SheafError("duplicate action_name in rules")
        for rule in rules:
            if rule.dimension not in CANONICAL_SYMBOL:
                raise SheafError(f"rule {rule.action_name!r} has unknown dimension {rule.dimension!r}")
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

    def _verdict(self, verdict: str, unit: str, reason: str, *, escalate: bool, **extra: Any) -> ActionVerdict:
        return ActionVerdict(
            verdict=verdict,
            approved=verdict in {"VALID", "RECOVERABLE_FAULT"},
            human_escalation_required=escalate,
            canonical_unit=unit,
            reason=reason,
            policy_sha256=self.policy_sha256,
            **extra,
        )

    def evaluate_transfer_action(
        self,
        *,
        action_name: str,
        actor_role: str,
        src_entity: str | None,
        dst_entity: str | None,
        src_witnesses: dict[str, Quantity],
        dst_initial: Quantity,
        transfer_amount: Quantity,
        claimed_post_src: Quantity | None = None,
        claimed_post_dst: Quantity | None = None,
    ) -> ActionVerdict:
        """Evaluate one proposed transfer. Never raises on bad input; bad input is a verdict."""
        rule = self._rules.get(action_name)
        if rule is None:
            return self._verdict("IMPOSSIBLE", "?", f"undeclared action {action_name!r}", escalate=True)
        unit = CANONICAL_SYMBOL[rule.dimension]

        if (
            not src_entity
            or not dst_entity
            or src_entity == "?"
            or dst_entity == "?"
            or src_entity == dst_entity
            or not src_witnesses
        ):
            return self._verdict(
                "TRANSLATION_AMBIGUOUS",
                unit,
                "missing or degenerate entity binding, or no source witness",
                escalate=True,
            )

        if actor_role not in rule.authorized_roles:
            return self._verdict(
                "IMPOSSIBLE", unit, f"role {actor_role!r} is not permitted for {action_name!r}", escalate=True
            )

        try:
            witnesses = {
                wid: _normalize_in(q, rule.dimension, f"witness {wid!r}") for wid, q in sorted(src_witnesses.items())
            }
            dst_val = _normalize_in(dst_initial, rule.dimension, "dst_initial")
            amount = _normalize_in(transfer_amount, rule.dimension, "transfer_amount")
            c_src = _normalize_in(claimed_post_src, rule.dimension, "claimed_post_src") if claimed_post_src else None
            c_dst = _normalize_in(claimed_post_dst, rule.dimension, "claimed_post_dst") if claimed_post_dst else None
        except SheafError as error:
            return self._verdict("IMPOSSIBLE", unit, str(error), escalate=True)

        regime, resolved, faults = witness_consensus(witnesses)
        if regime == "ABSTAIN_TIE" or resolved is None:
            return self._verdict(
                "ABSTAIN_TIE", unit, "source witnesses disagree with no single outlier; abstaining", escalate=True
            )
        balances = {"resolved_initial_src": str(resolved), "isolated_fault_sources": faults}

        if amount < rule.min_amount or (rule.max_amount is not None and amount > rule.max_amount):
            return self._verdict(
                "IMPOSSIBLE", unit, f"amount {amount} {unit} is outside the rule bounds", escalate=False, **balances
            )

        post_src = resolved - amount
        post_dst = dst_val + amount
        balances.update(post_src_balance=str(post_src), post_dst_balance=str(post_dst))
        if not rule.allow_negative_post_balance and post_src < 0:
            return self._verdict(
                "IMPOSSIBLE", unit, f"insufficient balance: post_src={post_src} {unit}", escalate=False, **balances
            )

        if c_src is not None or c_dst is not None:
            got_src = post_src if c_src is None else c_src
            got_dst = post_dst if c_dst is None else c_dst
            if rule.require_conservation and got_src + got_dst != resolved + dst_val:
                return self._verdict(
                    "INVALID", unit, "conservation violated: claimed post-sum differs", escalate=False, **balances
                )
            if got_src != post_src or got_dst != post_dst:
                return self._verdict(
                    "INVALID",
                    unit,
                    f"claimed post-balances ({got_src}, {got_dst}) disagree with ({post_src}, {post_dst})",
                    escalate=False,
                    **balances,
                )

        if faults:
            return self._verdict(
                "RECOVERABLE_FAULT", unit, f"set aside disagreeing witness {faults[0]!r}", escalate=False, **balances
            )
        return self._verdict("VALID", unit, "all checks passed", escalate=False, **balances)
