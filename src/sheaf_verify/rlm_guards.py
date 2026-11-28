"""Pure-stdlib SIIP dual-mode verification gate and RLM Appendix A pitfall guards.

Implements:
- ``SiipVerifyGate``: dual-mode speculative/formal commit gate + hesitation cosine damping
  (Paper 1 SIIP Landscape Analysis & ``c0_universal_kernel_v5.py`` ``op_21`` ``pi_6``).
- ``RLMBudgetPolicy`` & ``RLMBudgetState``: recursion depth, sub-call count, token cap,
  batching floor, and reasoning-token headroom governance (Paper 7 App. A/C).
- ``parse_final_directive``: fenced-code-block-aware ``FINAL(...)`` / ``FINAL_VAR(name)``
  parser rejecting empty variables and plan-text non-answers (Paper 7 App. A.5/D).
- ``AntiRepetitionTracker``: stagnant verification-loop circuit breaker (Paper 7 Ex. B.2).
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from sheaf_verify.hashing import SheafError, as_int, as_real

_FORMAL_MODES = frozenset({"formal", "verify"})
_SPECULATIVE_MODES = frozenset({"speculative", "yolo", "sandbox"})

_SCAN_TOKEN_RE = re.compile(r"`+|~{3,}|\b(?:FINAL_VAR|FINAL)\s*\(")
_BLANK_LINE_RE = re.compile(r"\n[ \t]*\r?\n")
_BACKTICK_FENCE_OPEN_RE = re.compile(r"^ {0,3}(`{3,})([^`]*)$")
_TILDE_FENCE_OPEN_RE = re.compile(r"^ {0,3}(~{3,})(.*)$")
_PLAN_TEXT_RE = re.compile(
    r"^(?:"
    r"i(?:'ll|\s+(?:will|am\s+going\s+to|need\s+to|should|plan\s+to))\b|"
    r"let\s+me\b|"
    r"my\s+plan\s+is\b|"
    r"we\s+(?:will|should|need\s+to)\b|"
    r"first,\s*i\s+(?:will|'ll)\b|"
    r"next,\s*i\s+(?:will|'ll)\b|"
    r"step\s+1(?:\s*[:\-]|\s*\.(?:\s|$))|"
    r"plan\s*[:\-]|"
    r"to\s+solve\s+this,\s*i\s+(?:will|'ll)\b"
    r")",
    re.IGNORECASE,
)


class SiipVerifyGate:
    """SIIP dual-mode speculative/formal verification gate with hesitation cosine damping."""

    def __init__(
        self,
        mode: str = "formal",
        *,
        hesitation_cos_threshold: float = -0.25,
        damping_factor: float = 0.5,
    ) -> None:
        thresh = as_real(hesitation_cos_threshold, "hesitation_cos_threshold")
        if not (-1.0 <= thresh <= 1.0):
            raise SheafError(f"hesitation_cos_threshold must lie in [-1, 1], got {thresh}")
        damp = as_real(damping_factor, "damping_factor")
        if not (0.0 < damp < 1.0):
            raise SheafError(f"damping_factor must lie in (0, 1), got {damp}")
        self.hesitation_cos_threshold: float = thresh
        self.damping_factor: float = damp
        self.hesitation_damping_events: int = 0
        self.mode: str = "formal"
        self.set_mode(mode)

    def set_mode(self, mode: str) -> str:
        if not isinstance(mode, str):
            raise SheafError("SiipVerifyGate mode must be a string")
        normalized = mode.strip().lower()
        if normalized in _FORMAL_MODES:
            self.mode = "formal"
        elif normalized in _SPECULATIVE_MODES:
            self.mode = "speculative"
        else:
            raise SheafError(f"unknown SIIP gate mode: {mode!r}")
        return self.mode

    def damp_direction(
        self,
        p_dir: Sequence[float],
        prev_dir: Sequence[float],
    ) -> list[float]:
        """Damp search direction by ``damping_factor`` strictly when ``cos(p_dir, prev_dir) < -0.25``."""
        if isinstance(p_dir, (str, bytes)) or isinstance(prev_dir, (str, bytes)):
            raise SheafError("directions must be numeric sequences")
        p = [as_real(x, "p_dir") for x in p_dir]
        prev = [as_real(x, "prev_dir") for x in prev_dir]
        if not p or len(p) != len(prev):
            raise SheafError("p_dir and prev_dir must be non-empty sequences of equal length")

        scale_p = max(abs(x) for x in p)
        scale_prev = max(abs(y) for y in prev)
        if scale_p == 0.0 or scale_prev == 0.0:
            return list(p)

        p_scaled = [x / scale_p for x in p]
        prev_scaled = [y / scale_prev for y in prev]
        norm_p = math.sqrt(math.fsum(x * x for x in p_scaled))
        norm_prev = math.sqrt(math.fsum(y * y for y in prev_scaled))
        cos_sim = math.fsum(a * b for a, b in zip(p_scaled, prev_scaled, strict=True)) / (norm_p * norm_prev)
        cos_sim = max(-1.0, min(1.0, cos_sim))
        if cos_sim < self.hesitation_cos_threshold:
            self.hesitation_damping_events += 1
            return [x * self.damping_factor for x in p]
        return list(p)

    apply_hesitation_damping = damp_direction

    def verify_commit(
        self,
        f_k: float,
        f_cand: float,
        x_cand: Sequence[float],
    ) -> dict[str, Any]:
        """Verify candidate state: finite in all modes, strict ``f_cand < f_k`` in formal mode."""
        fk = as_real(f_k, "f_k")
        fc = as_real(f_cand, "f_cand")
        if isinstance(x_cand, (str, bytes)):
            raise SheafError("x_cand must be a numeric sequence")
        x_vals = [as_real(v, "x_cand") for v in x_cand]
        if not x_vals:
            raise SheafError("x_cand must be non-empty")

        accepted = (fc < fk) if self.mode == "formal" else True
        return {
            "ok": accepted,
            "passed": accepted,
            "accepted": accepted,
            "mode": self.mode,
            "f_k": fk,
            "f_cand": fc,
        }

    verify_candidate = verify_commit


@dataclass
class RLMBudgetState:
    """Mutable budget usage counters for an RLM session."""

    subcalls: int = 0
    total_tokens: int = 0
    max_depth_seen: int = 0
    repl_steps: int = 0

    @property
    def subcall_count(self) -> int:
        return self.subcalls

    def record_subcall(
        self,
        policy: RLMBudgetPolicy,
        depth: int,
        chars: int,
        tokens: int = 0,
        *,
        max_output_tokens: int | None = None,
        thinking_budget: int | None = None,
    ) -> dict[str, Any]:
        return policy.check_subcall(
            self,
            depth,
            chars,
            tokens,
            max_output_tokens=max_output_tokens,
            thinking_budget=thinking_budget,
        )


@dataclass(frozen=True)
class RLMBudgetPolicy:
    """Hard Appendix A/C budget and batching bounds for recursive LM calls."""

    max_depth: int = 1
    max_subcalls: int = 16
    max_total_tokens: int = 100_000
    min_batch_chars: int = 10_000
    thinking_reserve: int = 1024
    max_repl_steps: int = 64

    def __post_init__(self) -> None:
        if as_int(self.max_depth, "max_depth") < 0:
            raise SheafError("max_depth must be >= 0")
        if as_int(self.max_subcalls, "max_subcalls") < 1:
            raise SheafError("max_subcalls must be >= 1")
        if as_int(self.max_total_tokens, "max_total_tokens") < 1:
            raise SheafError("max_total_tokens must be >= 1")
        if as_int(self.min_batch_chars, "min_batch_chars") < 0:
            raise SheafError("min_batch_chars must be >= 0")
        if as_int(self.thinking_reserve, "thinking_reserve") < 0:
            raise SheafError("thinking_reserve must be >= 0")
        if as_int(self.max_repl_steps, "max_repl_steps") < 1:
            raise SheafError("max_repl_steps must be >= 1")

    def check_subcall(
        self,
        state: RLMBudgetState,
        depth: int,
        chars: int,
        tokens: int = 0,
        *,
        max_output_tokens: int | None = None,
        thinking_budget: int | None = None,
    ) -> dict[str, Any]:
        """Validate depth, batch floor, subcall/token caps, and thinking reserve before mutating state."""
        if not isinstance(state, RLMBudgetState):
            raise SheafError("state must be an RLMBudgetState instance")
        d = as_int(depth, "depth")
        c = as_int(chars, "chars")
        tok = as_int(tokens, "tokens")
        if d < 0 or c < 0 or tok < 0:
            raise SheafError("depth, chars, and tokens must be non-negative")

        if d > self.max_depth:
            raise SheafError(f"RLM depth {d} exceeds max_depth {self.max_depth}")
        if c < self.min_batch_chars:
            raise SheafError(f"RLM subcall batch chars {c} below min_batch_chars {self.min_batch_chars}")
        if state.subcalls + 1 > self.max_subcalls:
            raise SheafError(f"RLM max_subcalls {self.max_subcalls} exceeded")
        if state.total_tokens + tok > self.max_total_tokens:
            raise SheafError(f"RLM max_total_tokens {self.max_total_tokens} exceeded")

        if max_output_tokens is not None or thinking_budget is not None:
            mot = as_int(max_output_tokens, "max_output_tokens") if max_output_tokens is not None else None
            tb = as_int(thinking_budget, "thinking_budget") if thinking_budget is not None else 0
            if tb < 0 or (mot is not None and mot < 0):
                raise SheafError("max_output_tokens and thinking_budget must be non-negative")
            if mot is not None and (mot - tb) < self.thinking_reserve:
                raise SheafError(
                    f"RLM thinking_budget {tb} leaves {mot - tb} < thinking_reserve {self.thinking_reserve}"
                )

        state.subcalls += 1
        state.total_tokens += tok
        if d > state.max_depth_seen:
            state.max_depth_seen = d

        return {
            "ok": True,
            "allowed": True,
            "subcalls": state.subcalls,
            "total_tokens": state.total_tokens,
            "max_depth_seen": state.max_depth_seen,
        }

    def check_repl_step(self, state: RLMBudgetState) -> dict[str, Any]:
        """Enforce ``state.repl_steps + 1 <= max_repl_steps`` before incrementing."""
        if not isinstance(state, RLMBudgetState):
            raise SheafError("state must be an RLMBudgetState instance")
        if state.repl_steps + 1 > self.max_repl_steps:
            raise SheafError(f"RLM max_repl_steps {self.max_repl_steps} exceeded")
        state.repl_steps += 1
        return {
            "ok": True,
            "allowed": True,
            "repl_steps": state.repl_steps,
            "max_repl_steps": self.max_repl_steps,
        }


def _find_matching_inline_span_end(
    text: str,
    start_after_opener: int,
    span_limit: int,
    delim_char: str,
    min_len: int,
    *,
    exact_len: bool,
) -> int | None:
    pos = start_after_opener
    while pos < span_limit:
        if text[pos] == delim_char:
            end = pos
            while end < span_limit and text[end] == delim_char:
                end += 1
            run_len = end - pos
            if (run_len == min_len) if exact_len else (run_len >= min_len):
                return end
            pos = end
        else:
            pos += 1
    return None


def _strip_fenced_code_blocks(text: str) -> str:
    """Remove CommonMark fenced code blocks (tracking fence char/len) while keeping non-fenced text."""
    kept: list[str] = []
    fence_char: str | None = None
    fence_len = 0
    for line in text.splitlines():
        if fence_char is not None:
            stripped = line.lstrip(" ")
            leading_spaces = len(line) - len(stripped)
            if leading_spaces <= 3:
                run_len = 0
                while run_len < len(stripped) and stripped[run_len] == fence_char:
                    run_len += 1
                if run_len >= fence_len and not stripped[run_len:].strip():
                    fence_char = None
                    fence_len = 0
            continue

        bt_match = _BACKTICK_FENCE_OPEN_RE.match(line)
        if bt_match is not None:
            if kept and kept[-1] != "":
                kept.append("")
            fence_char = "`"
            fence_len = len(bt_match.group(1))
            continue

        tl_match = _TILDE_FENCE_OPEN_RE.match(line)
        if tl_match is not None:
            marker = tl_match.group(1)
            info = tl_match.group(2)
            if re.search(r"~{" + str(len(marker)) + r",}\s*$", info) is None:
                if kept and kept[-1] != "":
                    kept.append("")
                fence_char = "~"
                fence_len = len(marker)
                continue

        kept.append(line)
    return "\n".join(kept)


def _extract_balanced_arg(text: str, open_paren_idx: int) -> str | None:
    depth = 1
    idx = open_paren_idx + 1
    while idx < len(text):
        ch = text[idx]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return text[open_paren_idx + 1 : idx]
        idx += 1
    return None


def _is_nonempty_var_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (bytes, bytearray, list, tuple, dict, set, frozenset)):
        return len(value) > 0
    return True


def parse_final_directive(
    text: str,
    namespace: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Parse ``FINAL(...)`` or ``FINAL_VAR(name)`` strictly outside fenced and inline code spans."""
    if not isinstance(text, str):
        raise SheafError("parse_final_directive text must be a str")
    if namespace is not None and not isinstance(namespace, Mapping):
        raise SheafError("namespace must be a Mapping or None")

    outside = _strip_fenced_code_blocks(text)
    n = len(outside)
    pos = 0
    while pos < n:
        match = _SCAN_TOKEN_RE.search(outside, pos)
        if match is None:
            break
        tok = match.group(0)
        if tok[0] == "`" or tok[0] == "~":
            if tok[0] == "`":
                blank_match = _BLANK_LINE_RE.search(outside, match.end())
                span_limit = blank_match.start() if blank_match is not None else n
            else:
                line_end = outside.find("\n", match.end())
                span_limit = line_end if line_end != -1 else n
            span_end = _find_matching_inline_span_end(
                outside,
                match.end(),
                span_limit,
                tok[0],
                len(tok),
                exact_len=(tok[0] == "`"),
            )
            pos = span_end if span_end is not None else match.end()
            continue

        kind = "FINAL_VAR" if tok.startswith("FINAL_VAR") else "FINAL"
        open_idx = match.end() - 1
        arg = _extract_balanced_arg(outside, open_idx)
        if arg is None:
            pos = match.end()
            continue

        if kind == "FINAL_VAR":
            var_name = arg.strip()
            if (
                not var_name.isidentifier()
                or namespace is None
                or var_name not in namespace
                or not _is_nonempty_var_value(namespace[var_name])
            ):
                return {
                    "found": False,
                    "accepted": False,
                    "ok": False,
                    "kind": "FINAL_VAR",
                    "value": None,
                    "var_name": var_name or None,
                }
            return {
                "found": True,
                "accepted": True,
                "ok": True,
                "kind": "FINAL_VAR",
                "value": namespace[var_name],
                "var_name": var_name,
            }

        cleaned = arg.strip()
        if not cleaned or _PLAN_TEXT_RE.match(cleaned):
            return {
                "found": False,
                "accepted": False,
                "ok": False,
                "kind": "FINAL",
                "value": None,
                "var_name": None,
            }
        return {
            "found": True,
            "accepted": True,
            "ok": True,
            "kind": "FINAL",
            "value": cleaned,
            "var_name": None,
        }

    return {
        "found": False,
        "accepted": False,
        "ok": False,
        "kind": None,
        "value": None,
        "var_name": None,
    }


@dataclass(frozen=True)
class AntiRepetitionStatus:
    """Turn status returned by ``AntiRepetitionTracker.record_turn`` (supports attr & key access)."""

    should_stop: bool
    consec_no_new_var: int
    new_vars: tuple[str, ...]
    best_verified_value: Any
    turns: int

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key) and not key.startswith("_"):
            return getattr(self, key)
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        if hasattr(self, key) and not key.startswith("_"):
            return getattr(self, key)
        return default

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and not key.startswith("_") and hasattr(self, key)


class AntiRepetitionTracker:
    """Detects stagnant RLM verification loops with zero newly created REPL variables."""

    def __init__(self, max_consec_no_new_var: int = 3) -> None:
        limit = as_int(max_consec_no_new_var, "max_consec_no_new_var")
        if limit < 1:
            raise SheafError(f"max_consec_no_new_var must be >= 1, got {limit}")
        self.max_consec_no_new_var: int = limit
        self.seen_vars: set[str] = set()
        self.consec_no_new_var: int = 0
        self.best_verified_value: Any = None
        self.turns: int = 0

    def record_turn(
        self,
        output: str = "",
        var_names: Iterable[str] = (),
        *,
        verified_value: Any = None,
    ) -> AntiRepetitionStatus:
        """Record one REPL turn and signal stop after ``max_consec_no_new_var`` stagnant turns."""
        if not isinstance(output, str):
            raise SheafError("output must be a str")
        if isinstance(var_names, (str, bytes)):
            raise SheafError("var_names must be an iterable of variable name strings")

        self.turns += 1
        if verified_value is not None:
            self.best_verified_value = verified_value

        vars_set = {str(name) for name in var_names}
        new_vars = vars_set - self.seen_vars
        if new_vars:
            self.seen_vars.update(new_vars)
            self.consec_no_new_var = 0
        else:
            self.consec_no_new_var += 1

        return AntiRepetitionStatus(
            should_stop=self.consec_no_new_var >= self.max_consec_no_new_var,
            consec_no_new_var=self.consec_no_new_var,
            new_vars=tuple(sorted(new_vars)),
            best_verified_value=self.best_verified_value,
            turns=self.turns,
        )

    step = record_turn
    update = record_turn
