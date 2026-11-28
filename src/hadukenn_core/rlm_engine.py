"""RLM Context-as-Environment Execution Engine (Feature 16, Paper 7 ``arXiv:2512.24601v1``).

Implements:
- ``RLMPromptProfile``, ``RLM_PROMPT_PROFILES``, ``get_rlm_prompt_profile``: per-model
  prompt profiles and batching heuristics for ``gpt5``, ``gemini``, ``qwen``, and ``litert``.
- ``SandboxedPythonREPL``: persistent Python REPL binding ``context`` (and metadata
  ``context_lengths``, ``context_total_length``, ``context_type``) under a dual-layer
  AST + restricted-builtins sandbox with bounded ``stdout`` capture, ``llm_query``,
  and semaphore-bounded async ``llm_query_batch``.
- ``RLMOrchestrator``: multi-turn RLM controller enforcing M2 governance primitives
  (``RLMBudgetPolicy``, ``thinking_reserve``, fenced-code-aware ``parse_final_directive``,
  and ``AntiRepetitionTracker`` stagnant-loop recovery).
- ``oolong_pairs_stitch``: balanced out-of-core chunking and all-pairs ``O(N^2)``
  stitching for arbitrary ``N`` (including ``N=0``, ``N=1``, ``N < n_chunks``, and
  non-divisible ``N``), optionally emitting replayable receipts into
  ``hadukenn_core.exec_bridge.IncrementalHypergraphKernel``.
"""

from __future__ import annotations

import ast
import asyncio
import concurrent.futures
import contextlib
import inspect
import io
import json
import re
import types
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from sheaf_verify.hashing import SheafError, as_int, sha256_hex
from sheaf_verify.rlm_guards import (
    AntiRepetitionTracker,
    RLMBudgetPolicy,
    RLMBudgetState,
    parse_final_directive,
)

ALLOWED_REPL_MODULES: frozenset[str] = frozenset(
    {
        "bisect",
        "collections",
        "copy",
        "datetime",
        "decimal",
        "fractions",
        "functools",
        "heapq",
        "itertools",
        "json",
        "math",
        "operator",
        "re",
        "statistics",
        "string",
        "textwrap",
        "typing",
    }
)

_FORBIDDEN_NAMES: frozenset[str] = frozenset(
    {
        "__builtins__",
        "__cached__",
        "__doc__",
        "__import__",
        "__loader__",
        "__name__",
        "__package__",
        "__spec__",
        "breakpoint",
        "compile",
        "copyright",
        "credits",
        "delattr",
        "dir",
        "eval",
        "exec",
        "exit",
        "getattr",
        "globals",
        "hasattr",
        "help",
        "input",
        "license",
        "locals",
        "memoryview",
        "object",
        "open",
        "quit",
        "setattr",
        "super",
        "type",
        "vars",
    }
)

_FORBIDDEN_ATTRS: frozenset[str] = frozenset(
    {
        "ag_await",
        "ag_code",
        "ag_frame",
        "cr_await",
        "cr_code",
        "cr_frame",
        "f_back",
        "f_builtins",
        "f_code",
        "f_globals",
        "f_locals",
        "func_closure",
        "func_code",
        "func_defaults",
        "func_globals",
        "gi_code",
        "gi_frame",
        "gi_yieldfrom",
        "im_class",
        "im_func",
        "im_self",
        "mro",
    }
)

_FORMAT_ESCAPE_RE = re.compile(r"\{[^{}]*(?:\.__|\._|\[[^\]]*__)")
_FENCE_OPEN_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")

_SAFE_BUILTINS: dict[str, Any] = {
    "ArithmeticError": ArithmeticError,
    "AssertionError": AssertionError,
    "Exception": Exception,
    "False": False,
    "IndexError": IndexError,
    "KeyError": KeyError,
    "LookupError": LookupError,
    "None": None,
    "RuntimeError": RuntimeError,
    "StopIteration": StopIteration,
    "True": True,
    "TypeError": TypeError,
    "ValueError": ValueError,
    "ZeroDivisionError": ZeroDivisionError,
    "abs": abs,
    "all": all,
    "any": any,
    "ascii": ascii,
    "bin": bin,
    "bool": bool,
    "bytearray": bytearray,
    "bytes": bytes,
    "callable": callable,
    "chr": chr,
    "complex": complex,
    "dict": dict,
    "divmod": divmod,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "format": format,
    "frozenset": frozenset,
    "hash": hash,
    "hex": hex,
    "int": int,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "iter": iter,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "next": next,
    "oct": oct,
    "ord": ord,
    "pow": pow,
    "range": range,
    "repr": repr,
    "reversed": reversed,
    "round": round,
    "set": set,
    "slice": slice,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}


@dataclass(frozen=True)
class RLMPromptProfile:
    """Per-model RLM system prompt profile and batch-size heuristics."""

    family: str
    system_prompt: str
    target_batch_chars: int
    min_batch_chars: int
    max_subcalls: int
    thinking_reserve: int
    recommended_max_concurrency: int = 4

    @property
    def model_family(self) -> str:
        return self.family

    @property
    def prompt(self) -> str:
        return self.system_prompt

    def __getitem__(self, key: str) -> Any:
        if key == "prompt":
            return self.system_prompt
        if key == "model_family":
            return self.family
        if hasattr(self, key) and not key.startswith("_"):
            return getattr(self, key)
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        if key in self:
            return self[key]
        return default

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str) or key.startswith("_"):
            return False
        return key in {"prompt", "model_family"} or hasattr(self, key)


RLM_PROMPT_PROFILES: dict[str, RLMPromptProfile] = {
    "gpt5": RLMPromptProfile(
        family="gpt5",
        system_prompt=(
            "You operate inside a persistent Python REPL where the input prompt is bound to the "
            "`context` variable (with metadata `context_type`, `context_lengths`, `context_total_length`). "
            "Never assume `context` fits in your direct token window. Write concise ```repl``` blocks to "
            "filter with regex/slices, batch large slices (~400,000 chars) into `llm_query` or async "
            "`llm_query_batch`, store verified intermediate results in variables, and emit `FINAL(answer)` "
            "or `FINAL_VAR(var_name)` strictly outside fenced code blocks."
        ),
        target_batch_chars=400_000,
        min_batch_chars=50_000,
        max_subcalls=16,
        thinking_reserve=1024,
        recommended_max_concurrency=4,
    ),
    "gemini": RLMPromptProfile(
        family="gemini",
        system_prompt=(
            "Your input resides in the REPL variable `context` (never inlined verbatim). Inspect "
            "`context_type`, `context_lengths`, and `context_total_length` using ```repl``` blocks, "
            "maintain prefix-stable slices of `context`, dispatch large (~350,000-char) chunks concurrently "
            "via `llm_query_batch`, cache verified outputs in REPL variables, and conclude with "
            "`FINAL(answer)` or `FINAL_VAR(var_name)` outside any code fence."
        ),
        target_batch_chars=350_000,
        min_batch_chars=50_000,
        max_subcalls=16,
        thinking_reserve=2048,
        recommended_max_concurrency=4,
    ),
    "qwen": RLMPromptProfile(
        family="qwen",
        system_prompt=(
            "The full payload is stored in REPL variable `context`. CRITICAL: do NOT issue tiny 1-line "
            "`llm_query` calls over `context`; always aggregate slices into ~200,000-char batches using "
            "Python loops inside ```repl``` blocks and invoke `llm_query` or `llm_query_batch` sparingly "
            "within your sub-call budget. Save final answers in a variable and terminate with "
            "`FINAL_VAR(var_name)` or `FINAL(answer)` outside fenced code blocks."
        ),
        target_batch_chars=200_000,
        min_batch_chars=25_000,
        max_subcalls=24,
        thinking_reserve=2048,
        recommended_max_concurrency=4,
    ),
    "litert": RLMPromptProfile(
        family="litert",
        system_prompt=(
            "You are running in a memory-bounded LiteRT edge REPL with input bound to `context`. "
            "Keep stdout brief, process `context` in compact ~32,000-char chunks inside ```repl``` blocks, "
            "avoid redundant sub-calls, accumulate verified results in REPL buffers, and report "
            "`FINAL(answer)` or `FINAL_VAR(var_name)` outside code blocks."
        ),
        target_batch_chars=32_000,
        min_batch_chars=4_000,
        max_subcalls=8,
        thinking_reserve=512,
        recommended_max_concurrency=2,
    ),
}

_MODEL_FAMILY_ALIASES: dict[str, str] = {
    "gpt5": "gpt5",
    "gpt-5": "gpt5",
    "gemini": "gemini",
    "gemini-2.5": "gemini",
    "qwen": "qwen",
    "qwen3": "qwen",
    "litert": "litert",
    "lite-rt": "litert",
}


def get_rlm_prompt_profile(model_family: str) -> RLMPromptProfile:
    """Resolve an ``RLMPromptProfile`` for a supported model family."""
    if not isinstance(model_family, str):
        raise SheafError("model_family must be a string")
    canonical = _MODEL_FAMILY_ALIASES.get(model_family.strip().lower())
    if canonical is None or canonical not in RLM_PROMPT_PROFILES:
        raise SheafError(f"Unknown RLM model family: {model_family!r}")
    return RLM_PROMPT_PROFILES[canonical]


class _SandboxASTValidator(ast.NodeVisitor):
    """Static AST guard blocking unsafe imports, builtins, dunder escapes, and frame attributes."""

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            top = alias.name.split(".")[0]
            if top not in ALLOWED_REPL_MODULES:
                raise ImportError(f"Module {alias.name!r} is not allowed in RLM REPL")
            if alias.asname and (alias.asname.startswith("__") or alias.asname in _FORBIDDEN_NAMES):
                raise PermissionError(f"Restricted import alias {alias.asname!r}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level > 0 or not node.module:
            raise ImportError("Relative imports are not allowed in RLM REPL")
        top = node.module.split(".")[0]
        if top not in ALLOWED_REPL_MODULES:
            raise ImportError(f"Module {node.module!r} is not allowed in RLM REPL")
        for alias in node.names:
            if alias.name == "*":
                raise ImportError("Wildcard imports are not allowed in RLM REPL")
            if alias.name.startswith("__") or alias.name in _FORBIDDEN_NAMES:
                raise PermissionError(f"Restricted import symbol {alias.name!r}")
            if alias.asname and (alias.asname.startswith("__") or alias.asname in _FORBIDDEN_NAMES):
                raise PermissionError(f"Restricted import alias {alias.asname!r}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("__") or node.attr in _FORBIDDEN_ATTRS:
            raise AttributeError(f"Access to restricted attribute {node.attr!r} is forbidden")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in _FORBIDDEN_NAMES or node.id.startswith("__"):
            raise PermissionError(f"Access to restricted name {node.id!r} is forbidden")
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> None:
        for name in node.names:
            if name in _FORBIDDEN_NAMES or name.startswith("__"):
                raise PermissionError(f"Restricted global name {name!r}")
        self.generic_visit(node)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        for name in node.names:
            if name in _FORBIDDEN_NAMES or name.startswith("__"):
                raise PermissionError(f"Restricted nonlocal name {name!r}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute) and node.func.attr in {"format", "format_map"}:
            target = node.func.value
            if isinstance(target, ast.Constant) and isinstance(target.value, str):
                if _FORMAT_ESCAPE_RE.search(target.value):
                    raise PermissionError("Format-string attribute traversal is forbidden in RLM REPL")
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str) and _FORMAT_ESCAPE_RE.search(node.value):
            raise PermissionError("Format-string attribute traversal pattern is forbidden in RLM REPL")
        self.generic_visit(node)


def _restricted_import(
    name: str,
    globals_dict: Mapping[str, Any] | None = None,
    locals_dict: Mapping[str, Any] | None = None,
    fromlist: Sequence[str] = (),
    level: int = 0,
) -> types.ModuleType:
    if level != 0:
        raise ImportError("Relative imports are not allowed in RLM REPL")
    top = name.split(".")[0]
    if top not in ALLOWED_REPL_MODULES:
        raise ImportError(f"Module {name!r} is not allowed in RLM REPL")
    return __import__(name, globals_dict, locals_dict, fromlist, level)


async def _await_value(awaitable: Any) -> Any:
    return await awaitable


def _run_awaitable_sync(awaitable: Any) -> Any:
    """Resolve an awaitable synchronously, whether or not an event loop is already running."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_await_value(awaitable))
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, _await_value(awaitable)).result()


def _compute_context_lengths(context: Any) -> list[int]:
    if isinstance(context, (str, bytes, bytearray)):
        return [len(context)]
    if isinstance(context, Mapping):
        return [len(v) if hasattr(v, "__len__") else len(str(v)) for v in context.values()]
    if isinstance(context, Sequence):
        return [len(item) if hasattr(item, "__len__") else len(str(item)) for item in context]
    if hasattr(context, "__len__"):
        return [len(context)]
    return [len(str(context))]


class SandboxedPythonREPL:
    """Persistent, sandboxed Python REPL binding ``context`` as an inspectable environment."""

    def __init__(
        self,
        context: Any,
        *,
        max_stdout_chars: int = 8192,
        budget_policy: RLMBudgetPolicy | None = None,
        budget_state: RLMBudgetState | None = None,
        llm_backend: Callable[[str], Any] | None = None,
    ) -> None:
        max_chars = as_int(max_stdout_chars, "max_stdout_chars")
        if max_chars < 1:
            raise SheafError("max_stdout_chars must be >= 1")
        if budget_policy is not None and not isinstance(budget_policy, RLMBudgetPolicy):
            raise SheafError("budget_policy must be an RLMBudgetPolicy or None")
        if budget_state is not None and not isinstance(budget_state, RLMBudgetState):
            raise SheafError("budget_state must be an RLMBudgetState or None")
        if budget_policy is not None and budget_state is None:
            budget_state = RLMBudgetState()

        self.context = context
        self.max_stdout_chars: int = max_chars
        self.budget_policy: RLMBudgetPolicy | None = budget_policy
        self.budget_state: RLMBudgetState | None = budget_state
        self.llm_backend: Callable[[str], Any] | None = llm_backend

        lengths = _compute_context_lengths(context)
        self.namespace: dict[str, Any] = {
            "context": context,
            "context_type": type(context).__name__,
            "context_lengths": lengths,
            "context_total_length": sum(lengths),
            "llm_query": self.llm_query,
            "llm_query_batch": self.llm_query_batch,
        }
        self._system_keys: frozenset[str] = frozenset(self.namespace.keys())
        self._initial_keys: frozenset[str] = self._system_keys | {"__builtins__"}
        self._exec_env: dict[str, Any] = dict(self.namespace)

    @property
    def variables(self) -> dict[str, Any]:
        return self.namespace

    @property
    def env(self) -> dict[str, Any]:
        return self.namespace

    def build_root_prompt(
        self,
        task: str,
        *,
        model_family: str = "gpt5",
        profile: str | None = None,
    ) -> str:
        """Build a metadata-only root prompt without ever inlining raw ``context`` text."""
        if not isinstance(task, str) or not task.strip():
            raise SheafError("task must be a non-empty string")
        prof = get_rlm_prompt_profile(profile if profile is not None else model_family)
        lengths = self.namespace["context_lengths"]
        preview = lengths[:16]
        suffix = ", ..." if len(lengths) > 16 else ""
        return (
            f"{prof.system_prompt}\n\n"
            f"Context Metadata:\n"
            f"- context_type: {self.namespace['context_type']}\n"
            f"- num_items: {len(lengths)}\n"
            f"- context_total_length: {self.namespace['context_total_length']} chars\n"
            f"- context_lengths_sample: {preview}{suffix}\n"
            f"- target_batch_chars: {prof.target_batch_chars}\n\n"
            f"Task:\n{task.strip()}"
        )

    def execute(self, code: str) -> dict[str, Any]:
        """Validate and execute Python code inside the AST + restricted-builtins sandbox."""
        if not isinstance(code, str):
            raise SheafError("code must be a string")
        try:
            tree = ast.parse(code, mode="exec")
        except SyntaxError as exc:
            raise ValueError(f"SyntaxError in RLM REPL code: {exc.msg}") from exc

        _SandboxASTValidator().visit(tree)

        if self.budget_policy is not None and self.budget_state is not None:
            self.budget_policy.check_repl_step(self.budget_state)

        buf = io.StringIO()
        soft_limit = self.max_stdout_chars + 4096
        written_chars = 0
        total_chars = 0

        def _safe_print(
            *args: Any,
            sep: str = " ",
            end: str = "\n",
            file: Any = None,
            flush: bool = False,
        ) -> None:
            del file, flush
            nonlocal written_chars, total_chars
            text = sep.join(str(a) for a in args) + end
            total_chars += len(text)
            remaining = soft_limit - written_chars
            if remaining > 0:
                chunk = text[:remaining]
                buf.write(chunk)
                written_chars += len(chunk)

        safe_builtins = {
            **_SAFE_BUILTINS,
            "print": _safe_print,
            "__import__": _restricted_import,
        }

        prev_user_keys = {k for k in self.namespace if k not in self._initial_keys}
        self._exec_env.clear()
        self._exec_env.update({k: v for k, v in self.namespace.items() if k != "__builtins__"})
        self._exec_env["__builtins__"] = safe_builtins

        compiled = compile(tree, "<rlm_repl>", "exec")
        try:
            with contextlib.redirect_stdout(buf):
                exec(compiled, self._exec_env, self._exec_env)
        finally:
            cleaned = {k: v for k, v in self._exec_env.items() if k != "__builtins__"}
            self.namespace.clear()
            self.namespace.update(cleaned)
            self._exec_env.pop("__builtins__", None)

        raw_stdout = buf.getvalue()
        actual_total = max(total_chars, len(raw_stdout))
        truncated = actual_total > self.max_stdout_chars
        if truncated:
            marker = "\n...[truncated]"
            if self.max_stdout_chars > len(marker):
                stdout = raw_stdout[: self.max_stdout_chars - len(marker)] + marker
            else:
                stdout = raw_stdout[: self.max_stdout_chars]
        else:
            stdout = raw_stdout

        user_vars = sorted(
            k
            for k, v in self.namespace.items()
            if k not in self._initial_keys and not k.startswith("_") and not isinstance(v, types.ModuleType)
        )
        new_vars = sorted(k for k in user_vars if k not in prev_user_keys)
        return {
            "ok": True,
            "passed": True,
            "stdout": stdout,
            "output": stdout,
            "truncated": truncated,
            "total_stdout_chars": actual_total,
            "new_vars": new_vars,
            "vars": user_vars,
        }

    def llm_query(
        self,
        prompt: str,
        *,
        depth: int = 1,
        tokens: int | None = None,
        max_output_tokens: int | None = None,
        thinking_budget: int | None = None,
    ) -> Any:
        """Issue a single budget-governed sub-LM call (supporting sync or async backends)."""
        if not isinstance(prompt, str):
            raise SheafError("prompt must be a string")
        tok = max(1, (len(prompt) + 3) // 4) if tokens is None else as_int(tokens, "tokens")
        if self.budget_policy is not None and self.budget_state is not None:
            self.budget_policy.check_subcall(
                self.budget_state,
                depth=depth,
                chars=len(prompt),
                tokens=tok,
                max_output_tokens=max_output_tokens,
                thinking_budget=thinking_budget,
            )
        if self.llm_backend is None:
            raise SheafError("llm_backend is not configured")
        out = self.llm_backend(prompt)
        if inspect.isawaitable(out):
            return _run_awaitable_sync(out)
        return out

    async def llm_query_batch(
        self,
        prompts: Sequence[str],
        *,
        max_concurrency: int = 4,
        depth: int = 1,
        tokens_per_prompt: Sequence[int] | None = None,
        max_output_tokens: int | None = None,
        thinking_budget: int | None = None,
    ) -> list[Any]:
        """Issue concurrent semaphore-bounded sub-LM calls with atomic upfront budget checks."""
        max_c = as_int(max_concurrency, "max_concurrency")
        if max_c < 1:
            raise SheafError("max_concurrency must be >= 1")
        if isinstance(prompts, (str, bytes)) or not isinstance(prompts, Sequence):
            raise SheafError("prompts must be a sequence of strings")
        prompt_list = list(prompts)
        for p in prompt_list:
            if not isinstance(p, str):
                raise SheafError("every prompt in prompts must be a string")
        if tokens_per_prompt is not None:
            if isinstance(tokens_per_prompt, (str, bytes)) or not isinstance(tokens_per_prompt, Sequence):
                raise SheafError("tokens_per_prompt must be a sequence of integers")
            if len(tokens_per_prompt) != len(prompt_list):
                raise SheafError("tokens_per_prompt length must match prompts length")

        if not prompt_list:
            return []

        if self.budget_policy is not None and self.budget_state is not None:
            snap = (
                self.budget_state.subcalls,
                self.budget_state.total_tokens,
                self.budget_state.max_depth_seen,
            )
            try:
                for idx, prompt in enumerate(prompt_list):
                    tok = (
                        max(1, (len(prompt) + 3) // 4)
                        if tokens_per_prompt is None
                        else as_int(tokens_per_prompt[idx], "tokens_per_prompt")
                    )
                    self.budget_policy.check_subcall(
                        self.budget_state,
                        depth=depth,
                        chars=len(prompt),
                        tokens=tok,
                        max_output_tokens=max_output_tokens,
                        thinking_budget=thinking_budget,
                    )
                if self.llm_backend is None:
                    raise SheafError("llm_backend is not configured")
            except Exception:
                self.budget_state.subcalls, self.budget_state.total_tokens, self.budget_state.max_depth_seen = snap
                raise
        elif self.llm_backend is None:
            raise SheafError("llm_backend is not configured")

        backend = self.llm_backend
        is_async = inspect.iscoroutinefunction(backend) or inspect.iscoroutinefunction(
            getattr(backend, "__call__", None)
        )
        sem = asyncio.Semaphore(max_c)

        async def _invoke_one(p: str) -> Any:
            async with sem:
                if is_async:
                    return await backend(p)
                res = await asyncio.to_thread(backend, p)
                if inspect.isawaitable(res):
                    return await res
                return res

        return list(await asyncio.gather(*(_invoke_one(p) for p in prompt_list)))


RLMContextREPL = SandboxedPythonREPL
RLMReplEnvironment = SandboxedPythonREPL
SandboxedRLMRepl = SandboxedPythonREPL


def _extract_repl_code_blocks(text: str) -> list[str]:
    """Extract ```repl / ```python fenced code blocks while honoring CommonMark fence nesting."""
    blocks: list[str] = []
    fence_char: str | None = None
    fence_len = 0
    capture_active = False
    current_lines: list[str] = []

    for line in text.splitlines():
        if fence_char is not None:
            stripped = line.lstrip(" ")
            leading_spaces = len(line) - len(stripped)
            if leading_spaces <= 3:
                run_len = 0
                while run_len < len(stripped) and stripped[run_len] == fence_char:
                    run_len += 1
                if run_len >= fence_len and not stripped[run_len:].strip():
                    if capture_active:
                        blocks.append("\n".join(current_lines))
                    fence_char = None
                    fence_len = 0
                    capture_active = False
                    current_lines = []
                    continue
            if capture_active:
                current_lines.append(line)
            continue

        match = _FENCE_OPEN_RE.match(line)
        if match is None:
            continue
        marker = match.group(1)
        info = match.group(2)
        if marker[0] == "`" and "`" in info:
            continue
        if marker[0] == "~" and re.search(r"~{" + str(len(marker)) + r",}\s*$", info) is not None:
            continue
        fence_char = marker[0]
        fence_len = len(marker)
        lang = info.strip().split()[0].lower() if info.strip() else ""
        capture_active = lang in {"repl", "python"}
        current_lines = []

    return blocks


def _is_nonempty_verified_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (bytes, bytearray, list, tuple, dict, set, frozenset)):
        return len(value) > 0
    return True


class RLMOrchestrator:
    """Multi-turn RLM orchestration loop with thinking-reserve, FINAL, and anti-repetition guards."""

    def __init__(
        self,
        *,
        budget_policy: RLMBudgetPolicy | None = None,
        max_turns: int = 8,
        max_consec_no_new_var: int = 3,
        max_stdout_chars: int = 4096,
        model_family: str = "gpt5",
    ) -> None:
        if budget_policy is not None and not isinstance(budget_policy, RLMBudgetPolicy):
            raise SheafError("budget_policy must be an RLMBudgetPolicy or None")
        turns = as_int(max_turns, "max_turns")
        if turns < 1:
            raise SheafError("max_turns must be >= 1")
        consec = as_int(max_consec_no_new_var, "max_consec_no_new_var")
        if consec < 1:
            raise SheafError("max_consec_no_new_var must be >= 1")
        stdout_cap = as_int(max_stdout_chars, "max_stdout_chars")
        if stdout_cap < 1:
            raise SheafError("max_stdout_chars must be >= 1")
        prof = get_rlm_prompt_profile(model_family)

        self.budget_policy: RLMBudgetPolicy | None = budget_policy
        self.max_turns: int = turns
        self.max_consec_no_new_var: int = consec
        self.max_stdout_chars: int = stdout_cap
        self.model_family: str = prof.family

    def run(
        self,
        query: str,
        context: Any,
        model_fn: Callable[[list[dict[str, Any]]], Any],
        *,
        llm_backend: Callable[[str], Any] | None = None,
        profile: str | None = None,
    ) -> dict[str, Any]:
        """Execute multi-turn RLM loop until FINAL directive, stagnation halt, or max_turns."""
        if not isinstance(query, str) or not query.strip():
            raise SheafError("query must be a non-empty string")
        if not callable(model_fn):
            raise SheafError("model_fn must be callable")

        prof = get_rlm_prompt_profile(self.model_family if profile is None else profile)
        budget_state = RLMBudgetState()
        repl = SandboxedPythonREPL(
            context,
            max_stdout_chars=self.max_stdout_chars,
            budget_policy=self.budget_policy,
            budget_state=budget_state,
            llm_backend=llm_backend,
        )
        tracker = AntiRepetitionTracker(max_consec_no_new_var=self.max_consec_no_new_var)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": prof.system_prompt},
            {"role": "user", "content": repl.build_root_prompt(query, model_family=prof.family)},
        ]

        for turn_idx in range(1, self.max_turns + 1):
            raw_resp = model_fn(list(messages))
            if inspect.isawaitable(raw_resp):
                raw_resp = _run_awaitable_sync(raw_resp)

            if isinstance(raw_resp, str):
                text = raw_resp
                max_output_tokens = None
                thinking_budget = None
            elif isinstance(raw_resp, Mapping):
                text = str(raw_resp.get("text", raw_resp.get("content", "")))
                max_output_tokens = raw_resp.get("max_output_tokens")
                thinking_budget = raw_resp.get("thinking_budget")
            else:
                raise SheafError("model_fn must return a string or mapping")

            if self.budget_policy is not None and (max_output_tokens is not None or thinking_budget is not None):
                mot = as_int(max_output_tokens, "max_output_tokens") if max_output_tokens is not None else None
                tb = as_int(thinking_budget, "thinking_budget") if thinking_budget is not None else 0
                if tb < 0 or (mot is not None and mot < 0):
                    raise SheafError("max_output_tokens and thinking_budget must be non-negative")
                if mot is not None and (mot - tb) < self.budget_policy.thinking_reserve:
                    raise SheafError(
                        f"RLM thinking_budget {tb} leaves {mot - tb} < thinking_reserve "
                        f"{self.budget_policy.thinking_reserve}"
                    )

            prev_snapshot = {
                k: v
                for k, v in repl.namespace.items()
                if k not in repl._initial_keys and not k.startswith("_") and not isinstance(v, types.ModuleType)
            }

            blocks = _extract_repl_code_blocks(text)
            repl_feedback: list[str] = []
            for block in blocks:
                try:
                    exec_res = repl.execute(block)
                    if exec_res["stdout"]:
                        repl_feedback.append(exec_res["stdout"])
                except SheafError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    repl_feedback.append(f"[REPL Error] {type(exc).__name__}: {exc}")

            user_vars = sorted(
                k
                for k, v in repl.namespace.items()
                if k not in repl._initial_keys and not k.startswith("_") and not isinstance(v, types.ModuleType)
            )
            verified_val: Any = None
            for var_name in user_vars:
                val = repl.namespace[var_name]
                changed = var_name not in prev_snapshot or prev_snapshot[var_name] is not val
                if changed and _is_nonempty_verified_value(val):
                    verified_val = val

            parsed = parse_final_directive(text, repl.namespace)
            if parsed["accepted"] is True:
                return {
                    "ok": True,
                    "passed": True,
                    "answer": parsed["value"],
                    "kind": parsed["kind"],
                    "var_name": parsed["var_name"],
                    "turns": turn_idx,
                    "termination": "final_directive",
                    "budget_state": budget_state,
                }

            rep_status = tracker.record_turn(text, user_vars, verified_value=verified_val)
            if rep_status.should_stop:
                if rep_status.best_verified_value is not None:
                    return {
                        "ok": True,
                        "passed": True,
                        "answer": rep_status.best_verified_value,
                        "turns": turn_idx,
                        "termination": "anti_repetition",
                        "budget_state": budget_state,
                    }
                return {
                    "ok": False,
                    "passed": False,
                    "status": "error",
                    "answer": None,
                    "turns": turn_idx,
                    "termination": "anti_repetition_no_value",
                    "budget_state": budget_state,
                }

            messages.append({"role": "assistant", "content": text})
            feedback = "\n".join(repl_feedback).strip()
            if parsed["kind"] is not None and not parsed["accepted"]:
                reject_note = f"FINAL directive ({parsed['kind']}) rejected: provide a verified non-empty result."
                feedback = f"{feedback}\n{reject_note}".strip()
            messages.append({"role": "user", "content": feedback or "(no stdout)"})

        return {
            "ok": False,
            "passed": False,
            "status": "error",
            "answer": None,
            "error": "max_turns exhausted without accepted FINAL directive",
            "turns": self.max_turns,
            "termination": "max_turns_exceeded",
            "budget_state": budget_state,
        }


RLMEngine = RLMOrchestrator
RLMRunner = RLMOrchestrator


def _balanced_index_chunks(n_items: int, n_chunks: int) -> list[list[int]]:
    if n_items <= 0:
        return []
    try:
        import numpy as np

        splits = np.array_split(np.arange(n_items, dtype=np.int64), n_chunks)
        return [arr.tolist() for arr in splits if len(arr) > 0]
    except ModuleNotFoundError:
        k = min(n_chunks, n_items)
        q, r = divmod(n_items, k)
        chunks: list[list[int]] = []
        cursor = 0
        for idx in range(k):
            size = q + (1 if idx < r else 0)
            chunks.append(list(range(cursor, cursor + size)))
            cursor += size
        return chunks


def oolong_pairs_stitch(
    items: Sequence[Any],
    n_chunks: int = 4,
    exec_kernel: Any | None = None,
    *,
    pair_predicate: Callable[..., bool] | None = None,
) -> dict[str, Any]:
    """Partition ``N`` items into balanced chunks, evaluate all ``N*(N-1)//2`` pairs, and log to kernel."""
    if isinstance(items, (str, bytes, Mapping)) or not isinstance(items, Iterable):
        raise SheafError("items must be a sequence or iterable of records")
    chunks_req = as_int(n_chunks, "n_chunks")
    if chunks_req < 1:
        raise SheafError("n_chunks must be >= 1")

    items_list = list(items)
    n_items = len(items_list)
    chunks = _balanced_index_chunks(n_items, chunks_req)
    chunk_sizes = [len(c) for c in chunks]
    chunk_slices = [(c[0], c[-1] + 1) for c in chunks]
    covered_indices = [idx for chunk in chunks for idx in chunk]

    matched_pairs: list[tuple[int, int]] = []
    total_pairs_evaluated = 0

    pred_arity = 0
    if pair_predicate is not None:
        if not callable(pair_predicate):
            raise SheafError("pair_predicate must be callable or None")
        params = [
            p
            for p in inspect.signature(pair_predicate).parameters.values()
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        pred_arity = 4 if len(params) >= 4 else 2

    def _matches(i: int, j: int) -> bool:
        assert pair_predicate is not None
        if pred_arity == 4:
            return bool(pair_predicate(i, items_list[i], j, items_list[j]))
        return bool(pair_predicate(items_list[i], items_list[j]))

    if pair_predicate is None and n_items > 1000:
        for c_idx, chunk in enumerate(chunks):
            sz = len(chunk)
            total_pairs_evaluated += sz * (sz - 1) // 2
            for other_idx in range(c_idx + 1, len(chunks)):
                total_pairs_evaluated += sz * len(chunks[other_idx])
    else:
        for c_idx, chunk in enumerate(chunks):
            sz = len(chunk)
            for pos_a in range(sz):
                idx_a = chunk[pos_a]
                for pos_b in range(pos_a + 1, sz):
                    idx_b = chunk[pos_b]
                    total_pairs_evaluated += 1
                    if pair_predicate is not None and _matches(idx_a, idx_b):
                        matched_pairs.append((idx_a, idx_b))
            for other_idx in range(c_idx + 1, len(chunks)):
                other_chunk = chunks[other_idx]
                for idx_a in chunk:
                    for idx_b in other_chunk:
                        total_pairs_evaluated += 1
                        lo, hi = (idx_a, idx_b) if idx_a < idx_b else (idx_b, idx_a)
                        if pair_predicate is not None and _matches(lo, hi):
                            matched_pairs.append((lo, hi))

    source_id: str | None = None
    kernel_head: str | None = None
    if exec_kernel is not None:
        if not hasattr(exec_kernel, "execute"):
            raise SheafError("exec_kernel must implement execute(command)")
        for c_idx, chunk in enumerate(chunks):
            chunk_payload = {
                "chunk_index": c_idx,
                "indices": chunk,
                "item_hashes": [sha256_hex(repr(items_list[i]).encode("utf-8"))[:16] for i in chunk],
                "intra_pairs": len(chunk) * (len(chunk) - 1) // 2,
            }
            chunk_text = json.dumps(chunk_payload, sort_keys=True, separators=(",", ":"))
            chunk_id = f"oolong_chunk_{c_idx}_{sha256_hex(chunk_text.encode('utf-8'))[:16]}"
            res = exec_kernel.execute({"action": "source", "id": chunk_id, "text": chunk_text})
            if res.get("status") != "ok":
                raise SheafError(f"exec_kernel rejected chunk source {chunk_id!r}: {res}")

        items_digest = sha256_hex(json.dumps([repr(x) for x in items_list], separators=(",", ":")).encode("utf-8"))
        summary_payload = {
            "n_items": n_items,
            "n_chunks": len(chunks),
            "chunk_sizes": chunk_sizes,
            "total_pairs_evaluated": total_pairs_evaluated,
            "matched_pair_count": len(matched_pairs),
            "items_sha256": items_digest,
        }
        summary_text = json.dumps(summary_payload, sort_keys=True, separators=(",", ":"))
        source_id = f"oolong_stitch_{n_items}_{len(chunks)}_{sha256_hex(summary_text.encode('utf-8'))[:16]}"
        src_res = exec_kernel.execute({"action": "source", "id": source_id, "text": summary_text})
        if src_res.get("status") != "ok":
            raise SheafError(f"exec_kernel rejected stitch source {source_id!r}: {src_res}")

        set_res = exec_kernel.execute(
            {
                "action": "set",
                "name": f"oolong_pairs_{source_id[-12:]}",
                "value": total_pairs_evaluated,
                "unit": "1",
                "source": source_id,
            }
        )
        if set_res.get("status") != "ok":
            raise SheafError(f"exec_kernel rejected stitch binding: {set_res}")
        kernel_head = getattr(exec_kernel, "head", None)

    return {
        "ok": True,
        "passed": True,
        "n_items": n_items,
        "n_chunks": len(chunks),
        "chunks": chunks,
        "chunk_sizes": chunk_sizes,
        "chunk_slices": chunk_slices,
        "covered_indices": covered_indices,
        "total_pairs_evaluated": total_pairs_evaluated,
        "matched_pairs": matched_pairs,
        "source_id": source_id,
        "kernel_head": kernel_head,
    }


run_oolong_pairs_stitching = oolong_pairs_stitch
rlm_subgraph_stitch = oolong_pairs_stitch
stitch_oolong_pairs = oolong_pairs_stitch

__all__ = [
    "ALLOWED_REPL_MODULES",
    "RLM_PROMPT_PROFILES",
    "RLMPromptProfile",
    "get_rlm_prompt_profile",
    "SandboxedPythonREPL",
    "RLMContextREPL",
    "RLMReplEnvironment",
    "SandboxedRLMRepl",
    "RLMOrchestrator",
    "RLMEngine",
    "RLMRunner",
    "oolong_pairs_stitch",
    "run_oolong_pairs_stitching",
    "rlm_subgraph_stitch",
    "stitch_oolong_pairs",
]
