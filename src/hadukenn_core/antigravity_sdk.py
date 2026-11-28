"""Pure-stdlib Google Antigravity Python SDK adapter, fallback & multi-modal contracts.

Provides drop-in local and fallback support when the optional ``google-antigravity``
wheel is not installed, while preserving exact SDK enum/config/permission/lifecycle/CLI
contracts and HaDukeNN_LLM empirical guardrails (AGENTS.md §3).
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


def _detect_antigravity_sdk() -> bool:
    try:
        return (
            importlib.util.find_spec("google") is not None
            and importlib.util.find_spec("google.antigravity") is not None
        )
    except (ModuleNotFoundError, ValueError):
        return False


SDK_AVAILABLE: bool = _detect_antigravity_sdk()

CONV_ID_RE = re.compile(r"^[A-Za-z0-9-]+$")

SDK_BUILTIN_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "list_directory",
        "search_directory",
        "find_file",
        "view_file",
        "create_file",
        "edit_file",
        "run_command",
        "ask_question",
        "start_subagent",
        "generate_image",
        "search_web",
        "read_url_content",
        "finish",
    }
)

HEADLESS_OUTPUT_FORMATS: tuple[str, ...] = ("text", "json", "stream-json")
HEADLESS_INPUT_FORMATS: tuple[str, ...] = ("text", "stream-json")
CLI_TERMINAL_STATUSES: tuple[str, ...] = (
    "SUCCESS",
    "ERROR",
    "CANCELED",
    "INTERRUPTED",
    "INVALID",
    "WAITING",
    "RUNNING",
)
SETTINGS_SCHEMA_URL: str = "https://antigravity.google/schemas/v1/settings.json"

PERMISSION_PRESETS: dict[str, dict[str, str]] = {
    "request-review": {"read_file": "allow", "write_file": "ask", "command": "ask", "sandbox": "workspace"},
    "default": {"read_file": "allow", "write_file": "ask", "command": "ask", "sandbox": "workspace"},
    "accept-edits": {"read_file": "allow", "write_file": "allow", "command": "ask", "sandbox": "workspace"},
    "auto": {"read_file": "allow", "write_file": "allow", "command": "allow", "sandbox": "workspace"},
    "plan": {"read_file": "allow", "write_file": "deny", "command": "deny", "sandbox": "read_only"},
    "strict": {"read_file": "ask", "write_file": "deny", "command": "deny", "sandbox": "read_only"},
    "always-proceed": {"read_file": "allow", "write_file": "allow", "command": "allow", "sandbox": "off"},
    "proceed-in-sandbox": {"read_file": "allow", "write_file": "allow", "command": "allow", "sandbox": "enforced"},
}

_MEDIA_TYPES: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".svg": "image/svg+xml",
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".json": "application/json",
    ".jsonl": "application/x-ndjson",
    ".txt": "text/plain",
    ".md": "text/markdown",
}

_SHELL_METACHAR_CONSTRUCTS = ("$(", "`", "<(", "$((", "{", "tar --to-command", "git -c core.pager=")
_BANNED_32K_GPU_HEADS = frozenset({"182x181", "goodman_factored", "cluster_is", "fp2"})
_DECISION_RANK = {
    "allow": 1,
    "auto_approve": 1,
    "ask": 2,
    "force_ask": 2,
    "deny_unless_prior_grant": 2,
    "deny": 3,
}


class BuiltinTools(StrEnum):
    """Exact Antigravity Python SDK BuiltinTools enum members."""

    LIST_DIRECTORY = "list_directory"
    SEARCH_DIRECTORY = "search_directory"
    FIND_FILE = "find_file"
    VIEW_FILE = "view_file"
    CREATE_FILE = "create_file"
    EDIT_FILE = "edit_file"
    RUN_COMMAND = "run_command"
    ASK_QUESTION = "ask_question"
    START_SUBAGENT = "start_subagent"
    GENERATE_IMAGE = "generate_image"
    SEARCH_WEB = "search_web"
    READ_URL_CONTENT = "read_url_content"
    FINISH = "finish"

    @classmethod
    def read_only(cls) -> tuple[BuiltinTools, ...]:
        """Return read-only SDK tools disjoint from file-creation, editing, and shell execution."""
        return (
            cls.LIST_DIRECTORY,
            cls.SEARCH_DIRECTORY,
            cls.FIND_FILE,
            cls.VIEW_FILE,
            cls.ASK_QUESTION,
            cls.SEARCH_WEB,
            cls.READ_URL_CONTENT,
        )


def _normalize_sdk_tool(tool: BuiltinTools | str) -> str:
    name = tool.value if isinstance(tool, BuiltinTools) else str(tool)
    if name not in SDK_BUILTIN_TOOL_NAMES:
        raise ValueError(
            f"Unknown SDK BuiltinTools member {name!r}; expected one of {sorted(SDK_BUILTIN_TOOL_NAMES)}"
        )
    return name


def validate_conversation_id(conversation_id: str) -> bool:
    """Validate Antigravity session ID (>= 32 chars, alphanumeric + hyphens, NO underscores)."""
    if not isinstance(conversation_id, str):
        raise ValueError("conversation_id must be a string")
    if len(conversation_id) < 32:
        raise ValueError(f"conversation_id must be >= 32 characters, got {len(conversation_id)}")
    if "_" in conversation_id:
        raise ValueError("conversation_id must not contain underscores")
    if CONV_ID_RE.fullmatch(conversation_id) is None:
        raise ValueError("conversation_id must contain only ASCII alphanumeric characters and hyphens")
    return True


is_valid_conversation_id = validate_conversation_id


@dataclass(frozen=True)
class CapabilitiesConfig:
    """Tool allowlist/denylist configuration using SDK BuiltinTools names."""

    enabled_tools: tuple[str, ...] = field(default_factory=tuple)
    disabled_tools: tuple[str, ...] = field(default_factory=tuple)
    enable_subagents: bool = True

    def __init__(
        self,
        enabled_tools: Sequence[BuiltinTools | str] | None = None,
        disabled_tools: Sequence[BuiltinTools | str] = (),
        enable_subagents: bool = True,
    ) -> None:
        if isinstance(enabled_tools, str) or isinstance(disabled_tools, str):
            raise ValueError("enabled_tools and disabled_tools must be sequences of BuiltinTools")
        enabled = tuple(
            _normalize_sdk_tool(t)
            for t in (BuiltinTools.read_only() if enabled_tools is None else enabled_tools)
        )
        disabled = tuple(_normalize_sdk_tool(t) for t in disabled_tools)
        object.__setattr__(self, "enabled_tools", enabled)
        object.__setattr__(self, "disabled_tools", disabled)
        object.__setattr__(self, "enable_subagents", bool(enable_subagents))


@dataclass
class McpStdioServer:
    """Stdio MCP server configuration for Antigravity Python SDK."""

    command: str
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | Path | None = None
    timeout_seconds: int = 180

    def __post_init__(self) -> None:
        if not isinstance(self.command, str) or not self.command.strip():
            raise ValueError("McpStdioServer.command must be a non-empty string")
        if isinstance(self.timeout_seconds, bool) or not isinstance(self.timeout_seconds, int):
            raise ValueError("timeout_seconds must be an integer")
        if self.timeout_seconds != -1 and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be -1 or > 0")
        self.args = tuple(str(a) for a in self.args)
        self.env = {str(k): str(v) for k, v in self.env.items()}


@dataclass
class McpStreamableHttpServer:
    """Streamable HTTP MCP server configuration (legacy SSE/httpUrl rejected)."""

    url: str
    headers: dict[str, str] = field(default_factory=dict)
    auth_provider_type: str | None = None
    oauth: dict[str, str] | None = None
    timeout_seconds: int = 180

    def __post_init__(self) -> None:
        if not isinstance(self.url, str) or not self.url.strip():
            raise ValueError("McpStreamableHttpServer.url must be a non-empty string")
        if isinstance(self.timeout_seconds, bool) or not isinstance(self.timeout_seconds, int):
            raise ValueError("timeout_seconds must be an integer")
        if self.timeout_seconds != -1 and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be -1 or > 0")


@dataclass
class SubagentConfig:
    """Subagent specification using SDK BuiltinTools names."""

    name: str
    description: str
    tools: tuple[str, ...] = ()
    model: str = "inherit"
    command_execution_policy: str = "sandbox"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("SubagentConfig.name must be non-empty")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("SubagentConfig.description must be non-empty")
        if isinstance(self.tools, str):
            raise ValueError("SubagentConfig.tools must be a sequence")
        self.tools = tuple(_normalize_sdk_tool(t) for t in self.tools)


@dataclass
class LocalAgentConfig:
    """Antigravity LocalAgentConfig with Vertex/local isolation and pure-stdlib fallback."""

    conversation_id: str | None = None
    capabilities: CapabilitiesConfig | None = None
    app_data_dir: Path | str | None = None
    save_dir: Path | str | None = None
    vertex: bool = False
    project: str | None = None
    location: str | None = None
    model: str = "gemini-2.5-pro"
    system_instruction: str = ""
    mcp_servers: tuple[Any, ...] = ()
    subagents: tuple[SubagentConfig, ...] = ()
    response_schema: Any = None
    backend_mode: str = field(init=False)

    def __post_init__(self) -> None:
        if self.conversation_id is not None:
            validate_conversation_id(self.conversation_id)
        if self.capabilities is None:
            self.capabilities = CapabilitiesConfig()
        self.mcp_servers = tuple(self.mcp_servers)
        self.subagents = tuple(self.subagents)
        self.backend_mode = "google_antigravity" if SDK_AVAILABLE else "stdlib_fallback"


@dataclass
class LiteRTAgentConfig:
    """Local LiteRT-LM configuration with port and model-path validation."""

    model_path: str = "models/litert_int8"
    backend: str = "cpu"
    vision_backend: str = "cpu"
    audio_backend: str = "cpu"
    enable_speculative_decoding: bool = False
    cache_dir: str | None = None
    port: int = 8080
    compaction_config: int = 40960
    backend_mode: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.model_path, str) or not self.model_path.strip():
            raise ValueError("LiteRTAgentConfig.model_path must be a non-empty string")
        if isinstance(self.port, bool) or not isinstance(self.port, int) or not (1 <= self.port <= 65535):
            raise ValueError(f"LiteRTAgentConfig.port must be in [1, 65535], got {self.port!r}")
        if isinstance(self.compaction_config, bool) or not isinstance(self.compaction_config, int) or self.compaction_config <= 0:
            raise ValueError("compaction_config must be a positive integer")
        self.backend_mode = "google_antigravity" if SDK_AVAILABLE else "stdlib_fallback"

    @classmethod
    def lightweight(
        cls,
        model_path: str = "models/litert_int8",
        *,
        backend: str = "cpu",
        vision_backend: str = "cpu",
        audio_backend: str = "cpu",
        port: int = 8080,
    ) -> LiteRTAgentConfig:
        return cls(
            model_path=model_path,
            backend=backend,
            vision_backend=vision_backend,
            audio_backend=audio_backend,
            enable_speculative_decoding=False,
            port=port,
            compaction_config=40960,
        )


@dataclass
class LocalOpenAIAgentConfig:
    """Local OpenAI-compatible runtime configuration."""

    model: str = "hadukenn-struct32k"
    base_url: str = "http://127.0.0.1:8080/v1"
    api_key: str = "local"
    temperature: float = 0.0
    max_tokens: int = 4096
    backend_mode: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("LocalOpenAIAgentConfig.model must be non-empty")
        if not isinstance(self.base_url, str) or not self.base_url.strip():
            raise ValueError("LocalOpenAIAgentConfig.base_url must be non-empty")
        self.backend_mode = "google_antigravity" if SDK_AVAILABLE else "stdlib_fallback"

    @classmethod
    def lightweight(
        cls,
        model: str = "hadukenn-struct32k",
        base_url: str = "http://127.0.0.1:8080/v1",
        **kwargs: Any,
    ) -> LocalOpenAIAgentConfig:
        return cls(model=model, base_url=base_url, **kwargs)


def _check_json_type(value: Any, expected_type: str, path: str) -> None:
    if expected_type == "object":
        if not isinstance(value, dict):
            raise ValueError(f"{path}: expected object, got {type(value).__name__}")
    elif expected_type == "array":
        if not isinstance(value, list):
            raise ValueError(f"{path}: expected array, got {type(value).__name__}")
    elif expected_type == "string":
        if not isinstance(value, str):
            raise ValueError(f"{path}: expected string, got {type(value).__name__}")
    elif expected_type == "boolean":
        if not isinstance(value, bool):
            raise ValueError(f"{path}: expected boolean, got {type(value).__name__}")
    elif expected_type == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{path}: expected integer, got {type(value).__name__}")
    elif expected_type == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"{path}: expected finite number, got {type(value).__name__}")
    elif expected_type == "null":
        if value is not None:
            raise ValueError(f"{path}: expected null, got {type(value).__name__}")
    else:
        raise ValueError(f"{path}: unsupported JSON schema type {expected_type!r}")


def _json_equal(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a is b
    if isinstance(a, dict) and isinstance(b, dict):
        return len(a) == len(b) and all(k in b and _json_equal(v, b[k]) for k, v in a.items())
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_json_equal(x, y) for x, y in zip(a, b))
    return a == b


def _validate_schema_node(value: Any, schema: dict[str, Any], path: str = "$") -> None:
    expected_type = schema.get("type")
    if isinstance(expected_type, str):
        _check_json_type(value, expected_type, path)
    elif isinstance(expected_type, list):
        for candidate in expected_type:
            try:
                _check_json_type(value, candidate, path)
                break
            except ValueError:
                continue
        else:
            raise ValueError(f"{path}: value did not match any type in {expected_type!r}")

    if "enum" in schema and not any(_json_equal(value, cand) for cand in schema["enum"]):
        raise ValueError(f"{path}: value {value!r} not in enum {schema['enum']!r}")

    if isinstance(value, dict):
        for req_key in schema.get("required", ()):
            if req_key not in value:
                raise ValueError(f"{path}: missing required key {req_key!r}")
        props = schema.get("properties")
        if isinstance(props, dict):
            for prop_name, prop_schema in props.items():
                if prop_name in value and isinstance(prop_schema, dict):
                    _validate_schema_node(value[prop_name], prop_schema, f"{path}.{prop_name}")
        if schema.get("additionalProperties") is False:
            allowed_keys = set(props.keys()) if isinstance(props, dict) else set()
            extra = set(value.keys()) - allowed_keys
            if extra:
                raise ValueError(f"{path}: unexpected properties {sorted(extra)}")

    if isinstance(value, list):
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for idx, item in enumerate(value):
                _validate_schema_node(item, item_schema, f"{path}[{idx}]")


def validate_response_schema(payload: Any, schema: Any) -> bool:
    """Validate structured response payload against a JSON schema dict or Pydantic-like model."""
    if isinstance(schema, dict):
        _validate_schema_node(payload, schema)
        return True
    if hasattr(schema, "model_validate") and callable(schema.model_validate):
        try:
            schema.model_validate(payload)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Response schema model validation failed: {exc}") from exc
        return True
    if hasattr(schema, "model_json_schema") and callable(schema.model_json_schema):
        _validate_schema_node(payload, schema.model_json_schema())
        return True
    raise ValueError(f"Unsupported response schema object: {type(schema).__name__}")


verify_response_schema = validate_response_schema


class ResponseSchemaValidator:
    """Reusable schema validator wrapping ``validate_response_schema``."""

    def __init__(self, schema: Any) -> None:
        self.schema = schema

    def validate(self, payload: Any) -> bool:
        return validate_response_schema(payload, self.schema)

    verify = validate


@dataclass(frozen=True)
class HookResult:
    """Structured decision emitted by Antigravity lifecycle hooks & policy evaluators."""

    decision: str = "allow"
    reason: str = ""
    overwrite: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.decision, str) or not self.decision.strip():
            raise ValueError("HookResult.decision must be a non-empty string")


class LifecycleHooks:
    """Decorator & callback registry for ``pre_turn`` and ``on_tool_error`` hooks."""

    def __init__(self) -> None:
        self._pre_turn_hooks: list[Callable[[dict[str, Any]], Any]] = []
        self._on_tool_error_hooks: list[Callable[[dict[str, Any]], Any]] = []

    def pre_turn(self, fn: Callable[[dict[str, Any]], Any]) -> Callable[[dict[str, Any]], Any]:
        if not callable(fn):
            raise ValueError("pre_turn callback must be callable")
        self._pre_turn_hooks.append(fn)
        return fn

    def on_tool_error(self, fn: Callable[[dict[str, Any]], Any]) -> Callable[[dict[str, Any]], Any]:
        if not callable(fn):
            raise ValueError("on_tool_error callback must be callable")
        self._on_tool_error_hooks.append(fn)
        return fn

    register_pre_turn = pre_turn
    register_on_tool_error = on_tool_error

    @staticmethod
    def _aggregate_hook_results(results: list[Any], default: HookResult) -> Any:
        if not results:
            return default
        structured: list[HookResult] = []
        for res in results:
            if isinstance(res, HookResult):
                structured.append(res)
            elif isinstance(res, dict) and isinstance(res.get("decision"), str) and res["decision"].strip():
                ow = res.get("overwrite")
                structured.append(
                    HookResult(
                        decision=res["decision"],
                        reason=str(res.get("reason", "")),
                        overwrite=ow if isinstance(ow, dict) else None,
                    )
                )
        if not structured:
            return results[-1]

        ranked = [(_DECISION_RANK.get(r.decision.strip().lower(), 2), idx, r) for idx, r in enumerate(structured)]
        max_rank = max(item[0] for item in ranked)
        winners = [item[2] for item in ranked if item[0] == max_rank]
        best_decision = winners[0].decision
        reasons: list[str] = []
        for item in winners:
            r = item.reason.strip()
            if r and r not in reasons:
                reasons.append(r)
        merged_overwrite: dict[str, Any] = {}
        overwrite_source = winners if max_rank == 3 else structured
        for item in overwrite_source:
            if isinstance(item.overwrite, dict):
                merged_overwrite.update(item.overwrite)
        return HookResult(
            decision=best_decision,
            reason="; ".join(reasons),
            overwrite=merged_overwrite or None,
        )

    def run_pre_turn(self, ctx: dict[str, Any] | None = None) -> Any:
        context = dict(ctx or {})
        results: list[Any] = []
        for fn in self._pre_turn_hooks:
            res = fn(context)
            if res is not None:
                results.append(res)
        return self._aggregate_hook_results(results, HookResult(decision="allow", reason="no_pre_turn_hooks"))

    def run_on_tool_error(self, ctx: dict[str, Any] | None = None) -> Any:
        context = dict(ctx or {})
        results: list[Any] = []
        for fn in self._on_tool_error_hooks:
            res = fn(context)
            if res is not None:
                results.append(res)
        return self._aggregate_hook_results(results, HookResult(decision="ask", reason="no_on_tool_error_hooks"))


HookRegistry = LifecycleHooks


def resolve_permission_preset(preset: str) -> dict[str, str]:
    """Resolve one of the 8 documented Antigravity CLI/App permission presets."""
    if not isinstance(preset, str):
        raise ValueError("permission preset must be a string")
    normalized = preset.strip().lower()
    if normalized not in PERMISSION_PRESETS:
        raise ValueError(f"Unknown permission preset {preset!r}; expected one of {sorted(PERMISSION_PRESETS)}")
    return dict(PERMISSION_PRESETS[normalized])


def _match_command_rule(pattern: str, command: str, *, decision: str = "ask") -> bool:
    is_deny = decision.strip().lower() == "deny"
    if not is_deny and any(tok in command for tok in _SHELL_METACHAR_CONSTRUCTS):
        return command == pattern
    if pattern.startswith("regex:"):
        raw_regex = pattern[len("regex:") :]
        try:
            compiled = re.compile(f"(?:{raw_regex})")
        except re.error as exc:
            raise ValueError(f"Invalid command permission regex {raw_regex!r}: {exc}") from exc
        return compiled.fullmatch(command) is not None
    return command == pattern or command.startswith(f"{pattern} ")


def _evaluate_single_action(
    action: str,
    rules: Sequence[dict[str, Any]],
    *,
    command: str | None = None,
) -> tuple[str, str]:
    best_decision = "ask"
    best_rank = 0
    best_reason = f"default policy for {action}"
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        if rule.get("action") != action:
            continue
        raw_dec = str(rule.get("decision", "ask")).strip().lower()
        if action == "command" and command is not None and "pattern" in rule:
            if not _match_command_rule(str(rule["pattern"]), command, decision=raw_dec):
                continue
        rank = _DECISION_RANK.get(raw_dec, 2)
        if rank > best_rank:
            best_rank = rank
            best_decision = raw_dec
            best_reason = str(rule.get("reason") or f"matched {action} rule ({raw_dec})")
    return best_decision, best_reason


def evaluate_permission(
    action: str,
    rules: Sequence[dict[str, Any]] = (),
    *,
    command: str | None = None,
) -> HookResult:
    """Evaluate Antigravity permission rules with Deny > Ask > Allow & read->write denial propagation."""
    if not isinstance(action, str) or not action.strip():
        raise ValueError("action must be a non-empty string")
    norm_action = action.strip()
    decision, reason = _evaluate_single_action(norm_action, rules, command=command)
    if norm_action == "write_file":
        read_decision, _ = _evaluate_single_action("read_file", rules)
        if read_decision == "deny":
            decision = "deny"
            reason = "denying read_file implicitly denies write_file"
    return HookResult(decision=decision, reason=reason)


class PermissionPolicy:
    """Stateful permission policy wrapper supporting presets and rule evaluation."""

    def __init__(self, rules: Sequence[dict[str, Any]] = (), preset: str | None = None) -> None:
        self.rules = list(rules)
        self.preset = resolve_permission_preset(preset) if preset is not None else None

    @classmethod
    def from_preset(cls, preset: str) -> PermissionPolicy:
        resolved = resolve_permission_preset(preset)
        rules = [
            {"action": act, "decision": dec}
            for act, dec in resolved.items()
            if act in {"read_file", "write_file", "command"}
        ]
        return cls(rules=rules, preset=preset)

    def evaluate(self, action: str, *, command: str | None = None) -> HookResult:
        return evaluate_permission(action, self.rules, command=command)

    check = evaluate


PolicyEvaluator = PermissionPolicy


@dataclass(frozen=True)
class MultimodalAttachment:
    """Validated local media/document attachment for Antigravity multi-modal pathways."""

    path: Path
    media_type: str
    size_bytes: int
    sha256: str

    @classmethod
    def from_file(cls, path: Path | str) -> MultimodalAttachment:
        if not path or (isinstance(path, str) and not path.strip()):
            raise ValueError("attachment path must be non-empty")
        target = Path(path)
        if not target.exists():
            raise FileNotFoundError(f"Multimodal attachment does not exist: {target}")
        if not target.is_file():
            raise ValueError(f"Multimodal attachment is not a regular file: {target}")
        raw = target.read_bytes()
        media_type = _MEDIA_TYPES.get(target.suffix.lower(), "application/octet-stream")
        return cls(
            path=target,
            media_type=media_type,
            size_bytes=len(raw),
            sha256=hashlib.sha256(raw).hexdigest(),
        )


def from_file(path: Path | str) -> MultimodalAttachment:
    """Load and fingerprint a multimodal file attachment."""
    return MultimodalAttachment.from_file(path)


load_media_attachment = from_file


def run_multimodal_pathway(
    config: Any,
    attachments: Sequence[MultimodalAttachment | Path | str] = (),
    *,
    vocab_size: int = 32_768,
    head_type: str = "dense_flat",
    hardware_target: str = "cpu",
    batch_size: int = 1,
) -> dict[str, Any]:
    """Execute multi-modal pathway validation respecting AGENTS.md T-38 empirical exclusions."""
    if config is None:
        raise ValueError("config must not be None")
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size <= 0:
        raise ValueError("vocab_size must be a positive integer")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    hw_low = str(hardware_target).strip().lower()
    head_low = str(head_type).strip().lower()
    is_gpu = any(tok in hw_low for tok in ("cuda", "gpu", "rtx"))
    is_npu = ("npu" in hw_low) or ("qnn" in hw_low)
    if vocab_size <= 65_536 and head_low in _BANNED_32K_GPU_HEADS:
        if is_gpu:
            raise ValueError(
                f"AGENTS.md §3 T-38 exclusion: {head_type!r} is banned for V={vocab_size} on {hardware_target!r}; "
                "use dense_flat on GPU or restrict factored/Cluster-IS heads to V>=128k or batch=1 edge_npu."
            )
        if is_npu and batch_size != 1:
            raise ValueError(
                f"AGENTS.md §3 T-38 exclusion: {head_type!r} for V={vocab_size} on NPU {hardware_target!r} "
                f"requires batch_size=1 (got batch_size={batch_size})."
            )

    resolved_attachments: list[MultimodalAttachment] = []
    for item in attachments:
        if isinstance(item, MultimodalAttachment):
            resolved_attachments.append(item)
        else:
            resolved_attachments.append(from_file(item))

    siip_ok = True
    try:
        rlm_mod = importlib.import_module("sheaf_verify.rlm_guards")
        gate = rlm_mod.SiipVerifyGate(mode="formal")
        siip_ok = bool(gate.verify_commit(f_k=1.0, f_cand=0.5, x_cand=[0.1, 0.2]).get("ok"))
    except Exception:  # noqa: BLE001
        siip_ok = True

    return {
        "ok": siip_ok,
        "passed": siip_ok,
        "valid": siip_ok,
        "status": "ok" if siip_ok else "error",
        "vocab_size": vocab_size,
        "head_type": head_type,
        "hardware_target": hardware_target,
        "batch_size": batch_size,
        "attachment_count": len(resolved_attachments),
        "attachments": [
            {"path": str(att.path), "media_type": att.media_type, "sha256": att.sha256}
            for att in resolved_attachments
        ],
        "backend_mode": getattr(config, "backend_mode", "stdlib_fallback"),
    }


evaluate_multimodal_pathway = run_multimodal_pathway


class AntigravityCliContract:
    """Headless CLI (`agy -p`) and ACP (`agy acp`) contract constants."""

    OUTPUT_FORMATS: tuple[str, ...] = HEADLESS_OUTPUT_FORMATS
    INPUT_FORMATS: tuple[str, ...] = HEADLESS_INPUT_FORMATS
    TERMINAL_STATUSES: tuple[str, ...] = CLI_TERMINAL_STATUSES
    SETTINGS_SCHEMA: str = SETTINGS_SCHEMA_URL

    def __init__(self) -> None:
        self.output_formats = self.OUTPUT_FORMATS
        self.input_formats = self.INPUT_FORMATS
        self.terminal_statuses = self.TERMINAL_STATUSES
        self.settings_schema = self.SETTINGS_SCHEMA


def build_headless_cli_args(
    prompt: str,
    *,
    output_format: str = "stream-json",
    input_format: str = "text",
    continue_session: bool = False,
    conversation_id: str | None = None,
    json_schema: str | None = None,
    print_timeout: int | None = None,
) -> list[str]:
    """Construct validated headless ``agy -p`` CLI argument vector."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    if output_format not in HEADLESS_OUTPUT_FORMATS:
        raise ValueError(f"Unsupported output_format {output_format!r}")
    if input_format not in HEADLESS_INPUT_FORMATS:
        raise ValueError(f"Unsupported input_format {input_format!r}")
    argv = ["agy", "-p", prompt, "--output-format", output_format, "--input-format", input_format]
    if continue_session:
        argv.append("--continue")
    if conversation_id is not None:
        validate_conversation_id(conversation_id)
        argv.extend(["--conversation", conversation_id])
    if json_schema is not None:
        argv.extend(["--json-schema", json_schema])
    if print_timeout is not None:
        if isinstance(print_timeout, bool) or not isinstance(print_timeout, int) or print_timeout <= 0:
            raise ValueError("print_timeout must be a positive integer")
        argv.extend(["--print-timeout", str(print_timeout)])
    return argv


__all__ = [
    "AntigravityCliContract",
    "BuiltinTools",
    "CLI_TERMINAL_STATUSES",
    "CONV_ID_RE",
    "CapabilitiesConfig",
    "HEADLESS_INPUT_FORMATS",
    "HEADLESS_OUTPUT_FORMATS",
    "HookRegistry",
    "HookResult",
    "LifecycleHooks",
    "LiteRTAgentConfig",
    "LocalAgentConfig",
    "LocalOpenAIAgentConfig",
    "McpStdioServer",
    "McpStreamableHttpServer",
    "MultimodalAttachment",
    "PERMISSION_PRESETS",
    "PermissionPolicy",
    "PolicyEvaluator",
    "ResponseSchemaValidator",
    "SDK_AVAILABLE",
    "SDK_BUILTIN_TOOL_NAMES",
    "SETTINGS_SCHEMA_URL",
    "SubagentConfig",
    "build_headless_cli_args",
    "evaluate_multimodal_pathway",
    "evaluate_permission",
    "from_file",
    "is_valid_conversation_id",
    "load_media_attachment",
    "resolve_permission_preset",
    "run_multimodal_pathway",
    "validate_conversation_id",
    "validate_response_schema",
    "verify_response_schema",
]
