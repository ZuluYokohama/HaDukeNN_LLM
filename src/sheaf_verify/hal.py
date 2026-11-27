"""Fail-closed HAL study profile.

A hardware / microbench study is selected with ``--profile hal`` and is
evaluated only by HAL0–HAL6. Training studies keep the G/H/P graph: this
profile cannot bypass those gates. Every HAL criterion recomputes from
registered payloads or a live probe; self-reported summary fields are not
accepted as evidence.
"""

from __future__ import annotations

import math
import platform
import subprocess
from collections.abc import Callable

from sheaf_verify.checks import (
    bad,
    code_tree_criteria,
    environment_criteria,
    kernel_criteria,
    ok,
)
from sheaf_verify.context import Context
from sheaf_verify.hashing import SheafError, parse_json_bytes

HAL_CLAIM_FORMAT = "sheaf-verify.hal-claim.v1"
HAL_PARITY_FORMAT = "sheaf-verify.hal-parity.v1"
HAL_TIMINGS_FORMAT = "sheaf-verify.hal-timings.v1"
HAL_GUARDS_FORMAT = "sheaf-verify.hal-guards.v1"

BASELINE_SERIAL = "serial_python_loop"
BASELINE_NUMPY = "packed_numpy"
REQUIRED_BASELINES = (BASELINE_SERIAL, BASELINE_NUMPY)
SUBJECT_C11 = "c11_tiled_cce"

# Studies may tighten, never loosen, this absolute parity cap.
HAL_MAX_ABS = 1e-4
MIN_TIMING_REPEATS = 3
MAX_PARITY_VALUES = 1_000_000

ProbeFn = Callable[[], dict[str, str]]
_PROBE: ProbeFn | None = None


def set_probe(fn: ProbeFn | None) -> None:
    """Tests replace the live environment probe. Production leaves this unset."""
    global _PROBE
    _PROBE = fn


def probe_environment() -> dict[str, str]:
    if _PROBE is not None:
        return _PROBE()
    cuda = _cmd_text(["nvcc", "--version"], pattern="release")
    driver, gpu, power = _nvidia_smi()
    return {
        "python": platform.python_version(),
        "cuda": cuda or "unknown",
        "driver": driver or "unknown",
        "gpu": gpu or "unknown",
        "platform": platform.platform(),
        "power": power or "unobserved",
    }


def _cmd_text(argv: list[str], pattern: str | None = None) -> str | None:
    try:
        proc = subprocess.run(argv, check=False, capture_output=True, text=True, timeout=8)
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        return None
    if pattern is None:
        return text.strip() or None
    for line in text.splitlines():
        if pattern in line:
            return line.strip()
    return text.strip() or None


def _nvidia_smi() -> tuple[str | None, str | None, str | None]:
    raw = _cmd_text([
        "nvidia-smi",
        "--query-gpu=driver_version,name,memory.total,compute_cap,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ])
    if not raw:
        return None, None, None
    parts = [item.strip() for item in raw.splitlines()[0].split(",")]
    if len(parts) < 4:
        return None, None, None
    driver, name, memory, cap = parts[0], parts[1], parts[2], parts[3]
    gpu = f"{name} ({memory} MiB, sm_{cap.replace('.', '')})" if name else None
    power = parts[5] if len(parts) > 5 else None
    return driver, gpu, power


def evaluate_hal0(ctx: Context) -> list:
    payload = ctx.payload("claim")
    if b"\x00" in payload:
        return [bad("hal.claim.text", "claim contains NUL")]
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        return [bad("hal.claim.text", "claim is not UTF-8")]
    if _contains_sha256(text):
        return [bad("hal.claim.frozen", "claim contains a 64-character hex digest")]
    try:
        document = parse_json_bytes(payload, "claim")
    except SheafError as error:
        return [bad("hal.claim.format", str(error))]
    if not isinstance(document, dict) or document.get("format") != HAL_CLAIM_FORMAT:
        return [bad("hal.claim.format", "claim format is not sheaf-verify.hal-claim.v1")]
    criteria = []
    if document.get("study_profile") != "hal":
        criteria.append(bad("hal.claim.profile", "study_profile must be hal and must be explicit"))
    else:
        criteria.append(ok("hal.claim.profile", "study_profile is hal"))
    hypothesis = document.get("hypothesis_id")
    description = document.get("description")
    if isinstance(hypothesis, str) and hypothesis.strip() and isinstance(description, str) and len(description.strip()) >= 20:
        criteria.append(ok("hal.claim.identity", hypothesis.strip()))
    else:
        criteria.append(bad("hal.claim.identity", "hypothesis_id and a description of at least 20 characters are required"))
    performance = document.get("performance") if isinstance(document.get("performance"), dict) else {}
    reported = performance.get("baselines_reported")
    if not isinstance(reported, list) or set(reported) < set(REQUIRED_BASELINES):
        criteria.append(bad(
            "hal.claim.baselines",
            "claim must report both packed_numpy and serial_python_loop baselines",
        ))
    else:
        criteria.append(ok("hal.claim.baselines", "both baselines are declared"))
    baseline = performance.get("baseline")
    if baseline not in REQUIRED_BASELINES:
        criteria.append(bad("hal.claim.baseline", "claim baseline must be serial_python_loop or packed_numpy"))
    else:
        criteria.append(ok("hal.claim.baseline", str(baseline)))
    if performance.get("metric") != "speedup" or performance.get("direction") != "greater":
        criteria.append(bad("hal.claim.direction", "performance metric must be speedup in the greater direction"))
    else:
        criteria.append(ok("hal.claim.direction", "speedup, greater"))
    min_speedup = performance.get("min_speedup")
    if isinstance(min_speedup, bool) or not isinstance(min_speedup, (int, float)) or float(min_speedup) <= 0:
        criteria.append(bad("hal.claim.min_speedup", "min_speedup must be a positive number"))
    else:
        criteria.append(ok("hal.claim.min_speedup", str(float(min_speedup))))
    if performance.get("subject") != SUBJECT_C11:
        criteria.append(bad("hal.claim.subject", "subject must be c11_tiled_cce"))
    else:
        criteria.append(ok("hal.claim.subject", SUBJECT_C11))
    parity = document.get("parity") if isinstance(document.get("parity"), dict) else {}
    max_abs = parity.get("max_abs", HAL_MAX_ABS)
    if isinstance(max_abs, bool) or not isinstance(max_abs, (int, float)) or float(max_abs) <= 0 or float(max_abs) > HAL_MAX_ABS + 1e-18:
        criteria.append(bad("hal.claim.parity", f"parity max_abs must be positive and at most {HAL_MAX_ABS}"))
    else:
        criteria.append(ok("hal.claim.parity", str(float(max_abs))))
    if ctx.has("metrics"):
        claim_time = ctx.manifest("claim")["registered_at"]
        metrics_time = ctx.manifest("metrics")["registered_at"]
        if isinstance(claim_time, str) and isinstance(metrics_time, str) and claim_time < metrics_time:
            criteria.append(ok("hal.claim.before_metrics", "the claim was registered before the metrics"))
        else:
            criteria.append(bad("hal.claim.before_metrics", "the claim was not registered before the metrics"))
    return criteria


def evaluate_hal1(ctx: Context) -> list:
    document = ctx.parse("environment")
    criteria = list(environment_criteria(document))
    if any(not item.as_dict()["passed"] for item in criteria):
        return criteria
    live = probe_environment()
    for field in ("python", "cuda", "driver", "gpu", "platform"):
        recorded = str(document.get(field, "")).strip()
        observed = str(live.get(field, "")).strip()
        if not observed or observed.casefold() == "unknown":
            criteria.append(bad(f"environment.live.{field}", f"live probe could not observe {field}"))
        elif _norm(recorded) != _norm(observed):
            criteria.append(bad(
                f"environment.live.{field}",
                f"registered {field} does not match the live probe",
            ))
        else:
            criteria.append(ok(f"environment.live.{field}", observed))
    return criteria


def evaluate_hal2(ctx: Context) -> list:
    listing = ctx.parse("code")
    if not isinstance(listing, list):
        return [bad("code.listing", "code listing is not a list")]
    digest = ctx.manifest("code")["payload_sha256"]
    files = [(entry["path"], entry["mode"], ctx.snapshot_bytes(digest, entry["path"])) for entry in listing]
    return list(code_tree_criteria(ctx.manifest("code"), listing, files))


def evaluate_hal3(ctx: Context) -> list:
    criteria = list(_kernel_role(ctx, "kernel"))
    if ctx.has("kernel.cuda"):
        criteria.extend(_kernel_role(ctx, "kernel.cuda"))
    return criteria


def _kernel_role(ctx: Context, role: str) -> list:
    kernel = ctx.manifest(role)
    input_payloads = set()
    for manifest_sha in kernel.get("inputs", []):
        input_payloads.add(ctx.store.read_manifest(manifest_sha)["payload_sha256"])
    return kernel_criteria(
        kernel.get("metadata"),
        kernel["payload_sha256"],
        kernel["size_bytes"],
        input_payloads,
    )


def evaluate_hal4(ctx: Context) -> list:
    try:
        document = ctx.parse("log.parity")
    except SheafError as error:
        return [bad("hal.parity.format", str(error))]
    if not isinstance(document, dict) or document.get("format") != HAL_PARITY_FORMAT:
        return [bad("hal.parity.format", "parity log is not sheaf-verify.hal-parity.v1")]
    cases = document.get("cases")
    if not isinstance(cases, list) or not cases:
        return [bad("hal.parity.cases", "parity log has no cases")]
    max_abs = _claim_max_abs(ctx)
    criteria = [ok("hal.parity.format", HAL_PARITY_FORMAT)]
    for index, case in enumerate(cases):
        criteria.extend(_parity_case(case, index, max_abs))
    return criteria


def evaluate_hal5(ctx: Context) -> list:
    claim = parse_json_bytes(ctx.payload("claim"), "claim")
    if not isinstance(claim, dict):
        return [bad("hal.perf.claim", "claim is not a JSON object")]
    performance = claim.get("performance") if isinstance(claim.get("performance"), dict) else {}
    baseline_name = performance.get("baseline")
    min_speedup = performance.get("min_speedup")
    subject = performance.get("subject")
    try:
        document = ctx.parse("log.timings")
    except SheafError as error:
        return [bad("hal.perf.format", str(error))]
    if not isinstance(document, dict) or document.get("format") != HAL_TIMINGS_FORMAT:
        return [bad("hal.perf.format", "timings log is not sheaf-verify.hal-timings.v1")]
    cases = document.get("cases")
    if not isinstance(cases, list) or not cases:
        return [bad("hal.perf.cases", "timings log has no cases")]
    if subject != SUBJECT_C11 or baseline_name not in REQUIRED_BASELINES:
        return [bad("hal.perf.claim", "claim subject or baseline is not usable")]
    if isinstance(min_speedup, bool) or not isinstance(min_speedup, (int, float)) or float(min_speedup) <= 0:
        return [bad("hal.perf.claim", "claim min_speedup is not a positive number")]
    criteria = [ok("hal.perf.format", HAL_TIMINGS_FORMAT)]
    claimed_speedups: list[float] = []
    for index, case in enumerate(cases):
        label = _case_name(case, index)
        if not isinstance(case, dict):
            criteria.append(bad(f"hal.perf.{label}", "case is not an object"))
            continue
        subject_times = _repeats(case.get("subject_ms"))
        if subject_times is None:
            criteria.append(bad(f"hal.perf.{label}.subject", "subject_ms must be at least 3 raw finite repeats"))
            continue
        criteria.append(ok(f"hal.perf.{label}.subject", _stats_detail(subject_times)))
        baselines = case.get("baselines_ms")
        if not isinstance(baselines, dict):
            criteria.append(bad(f"hal.perf.{label}.baselines", "baselines_ms is missing"))
            continue
        for required in REQUIRED_BASELINES:
            times = _repeats(baselines.get(required))
            if times is None:
                criteria.append(bad(
                    f"hal.perf.{label}.{required}",
                    f"{required} must provide at least {MIN_TIMING_REPEATS} raw finite repeats",
                ))
            else:
                speedup = _mean(times) / max(_mean(subject_times), 1e-12)
                criteria.append(ok(f"hal.perf.{label}.{required}", f"{_stats_detail(times)}; speedup {speedup:.4f}"))
                if required == baseline_name:
                    claimed_speedups.append(speedup)
        # Reject trusting a pre-written speedup field: if present it must match the recomputation.
        reported = case.get("speedup")
        if reported is not None and baseline_name in baselines and _repeats(baselines.get(baseline_name)):
            recomputed = _mean(_repeats(baselines[baseline_name])) / max(_mean(subject_times), 1e-12)
            if not isinstance(reported, (int, float)) or abs(float(reported) - recomputed) > 1e-6 * max(1.0, abs(recomputed)):
                criteria.append(bad(f"hal.perf.{label}.speedup_field", "self-reported speedup does not match raw repeats"))
    if not claimed_speedups:
        criteria.append(bad("hal.perf.threshold", "no case produced a speedup against the claim baseline"))
        return criteria
    worst = min(claimed_speedups)
    if worst + 1e-12 < float(min_speedup):
        criteria.append(bad(
            "hal.perf.threshold",
            f"recomputed speedup vs {baseline_name} is {worst:.4f}, below claimed min_speedup {float(min_speedup)}",
        ))
    else:
        criteria.append(ok(
            "hal.perf.threshold",
            f"recomputed speedup vs {baseline_name} is {worst:.4f} (min_speedup {float(min_speedup)})",
        ))
    return criteria


def evaluate_hal6(ctx: Context) -> list:
    try:
        document = ctx.parse("log.guards")
    except SheafError as error:
        return [bad("hal.guard.format", str(error))]
    if not isinstance(document, dict) or document.get("format") != HAL_GUARDS_FORMAT:
        return [bad("hal.guard.format", "guards log is not sheaf-verify.hal-guards.v1")]
    criteria = [ok("hal.guard.format", HAL_GUARDS_FORMAT)]
    thermal = document.get("thermal") if isinstance(document.get("thermal"), dict) else {}
    stop = thermal.get("stop_celsius")
    if isinstance(stop, bool) or not isinstance(stop, (int, float)) or float(stop) <= 0:
        criteria.append(bad("hal.guard.stop", "thermal stop_celsius is missing"))
        return criteria
    samples = thermal.get("samples")
    temps = []
    if not isinstance(samples, list) or len(samples) < 2:
        criteria.append(bad("hal.guard.thermal", "at least two thermal samples are required"))
    else:
        for item in samples:
            if isinstance(item, dict) and isinstance(item.get("celsius"), (int, float)) and not isinstance(item.get("celsius"), bool):
                temps.append(float(item["celsius"]))
        if len(temps) < 2:
            criteria.append(bad("hal.guard.thermal", "thermal samples must include numeric celsius readings"))
        else:
            peak = max(temps)
            if peak >= float(stop):
                criteria.append(bad("hal.guard.thermal", f"peak {peak} C reached the stop threshold {float(stop)} C"))
            else:
                criteria.append(ok("hal.guard.thermal", f"{len(temps)} samples, peak {peak} C < stop {float(stop)} C"))
    power = document.get("power") if isinstance(document.get("power"), dict) else {}
    if power.get("observed") is True:
        draw = power.get("draw_watts")
        if isinstance(draw, bool) or not isinstance(draw, (int, float)):
            criteria.append(bad("hal.guard.power", "power was marked observed but draw_watts is missing"))
        else:
            criteria.append(ok("hal.guard.power", f"{float(draw)} W"))
    else:
        reason = power.get("reason")
        probe = power.get("probe")
        if isinstance(reason, str) and reason.strip() and isinstance(probe, str) and probe.strip():
            criteria.append(ok("hal.guard.power", "power unobserved, probe and reason recorded"))
        else:
            criteria.append(bad(
                "hal.guard.power",
                "unobserved power must record the probe command and the reason it failed",
            ))
    return criteria


def cce_mup_loss_f64(hidden, embedding, targets, mup_base_d: float) -> list[float]:
    """Stdlib float64 Cut-Cross-Entropy with muP logit scale. Does not trust a payload loss."""
    rows = _matrix(hidden, "H")
    table = _matrix(embedding, "E")
    labels = _int_vector(targets, "y")
    if not rows or not table:
        raise SheafError("parity case is empty")
    width = len(rows[0])
    if any(len(row) != width for row in rows) or any(len(row) != width for row in table):
        raise SheafError("H and E width must match")
    if len(labels) != len(rows):
        raise SheafError("y length must match H rows")
    vocab = len(table)
    if any(label < 0 or label >= vocab for label in labels):
        raise SheafError("target index is out of range")
    scale = float(mup_base_d) / float(width)
    losses = []
    for row, label in zip(rows, labels, strict=True):
        logits = [scale * _dot(row, weight) for weight in table]
        losses.append(_logsumexp(logits) - logits[label])
    return losses


def _parity_case(case, index: int, max_abs: float) -> list:
    label = _case_name(case, index)
    if not isinstance(case, dict):
        return [bad(f"hal.parity.{label}", "case is not an object")]
    if case.get("op") != "cce_mup_fwd":
        return [bad(f"hal.parity.{label}.op", "unsupported op; only cce_mup_fwd is implemented")]
    inputs = case.get("inputs") if isinstance(case.get("inputs"), dict) else {}
    outputs = case.get("kernel_outputs") if isinstance(case.get("kernel_outputs"), dict) else {}
    try:
        count = _count_values(inputs) + _count_values(outputs)
        if count > MAX_PARITY_VALUES:
            raise SheafError("parity case exceeds the value cap")
        reference = cce_mup_loss_f64(
            inputs.get("H"),
            inputs.get("E"),
            inputs.get("y"),
            float(inputs.get("mup_base_d")),
        )
        reported = _float_vector(outputs.get("loss"), "loss")
        if len(reported) != len(reference):
            raise SheafError("kernel loss length does not match the reference")
    except (SheafError, TypeError, ValueError, OverflowError) as error:
        return [bad(f"hal.parity.{label}", str(error))]
    hidden = inputs.get("H")
    width = len(hidden[0]) if isinstance(hidden, list) and hidden and isinstance(hidden[0], list) and hidden[0] else 0
    try:
        scale = float(inputs.get("mup_base_d")) / float(width)
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        scale = float("nan")
    errors = [abs(left - right) for left, right in zip(reported, reference, strict=True)]
    finite = (
        math.isfinite(scale)
        and all(math.isfinite(item) for item in reference)
        and all(math.isfinite(item) for item in reported)
        and all(math.isfinite(item) for item in errors)
    )
    if not finite:
        return [bad(f"hal.parity.{label}", "non-finite scale, reference loss, kernel loss, or parity error")]
    peak = max(errors)
    if peak > max_abs:
        return [bad(f"hal.parity.{label}", f"max abs error {peak} exceeds {max_abs} vs float64 reference")]
    return [ok(f"hal.parity.{label}", f"max abs error {peak} vs float64 reference (cap {max_abs})")]


def _claim_max_abs(ctx: Context) -> float:
    cap = HAL_MAX_ABS
    if not ctx.has("claim"):
        return cap
    try:
        claim = parse_json_bytes(ctx.payload("claim"), "claim")
    except SheafError:
        return cap
    if not isinstance(claim, dict):
        return cap
    parity = claim.get("parity") if isinstance(claim.get("parity"), dict) else {}
    value = parity.get("max_abs", cap)
    if isinstance(value, (int, float)) and not isinstance(value, bool) and 0 < float(value) <= cap:
        return float(value)
    return cap


def _repeats(value) -> list[float] | None:
    if not isinstance(value, list) or len(value) < MIN_TIMING_REPEATS:
        return None
    times = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item) or float(item) <= 0:
            return None
        times.append(float(item))
    return times


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    centre = _mean(values)
    return math.sqrt(sum((item - centre) ** 2 for item in values) / (len(values) - 1))


def _stats_detail(values: list[float]) -> str:
    return f"n={len(values)} mean={_mean(values):.6g} std={_std(values):.6g}"


def _case_name(case, index: int) -> str:
    if isinstance(case, dict) and isinstance(case.get("name"), str) and case["name"].strip():
        return case["name"].strip()
    return f"case{index}"


def _contains_sha256(text: str) -> bool:
    hexdigits = set("0123456789abcdefABCDEF")
    run = 0
    for char in text:
        if char in hexdigits:
            run += 1
            if run >= 64:
                return True
        else:
            run = 0
    return False


def _norm(value: str) -> str:
    return " ".join(value.strip().casefold().split())


def _matrix(value, label: str) -> list[list[float]]:
    if not isinstance(value, list) or not value:
        raise SheafError(f"{label} must be a non-empty matrix")
    rows = []
    width = None
    for row in value:
        vector = _float_vector(row, label)
        if width is None:
            width = len(vector)
        elif len(vector) != width:
            raise SheafError(f"{label} is ragged")
        rows.append(vector)
    return rows


def _float_vector(value, label: str) -> list[float]:
    if not isinstance(value, list) or not value:
        raise SheafError(f"{label} must be a non-empty vector")
    out = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item):
            raise SheafError(f"{label} contains a non-finite number")
        out.append(float(item))
    return out


def _int_vector(value, label: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise SheafError(f"{label} must be a non-empty vector")
    out = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise SheafError(f"{label} must be integers")
        out.append(int(item))
    return out


def _count_values(value) -> int:
    if isinstance(value, list):
        return sum(_count_values(item) for item in value)
    if isinstance(value, dict):
        return sum(_count_values(item) for item in value.values())
    return 1


def _dot(left: list[float], right: list[float]) -> float:
    return math.fsum(a * b for a, b in zip(left, right, strict=True))


def _logsumexp(values: list[float]) -> float:
    peak = max(values)
    return peak + math.log(math.fsum(math.exp(item - peak) for item in values))

