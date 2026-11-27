"""Live and replay evaluation for sheaf-verify campaign modes.

JSON-only evaluation is always labeled *replay*: the criteria are scored from a
registered document, not from a fresh measurement. ``--live`` re-runs an
in-process suite for the modes that have one (T-46 HAL, T-49 HST toy campaign)
and writes a sibling live JSON. The sealed result files are never overwritten.

T-46 live measurements on this host are for the machine that is actually
running. The in-process Zenbook looped INT8 engine is a host simulation, not a
live Snapdragon NPU. WBeast / Zenbook hardware fields in the sealed document
remain replay. GPU work stays on the host; this module does not use a container
runtime.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sheaf_verify.attestation import validate_strict_json_bytes
from sheaf_verify.campaign import (
    audit_t38_prereg,
    verify_hg_v3_gate,
    verify_t37_grid,
    verify_t40_local_stack,
    verify_t41_frontier5070,
    verify_t42_exec_bridge,
    verify_t43_all_realms,
    verify_t44_glassbox,
    verify_t45_sheaf_bridge,
    verify_t46_tri_machine_hal,
    verify_t47_cosimilar_trace,
    verify_t48_roadmap_complete,
    verify_t49_hst_foundation,
)
from sheaf_verify.checks import bad, ok
from sheaf_verify.hashing import SheafError, reject_symlink

ROOT = Path(__file__).resolve().parents[2]
SEALED_T46 = ROOT / "iterations" / "t46-tri-machine-hal" / "T46_TRI_MACHINE_RESULTS.json"
DEFAULT_LIVE_T46 = ROOT / "iterations" / "t46-tri-machine-hal" / "T46_TRI_MACHINE_RESULTS.lapbeast-live.json"
SEALED_T49 = ROOT / "iterations" / "t49-hst-foundation-stage" / "T49_HST_FOUNDATION_RESULTS.json"
DEFAULT_LIVE_T49 = ROOT / "iterations" / "t49-hst-foundation-stage" / "T49_HST_FOUNDATION_RESULTS.lapbeast-live.json"

LIVE_MODES = frozenset({"t46-tri-machine-hal", "t49-hst-foundation"})
REPLAY_LABEL = (
    "JSON-only replay: criteria were scored from the registered document, not from a fresh in-process measurement."
)
LIVE_LABEL = "Live in-process measurement. Sealed JSON was not overwritten."

JSON_MODES = {
    "t37-grid": lambda doc, addendum=None: _triple(verify_t37_grid(doc)),
    "t38-prereg": lambda doc, addendum=None: audit_t38_prereg(doc, addendum),
    "t40-stack": lambda doc, addendum=None: verify_t40_local_stack(doc),
    "t41-frontier5070": lambda doc, addendum=None: verify_t41_frontier5070(doc),
    "t42-exec-bridge": lambda doc, addendum=None: verify_t42_exec_bridge(doc),
    "t43-all-realms": lambda doc, addendum=None: verify_t43_all_realms(doc),
    "t44-glassbox": lambda doc, addendum=None: verify_t44_glassbox(doc),
    "t45-sheaf-bridge": lambda doc, addendum=None: verify_t45_sheaf_bridge(doc),
    "t46-tri-machine-hal": lambda doc, addendum=None: verify_t46_tri_machine_hal(doc),
    "t47-cosimilar-trace": lambda doc, addendum=None: verify_t47_cosimilar_trace(doc),
    "t48-roadmap-complete": lambda doc, addendum=None: verify_t48_roadmap_complete(doc),
    "t49-hst-foundation": lambda doc, addendum=None: verify_t49_hst_foundation(doc),
}


def _triple(result):
    if len(result) == 3:
        return result
    ok_flag, criteria = result
    return ok_flag, criteria, []


def in_ci() -> bool:
    return bool(os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_json(document: dict) -> str:
    return hashlib.sha256(json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def t46_hardware_present() -> bool:
    try:
        mod = importlib.import_module("hadukenn_core.tri_machine")

        return mod.get_tri_machine_lib() is not None
    except Exception:
        return False


def resolve_want_live(*, mode: str, live: bool, replay: bool) -> bool:
    if live and replay:
        raise SheafError("--live and --replay cannot be combined")
    if replay:
        return False
    if live:
        if mode not in LIVE_MODES:
            raise SheafError(
                f"--live is not implemented for campaign mode {mode!r}; "
                "only t46-tri-machine-hal and t49-hst-foundation can re-measure. "
                "JSON-only evaluation is available as replay."
            )
        return True
    # Default is replay. Auto-live was rejected: it would change fail-closed
    # CLI exit status for JSON-only invocations and could overwrite the live
    # sibling JSON without an explicit --live.
    return False


def _criteria_dicts(criteria) -> list[dict]:
    out = []
    for item in criteria:
        if hasattr(item, "as_dict"):
            out.append(item.as_dict())
        else:
            out.append({"id": item["id"], "passed": bool(item["passed"]), "detail": item.get("detail", "")})
    return out


def _load_json(path: Path, label: str) -> dict:
    if path is None or not path.is_file():
        raise SheafError(f"--path is required for campaign mode {label}")
    return validate_strict_json_bytes(reject_symlink(path, "campaign JSON").read_bytes())


def evaluate_campaign(
    *,
    mode: str,
    path: Path | None,
    addendum: Path | None = None,
    live: bool = False,
    replay: bool = False,
    live_out: Path | None = None,
    register_store: Path | None = None,
) -> dict[str, Any]:
    want_live = resolve_want_live(mode=mode, live=live, replay=replay)
    add_doc = None
    if addendum is not None:
        add_doc = validate_strict_json_bytes(reject_symlink(addendum, "addendum JSON").read_bytes())
    if mode == "hg-v3":
        doc = _load_json(path, mode)
        ok_flag, criteria = verify_hg_v3_gate(doc)
        return _replay_payload(mode, path, ok_flag, criteria, [])

    if mode not in JSON_MODES:
        raise SheafError(f"unknown campaign mode {mode!r}")

    if not want_live:
        doc = _load_json(path, mode)
        ok_flag, criteria, advisories = _triple(JSON_MODES[mode](doc, add_doc))
        return _replay_payload(mode, path, ok_flag, criteria, advisories)

    if mode == "t46-tri-machine-hal":
        return run_t46_live(sealed_path=path or SEALED_T46, live_out=live_out, register_store=register_store)
    if mode == "t49-hst-foundation":
        return run_t49_live(sealed_path=path or SEALED_T49, live_out=live_out)
    raise SheafError(f"--live is not implemented for {mode!r}")


def _replay_payload(mode: str, path: Path | None, ok_flag: bool, criteria, advisories) -> dict[str, Any]:
    payload = {
        "evaluation": "replay",
        "evaluation_label": REPLAY_LABEL,
        "mode": mode,
        "ok": bool(ok_flag),
        "criteria": _criteria_dicts(criteria),
        "advisories": list(advisories or []),
    }
    if path is not None and path.is_file():
        payload["path"] = str(path)
        payload["document_sha256"] = sha256_file(path)
    return payload


def _refuse_sealed(dest: Path, sealed: Path) -> None:
    if dest.resolve() == sealed.resolve():
        raise SheafError(f"refusing to overwrite the sealed result file: {sealed}")


def run_t46_live(
    *,
    sealed_path: Path,
    live_out: Path | None = None,
    register_store: Path | None = None,
) -> dict[str, Any]:
    sealed_path = sealed_path.expanduser()
    live_out = (live_out or DEFAULT_LIVE_T46).expanduser()
    _refuse_sealed(live_out, SEALED_T46)
    _refuse_sealed(live_out, sealed_path)

    sealed = _load_json(sealed_path, "t46-tri-machine-hal")
    replay_ok, replay_criteria, replay_adv = verify_t46_tri_machine_hal(sealed)

    suite = importlib.import_module("hadukenn_core.tri_machine").run_tri_machine_verification_suite()
    if not isinstance(suite, dict) or suite.get("schema") != "sheaf-verify.tri-machine-hal.v1":
        raise SheafError("live T-46 suite did not return a sheaf-verify.tri-machine-hal.v1 document")

    live_doc = dict(suite)
    live_doc["measurement_mode"] = "live"
    live_doc["measured_at"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    live_doc["sealed_reference"] = {
        "path": str(sealed_path),
        "sha256": sha256_file(sealed_path),
    }
    live_doc["measurement_notes"] = [
        "Parity, C11 timings, sheaf KV-prefix, causal spans, slab overflow, and the watchdog state machine were measured in-process on this host.",
        "zenbook_npu_* fields are a host-simulated looped INT8 engine, not a live Snapdragon NPU run.",
        "WBeast and Zenbook hardware inventory in the sealed document was not re-measured and is reported only under replay.",
    ]
    # Do not persist the in-suite SCITT bundle (contains a built-in audit key).
    compliance = dict(live_doc.get("regulatory_compliance") or {})
    compliance.pop("bundle", None)
    live_doc["regulatory_compliance"] = compliance

    evidence = _collect_raw_evidence()
    live_doc["live_evidence"] = evidence["summaries"]
    live_criteria = evaluate_t46_live_criteria(live_doc, evidence)

    live_out.parent.mkdir(parents=True, exist_ok=True)
    live_out.write_text(json.dumps(live_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Also score the T46 document-shaped criteria against the live numbers (honest host sim included).
    doc_ok, doc_criteria, doc_adv = verify_t46_tri_machine_hal(live_doc)

    store_info = None
    if register_store is not None:
        store_info = _register_hal_store(register_store, evidence, live_doc)

    live_ok = all(item["passed"] for item in live_criteria)
    payload = {
        "evaluation": "live",
        "evaluation_label": LIVE_LABEL,
        "mode": "t46-tri-machine-hal",
        "ok": bool(live_ok),
        "live_out": str(live_out),
        "live_sha256": sha256_file(live_out),
        "criteria": live_criteria,
        "advisories": list(doc_adv) + [note for note in live_doc["measurement_notes"]],
        "live_document_criteria": _criteria_dicts(doc_criteria),
        "live_document_ok": bool(doc_ok),
        "replay": {
            "evaluation": "replay",
            "evaluation_label": REPLAY_LABEL,
            "path": str(sealed_path),
            "document_sha256": sha256_file(sealed_path),
            "ok": bool(replay_ok),
            "criteria": _criteria_dicts(replay_criteria),
            "advisories": list(replay_adv),
        },
    }
    if store_info is not None:
        payload["register_store"] = store_info
    return payload


def _collect_raw_evidence() -> dict[str, Any]:
    """Raw-repeat timings, float64 CCE parity, and thermal/power samples."""
    import importlib.util
    from tempfile import TemporaryDirectory

    from sheaf_verify.hal import cce_mup_loss_f64, evaluate_hal4, evaluate_hal5, evaluate_hal6
    from sheaf_verify.hashing import parse_json_bytes

    driver = ROOT / "scripts" / "live_gated_run.py"
    spec = importlib.util.spec_from_file_location("live_gated_run", driver)
    if spec is None or spec.loader is None:
        raise SheafError("scripts/live_gated_run.py is missing")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    summaries = {}
    criteria: dict[str, list] = {}
    with TemporaryDirectory(prefix="t46-live-evidence-") as tmp:
        work = Path(tmp)
        paths = mod.collect_hal_evidence(work)
        claim = {
            "format": "sheaf-verify.hal-claim.v1",
            "study_profile": "hal",
            "hypothesis_id": "campaign-live-t46",
            "description": "Live campaign C11 tiled CCE vs a serial Python loop on this host, both baselines reported.",
            "performance": {
                "metric": "speedup",
                "direction": "greater",
                "baseline": "serial_python_loop",
                "baselines_reported": ["serial_python_loop", "packed_numpy"],
                "min_speedup": 1.5,
                "subject": "c11_tiled_cce",
            },
            "parity": {"max_abs": 1e-4, "ops": ["cce_mup_fwd"]},
            "guards": {"thermal_stop_celsius": 85},
        }
        parity = json.loads(paths["parity"].read_text())
        timings = json.loads(paths["timings"].read_text())
        guards = json.loads(paths["guards"].read_text())

        # Recompute CCE vs stdlib float64 (does not trust kernel_outputs summary).
        class _Ctx:
            def __init__(self, **docs):
                self._raw = {k: json.dumps(v).encode("utf-8") for k, v in docs.items()}

            def has(self, role: str) -> bool:
                return role in self._raw

            def payload(self, role: str) -> bytes:
                return self._raw[role]

            def parse(self, role: str):
                return parse_json_bytes(self._raw[role], role)

        ctx = _Ctx(claim=claim, **{"log.parity": parity, "log.timings": timings, "log.guards": guards})
        # Also confirm the payload functions if needed — use HAL evaluators (source of truth).
        parity_c = evaluate_hal4(ctx)
        timing_c = evaluate_hal5(ctx)
        guard_c = evaluate_hal6(ctx)
        # Independent stdlib loss on the first case.
        case = parity["cases"][0]
        ref = cce_mup_loss_f64(
            case["inputs"]["H"],
            case["inputs"]["E"],
            case["inputs"]["y"],
            float(case["inputs"]["mup_base_d"]),
        )
        reported = [float(v) for v in case["kernel_outputs"]["loss"]]
        peak = max(abs(a - b) for a, b in zip(reported, ref, strict=True))
        summaries = {
            "parity_max_abs_vs_f64": peak,
            "timings_cases": [
                {
                    "name": c["name"],
                    "n_subject": len(c["subject_ms"]),
                    "n_serial": len(c["baselines_ms"]["serial_python_loop"]),
                    "n_numpy": len(c["baselines_ms"]["packed_numpy"]),
                }
                for c in timings["cases"]
            ],
            "guards": {
                "thermal_samples": len(guards.get("thermal", {}).get("samples") or []),
                "power": guards.get("power"),
            },
        }
        criteria = {
            "parity": [c.as_dict() for c in parity_c],
            "timings": [c.as_dict() for c in timing_c],
            "guards": [c.as_dict() for c in guard_c],
        }
        # Keep a copy of the sidecar docs in the live JSON, not the temp dir.
        sidecars = {"parity": parity, "timings": timings, "guards": guards, "claim": claim}
    return {"summaries": {**summaries, "hal_criteria": criteria, "sidecars": sidecars}}


def evaluate_t46_live_criteria(live_doc: dict, evidence: dict) -> list[dict]:
    criteria = []
    bm = live_doc.get("benchmarks_and_parity") or {}
    probe = live_doc.get("host_probe") or {}
    summaries = evidence.get("summaries") or {}

    # Document-level live numbers (re-measured this process).
    parity_ok = (
        float(bm.get("cce_loss_max_abs_err", 1.0)) < 1e-4
        and float(bm.get("cce_dh_max_abs_err", 1.0)) < 1e-4
        and float(bm.get("cce_de_max_abs_err", 1.0)) < 1e-4
    )
    criteria.append(
        (ok if parity_ok else bad)(
            "T46_LIVE_CCE_PARITY",
            f"In-process CCE loss/dH/dE max abs vs float64 reference: "
            f"{bm.get('cce_loss_max_abs_err')}, {bm.get('cce_dh_max_abs_err')}, {bm.get('cce_de_max_abs_err')}.",
        ).as_dict()
    )
    f64_peak = float(summaries.get("parity_max_abs_vs_f64", 1.0))
    criteria.append(
        (ok if f64_peak <= 1e-4 else bad)(
            "T46_LIVE_F64_REFERENCE",
            f"Registered small-case kernel loss max abs vs sheaf_verify.hal.cce_mup_loss_f64 is {f64_peak}.",
        ).as_dict()
    )

    speedup = float(bm.get("c11_speedup_over_py_loop", 0.0))
    criteria.append(
        (ok if speedup > 1.5 else bad)(
            "T46_LIVE_C11_SPEEDUP_VS_SERIAL",
            f"Suite-mean C11 speedup vs serial Python loop is {speedup:.4f} (threshold 1.5). "
            f"Vs packed NumPy: {bm.get('c11_speedup_over_numpy_gemm')}.",
        ).as_dict()
    )

    # Raw-repeat sidecars: require well-formed repeats and recomputation. The official
    # 1.5x speedup claim is judged on the T-46 suite shape (T46_LIVE_C11_SPEEDUP_VS_SERIAL),
    # not on the smaller HAL-study shapes, which can land under 1.5x depending on load.
    timing_items = ((summaries.get("hal_criteria") or {}).get("timings")) or []
    timing_structural = [item for item in timing_items if item.get("id") != "hal.perf.threshold"]
    timing_failed = [item for item in timing_structural if not item.get("passed")]
    threshold = next((item for item in timing_items if item.get("id") == "hal.perf.threshold"), None)
    timing_note = threshold["detail"] if threshold else "no sidecar threshold"
    criteria.append(
        (ok if timing_items and not timing_failed else bad)(
            "T46_LIVE_RAW_TIMINGS",
            ("raw repeats recomputed; " + timing_note + " (informational for small sidecar shapes)")
            if timing_items and not timing_failed
            else f"failed: {timing_failed}",
        ).as_dict()
    )
    for family, label in (("guards", "T46_LIVE_GUARDS"), ("parity", "T46_LIVE_SIDECAR_PARITY")):
        family_items = ((summaries.get("hal_criteria") or {}).get(family)) or []
        failed = [item for item in family_items if not item.get("passed")]
        criteria.append(
            (ok if family_items and not failed else bad)(
                label,
                "all HAL recomputations passed" if family_items and not failed else f"failed: {failed}",
            ).as_dict()
        )

    sheaf_ok = (
        float(bm.get("sheaf_kv_gate_max_abs_err", 1.0)) < 1e-4
        and float(bm.get("sheaf_coherent_neighbor_mass", 0.0)) > 0.99
        and bm.get("causal_span_prefix_invariance") is True
    )
    criteria.append(
        (ok if sheaf_ok else bad)(
            "T46_LIVE_SHEAF_AND_SPANS",
            "In-process sheaf KV-prefix gate parity and causal span prefix invariance.",
        ).as_dict()
    )

    arch = str(
        probe.get("cuda_arch_primary")
        or (live_doc.get("profiles") or {}).get("lapbeast", {}).get("cuda_arch_primary")
        or ""
    )
    # Honesty: we do not require sm_120. We require the probe to record whatever is actually present.
    criteria.append(
        (ok if arch else bad)(
            "T46_LIVE_HOST_CUDA_ARCH",
            f"Live host CUDA arch is {arch or 'unrecorded'} (not assumed to be sm_120).",
        ).as_dict()
    )

    # Host-simulated INT8 loop: report, but label it clearly. Still fail-closed on the cosine threshold.
    cosine = float(bm.get("zenbook_npu_int8_cosine_sim", 0.0))
    allocs = int(bm.get("zenbook_npu_loop_dynamic_allocs", 99))
    criteria.append(
        (ok if cosine >= 0.995 and allocs == 0 else bad)(
            "T46_LIVE_HOST_SIM_INT8_LOOP",
            f"Host-simulated looped INT8 engine cosine {cosine}, dynamic allocs {allocs}. "
            "Not a live Zenbook NPU measurement.",
        ).as_dict()
    )

    watchdog = bm.get("lapbeast_watchdog_rc143_verified") is True and bm.get("slab_overflow_refusal_verified") is True
    criteria.append(
        (ok if watchdog else bad)(
            "T46_LIVE_WATCHDOG_STATE_MACHINE",
            "LapBeast watchdog state machine and slab overflow refusal (in-process, not a field battery event).",
        ).as_dict()
    )
    return criteria


def run_t49_live(*, sealed_path: Path, live_out: Path | None = None) -> dict[str, Any]:
    sealed_path = sealed_path.expanduser()
    live_out = (live_out or DEFAULT_LIVE_T49).expanduser()
    _refuse_sealed(live_out, SEALED_T49)
    _refuse_sealed(live_out, sealed_path)

    sealed = _load_json(sealed_path, "t49-hst-foundation")
    replay_ok, replay_criteria, replay_adv = verify_t49_hst_foundation(sealed)

    live_doc = importlib.import_module("hadukenn_core.hst_foundation_lm").run_t49_hst_foundation_campaign()
    if not isinstance(live_doc, dict):
        raise SheafError("live T-49 campaign did not return a document")
    live_doc["measurement_mode"] = "live"
    live_doc["measured_at"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    live_doc["sealed_reference"] = {"path": str(sealed_path), "sha256": sha256_file(sealed_path)}
    live_doc["measurement_notes"] = [
        "Re-ran run_t49_hst_foundation_campaign() in-process (48,000 training-byte toy budget, not a multi-hour pretrain).",
    ]
    live_out.parent.mkdir(parents=True, exist_ok=True)
    live_out.write_text(json.dumps(live_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    live_ok, live_criteria, live_adv = verify_t49_hst_foundation(live_doc)
    return {
        "evaluation": "live",
        "evaluation_label": LIVE_LABEL,
        "mode": "t49-hst-foundation",
        "ok": bool(live_ok),
        "live_out": str(live_out),
        "live_sha256": sha256_file(live_out),
        "criteria": _criteria_dicts(live_criteria),
        "advisories": list(live_adv) + list(live_doc["measurement_notes"]),
        "replay": {
            "evaluation": "replay",
            "evaluation_label": REPLAY_LABEL,
            "path": str(sealed_path),
            "document_sha256": sha256_file(sealed_path),
            "ok": bool(replay_ok),
            "criteria": _criteria_dicts(replay_criteria),
            "advisories": list(replay_adv),
        },
    }


def _register_hal_store(store: Path, evidence: dict, live_doc: dict) -> dict[str, Any]:
    """Optional: register live evidence into a sheaf-verify store under --profile hal."""
    # Conservative pointer only: a full HAL ledger still comes from scripts/live_gated_run.sh.
    # Full register_all needs many host paths (seal key, kernels, etc.). We write
    # a pointer file rather than inventing a sealed training store.
    store = store.expanduser()
    if store.resolve() == SEALED_T46.resolve():
        raise SheafError("refusing to use the sealed T46 JSON as a sheaf store")
    store.mkdir(parents=True, exist_ok=True)
    pointer = store / "campaign-live-pointer.json"
    payload = {
        "format": "hadukenn.campaign-live-pointer.v1",
        "profile": "hal",
        "note": "Pointer to a live T-46 campaign result. Run scripts/live_gated_run.sh to build a full HAL ledger.",
        "live_document_sha256": sha256_json({k: v for k, v in live_doc.items() if k != "regulatory_compliance"}),
    }
    pointer.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"store": str(store), "pointer": str(pointer), "profile": "hal", "full_ledger": False}
