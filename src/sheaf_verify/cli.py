"""Command line for registration, gates, verification, provenance receipts, and campaign audits."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

from sheaf_verify.anticheat import audit_kernel_source
from sheaf_verify.attestation import (
    build_ledger_receipt,
    dsse_sign,
    dsse_verify,
    intoto_statement_v1,
    merkle_tree_head,
    validate_strict_json_bytes,
    verify_ledger_receipt,
)
from sheaf_verify.campaign import (
    audit_cluster_geometry,
    head_flop_scaling_table,
    verify_t46_snapdragon_local,
)
from sheaf_verify.hashing import SheafError, canonical_bytes, is_sha256, reject_symlink
from sheaf_verify.ledger import Ledger
from sheaf_verify.mutation import run_mutation_suite
from sheaf_verify.report import render
from sheaf_verify.runner import UtcClock, run
from sheaf_verify.seal import load_key
from sheaf_verify.sheaf_cohomology import CellularSheaf, diagnose_contradictions, grounded_mapping_cone_energy
from sheaf_verify.store import Store, dump_pretty
from sheaf_verify.verify import verify_store


def main(argv: list[str] | None = None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--store", default=".sheaf", help="artifact store directory")
    common.add_argument("--json", action="store_true", help="print machine-readable JSON")
    parser = argparse.ArgumentParser(prog="sheaf-verify")
    sub = parser.add_subparsers(dest="command", required=True)

    register = sub.add_parser("register", parents=[common], help="register an inline file")
    register.add_argument("--kind", required=True)
    register.add_argument("--role", required=True)
    register.add_argument("--path", required=True, type=Path)
    register.add_argument("--registered-at")
    register.add_argument("--input", action="append", default=[])
    register.add_argument("--metadata", default="{}")
    register.set_defaults(func=_register)

    external = sub.add_parser("register-external", parents=[common], help="register a checkpoint by hash")
    external.add_argument("--role", required=True)
    external.add_argument("--payload-sha256", required=True)
    external.add_argument("--size-bytes", required=True, type=int)
    external.add_argument("--locator", required=True)
    external.add_argument("--registered-at")
    external.add_argument("--input", action="append", default=[])
    external.set_defaults(func=_register_external)

    tree = sub.add_parser("register-tree", parents=[common], help="register a code tree")
    tree.add_argument("--role", default="code")
    tree.add_argument("--path", required=True, type=Path)
    tree.add_argument("--registered-at")
    tree.add_argument("--input", action="append", default=[])
    tree.set_defaults(func=_register_tree)

    env = sub.add_parser("register-env", parents=[common], help="register the environment lock")
    env.add_argument("--role", default="environment")
    env.add_argument("--python", required=True)
    env.add_argument("--cuda", required=True)
    env.add_argument("--driver", required=True)
    env.add_argument("--gpu", required=True)
    env.add_argument("--platform", required=True)
    env.add_argument("--registered-at")
    env.set_defaults(func=_register_env)

    run_cmd = sub.add_parser("run", parents=[common], help="evaluate every gate and write a result")
    run_cmd.add_argument(
        "--profile",
        choices=("training", "hal"),
        default="training",
        help="Study kind. training (default) evaluates G/H/P and cannot skip those gates. hal evaluates only HAL0-HAL6.",
    )
    _add_key(run_cmd)
    run_cmd.set_defaults(func=_run)

    verify_cmd = sub.add_parser("verify", parents=[common], help="recompute hashes, the ledger, and gate replay")
    _add_key(verify_cmd)
    verify_cmd.add_argument("--allow-unsigned", action="store_true")
    verify_cmd.add_argument("--strict", action="store_true")
    verify_cmd.set_defaults(func=_verify)

    bundle = sub.add_parser("bundle", parents=[common], help="export a reviewer bundle")
    _add_key(bundle)
    bundle.add_argument("--out", required=True, type=Path)
    bundle.add_argument("--allow-unsigned", action="store_true")
    bundle.set_defaults(func=_bundle)

    checklist = sub.add_parser("checklist", parents=[common], help="print the adversarial checklist")
    _add_key(checklist)
    checklist.add_argument("--allow-unsigned", action="store_true")
    checklist.set_defaults(func=_checklist)

    anchor = sub.add_parser(
        "anchor",
        parents=[common],
        help="print the ledger head hash so it can be recorded outside the store",
    )
    anchor.add_argument("--out", type=Path, help="append the same line to a file outside the store")
    anchor.set_defaults(func=_anchor)

    attest = sub.add_parser("attest", parents=[common], help="emit an in-toto v1 + DSSE v1.0 store attestation")
    _add_key(attest)
    attest.add_argument("--out", type=Path, help="write the DSSE envelope outside the store")
    attest.set_defaults(func=_attest)

    verify_att = sub.add_parser(
        "verify-attestation", parents=[common], help="verify an in-toto v1 + DSSE v1.0 envelope"
    )
    _add_key(verify_att)
    verify_att.add_argument("--envelope", required=True, type=Path)
    verify_att.set_defaults(func=_verify_attestation)

    receipt = sub.add_parser(
        "receipt", parents=[common], help="emit an RFC 9162 Merkle inclusion receipt for a ledger line"
    )
    _add_key(receipt)
    receipt.add_argument("--allow-unsigned", action="store_true")
    receipt.add_argument("--seq", required=True, type=int, help="0-based ledger record sequence")
    receipt.add_argument("--out", type=Path, help="optional receipt output path outside the store")
    receipt.set_defaults(func=_receipt)

    verify_rec = sub.add_parser("verify-receipt", parents=[common], help="verify a standalone Merkle inclusion receipt")
    _add_key(verify_rec)
    verify_rec.add_argument("--allow-unsigned", action="store_true")
    verify_rec.add_argument("--receipt", required=True, type=Path)
    verify_rec.set_defaults(func=_verify_receipt)

    audit_kern = sub.add_parser("audit-kernel", parents=[common], help="run AST anti-cheat audit on a kernel file")
    audit_kern.add_argument("--path", required=True, type=Path)
    audit_kern.add_argument("--require-triton", action="store_true")
    audit_kern.set_defaults(func=_audit_kernel)

    sheaf_cmd = sub.add_parser(
        "sheaf-check", parents=[common], help="compute cellular sheaf cohomology and cone diagnostics"
    )
    sheaf_cmd.add_argument("--path", required=True, type=Path)
    sheaf_cmd.set_defaults(func=_sheaf_check)

    campaign_cmd = sub.add_parser(
        "campaign", parents=[common], help="verify T-37..T-46, cluster-geometry, or hg-v3 records"
    )
    campaign_cmd.add_argument(
        "--mode",
        required=True,
        choices=(
            "t37-grid",
            "t38-prereg",
            "cluster-geometry",
            "hg-v3",
            "flop-scaling",
            "t40-stack",
            "t41-frontier5070",
            "t42-exec-bridge",
            "t43-all-realms",
            "t44-glassbox",
            "t45-sheaf-bridge",
            "t46-snapdragon-local",
            "t46-tri-machine-hal",
            "t47-cosimilar-trace",
            "t48-roadmap-complete",
            "t49-hst-foundation",
            "t49-hst-foundation-stage",
        ),
    )
    campaign_cmd.add_argument(
        "--path",
        "--in",
        dest="path",
        type=Path,
        help="input JSON file for the selected campaign mode",
    )
    campaign_cmd.add_argument("--addendum", type=Path, help="optional paired-seed addendum JSON for t38-prereg")
    campaign_cmd.add_argument("--vocab", type=int, default=32768)
    campaign_cmd.add_argument("--nc", type=int, default=256)
    campaign_cmd.add_argument("--cluster-size", type=int, default=128)
    campaign_cmd.add_argument("--d-model", type=int, default=576)
    campaign_cmd.add_argument("--k-neg", type=int, default=3)
    campaign_cmd.add_argument("--struct-rank", type=int, default=0)
    campaign_cmd.add_argument(
        "--live",
        action="store_true",
        help="Re-run the measurement suite in-process (t46 HAL, t49 HST toy campaign). Never overwrites the sealed result JSON.",
    )
    campaign_cmd.add_argument(
        "--replay",
        action="store_true",
        help="Force JSON-only replay even if hardware is present.",
    )
    campaign_cmd.add_argument(
        "--live-out",
        type=Path,
        help="Where to write a live result JSON (must not be the sealed path).",
    )
    campaign_cmd.add_argument(
        "--register-store",
        type=Path,
        help="Optional sheaf-verify store to register a HAL-profile live t46 run.",
    )
    campaign_cmd.set_defaults(func=_campaign)

    mut_cmd = sub.add_parser(
        "mutation-score", parents=[common], help="run scientific and cryptographic mutation adequacy suite"
    )
    mut_cmd.set_defaults(func=_mutation_score)

    self_cmd = sub.add_parser("selftest", parents=[common], help="run full 8-gate Antigravity repository verification")
    self_cmd.add_argument("--report-out", type=Path, default=None)
    self_cmd.set_defaults(func=_selftest)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except SheafError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


def _add_key(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seal-key-file", type=Path)


def _store(args) -> Store:
    return Store(Path(args.store))


def _when(args) -> str:
    if args.registered_at:
        return args.registered_at
    return UtcClock().now()


def _key(args) -> bytes | None:
    if args.seal_key_file is None:
        if getattr(args, "allow_unsigned", False):
            return None
        raise SheafError("a seal key file is required")
    return load_key(args.seal_key_file)


def _check_outside_store(store: Store, out_path: Path, label: str) -> None:
    root = store.root.resolve()
    target = out_path.resolve()
    if target == root or root in target.parents:
        raise SheafError(f"{label} must sit outside the store")


def _register(args) -> int:
    metadata = json.loads(args.metadata)
    if not isinstance(metadata, dict):
        raise SheafError("metadata must be a JSON object")
    digest, manifest = _store(args).register_file(
        kind=args.kind,
        role=args.role,
        path=args.path,
        registered_at=_when(args),
        inputs=args.input,
        metadata=metadata,
    )
    return _emit(args, {"manifest_sha256": digest, "manifest": manifest})


def _register_external(args) -> int:
    digest, manifest = _store(args).register(
        kind="checkpoint",
        role=args.role,
        registered_at=_when(args),
        storage="external",
        payload_sha256=args.payload_sha256,
        size_bytes=args.size_bytes,
        locator=args.locator,
        inputs=args.input,
    )
    return _emit(args, {"manifest_sha256": digest, "manifest": manifest})


def _register_tree(args) -> int:
    digest, manifest = _store(args).register_tree(
        role=args.role,
        path=args.path,
        registered_at=_when(args),
        inputs=args.input,
    )
    return _emit(args, {"manifest_sha256": digest, "manifest": manifest})


def _register_env(args) -> int:
    digest, manifest = _store(args).register_environment(
        role=args.role,
        registered_at=_when(args),
        python=args.python,
        cuda=args.cuda,
        driver=args.driver,
        gpu=args.gpu,
        platform=args.platform,
    )
    return _emit(args, {"manifest_sha256": digest, "manifest": manifest})


def _run(args) -> int:
    records = run(_store(args), key=_key(args), clock=UtcClock(), profile=args.profile)
    failed = [record["gate_id"] for record in records if record["status"] != "pass"]
    payload = {
        "failed": failed,
        "gates": [{"gate_id": record["gate_id"], "status": record["status"]} for record in records],
    }
    _emit(args, payload)
    return 1 if failed else 0


def _verify(args) -> int:
    report = verify_store(_store(args), key=_key(args), allow_unsigned=args.allow_unsigned, strict=args.strict)
    _emit_report(args, report)
    return 0 if report.ok else 1


def _checklist(args) -> int:
    report = verify_store(_store(args), key=_key(args), allow_unsigned=args.allow_unsigned)
    _emit(args, {"checklist": report.checklist, "problems": report.problems})
    return 0 if report.checklist and all(item["answered"] for item in report.checklist) else 1


def _bundle(args) -> int:
    store = _store(args)
    report = verify_store(store, key=_key(args), allow_unsigned=args.allow_unsigned)
    out = args.out
    if out.exists():
        raise SheafError("bundle output already exists")
    out.mkdir(parents=True)
    for name in ("manifests", "payloads", "snapshots", "ledger"):
        source = store.root / name
        if source.is_dir():
            shutil.copytree(source, out / name, ignore=shutil.ignore_patterns(".lock"))
    (out / "REPORT.md").write_text(render(report), encoding="utf-8")
    science = report.ok and all(row["status"] == "pass" for row in report.gate_rows)
    clear = science and all(item["answered"] for item in report.checklist)
    _emit_report(args, report)
    return 0 if clear else 1


def _anchor(args) -> int:
    """Print the ledger head. This does not call any external service."""
    store = _store(args)
    head_path = store.root / "ledger" / "head.json"
    if not head_path.is_file():
        raise SheafError("ledger head is missing")
    head = json.loads(head_path.read_text(encoding="utf-8"))
    digest = head.get("head_sha256") if isinstance(head, dict) else None
    if not is_sha256(digest):
        raise SheafError("ledger head hash is missing")
    line = f"sheaf-verify.anchor.v1 {digest}\n"
    if args.out is not None:
        _check_outside_store(store, args.out, "anchor file")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("a", encoding="utf-8") as handle:
            handle.write(line)
    sys.stdout.write(line)
    return 0


def _attest(args) -> int:
    store = _store(args)
    key = _key(args)
    assert key is not None
    report = verify_store(store, key=key)
    if not report.ok or report.head is None:
        raise SheafError("cannot attest an unverified or empty store")
    records = Ledger(store.root).records()
    merkle_root = merkle_tree_head([canonical_bytes(r) for r in records])
    binding = store.binding()
    subjects = [{"name": "ledger/head", "digest": {"sha256": report.head}}]
    if "result" in binding:
        subjects.append({"name": "role/result", "digest": {"sha256": binding["result"][0]}})
    statement = intoto_statement_v1(
        subjects=subjects,
        predicate={
            "gates": report.gate_rows,
            "ledger_head_sha256": report.head,
            "merkle_root_sha256": merkle_root,
            "open_objections": [item["id"] for item in report.checklist if not item["answered"]],
            "tree_size": len(records),
        },
    )
    envelope = dsse_sign(statement, key=key)
    if args.out is not None:
        _check_outside_store(store, args.out, "attestation file")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(canonical_bytes(envelope) + b"\n")
    return _emit(args, envelope)


def _verify_attestation(args) -> int:
    key = _key(args)
    assert key is not None
    path = reject_symlink(args.envelope, "envelope file")
    if not path.is_file():
        raise SheafError("envelope file is missing")
    envelope = validate_strict_json_bytes(path.read_bytes())
    statement = dsse_verify(envelope, key=key)
    return _emit(args, {"ok": True, "statement": statement})


def _receipt(args) -> int:
    store = _store(args)
    key = _key(args)
    records = Ledger(store.root).records()
    receipt = build_ledger_receipt(records, args.seq, key=key)
    if args.out is not None:
        _check_outside_store(store, args.out, "receipt file")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(canonical_bytes(receipt) + b"\n")
    return _emit(args, receipt)


def _verify_receipt(args) -> int:
    key = _key(args)
    path = reject_symlink(args.receipt, "receipt file")
    if not path.is_file():
        raise SheafError("receipt file is missing")
    doc = validate_strict_json_bytes(path.read_bytes())
    ok_flag, problems = verify_ledger_receipt(doc, key=key)
    _emit(args, {"ok": ok_flag, "problems": problems})
    return 0 if ok_flag else 1


def _audit_kernel(args) -> int:
    path = reject_symlink(args.path, "kernel file")
    if not path.is_file():
        raise SheafError("kernel file is missing")
    rep = audit_kernel_source(path.read_text(encoding="utf-8"), require_triton_or_custom=args.require_triton)
    payload = {
        "criteria": [c.as_dict() for c in rep.criteria],
        "has_triton_jit": rep.has_triton_jit,
        "ok": rep.ok,
        "violations": list(rep.violations),
    }
    _emit(args, payload)
    return 0 if rep.ok else 1


def _sheaf_check(args) -> int:
    path = reject_symlink(args.path, "sheaf file")
    if not path.is_file():
        raise SheafError("sheaf file is missing")
    doc = validate_strict_json_bytes(path.read_bytes())
    sheaf = CellularSheaf.from_dict(doc)
    summary = sheaf.cohomology()
    out: dict = {
        "boundary_defect_frobenius": summary.boundary_defect_frobenius,
        "dim_c0": summary.dim_c0,
        "dim_c1": summary.dim_c1,
        "dim_h0": summary.dim_h0,
        "dim_h1": summary.dim_h1,
        "rank_d0": summary.rank_d0,
        "spectral_gap": summary.spectral_gap,
    }
    if isinstance(doc.get("assignment"), dict):
        verdict = diagnose_contradictions(sheaf, doc["assignment"])
        out["contradiction_regime"] = verdict.regime
        out["faulty_vertices"] = list(verdict.faulty_vertices)
        out["section_energy"] = verdict.total_energy
        if isinstance(doc.get("anchors"), dict):
            internal, cone_e, obs = grounded_mapping_cone_energy(sheaf, doc["assignment"], doc["anchors"])
            out["grounded_cone"] = {"cone_energy": cone_e, "internal_energy": internal, "obstructions": obs}
    return _emit(args, out)


def _campaign(args) -> int:
    from sheaf_verify.campaign_live import evaluate_campaign

    mode = args.mode
    if mode == "flop-scaling":
        return _emit(
            args, {"evaluation": "computed", "evaluation_label": "computed table", "rows": head_flop_scaling_table()}
        )
    if mode == "cluster-geometry":
        res = audit_cluster_geometry(
            vocab_size=args.vocab,
            nc=args.nc,
            cluster_size=args.cluster_size,
            d_model=args.d_model,
            k_neg=args.k_neg,
            struct_rank=args.struct_rank,
        )
        payload = {
            "evaluation": "computed",
            "evaluation_label": "computed geometry audit",
            "cluster_size": res.cluster_size,
            "criteria": [c.as_dict() for c in res.criteria],
            "dead_pad_slots": res.dead_pad_slots,
            "gather_mib_bf16": res.gather_mib_bf16,
            "head_mflops_per_tok_kneg": res.head_mflops_per_tok_kneg,
            "head_mflops_per_tok_kneg0": res.head_mflops_per_tok_kneg0,
            "nc": res.nc,
            "passes_p1_size_gate": res.passes_p1_size_gate,
            "recommended": res.recommended,
            "redundant_unit_table_params": res.redundant_unit_table_params,
            "tensor_core_128_aligned": res.tensor_core_128_aligned,
            "vocab_size": res.vocab_size,
        }
        _emit(args, payload)
        return 0 if res.recommended else 1

    if mode == "t46-snapdragon-local":
        if args.path is None or not args.path.is_file():
            raise SheafError(f"--path is required for campaign mode {mode!r}")
        doc = validate_strict_json_bytes(reject_symlink(args.path, "campaign JSON").read_bytes())
        ok_flag, criteria, advisories = verify_t46_snapdragon_local(doc)
        _emit(args, {"advisories": advisories, "criteria": [c.as_dict() for c in criteria], "ok": ok_flag})
        return 0 if ok_flag else 1

    live = bool(getattr(args, "live", False))
    replay = bool(getattr(args, "replay", False))
    if live and replay:
        raise SheafError("--live and --replay cannot be combined")
    payload = evaluate_campaign(
        mode=mode,
        path=getattr(args, "path", None),
        addendum=getattr(args, "addendum", None),
        live=live,
        replay=replay,
        live_out=getattr(args, "live_out", None),
        register_store=getattr(args, "register_store", None),
    )
    _emit(args, payload)
    return 0 if payload.get("ok") else 1


def _selftest(args) -> int:
    from sheaf_verify.selftest import run_full_verification

    ok_flag, report = run_full_verification()
    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _emit(args, report)
    return 0 if ok_flag else 1


def _mutation_score(args) -> int:
    with tempfile.TemporaryDirectory(prefix="sheaf_mut_") as tmp:
        rep = run_mutation_suite(Path(tmp))
    payload = {
        "equivalent": rep.equivalent,
        "generated": rep.generated,
        "killed": rep.killed,
        "mutation_score": rep.mutation_score,
        "outcomes": [
            {"category": m.category, "description": m.description, "detail": m.detail, "id": m.id, "killed": m.killed}
            for m in rep.outcomes
        ],
        "survived": rep.survived,
    }
    _emit(args, payload)
    return 0 if rep.survived == 0 else 1


def _emit(args, payload: dict) -> int:
    if args.json:
        sys.stdout.write(canonical_bytes(payload).decode("utf-8") + "\n")
    else:
        sys.stdout.write(dump_pretty(payload))
    return 0


def _emit_report(args, report) -> None:
    payload = {
        "checklist_open": [item["id"] for item in report.checklist if not item["answered"]],
        "gates": report.gate_rows,
        "head": report.head,
        "notes": report.notes,
        "ok": report.ok,
        "problems": report.problems,
    }
    if args.json:
        sys.stdout.write(canonical_bytes(payload).decode("utf-8") + "\n")
    else:
        sys.stdout.write(render(report))
