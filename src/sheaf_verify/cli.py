"""Command line for registration, gates, verification, and the review bundle."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from sheaf_verify.hashing import SheafError, canonical_bytes, is_sha256
from sheaf_verify.report import render
from sheaf_verify.runner import UtcClock, run
from sheaf_verify.seal import load_key
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
    records = run(_store(args), key=_key(args), clock=UtcClock())
    failed = [record["gate_id"] for record in records if record["status"] != "pass"]
    payload = {"failed": failed, "gates": [{"gate_id": record["gate_id"], "status": record["status"]} for record in records]}
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
    """Print the ledger head. This does not call any external service.

    Copy or print the line, or pass ``--out`` to append it to a file that
    sits outside the store. The seal key can rewrite the ledger; the copy
    is only an anchor if it is kept where that key cannot rewrite it.
    ``verify`` is what checks the chain. This command only reads ``head.json``.
    """
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
        root = store.root.resolve()
        target = args.out.resolve()
        if target == root or root in target.parents:
            raise SheafError("anchor file must sit outside the store")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("a", encoding="utf-8") as handle:
            handle.write(line)
    sys.stdout.write(line)
    return 0


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

