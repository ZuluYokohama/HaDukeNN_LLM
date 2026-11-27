"""Unit tests for T-46 Snapdragon X Plus campaign verifier, CLI, planner, and runner."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from hadukenn_core import local_e2e_runner, snapdragon_local
from sheaf_verify.campaign import verify_t46_snapdragon_local
from sheaf_verify.cli import main as cli_main
from sheaf_verify.hashing import canonical_bytes, sha256_hex


def _sign_t46_doc(doc: dict[str, Any]) -> dict[str, Any]:
    body = {k: v for k, v in doc.items() if k != "manifest_sha256"}
    signed = dict(body)
    signed["manifest_sha256"] = sha256_hex(canonical_bytes(body))
    return signed


def _valid_t46_doc(*, avail_gib: float = 2.0) -> dict[str, Any]:
    tiers_raw = [
        ("51M", 50.72, 0.18, 1200.0, 8.12, True),
        ("125M", 124.26, 0.35, 650.0, 9.86, True),
        ("1.5B", 1479.02, 1.28, 78.0, 12.18, False),
        ("3.2B", 3242.72, 3.74, 24.0, 13.92, False),
        ("8B", 8053.06, 5.48, 16.0, 14.5, False),
    ]
    tiers = []
    for name, params_m, ram_gib, tok_s, power_w, want_live in tiers_raw:
        fits_wsl2 = ram_gib <= avail_gib
        tiers.append(
            {
                "name": name,
                "tier": name,
                "vocab_size": 32768,
                "head_kind": "flat_dense_struct32k",
                "total_params_m": params_m,
                "required_ram_gib": ram_gib,
                "fits_physical_ram": ram_gib <= 15.61,
                "fits_wsl2_mem_available": fits_wsl2,
                "bandwidth_tok_s": tok_s,
                "power_w": power_w,
                "live_executed": bool(want_live and fits_wsl2),
            }
        )
    doc: dict[str, Any] = {
        "schema": "sheaf-verify.t46-snapdragon-local.v1",
        "version": "0.9.0",
        "hardware": {
            "soc": "X1P-42-100",
            "cores": 8,
            "bandwidth_gb_s": 135.0,
            "tdp_watts": 14.5,
            "physical_ram_gib": 15.61,
            "wsl2_mem_available_gib": avail_gib,
        },
        "tiers": tiers,
        "sheaf_bridge_csr": {
            "vocab_size": 32768,
            "csr_bytes": 2424828,
            "peak_ram_bytes": 27663136,
            "within_budget": True,
            "dense_flat_head_preserved": True,
        },
        "torch_bridge_audit_ok": True,
        "torch_bridge_violations": [],
        "int8_parity_ok": True,
        "int8_max_abs_err": 0.0,
        "oryon_simd_speedup_vs_numpy": 18.4,
        "op_counter": {"mac_ops": 16777216, "calls": 4},
        "execution_tiers": {
            "oryon_i8mm": {"parity_ok": True, "speedup_vs_numpy": 18.4},
            "numpy_ref": {"parity_ok": True},
        },
        "npu_tier_used": "htp_qnn",
        "npu_canary_ok": True,
        "no_wslconfig_modified": True,
        "no_network_downloads": True,
    }
    return _sign_t46_doc(doc)


def test_verify_t46_snapdragon_local_pass_and_advisories() -> None:
    doc = _valid_t46_doc()
    ok_flag, criteria, advisories = verify_t46_snapdragon_local(doc)
    assert ok_flag is True
    assert len(criteria) == 6
    assert all(c.passed for c in criteria)
    assert len(advisories) == 1 and "SNAPDRAGON_X_PLUS_LOCAL_READY" in advisories[0]


def test_verify_t46_rejects_non_dict_and_bad_schema() -> None:
    ok_none, crit_none, adv_none = verify_t46_snapdragon_local(None)  # type: ignore[arg-type]
    assert ok_none is False and crit_none[0].id == "T46_SCHEMA_VERSION" and adv_none == []

    ok_empty, crit_empty, _ = verify_t46_snapdragon_local({})
    assert ok_empty is False and any(not c.passed for c in crit_empty)

    bad_ver = _valid_t46_doc()
    bad_ver["version"] = "0.8.0"
    ok_ver, crit_ver, _ = verify_t46_snapdragon_local(_sign_t46_doc(bad_ver))
    assert ok_ver is False and not crit_ver[0].passed


def test_verify_t46_rejects_bool_nan_inf_overflow_in_hardware() -> None:
    for field, bad_val in (
        ("cores", True),
        ("bandwidth_gb_s", float("nan")),
        ("tdp_watts", float("inf")),
        ("physical_ram_gib", 10**400),
        ("wsl2_mem_available_gib", -0.1),
        ("soc", "RTX-5070"),
    ):
        doc = _valid_t46_doc()
        doc["hardware"][field] = bad_val
        ok_flag, crit, _ = verify_t46_snapdragon_local(doc)
        assert ok_flag is False
        assert any(c.id == "T46_HARDWARE_AND_MEMINFO" and not c.passed for c in crit)

    doc_bad_hw_type = _valid_t46_doc()
    doc_bad_hw_type["hardware"] = "invalid"
    ok_hw, crit_hw, _ = verify_t46_snapdragon_local(doc_bad_hw_type)
    assert ok_hw is False
    assert any(c.id == "T46_HARDWARE_AND_MEMINFO" and not c.passed for c in crit_hw)


def test_verify_t46_rejects_tier_mutations_and_gating_violations() -> None:
    # Wrong length / non-list
    d_list = _valid_t46_doc()
    d_list["tiers"] = d_list["tiers"][:4]
    assert verify_t46_snapdragon_local(_sign_t46_doc(d_list))[0] is False

    # Wrong tier name / mismatched tier alias
    d_name = _valid_t46_doc()
    d_name["tiers"][0]["name"] = "42M"
    assert verify_t46_snapdragon_local(_sign_t46_doc(d_name))[0] is False

    d_alias = _valid_t46_doc()
    d_alias["tiers"][0]["tier"] = "wrong"
    assert verify_t46_snapdragon_local(_sign_t46_doc(d_alias))[0] is False

    # 182x181 / factored head regression
    d_head = _valid_t46_doc()
    d_head["tiers"][0]["head_kind"] = "factored_182x181"
    assert verify_t46_snapdragon_local(_sign_t46_doc(d_head))[0] is False

    # Non-finite / non-positive / power > 14.5
    d_nan = _valid_t46_doc()
    d_nan["tiers"][0]["bandwidth_tok_s"] = "fast"
    assert verify_t46_snapdragon_local(d_nan)[0] is False

    d_pow = _valid_t46_doc()
    d_pow["tiers"][4]["power_w"] = 18.0
    assert verify_t46_snapdragon_local(_sign_t46_doc(d_pow))[0] is False

    # Non-bool flags
    d_bool = _valid_t46_doc()
    d_bool["tiers"][0]["live_executed"] = 1
    assert verify_t46_snapdragon_local(_sign_t46_doc(d_bool))[0] is False

    # Live execution on non-fitting tier (8B live_executed=True)
    d_oom = _valid_t46_doc()
    d_oom["tiers"][4]["live_executed"] = True
    assert verify_t46_snapdragon_local(_sign_t46_doc(d_oom))[0] is False

    # Inconsistent fits_physical_ram and fits_wsl2_mem_available
    d_phys = _valid_t46_doc()
    d_phys["tiers"][4]["fits_physical_ram"] = False
    assert verify_t46_snapdragon_local(_sign_t46_doc(d_phys))[0] is False

    d_wsl = _valid_t46_doc()
    d_wsl["tiers"][4]["fits_wsl2_mem_available"] = True
    assert verify_t46_snapdragon_local(_sign_t46_doc(d_wsl))[0] is False

    # Non-monotonic tok/s
    d_mono = _valid_t46_doc()
    d_mono["tiers"][1]["bandwidth_tok_s"] = 1500.0
    assert verify_t46_snapdragon_local(_sign_t46_doc(d_mono))[0] is False

    # No live execution despite >= 0.25 GiB available
    d_nolive = _valid_t46_doc()
    d_nolive["tiers"][0]["live_executed"] = False
    d_nolive["tiers"][1]["live_executed"] = False
    assert verify_t46_snapdragon_local(_sign_t46_doc(d_nolive))[0] is False


def test_verify_t46_rejects_csr_torch_parity_and_manifest_tamper() -> None:
    d_csr = _valid_t46_doc()
    d_csr["sheaf_bridge_csr"]["csr_bytes"] = 9 * 1024 * 1024
    assert verify_t46_snapdragon_local(_sign_t46_doc(d_csr))[0] is False

    d_top_ram = _valid_t46_doc()
    d_top_ram["peak_ram_bytes"] = 65 * 1024 * 1024
    assert verify_t46_snapdragon_local(_sign_t46_doc(d_top_ram))[0] is False

    d_sb_type = _valid_t46_doc()
    d_sb_type["sheaf_bridge_csr"] = None
    assert verify_t46_snapdragon_local(_sign_t46_doc(d_sb_type))[0] is False

    d_torch = _valid_t46_doc()
    d_torch["torch_bridge_audit_ok"] = False
    assert verify_t46_snapdragon_local(_sign_t46_doc(d_torch))[0] is False

    d_speed = _valid_t46_doc()
    d_speed["oryon_simd_speedup_vs_numpy"] = 0.9
    assert verify_t46_snapdragon_local(_sign_t46_doc(d_speed))[0] is False

    d_canary = _valid_t46_doc()
    d_canary["npu_canary_ok"] = False
    assert verify_t46_snapdragon_local(_sign_t46_doc(d_canary))[0] is False

    # Manifest mismatch (tamper without re-signing)
    d_tamper = copy.deepcopy(_valid_t46_doc())
    d_tamper["int8_parity_ok"] = False
    assert verify_t46_snapdragon_local(d_tamper)[0] is False

    # Drive-letter backslash leak caught even if re-signed
    d_drive = _valid_t46_doc()
    d_drive["extra_note"] = ("C:" + "\\") + "Users"
    assert verify_t46_snapdragon_local(_sign_t46_doc(d_drive))[0] is False

    # Non-canonical JSON value (NaN inside extra field) caught in manifest step
    d_noncanon = _valid_t46_doc()
    d_noncanon["extra_nan"] = float("nan")
    assert verify_t46_snapdragon_local(d_noncanon)[0] is False


def test_cli_campaign_t46_path_in_and_errors(tmp_path: Path, capsys: Any) -> None:
    good_path = tmp_path / "good_t46.json"
    good_path.write_text(json.dumps(_valid_t46_doc()), encoding="utf-8")

    bad_doc = _valid_t46_doc()
    bad_doc["torch_bridge_audit_ok"] = False
    bad_path = tmp_path / "bad_t46.json"
    bad_path.write_text(json.dumps(_sign_t46_doc(bad_doc)), encoding="utf-8")

    capsys.readouterr()
    assert cli_main(["campaign", "--mode", "t46-snapdragon-local", "--path", str(good_path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True

    assert cli_main(["campaign", "--mode", "t46-snapdragon-local", "--in", str(good_path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True

    assert cli_main(["campaign", "--mode", "t46-snapdragon-local", "--in", str(bad_path), "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False

    assert cli_main(["campaign", "--mode", "t46-snapdragon-local"]) == 2
    assert cli_main(["campaign", "--mode", "t46-snapdragon-local", "--in", str(tmp_path / "nope.json")]) == 2


def test_snapdragon_local_planner_validation_and_meminfo(monkeypatch: Any) -> None:
    orig_read_text = Path.read_text

    def _fake_meminfo(self: Path, *args: Any, **kwargs: Any) -> str:
        if str(self) == "/proc/meminfo":
            return "MemTotal: 8388608 kB\nMemAvailable: 2097152 kB\n"
        return orig_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", _fake_meminfo)
    spec = snapdragon_local.SnapdragonXPlusSpec()
    assert spec.wsl2_mem_available_gib == pytest.approx(2.0)

    tiers = snapdragon_local.plan_snapdragon_local_tiers(spec)
    assert [t["name"] for t in tiers] == ["51M", "125M", "1.5B", "3.2B", "8B"]
    assert tiers[0]["fits_wsl2_mem_available"] is True
    assert tiers[3]["fits_wsl2_mem_available"] is False
    assert tiers[4]["fits_wsl2_mem_available"] is False

    with pytest.raises(ValueError):
        snapdragon_local.plan_snapdragon_local_tiers(vocab_size=True)
    with pytest.raises(ValueError):
        snapdragon_local.plan_snapdragon_local_tiers(snapdragon_local.SnapdragonXPlusSpec(bandwidth_gb_s=-1.0))
    with pytest.raises(ValueError):
        snapdragon_local.plan_snapdragon_local_tiers(snapdragon_local.SnapdragonXPlusSpec(cores=0))
    with pytest.raises(ValueError):
        snapdragon_local.plan_snapdragon_local_tiers(snapdragon_local.SnapdragonXPlusSpec(tdp_watts=0.0))
    with pytest.raises(ValueError):
        snapdragon_local.plan_snapdragon_local_tiers(snapdragon_local.SnapdragonXPlusSpec(wsl2_mem_available_gib=-0.5))


def test_local_e2e_runner_cli_and_early_path_validation(tmp_path: Path, monkeypatch: Any, capsys: Any) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("file", encoding="utf-8")
    assert local_e2e_runner.main(["--json-out", str(blocker / "out.json")]) == 1
    assert local_e2e_runner.main(["--json-out", str(tmp_path)]) == 1

    fake_rep = _valid_t46_doc()
    monkeypatch.setattr(
        local_e2e_runner,
        "run_t46_snapdragon_local_suite",
        lambda **_kw: copy.deepcopy(fake_rep),
    )
    out_file = tmp_path / "e2e_out.json"
    assert local_e2e_runner.main(["--json-out", str(out_file)]) == 0
    assert out_file.is_file()
    loaded = json.loads(out_file.read_text(encoding="utf-8"))
    assert verify_t46_snapdragon_local(loaded)[0] is True

    capsys.readouterr()
    assert local_e2e_runner.main([]) == 0
    assert json.loads(capsys.readouterr().out)["schema"] == "sheaf-verify.t46-snapdragon-local.v1"


def test_snapdragon_local_low_meminfo_slow_simd_unclipped_rejected(monkeypatch: Any) -> None:
    if not snapdragon_local.detect_oryon_capabilities().get("has_i8mm"):
        pytest.skip("requires ARM64 Oryon i8mm CPU")
    orig_read_text = Path.read_text
    orig_bench = snapdragon_local.benchmark_kernel_parity

    def _low_meminfo(self: Path, *args: Any, **kwargs: Any) -> str:
        if str(self) == "/proc/meminfo":
            return "MemTotal: 8388608 kB\nMemAvailable: 100000 kB\n"
        return orig_read_text(self, *args, **kwargs)

    def _slow_gemm_bench(name: str, fn: Any, ref_fn: Any, **kwargs: Any) -> dict[str, Any]:
        res = orig_bench(name, fn, ref_fn, **kwargs)
        if name == "int8_gemm_oryon":
            res["speedup_x"] = 0.5
        return res

    monkeypatch.setattr(Path, "read_text", _low_meminfo)
    monkeypatch.setattr(snapdragon_local, "benchmark_kernel_parity", _slow_gemm_bench)
    rep = snapdragon_local.run_t46_snapdragon_local_suite()
    assert rep["oryon_simd_speedup_vs_numpy"] == pytest.approx(0.5)
    assert rep["execution_tiers"]["oryon_i8mm"]["gemm_speedup_vs_numpy"] == pytest.approx(0.5)
    assert all(not t["live_executed"] for t in rep["tiers"])
    ok_flag, crit, _ = verify_t46_snapdragon_local(rep)
    assert ok_flag is False
    assert any(c.id == "T46_TORCH_AUDIT_AND_PARITY" and not c.passed for c in crit)

    with pytest.raises(ValueError):
        snapdragon_local.SnapdragonXPlusSpec(wsl2_mem_available_gib=16.0).validate()


def test_verify_t46_hardened_execution_tiers_and_auto_selected() -> None:
    d_bad_type = _valid_t46_doc()
    d_bad_type["execution_tiers"] = []
    assert verify_t46_snapdragon_local(_sign_t46_doc(d_bad_type))[0] is False

    d_oryon_str = _valid_t46_doc()
    d_oryon_str["execution_tiers"]["oryon_i8mm"] = "invalid"
    assert verify_t46_snapdragon_local(_sign_t46_doc(d_oryon_str))[0] is False

    d_oryon_fail = _valid_t46_doc()
    d_oryon_fail["execution_tiers"]["oryon_i8mm"]["parity_ok"] = False
    assert verify_t46_snapdragon_local(_sign_t46_doc(d_oryon_fail))[0] is False

    d_np_fail = _valid_t46_doc()
    d_np_fail["execution_tiers"]["numpy_ref"] = {"parity_ok": False}
    assert verify_t46_snapdragon_local(_sign_t46_doc(d_np_fail))[0] is False

    d_auto_str = _valid_t46_doc()
    d_auto_str["execution_tiers"]["auto_selected"] = "invalid"
    assert verify_t46_snapdragon_local(_sign_t46_doc(d_auto_str))[0] is False

    d_auto_htp_bad_canary = _valid_t46_doc()
    d_auto_htp_bad_canary["execution_tiers"]["auto_selected"] = {
        "tier_used": "htp_qnn",
        "canary_ok": False,
        "parity_ok": True,
    }
    assert verify_t46_snapdragon_local(_sign_t46_doc(d_auto_htp_bad_canary))[0] is False

    d_auto_oryon_bad_canary = _valid_t46_doc()
    d_auto_oryon_bad_canary["execution_tiers"]["auto_selected"] = {
        "tier_used": "oryon_i8mm",
        "canary_ok": False,
        "parity_ok": True,
    }
    assert verify_t46_snapdragon_local(_sign_t46_doc(d_auto_oryon_bad_canary))[0] is False

    d_auto_bad_parity = _valid_t46_doc()
    d_auto_bad_parity["execution_tiers"]["auto_selected"] = {
        "tier_used": "htp_qnn",
        "canary_ok": True,
        "parity_ok": False,
    }
    assert verify_t46_snapdragon_local(_sign_t46_doc(d_auto_bad_parity))[0] is False

    d_auto_ok = _valid_t46_doc()
    d_auto_ok["execution_tiers"]["auto_selected"] = {
        "tier_used": "htp_qnn",
        "canary_ok": True,
        "parity_ok": True,
    }
    assert verify_t46_snapdragon_local(_sign_t46_doc(d_auto_ok))[0] is True


def test_verify_t46_hardened_top_level_csr_ram_equality_and_safe_int() -> None:
    d_csr_mismatch = _valid_t46_doc()
    d_csr_mismatch["csr_bytes"] = 12345
    assert verify_t46_snapdragon_local(_sign_t46_doc(d_csr_mismatch))[0] is False

    d_ram_mismatch = _valid_t46_doc()
    d_ram_mismatch["peak_ram_bytes"] = 12345
    assert verify_t46_snapdragon_local(_sign_t46_doc(d_ram_mismatch))[0] is False

    d_csr_match = _valid_t46_doc()
    d_csr_match["csr_bytes"] = d_csr_match["sheaf_bridge_csr"]["csr_bytes"]
    d_csr_match["peak_ram_bytes"] = d_csr_match["sheaf_bridge_csr"]["peak_ram_bytes"]
    assert verify_t46_snapdragon_local(_sign_t46_doc(d_csr_match))[0] is True

    d_big_mac = _valid_t46_doc()
    d_big_mac["op_counter"]["mac_ops"] = 10**400
    assert verify_t46_snapdragon_local(_sign_t46_doc(d_big_mac))[0] is False

    d_edge_mac = _valid_t46_doc()
    d_edge_mac["op_counter"]["mac_ops"] = 1 << 53
    assert verify_t46_snapdragon_local(_sign_t46_doc(d_edge_mac))[0] is False

    d_safe_mac = _valid_t46_doc()
    d_safe_mac["op_counter"]["mac_ops"] = (1 << 53) - 1
    assert verify_t46_snapdragon_local(_sign_t46_doc(d_safe_mac))[0] is True


def test_verify_t46_hardened_low_mem_020_gating_boundary() -> None:
    d_020_ok = _valid_t46_doc(avail_gib=0.20)
    assert d_020_ok["tiers"][0]["fits_wsl2_mem_available"] is True
    assert d_020_ok["tiers"][0]["live_executed"] is True
    assert verify_t46_snapdragon_local(d_020_ok)[0] is True

    d_020_nolive = _valid_t46_doc(avail_gib=0.20)
    d_020_nolive["tiers"][0]["live_executed"] = False
    ok_flag, crit, _ = verify_t46_snapdragon_local(_sign_t46_doc(d_020_nolive))
    assert ok_flag is False
    assert any(c.id == "T46_TIERS_STRUCT32K_AND_GATING" and not c.passed for c in crit)

    d_010_ok = _valid_t46_doc(avail_gib=0.10)
    assert all(not t["fits_wsl2_mem_available"] and not t["live_executed"] for t in d_010_ok["tiers"])
    assert verify_t46_snapdragon_local(d_010_ok)[0] is True


def test_local_e2e_runner_readonly_target_fails_fast(tmp_path: Path, monkeypatch: Any) -> None:
    ro_file = tmp_path / "readonly.json"
    ro_file.write_text("{}", encoding="utf-8")
    ro_file.chmod(0o444)

    live_link = tmp_path / "live_link.json"
    live_link.symlink_to(ro_file)
    dangling_link = tmp_path / "dangling_link.json"
    dangling_link.symlink_to(tmp_path / "missing_target.json")

    def _boom(**_kw: Any) -> dict[str, Any]:
        raise AssertionError("run_t46_snapdragon_local_suite should not run for unwritable --json-out")

    monkeypatch.setattr(local_e2e_runner, "run_t46_snapdragon_local_suite", _boom)
    try:
        assert local_e2e_runner.main(["--json-out", str(ro_file)]) == 1
        assert local_e2e_runner.main(["--json-out", str(live_link)]) == 1
        assert local_e2e_runner.main(["--json-out", str(dangling_link)]) == 1
        assert local_e2e_runner.main([]) == 1
    finally:
        ro_file.chmod(0o644)
