from pathlib import Path

from sheaf_verify.checklist import OBJECTIONS
from sheaf_verify.gates import GATES

ROOT = Path(__file__).resolve().parents[1]


def test_gate_ids_and_thresholds_are_in_the_gate_doc():
    text = (ROOT / "GATES.md").read_text(encoding="utf-8")
    for gate in GATES:
        assert gate.id in text
    for line in (
        "PILOT_SEEDS = 3",
        "HEADLINE_SEEDS = 5",
        "G0_MIN_UNIT_REDUCTION = 0.15",
        "G0_MAX_BYTE_FALLBACK_PERCENT = 1",
        "G0_VOCAB_CAP = 64000",
        "G0_RECORDED_VOCAB = 65536",
        "G0_MAX_BUILD_CPU_HOURS = 2",
        "G0_MAX_BUILD_RAM_GB = 16",
        "G1_SD_MARGIN = 1",
        "G3_MAX_REL_ERR = 0.001",
        "G3_MAX_SHEAFHEAT_FRACTION = 0.15",
        "G4_MIN_BASELINE_SD = 2",
        "P4_SEEDS_15M = 3",
        "P4_SEEDS_30M = 3",
        "P4_SEEDS_60M = 2",
        "P4_SEEDS_120M = 1",
        "MAX_FLOP_RELATIVE_TOLERANCE = 0.05",
        "ALPHA = 0.05",
        "MIN_BOOTSTRAP_DRAWS = 1000",
        "MAX_BOOTSTRAP_DRAWS = 100000",
    ):
        assert line in text


def test_reviewer_guide_lists_every_objection():
    text = (ROOT / "REVIEWER_GUIDE.md").read_text(encoding="utf-8")
    for objection in OBJECTIONS:
        assert objection.id in text


def test_license_notice_and_export_hygiene():
    banned = ("Permission is hereby " + "granted", "well" + "bore", "drill" + "ing", "DEVn" + "-1")
    notice = (ROOT / "NOTICE").read_text(encoding="utf-8")
    required = (
        "Required Notice: Copyright (c) 2026 Blake Jones (ZuluYokohama). "
        "Commercial use requires a separate license from the author; contact via GitHub @ZuluYokohama."
    )
    assert required in notice
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert license_text.startswith("# PolyForm Noncommercial License 1.0.0\n")
    assert "Noncommercial Purposes" in license_text
    overview = (ROOT / "docs" / "index.html").read_text(encoding="utf-8")
    assert "HaDukeN //\\\\// HGNLLM Experiments" in overview
    assert "partial" in overview
    assert "65,536" in overview
    assert "64,000" in overview
    assert ">fail<" in overview or 'class="status fail"' in overview
    assert "https://github.com/ZuluYokohama/HaDukeNN_LLM/tree/iter/g0" in overview
    assert (ROOT / "docs" / "briefing.html").is_file()
    assert "iter/g0" in (ROOT / "iterations" / "INDEX.md").read_text(encoding="utf-8")
    diagram = (ROOT / "docs" / "control-graph.mmd").read_text(encoding="utf-8").strip()
    assert diagram in (ROOT / "GATES.md").read_text(encoding="utf-8")
    graph = (ROOT / "docs" / "graph.html").read_text(encoding="utf-8")
    assert diagram in graph
    for phrase in ("pe_heat", "equal-size", "struct32k", "GPT-2", 'class="compare"'):
        assert phrase in graph
        assert phrase in overview
    contributing = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    assert "main is protected" in contributing
    assert "pull request" in contributing
    template = (ROOT / ".github" / "PULL_REQUEST_TEMPLATE.md").read_text(encoding="utf-8")
    for phrase in (
        "Gates touched",
        "Evidence files",
        "Ledger head before",
        "Ledger head after",
        "amendment",
        "weights",
    ):
        assert phrase in template
    skip = {".git", ".pytest_cache", ".venv", "__pycache__"}
    scanned = {".py", ".md", ".toml", ".yml", ".json", ".html", ".css", ""}
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in skip or part.endswith(".egg-info") for part in path.parts):
            continue
        assert path.suffix.lower() not in {".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".bin"}, path
        if "fixtures/passing" in path.as_posix():
            continue
        if path.suffix in scanned or path.name == "NOTICE":
            text = path.read_text(encoding="utf-8", errors="ignore")
            for phrase in banned:
                assert phrase not in text, path
            for drive in ("C:" + "\\", "E:" + "\\"):
                assert drive not in text, path
