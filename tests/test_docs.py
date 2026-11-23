from pathlib import Path

from sheaf_verify.checklist import OBJECTIONS
from sheaf_verify.gates import GATES

ROOT = Path(__file__).resolve().parents[1]


def test_gate_ids_and_thresholds_are_in_the_gate_doc():
    text = (ROOT / "GATES.md").read_text(encoding="utf-8")
    for gate in GATES:
        assert gate.id in text
    assert "MIN_SEEDS = 5" in text
    assert "MAX_FLOP_RELATIVE_TOLERANCE = 0.05" in text
    assert "ALPHA = 0.05" in text
    assert "MIN_BOOTSTRAP_DRAWS = 1000" in text
    assert "MAX_BOOTSTRAP_DRAWS = 100000" in text


def test_reviewer_guide_lists_every_objection():
    text = (ROOT / "REVIEWER_GUIDE.md").read_text(encoding="utf-8")
    for objection in OBJECTIONS:
        assert objection.id in text


def test_tree_grants_no_license_and_has_no_operational_sensor_data():
    banned = ("Permission is hereby " + "granted", "well" + "bore", "drill" + "ing")
    skip = {".git", ".pytest_cache", ".venv", "__pycache__"}
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in skip or part.endswith(".egg-info") for part in path.parts):
            continue
        if "fixtures/passing" in path.as_posix():
            continue
        if path.suffix in {".py", ".md", ".toml", ".yml", ".json", ""} or path.name == "NOTICE":
            text = path.read_text(encoding="utf-8", errors="ignore")
            for phrase in banned:
                assert phrase not in text, path
    assert not (ROOT / "LICENSE").exists()
