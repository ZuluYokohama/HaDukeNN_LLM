from sheaf_verify.hashing import canonical_bytes, parse_json_bytes
from sheaf_verify.spec_gates import g0_criteria, recorded_g0


def test_recorded_g0_file_matches_the_metadata_record():
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "fixtures" / "g0-recorded" / "g0.json"
    document = parse_json_bytes(path.read_bytes(), "g0")
    assert document == recorded_g0()
    assert path.read_bytes() == canonical_bytes(recorded_g0())
    note = (path.parent / "README.md").read_text(encoding="utf-8").lower()
    assert "recorded metadata" in note
    assert "not corpus" in note or "not data" in note


def test_recorded_g0_flags_the_vocab_and_passes_the_other_counts():
    rows = {item.id: item for item in g0_criteria(recorded_g0())}
    assert rows["g0.fewer_units"].passed is True
    assert "27.35%" in rows["g0.fewer_units"].detail
    assert "181544" in rows["g0.fewer_units"].detail
    assert "249887" in rows["g0.fewer_units"].detail
    assert rows["g0.fallback"].passed is True
    assert rows["g0.build_cpu"].passed is True
    assert rows["g0.build_ram"].passed is True
    assert rows["g0.label"].passed is True
    assert rows["g0.vocab_cap"].passed is False
    assert "65536" in rows["g0.vocab_cap"].detail
    assert "64000" in rows["g0.vocab_cap"].detail


def test_swapping_the_recorded_g0_into_a_study_fails_only_the_cap(tmp_path):
    from tests.support import criterion, latest

    from sheaf_verify.synthetic import build

    def mutate(parts):
        parts["metrics"]["spec"]["g0"] = recorded_g0()

    store = build(tmp_path / "recorded", mutate=mutate)
    rows = latest(store)
    assert rows["G0"]["status"] == "fail"
    assert criterion(rows["G0"], "g0.vocab_cap")["passed"] is False
    assert criterion(rows["G0"], "g0.fewer_units")["passed"] is True
    assert criterion(rows["G0"], "g0.fallback")["passed"] is True
    assert criterion(rows["G0"], "g0.build_cpu")["passed"] is True
    assert criterion(rows["G0"], "g0.build_ram")["passed"] is True
