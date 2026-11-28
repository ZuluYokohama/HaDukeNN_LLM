import json
from pathlib import Path

from sheaf_verify.store import Store
from sheaf_verify.synthetic import TEST_KEY, build
from sheaf_verify.verify import verify_store

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "fixtures" / "passing"
PROFILE_FIXTURE = ROOT / "fixtures" / "profile-training"
KEY = ROOT / "fixtures" / "test_seal.key"


def _files(root: Path) -> dict[str, bytes]:
    found = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != ".lock":
            found[path.relative_to(root).as_posix()] = path.read_bytes()
    return found


def test_committed_fixture_matches_regeneration(tmp_path: Path):
    built = tmp_path / "passing"
    build(built)
    assert _files(built) == _files(FIXTURE)


def test_committed_fixture_verifies():
    report = verify_store(Store(FIXTURE), key=TEST_KEY)
    assert report.problems == []
    assert KEY.read_bytes() == TEST_KEY


def test_profile_records_stay_outside_the_canonical_store():
    canonical_head = json.loads((FIXTURE / "ledger" / "head.json").read_text(encoding="utf-8"))
    assert canonical_head["head_sha256"] == "307711a4d841c2c8fa0bbd954d15b2a585012608528c9f53b9ea08ad717bd7c4"
    canonical_records = [
        json.loads(line)
        for line in (FIXTURE / "ledger" / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(canonical_records) == 22
    assert all("profile" not in rec for rec in canonical_records)

    profile_head = json.loads((PROFILE_FIXTURE / "ledger" / "head.json").read_text(encoding="utf-8"))
    assert profile_head["head_sha256"] == "e247484042475d381eb454df95b718f55739a6cd0385006f142be257b8b86567"
    profile_records = [
        json.loads(line)
        for line in (PROFILE_FIXTURE / "ledger" / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(profile_records) == 22
    assert all(rec.get("profile") == "training" for rec in profile_records)

    report = verify_store(Store(PROFILE_FIXTURE), key=TEST_KEY)
    assert report.ok is True
    assert report.problems == []
