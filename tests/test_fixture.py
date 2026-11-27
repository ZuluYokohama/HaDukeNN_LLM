from pathlib import Path

from sheaf_verify.store import Store
from sheaf_verify.synthetic import TEST_KEY, build
from sheaf_verify.verify import verify_store

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "fixtures" / "passing"
PROFILE = ROOT / "fixtures" / "profile-training"
KEY = ROOT / "fixtures" / "test_seal.key"
CANONICAL_HEAD = "307711a4d841c2c8fa0bbd954d15b2a585012608528c9f53b9ea08ad717bd7c4"
PROFILE_HEAD = "e247484042475d381eb454df95b718f55739a6cd0385006f142be257b8b86567"


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
    report = verify_store(__import__("sheaf_verify.store", fromlist=["Store"]).Store(FIXTURE), key=TEST_KEY)
    assert report.problems == []
    assert KEY.read_bytes() == TEST_KEY


def test_profile_records_stay_outside_the_canonical_store():
    import json

    passing_head = json.loads((FIXTURE / "ledger" / "head.json").read_text(encoding="utf-8"))["head_sha256"]
    profile_head = json.loads((PROFILE / "ledger" / "head.json").read_text(encoding="utf-8"))["head_sha256"]
    assert passing_head == CANONICAL_HEAD
    assert profile_head == PROFILE_HEAD
    passing_line = (FIXTURE / "ledger" / "ledger.jsonl").read_text(encoding="utf-8").splitlines()[0]
    profile_line = (PROFILE / "ledger" / "ledger.jsonl").read_text(encoding="utf-8").splitlines()[0]
    assert '"profile"' not in passing_line
    assert '"profile":"training"' in profile_line
    report = verify_store(Store(PROFILE), key=TEST_KEY)
    assert report.problems == []
