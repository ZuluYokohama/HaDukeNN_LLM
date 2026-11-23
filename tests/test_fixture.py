from pathlib import Path

from sheaf_verify.synthetic import TEST_KEY, build
from sheaf_verify.verify import verify_store

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "fixtures" / "passing"
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
    report = verify_store(__import__("sheaf_verify.store", fromlist=["Store"]).Store(FIXTURE), key=TEST_KEY)
    assert report.problems == []
    assert KEY.read_bytes() == TEST_KEY
