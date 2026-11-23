from sheaf_verify.ledger import Ledger
from sheaf_verify.synthetic import TEST_KEY, build
from sheaf_verify.verify import verify_store


def test_second_run_still_verifies(tmp_path):
    from sheaf_verify.runner import SequenceClock, run

    store = build(tmp_path / "store")
    run(store, key=TEST_KEY, clock=SequenceClock("2026-06-01T00:00:00Z"))
    report = verify_store(store, key=TEST_KEY)
    assert report.problems == []


def test_verify_passes_on_synthetic_store(tmp_path):
    store = build(tmp_path / "store")
    report = verify_store(store, key=TEST_KEY)
    assert report.problems == []
    assert all(row["status"] == "pass" for row in report.gate_rows)
    assert all(item["answered"] for item in report.checklist)


def test_edited_ledger_line_fails(tmp_path):
    store = build(tmp_path / "store")
    path = Ledger(store.root).path
    text = path.read_bytes()
    path.write_bytes(text.replace(b'"pass"', b'"fail"', 1))
    report = verify_store(store, key=TEST_KEY)
    assert report.problems


def test_deleted_ledger_line_fails(tmp_path):
    store = build(tmp_path / "store")
    path = Ledger(store.root).path
    lines = path.read_bytes().splitlines(keepends=True)
    path.write_bytes(b"".join(lines[:-1]))
    report = verify_store(store, key=TEST_KEY)
    assert any("prev_sha256" in problem or "head" in problem or "sequence" in problem for problem in report.problems)


def test_reordered_ledger_fails(tmp_path):
    store = build(tmp_path / "store")
    path = Ledger(store.root).path
    lines = path.read_bytes().splitlines(keepends=True)
    swapped = lines[:1] + lines[2:3] + lines[1:2] + lines[3:]
    path.write_bytes(b"".join(swapped))
    report = verify_store(store, key=TEST_KEY)
    assert report.problems


def test_bad_signature_fails(tmp_path):
    store = build(tmp_path / "store")
    report = verify_store(store, key=b"sheaf-verify-other-test-key!!")
    assert any("signature" in problem for problem in report.problems)
