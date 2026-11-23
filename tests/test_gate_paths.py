"""Own pass/fail paths for gates the other tests do not fail on their criteria."""

from tests.support import criterion, latest

from sheaf_verify.synthetic import build


def _rows(tmp_path, name, mutate):
    store = build(tmp_path / name, mutate=mutate)
    return latest(store)


def test_h1_fails_when_gpt2_is_not_beside_hnt(tmp_path):
    def mutate(parts):
        del parts["metrics"]["spec"]["g1"]["arms"]["gpt2_bpe"]

    rows = _rows(tmp_path, "h1", mutate)
    assert rows["G1"]["status"] == "pass"
    assert rows["H1"]["status"] == "fail"
    assert criterion(rows["H1"], "h1.gpt2_bpe")["passed"] is False


def test_g2_fails_when_spectral_loses_to_its_control(tmp_path):
    def mutate(parts):
        row = parts["metrics"]["spec"]["g2"]["comparisons"][0]
        row["bpb_per_seed"] = [2.5, 2.6, 2.7]
        row["control_bpb_per_seed"] = [1.0, 1.1, 1.2]

    rows = _rows(tmp_path, "g2", mutate)
    assert rows["G1"]["status"] == "pass"
    assert rows["G2"]["status"] == "fail"


def test_stage_cards_fail_on_their_own_gate(tmp_path):
    def mutate(parts):
        for stage in ("H2", "H3", "H6"):
            parts["metrics"]["spec"]["stages"][stage]["work"] = "renamed"

    rows = _rows(tmp_path, "cards", mutate)
    assert rows["G2"]["status"] == "pass"
    assert rows["G3"]["status"] == "pass"
    assert rows["G6"]["status"] == "pass"
    for stage in ("H2", "H3", "H6"):
        assert rows[stage]["status"] == "fail"
        assert criterion(rows[stage], f"stage.{stage}")["passed"] is False


def test_p1_fails_when_an_ablation_is_missing(tmp_path):
    def mutate(parts):
        parts["metrics"]["spec"]["p1"]["arms"] = ["s4d"]

    rows = _rows(tmp_path, "p1", mutate)
    assert rows["P0"]["status"] == "pass"
    assert rows["P1"]["status"] == "fail"
    assert criterion(rows["P1"], "p1.arms")["passed"] is False


def test_p2_fails_when_heat_is_missing_from_the_pe_controls(tmp_path):
    def mutate(parts):
        parts["metrics"]["spec"]["p2"]["pe_controls"] = ["sinusoidal", "rope", "alibi", "nope", "heat_rope"]

    rows = _rows(tmp_path, "p2", mutate)
    assert rows["P1"]["status"] == "pass"
    assert rows["P2"]["status"] == "fail"
    assert criterion(rows["P2"], "p2.pe")["passed"] is False


def test_p3_and_h5_fail_when_fineweb_is_missing(tmp_path):
    def mutate(parts):
        del parts["metrics"]["spec"]["p3"]["fineweb_edu_val_bpb"]

    rows = _rows(tmp_path, "p3", mutate)
    assert rows["G5"]["status"] == "pass"
    assert rows["P3"]["status"] == "fail"
    assert rows["H5"]["status"] == "fail"
    assert criterion(rows["H5"], "p3.fineweb")["passed"] is False


def test_p5_fails_when_the_cuda_graph_is_off(tmp_path):
    def mutate(parts):
        parts["metrics"]["spec"]["p5"]["cuda_graph_all_arms"] = False

    rows = _rows(tmp_path, "p5", mutate)
    assert rows["G3"]["status"] == "pass"
    assert rows["P5"]["status"] == "fail"
    assert criterion(rows["P5"], "p5.cuda_graph")["passed"] is False


def test_g5_fails_when_a_size_loses_its_gain(tmp_path):
    def mutate(parts):
        parts["metrics"]["spec"]["g5"]["sizes"][0]["gain"] = 0

    rows = _rows(tmp_path, "g5", mutate)
    assert rows["G4"]["status"] == "pass"
    assert rows["G5"]["status"] == "fail"
    assert criterion(rows["G5"], "g5.gains")["passed"] is False


def test_p4_fails_when_the_15m_ladder_has_one_seed(tmp_path):
    def mutate(parts):
        for row in parts["metrics"]["spec"]["g5"]["sizes"]:
            if row["params"] == 15000000:
                row["seeds"] = 1

    rows = _rows(tmp_path, "p4", mutate)
    assert rows["G5"]["status"] == "pass"
    assert rows["P4"]["status"] == "fail"
    assert criterion(rows["P4"], "p4.seeds.15000000")["passed"] is False


def test_g6_fails_when_it_does_not_beat_a7(tmp_path):
    def mutate(parts):
        parts["metrics"]["spec"]["g6"]["bpb"] = 2.0

    rows = _rows(tmp_path, "g6", mutate)
    assert rows["G5"]["status"] == "pass"
    assert rows["G6"]["status"] == "fail"
    assert criterion(rows["G6"], "g6.bpb")["passed"] is False


def test_h7_fails_when_lm_eval_is_treated_as_the_headline(tmp_path):
    def mutate(parts):
        parts["metrics"]["spec"]["h7"]["headline"] = True

    rows = _rows(tmp_path, "h7", mutate)
    assert rows["H6"]["status"] == "pass"
    assert rows["H7"]["status"] == "fail"
    assert criterion(rows["H7"], "h7.headline")["passed"] is False


def test_p6_fails_when_the_replication_note_is_too_short(tmp_path):
    def mutate(parts):
        parts["metrics"]["spec"]["p6"]["replication_note"] = "too short"

    rows = _rows(tmp_path, "p6", mutate)
    assert rows["H7"]["status"] == "pass"
    assert rows["P6"]["status"] == "fail"
    assert criterion(rows["P6"], "p6.replication")["passed"] is False
