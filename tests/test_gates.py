from tests.support import criterion, latest

from sheaf_verify.synthetic import build


def _fail(tmp_path, name, mutate):
    store = build(tmp_path / name, mutate=mutate)
    return latest(store)


def test_passing_study(tmp_path):
    store = build(tmp_path / "store")
    rows = latest(store)
    assert all(row["status"] == "pass" for row in rows.values())
    assert criterion(rows["G0"], "g0.vocab_cap")["passed"] is True
    assert criterion(rows["G1"], "g1.bpb")["passed"] is True
    assert criterion(rows["G1"], "g1.seeds.hnt64k")["passed"] is True
    agreement = criterion(rows["G4"], "stats.control_agreement")
    assert agreement["passed"] is True
    assert "disagree" in agreement["detail"]


def test_held_out_document_fails_h0(tmp_path):
    def mutate(parts):
        parts["hypergraph"]["document_ids"] = ["doc-train", "doc-test"]
    rows = _fail(tmp_path, "leak", mutate)
    assert rows["H0"]["status"] == "fail"
    assert criterion(rows["H0"], "hypergraph.documents")["passed"] is False


def test_missing_graph_flops_fail_g1(tmp_path):
    def mutate(parts):
        flops = parts["metrics"]["arms"]["hypergraph"]["flops"]
        flops["graph_build"] = 0
        flops["total"] = flops["train"]
    rows = _fail(tmp_path, "flops", mutate)
    assert rows["G1"]["status"] == "fail"


def test_unequal_train_bytes_fail_g1(tmp_path):
    def mutate(parts):
        parts["metrics"]["arms"]["superword_bpe"]["train_bytes"] = 1
    rows = _fail(tmp_path, "bytes", mutate)
    assert criterion(rows["G1"], "bytes.superword_bpe")["passed"] is False


def test_three_seeds_fail_h4(tmp_path):
    def mutate(parts):
        for arm in parts["seeds"]["arms"]:
            parts["seeds"]["arms"][arm] = parts["seeds"]["arms"][arm][:3]
            parts["metrics"]["arms"][arm]["bpb_per_seed"] = parts["metrics"]["arms"][arm]["bpb_per_seed"][:3]
    rows = _fail(tmp_path, "seeds", mutate)
    assert rows["H4"]["status"] == "fail"
    assert rows["H7"]["status"] == "fail"


def test_wrong_reported_statistic_fails_h7(tmp_path):
    def mutate(parts):
        parts["metrics"]["reported"] = {
            "hypergraph_vs_randomized_hypergraph": {"df": 0, "g": 0, "high": 0, "low": 0, "p": 0, "t": 0},
            "hypergraph_vs_superword_bpe": {"df": 0, "g": 0, "high": 0, "low": 0, "p": 0, "t": 0},
        }
    rows = _fail(tmp_path, "reported", mutate)
    assert rows["G4"]["status"] == "fail"
    assert criterion(rows["G4"], "stats.reported.hypergraph_vs_superword_bpe")["passed"] is False


def test_missing_bpe_arm_fails_h1(tmp_path):
    def mutate(parts):
        del parts["metrics"]["arms"]["superword_bpe"]
    rows = _fail(tmp_path, "bpe", mutate)
    assert rows["H1"]["status"] == "fail"


def test_same_incidence_fails_h3(tmp_path):
    def mutate(parts):
        parts["random"]["hyperedges"] = parts["hypergraph"]["hyperedges"]
        arm = parts["metrics"]["arms"]["randomized_hypergraph"]
        from sheaf_verify.checks import incidence_sha256
        arm["incidence_sha256"] = incidence_sha256(parts["random"])
    rows = _fail(tmp_path, "incidence", mutate)
    assert criterion(rows["G4"], "random.incidence")["passed"] is False


def test_token_denominator_fails_h6(tmp_path):
    def mutate(parts):
        parts["metrics"]["arms"]["hypergraph"]["eval"]["denominator"] = "tokens"
    rows = _fail(tmp_path, "denom", mutate)
    assert criterion(rows["G1"], "bpb.denominator.hypergraph")["passed"] is False


def test_unequal_scored_bytes_fail_h6(tmp_path):
    def mutate(parts):
        parts["metrics"]["arms"]["hypergraph"]["eval"]["scored_bytes"] = 8
    rows = _fail(tmp_path, "scored", mutate)
    assert rows["G1"]["status"] == "fail"


def test_future_perturbation_fails_h2(tmp_path):
    def mutate(parts):
        parts["trace"]["perturbed_outputs"][0] = "changed"
    rows = _fail(tmp_path, "causal", mutate)
    assert criterion(rows["H4"], "causality.past")["passed"] is False


def test_vacuous_probe_fails_h2(tmp_path):
    def mutate(parts):
        parts["trace"]["perturbed_outputs"][3] = "c3"
    rows = _fail(tmp_path, "vacuous", mutate)
    assert criterion(rows["H4"], "causality.probe")["passed"] is False


def test_superiority_fails_when_controls_disagree(tmp_path):
    def mutate(parts):
        parts["metrics"]["superiority"]["claimed"] = True
    rows = _fail(tmp_path, "claim", mutate)
    assert criterion(rows["G4"], "stats.superiority")["passed"] is True
    assert criterion(rows["G4"], "stats.control_agreement")["passed"] is False
    assert rows["G4"]["status"] == "fail"


def test_broken_complex_fails_h0(tmp_path):
    def mutate(parts):
        parts["hypergraph"]["complex"] = {"edges": [[0, 1], [1, 2]], "faces": [[0, 1, 2]], "num_vertices": 3}
    rows = _fail(tmp_path, "complex", mutate)
    assert criterion(rows["H0"], "hypergraph.boundary")["passed"] is False


def test_open_sheaf_loop_fails_h2(tmp_path):
    def mutate(parts):
        parts["hypergraph"]["sheaf"]["operations"][1]["morphism"] = {"a": "c"}
    rows = _fail(tmp_path, "loop", mutate)
    assert criterion(rows["H4"], "sheaf.loop_closure")["passed"] is False


def test_cuda_kernel_without_host_reference_fails_h5(tmp_path):
    def mutate(parts):
        parts["kernel_meta"] = {"backend": "cuda", "entry": "apply"}
    rows = _fail(tmp_path, "cuda", mutate)
    assert criterion(rows["G3"], "kernel.host_reference")["passed"] is False


def test_cuda_kernel_with_host_reference_passes(tmp_path):
    def mutate(parts):
        parts["kernel_meta"] = {"backend": "cuda", "entry": "apply"}
        parts["host_reference"] = b"def apply_reference(tokens):\n    return list(tokens)\n"
    store = build(tmp_path / "cuda-ok", mutate=mutate)
    rows = latest(store)
    assert rows["G3"]["status"] == "pass"
    assert all(row["status"] == "pass" for row in rows.values())


def test_claim_registered_after_metrics_fails_p1(tmp_path):
    from sheaf_verify.hashing import canonical_bytes
    from sheaf_verify.runner import run
    from sheaf_verify.synthetic import TEST_KEY

    store = build(tmp_path / "late-claim")
    claim = store.binding()["claim"][1]
    store.register(
        kind="claim",
        role="claim",
        registered_at="2026-02-01T00:00:00Z",
        payload=canonical_bytes({
            "alpha": 0.05,
            "description": "A later claim that was written after the metrics existed.",
            "direction": "less",
            "format": "sheaf-verify.claim.v1",
            "hypothesis_id": "late",
            "primary_contrast": "superword_bpe",
            "primary_metric": "bpb",
        }),
        inputs=claim["inputs"],
    )
    run(store, key=TEST_KEY)
    rows = latest(store)
    assert criterion(rows["P0"], "stage.claim_before_metrics")["passed"] is False


def test_two_pilot_seeds_fail_g1(tmp_path):
    def mutate(parts):
        for arm in parts["metrics"]["spec"]["g1"]["arms"].values():
            arm["bpb_per_seed"] = arm["bpb_per_seed"][:2]
    rows = _fail(tmp_path, "pilot-seeds", mutate)
    assert rows["G1"]["status"] == "fail"
    assert criterion(rows["G1"], "g1.seeds.hnt64k")["passed"] is False


def test_three_headline_seeds_fail_g4_only(tmp_path):
    def mutate(parts):
        parts["metrics"]["spec"]["g4"]["headline"]["bpb_per_seed"] = [1.40, 1.50, 1.45]
    rows = _fail(tmp_path, "headline-seeds", mutate)
    assert rows["G1"]["status"] == "pass"
    assert rows["G4"]["status"] == "fail"
    assert criterion(rows["G4"], "g4.seeds.headline")["passed"] is False
