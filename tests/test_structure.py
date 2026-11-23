from sheaf_verify.checks import _boundary_identity, _control_agreement, _loop_closure, _supports
from sheaf_verify.synthetic import COMPLEX


def test_triangle_boundary_is_zero():
    passed, detail = _boundary_identity(COMPLEX)
    assert passed, detail


def test_open_face_fails_boundary():
    broken = {"edges": [[0, 1], [1, 2]], "faces": [[0, 1, 2]], "num_vertices": 3}
    passed, detail = _boundary_identity(broken)
    assert not passed
    assert "missing edge" in detail


def test_loop_closure_detects_disagreement():
    sheaf = {
        "loops": [{"path_a": ["op0"], "path_b": ["op1"]}],
        "operations": [
            {"id": "op0", "morphism": {"a": "b"}, "source": "s0", "target": "s1"},
            {"id": "op1", "morphism": {"a": "c"}, "source": "s0", "target": "s1"},
        ],
        "states": ["s0", "s1"],
    }
    passed, detail = _loop_closure(sheaf)
    assert not passed
    assert "do not compose" in detail


def test_superiority_requires_both_controls():
    computed = {
        "superword_bpe": {"p": 0.01, "high": -0.1, "g": -0.4, "t": -2.0, "df": 8.0, "low": -0.5},
        "randomized_hypergraph": {"p": 0.4, "high": 0.2, "g": -0.1, "t": -0.5, "df": 8.0, "low": -0.4},
    }
    claim = {"claimed": True, "arm": "hypergraph", "against": "superword_bpe", "metric": "bpb"}
    assert _supports(computed["superword_bpe"])
    assert not _supports(computed["randomized_hypergraph"])
    assert _control_agreement(claim, computed).passed is False
    assert _control_agreement({"claimed": False}, computed).passed is True
