"""Negative controls for the evaluator's scientific acceptance boundary."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hadukenn_exec.runner import evaluate


def small_case(wrong=False):
    return {
        "id": "literal-negative-control",
        "family": "literal",
        "seed": 11,
        "depth": 1,
        "commands": [{"action": "set", "name": "x", "value": 3}, {"action": "query", "name": "x"}],
        "expected": [
            {"index": 0, "result": {"status": "ok"}},
            {"index": 1, "result": {"status": "ok", "value": "4" if wrong else "3", "unit": "1"}},
        ],
    }


class EvaluatorControls(unittest.TestCase):
    def test_wrong_golden_answer_cannot_pass(self):
        with tempfile.TemporaryDirectory() as temp, patch("hadukenn_exec.runner.generate_suite", return_value=[small_case(True)]), patch("hadukenn_exec.runner.intervention_pairs", return_value=[]):
            report = evaluate(Path(temp), benchmark=False)
        self.assertEqual(report["status"], "fail")
        self.assertEqual(len(report["failures"]), 3)

    def test_false_intervention_claim_cannot_pass(self):
        pair = {"id": "false-difference", "kind": "negative-control", "base": small_case(), "changed": small_case(), "relation": "different"}
        with tempfile.TemporaryDirectory() as temp, patch("hadukenn_exec.runner.generate_suite", return_value=[small_case()]), patch("hadukenn_exec.runner.intervention_pairs", return_value=[pair]):
            report = evaluate(Path(temp), benchmark=False)
        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["interventions"]["passed"], 0)

    def test_timed_workload_failure_cannot_pass(self):
        with tempfile.TemporaryDirectory() as temp, patch("hadukenn_exec.runner.generate_suite", return_value=[small_case()]), patch("hadukenn_exec.runner.intervention_pairs", return_value=[]), patch("hadukenn_exec.runner.benchmark_workloads", return_value=[small_case(True)]):
            report = evaluate(Path(temp), benchmark=True, repeats=1)
        self.assertEqual(report["status"], "fail")
        self.assertEqual(len(report["timing_failures"]), 60)


if __name__ == "__main__":
    unittest.main()
