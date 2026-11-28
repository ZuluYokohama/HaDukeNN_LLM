"""Exercise the user-facing commands through real subprocesses and files."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class CommandLineTests(unittest.TestCase):
    def invoke(self, *args):
        return subprocess.run([sys.executable, "-m", "hadukenn_exec", *map(str, args)], cwd=ROOT, capture_output=True, text=True, check=False)

    def test_run_and_anchored_replay(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            result = self.invoke("run", "examples/transfers.json", "--out", out)
            self.assertEqual(result.returncode, 0, result.stderr)
            outcomes = json.loads((out / "outcomes.json").read_text())
            self.assertEqual(outcomes[10]["value"], "11")
            self.assertEqual(outcomes[15]["value"], "15")
            self.assertEqual(outcomes[18]["value"], "12")
            self.assertEqual(outcomes[20]["error"], "conflict")
            self.assertEqual(outcomes[22]["error"], "missing_reference")
            anchor = json.loads((out / "anchor.json").read_text())
            replay = self.invoke("replay", out / "trace.json", "--expected-head", anchor["head"])
            self.assertEqual(replay.returncode, 0, replay.stderr)
            self.assertEqual(json.loads(replay.stdout)["state_digest"], anchor["state_digest"])
            trace = json.loads((out / "trace.json").read_text())
            (out / "truncated.json").write_text(json.dumps(trace[:-1]))
            truncated = self.invoke("replay", out / "truncated.json", "--expected-head", anchor["head"])
            self.assertNotEqual(truncated.returncode, 0)

    def test_spans_cli_preserves_binary_prefix(self):
        from hadukenn_exec.spans import decode_graph
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            raw = b"\xff" + "\u03b1\u03b2".encode()
            (out / "source.bin").write_bytes(raw)
            result = self.invoke("spans", out / "source.bin", "--visible-bytes", "2", "--out", out / "spans.json")
            self.assertEqual(result.returncode, 0, result.stderr)
            graph = json.loads((out / "spans.json").read_text())
            self.assertEqual(decode_graph(graph), raw[:2])


if __name__ == "__main__":
    unittest.main()
