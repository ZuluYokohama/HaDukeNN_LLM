"""Independent semantic and integrity controls for the structured-command pilot."""

from __future__ import annotations

import copy
import hashlib
import json
import random
import unittest
from fractions import Fraction

from hadukenn_exec.baselines import CachedDAG, Reference
from hadukenn_exec.core import Kernel


def outcome(result):
    """Compare the command's meaning separately from audit metadata."""
    fields = ("status", "error") if result.get("status") == "error" else ("status", "value", "unit")
    return {key: result[key] for key in fields if key in result}


class KernelSemanticTests(unittest.TestCase):
    def setUp(self):
        self.kernel = Kernel()

    def execute(self, action, **kwargs):
        return self.kernel.execute({"action": action, **kwargs})

    def put(self, name, value, unit="1", scope="main", **kwargs):
        result = self.execute("set", name=name, value=value, unit=unit, scope=scope, **kwargs)
        self.assertEqual(result["status"], "ok", result)
        return result

    def derive(self, name, op, left=None, right=None, scope="main", value=None):
        args = {"value": value} if op in ("copy", "neg") else {"left": left, "right": right}
        result = self.execute("derive", name=name, op=op, args=args, scope=scope)
        self.assertEqual(result["status"], "ok", result)
        return result

    def check(self, name, value, unit="1", scope="main"):
        self.assertEqual(
            outcome(self.execute("query", name=name, scope=scope)),
            {"status": "ok", "value": str(Fraction(value)), "unit": unit},
        )

    def check_error(self, result, error):
        self.assertEqual(outcome(result), {"status": "error", "error": error}, result)

    def test_role_order_changes_subtraction(self):
        self.put("a", 20, "L")
        self.put("b", 5, "L")
        self.derive("ab", "sub", "a", "b")
        self.derive("ba", "sub", "b", "a")
        self.check("ab", 15, "L")
        self.check("ba", -15, "L")

    def test_exact_fraction_chain(self):
        self.put("a", "1/3")
        self.put("b", "1/6")
        self.put("three", 3)
        self.derive("sum", "add", "a", "b")
        self.derive("product", "mul", "sum", "three")
        self.derive("ratio", "div", "product", "three")
        self.check("sum", "1/2")
        self.check("product", "3/2")
        self.check("ratio", "1/2")

    def test_negation_and_copy_preserve_units(self):
        self.put("x", "2/7", "m")
        self.derive("negative", "neg", value="x")
        self.derive("copy", "copy", value="negative")
        self.check("negative", "-2/7", "m")
        self.check("copy", "-2/7", "m")

    def test_dimensional_scalar_and_ratio_rules(self):
        self.put("x", 12, "m")
        self.put("y", 3, "m")
        self.put("s", 2)
        self.derive("scaled_left", "mul", "s", "x")
        self.derive("scaled_right", "mul", "x", "s")
        self.derive("halved", "div", "x", "s")
        self.derive("ratio", "div", "x", "y")
        self.check("scaled_left", 24, "m")
        self.check("scaled_right", 24, "m")
        self.check("halved", 6, "m")
        self.check("ratio", 4)

    def test_incompatible_units_are_explicit(self):
        self.put("distance", 3, "m")
        self.put("time", 2, "s")
        for op in ("add", "sub", "mul", "div"):
            with self.subTest(op=op):
                self.derive(op, op, "distance", "time")
                self.check_error(self.execute("query", name=op), "unit_mismatch")

    def test_division_by_zero(self):
        self.put("a", 7)
        self.put("zero", 0)
        self.derive("q", "div", "a", "zero")
        self.check_error(self.execute("query", name="q"), "division_by_zero")

    def test_float_bool_and_invalid_rationals_refused_atomically(self):
        self.put("x", 7)
        for value in (True, False, 1.25, "not-a-number", "1/0", None, [], {}):
            with self.subTest(value=repr(value)):
                before = self.kernel.state_digest()
                result = self.execute("set", name="x", value=value)
                self.check_error(result, "invalid_command")
                self.assertEqual(before, self.kernel.state_digest())
                self.check("x", 7)

    def test_malformed_actions_refused_without_uncaught_exception(self):
        self.put("x", 7)
        for action in ([], {}, None, 3, True, "unknown"):
            with self.subTest(action=action):
                before = self.kernel.state_digest()
                self.check_error(self.kernel.execute({"action": action}), "invalid_command")
                self.assertEqual(before, self.kernel.state_digest())

    def test_non_json_input_refusal_is_replayable(self):
        for command in (None, [], {"action": "set", "name": "a", "value": float("nan")},
                        {"action": "set", "name": "a", "value": 1, "extra": (1, 2)}):
            with self.subTest(command=repr(command)):
                kernel = Kernel()
                result = kernel.execute(command)
                self.check_error(result, "invalid_command")
                replayed = Kernel.replay(kernel.export_trace())
                self.assertEqual(replayed.export_trace(), kernel.export_trace())

    def test_deep_metadata_is_refused_before_state_mutation(self):
        nested = "leaf"
        for _ in range(80):
            nested = {"nested": nested}
        before = self.kernel.state_digest()
        result = self.kernel.execute({"action": "set", "name": "a", "value": 7, "metadata": nested})
        self.check_error(result, "invalid_command")
        self.assertEqual(before, self.kernel.state_digest())
        self.assertEqual(Kernel.replay(self.kernel.export_trace()).export_trace(), self.kernel.export_trace())

    def test_unknown_operator_refused_without_replacement(self):
        self.put("x", 7)
        before = self.kernel.state_digest()
        result = self.execute("derive", name="x", op="shell", args={"left": "x", "right": "x"})
        self.check_error(result, "invalid_operator")
        self.assertEqual(before, self.kernel.state_digest())
        self.check("x", 7)

    def test_named_roles_are_required(self):
        for args in ({"a": "x", "b": "y"}, {"left": "x"}, {"left": "x", "right": "y", "extra": "x"}):
            with self.subTest(args=args):
                before = self.kernel.state_digest()
                result = self.execute("derive", name="bad", op="sub", args=args)
                self.check_error(result, "invalid_command")
                self.assertEqual(before, self.kernel.state_digest())

    def test_forward_reference_is_resolved_only_when_available(self):
        self.derive("later", "copy", value="future")
        self.check_error(self.execute("query", name="later"), "missing_reference")
        self.put("future", 19)
        self.check("later", 19)

    def test_revisions_invalidate_transitive_cached_results(self):
        self.put("a", 20)
        self.put("b", 5)
        self.put("two", 2)
        self.derive("difference", "sub", "a", "b")
        self.derive("twice", "mul", "difference", "two")
        self.check("twice", 30)
        self.check("twice", 30)
        self.put("a", 40)
        self.check("difference", 35)
        self.check("twice", 70)

    def test_replacing_operator_updates_descendants(self):
        self.put("a", 20)
        self.put("b", 5)
        self.derive("mid", "sub", "a", "b")
        self.derive("out", "neg", value="mid")
        self.check("out", -15)
        self.derive("mid", "add", "a", "b")
        self.check("out", -25)

    def test_retract_and_recreate_never_resurrect_stale_cache(self):
        self.put("a", 20)
        self.put("b", 5)
        self.derive("out", "sub", "a", "b")
        self.check("out", 15)
        self.assertEqual(self.execute("retract", name="a")["status"], "ok")
        self.check_error(self.execute("query", name="out"), "missing_reference")
        self.put("a", 8)
        self.check("out", 3)

    def test_changed_dependencies_invalidate_new_inputs(self):
        self.put("a", 3)
        self.put("b", 7)
        self.derive("out", "copy", value="a")
        self.check("out", 3)
        self.derive("out", "copy", value="b")
        self.check("out", 7)
        self.put("b", 11)
        self.check("out", 11)
        self.put("a", 999)
        self.check("out", 11)

    def test_snapshot_branch_remains_independent_of_parent(self):
        self.put("a", 20)
        self.put("b", 5)
        self.derive("out", "sub", "a", "b")
        self.check("out", 15)
        self.assertEqual(self.execute("fork", new_scope="hypothesis")["status"], "ok")
        self.put("a", 100)
        self.check("out", 95)
        self.check("out", 15, scope="hypothesis")
        self.put("b", 10, scope="hypothesis")
        self.check("out", 10, scope="hypothesis")
        self.check("out", 95)

    def test_snapshot_retraction_does_not_cross_scopes(self):
        self.put("a", 5)
        self.derive("out", "copy", value="a")
        self.execute("fork", new_scope="past")
        self.execute("retract", name="a")
        self.check_error(self.execute("query", name="out"), "missing_reference")
        self.check("out", 5, scope="past")

    def test_later_parent_binding_is_not_visible_inside_snapshot(self):
        self.execute("fork", new_scope="branch")
        self.put("new", 23)
        self.check_error(self.execute("query", scope="branch", name="new"), "missing_reference")

    def test_cycles_are_refused_at_evaluation_and_can_be_repaired(self):
        self.derive("a", "copy", value="b")
        self.derive("b", "copy", value="a")
        self.check_error(self.execute("query", name="a"), "cycle")
        self.put("b", 8)
        self.check("a", 8)

    def test_self_cycle(self):
        self.derive("a", "copy", value="a")
        self.check_error(self.execute("query", name="a"), "cycle")

    def test_shared_dependencies_are_not_mistaken_for_cycles(self):
        self.put("x", 3)
        self.derive("left", "copy", value="x")
        self.derive("right", "copy", value="x")
        self.derive("diamond", "add", "left", "right")
        self.check("diamond", 6)
        self.put("x", 4)
        self.check("diamond", 8)

    def test_source_identity_is_immutable_and_unicode_exact(self):
        text = "A → B; μ=⅓; café\n"
        result = self.execute("source", id="observation", text=text)
        self.assertEqual(result["status"], "ok")
        self.put("x", 1, source="observation")
        before = self.kernel.state_digest()
        self.assertEqual(self.execute("source", id="observation", text=text)["status"], "ok")
        self.assertEqual(before, self.kernel.state_digest())
        self.check_error(self.execute("source", id="observation", text="changed"), "conflict")
        self.assertEqual(before, self.kernel.state_digest())
        trace = self.kernel.export_trace()
        self.assertEqual(trace[0]["command"]["text"].encode("utf-8"), text.encode("utf-8"))
        source = next(s for s in self.kernel.graph()["sources"] if s["id"] == "observation")
        self.assertEqual(source["text"], text)
        self.assertEqual(source["byte_length"], len(text.encode("utf-8")))
        self.assertEqual(source["sha256"], hashlib.sha256(text.encode("utf-8")).hexdigest())

    def test_explicit_missing_source_refused_atomically(self):
        before = self.kernel.state_digest()
        result = self.execute("set", name="a", value=1, source="missing")
        self.check_error(result, "missing_reference")
        self.assertEqual(before, self.kernel.state_digest())

    def test_assertion_conflict_is_recorded_without_state_change(self):
        self.put("x", 7, "m")
        self.assertEqual(outcome(self.execute("assert", name="x", value=7, unit="m")),
                         {"status": "ok", "value": "7", "unit": "m"})
        before = self.kernel.state_digest()
        self.check_error(self.execute("assert", name="x", value=8, unit="m"), "conflict")
        self.assertEqual(before, self.kernel.state_digest())
        self.assertEqual(self.kernel.export_trace()[-1]["result"]["error"], "conflict")

    def test_nuisance_information_does_not_change_result(self):
        self.put("a", 12)
        self.put("b", 7)
        self.derive("out", "sub", "a", "b")
        self.check("out", 5)
        self.put("unrelated", 999999)
        self.check("out", 5)
        self.put("unrelated", -999999)
        self.check("out", 5)

    def test_consistent_renaming_preserves_result(self):
        results = []
        for names in (("a", "b", "out"), ("entity_71", "entity_12", "answer_44")):
            kernel = Kernel()
            commands = [
                {"action": "set", "name": names[0], "value": 12},
                {"action": "set", "name": names[1], "value": 7},
                {"action": "derive", "name": names[2], "op": "sub", "args": {"left": names[0], "right": names[1]}},
                {"action": "query", "name": names[2]},
            ]
            for command in commands:
                result = kernel.execute(command)
            results.append(outcome(result))
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[0]["value"], "5")

    def test_bounded_steps_fail_explicitly(self):
        kernel = Kernel(max_steps=2)
        for command in (
            {"action": "set", "name": "a", "value": 2},
            {"action": "set", "name": "b", "value": 3},
            {"action": "derive", "name": "out", "op": "add", "args": {"left": "a", "right": "b"}},
        ):
            self.assertEqual(kernel.execute(command)["status"], "ok")
        before = kernel.state_digest()
        self.check_error(kernel.execute({"action": "query", "name": "out"}), "resource_limit")
        self.assertEqual(before, kernel.state_digest())

    def test_active_node_limit_and_replacement(self):
        kernel = Kernel(max_nodes=2)
        self.assertEqual(kernel.execute({"action": "set", "name": "a", "value": 1})["status"], "ok")
        self.assertEqual(kernel.execute({"action": "set", "name": "b", "value": 2})["status"], "ok")
        before = kernel.state_digest()
        self.check_error(kernel.execute({"action": "set", "name": "c", "value": 3}), "resource_limit")
        self.assertEqual(before, kernel.state_digest())
        self.assertEqual(kernel.execute({"action": "set", "name": "a", "value": 7})["status"], "ok")
        self.assertEqual(outcome(kernel.execute({"action": "query", "name": "a"})),
                         {"status": "ok", "value": "7", "unit": "1"})

    def test_rejected_fork_does_not_create_partial_scope(self):
        kernel = Kernel(max_nodes=3)
        kernel.execute({"action": "set", "name": "a", "value": 1})
        kernel.execute({"action": "set", "name": "b", "value": 2})
        before = kernel.state_digest()
        self.check_error(kernel.execute({"action": "fork", "new_scope": "large"}), "resource_limit")
        self.assertEqual(before, kernel.state_digest())

    def test_query_reads_record_named_roles_and_current_revisions(self):
        self.put("a", 20)
        self.put("b", 5)
        self.derive("difference", "sub", "a", "b")
        self.check("difference", 15)
        receipt = self.kernel.export_trace()[-1]
        evaluated = next(event for event in receipt["evidence"]
                         if event["event"] == "evaluate" and event["name"] == "difference")
        self.assertEqual(evaluated["reads"]["left"]["name"], "a")
        self.assertEqual(evaluated["reads"]["right"]["name"], "b")
        old_revision = evaluated["reads"]["left"]["revision"]
        self.put("a", 30)
        self.check("difference", 25)
        fresh = next(event for event in self.kernel.export_trace()[-1]["evidence"]
                     if event["event"] == "evaluate" and event["name"] == "difference")
        self.assertGreater(fresh["reads"]["left"]["revision"], old_revision)

    def test_cached_result_receipt_points_to_actual_previous_computation(self):
        self.put("a", 3)
        self.check("a", 3)
        first_query = self.kernel.export_trace()[-1]
        self.check("a", 3)
        reused = next(event for event in self.kernel.export_trace()[-1]["evidence"]
                      if event["event"] == "cache_reuse" and event["name"] == "a")
        self.assertEqual(reused["computed_at"], first_query["sequence"])
        self.assertTrue(any(event["event"] == "evaluate" and event["name"] == "a"
                            for event in first_query["evidence"]))

    def test_overflowing_derived_value_is_never_cached_as_accepted(self):
        self.put("a", 2**2500)
        self.put("b", 2**2500)
        self.derive("too_big", "mul", "a", "b")
        for _ in range(3):
            self.check_error(self.execute("query", name="too_big"), "resource_limit")
        self.put("b", 1)
        self.check("too_big", 2**2500)

    def test_command_inputs_are_copied(self):
        command = {"action": "set", "name": "x", "value": 3, "metadata": {"note": "original"}}
        self.kernel.execute(command)
        command["value"] = 999
        command["metadata"]["note"] = "changed"
        self.check("x", 3)
        self.assertEqual(self.kernel.export_trace()[0]["command"]["value"], 3)
        self.assertEqual(self.kernel.export_trace()[0]["command"]["metadata"]["note"], "original")

    def test_exported_trace_is_detached(self):
        self.put("x", 7)
        trace = self.kernel.export_trace()
        trace[0]["command"]["value"] = 999
        self.assertEqual(self.kernel.export_trace()[0]["command"]["value"], 7)
        self.check("x", 7)

    def test_graph_export_is_detached(self):
        self.put("x", 7)
        before = self.kernel.state_digest()
        graph = self.kernel.graph()
        self.assertIsInstance(graph, (dict, list))
        json.dumps(graph, allow_nan=False)
        if isinstance(graph, dict):
            graph.clear()
        else:
            graph[:] = []
        self.assertEqual(before, self.kernel.state_digest())
        self.check("x", 7)


class KernelReplayTests(unittest.TestCase):
    def make_trace(self):
        kernel = Kernel()
        commands = [
            {"action": "source", "id": "original", "text": "a=20, b=5"},
            {"action": "set", "name": "a", "value": 20, "source": "original"},
            {"action": "set", "name": "b", "value": 5, "source": "original"},
            {"action": "derive", "name": "out", "op": "sub", "args": {"left": "a", "right": "b"}},
            {"action": "query", "name": "out"},
            {"action": "fork", "new_scope": "past"},
            {"action": "set", "name": "a", "value": 9},
            {"action": "query", "name": "out"},
            {"action": "query", "scope": "past", "name": "out"},
            {"action": "assert", "name": "out", "value": 999},
        ]
        for command in commands:
            kernel.execute(command)
        return kernel, commands

    def test_replay_reproduces_state_and_receipts(self):
        kernel, _ = self.make_trace()
        replayed = Kernel.replay(kernel.export_trace())
        self.assertEqual(replayed.state_digest(), kernel.state_digest())
        self.assertEqual(replayed.export_trace(), kernel.export_trace())

    def test_repeat_execution_has_same_non_timing_record(self):
        kernel, commands = self.make_trace()
        another = Kernel()
        for command in commands:
            another.execute(command)
        self.assertEqual(kernel.export_trace(), another.export_trace())

    def test_changed_command_refused(self):
        kernel, _ = self.make_trace()
        trace = kernel.export_trace()
        trace[1]["command"]["value"] = 21
        with self.assertRaises(ValueError):
            Kernel.replay(trace)

    def test_changed_answer_refused(self):
        kernel, _ = self.make_trace()
        trace = kernel.export_trace()
        trace[4]["result"]["value"] = "999"
        with self.assertRaises(ValueError):
            Kernel.replay(trace)

    def test_reordered_history_refused(self):
        kernel, _ = self.make_trace()
        trace = kernel.export_trace()
        trace[1], trace[2] = trace[2], trace[1]
        with self.assertRaises(ValueError):
            Kernel.replay(trace)

    def test_rehashed_fabricated_answer_fails_semantic_reexecution(self):
        kernel, _ = self.make_trace()
        trace = copy.deepcopy(kernel.export_trace())
        trace[4]["result"]["value"] = "999"
        previous = "0" * 64
        for receipt in trace:
            receipt["previous_hash"] = previous
            payload = {key: value for key, value in receipt.items() if key != "hash"}
            encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                 ensure_ascii=False, allow_nan=False).encode("utf-8")
            receipt["hash"] = hashlib.sha256(encoded).hexdigest()
            previous = receipt["hash"]
        with self.assertRaisesRegex(ValueError, "semantic replay"):
            Kernel.replay(trace)

    def test_missing_interior_receipt_refused(self):
        kernel, _ = self.make_trace()
        trace = kernel.export_trace()
        del trace[3]
        with self.assertRaises(ValueError):
            Kernel.replay(trace)

    def test_valid_prefix_is_not_misrepresented_as_authenticated_full_history(self):
        kernel, _ = self.make_trace()
        prefix = kernel.export_trace()[:5]
        replayed = Kernel.replay(prefix)
        self.assertEqual(len(replayed.export_trace()), 5)
        self.assertNotEqual(replayed.state_digest(), kernel.state_digest())

    def test_retained_head_rejects_truncation(self):
        kernel, _ = self.make_trace()
        trace = kernel.export_trace()
        self.assertEqual(Kernel.replay(trace, expected_head=trace[-1]["hash"]).head, trace[-1]["hash"])
        with self.assertRaisesRegex(ValueError, "anchor"):
            Kernel.replay(trace[:5], expected_head=trace[-1]["hash"])


class FiniteCapacityTests(unittest.TestCase):
    def assert_refused_without_change(self, kernel, command):
        before = kernel.state_digest()
        self.assertEqual(outcome(kernel.execute(command)), {"status": "error", "error": "resource_limit"})
        self.assertEqual(before, kernel.state_digest())

    def test_scope_quota_counts_empty_scopes(self):
        kernel = Kernel(max_scopes=2)
        self.assertEqual(kernel.execute({"action": "fork", "new_scope": "first"})["status"], "ok")
        self.assert_refused_without_change(kernel, {"action": "fork", "new_scope": "second"})
        self.assertEqual(kernel.graph()["scopes"], ["first", "main"])

    def test_revision_quota_blocks_repeated_overwrite(self):
        kernel = Kernel(max_revisions=2)
        kernel.execute({"action": "set", "name": "a", "value": 1})
        kernel.execute({"action": "set", "name": "a", "value": 2})
        self.assert_refused_without_change(kernel, {"action": "set", "name": "a", "value": 3})
        self.assertEqual(outcome(kernel.execute({"action": "query", "name": "a"})),
                         {"status": "ok", "value": "2", "unit": "1"})

    def test_revision_quota_rejects_fork_atomically(self):
        kernel = Kernel(max_revisions=3)
        kernel.execute({"action": "set", "name": "a", "value": 1})
        kernel.execute({"action": "set", "name": "b", "value": 2})
        self.assert_refused_without_change(kernel, {"action": "fork", "new_scope": "over"})
        self.assertEqual(kernel.graph()["scopes"], ["main"])

    def test_total_source_bytes_are_counted_as_utf8(self):
        kernel = Kernel(max_source_bytes=5)
        self.assertEqual(kernel.execute({"action": "source", "id": "a", "text": "abc"})["status"], "ok")
        self.assertEqual(kernel.execute({"action": "source", "id": "b", "text": "é"})["status"], "ok")
        self.assertEqual(kernel.execute({"action": "source", "id": "a", "text": "abc"})["status"], "ok")
        self.assert_refused_without_change(kernel, {"action": "source", "id": "c", "text": "x"})

    def test_command_size_limit_refuses_before_mutation(self):
        kernel = Kernel(max_command_bytes=128)
        self.assert_refused_without_change(kernel, {"action": "set", "name": "a", "value": 1,
                                                    "metadata": "x" * 129})
        self.assertEqual(Kernel.replay(kernel.export_trace()).state_digest(), kernel.state_digest())

    def test_receipt_capacity_stops_before_unrecorded_execution(self):
        kernel = Kernel(max_receipts=2)
        kernel.execute({"action": "set", "name": "a", "value": 1})
        kernel.execute({"action": "query", "name": "a"})
        before_trace = kernel.export_trace()
        self.assert_refused_without_change(kernel, {"action": "set", "name": "a", "value": 99})
        self.assertEqual(kernel.export_trace(), before_trace)
        self.assertEqual(Kernel.replay(before_trace).state_digest(), kernel.state_digest())


class IndependentArithmeticTests(unittest.TestCase):
    def test_multiple_failures_follow_left_to_right_role_order(self):
        for factory in (Kernel, Reference, CachedDAG):
            with self.subTest(factory=factory.__name__):
                engine = factory()
                engine.execute({"action": "derive", "name": "out", "op": "add",
                                "args": {"left": "missing", "right": "out"}})
                self.assertEqual(outcome(engine.execute({"action": "query", "name": "out"})),
                                 {"status": "error", "error": "missing_reference"})
                engine.execute({"action": "derive", "name": "out", "op": "add",
                                "args": {"left": "out", "right": "missing"}})
                self.assertEqual(outcome(engine.execute({"action": "query", "name": "out"})),
                                 {"status": "error", "error": "cycle"})

    def test_mixed_mutations_scopes_cycles_and_errors_against_reference(self):
        """Supplemental adversarial regression, not the preregistered task grid."""
        for seed in range(30):
            with self.subTest(seed=seed):
                rng = random.Random(seed + 20261125)
                engines = [Kernel(), Reference(), CachedDAG()]
                scopes = ["main"]
                for index in range(80):
                    scope = rng.choice(scopes)
                    name = f"n{rng.randrange(10)}"
                    action = rng.choice(["set", "set", "derive", "derive", "retract", "query", "assert", "fork"])
                    if action == "set":
                        command = {"action": action, "scope": scope, "name": name,
                                   "value": str(rng.randint(-5, 5)), "unit": rng.choice(["1", "m", "s"])}
                    elif action == "derive":
                        operator = rng.choice(["add", "sub", "mul", "div", "copy", "neg"])
                        if operator in ("copy", "neg"):
                            args = {"value": f"n{rng.randrange(10)}"}
                        else:
                            args = {"left": f"n{rng.randrange(10)}", "right": f"n{rng.randrange(10)}"}
                        command = {"action": action, "scope": scope, "name": name, "op": operator, "args": args}
                    elif action == "fork" and len(scopes) < 4:
                        new_scope = f"b{len(scopes)}"
                        command = {"action": action, "scope": scope, "new_scope": new_scope}
                        scopes.append(new_scope)
                    elif action == "assert":
                        command = {"action": action, "scope": scope, "name": name,
                                   "value": rng.randint(-5, 5), "unit": rng.choice(["1", "m", "s"])}
                    else:
                        command = {"action": "query" if action == "fork" else action, "scope": scope, "name": name}
                    results = [outcome(engine.execute(command)) for engine in engines]
                    self.assertEqual(results[0], results[1], (seed, index, command, results))
                    self.assertEqual(results[1], results[2], (seed, index, command, results))
                self.assertEqual(Kernel.replay(engines[0].export_trace()).state_digest(), engines[0].state_digest())

    def test_seeded_chains_match_hand_computed_fraction_recurrence(self):
        for seed in (11, 29, 47, 71, 101):
            with self.subTest(seed=seed):
                rng = random.Random(seed)
                kernel = Kernel()
                expected = Fraction(rng.randint(-20, 20), rng.randint(1, 9))
                kernel.execute({"action": "set", "name": "x0", "value": str(expected)})
                for i in range(1, 65):
                    rhs = Fraction(rng.randint(-10, 10), rng.randint(1, 9))
                    operation = "add" if rng.randrange(2) else "sub"
                    kernel.execute({"action": "set", "name": f"c{i}", "value": str(rhs)})
                    kernel.execute({"action": "derive", "name": f"x{i}", "op": operation,
                                    "args": {"left": f"x{i-1}", "right": f"c{i}"}})
                    expected = expected + rhs if operation == "add" else expected - rhs
                self.assertEqual(outcome(kernel.execute({"action": "query", "name": "x64"})),
                                 {"status": "ok", "value": str(expected), "unit": "1"})

    def test_mutation_stream_matches_independent_current_value_formula(self):
        kernel = Kernel()
        kernel.execute({"action": "set", "name": "left", "value": 0})
        kernel.execute({"action": "set", "name": "right", "value": 0})
        kernel.execute({"action": "derive", "name": "answer", "op": "sub",
                        "args": {"left": "left", "right": "right"}})
        rng = random.Random(37)
        values = {"left": 0, "right": 0}
        for _ in range(100):
            chosen = rng.choice(("left", "right"))
            values[chosen] = rng.randint(-1000, 1000)
            kernel.execute({"action": "set", "name": chosen, "value": values[chosen]})
            self.assertEqual(outcome(kernel.execute({"action": "query", "name": "answer"})),
                             {"status": "ok", "value": str(values["left"] - values["right"]), "unit": "1"})


if __name__ == "__main__":
    unittest.main()
