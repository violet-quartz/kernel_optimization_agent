"""Tests for write_triton_kernel / record_result.

The validation half matters most: every rule in `_validate_kernel_source`
mirrors a way `build_case` fails, and two of those failures are silent — a
`ModelNew = Model` alias and a computed module-level constant are both dropped
by the loader and only surface much later as `must define ModelNew` or a
`NameError` from inside the kernel.
"""

import json
import unittest

from anthropic.lib.tools import ToolError

from tests._helpers import TempDirCase, kernel_source


class RecordCase(TempDirCase):
    def setUp(self):
        super().setUp()
        from record import make_kernel_tools

        self.run = self.make_run()
        self.write_kernel, self.record = make_kernel_tools(self.run)

    def write_ok(self, **kwargs) -> dict:
        payload = {"code": kernel_source(), "strategy": "baseline port"}
        payload.update(kwargs)
        return json.loads(self.write_kernel.call(payload))

    def record_ok(self, **kwargs) -> dict:
        payload = {"version": "v001", "correct": True, "v0_ms": 2.0, "v1_ms": 1.0}
        payload.update(kwargs)
        return json.loads(self.record.call(payload))


class TestWriteKernel(RecordCase):
    def test_writes_the_kernel_and_its_attempt_record(self):
        out = self.write_ok()
        self.assertEqual(out["version"], "v001")
        self.assertEqual(out["v0_path"], str(self.run.v0_path))
        self.assertIn("import triton", self.run.kernel_path("v001").read_text(encoding="utf-8"))

        attempt = json.loads(
            (self.run.version_dir("v001") / "attempt.json").read_text(encoding="utf-8")
        )
        self.assertEqual(attempt["strategy"], "baseline port")
        self.assertIn("created_at", attempt)

    def test_parent_defaults_to_the_current_best(self):
        """Without a parent the model's next version should build on the best
        one so far, not on whatever it happened to write last."""
        self.write_ok()
        self.record_ok()  # v001 becomes best
        self.write_ok()  # v002
        self.record_ok(version="v002", v1_ms=5.0)  # slower, best stays v001
        self.write_ok()  # v003
        attempt = json.loads(
            (self.run.version_dir("v003") / "attempt.json").read_text(encoding="utf-8")
        )
        self.assertEqual(attempt["parent"], "v001")

    def test_warns_but_accepts_a_torch_only_change(self):
        out = self.write_ok(code=kernel_source().replace("import triton\n", ""))
        self.assertTrue(any("triton" in w for w in out["warnings"]))
        self.assertEqual(out["version"], "v001")


class TestValidation(RecordCase):
    def reject(self, code: str) -> str:
        with self.assertRaises(ToolError) as ctx:
            self.write_kernel.call({"code": code, "strategy": "s"})
        return str(ctx.exception)

    def test_rejects_empty_source(self):
        self.assertIn("empty", self.reject("   \n").lower())

    def test_rejects_a_syntax_error_with_a_line_number(self):
        message = self.reject("class ModelNew(:\n    pass\n")
        self.assertIn("line 1", message)

    def test_rejects_a_missing_model_new(self):
        self.assertIn("ModelNew", self.reject("def get_inputs():\n    return []\n"))

    def test_rejects_model_new_defined_by_assignment(self):
        """`ModelNew = Model` parses fine and reads fine, but the loader strips
        non-literal module-level assignments, so ModelNew never exists."""
        code = kernel_source().replace("class ModelNew(nn.Module):", "class _Impl(nn.Module):")
        code += "\n\nModelNew = _Impl\n"
        self.assertIn("assignment", self.reject(code))

    def test_rejects_a_model_new_without_forward(self):
        code = kernel_source().replace("    def forward(self, x):\n        return x", "    pass")
        self.assertIn("forward", self.reject(code))

    def test_rejects_missing_module_level_functions(self):
        code = kernel_source().replace("def get_inputs():", "def _unused():")
        self.assertIn("get_inputs", self.reject(code))

    def test_rejects_a_computed_module_level_constant_that_is_used(self):
        code = kernel_source(body="return x.to(DEV)")
        code = code.replace("import triton\n", "import triton\n\nDEV = torch.device('cuda')\n")
        message = self.reject(code)
        self.assertIn("DEV", message)
        self.assertIn("NameError", message)

    def test_allows_an_unused_computed_constant(self):
        """Only dropped names that something reads can break the file."""
        code = kernel_source().replace("import triton\n", "import triton\n\n_UNUSED = 1 + 1\n")
        self.assertEqual(json.loads(self.write_kernel.call({"code": code, "strategy": "s"}))["version"], "v001")

    def test_allows_literal_module_level_constants(self):
        code = kernel_source(body="return x * BLOCK")
        code = code.replace("import triton\n", "import triton\n\nBLOCK = 128\n")
        self.assertEqual(json.loads(self.write_kernel.call({"code": code, "strategy": "s"}))["version"], "v001")

    def test_a_rejected_kernel_consumes_no_version(self):
        """A validation failure must be free — otherwise the model burns its
        version budget on files that were never written."""
        self.reject("nonsense(")
        self.assertEqual(self.run.versions(), [])
        self.assertEqual(self.write_ok()["version"], "v001")


class TestRecordResult(RecordCase):
    def test_computes_the_speedup_itself(self):
        self.write_ok()
        out = self.record_ok(v0_ms=4.0, v1_ms=1.0)
        self.assertEqual(out["recorded"]["speedup"], 4.0)
        self.assertTrue(out["is_best"])
        self.assertEqual(out["best"], "v001")

    def test_appends_one_history_row_per_version(self):
        from record import load_history

        self.write_ok()
        self.record_ok()
        self.write_ok()
        self.record_ok(version="v002", correct=False, error="illegal memory access")

        rows = load_history(self.run)
        self.assertEqual([r["version"] for r in rows], ["v001", "v002"])
        self.assertFalse(rows[1]["correct"])
        self.assertIn("illegal memory access", rows[1]["error"])
        self.assertEqual(rows[1]["strategy"], "baseline port")

    def test_best_only_moves_on_a_correct_and_faster_version(self):
        for _ in range(3):
            self.write_ok()
        self.record_ok(version="v001", v1_ms=1.0)
        self.assertEqual(self.record_ok(version="v002", v1_ms=2.0)["best"], "v001")
        self.assertEqual(self.record_ok(version="v003", v1_ms=0.5)["best"], "v003")
        self.assertEqual(self.run.best_link.resolve(), self.run.version_dir("v003").resolve())

    def test_an_incorrect_version_never_becomes_best(self):
        self.write_ok()
        self.write_ok()
        self.record_ok(version="v001", v1_ms=1.0)
        out = self.record_ok(version="v002", correct=False, v1_ms=0.001)
        self.assertFalse(out["is_best"])
        self.assertEqual(out["best"], "v001")

    def test_records_a_failure_with_no_timings(self):
        self.write_ok()
        out = self.record_ok(correct=False, v0_ms=None, v1_ms=None, error="compile error")
        self.assertIsNone(out["recorded"]["speedup"])
        self.assertIsNone(out["best"])
        self.assertEqual(out["attempts"], 1)

    def test_rejects_an_unknown_version(self):
        with self.assertRaises(ToolError) as ctx:
            self.record.call({"version": "v042", "correct": True})
        self.assertIn("v042", str(ctx.exception))

    def test_leaderboard_is_ordered_by_speedup(self):
        for _ in range(3):
            self.write_ok()
        self.record_ok(version="v001", v1_ms=2.0)
        self.record_ok(version="v002", v1_ms=1.0)
        out = self.record_ok(version="v003", v1_ms=4.0)
        speedups = [row["speedup"] for row in out["leaderboard"]]
        self.assertEqual(speedups, sorted(speedups, reverse=True))


if __name__ == "__main__":
    unittest.main()
