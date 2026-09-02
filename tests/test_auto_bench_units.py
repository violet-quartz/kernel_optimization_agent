"""Tests for the CPU-reachable parts of the benchmark harness.

`check_correctness` and `bench_mark` need an accelerator by design — see
`_detect_target_device`. What is testable without one is everything that
decides *whether a v1 file can be loaded at all*, which is where the agent's
generated code actually goes wrong, plus the failure modes of the two tools.
"""

import ast
import unittest
from pathlib import Path

from tests._helpers import TempDirCase, V0_SOURCE, V1_SOURCE


class TestLiteralFilter(TempDirCase):
    """`_filter_module_ast` runs on every loaded file and silently drops
    module-level statements it does not keep. `record._validate_kernel_source`
    warns the model about exactly this set, so the two must agree."""

    def kept(self, source: str) -> set:
        from bench import _filter_module_ast

        tree = _filter_module_ast(ast.parse(source))
        names = set()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, ast.Assign):
                names.update(t.id for t in node.targets if isinstance(t, ast.Name))
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names.add(node.target.id)
        return names

    def test_keeps_definitions_and_literal_constants(self):
        kept = self.kept(
            "import torch\n"
            "BLOCK = 128\n"
            "SHAPES = [(1, 2), (3, 4)]\n"
            "NAME: str = 'x'\n"
            "def f(): pass\n"
            "class C: pass\n"
        )
        self.assertEqual(kept, {"BLOCK", "SHAPES", "NAME", "f", "C"})

    def test_drops_computed_constants(self):
        kept = self.kept(
            "import torch\n"
            "DEV = torch.device('cuda')\n"
            "ALIAS = SomeClass\n"
            "SIZE = 2 * 64\n"
            "LAZY: int\n"
        )
        self.assertEqual(kept, set())

    def test_agrees_with_the_validator_in_record(self):
        import bench
        import record

        cases = [
            "1", "'x'", "None", "True", "(1, 2)", "[1, 2]", "{'a': 1}", "-1",
            "torch.device('cuda')", "SomeName", "2 * 64", "f(1)", "[x for x in y]",
        ]
        for case in cases:
            node = ast.parse(case).body[0].value
            with self.subTest(expr=case):
                self.assertEqual(
                    bench._is_safe_literal(node),
                    record._is_safe_literal(node),
                    f"the two copies of _is_safe_literal disagree on `{case}`",
                )


class TestLoadKsModule(TempDirCase):
    def test_loads_a_file_without_importing_it(self):
        from bench import load_ks_module

        module = load_ks_module(self.write("v0.py", V0_SOURCE))
        self.assertTrue(hasattr(module, "Model"))
        self.assertEqual(module.get_init_inputs(), [2.0])

    def test_module_level_side_effects_are_dropped(self):
        """A file that would crash on import at module level still loads,
        because only defs, imports and literal assignments survive."""
        from bench import load_ks_module

        module = load_ks_module(
            self.write("boom.py", V0_SOURCE + "\nraise RuntimeError('never runs')\n")
        )
        self.assertTrue(hasattr(module, "Model"))

    def test_rejects_a_missing_or_non_python_file(self):
        from bench import KsCompareError, load_ks_module

        with self.assertRaises(KsCompareError):
            load_ks_module(self.tmp / "missing.py")
        with self.assertRaises(KsCompareError):
            load_ks_module(self.write("v0.txt", V0_SOURCE))


class TestBuildCase(TempDirCase):
    """build_case runs entirely on CPU — it loads both files, builds both
    models and generates inputs. Everything after it needs a device."""

    def test_builds_both_models_from_the_same_seed(self):
        from bench import build_case

        case = build_case(
            self.write("v0.py", V0_SOURCE), self.write("v1.py", V1_SOURCE), seed=42
        )
        self.assertIsNotNone(case)

    def test_reports_a_v1_missing_model_new(self):
        from bench import KsCompareError, build_case

        with self.assertRaises(KsCompareError) as ctx:
            build_case(
                self.write("v0.py", V0_SOURCE), self.write("v1.py", V0_SOURCE), seed=42
            )
        self.assertIn("ModelNew", str(ctx.exception))

    def test_reports_a_model_new_that_cannot_be_constructed(self):
        from bench import KsCompareError, build_case

        broken = V1_SOURCE.replace("def __init__(self, scale):", "def __init__(self):")
        with self.assertRaises(KsCompareError):
            build_case(
                self.write("v0.py", V0_SOURCE), self.write("v1.py", broken), seed=42
            )


class TestPathChecking(TempDirCase):
    def test_returns_resolved_paths(self):
        from bench import check_input_file_path

        v0 = self.write("v0.py", V0_SOURCE)
        v1 = self.write("nested/../v1.py", V1_SOURCE)
        a, b = check_input_file_path(str(v0), str(v1))
        self.assertTrue(a.is_absolute() and b.is_absolute())
        self.assertNotIn("..", str(b))

    def test_failures_are_tool_errors_not_system_exit(self):
        """`SystemExit` is a BaseException: the tool runner would not catch it
        and the agent loop would die instead of retrying."""
        from anthropic.lib.tools import ToolError
        from bench import check_input_file_path

        v0 = self.write("v0.py", V0_SOURCE)
        cases = {
            "missing": (str(self.tmp / "nope.py"), str(v0)),
            "not a .py": (str(v0), str(self.write("v1.txt", V1_SOURCE))),
            "a directory": (str(v0), str(self.tmp)),
        }
        for label, pair in cases.items():
            with self.subTest(case=label):
                with self.assertRaises(ToolError):
                    check_input_file_path(*pair)


class TestAcceleratorRequirement(TempDirCase):
    def test_a_cpu_only_box_gets_a_clear_message(self):
        """On a machine with no accelerator both tools must fail loudly rather
        than quietly timing CPU tensors and reporting a meaningless speedup."""
        from bench import _auto_accel_name, bench_mark, check_correctness

        if _auto_accel_name() is not None:
            self.skipTest("this box has an accelerator; the GPU path is exercised for real")

        v0 = str(self.write("v0.py", V0_SOURCE))
        v1 = str(self.write("v1.py", V1_SOURCE))
        for tool in (check_correctness, bench_mark):
            with self.subTest(tool=tool.name):
                with self.assertRaises(Exception) as ctx:
                    tool.call({"v0_file": v0, "v1_file": v1})
                self.assertIn("accelerator", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
