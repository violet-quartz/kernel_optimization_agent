"""Contract tests for every @beta_tool.

These exist because both production failures so far were contract bugs, not
logic bugs: a tool returned a dict (`tool_result.content` must be text) and a
tool annotated a path as `str` then called `.resolve()` on it. Neither shows up
until the agent is several turns into a run on a GPU box. All of it is
checkable locally in milliseconds.
"""

import inspect
import json
import unittest
from pathlib import Path

from pydantic import ValidationError

from tests._helpers import TempDirCase, V0_SOURCE, V1_SOURCE, kernel_source, write_outcome


def all_tools(run=None):
    """Every tool the agent is given, as (name, tool) pairs."""
    import simple_agent
    from bench import bench_mark, check_correctness

    tools = [
        ("read_file", simple_agent.read_file),
        ("check_correctness", check_correctness),
        ("bench_mark", bench_mark),
    ]
    if run is not None:
        from record import make_kernel_tools

        write_triton_kernel, record_result = make_kernel_tools(run)
        tools += [
            ("write_triton_kernel", write_triton_kernel),
            ("record_result", record_result),
        ]
    return tools


class TestSchemas(TempDirCase):
    def tools(self):
        return all_tools(self.make_run())

    def test_every_parameter_has_a_concrete_type(self):
        """A property with no `type`/`anyOf` means an annotation pydantic could
        not render — the model then guesses at the argument shape."""
        for name, tool in self.tools():
            schema = tool.input_schema
            for prop, spec in schema.get("properties", {}).items():
                with self.subTest(tool=name, param=prop):
                    self.assertTrue(
                        {"type", "anyOf", "enum", "$ref"} & set(spec),
                        f"{name}.{prop} has no type in its schema: {spec}",
                    )

    def test_every_tool_and_parameter_is_documented(self):
        for name, tool in self.tools():
            with self.subTest(tool=name):
                self.assertTrue((tool.description or "").strip(), f"{name} has no description")
            for prop, spec in tool.input_schema.get("properties", {}).items():
                with self.subTest(tool=name, param=prop):
                    self.assertTrue(
                        (spec.get("description") or "").strip(),
                        f"{name}.{prop} is undocumented — the model sees only its name",
                    )

    def test_none_defaults_are_annotated_optional(self):
        """`atol: float = None` builds a schema whose type is plain `number`,
        so the model has no way to express "leave it at the default" other than
        sending `null` — which pydantic then rejects. A parameter defaulting to
        None must be annotated `X | None`. Parameters with a real default
        (`seed: int = 42`) are fine: the model just omits them."""
        for name, tool in self.tools():
            signature = inspect.signature(tool.func)
            spec_by_name = tool.input_schema.get("properties", {})
            for param in signature.parameters.values():
                if param.default is not None or param.default is inspect.Parameter.empty:
                    continue
                spec = spec_by_name.get(param.name, {})
                types = {spec.get("type")} | {
                    entry.get("type") for entry in spec.get("anyOf", [])
                }
                with self.subTest(tool=name, param=param.name):
                    self.assertIn(
                        "null",
                        types,
                        f"{name}.{param.name} defaults to None but its schema is "
                        f"{types - {None}} — sending null raises a validation error",
                    )
                    # Only argument *validation* matters here; the tool is free
                    # to reject the placeholder values with a ToolError.
                    try:
                        tool.call({**self._required_args(tool), param.name: None})
                    except ValidationError as exc:
                        self.fail(f"{name} rejects {param.name}=null: {exc}")
                    except Exception:
                        pass

    def _required_args(self, tool) -> dict:
        """Placeholder values for a tool's required parameters, so a single
        optional parameter can be exercised in isolation."""
        filler = {"string": "x", "integer": 1, "number": 1.0, "boolean": True}
        args = {}
        for param in tool.input_schema.get("required", []):
            spec = tool.input_schema["properties"][param]
            args[param] = filler.get(spec.get("type"), "x")
        return args


class TestReturnValues(TempDirCase):
    def test_results_are_text(self):
        """The runner puts a tool's return value straight into
        `{"type": "tool_result", "content": ...}`, which only accepts a string
        or a list of blocks. A dict there fails the whole request with a 400."""
        run = self.make_run()
        tools = dict(all_tools(run))
        calls = [
            ("read_file", {"file_path": str(run.v0_path)}),
            (
                "write_triton_kernel",
                {"code": kernel_source(), "strategy": "identity passthrough"},
            ),
            ("record_result", {"version": "v001"}),
        ]
        for name, payload in calls:
            if name == "record_result":
                # It reports what check_correctness and bench_mark measured, so
                # it needs the file they leave beside the kernel to exist.
                write_outcome(run, payload["version"])
            with self.subTest(tool=name):
                result = tools[name].call(payload)
                self.assertIsInstance(
                    result, str, f"{name} returned {type(result).__name__}, not text"
                )

    def test_json_results_parse(self):
        run = self.make_run()
        tools = dict(all_tools(run))
        out = json.loads(
            tools["write_triton_kernel"].call({"code": kernel_source(), "strategy": "s"})
        )
        self.assertEqual(out["version"], "v001")
        write_outcome(run, "v001", v0_ms=2.0, v1_ms=1.0)
        out = json.loads(tools["record_result"].call({"version": "v001"}))
        self.assertEqual(out["recorded"]["speedup"], 2.0)


class TestFailureModes(TempDirCase):
    def test_bad_input_raises_a_catchable_exception(self):
        """The runner catches `Exception` and hands the text back to the model.
        A `SystemExit` is a BaseException — it escapes and kills the agent."""
        run = self.make_run()
        tools = dict(all_tools(run))
        missing = str(self.tmp / "nope.py")
        calls = [
            ("read_file", {"file_path": missing}),
            ("check_correctness", {"v0_file": missing, "v1_file": missing}),
            ("bench_mark", {"v0_file": missing, "v1_file": missing}),
            ("write_triton_kernel", {"code": "def broken(:", "strategy": "s"}),
            ("record_result", {"version": "v999"}),
        ]
        for name, payload in calls:
            with self.subTest(tool=name):
                try:
                    tools[name].call(payload)
                except Exception as exc:  # noqa: BLE001 — that is the assertion
                    self.assertTrue(str(exc).strip(), f"{name} raised an empty message")
                except BaseException as exc:  # pragma: no cover
                    self.fail(
                        f"{name} raised {type(exc).__name__}, which the tool runner does "
                        f"not catch — the agent loop would die here"
                    )

    def test_path_parameters_accept_plain_strings(self):
        """The model always sends JSON strings. A `Path` annotation makes
        pydantic coerce them; a `str` annotation does not, and any later
        `.resolve()` raises AttributeError mid-run."""
        from bench import check_input_file_path

        v0 = self.write("v0.py", V0_SOURCE)
        v1 = self.write("v1.py", V1_SOURCE)
        for pair in ((str(v0), str(v1)), (v0, v1)):
            with self.subTest(kind=type(pair[0]).__name__):
                a, b = check_input_file_path(*pair)
                self.assertIsInstance(a, Path)
                self.assertIsInstance(b, Path)


if __name__ == "__main__":
    unittest.main()
