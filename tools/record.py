"""Tools the agent uses to write a kernel and record how it did.

Both tools are bound to one `Run` (see `run_dir.py`) by `make_kernel_tools`,
so the model never supplies a path or a version number — it only supplies the
code and its reasoning. Version allocation, file layout and the `best` pointer
stay under the harness's control.

`write_triton_kernel` validates the generated source against the contract
`auto_bench_v2.build_case` imposes on a v1 file *before* writing it, because
two of that contract's failure modes are silent: a `ModelNew = Model` alias and
a non-literal module-level assignment are both dropped by `_filter_module_ast`,
surfacing much later as `must define ModelNew` or a `NameError` from inside the
kernel. Catching them here costs one AST walk instead of one benchmark round.
"""

import ast
import json
from datetime import datetime
from pathlib import Path

from anthropic import beta_tool
from anthropic.lib.tools import ToolError

try:  # importable both as `tools.record` and as a top-level module
    from .run_dir import Run
except ImportError:
    from run_dir import Run

# What build_case looks up in the v1 file. `get_init_inputs` / `get_inputs` are
# required even though v1's inputs are discarded in favour of a clone of v0's:
# build_case still calls both, and compares the argument counts.
REQUIRED_V1_FUNCS = ("get_init_inputs", "get_inputs")

_MAX_ERROR_CHARS = 2000
_MAX_HISTORY_ERROR_CHARS = 400


# --------------------------------------------------------------------------
# source validation
# --------------------------------------------------------------------------


def _is_safe_literal(node) -> bool:
    """Mirror of auto_bench_v2._is_safe_literal — kept in sync deliberately.

    Importing it would drag torch into this module for one predicate, and this
    module has to stay importable on a machine with no accelerator.
    """
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return all(_is_safe_literal(elt) for elt in node.elts)
    if isinstance(node, ast.Dict):
        return all(
            (key is None or _is_safe_literal(key)) and _is_safe_literal(value)
            for key, value in zip(node.keys, node.values)
        )
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return _is_safe_literal(node.operand)
    return False


def _loaded_names(tree: ast.Module) -> set[str]:
    return {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }


def _validate_kernel_source(code: str) -> list[str]:
    """Check `code` against the v1 contract. Returns non-fatal warnings.

    Raises:
        ToolError: The file would fail (or silently misbehave) in build_case.
            The message is written for the model — it says what to change.
    """
    if not code.strip():
        raise ToolError("The kernel source is empty.")

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise ToolError(
            f"Syntax error on line {exc.lineno}: {exc.msg}. Nothing was written; "
            "fix the code and call write_triton_kernel again."
        ) from exc

    classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    functions = {
        n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    if "ModelNew" not in classes:
        aliased = any(
            isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "ModelNew" for t in n.targets)
            for n in tree.body
        )
        if aliased:
            raise ToolError(
                "`ModelNew` is defined by assignment (e.g. `ModelNew = Model`). The "
                "benchmark harness strips module-level assignments whose value is not "
                "a literal, so `ModelNew` would not exist at load time. Define it as a "
                "real class: `class ModelNew(nn.Module):`."
            )
        raise ToolError(
            "The file must define `class ModelNew` at module level — that is the class "
            "the harness benchmarks against `Model`."
        )

    methods = {
        n.name
        for n in classes["ModelNew"].body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if "forward" not in methods:
        raise ToolError("`ModelNew` must define a `forward` method.")

    missing = [name for name in REQUIRED_V1_FUNCS if name not in functions]
    if missing:
        raise ToolError(
            f"The file must also define {', '.join(missing)} at module level. The "
            "harness calls them to build ModelNew and to size the inputs, and compares "
            "the argument count against v0 — copy them from the v0 file unchanged "
            "unless the kernel genuinely needs a different signature."
        )

    # Module-level assignments that build_case would silently drop. Only the
    # ones actually used elsewhere matter — an unused constant is harmless.
    dropped = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and not _is_safe_literal(node.value):
            dropped.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and (node.value is None or not _is_safe_literal(node.value))
        ):
            dropped.add(node.target.id)

    used = sorted(dropped & _loaded_names(tree))
    if used:
        raise ToolError(
            f"Module-level assignment(s) {', '.join(used)} are computed rather than "
            "plain literals, so the harness drops them before loading the file and the "
            "code that reads them raises NameError. Move them inside a function or "
            "`ModelNew.__init__`, or make them literal constants."
        )

    warnings = []
    if "triton" not in {
        alias.name.split(".")[0]
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module.split(".")[0]
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module
    }:
        warnings.append("no `triton` import found — this version is a torch-level change")
    return warnings


# --------------------------------------------------------------------------
# history
# --------------------------------------------------------------------------


def load_history(run: Run) -> list[dict]:
    """Every recorded attempt, oldest first. Skips malformed lines."""
    if not run.history_path.is_file():
        return []
    rows = []
    for line in run.history_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _append_history(run: Run, row: dict) -> None:
    # One short line, opened in append mode: concurrent writers cannot
    # interleave a single small write on POSIX.
    with run.history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _leaderboard(run: Run, limit: int = 3) -> list[dict]:
    correct = [r for r in load_history(run) if r.get("correct") and r.get("v1_ms")]
    correct.sort(key=lambda r: r["v1_ms"])
    return [
        {"version": r["version"], "v1_ms": r["v1_ms"], "speedup": r.get("speedup")}
        for r in correct[:limit]
    ]


def _as_tool_text(payload: dict) -> str:
    """Render a tool result as text.

    The tool runner passes a tool's return value straight into
    ``{"type": "tool_result", "content": <value>}`` without serialising it
    (`anthropic/lib/tools/_beta_runner.py`), and that field only accepts a
    string or a list of content blocks. Returning a dict makes the API reject
    the whole request with `expected a string or a list`, so encode here.
    """
    return json.dumps(payload, indent=2, ensure_ascii=False, default=str)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


# --------------------------------------------------------------------------
# tools
# --------------------------------------------------------------------------


def make_kernel_tools(run: Run):
    """Build the two run-bound tools. Returns (write_triton_kernel, record_result)."""

    @beta_tool
    def write_triton_kernel(code: str, strategy: str, parent: str | None = None) -> str:
        """Save a new version of the optimized kernel and return its path.

        The version number and the file location are assigned by the harness —
        you cannot overwrite an earlier attempt. Validation runs before
        anything is written, so a rejected kernel does not consume a version.

        The file must define, at module level: `class ModelNew` (with a
        `forward` method), `get_init_inputs`, and `get_inputs`. `ModelNew` must
        accept `*get_init_inputs()` and keep the same `state_dict` keys and
        shapes as `Model`, or the correctness check cannot copy v0's weights
        into it. Module-level constants must be plain literals.

        Args:
            code: Complete Python source of the optimized version. It is loaded
                on its own, so it must contain every import it needs.
            strategy: One or two sentences on what this version changes and why
                you expect it to be faster. Recorded in the run history and
                shown back to you on later turns, so future attempts do not
                repeat a approach that already failed.
            parent: Version this one is derived from, e.g. "v002". Defaults to
                the fastest correct version so far, or the previous version if
                none has passed yet.

        Returns:
            A dict with `version`, `kernel_path`, `v0_path` and any non-fatal
            `warnings`. Pass `kernel_path` as `v1_file` and `v0_path` as
            `v0_file` to check_correctness and bench_mark, then report the
            outcome with record_result.

        Raises:
            ToolError: The source does not satisfy the contract above. Nothing
                is written; the message says what to fix.
        """
        warnings = _validate_kernel_source(code)

        if parent is None:
            existing = run.versions()
            best = run.read_meta().get("best")
            parent = best or (existing[-1] if existing else None)
        elif parent not in run.versions():
            raise ToolError(f"No such parent version: {parent!r}.")

        version, vdir = run.new_version()
        (vdir / "kernel.py").write_text(code, encoding="utf-8")
        # Written now rather than at record time, so a version whose benchmark
        # crashes still leaves its intent behind in the run directory.
        (vdir / "attempt.json").write_text(
            json.dumps(
                {
                    "version": version,
                    "parent": parent,
                    "strategy": strategy,
                    "created_at": _now(),
                    "warnings": warnings,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        return _as_tool_text(
            {
                "version": version,
                "kernel_path": str(run.kernel_path(version)),
                "v0_path": str(run.v0_path),
                "warnings": warnings,
            }
        )

    @beta_tool
    def record_result(
        version: str,
        correct: bool,
        v0_ms: float | None = None,
        v1_ms: float | None = None,
        error: str | None = None,
        notes: str | None = None,
    ) -> str:
        """Record how a version performed, and update the best-so-far pointer.

        Call this once per version, after check_correctness and (if it passed)
        bench_mark. The speedup is computed here from the two timings — do not
        compute it yourself.

        Args:
            version: The version being reported, e.g. "v003".
            correct: Whether check_correctness passed.
            v0_ms: Median milliseconds for the original — bench_mark's
                `v0_median_ms`. Omit when the kernel was not benchmarked.
            v1_ms: Median milliseconds for this version — bench_mark's
                `v1_median_ms`. Omit when the kernel was not benchmarked.
            error: The failure message, when `correct` is false or the
                benchmark raised. Paste the harness's message verbatim.
            notes: Anything worth knowing on a later attempt — which shape hurt,
                what the profiler showed, what to try next.

        Returns:
            A dict with the recorded row, whether this version is the new best,
            and the current top-3 leaderboard by measured time.

        Raises:
            ToolError: Unknown version, or a timing that is not a positive
                number.
        """
        if version not in run.versions():
            raise ToolError(
                f"No such version: {version!r}. Known versions: "
                f"{', '.join(run.versions()) or '(none)'}."
            )
        for label, value in (("v0_ms", v0_ms), ("v1_ms", v1_ms)):
            if value is not None and (value != value or value <= 0):
                raise ToolError(f"{label} must be a positive number, got {value!r}.")

        vdir = run.version_dir(version)
        attempt = {}
        attempt_path = vdir / "attempt.json"
        if attempt_path.is_file():
            attempt = json.loads(attempt_path.read_text(encoding="utf-8"))

        speedup = round(v0_ms / v1_ms, 4) if (v0_ms and v1_ms) else None
        row = {
            "version": version,
            "parent": attempt.get("parent"),
            "strategy": attempt.get("strategy"),
            "correct": bool(correct),
            "v0_ms": v0_ms,
            "v1_ms": v1_ms,
            "speedup": speedup,
            "error": (error or "")[:_MAX_ERROR_CHARS] or None,
            "notes": notes,
            "recorded_at": _now(),
        }
        (vdir / "result.json").write_text(
            json.dumps(row, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

        # A correct-but-unbenchmarked version cannot win: `best` means fastest
        # measured, and an unknown time must not displace a known one.
        previous_best = run.read_meta().get("best")
        is_best = False
        if correct and v1_ms:
            best_ms = None
            if previous_best:
                for candidate in load_history(run):
                    if candidate.get("version") == previous_best:
                        best_ms = candidate.get("v1_ms")
            if best_ms is None or v1_ms < best_ms:
                run.set_best(version)
                run.update_meta(best=version)
                is_best = True

        history_row = dict(row)
        if history_row["error"]:
            history_row["error"] = history_row["error"][:_MAX_HISTORY_ERROR_CHARS]
        _append_history(run, history_row)

        return _as_tool_text(
            {
                "recorded": row,
                "is_best": is_best,
                "best": run.read_meta().get("best"),
                "attempts": len(load_history(run)),
                "leaderboard": _leaderboard(run),
            }
        )

    return write_triton_kernel, record_result
