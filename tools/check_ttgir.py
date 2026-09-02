"""What the Triton compiler actually produced for a kernel, read off its TTGIR.

`check_resource_usage` gives the cost of the generated code; this gives its
shape. TTGIR is the last IR before codegen, so it answers the questions that
decide the next optimization and that no timing number can:

  - how many layout conversions survived (every one is a round trip through
    shared memory that the source never asked for),
  - how many reductions there are,
  - whether the loads were pipelined into async copies,
  - which layouts the compiler chose,
  - whether the dots became tensor-core ops at all.

The compile runs in a subprocess: `TRITON_ALWAYS_COMPILE` and
`TRITON_KERNEL_DUMP` have to be set before Triton reads them, and they must not
leak into the benchmark's own process, where forcing a recompile on every
launch would distort every later timing.
"""

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from anthropic import beta_tool
from anthropic.lib.tools import ToolError

try:  # importable both as `tools.check_ttgir` and as a top-level module
    from .bench import (
        _detect_target_device,
        _move_to_device,
        as_args,
        load_ks_module,
        require_attr,
        run_forward,
        set_seed,
        step,
    )
except ImportError:
    from bench import (
        _detect_target_device,
        _move_to_device,
        as_args,
        load_ks_module,
        require_attr,
        run_forward,
        set_seed,
        step,
    )

_REPO_ROOT = Path(__file__).resolve().parent.parent

# One TTGIR file per compiled kernel. With TRITON_ALWAYS_COMPILE=1 an autotuned
# kernel dumps one per candidate config, so the list needs a ceiling.
_MAX_FILES = 20

_COMPILE_TIMEOUT_S = 600

# A layout conversion is a real shared-memory round trip; `local_alloc` is the
# buffer it goes through. Counted together, the way you would grep for them.
_LAYOUT_CONVERSION = re.compile(r"convert_layout|local_alloc")
_REDUCE = re.compile(r"tt\.reduce")
_ASYNC_COPY = re.compile(r"async_copy")
_DOT = re.compile(r"tt\.dot")
_LAYOUT_DECL = re.compile(r"^#(blocked|mma|shared)")


def _build_and_run(v1_file: str, seed: int) -> None:
    """Compile and run `ModelNew` once. Runs in the dump subprocess only.

    Loaded through the benchmark's own loader, so the IR belongs to exactly the
    program that gets timed.
    """
    v1_path = Path(v1_file)
    module = load_ks_module(v1_path)
    model_cls = require_attr(module, "ModelNew", v1_path)
    get_init_inputs = require_attr(module, "get_init_inputs", v1_path)
    get_inputs = require_attr(module, "get_inputs", v1_path)

    set_seed(seed)
    with step(f"{v1_path}: get_init_inputs()"):
        init_args = as_args(get_init_inputs(), f"{v1_path}: get_init_inputs()")
    with step(f"{v1_path}: ModelNew(...)"):
        model = model_cls(*init_args)
    set_seed(seed)
    with step(f"{v1_path}: get_inputs()"):
        inputs = as_args(get_inputs(), f"{v1_path}: get_inputs()")

    # The same model twice: _detect_target_device takes a (v0, v1) pair, and
    # here there is only one side to detect from.
    device = _detect_target_device(model, model, inputs, inputs)
    if hasattr(model, "to"):
        model = model.to(device)
    inputs = _move_to_device(inputs, device)
    run_forward(model, inputs, seed, f"{v1_path}: ModelNew")


def _dump_ttgir(v1_path: Path, seed: int) -> Path:
    """Run one forward with kernel dumping on, and return the dump directory."""
    dump_dir = Path(tempfile.mkdtemp(prefix="ttgir_"))
    env = dict(os.environ)
    env.update(
        TRITON_ALWAYS_COMPILE="1",  # otherwise a cached kernel dumps nothing
        TRITON_KERNEL_DUMP="1",
        TRITON_DUMP_DIR=str(dump_dir),
    )
    script = (
        f"import sys; sys.path.insert(0, {str(_REPO_ROOT)!r})\n"
        "from tools.check_ttgir import _build_and_run\n"
        f"_build_and_run({str(v1_path)!r}, {seed!r})\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=_COMPILE_TIMEOUT_S,
    )
    if result.returncode != 0:
        # The child's traceback is the only diagnosis there is — pass it on
        # rather than the return code.
        raise ToolError(
            f"compiling {v1_path} failed (exit {result.returncode}):\n"
            f"{(result.stderr or result.stdout or '').strip()[-2000:]}"
        )
    return dump_dir


def _summarize(path: Path) -> dict:
    """Count the constructs that matter in one .ttgir file."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return {
        "kernel": path.stem,
        # Counts are of matching lines, not occurrences — one line is one op here.
        "layout_conversions": sum(1 for l in lines if _LAYOUT_CONVERSION.search(l)),
        "reductions": sum(1 for l in lines if _REDUCE.search(l)),
        "async_copies": sum(1 for l in lines if _ASYNC_COPY.search(l)),
        "dots": sum(1 for l in lines if _DOT.search(l)),
        "layouts": [l.strip() for l in lines if _LAYOUT_DECL.match(l)],
        "ttgir_path": str(path),
    }


@beta_tool
def check_ttgir(v1_file: Path, seed: int = 42) -> str:
    """Report what the Triton compiler generated for a kernel, from its TTGIR.

    Compiles the kernel with dumping enabled, in a separate process, and reads
    the resulting TTGIR — the last IR before codegen. Use this when a kernel is
    correct but slower than the source suggests it should be: it shows the
    decisions the compiler made that the Python source does not.

    Args:
        v1_file: Path of the optimized kernel .py file — the `kernel_path`
            returned by `write_triton_kernel`. Must define `ModelNew`,
            `get_init_inputs` and `get_inputs`.
        seed: RNG seed for building the model and its inputs. Defaults to 42.

    Returns:
        A JSON list, one entry per compiled kernel, with:
          - `layout_conversions`: `convert_layout` / `local_alloc` lines. Each
            one is a round trip through shared memory that your source never
            asked for, inserted because two ops wanted the data in different
            layouts. A high count on a simple kernel is the single most common
            reason a hand-written Triton kernel underperforms — usually fixed by
            making the loads and the dot agree on one layout.
          - `reductions`: `tt.reduce` lines. Each is a cross-lane shuffle plus a
            barrier; several in one kernel is often one fused pass in disguise.
          - `async_copies`: `async_copy` lines. Zero inside a loop means the
            loads were not pipelined, so every iteration waits on global memory
            — raising `num_stages` is the usual answer.
          - `dots`: `tt.dot` lines. Zero in a matmul-shaped kernel means it
            never reached the tensor cores.
          - `layouts`: the `#blocked` / `#mma` / `#shared` declarations, i.e.
            what the compiler actually chose.
          - `ttgir_path`: the dumped file, readable with `read_file` when the
            counts alone are not enough.

    Raises:
        ToolError: The path is missing or is not a .py file, the kernel failed
            to compile, or no TTGIR was produced (which means the version
            launches no Triton kernel — a torch-level change).
    """
    v1_path = Path(v1_file).resolve()
    if not v1_path.is_file():
        raise ToolError(f"v1_file is not a file: {v1_path}")
    if v1_path.suffix != ".py":
        raise ToolError(f"v1_file must be a .py file: {v1_path}")

    dump_dir = _dump_ttgir(v1_path, seed)
    # The dump directory is fresh, so everything under it belongs to this run —
    # no need to pick the newest subdirectory the way a shell one-liner does.
    files = sorted(dump_dir.rglob("*.ttgir"))
    if not files:
        raise ToolError(
            f"no .ttgir was dumped for {v1_path}: the forward pass launched no "
            "Triton kernel, so this version is a pure torch-level change."
        )

    rows = [_summarize(path) for path in files[:_MAX_FILES]]
    payload = {"kernels": rows}
    if len(files) > _MAX_FILES:
        payload["note"] = (
            f"{len(files)} kernels were dumped; only the first {_MAX_FILES} are "
            f"reported. All of them are under {dump_dir}."
        )
    # The tool runner puts this value straight into the request body, where
    # tool_result content must be a string — a dict there is rejected.
    return json.dumps(payload, indent=2, ensure_ascii=False)
