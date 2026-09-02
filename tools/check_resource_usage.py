"""Register and shared-memory usage of the Triton kernels in a v1 file.

`bench_mark` says a version is slow; this says whether it is slow because the
kernel ran out of registers. The three numbers come straight from the Triton
compiler, so no timing and no profiler is involved.

The probe never needs the kernel's signature: it runs one forward pass of
`ModelNew` with `JITFunction.run` patched, and reads the counters off the
`CompiledKernel` that each real launch produces.
"""

import json
from contextlib import contextmanager
from pathlib import Path

from anthropic import beta_tool
from anthropic.lib.tools import ToolError

try:  # importable both as `tools.check_resource_usage` and as a top-level module
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


@contextmanager
def _watch_launches(compiled: dict):
    """Record the CompiledKernel of every Triton launch made inside the block.

    Keyed by the compiled kernel's identity, not by name: one `@triton.jit`
    function called with different shapes or constexprs compiles a separate
    kernel per specialization, each with its own register cost, and keying by
    name would keep only the last one. Repeated launches of the same
    specialization hit Triton's cache and return the same object, so they
    collapse on their own. The launches themselves still happen for real — the
    hook only observes.
    """
    try:
        from triton.runtime.jit import JITFunction
    except ImportError as exc:
        raise ToolError("triton is not installed here, so kernel resource "
                        "usage cannot be read.") from exc

    original_run = JITFunction.run

    def run(self, *args, **kwargs):
        kernel = original_run(self, *args, **kwargs)
        if getattr(kernel, "metadata", None) is not None:
            compiled[id(kernel)] = (getattr(self, "__name__", repr(self)), kernel)
        return kernel

    JITFunction.run = run
    try:
        yield
    finally:
        JITFunction.run = original_run


@beta_tool
def check_resource_usage(v1_file: Path, seed: int = 42) -> str:
    """Report register and shared-memory usage of each Triton kernel in a file.

    Runs one forward pass of `ModelNew` and reads the compiler's own counters
    off every kernel it launches. Call this when a version is correct but
    slower than expected: `n_spills > 0` means the kernel needs more registers
    than the hardware has and the overflow goes to local memory, which no
    timing number tells you.

    Args:
        v1_file: Path of the optimized kernel .py file — the `kernel_path`
            returned by `write_triton_kernel`. Must define `ModelNew`,
            `get_init_inputs` and `get_inputs`.
        seed: RNG seed for building the model and its inputs. Defaults to 42.

    Returns:
        A JSON list with `kernel`, `n_regs` (registers per thread), `n_spills`
        (spilled registers per thread) and `smem` (shared memory per block, in
        bytes). One entry per compiled specialization, so the same `kernel`
        name appears more than once when it is launched with different shapes
        or constexprs — those are separately compiled and can differ widely in
        register cost. The list is empty when the version launches no Triton
        kernel — that is a torch-level change, not an error.

    Raises:
        ToolError: The path is missing, is not a .py file, or triton is absent.
        KsCompareError: The file failed to load, build, or run.
    """
    v1_path = Path(v1_file).resolve()
    if not v1_path.is_file():
        raise ToolError(f"v1_file is not a file: {v1_path}")
    if v1_path.suffix != ".py":
        raise ToolError(f"v1_file must be a .py file: {v1_path}")

    # Loaded through the benchmark's own loader, so the probe reports on
    # exactly the program that gets timed.
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

    # Uncaptured first pass: it pays the JIT compile and lets @triton.autotune
    # settle on a config. Capturing it would report every candidate the
    # autotuner benchmarked alongside the one that actually runs.
    run_forward(model, inputs, seed, f"{v1_path}: ModelNew warmup")

    compiled: dict = {}
    with _watch_launches(compiled):
        run_forward(model, inputs, seed, f"{v1_path}: ModelNew")

    rows = []
    for name, kernel in compiled.values():
        # A launched kernel already has its handles; warmup-compiled ones do
        # not, and n_regs stays unset until they are loaded.
        if getattr(kernel, "n_regs", None) is None and hasattr(kernel, "_init_handles"):
            kernel._init_handles()
        rows.append({
            "kernel": name,
            "n_regs": getattr(kernel, "n_regs", None),      # 每个 thread 用的寄存器数量
            "n_spills": getattr(kernel, "n_spills", None),  # 每个 thread 的寄存器溢出数量
            "smem": getattr(kernel.metadata, "shared", None),  # 每个 block 的 shared memory 字节数
        })

    # The tool runner puts this value straight into the request body, where
    # tool_result content must be a string — a dict there is rejected.
    return json.dumps(rows, indent=2, ensure_ascii=False)
