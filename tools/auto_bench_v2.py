import argparse
import ast
import importlib.util
import itertools  # updated: for _get_model_device's chain(parameters, buffers)
import statistics
import sys
import time
import traceback
import types
from contextlib import contextmanager  # updated: for the step() error-attribution helper
from dataclasses import dataclass
from pathlib import Path

import torch


class KsCompareError(Exception):
    pass


# updated: per-dtype default tolerances replace the flat 1e-2/1e-2 defaults.
# Comparing fp32 with the fp16-sized 1e-2 lets a visibly wrong kernel pass, so
# the tolerance follows the data type unless the caller overrides it.
DEFAULT_TOL = {
    torch.float64: (1e-8, 1e-8),
    torch.float32: (1e-5, 1e-5),
    torch.bfloat16: (1e-2, 2e-2),
    torch.float16: (1e-2, 1e-2),
    torch.complex128: (1e-8, 1e-8),
    torch.complex64: (1e-5, 1e-5),
}
_FALLBACK_TOL = (1e-2, 1e-2)

# updated: warn-once channel, so a silently degraded run (failed sync, noisy
# timings) is reported instead of quietly producing meaningless numbers.
_WARNED = set()


def _warn_once(message):
    if message not in _WARNED:
        _WARNED.add(message)
        print(f"WARN {message}", file=sys.stderr)


# updated: timing now carries median + min + stdev + which clock was used,
# instead of a bare median float. median resists outliers, min approximates the
# undisturbed best case, and a large stdev flags a dirty measurement.
@dataclass
class TimingStats:
    median_ms: float
    min_ms: float
    stdev_ms: float
    method: str


@dataclass
class CaseResult:
    name: str
    passed: bool
    v0_ms: float | None = None
    v1_ms: float | None = None
    speedup: float | None = None
    message: str = ""
    # updated: full timing stats and the tolerance actually applied.
    v0_stats: TimingStats | None = None
    v1_stats: TimingStats | None = None
    tol_note: str = ""


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare KS competition v0/v1 Python files. The v0 file must define "
            "Model/get_init_inputs/get_inputs, and the v1 file must define "
            "ModelNew/get_init_inputs/get_inputs. All tensors and models must be on the same device! "
            "For example: python benchmarks/ks/auto_bench.py --v0_file dlblas/kernels/ks_competition/torch/layer_norm.py "
            "--v1_file dlblas/kernels/ks_competition/triton/layer_norm.py "
        )
    )
    parser.add_argument("--v0_file", type=Path, help="Path to the v0 .py file.")
    parser.add_argument("--v1_file", type=Path, help="Path to the v1 .py file.")
    parser.add_argument("--seed", type=int, default=42)
    # updated: default None means "pick from the output dtype"; an explicit
    # value still overrides, and the value used is printed with the result.
    parser.add_argument(
        "--atol",
        type=float,
        default=None,
        help="Absolute tolerance. Default: chosen per output dtype (see DEFAULT_TOL).",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=None,
        help="Relative tolerance. Default: chosen per output dtype (see DEFAULT_TOL).",
    )
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--repeat", type=int, default=500)
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first failed case.",
    )
    parser.add_argument(
        "--full-traceback",
        action="store_true",
        help="Print full Python traceback for load/run failures.",
    )
    return parser.parse_args()


def _is_safe_literal(node):
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


def _filter_module_ast(tree):
    kept_nodes = []
    for node in tree.body:
        if isinstance(
            node,
            (
                ast.Import,
                ast.ImportFrom,
                ast.ClassDef,
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        ):
            kept_nodes.append(node)
        elif isinstance(node, ast.Assign) and _is_safe_literal(node.value):
            kept_nodes.append(node)
        elif (
            isinstance(node, ast.AnnAssign)
            and node.value is not None
            and _is_safe_literal(node.value)
        ):
            kept_nodes.append(node)
    tree.body = kept_nodes
    ast.fix_missing_locations(tree)
    return tree


def _auto_accel_name() -> str | None:
    """Name of the first available accelerator (cuda/npu/mlu), or None."""
    for name, _ in _iter_accelerators():
        return name
    return None


class _RewriteDeviceStr(ast.NodeTransformer):
    """Rewrite device string literals in ks source so a file written for one
    backend runs on whatever accelerator is available here.

    Bare string constants equal to the source device name (e.g. 'npu') are
    rewritten to the detected target. In ks files these only appear as
    `device='npu'` / `.to('npu')`, so this is a safe, targeted swap.
    """

    def __init__(self, src: str, dst: str):
        self.src = src
        self.dst = dst

    def visit_Constant(self, node):
        if isinstance(node.value, str) and node.value == self.src:
            return ast.copy_location(ast.Constant(value=self.dst), node)
        return node


def _rewrite_device_for_backend(tree: ast.AST) -> None:
    """In-place: remap 'npu' device strings to the available accelerator.

    ks competition files are written against Ascend ('npu'); on other backends
    (mlu/cuda) the literal is rejected by torch at runtime, so rewrite it
    before exec. No-op on npu hosts or when no accelerator is present.
    """
    target = _auto_accel_name()
    if target is None or target == "npu":
        return
    _RewriteDeviceStr("npu", target).visit(tree)
    if target == "gcu":
        _RewriteDeviceStr("cuda", target).visit(tree)
    ast.fix_missing_locations(tree)


def load_ks_module(path: Path) -> types.ModuleType:
    if not path.exists():
        raise KsCompareError(f"file does not exist: {path}")
    if path.suffix != ".py":
        raise KsCompareError(f"expected a .py file, got: {path}")

    try:
        source = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        source = path.read_text()
    except OSError as exc:
        raise KsCompareError(f"failed to read {path}: {exc}") from exc

    try:
        tree = ast.parse(source, filename=str(path))
        _rewrite_device_for_backend(tree)
    except SyntaxError as exc:
        raise KsCompareError(f"syntax error in {path}:{exc.lineno}: {exc.msg}") from exc

    module_name = f"_ks_compare_{path.stem}_{abs(hash(path.resolve()))}"
    module = types.ModuleType(module_name)
    module.__file__ = str(path)
    module.__package__ = ""
    module.__spec__ = importlib.util.spec_from_loader(module_name, loader=None)
    sys.modules[module_name] = module
    old_sys_path = list(sys.path)
    sys.path.insert(0, str(path.parent))
    try:
        code = compile(_filter_module_ast(tree), filename=str(path), mode="exec")
        exec(code, module.__dict__)
    except Exception as exc:
        raise KsCompareError(f"failed to load definitions from {path}: {exc}") from exc
    finally:
        sys.path[:] = old_sys_path
        sys.modules.pop(module_name, None)
    return module


def require_attr(module, attr_name, path: Path):
    if not hasattr(module, attr_name):
        raise KsCompareError(f"{path} must define `{attr_name}`.")
    return getattr(module, attr_name)


# updated: replaces call_with_context(func, description). The block runs inline
# instead of inside a lambda, so the call site reads as ordinary code and does
# not need a closure just to defer execution. KsCompareError passes through
# untouched — it already carries a precise location and must not be re-wrapped.
@contextmanager
def step(description):
    """Attribute any failure inside the block to *description*."""
    try:
        yield
    except KsCompareError:
        raise
    except Exception as exc:
        raise KsCompareError(
            f"{description} failed: {type(exc).__name__}: {exc}"
        ) from exc


def as_args(value, description):
    if value is None:
        return ()
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    raise KsCompareError(
        f"{description} must return a list or tuple, got {type(value).__name__}."
    )


def _iter_accelerators():
    """Yield (name, module) for each available accelerator backend.

    Covers cuda / npu (Ascend) / mlu (Cambricon) / gcu (Enflame).
    Add more backends here asneeded;
    set_seed / device detection all derive from this.
    """
    for name in ("gcu", "cuda", "npu", "mlu"):
        mod = getattr(torch, name, None)
        if mod is None:
            continue
        try:
            if mod.is_available():
                yield name, mod
        except Exception:
            continue


def set_seed(seed):
    torch.manual_seed(seed)
    for _name, mod in _iter_accelerators():
        try:
            mod.manual_seed_all(seed)
        except Exception:
            pass


# updated: replaces sync_devices(), which synchronized every installed backend
# and swallowed every failure. Two problems fixed here:
#   - this runs inside the timed region, and syncing all backends adds host
#     overhead to both v0 and v1, which deflates the measured speedup on small
#     kernels; only the device under test is synchronized now.
#   - a failed sync silently yields launch-latency numbers instead of real
#     kernel time, so it now warns instead of passing unnoticed.
def sync_device(device):
    """Block until *device* has finished its queued work."""
    if device is None or device.type == "cpu":
        return
    mod = getattr(torch, device.type, None)
    if mod is None:
        _warn_once(
            f"no torch.{device.type} module to synchronize; timings may be async "
            "and therefore meaningless"
        )
        return
    try:
        mod.synchronize()
    except Exception as exc:
        _warn_once(
            f"torch.{device.type}.synchronize() failed: {exc}; "
            "reported timings are unreliable"
        )


def clone_value(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, list):
        return [clone_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_value(item) for item in value)
    if isinstance(value, dict):
        return {key: clone_value(item) for key, item in value.items()}
    return value


def describe_value(value):
    if isinstance(value, torch.Tensor):
        return (
            f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype}, "
            f"device={value.device})"
        )
    if isinstance(value, (list, tuple)):
        inner = ", ".join(describe_value(item) for item in value)
        return f"{type(value).__name__}({inner})"
    if isinstance(value, dict):
        inner = ", ".join(
            f"{key}: {describe_value(item)}" for key, item in value.items()
        )
        return f"dict({inner})"
    return repr(value)


# updated: resolves the effective tolerance from the tensor dtype when the
# caller left --atol/--rtol unset.
def resolve_tol(dtype, atol, rtol):
    """Fill in per-dtype defaults for whichever tolerance was not given."""
    default_atol, default_rtol = DEFAULT_TOL.get(dtype, _FALLBACK_TOL)
    return (
        default_atol if atol is None else atol,
        default_rtol if rtol is None else rtol,
    )


# updated: helper for the richer failure diagnostics below.
def _unravel_index(index, shape):
    """Flat index -> per-dimension coordinates (row-major, like Tensor.flatten)."""
    coords = []
    for dim in reversed(shape):
        coords.append(index % dim)
        index //= dim
    return tuple(reversed(coords))


# updated: helper for the richer failure diagnostics below; complex .item()
# returns a Python complex, which the plain :.6e format cannot render.
def _fmt_scalar(value):
    if isinstance(value, complex):
        return f"{value.real:.6e}{value.imag:+.6e}j"
    return f"{float(value):.6e}"


# updated: extracted from compare_values and extended with the location of the
# worst element, the two values at that location, and the share of elements out
# of tolerance. These are what make a failure actionable: one bad row means a
# boundary bug, a bad tail means the remainder is unhandled, a uniform small
# error means the tolerance is too tight for the dtype, all bad means the
# algorithm is wrong. max_abs_diff alone leaves the agent guessing.
def _diff_summary(lhs, rhs, atol, rtol):
    """Describe where and how badly *rhs* deviates from reference *lhs*."""
    if lhs.is_complex() or rhs.is_complex():
        # complex tensors cannot go through .float(); abs() of the difference
        # is already the magnitude we want.
        diff = (lhs - rhs).abs()
        ref = lhs.abs()
    else:
        # fp16 subtraction overflows past 65504 and its mean() accumulates
        # garbage over large tensors, so widen before computing statistics.
        diff = (lhs.float() - rhs.float()).abs()
        ref = lhs.float().abs()

    total = diff.numel()
    if total == 0:
        return "empty tensor"

    flat = diff.flatten()
    idx = int(flat.argmax().item())
    pos = _unravel_index(idx, tuple(lhs.shape))
    exceed = int((diff > atol + rtol * ref).sum().item())
    return (
        f"max_abs_diff={_fmt_scalar(flat[idx].item())} at index {pos} "
        f"(v0={_fmt_scalar(lhs.flatten()[idx].item())}, "
        f"v1={_fmt_scalar(rhs.flatten()[idx].item())}), "
        f"mean_abs_diff={_fmt_scalar(diff.mean().item())}, "
        f"elements_exceeding_tol={exceed}/{total}"
    )


# updated: signature gained tol_used, an accumulator recording the tolerance
# actually applied per tensor so the caller can report it.
def compare_values(v0, v1, path, atol, rtol, tol_used=None):
    if tol_used is None:
        tol_used = []

    if isinstance(v0, torch.Tensor) or isinstance(v1, torch.Tensor):
        if not isinstance(v0, torch.Tensor) or not isinstance(v1, torch.Tensor):
            raise KsCompareError(
                f"{path}: output type mismatch: {type(v0).__name__} vs {type(v1).__name__}"
            )
        if v0.shape != v1.shape:
            raise KsCompareError(
                f"{path}: tensor shape mismatch: {v0.shape} vs {v1.shape}"
            )
        # updated: dtype was never checked. torch.allclose refuses mixed dtypes
        # with a raw RuntimeError that bypasses KsCompareError entirely, and
        # torch.equal reports False on a dtype difference while every element
        # matches — producing the self-contradictory "mismatched_elements=0".
        if v0.dtype != v1.dtype:
            raise KsCompareError(
                f"{path}: tensor dtype mismatch: {v0.dtype} vs {v1.dtype}"
            )

        # updated: dtype is now known to match, so the branch tests v0 only
        # (was: `v0.dtype.is_floating_point or v1.dtype... or v0.is_complex()...`).
        if v0.dtype.is_floating_point or v0.is_complex():
            # updated: tolerance resolved per dtype and recorded for reporting.
            eff_atol, eff_rtol = resolve_tol(v0.dtype, atol, rtol)
            tol_used.append((path, v0.dtype, eff_atol, eff_rtol))
            lhs = v0.detach()
            rhs = v1.detach().to(lhs.device)
            # updated: argument order swapped (was allclose(lhs, rhs, ...)).
            # allclose scales rtol by the magnitude of its *second* argument, so
            # the reference has to sit there — otherwise an inflated candidate
            # widens its own tolerance and a wrong kernel can pass.
            if not torch.allclose(
                rhs, lhs, atol=eff_atol, rtol=eff_rtol, equal_nan=True
            ):
                # updated: failure message now carries the full diff summary.
                raise KsCompareError(
                    f"{path}: tensor values differ; "
                    f"{_diff_summary(lhs, rhs, eff_atol, eff_rtol)}, "
                    f"atol={eff_atol}, rtol={eff_rtol}, "
                    f"v0={describe_value(v0)}, v1={describe_value(v1)}"
                )
        else:
            lhs = v0.detach()
            rhs = v1.detach().to(lhs.device)
            if not torch.equal(lhs, rhs):
                # updated: report the mismatch count against the total and
                # locate the first differing element, instead of a bare count.
                ne = lhs != rhs
                mismatch = int(ne.sum().item())
                total = lhs.numel()
                detail = ""
                if mismatch:
                    idx = int(ne.flatten().nonzero()[0].item())
                    pos = _unravel_index(idx, tuple(lhs.shape))
                    detail = (
                        f", first at index {pos} "
                        f"(v0={lhs.flatten()[idx].item()!r}, "
                        f"v1={rhs.flatten()[idx].item()!r})"
                    )
                raise KsCompareError(
                    f"{path}: tensor values differ; "
                    f"mismatched_elements={mismatch}/{total}{detail}, "
                    f"v0={describe_value(v0)}, v1={describe_value(v1)}"
                )
        return

    if isinstance(v0, tuple) or isinstance(v1, tuple):
        if not isinstance(v0, tuple) or not isinstance(v1, tuple):
            raise KsCompareError(
                f"{path}: output type mismatch: {type(v0).__name__} vs {type(v1).__name__}"
            )
        if len(v0) != len(v1):
            raise KsCompareError(
                f"{path}: tuple length mismatch: {len(v0)} vs {len(v1)}"
            )
        for i, (item0, item1) in enumerate(zip(v0, v1)):
            # updated: thread tol_used through the recursion.
            compare_values(item0, item1, f"{path}[{i}]", atol, rtol, tol_used)
        return

    if isinstance(v0, list) or isinstance(v1, list):
        if not isinstance(v0, list) or not isinstance(v1, list):
            raise KsCompareError(
                f"{path}: output type mismatch: {type(v0).__name__} vs {type(v1).__name__}"
            )
        if len(v0) != len(v1):
            raise KsCompareError(
                f"{path}: list length mismatch: {len(v0)} vs {len(v1)}"
            )
        for i, (item0, item1) in enumerate(zip(v0, v1)):
            # updated: thread tol_used through the recursion.
            compare_values(item0, item1, f"{path}[{i}]", atol, rtol, tol_used)
        return

    if isinstance(v0, dict) or isinstance(v1, dict):
        if not isinstance(v0, dict) or not isinstance(v1, dict):
            raise KsCompareError(
                f"{path}: output type mismatch: {type(v0).__name__} vs {type(v1).__name__}"
            )
        if set(v0) != set(v1):
            raise KsCompareError(
                f"{path}: dict keys mismatch: {sorted(v0)} vs {sorted(v1)}"
            )
        for key in sorted(v0):
            # updated: thread tol_used through the recursion.
            compare_values(v0[key], v1[key], f"{path}[{key!r}]", atol, rtol, tol_used)
        return

    if v0 != v1:
        raise KsCompareError(f"{path}: values differ: {v0!r} vs {v1!r}")


# updated: renders the tolerance actually applied. Printing it matters in an
# agent loop: --atol/--rtol are command-line arguments, so an agent that widens
# them to "pass" the check would otherwise be indistinguishable from a real win.
def format_tol_used(tol_used):
    """One line naming the tolerance actually applied, per dtype."""
    if not tol_used:
        return "no floating-point output compared"
    seen = []
    for _path, dtype, atol, rtol in tol_used:
        key = (dtype, atol, rtol)
        if key not in seen:
            seen.append(key)
    return "; ".join(
        f"{str(dtype).replace('torch.', '')} atol={atol:g} rtol={rtol:g}"
        for dtype, atol, rtol in seen
    )


def build_case(v0_path: Path, v1_path: Path, seed: int):
    v0_module = load_ks_module(v0_path)
    v1_module = load_ks_module(v1_path)

    model_cls = require_attr(v0_module, "Model", v0_path)
    model_new_cls = require_attr(v1_module, "ModelNew", v1_path)
    v0_get_init_inputs = require_attr(v0_module, "get_init_inputs", v0_path)
    v1_get_init_inputs = require_attr(v1_module, "get_init_inputs", v1_path)
    v0_get_inputs = require_attr(v0_module, "get_inputs", v0_path)
    v1_get_inputs = require_attr(v1_module, "get_inputs", v1_path)

    for func, name, path in (
        (v0_get_init_inputs, "get_init_inputs", v0_path),
        (v1_get_init_inputs, "get_init_inputs", v1_path),
        (v0_get_inputs, "get_inputs", v0_path),
        (v1_get_inputs, "get_inputs", v1_path),
    ):
        if not callable(func):
            raise KsCompareError(f"{path}: `{name}` must be callable.")

    # updated: the six as_args(call_with_context(...)) nests below became
    # `with step(...)` blocks — same error attribution, no lambdas.
    set_seed(seed)
    with step(f"{v0_path}: get_init_inputs()"):
        v0_init_raw = v0_get_init_inputs()
    v0_init_args = as_args(v0_init_raw, f"{v0_path}: get_init_inputs()")

    set_seed(seed)
    with step(f"{v1_path}: get_init_inputs()"):
        v1_init_raw = v1_get_init_inputs()
    v1_init_args = as_args(v1_init_raw, f"{v1_path}: get_init_inputs()")

    with step(f"{v0_path}: Model(...)"):
        model = model_cls(*v0_init_args)
    with step(f"{v1_path}: ModelNew(...)"):
        model_new = model_new_cls(*v1_init_args)

    if hasattr(model, "eval"):
        model.eval()
    if hasattr(model_new, "eval"):
        model_new.eval()

    set_seed(seed)
    with step(f"{v0_path}: get_inputs()"):
        v0_inputs_raw = v0_get_inputs()
    v0_inputs = as_args(v0_inputs_raw, f"{v0_path}: get_inputs()")

    set_seed(seed)
    with step(f"{v1_path}: get_inputs()"):
        v1_inputs_raw = v1_get_inputs()
    v1_inputs = as_args(v1_inputs_raw, f"{v1_path}: get_inputs()")

    if len(v0_inputs) != len(v1_inputs):
        raise KsCompareError(
            f"get_inputs argument count mismatch: {v0_path} returns {len(v0_inputs)} "
            f"args, {v1_path} returns {len(v1_inputs)} args."
        )
    return model, model_new, v0_inputs, v1_inputs


# updated: extracted from compare_case, where it was
#     try: model_new.load_state_dict(model.state_dict())
#     except Exception: pass
# Swallowing that exception leaves the two models on different random weights,
# and the accuracy check then fails as a bogus "output mismatch" — a real
# failure disguised as a different one. An empty state_dict (purely functional
# Model) is the one legitimate no-op and is now distinguished from a genuine
# structural mismatch, which raises.
def sync_weights(model, model_new, description):
    """Copy v0's parameters and buffers into ModelNew.

    set_seed alone does not guarantee identical weights: ModelNew draws from the
    RNG in its own order, so any change in module layout desynchronizes the
    initialization and the accuracy check then fails for a reason that has
    nothing to do with the kernel.
    """
    state = model.state_dict()
    if not state:
        return
    try:
        model_new.load_state_dict(state)
    except Exception as exc:
        raise KsCompareError(
            f"{description}: cannot load v0 weights into ModelNew "
            f"({type(exc).__name__}: {exc}). The two models expose different "
            "parameter structures, so an accuracy comparison would be "
            "meaningless. Make ModelNew keep v0's state_dict keys and shapes."
        ) from exc


def run_forward(model, inputs, seed, description):
    set_seed(seed)
    cloned_inputs = clone_value(inputs)
    try:
        with torch.no_grad():
            return model.forward(*cloned_inputs)
    except Exception as exc:
        raise KsCompareError(f"{description} forward failed: {exc}") from exc


# updated: takes the target device (was: synced every backend), times with CUDA
# events where available, and returns TimingStats instead of a median float.
def time_forward(model, inputs, seed, warmup, repeat, device):
    def one_call():
        with torch.no_grad():
            model.forward(*inputs)

    # Warmup output is discarded, so its RNG state does not matter; it only has
    # to pay the one-off costs (context init, autotune, JIT, allocator, clocks).
    for _ in range(warmup):
        one_call()
    sync_device(device)  # updated: drain warmup on the target device only

    samples = []
    # updated: CUDA event timing measures device-side elapsed time, excluding
    # the Python and kernel-launch overhead that dominates perf_counter on short
    # kernels. Events are CUDA-only, so other backends keep perf_counter.
    use_events = device is not None and device.type == "cuda"
    if use_events:
        method = "cuda_event"
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        for _ in range(repeat):
            set_seed(seed)
            start_ev.record()
            one_call()
            end_ev.record()
            end_ev.synchronize()
            samples.append(start_ev.elapsed_time(end_ev))
    else:
        method = "perf_counter"
        for _ in range(repeat):
            set_seed(seed)
            start = time.perf_counter()
            one_call()
            sync_device(device)  # updated: target device only
            samples.append((time.perf_counter() - start) * 1000.0)

    # updated: min and stdev kept alongside the median.
    median_ms = statistics.median(samples)
    stdev_ms = statistics.stdev(samples) if len(samples) > 1 else 0.0
    return TimingStats(
        median_ms=median_ms,
        min_ms=min(samples),
        stdev_ms=stdev_ms,
        method=method,
    )


def _get_model_device(model):
    """Return the device of *model*'s first parameter or buffer, or None."""
    # updated: `for ... return` over chain(parameters, buffers) replaces two
    # next()/except StopIteration blocks. Behaviour is identical, including the
    # None for a model that has neither (a purely functional Model), which means
    # "unknown" and lets the caller fall through to the next signal.
    for tensor in itertools.chain(model.parameters(), model.buffers()):
        return tensor.device
    return None


def _first_input_device(inputs):
    """Return the device of the first tensor found in nested *inputs*, or None."""
    if isinstance(inputs, torch.Tensor):
        return inputs.device
    if isinstance(inputs, (list, tuple)):
        for item in inputs:
            d = _first_input_device(item)
            if d is not None:
                return d
    if isinstance(inputs, dict):
        for item in inputs.values():
            d = _first_input_device(item)
            if d is not None:
                return d
    return None


def _detect_target_device(model, model_new, v0_inputs, v1_inputs):
    """Pick a non-CPU device from models/inputs, or auto-detect one.

    Priority: model device > input device > auto-detect (cuda → npu).
    Raises KsCompareError if no accelerator is available.
    """
    for m in (model, model_new):
        d = _get_model_device(m)
        if d is not None and d.type != "cpu":
            return d
    for inputs in (v0_inputs, v1_inputs):
        d = _first_input_device(inputs)
        if d is not None and d.type != "cpu":
            return d
    for name, _ in _iter_accelerators():
        return torch.device(name)
    raise KsCompareError(
        "no accelerator device available (cuda/npu/mlu); "
        "cannot run accuracy or performance comparison on CPU."
    )


def _move_to_device(value, device):
    """Recursively copy every tensor in *value* to *device*."""
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    return value


def compare_case(name, v0_path, v1_path, args):
    model, model_new, v0_inputs, v1_inputs = build_case(v0_path, v1_path, args.seed)

    target_device = _detect_target_device(model, model_new, v0_inputs, v1_inputs)

    # updated: was a try/except-pass around load_state_dict; see sync_weights.
    sync_weights(model, model_new, name)

    if hasattr(model, "to"):
        model = model.to(target_device)
    if hasattr(model_new, "to"):
        model_new = model_new.to(target_device)

    v0_inputs = _move_to_device(v0_inputs, target_device)
    # updated: dropped the dead `v1_inputs = _move_to_device(v1_inputs, ...)`,
    # which the next line overwrote immediately. Cloning v0's tensors is the
    # intended behaviour — v1 gets a private copy of the very same values, so an
    # in-place kernel cannot corrupt the reference run, and v1's own
    # get_inputs() result is deliberately unused.
    v1_inputs = clone_value(v0_inputs)

    v0_output = run_forward(model, v0_inputs, args.seed, f"{name}: v0")
    v1_output = run_forward(model_new, v1_inputs, args.seed, f"{name}: v1")
    # updated: collect the tolerance actually applied.
    tol_used = []
    compare_values(v0_output, v1_output, "output", args.atol, args.rtol, tol_used)

    # updated: pass the target device through to the timing loop.
    v0_stats = time_forward(
        model, v0_inputs, args.seed, args.warmup, args.repeat, target_device
    )
    v1_stats = time_forward(
        model_new, v1_inputs, args.seed, args.warmup, args.repeat, target_device
    )
    speedup = (
        v0_stats.median_ms / v1_stats.median_ms
        if v1_stats.median_ms > 0
        else float("inf")
    )
    # updated: a wide spread means something else was using the device or the
    # clocks were not locked, which makes the speedup untrustworthy — say so.
    for label, stats in (("v0", v0_stats), ("v1", v1_stats)):
        if stats.median_ms > 0 and stats.stdev_ms / stats.median_ms > 0.1:
            _warn_once(
                f"{label} timing is noisy (stdev/median="
                f"{stats.stdev_ms / stats.median_ms:.1%}); another process may be "
                "sharing the device or clocks are not locked"
            )
    return CaseResult(
        name=name,
        passed=True,
        v0_ms=v0_stats.median_ms,
        v1_ms=v1_stats.median_ms,
        speedup=speedup,
        # updated: carry the timing stats and tolerance note to the caller.
        v0_stats=v0_stats,
        v1_stats=v1_stats,
        tol_note=format_tol_used(tol_used),
    )

from anthropic import beta_tool

def check_input_file_path(v0_file: Path, v1_file: Path) -> tuple[Path]:
    v0_path = v0_file.resolve()
    v1_path = v1_file.resolve()
    if not v0_path.is_file():
        raise SystemExit(f"v0_file is not a file: {v0_path}")
    if not v1_path.is_file():
        raise SystemExit(f"v1_file is not a file: {v1_path}")
    if v0_path.suffix != ".py":
        raise SystemExit(f"v0_file must be a .py file: {v0_path}")
    if v1_path.suffix != ".py":
        raise SystemExit(f"v1_file must be a .py file: {v1_path}")
    return v0_path, v1_path

@beta_tool
def check_correctness(v0_file: Path, v1_file: Path, seed: int=42, atol: float | None=None, rtol: float | None=None):
    """Check that an optimized kernel produces the same output as the original.

    Loads `Model` from the v0 file and `ModelNew` from the v1 file, copies v0's
    weights into v1 so both start from identical parameters, moves both to the
    same device, and runs a single forward pass on each with the same inputs
    (v1 gets a private clone, so an in-place kernel cannot corrupt the
    reference). The two outputs are then compared element by element.

    Args:
        v0_file: Path of original kernel code python file. Must be an existing
            .py file that defines `Model`, `get_init_inputs` and `get_inputs`.
        v1_file: Path of optimized kernel code python file. Must be an existing
            .py file that defines `ModelNew`, `get_init_inputs` and
            `get_inputs`.
        seed: RNG seed used for input generation and for both forward passes,
            so the two versions see identical random state. Defaults to 42.
        atol: Absolute tolerance for the output comparison. Defaults to None,
            which picks a per-dtype default (looser for fp16/bf16 than fp32).
        rtol: Relative tolerance for the output comparison. Defaults to None,
            which picks a per-dtype default.

    Returns:
        A short confirmation string naming the tolerance that was applied.

    Raises:
        KsCompareError: The outputs differ, the shapes/dtypes/structures do not
            match, or a model failed to build or run. The message reports the
            worst mismatch (index, both values, absolute and relative error).
        SystemExit: A path is missing or is not a .py file.
    """
    v0_path, v1_path = check_input_file_path(v0_file=v0_file, v1_file=v1_file)
    name = str(v0_path)
    model, model_new, v0_inputs, v1_inputs = build_case(v0_path, v1_path, seed)
    
    target_device = _detect_target_device(model, model_new, v0_inputs, v1_inputs)

    # updated: was a try/except-pass around load_state_dict; see sync_weights.
    sync_weights(model, model_new, name)

    if hasattr(model, "to"):
        model = model.to(target_device)
    if hasattr(model_new, "to"):
        model_new = model_new.to(target_device)

    v0_inputs = _move_to_device(v0_inputs, target_device)
    # updated: dropped the dead `v1_inputs = _move_to_device(v1_inputs, ...)`,
    # which the next line overwrote immediately. Cloning v0's tensors is the
    # intended behaviour — v1 gets a private copy of the very same values, so an
    # in-place kernel cannot corrupt the reference run, and v1's own
    # get_inputs() result is deliberately unused.
    v1_inputs = clone_value(v0_inputs)

    v0_output = run_forward(model, v0_inputs, seed, f"{name}: v0")
    v1_output = run_forward(model_new, v1_inputs, seed, f"{name}: v1")
    # updated: collect the tolerance actually applied.
    tol_used = []
    compare_values(v0_output, v1_output, "output", atol, rtol, tol_used)
    # A bare True cannot be a tool_result: the runner passes the return value
    # straight into the request body, where content must be text.
    return f"correct: outputs match ({format_tol_used(tol_used) or 'no tolerance applied'})"

@beta_tool
def bench_mark(v0_file: str, v1_file: str, seed: int=42, warmup: int=200, repeat: int=500):
    """Benchmark an optimized kernel against the original and report the speedup.

    Loads `Model` from the v0 file and `ModelNew` from the v1 file, syncs
    weights, moves both to the same device, and times `repeat` forward passes of
    each after `warmup` untimed calls. On CUDA the timing uses CUDA events
    (device-side elapsed time); other backends fall back to `time.perf_counter`.
    This does NOT verify correctness — run `check_correctness` first, otherwise
    a wrong kernel can look arbitrarily fast.

    Args:
        v0_file: Path of original kernel code python file. Must be an existing
            .py file that defines `Model`, `get_init_inputs` and `get_inputs`.
        v1_file: Path of optimized kernel code python file. Must be an existing
            .py file that defines `ModelNew`, `get_init_inputs` and
            `get_inputs`.
        seed: RNG seed used for input generation and re-applied before every
            timed call, so both versions run on identical data. Defaults to 42.
        warmup: Number of untimed forward passes per version, used to absorb
            one-off costs (context init, autotune, JIT, allocator warmup).
            Defaults to 200; lower it for very slow kernels.
        repeat: Number of timed forward passes per version that the statistics
            are computed from. Defaults to 500; more repeats means less noise.

    Returns:
        A dict with `speedup` (v0 median / v1 median — greater than 1 means v1
        is faster), `v0_median_ms` / `v1_median_ms`, `v0_min_ms` / `v1_min_ms`,
        `v0_stdev_ms` / `v1_stdev_ms`, the `timing_method` used, and
        `warnings`. A warning appears when stdev/median exceeds 10% for either
        version: the device was likely shared or its clocks were not locked,
        and the speedup should not be trusted.

    Raises:
        KsCompareError: A model failed to build or run.
        SystemExit: A path is missing or is not a .py file.
    """
    v0_path, v1_path = check_input_file_path(v0_file=v0_file, v1_file=v1_file)
    name = str(v0_path)
    model, model_new, v0_inputs, v1_inputs = build_case(v0_path, v1_path, seed)
    
    target_device = _detect_target_device(model, model_new, v0_inputs, v1_inputs)

    # updated: was a try/except-pass around load_state_dict; see sync_weights.
    sync_weights(model, model_new, name)

    if hasattr(model, "to"):
        model = model.to(target_device)
    if hasattr(model_new, "to"):
        model_new = model_new.to(target_device)

    v0_inputs = _move_to_device(v0_inputs, target_device)
    # updated: dropped the dead `v1_inputs = _move_to_device(v1_inputs, ...)`,
    # which the next line overwrote immediately. Cloning v0's tensors is the
    # intended behaviour — v1 gets a private copy of the very same values, so an
    # in-place kernel cannot corrupt the reference run, and v1's own
    # get_inputs() result is deliberately unused.
    v1_inputs = clone_value(v0_inputs)

    v0_stats = time_forward(
        model, v0_inputs, seed, warmup, repeat, target_device
    )
    v1_stats = time_forward(
        model_new, v1_inputs, seed, warmup, repeat, target_device
    )
    speedup = (
        v0_stats.median_ms / v1_stats.median_ms
        if v1_stats.median_ms > 0
        else float("inf")
    )
    # updated: a wide spread means something else was using the device or the
    # clocks were not locked, which makes the speedup untrustworthy — say so.
    # The warnings are returned as well as logged: as a tool this is the only
    # channel the caller sees.
    warnings = []
    for label, stats in (("v0", v0_stats), ("v1", v1_stats)):
        if stats.median_ms > 0 and stats.stdev_ms / stats.median_ms > 0.1:
            message = (
                f"{label} timing is noisy (stdev/median="
                f"{stats.stdev_ms / stats.median_ms:.1%}); another process may be "
                "sharing the device or clocks are not locked"
            )
            warnings.append(message)
            _warn_once(message)
    # TimingStats is a dataclass and cannot cross the tool boundary — the
    # runner puts this value straight into the request body, which must be JSON.
    return {
        "speedup": round(speedup, 4),
        "v0_median_ms": round(v0_stats.median_ms, 6),
        "v1_median_ms": round(v1_stats.median_ms, 6),
        "v0_min_ms": round(v0_stats.min_ms, 6),
        "v1_min_ms": round(v1_stats.min_ms, 6),
        "v0_stdev_ms": round(v0_stats.stdev_ms, 6),
        "v1_stdev_ms": round(v1_stats.stdev_ms, 6),
        "timing_method": v0_stats.method,
        "warnings": warnings,
    }


def main():
    args = parse_args()
    v0_path = args.v0_file.resolve()
    v1_path = args.v1_file.resolve()
    if not v0_path.is_file():
        raise SystemExit(f"v0_file is not a file: {v0_path}")
    if not v1_path.is_file():
        raise SystemExit(f"v1_file is not a file: {v1_path}")
    if v0_path.suffix != ".py":
        raise SystemExit(f"v0_file must be a .py file: {v0_path}")
    if v1_path.suffix != ".py":
        raise SystemExit(f"v1_file must be a .py file: {v1_path}")
    if args.warmup < 0 or args.repeat <= 0:
        raise SystemExit("--warmup must be >= 0 and --repeat must be > 0.")

    name = str(v0_path)

    try:
        result = compare_case(name, v0_path, v1_path, args)
        # updated: the PASS line now reports min and stdev next to the median,
        # which clock produced the numbers, and the tolerance actually applied.
        print(
            f"PASS accuracy; "
            f"v0={result.v0_ms:.6f} ms "
            f"(min={result.v0_stats.min_ms:.6f}, sd={result.v0_stats.stdev_ms:.6f}), "
            f"v1={result.v1_ms:.6f} ms "
            f"(min={result.v1_stats.min_ms:.6f}, sd={result.v1_stats.stdev_ms:.6f}), "
            f"speedup={result.speedup:.3f}x, "
            f"timing={result.v0_stats.method}, tol[{result.tol_note}]"
        )
        passed = 1
        failed = 0
    except Exception as exc:
        if args.full_traceback:
            traceback.print_exc()
        message = str(exc)
        result = CaseResult(name=name, passed=False, message=message)
        print(f"FAIL {message}")
        passed = 0
        failed = 1

    print(f"\nSummary: {passed} passed, {failed} failed, 1 total.")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
