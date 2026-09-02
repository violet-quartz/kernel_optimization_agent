"""Run-directory layout for the kernel-optimization agent.

One run is one agent session working on one task. Everything that session
produces lives under a single self-contained directory, so it can be archived,
replayed, or handed to someone else without dragging the rest of the repo
along:

    runs/20260826-140311_centre_random_augmentation/
        v0.py            frozen copy of the task file — the baseline
        meta.json        task provenance + the environment the run started in
        history.jsonl    one line per attempt, appended by the recorder
        triton_cache/    TRITON_CACHE_DIR for this run only
        v001/            one immutable directory per generated kernel
            kernel.py
        best -> v003     symlink to the fastest correct version so far

The task file is *copied* rather than referenced: a speedup is only meaningful
against the exact baseline it was measured on, and editing `tasks/` later must
not silently rewrite history.

Version directories are allocated here (`Run.new_version`) instead of by the
tool that writes the kernel, so the model never picks a directory name and can
never overwrite its own earlier attempt.
"""

import ast
import hashlib
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

DEFAULT_RUNS_ROOT = Path(__file__).resolve().parent.parent / "runs"

# `Model` and the two input factories are what auto_bench_v2.build_case looks
# up; a task missing any of them cannot be benchmarked at all, so it is worth
# failing here rather than after the first kernel has been generated.
REQUIRED_TASK_ATTRS = ("Model", "get_init_inputs", "get_inputs")

_VERSION_RE = re.compile(r"^v(\d{3,})$")
_SLUG_RE = re.compile(r"[^0-9A-Za-z._-]+")


class RunDirError(Exception):
    pass


def _slug(text: str) -> str:
    """Reduce arbitrary text to something safe to put in a directory name."""
    cleaned = _SLUG_RE.sub("-", text).strip("-._")
    return cleaned[:60] or "task"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _top_level_names(tree: ast.Module) -> set[str]:
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _env_info() -> dict:
    """Snapshot whatever identifies the machine a timing came from.

    Every import is optional: the run directory must be creatable on a laptop
    without CUDA, or without torch at all, so a missing dependency is recorded
    as null instead of raising.
    """
    info = {
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "torch": None,
        "triton": None,
        "device": None,
        "device_name": None,
        "device_capability": None,
        "device_count": None,
    }
    try:
        import torch
    except Exception:
        return info

    info["torch"] = torch.__version__
    try:
        import triton

        info["triton"] = triton.__version__
    except Exception:
        pass

    if torch.cuda.is_available():
        info["device"] = "cuda"
        info["device_count"] = torch.cuda.device_count()
        try:
            info["device_name"] = torch.cuda.get_device_name(0)
            info["device_capability"] = ".".join(
                str(x) for x in torch.cuda.get_device_capability(0)
            )
        except Exception:
            pass
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        info["device"] = "mps"
    else:
        info["device"] = "cpu"
    return info


@dataclass(frozen=True)
class Run:
    """Handle on one run directory. Cheap to construct; holds no open state."""

    root: Path
    run_id: str

    # --- fixed layout ---------------------------------------------------

    @property
    def v0_path(self) -> Path:
        return self.root / "v0.py"

    @property
    def meta_path(self) -> Path:
        return self.root / "meta.json"

    @property
    def history_path(self) -> Path:
        return self.root / "history.jsonl"

    @property
    def transcript_path(self) -> Path:
        """Full conversation of the run, one message per line.

        `history.jsonl` records what the model *decided*; this records how it
        got there — every assistant message and every tool result, in order.
        """
        return self.root / "transcript.jsonl"

    @property
    def triton_cache_dir(self) -> Path:
        return self.root / "triton_cache"

    @property
    def best_link(self) -> Path:
        return self.root / "best"

    def version_dir(self, version: str) -> Path:
        if not _VERSION_RE.match(version):
            raise RunDirError(f"not a version name: {version!r} (expected e.g. 'v001')")
        return self.root / version

    def kernel_path(self, version: str) -> Path:
        return self.version_dir(version) / "kernel.py"

    # --- version allocation ---------------------------------------------

    def versions(self) -> list[str]:
        """Existing version names, ascending. `v0.py` is a file, not a version."""
        found = [p.name for p in self.root.iterdir() if p.is_dir() and _VERSION_RE.match(p.name)]
        return sorted(found, key=lambda name: int(_VERSION_RE.match(name).group(1)))

    def new_version(self) -> tuple[str, Path]:
        """Atomically claim the next version directory.

        `mkdir(exist_ok=False)` is the lock: two writers racing for the same
        number cannot both succeed, so the loser just retries with the next one.
        Returns (version_name, version_dir).
        """
        existing = self.versions()
        nxt = int(_VERSION_RE.match(existing[-1]).group(1)) + 1 if existing else 1
        for candidate in range(nxt, nxt + 1000):
            version = f"v{candidate:03d}"
            path = self.root / version
            try:
                path.mkdir(exist_ok=False)
            except FileExistsError:
                continue
            return version, path
        raise RunDirError(f"could not allocate a version directory under {self.root}")

    def set_best(self, version: str) -> Path:
        """Point `best` at `version`, replacing any previous target atomically."""
        target = self.version_dir(version)
        if not target.is_dir():
            raise RunDirError(f"no such version: {target}")
        tmp = self.root / f".best.{os.getpid()}.tmp"
        tmp.unlink(missing_ok=True)
        tmp.symlink_to(version)  # relative, so the run stays movable
        os.replace(tmp, self.best_link)
        return self.best_link

    # --- metadata ---------------------------------------------------------

    def read_meta(self) -> dict:
        return json.loads(self.meta_path.read_text(encoding="utf-8"))

    def update_meta(self, **fields) -> dict:
        meta = self.read_meta()
        meta.update(fields)
        _write_json(self.meta_path, meta)
        return meta


def _write_json(path: Path, payload: dict) -> None:
    """Write JSON atomically, so a crash mid-write cannot truncate meta.json."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _check_task_file(path: Path) -> list[str]:
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RunDirError(f"failed to read task file {path}: {exc}") from exc
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        raise RunDirError(f"syntax error in task file {path}:{exc.lineno}: {exc.msg}") from exc

    defined = _top_level_names(tree)
    missing = [attr for attr in REQUIRED_TASK_ATTRS if attr not in defined]
    if missing:
        raise RunDirError(
            f"{path} does not define {', '.join(missing)} at module level; "
            "the benchmark harness needs all of "
            f"{', '.join(REQUIRED_TASK_ATTRS)}"
        )
    return sorted(defined)


def start_run(
    task_file,
    runs_root=None,
    model: str | None = None,
    note: str | None = None,
    set_triton_cache: bool = True,
) -> Run:
    """Create a fresh run directory for one task and return a handle on it.

    Freezes a copy of the task file as `v0.py`, writes `meta.json` with the
    task's provenance and the environment the run started in, and creates the
    empty `history.jsonl` the recorder appends to.

    Args:
        task_file: Path to the reference implementation (the `tasks/*.py` file
            defining `Model`, `get_init_inputs` and `get_inputs`).
        runs_root: Directory the run is created under. Defaults to `runs/` next
            to the repository root.
        model: Identifier of the agent model driving this run, recorded in
            meta.json so timings can be attributed later.
        note: Free-form description of what this run is trying, recorded in
            meta.json.
        set_triton_cache: When True (default), point `TRITON_CACHE_DIR` at this
            run's own cache directory, so autotuning results from different
            runs cannot pollute each other and are cleaned up with the run.
            Note this mutates `os.environ` for the current process.

    Returns:
        A `Run` handle. `run.root` is the created directory.

    Raises:
        RunDirError: The task file is missing, unreadable, not parseable, or
            does not define the attributes the benchmark harness requires.
    """
    task_path = Path(task_file).resolve()
    if not task_path.is_file():
        raise RunDirError(f"task file is not a file: {task_path}")
    if task_path.suffix != ".py":
        raise RunDirError(f"task file must be a .py file: {task_path}")

    defined = _check_task_file(task_path)

    root_dir = Path(runs_root).resolve() if runs_root else DEFAULT_RUNS_ROOT
    root_dir.mkdir(parents=True, exist_ok=True)

    # Two runs started in the same second get -2, -3, ... — mkdir is the lock.
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = f"{stamp}_{_slug(task_path.stem)}"
    for attempt in range(1, 100):
        run_id = base if attempt == 1 else f"{base}-{attempt}"
        run_root = root_dir / run_id
        try:
            run_root.mkdir(exist_ok=False)
            break
        except FileExistsError:
            continue
    else:
        raise RunDirError(f"could not allocate a run directory under {root_dir}")

    run = Run(root=run_root, run_id=run_id)
    shutil.copy2(task_path, run.v0_path)
    run.triton_cache_dir.mkdir()
    run.history_path.touch()

    _write_json(
        run.meta_path,
        {
            "run_id": run_id,
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "task_name": task_path.stem,
            "task_source": str(task_path),
            "task_sha256": _sha256(run.v0_path),
            "task_defines": defined,
            "agent_model": model,
            "note": note,
            "env": _env_info(),
            "best": None,
        },
    )

    if set_triton_cache:
        os.environ["TRITON_CACHE_DIR"] = str(run.triton_cache_dir)

    _point_latest(root_dir, run_id)
    return run


def _point_latest(runs_root: Path, run_id: str) -> None:
    """Best-effort `runs/latest` symlink. Never fails a run over convenience."""
    link = runs_root / "latest"
    try:
        if link.is_symlink() or not link.exists():
            tmp = runs_root / f".latest.{os.getpid()}.tmp"
            tmp.unlink(missing_ok=True)
            tmp.symlink_to(run_id)
            os.replace(tmp, link)
    except OSError:
        pass


def open_run(run_id_or_path, runs_root=None) -> Run:
    """Reopen an existing run by id, by path, or by the alias `latest`."""
    candidate = Path(run_id_or_path)
    if candidate.is_dir() and (candidate / "meta.json").is_file():
        root = candidate.resolve()
    else:
        root_dir = Path(runs_root).resolve() if runs_root else DEFAULT_RUNS_ROOT
        root = (root_dir / str(run_id_or_path)).resolve()
    if not (root / "meta.json").is_file():
        raise RunDirError(f"not a run directory (no meta.json): {root}")
    return Run(root=root, run_id=root.name)


def latest_run(runs_root=None) -> Run:
    """Most recently created run, by directory name (which is timestamped)."""
    root_dir = Path(runs_root).resolve() if runs_root else DEFAULT_RUNS_ROOT
    # `latest` is a symlink to one of these — including it would make it win
    # the name sort and report itself as the newest run.
    runs = sorted(
        (
            p
            for p in root_dir.iterdir()
            if not p.is_symlink() and (p / "meta.json").is_file()
        ),
        key=lambda p: p.name,
    ) if root_dir.is_dir() else []
    if not runs:
        raise RunDirError(f"no runs found under {root_dir}")
    return Run(root=runs[-1], run_id=runs[-1].name)
