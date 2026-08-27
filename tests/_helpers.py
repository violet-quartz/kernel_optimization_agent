"""Shared fixtures for the local test suite.

Everything here is CPU-only. The two GPU-bound tools (`check_correctness`,
`bench_mark`) cannot produce numbers without an accelerator, so the suite
covers their *contract* — argument coercion, error type, result type — rather
than their measurements. That is where every bug so far has actually been.
"""

import sys
import shutil
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = REPO_ROOT / "tools"
for _p in (str(REPO_ROOT), str(TOOLS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# A task file: what tasks/*.py looks like, minus the hardcoded cuda device.
V0_SOURCE = '''\
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
        self.weight = nn.Parameter(torch.randn(8, 8))

    def forward(self, x):
        return (x @ self.weight) * self.scale


def get_init_inputs():
    return [2.0]


def get_inputs():
    return [torch.randn(4, 8)]
'''

# A valid optimized file: same public surface, ModelNew as a real class.
# Deliberately torch-only — the tests that actually *load* this file would
# otherwise need triton installed, which the dev machine has no reason to have.
V1_SOURCE = V0_SOURCE.replace("class Model(nn.Module):", "class ModelNew(nn.Module):")


def kernel_source(body: str = "return x") -> str:
    """A minimal valid v1 file with a customisable forward body."""
    return f'''\
import torch
import torch.nn as nn
import triton


class ModelNew(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
        self.weight = nn.Parameter(torch.randn(8, 8))

    def forward(self, x):
        {body}


def get_init_inputs():
    return [2.0]


def get_inputs():
    return [torch.randn(4, 8)]
'''


class TempDirCase(unittest.TestCase):
    """Gives each test an isolated tmp dir, removed afterwards."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="koa-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def write(self, name: str, text: str) -> Path:
        path = self.tmp / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def make_run(self, task_source: str = V0_SOURCE, **kwargs):
        from run_dir import start_run

        task = self.write("task.py", task_source)
        kwargs.setdefault("runs_root", self.tmp / "runs")
        kwargs.setdefault("set_triton_cache", False)
        return start_run(task, **kwargs)
