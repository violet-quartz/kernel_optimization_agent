"""Tests for the run directory: layout, version allocation, best pointer."""

import json
import unittest

from tests._helpers import TempDirCase, V0_SOURCE


class TestStartRun(TempDirCase):
    def test_layout(self):
        run = self.make_run()
        self.assertTrue(run.root.is_dir())
        self.assertTrue(run.v0_path.is_file())
        self.assertTrue(run.meta_path.is_file())
        self.assertTrue(run.history_path.is_file())
        self.assertIn("task", run.run_id)

    def test_v0_is_a_frozen_copy(self):
        """The run must not depend on the task file staying put — editing
        tasks/ mid-run would otherwise silently change what v1 is compared to."""
        run = self.make_run()
        self.assertEqual(run.v0_path.read_text(encoding="utf-8"), V0_SOURCE)
        (self.tmp / "task.py").write_text("# gone\n", encoding="utf-8")
        self.assertEqual(run.v0_path.read_text(encoding="utf-8"), V0_SOURCE)

    def test_meta_records_provenance(self):
        run = self.make_run(model="some-model", note="a note")
        meta = json.loads(run.meta_path.read_text(encoding="utf-8"))
        self.assertEqual(meta["agent_model"], "some-model")
        self.assertEqual(meta["note"], "a note")
        self.assertIn("task_sha256", meta)
        self.assertIn("env", meta)

    def test_rejects_a_task_that_is_not_a_python_file(self):
        from run_dir import RunDirError, start_run

        bad = self.write("task.txt", V0_SOURCE)
        with self.assertRaises(RunDirError):
            start_run(bad, runs_root=self.tmp / "runs", set_triton_cache=False)
        with self.assertRaises(RunDirError):
            start_run(self.tmp / "missing.py", runs_root=self.tmp / "runs", set_triton_cache=False)

    def test_rejects_a_task_missing_the_required_names(self):
        from run_dir import RunDirError

        with self.assertRaises(RunDirError) as ctx:
            self.make_run("import torch\n")
        self.assertIn("get_inputs", str(ctx.exception))

    def test_rejects_a_task_that_does_not_parse(self):
        from run_dir import RunDirError

        with self.assertRaises(RunDirError) as ctx:
            self.make_run("class Model(:\n")
        self.assertIn("syntax error", str(ctx.exception).lower())


class TestVersions(TempDirCase):
    def test_allocation_is_sequential_and_zero_padded(self):
        run = self.make_run()
        self.assertEqual(run.versions(), [])
        names = [run.new_version()[0] for _ in range(3)]
        self.assertEqual(names, ["v001", "v002", "v003"])
        self.assertEqual(run.versions(), names)

    def test_versions_ignores_non_version_entries(self):
        """`v0.py`, `triton_cache/` and the `best` symlink all live in the run
        root; only vNNN directories are versions."""
        run = self.make_run()
        run.new_version()
        (run.root / "triton_cache").mkdir(exist_ok=True)
        (run.root / "notes").mkdir()
        run.set_best("v001")
        self.assertEqual(run.versions(), ["v001"])

    def test_best_points_at_the_named_version(self):
        run = self.make_run()
        run.new_version()
        run.new_version()
        run.set_best("v002")
        self.assertTrue(run.best_link.is_symlink())
        self.assertEqual(run.best_link.resolve(), run.version_dir("v002").resolve())
        run.set_best("v001")  # replacing an existing pointer must not fail
        self.assertEqual(run.best_link.resolve(), run.version_dir("v001").resolve())

    def test_best_is_relative_so_the_run_stays_movable(self):
        import shutil

        run = self.make_run()
        run.new_version()
        run.set_best("v001")
        moved = self.tmp / "moved"
        shutil.move(str(run.root), str(moved))
        self.assertEqual((moved / "best").resolve(), (moved / "v001").resolve())

    def test_rejects_unknown_versions(self):
        from run_dir import RunDirError

        run = self.make_run()
        with self.assertRaises(RunDirError):
            run.version_dir("best")
        with self.assertRaises(RunDirError):
            run.set_best("v009")


class TestLookup(TempDirCase):
    def test_open_run_round_trips(self):
        from run_dir import open_run

        run = self.make_run()
        run.update_meta(best="v001")
        reopened = open_run(run.root)
        self.assertEqual(reopened.run_id, run.run_id)
        self.assertEqual(reopened.read_meta()["best"], "v001")

    def test_open_run_rejects_a_directory_without_meta(self):
        from run_dir import RunDirError, open_run

        (self.tmp / "not-a-run").mkdir()
        with self.assertRaises(RunDirError):
            open_run(self.tmp / "not-a-run")

    def test_latest_run_skips_the_latest_symlink(self):
        """`runs/latest` is a symlink sitting alongside the real runs, and it
        sorts last by name — counting it would make latest_run return itself."""
        from run_dir import latest_run

        runs_root = self.tmp / "runs"
        first = self.make_run()
        second = self.make_run()
        self.assertNotEqual(first.run_id, second.run_id)
        self.assertTrue((runs_root / "latest").is_symlink())
        self.assertEqual(latest_run(runs_root).run_id, second.run_id)

    def test_latest_run_errors_when_there_are_no_runs(self):
        from run_dir import RunDirError, latest_run

        (self.tmp / "empty").mkdir()
        with self.assertRaises(RunDirError):
            latest_run(self.tmp / "empty")


if __name__ == "__main__":
    unittest.main()
