"""Pure /proc fixtures; no process control, cloud connection or GPU use."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from scripts import continue_after_quality_20261114 as continuation


class ContinuationGates(unittest.TestCase):
    def test_linux_stat_field22_and_process_absence(self):
        with tempfile.TemporaryDirectory() as temporary:
            proc = Path(temporary)
            directory = proc / "30681"
            directory.mkdir()
            (directory / "stat").write_text(
                "30681 (python (worker) name) " + " ".join(["S"] + ["0"] * 18 + ["987654321"]) + "\n",
                encoding="utf-8")
            self.assertEqual(continuation.proc_identity(30681, proc),
                             {"pid": 30681, "state": "S", "starttime_ticks": 987654321})
            self.assertIsNone(continuation.proc_identity(30682, proc))

    def test_exact_torchrun_configuration_and_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            proc = Path(temporary)
            directory = proc / "30681"
            directory.mkdir()
            # An ordinary directory works as a readlink-resolved fake cwd.
            project = directory / "cwd"
            project.mkdir()
            config = project / "configs/quality.json"
            run = project / "quality_run"
            command = ["/venv/bin/python", "/venv/bin/torchrun", "--standalone", "--nproc_per_node=2",
                       "-m", "training.train", "--config", "configs/quality.json", "--output", str(run)]
            def write_command(parts):
                (directory / "cmdline").write_bytes(b"\0".join(part.encode() for part in parts) + b"\0")
            write_command(command)
            self.assertEqual(continuation.verify_training_command(30681, project, config, run, proc), command)
            bad = list(command)
            bad[bad.index("configs/quality.json")] = "configs/dr.json"
            write_command(bad)
            with self.assertRaisesRegex(ValueError, "different configuration"):
                continuation.verify_training_command(30681, project, config, run, proc)
            bad = list(command)
            bad[-1] = str(project / "other_run")
            write_command(bad)
            with self.assertRaisesRegex(ValueError, "different run"):
                continuation.verify_training_command(30681, project, config, run, proc)

    def test_pid_reuse_stops_waiting_without_signalling_or_sleeping(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = SimpleNamespace(operational_root=root, project_root=root, runtime_root=root,
                                   training_pid=30681, quality_run=root / "quality", quality_config=root / "quality.json",
                                   suite_root=root / "remaining")
            worker = continuation.Continuation(args)
            original = {"pid": 30681, "state": "S", "starttime_ticks": 111}
            replacement = {"pid": 30681, "state": "R", "starttime_ticks": 999}
            with patch.object(continuation, "proc_identity", side_effect=[original, original, replacement]), \
                 patch.object(continuation, "verify_training_command", return_value=["torchrun"]), \
                 patch.object(continuation.time, "sleep") as sleep:
                worker.wait_quality()
                sleep.assert_not_called()
            self.assertEqual(worker.state["wait_exit_reason"], "pid_reused_expected_process_ended")
            self.assertFalse(worker.state["signals_sent_to_training_job"])


if __name__ == "__main__":
    unittest.main()
