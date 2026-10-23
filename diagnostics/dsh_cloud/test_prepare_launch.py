"""CPU-only preparation/launch checks; process creation is always mocked."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import launch
import prepare


class PreparedCaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="dsh-custom-case-")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.harness = self.base / "harness"
        for relative in ("node-v24.19.0-linux-x64/bin/node", "node_modules/@deepseek-ai/dsh/lib/bin.js"):
            file = self.harness / relative
            file.parent.mkdir(parents=True, exist_ok=True)
            file.touch()
        self.root = self.base / "run"
        self.task = self.base / "expense.txt"
        self.task.write_text("请制作费用记录网页，数据保存到 localStorage。\n", encoding="utf-8")

    def invoke(self, module, *args):
        out, err = io.StringIO(), io.StringIO()
        with patch("sys.argv", [module.__file__, *map(str, args)]), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                module.main()
            except SystemExit as error:
                return error.code, out.getvalue(), err.getvalue()
        return 0, out.getvalue(), err.getvalue()

    def prepare(self, *extra):
        code, out, err = self.invoke(prepare, "--run-root", self.root, "--harness", self.harness,
                                    "--context-window", "32768", *extra)
        self.assertEqual(code, 0, err)
        return json.loads((self.root / "manifest.json").read_text())

    def plan(self, case, *extra):
        return self.invoke(launch, "--run-root", self.root, "--case", case, *extra)

    def test_defaults_preserved_and_dry_run_never_launches(self):
        with patch("subprocess.Popen") as spawn:
            manifest = self.prepare()
            self.assertEqual(set(manifest["cases"]), {"marker", "csgo"})
            self.assertEqual(manifest["max_tokens"], 8192)
            self.assertEqual(manifest["context_window"], 32768)
            for case, queued in (("marker", False), ("csgo", True)):
                code, out, err = self.plan(case)
                self.assertEqual(code, 0, err)
                plan = json.loads(out)
                self.assertEqual(any("session_runner.patch.yml" in x for x in plan["command"]), queued)
                self.assertFalse((self.root / case / "launch.json").exists())
            spawn.assert_not_called()
        self.assertEqual((self.root / "csgo/task.txt").read_text().strip(), "做个网页版的csgo。")

    def test_custom_task_is_snapshotted_and_queued_once(self):
        original = self.task.read_text()
        manifest = self.prepare("--case-name", "expense-tracker", "--task-file", self.task)
        self.task.write_text("An unrelated later edit")
        self.assertEqual(set(manifest["cases"]), {"marker", "csgo", "expense-tracker"})
        entry = manifest["cases"]["expense-tracker"]
        self.assertEqual(entry["runner"], "session")
        self.assertIn("task_sha256", entry)
        self.assertIn("initial_command_sha256", entry)
        case = self.root / "expense-tracker"
        self.assertEqual((case / "task.txt").read_text(), original)
        commands = list((case / "commands").glob("*.json"))
        self.assertEqual(len(commands), 1)
        self.assertEqual(json.loads(commands[0].read_text()), {"index": 1, "kind": "user", "text": original.strip()})
        code, out, err = self.plan("expense-tracker")
        self.assertEqual(code, 0, err)
        plan = json.loads(out)
        self.assertEqual(plan["command"][-1], original.strip())
        self.assertTrue(any("session_runner.patch.yml" in x for x in plan["command"]))

    def test_invalid_custom_arguments_reject_before_creating_root(self):
        variants = [
            ["--case-name", "expense"],
            ["--task-file", self.task],
            ["--case-name", "../outside", "--task-file", self.task],
            ["--case-name", "marker", "--task-file", self.task],
            ["--case-name", "configuration", "--task-file", self.task],
            ["--case-name", "expense", "--task-file", self.base / "missing"],
        ]
        for extra in variants:
            with self.subTest(extra=extra):
                code, _, _ = self.invoke(prepare, "--run-root", self.root, "--harness", self.harness, *extra)
                self.assertEqual(code, 2)
                self.assertFalse(self.root.exists())
        self.task.write_text(" \n")
        code, _, _ = self.invoke(prepare, "--run-root", self.root, "--harness", self.harness,
                                "--case-name", "expense", "--task-file", self.task)
        self.assertEqual(code, 2)
        self.assertFalse(self.root.exists())

    def test_modified_task_or_initial_queue_is_rejected(self):
        self.prepare("--case-name", "expense", "--task-file", self.task)
        for relative in ("task.txt", "commands/000001.json"):
            file = self.root / "expense" / relative
            original = file.read_bytes()
            file.write_bytes(original + b" ")
            with patch("subprocess.Popen") as spawn:
                code, _, err = self.plan("expense", "--execute")
                self.assertEqual(code, 2)
                self.assertIn("Prepared task changed", err)
                spawn.assert_not_called()
            file.write_bytes(original)

    def test_unknown_or_path_case_rejected(self):
        self.prepare()
        for case in ("missing", "../marker", "/tmp", "configuration"):
            with self.subTest(case=case), patch("subprocess.Popen") as spawn:
                code, _, _ = self.plan(case, "--execute")
                self.assertEqual(code, 2)
                spawn.assert_not_called()

    def test_explicit_launch_is_single_use_and_dry_run_remains_read_only(self):
        self.prepare("--case-name", "expense", "--task-file", self.task)
        with patch("subprocess.Popen") as spawn:
            spawn.return_value.pid = 12345
            spawn.return_value.wait.return_value = 0
            code, _, err = self.plan("expense", "--execute")
            self.assertEqual(code, 0, err)
            spawn.assert_called_once()
            self.assertEqual(spawn.call_args.kwargs["cwd"], self.root / "expense/workspace")
            self.assertEqual(spawn.call_args.kwargs["env"]["DSH_HOME"], str(self.root / "expense/home"))
            code, _, err = self.plan("expense", "--execute")
            self.assertEqual(code, 2)
            self.assertIn("already launched", err)
            spawn.assert_called_once()
            code, out, err = self.plan("expense")
            self.assertEqual(code, 0, err)
            self.assertFalse(json.loads(out)["execute"])
            spawn.assert_called_once()
        self.assertEqual(json.loads((self.root / "expense/exit.json").read_text())["exit_code"], 0)

    def test_legacy_default_manifest_does_not_need_rewriting(self):
        manifest = self.prepare()
        del manifest["cases"]
        (self.root / "manifest.json").write_text(json.dumps(manifest))
        for case in ("marker", "csgo"):
            code, _, err = self.plan(case)
            self.assertEqual(code, 0, err)
        code, _, _ = self.plan("expense")
        self.assertEqual(code, 2)

    def test_explicit_output_cap_changes_only_prepared_provider(self):
        manifest = self.prepare("--max-output-tokens", "16384")
        self.assertEqual(manifest["max_tokens"], 16384)
        self.assertEqual(manifest["native_g"], 4096)
        provider = (self.root / "configuration/provider.patch.yml").read_text()
        self.assertIn("maxTokens: 16384", provider)
        self.assertIn("contextWindow: 32768", provider)
        code, _, err = self.plan("csgo")
        self.assertEqual(code, 0, err)

    def test_invalid_output_cap_rejected_before_preparation(self):
        for cap in ("0", "-1", "16385", "1.5"):
            with self.subTest(cap=cap):
                code, _, _ = self.invoke(prepare, "--run-root", self.root, "--harness", self.harness,
                                        "--context-window", "16384", "--max-output-tokens", cap)
                self.assertEqual(code, 2)
                self.assertFalse(self.root.exists())

    def test_configuration_fingerprint_still_checked(self):
        self.prepare()
        with (self.root / "configuration/provider.patch.yml").open("a") as stream:
            stream.write("# changed\n")
        code, _, err = self.plan("csgo")
        self.assertEqual(code, 2)
        self.assertIn("Prepared configuration changed", err)


if __name__ == "__main__":
    unittest.main()
