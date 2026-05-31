import tempfile
import unittest
from pathlib import Path

from notify_when_done import collect_summary, load_env_file


class NotifyWhenDoneTest(unittest.TestCase):
    def test_load_env_file_parses_quoted_values_and_comments(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / "brevo.env"
            env_path.write_text(
                'BREVO_API_KEY="secret-value"\n'
                "# ignored\n"
                "SPO_NOTIFY_FROM=notify@example.com\n"
                "SPO_NOTIFY_TO='user@example.com'\n"
            )

            env = load_env_file(env_path)

            self.assertEqual(env["BREVO_API_KEY"], "secret-value")
            self.assertEqual(env["SPO_NOTIFY_FROM"], "notify@example.com")
            self.assertEqual(env["SPO_NOTIFY_TO"], "user@example.com")

    def test_collect_summary_counts_results_and_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_dir = root / "res"
            log_dir = root / "logs"
            result_dir.mkdir()
            log_dir.mkdir()
            (result_dir / "a.csv").write_text("x\n1\n2\n")
            (result_dir / "b.csv").write_text("x\n1\n")
            (log_dir / "ok.log").write_text("Loading fixed synthetic data\n")
            (log_dir / "bad.log").write_text("Traceback: failed\n")

            summary = collect_summary(result_dir, log_dir)

            self.assertEqual(summary["csv_files"], 2)
            self.assertEqual(summary["result_rows"], 3)
            self.assertEqual(summary["error_hits"], 1)
            self.assertIn("bad.log", summary["error_logs"])


if __name__ == "__main__":
    unittest.main()
