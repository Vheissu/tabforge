from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class BenchmarkAccuracyTests(unittest.TestCase):
    def test_manifest_runner_reports_case_gaps_and_averages(self) -> None:
        from app.tools.benchmark_accuracy import main

        reference = {
            "metadata": {"tempo": 120},
            "constraints": {"time_signature": "4/4", "capo_fret": 0},
            "tuning": {"name": "standard"},
            "tracks": [
                {
                    "name": "guitar",
                    "notes": [{"pitch_midi": 64, "start_beat": 0, "duration": 1, "string": 1, "fret": 0}],
                }
            ],
        }
        bad_candidate = {
            "metadata": {"tempo": 120},
            "constraints": {"time_signature": "4/4", "capo_fret": 0},
            "tuning": {"name": "standard"},
            "tracks": [
                {
                    "name": "guitar",
                    "notes": [{"pitch_midi": 64, "start_beat": 0, "duration": 1, "string": 2, "fret": 5}],
                }
            ],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reference_path = root / "reference.draft.json"
            candidate_path = root / "candidate.draft.json"
            manifest_path = root / "manifest.json"
            report_path = root / "report.json"
            reference_path.write_text(json.dumps(reference), encoding="utf-8")
            candidate_path.write_text(json.dumps(bad_candidate), encoding="utf-8")
            manifest_path.write_text(
                json.dumps({"cases": [{"name": "wrong-fret", "reference": reference_path.name, "candidate": candidate_path.name}]}),
                encoding="utf-8",
            )

            output = io.StringIO()
            with patch(
                "sys.argv",
                [
                    "benchmark_accuracy",
                    "--manifest",
                    str(manifest_path),
                    "--output",
                    str(report_path),
                    "--min-f1",
                    "0.9",
                    "--min-strict-accuracy",
                    "0.9",
                ],
            ):
                with contextlib.redirect_stdout(output):
                    exit_code = main()

            result = json.loads(output.getvalue())
            written = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 1)
        self.assertFalse(result["passed"])
        self.assertEqual(result["case_count"], 1)
        self.assertEqual(result["average_f1"], 1)
        self.assertLess(result["average_strict_accuracy"], 0.9)
        self.assertEqual(result["cases"][0]["name"], "wrong-fret")
        self.assertEqual(result["cases"][0]["summary"]["position_mismatch_count"], 1)
        self.assertTrue(any(gap["metric"] == "position" for gap in result["cases"][0]["gaps"]))
        self.assertEqual(written["cases"][0]["name"], "wrong-fret")

    def test_manifest_runner_accepts_case_list(self) -> None:
        from app.tools.benchmark_accuracy import main

        draft = {
            "metadata": {"tempo": 120},
            "constraints": {"time_signature": "4/4", "capo_fret": 0},
            "tuning": {"name": "standard"},
            "tracks": [{"name": "guitar", "notes": [{"pitch_midi": 64, "start_beat": 0, "duration": 1}]}],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reference_path = root / "reference.draft.json"
            candidate_path = root / "candidate.draft.json"
            manifest_path = root / "manifest.json"
            reference_path.write_text(json.dumps(draft), encoding="utf-8")
            candidate_path.write_text(json.dumps(draft), encoding="utf-8")
            manifest_path.write_text(
                json.dumps([{"reference": reference_path.name, "candidate": candidate_path.name}]),
                encoding="utf-8",
            )

            output = io.StringIO()
            with patch("sys.argv", ["benchmark_accuracy", "--manifest", str(manifest_path)]):
                with contextlib.redirect_stdout(output):
                    exit_code = main()

            result = json.loads(output.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertTrue(result["passed"])
        self.assertEqual(result["cases"][0]["name"], "case-1")


if __name__ == "__main__":
    unittest.main()
