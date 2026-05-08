from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(importlib.util.find_spec("guitarpro"), "pyguitarpro is not installed")
class GuitarProGenerationTests(unittest.TestCase):
    def test_writes_gp5_with_guitar_bass_and_drums(self) -> None:
        from app.services.gp import create_guitar_pro_file

        transcription = {
            "title": "Smoke Test",
            "artist": "TabForge",
            "tempo": 120,
            "tuning": "standard",
            "guitar": {
                "notes": [
                    {
                        "pitch": "E4",
                        "start_beat": 0,
                        "duration": 1,
                        "string": 1,
                        "fret": 0,
                        "velocity": 100,
                    }
                ]
            },
            "bass": {
                "notes": [
                    {
                        "pitch": "E2",
                        "start_beat": 0,
                        "duration": 1,
                        "string": 4,
                        "fret": 0,
                        "velocity": 100,
                    }
                ]
            },
            "drums": [
                {
                    "drum": "kick",
                    "start_beat": 0,
                    "duration": 1,
                    "velocity": 100,
                }
            ],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "smoke.gp5"
            create_guitar_pro_file(transcription, str(output_path))
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
