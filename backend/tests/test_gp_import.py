from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(importlib.util.find_spec("guitarpro"), "pyguitarpro is not installed")
class GuitarProImportTests(unittest.TestCase):
    def test_imports_generated_gp5_into_draft_schema(self) -> None:
        from app.services.gp import create_guitar_pro_file
        from app.services.gp_import import import_guitar_pro_file

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "source.gp5"
            create_guitar_pro_file(
                {
                    "title": "Reference Tab",
                    "artist": "Human",
                    "tempo": 100,
                    "time_signature": "3/4",
                    "capo_fret": 2,
                    "tuning": "standard",
                    "guitar": {
                        "notes": [
                            {"pitch": "F#4", "start_beat": 0, "duration": 0.25, "string": 1, "fret": 0, "velocity": 100},
                            {"pitch": "A4", "start_beat": 1, "duration": 0.25, "string": 1, "fret": 3, "velocity": 100},
                        ]
                    },
                    "drums": [
                        {"drum": "kick", "start_beat": 0, "duration": 0.25, "velocity": 100},
                    ],
                },
                str(path),
            )

            draft = import_guitar_pro_file(path)

        self.assertEqual(draft["schema_version"], "tabforge-draft-v1")
        self.assertEqual(draft["metadata"]["title"], "Reference Tab")
        self.assertEqual(draft["constraints"]["time_signature"], "3/4")
        self.assertEqual(draft["constraints"]["capo_fret"], 2)
        self.assertEqual(draft["tracks"][0]["capo_fret"], 2)
        self.assertEqual(draft["tracks"][0]["notes"][0]["pitch"], "F#4")
        self.assertEqual(draft["tracks"][0]["notes"][0]["pitch_midi"], 66)
        self.assertEqual(draft["tracks"][0]["strings"][0]["value"], 64)
        self.assertEqual(draft["tracks"][0]["statistics"]["note_count"], 2)
        self.assertEqual(draft["tracks"][1]["name"], "drums")


if __name__ == "__main__":
    unittest.main()
