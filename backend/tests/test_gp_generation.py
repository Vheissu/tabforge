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
                    },
                    {
                        "pitch": "G4",
                        "start_beat": 4,
                        "duration": 1,
                        "string": 1,
                        "fret": 3,
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
                    },
                    {
                        "pitch": "A2",
                        "start_beat": 4,
                        "duration": 1,
                        "string": 3,
                        "fret": 2,
                        "velocity": 100,
                    },
                    {
                        "pitch": "C3",
                        "start_beat": 4,
                        "duration": 1,
                        "string": 9,
                        "fret": 40,
                        "velocity": 90,
                    }
                ]
            },
            "drums": [
                {
                    "drum": "kick",
                    "start_beat": 0,
                    "duration": 1,
                    "velocity": 100,
                },
                {
                    "drum": "snare",
                    "start_beat": 4,
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

            import guitarpro

            song = guitarpro.parse(str(output_path))
            track_names = [track.name for track in song.tracks]
            self.assertEqual(track_names, ["Guitar", "Bass", "Drums"])
            self.assertTrue(all(len(track.measures) == 2 for track in song.tracks))
            note_count = sum(
                len(beat.notes)
                for track in song.tracks
                for measure in track.measures
                for voice in measure.voices
                for beat in voice.beats
            )
            self.assertGreater(note_count, 0)

    def test_long_bass_track_with_duplicate_slots_parses_back(self) -> None:
        from app.services.gp import create_guitar_pro_file

        notes = []
        for measure in range(80):
            start_beat = measure * 4 + 3.5
            notes.extend(
                [
                    {
                        "pitch": "A2",
                        "start_beat": start_beat,
                        "duration": 0.25,
                        "string": 9,
                        "fret": 40,
                        "velocity": 80,
                    },
                    {
                        "pitch": "C3",
                        "start_beat": start_beat,
                        "duration": 0.25,
                        "string": 3,
                        "fret": 2,
                        "velocity": 100,
                    },
                ]
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "long-bass.gp5"
            create_guitar_pro_file(
                {
                    "title": "Long Bass",
                    "artist": "TabForge",
                    "tempo": 120,
                    "tuning": "standard",
                    "bass": {"notes": notes},
                },
                str(output_path),
            )

            import guitarpro

            song = guitarpro.parse(str(output_path))

        self.assertEqual([track.name for track in song.tracks], ["Bass"])
        self.assertEqual(len(song.tracks[0].measures), 80)

    def test_long_drum_track_with_duplicate_slots_parses_back(self) -> None:
        from app.services.gp import create_guitar_pro_file

        hits = []
        for measure in range(80):
            start_beat = measure * 4 + 0.5
            hits.extend(
                [
                    {"drum": "kick", "start_beat": start_beat, "duration": 0.25, "velocity": 80},
                    {"drum": "snare", "start_beat": start_beat, "duration": 0.25, "velocity": 100},
                ]
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "long-drums.gp5"
            create_guitar_pro_file(
                {
                    "title": "Long Drums",
                    "artist": "TabForge",
                    "tempo": 120,
                    "drums": hits,
                },
                str(output_path),
            )

            import guitarpro

            song = guitarpro.parse(str(output_path))

        self.assertEqual([track.name for track in song.tracks], ["Drums"])
        self.assertEqual(len(song.tracks[0].measures), 80)


if __name__ == "__main__":
    unittest.main()
