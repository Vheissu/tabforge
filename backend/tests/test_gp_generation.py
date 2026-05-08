from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


def _beat_duration_beats(duration) -> float:
    beats = 4 / int(duration.value)
    if getattr(duration, "isDotted", False):
        beats *= 1.5
    tuplet = getattr(duration, "tuplet", None)
    if tuplet and getattr(tuplet, "enters", 1) and getattr(tuplet, "times", 1):
        beats *= float(tuplet.times) / float(tuplet.enters)
    return beats


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
            self.assertEqual(song.tracks[0].channel.channel, 0)
            self.assertEqual(song.tracks[1].channel.channel, 2)
            self.assertEqual(song.tracks[1].channel.instrument, 33)
            self.assertEqual(song.tracks[2].channel.channel, 9)
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

    def test_pitched_track_preserves_multiple_strings_in_same_slot(self) -> None:
        from app.services.gp import create_guitar_pro_file

        notes = [
            {"pitch": "E4", "start_beat": 0, "duration": 1, "string": 1, "fret": 0, "velocity": 90},
            {"pitch": "B3", "start_beat": 0, "duration": 1, "string": 2, "fret": 0, "velocity": 85},
            {"pitch": "G3", "start_beat": 0, "duration": 1, "string": 3, "fret": 0, "velocity": 80},
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "chord.gp5"
            create_guitar_pro_file(
                {
                    "title": "Chord",
                    "artist": "Test",
                    "tempo": 120,
                    "tuning": "standard",
                    "guitar": {"notes": notes},
                },
                str(output_path),
            )

            import guitarpro

            song = guitarpro.parse(str(output_path))
            first_beat = song.tracks[0].measures[0].voices[0].beats[0]

        self.assertEqual(len(first_beat.notes), 3)

    def test_pitched_track_revoices_unpositioned_notes_from_pitch(self) -> None:
        from app.services.gp import create_guitar_pro_file

        notes = [
            {"pitch": "E4", "start_beat": 0, "duration": 1, "string": None, "fret": None, "velocity": 120},
            {"pitch": "B3", "start_beat": 0, "duration": 1, "string": 2, "fret": 0, "velocity": 90},
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "skip-invalid.gp5"
            create_guitar_pro_file(
                {
                    "title": "Skip Invalid",
                    "artist": "Test",
                    "tempo": 120,
                    "tuning": "standard",
                    "guitar": {"notes": notes},
                },
                str(output_path),
            )

            import guitarpro

            song = guitarpro.parse(str(output_path))
            first_beat = song.tracks[0].measures[0].voices[0].beats[0]

        self.assertEqual(len(first_beat.notes), 2)
        self.assertEqual(sorted((note.string, note.value) for note in first_beat.notes), [(1, 0), (2, 0)])

    def test_pitched_track_recomputes_stale_position_from_pitch(self) -> None:
        from app.services.gp import create_guitar_pro_file

        notes = [
            {"pitch": "E4", "pitch_midi": 64, "start_beat": 0, "duration": 1, "string": 1, "fret": 7, "velocity": 120},
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "revoice-stale.gp5"
            create_guitar_pro_file(
                {
                    "title": "Revoice Stale",
                    "artist": "Test",
                    "tempo": 120,
                    "tuning": "standard",
                    "guitar": {"notes": notes},
                },
                str(output_path),
            )

            import guitarpro

            song = guitarpro.parse(str(output_path))
            first_note = song.tracks[0].measures[0].voices[0].beats[0].notes[0]

        self.assertEqual(first_note.string, 1)
        self.assertEqual(first_note.value, 0)

    def test_drum_track_preserves_multiple_kit_pieces_in_same_slot(self) -> None:
        from app.services.gp import create_guitar_pro_file

        hits = [
            {"drum": "kick", "start_beat": 0, "duration": 0.25, "velocity": 100},
            {"drum": "snare", "start_beat": 0, "duration": 0.25, "velocity": 95},
            {"drum": "hihat_closed", "start_beat": 0, "duration": 0.25, "velocity": 80},
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "drum-stack.gp5"
            create_guitar_pro_file(
                {
                    "title": "Drum Stack",
                    "artist": "Test",
                    "tempo": 120,
                    "drums": hits,
                },
                str(output_path),
            )

            import guitarpro

            song = guitarpro.parse(str(output_path))
            first_beat = song.tracks[0].measures[0].voices[0].beats[0]

        self.assertEqual(len(first_beat.notes), 3)

    def test_drum_track_preserves_same_string_kit_collisions(self) -> None:
        from app.services.gp import create_guitar_pro_file

        hits = [
            {"drum": "snare", "start_beat": 0, "duration": 0.25, "velocity": 100},
            {"drum": "tom_low", "start_beat": 0, "duration": 0.25, "velocity": 95},
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "drum-collision.gp5"
            create_guitar_pro_file(
                {
                    "title": "Drum Collision",
                    "artist": "Test",
                    "tempo": 120,
                    "drums": hits,
                },
                str(output_path),
            )

            import guitarpro

            song = guitarpro.parse(str(output_path))
            first_beat = song.tracks[0].measures[0].voices[0].beats[0]

        self.assertEqual(sorted(note.value for note in first_beat.notes), [38, 43])

    def test_writes_non_four_four_time_signature(self) -> None:
        from app.services.gp import create_guitar_pro_file

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "three-four.gp5"
            create_guitar_pro_file(
                {
                    "title": "Three Four",
                    "artist": "Test",
                    "tempo": 120,
                    "time_signature": "3/4",
                    "tuning": "standard",
                    "guitar": {
                        "notes": [
                            {"pitch": "E4", "start_beat": 0, "duration": 0.25, "string": 1, "fret": 0, "velocity": 100},
                            {"pitch": "G4", "start_beat": 3, "duration": 0.25, "string": 1, "fret": 3, "velocity": 100},
                        ]
                    },
                },
                str(output_path),
            )

            import guitarpro

            song = guitarpro.parse(str(output_path))

        self.assertEqual(song.measureHeaders[0].timeSignature.numerator, 3)
        self.assertEqual(len(song.tracks[0].measures), 2)
        self.assertEqual(
            sum(_beat_duration_beats(beat.duration) for beat in song.tracks[0].measures[0].voices[0].beats),
            3,
        )

    def test_writes_pickup_bar_and_capo(self) -> None:
        from app.services.gp import create_guitar_pro_file

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "pickup-capo.gp5"
            create_guitar_pro_file(
                {
                    "title": "Pickup Capo",
                    "artist": "Test",
                    "tempo": 120,
                    "time_signature": "4/4",
                    "pickup_bar_beats": 1.0,
                    "capo_fret": 2,
                    "tuning": "standard",
                    "guitar": {
                        "notes": [
                            {"pitch": "F#4", "start_beat": 0, "duration": 0.25, "string": 1, "fret": 0, "velocity": 100},
                            {"pitch": "A4", "start_beat": 1, "duration": 0.25, "string": 1, "fret": 3, "velocity": 100},
                        ]
                    },
                },
                str(output_path),
            )

            import guitarpro

            song = guitarpro.parse(str(output_path))

        self.assertEqual(song.measureHeaders[0].timeSignature.numerator, 1)
        self.assertEqual(song.measureHeaders[0].timeSignature.denominator.value, 4)
        self.assertEqual(song.tracks[0].offset, 2)
        self.assertEqual(
            sum(_beat_duration_beats(beat.duration) for beat in song.tracks[0].measures[0].voices[0].beats),
            1,
        )

    def test_pitched_track_preserves_note_duration(self) -> None:
        from app.services.gp import create_guitar_pro_file

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "duration.gp5"
            create_guitar_pro_file(
                {
                    "title": "Duration",
                    "artist": "Test",
                    "tempo": 120,
                    "tuning": "standard",
                    "guitar": {
                        "notes": [
                            {"pitch": "E4", "start_beat": 0, "duration": 1, "string": 1, "fret": 0, "velocity": 100},
                            {"pitch": "G4", "start_beat": 2, "duration": 0.5, "string": 1, "fret": 3, "velocity": 100},
                        ]
                    },
                },
                str(output_path),
            )

            import guitarpro

            song = guitarpro.parse(str(output_path))
            beats = song.tracks[0].measures[0].voices[0].beats

        self.assertEqual(_beat_duration_beats(beats[0].duration), 1)
        self.assertEqual(_beat_duration_beats(beats[2].duration), 0.5)

    def test_pitched_track_round_trips_tied_sustain_duration(self) -> None:
        from app.services.gp import create_guitar_pro_file
        from app.services.gp_import import import_guitar_pro_file

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "tied-duration.gp5"
            create_guitar_pro_file(
                {
                    "title": "Tied Duration",
                    "artist": "Test",
                    "tempo": 120,
                    "time_signature": "4/4",
                    "tuning": "standard",
                    "guitar": {
                        "notes": [
                            {"pitch": "E4", "start_beat": 0, "duration": 1.25, "string": 1, "fret": 0, "velocity": 100},
                            {"pitch": "G4", "start_beat": 3.5, "duration": 1.0, "string": 1, "fret": 3, "velocity": 100},
                        ]
                    },
                },
                str(output_path),
            )

            draft = import_guitar_pro_file(output_path)

        imported_notes = draft["tracks"][0]["notes"]
        self.assertEqual(len(imported_notes), 2)
        self.assertAlmostEqual(imported_notes[0]["duration"], 1.25)
        self.assertAlmostEqual(imported_notes[1]["duration"], 1.0)

    def test_pitched_track_preserves_overlapping_different_string_sustain(self) -> None:
        from app.services.gp import create_guitar_pro_file
        from app.services.gp_import import import_guitar_pro_file

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "overlap.gp5"
            create_guitar_pro_file(
                {
                    "title": "Overlap",
                    "artist": "Test",
                    "tempo": 120,
                    "time_signature": "4/4",
                    "tuning": "standard",
                    "guitar": {
                        "notes": [
                            {"pitch": "E4", "start_beat": 0, "duration": 2, "string": 1, "fret": 0, "velocity": 100},
                            {"pitch": "B3", "start_beat": 1, "duration": 1, "string": 2, "fret": 0, "velocity": 100},
                        ]
                    },
                },
                str(output_path),
            )

            draft = import_guitar_pro_file(output_path)

        imported = sorted(draft["tracks"][0]["notes"], key=lambda note: (note["start_beat"], note["pitch_midi"]))
        self.assertEqual(len(imported), 2)
        self.assertEqual(imported[0]["pitch"], "E4")
        self.assertAlmostEqual(imported[0]["duration"], 2)
        self.assertEqual(imported[1]["pitch"], "B3")
        self.assertAlmostEqual(imported[1]["duration"], 1)

    def test_triplet_feel_preserves_triplet_start(self) -> None:
        from app.services.gp import create_guitar_pro_file

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "triplet.gp5"
            create_guitar_pro_file(
                {
                    "title": "Triplet",
                    "artist": "Test",
                    "tempo": 120,
                    "tuning": "standard",
                    "triplet_feel": True,
                    "guitar": {
                        "notes": [
                            {
                                "pitch": "E4",
                                "start_beat": 1 / 3,
                                "duration": 1 / 3,
                                "string": 1,
                                "fret": 0,
                                "velocity": 100,
                            },
                        ]
                    },
                },
                str(output_path),
            )

            import guitarpro

            song = guitarpro.parse(str(output_path))
            beats = song.tracks[0].measures[0].voices[0].beats

        self.assertAlmostEqual(_beat_duration_beats(beats[0].duration), 1 / 3)
        self.assertAlmostEqual(_beat_duration_beats(beats[1].duration), 1 / 3)
        self.assertEqual(len(beats[1].notes), 1)

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
