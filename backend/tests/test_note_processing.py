from __future__ import annotations

import unittest


class NoteProcessingTests(unittest.TestCase):
    def test_filters_notes_outside_playable_guitar_range(self) -> None:
        from app.services.note_processing import prepare_note_events_for_tab

        events = [
            {"start_time": 0, "end_time": 0.3, "pitch_midi": 20, "velocity": 120},
            {"start_time": 0, "end_time": 0.3, "pitch_midi": 64, "velocity": 120},
        ]

        notes = prepare_note_events_for_tab(events, "guitar", 120, "standard")

        self.assertEqual([note.pitch_midi for note in notes], [64])

    def test_bass_keeps_one_low_confident_note_per_grid_slot(self) -> None:
        from app.services.note_processing import prepare_note_events_for_tab

        events = [
            {"start_time": 0.0, "end_time": 0.2, "pitch_midi": 52, "velocity": 105},
            {"start_time": 0.02, "end_time": 0.2, "pitch_midi": 40, "velocity": 80},
            {"start_time": 0.03, "end_time": 0.2, "pitch_midi": 45, "velocity": 95},
        ]

        notes = prepare_note_events_for_tab(events, "bass", 120, "standard")

        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0].pitch_midi, 40)

    def test_merges_repeated_notes_across_tiny_gaps(self) -> None:
        from app.services.note_processing import prepare_note_events_for_tab

        events = [
            {"start_time": 0.0, "end_time": 0.24, "pitch_midi": 64, "velocity": 90},
            {"start_time": 0.26, "end_time": 0.5, "pitch_midi": 64, "velocity": 100},
        ]

        notes = prepare_note_events_for_tab(events, "guitar", 120, "standard")

        self.assertEqual(len(notes), 1)
        self.assertGreaterEqual(notes[0].duration, 1.0)
        self.assertEqual(notes[0].velocity, 100)

    def test_triplet_feel_uses_triplet_grid(self) -> None:
        from app.services.note_processing import prepare_note_events_for_tab

        events = [
            {"start_time": 0.16, "end_time": 0.32, "pitch_midi": 64, "velocity": 90},
        ]

        notes = prepare_note_events_for_tab(events, "guitar", 120, "standard", triplet_feel=True)

        self.assertAlmostEqual(notes[0].start_beat, 1 / 3)

    def test_capo_filters_notes_below_capo_adjusted_open_string(self) -> None:
        from app.services.note_processing import prepare_note_events_for_tab

        events = [
            {"start_time": 0, "end_time": 0.3, "pitch_midi": 40, "velocity": 120},
            {"start_time": 0, "end_time": 0.3, "pitch_midi": 42, "velocity": 120},
        ]

        notes = prepare_note_events_for_tab(events, "guitar", 120, "standard", capo_fret=2)

        self.assertEqual([note.pitch_midi for note in notes], [42])


if __name__ == "__main__":
    unittest.main()
