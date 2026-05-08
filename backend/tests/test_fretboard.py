from __future__ import annotations

import unittest


class FretboardTests(unittest.TestCase):
    def test_note_to_midi_accepts_flat_names(self) -> None:
        from app.services.fretboard import note_to_midi

        self.assertEqual(note_to_midi("Bb3"), note_to_midi("A#3"))

    def test_bass_positions_use_four_string_numbering(self) -> None:
        from app.services.fretboard import BASS_STANDARD_TUNING, get_all_positions

        positions = get_all_positions(28, BASS_STANDARD_TUNING)

        self.assertTrue(positions)
        self.assertTrue(all(1 <= position.string <= 4 for position in positions))

    def test_drop_d_lowers_bass_string_too(self) -> None:
        from app.services.fretboard import get_tuning_midi

        self.assertEqual(get_tuning_midi("drop_d", is_bass=True)[0], 26)

    def test_chord_voicing_uses_unique_strings_when_available(self) -> None:
        from app.services.fretboard import Note, optimize_positions

        notes = [
            Note("E4", start_beat=0, duration=1),
            Note("B3", start_beat=0, duration=1),
            Note("G3", start_beat=0, duration=1),
        ]

        voiced = optimize_positions(notes)
        strings = [note.position.string for note in voiced if note.position is not None]

        self.assertEqual(len(strings), 3)
        self.assertEqual(len(set(strings)), 3)


if __name__ == "__main__":
    unittest.main()
