from __future__ import annotations

import unittest

from app.services.fretboard import get_all_positions, get_tuning_midi, midi_to_note_name, note_to_midi


class FretboardTests(unittest.TestCase):
    def test_round_trips_midi_note_names(self) -> None:
        for midi_value in [28, 40, 52, 64, 76]:
            self.assertEqual(note_to_midi(midi_to_note_name(midi_value)), midi_value)

    def test_standard_guitar_positions_high_e(self) -> None:
        positions = get_all_positions(note_to_midi("E4"))
        self.assertEqual(positions[-1].string, 1)
        self.assertEqual(positions[-1].fret, 0)

    def test_drop_d_only_changes_low_guitar_string(self) -> None:
        self.assertEqual(get_tuning_midi("drop_d"), [38, 45, 50, 55, 59, 64])
        self.assertEqual(get_tuning_midi("drop_d", is_bass=True), [28, 33, 38, 43])


if __name__ == "__main__":
    unittest.main()
