from __future__ import annotations

import unittest


class FretboardTests(unittest.TestCase):
    def test_bass_positions_use_four_string_numbering(self) -> None:
        from app.services.fretboard import BASS_STANDARD_TUNING, get_all_positions

        positions = get_all_positions(28, BASS_STANDARD_TUNING)

        self.assertTrue(positions)
        self.assertTrue(all(1 <= position.string <= 4 for position in positions))


if __name__ == "__main__":
    unittest.main()
