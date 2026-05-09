from __future__ import annotations

import unittest


class AsciiTabTests(unittest.TestCase):
    def test_converts_six_line_ascii_tab_to_reference_draft(self) -> None:
        from app.services.ascii_tab import ascii_tab_to_draft

        draft = ascii_tab_to_draft(
            """
[Intro]
e|---------|
B|---------|
G|-2---0h2-|
D|---3-----|
A|---------|
E|-0---1---|
""",
            title="Reference",
            artist="Fixture",
            tempo=152,
            key="A",
            columns_per_beat=4,
        )

        notes = draft["tracks"][0]["notes"]

        self.assertEqual(draft["schema_version"], "tabforge-draft-v1")
        self.assertEqual(draft["metadata"]["tempo"], 152)
        self.assertEqual(draft["metadata"]["key"], "A")
        self.assertEqual(draft["tuning"]["name"], "standard")
        self.assertEqual(draft["tracks"][0]["statistics"]["note_count"], 6)
        self.assertIn({"string": 6, "fret": 0, "pitch_midi": 40, "start_beat": 0.25}, [_compact(note) for note in notes])
        self.assertIn({"string": 4, "fret": 3, "pitch_midi": 53, "start_beat": 0.75}, [_compact(note) for note in notes])
        self.assertIn({"string": 3, "fret": 2, "pitch_midi": 57, "start_beat": 0.25}, [_compact(note) for note in notes])
        self.assertEqual(notes[-1]["technique"], "hammer_on")

    def test_ignores_parenthesized_bend_targets(self) -> None:
        from app.services.ascii_tab import ascii_tab_to_draft

        draft = ascii_tab_to_draft(
            """
e|---------|
B|-5b(7)r5-|
G|---------|
D|---------|
A|---------|
E|---------|
""",
            columns_per_beat=4,
        )

        notes = draft["tracks"][0]["notes"]

        self.assertEqual([note["fret"] for note in notes], [5, 5])
        self.assertEqual([note["technique"] for note in notes], ["bend", "release"])

    def test_ignores_incomplete_tab_blocks(self) -> None:
        from app.services.ascii_tab import ascii_tab_to_draft

        draft = ascii_tab_to_draft(
            """
e|---0---|
B|---1---|
""",
        )

        self.assertEqual(draft["tracks"][0]["notes"], [])
        self.assertEqual(draft["sources"]["ascii_tab"]["block_count"], 0)

    def test_rejects_invalid_column_scale(self) -> None:
        from app.services.ascii_tab import ascii_tab_to_draft

        with self.assertRaises(ValueError):
            ascii_tab_to_draft("", columns_per_beat=0)


def _compact(note: dict) -> dict:
    return {
        "string": note["string"],
        "fret": note["fret"],
        "pitch_midi": note["pitch_midi"],
        "start_beat": note["start_beat"],
    }


if __name__ == "__main__":
    unittest.main()
