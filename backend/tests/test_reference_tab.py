from __future__ import annotations

import unittest


class ReferenceTabTests(unittest.TestCase):
    def test_builds_reference_tab_draft_with_audio_metadata(self) -> None:
        from app.services.reference_tab import build_reference_tab_draft

        draft = build_reference_tab_draft(
            {
                "ascii_tab": """
e|-------|
B|-------|
G|-2-0h2-|
D|-------|
A|-------|
E|-0---1-|
""",
                "columns_per_beat": 4,
            },
            metadata={"title": "Video Title", "artist": "Uploader"},
            tempo=152,
            key="A",
            tuning="standard",
            constraints={"time_signature": "4/4", "time_signature_source": "user", "capo_fret": 0},
        )

        warning_codes = {warning["code"] for warning in draft["quality"]["warnings"]}

        self.assertEqual(draft["metadata"]["title"], "Video Title")
        self.assertEqual(draft["metadata"]["tempo"], 152)
        self.assertEqual(draft["metadata"]["key"], "A")
        self.assertEqual(draft["tracks"][0]["name"], "guitar")
        self.assertGreater(draft["tracks"][0]["statistics"]["note_count"], 0)
        self.assertEqual(draft["sources"]["youtube"]["artist"], "Uploader")
        self.assertIn("reference_tab_used", warning_codes)

    def test_empty_reference_payload_is_rejected(self) -> None:
        from app.services.reference_tab import build_reference_tab_draft, has_reference_tab

        self.assertFalse(has_reference_tab({"ascii_tab": "  "}))
        with self.assertRaises(ValueError):
            build_reference_tab_draft(
                {"ascii_tab": ""},
                metadata={},
                tempo=120,
                key=None,
                tuning="standard",
                constraints={},
            )


if __name__ == "__main__":
    unittest.main()
