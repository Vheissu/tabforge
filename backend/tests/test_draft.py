from __future__ import annotations

import unittest


class DraftTests(unittest.TestCase):
    def test_builds_versioned_draft_with_track_stats(self) -> None:
        from app.services.draft import build_tab_draft

        draft = build_tab_draft(
            {
                "title": "Draft Song",
                "artist": "TabForge",
                "tempo": 120,
                "detected_tempo": 118,
                "key": "E",
                "constraints": {"tempo_source": "user", "time_signature_source": "user"},
                "tuning": "standard",
                "source_stems": {"guitar": "guitar.wav", "bass": "bass.wav"},
                "guitar": {
                    "notes": [
                        {"start_beat": 0, "duration": 0.25, "pitch": "E4", "string": 1, "fret": 0},
                        {"start_beat": 0, "duration": 0.25, "pitch": "B3", "string": 2, "fret": 0},
                    ]
                },
                "bass": {"notes": [{"start_beat": 1, "duration": 1, "pitch": "E2", "string": 4, "fret": 0}]},
            },
            ["guitar", "bass"],
        )

        self.assertEqual(draft["schema_version"], "tabforge-draft-v1")
        self.assertEqual(draft["metadata"]["title"], "Draft Song")
        self.assertEqual(draft["tracks"][0]["source_stem"], "guitar.wav")
        self.assertEqual(draft["tracks"][0]["statistics"]["note_count"], 2)
        self.assertEqual(draft["tracks"][0]["statistics"]["chord_slot_count"], 1)

    def test_warns_when_guitar_uses_mixed_other_stem(self) -> None:
        from app.services.draft import build_tab_draft

        draft = build_tab_draft(
            {
                "title": "Mixed Stem",
                "constraints": {"tempo_source": "detected", "time_signature_source": "auto"},
                "source_stems": {"other": "other.wav", "bass": "bass.wav"},
                "guitar": {"notes": []},
            },
            ["guitar"],
        )

        warning_codes = {warning["code"] for warning in draft["quality"]["warnings"]}

        self.assertIn("mixed_guitar_stem", warning_codes)
        self.assertIn("detected_tempo", warning_codes)
        self.assertIn("assumed_meter", warning_codes)


if __name__ == "__main__":
    unittest.main()
