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

    def test_surfaces_track_analysis_warnings(self) -> None:
        from app.services.draft import build_tab_draft, summarise_draft

        draft = build_tab_draft(
            {
                "title": "Analysis",
                "constraints": {},
                "source_stems": {"guitar": "guitar.wav"},
                "guitar": {
                    "notes": [{"pitch": "E4", "pitch_midi": 64, "start_beat": 0, "duration": 1}],
                    "analysis": {
                        "raw_event_count": 3,
                        "final_note_count": 1,
                        "dropped_unpositioned_count": 2,
                        "refinement_status": "applied",
                    },
                },
            },
            ["guitar"],
        )
        summary = summarise_draft(draft)
        warning_codes = {warning["code"] for warning in draft["quality"]["warnings"]}

        self.assertIn("guitar_dropped_unpositioned", warning_codes)
        self.assertIn("guitar_refinement_applied", warning_codes)
        self.assertEqual(summary["tracks"][0]["analysis"]["raw_event_count"], 3)

    def test_warns_when_detected_tuning_has_low_confidence(self) -> None:
        from app.services.draft import build_tab_draft, summarise_draft

        draft = build_tab_draft(
            {
                "title": "Weak Tuning",
                "constraints": {},
                "tuning": "standard",
                "tuning_info": {"tuning": "standard", "confidence": 0.2},
                "source_stems": {"guitar": "guitar.wav"},
                "guitar": {"notes": []},
            },
            ["guitar"],
        )
        summary = summarise_draft(draft)
        warning_codes = {warning["code"] for warning in draft["quality"]["warnings"]}

        self.assertIn("low_confidence_tuning", warning_codes)
        self.assertTrue(any("Set tuning manually" in action for action in summary["quality"]["next_actions"]))

    def test_summarises_draft_without_note_payloads(self) -> None:
        from app.services.draft import summarise_draft

        draft = {
            "schema_version": "tabforge-draft-v1",
            "metadata": {"title": "Summary"},
            "constraints": {"tempo_source": "detected", "time_signature_source": "auto"},
            "tuning": {"name": "standard"},
            "sources": {"stems": {"guitar": "guitar.wav"}},
            "tracks": [
                {
                    "name": "guitar",
                    "source_stem": "guitar.wav",
                    "statistics": {"note_count": 2, "chord_slot_count": 1, "last_beat": 4},
                    "notes": [{"pitch": "E4"}, {"pitch": "G4"}],
                }
            ],
            "quality": {"warnings": [{"code": "detected_tempo", "severity": "info", "message": "Tempo was inferred."}]},
        }

        summary = summarise_draft(draft)

        self.assertEqual(summary["statistics"]["note_count"], 2)
        self.assertEqual(summary["tracks"][0]["source_stem"], "guitar.wav")
        self.assertNotIn("notes", summary["tracks"][0])
        self.assertTrue(summary["quality"]["next_actions"])

    def test_applies_corrections_and_revoices_pitched_tracks(self) -> None:
        from app.services.draft import apply_draft_corrections, draft_to_transcription

        draft = {
            "schema_version": "tabforge-draft-v1",
            "metadata": {"title": "Correct Me", "artist": "Band", "tempo": 118},
            "constraints": {"time_signature": "4/4", "tempo_source": "detected", "capo_fret": 0},
            "tuning": {"name": "standard"},
            "tracks": [
                {
                    "name": "guitar",
                    "source_stem": "guitar.wav",
                    "statistics": {},
                    "notes": [
                        {"pitch": "E4", "start_beat": 0, "duration": 0.25, "string": 1, "fret": 0, "velocity": 20},
                        {"pitch": "F#4", "start_beat": 0.25, "duration": 0.25, "string": 1, "fret": 2, "velocity": 100},
                    ],
                }
            ],
            "quality": {"warnings": []},
        }

        corrected = apply_draft_corrections(
            draft,
            {
                "tempo_bpm": 120,
                "time_signature": "3/4",
                "capo_fret": 2,
                "tracks": {"guitar": {"min_velocity": 50}},
            },
        )
        transcription = draft_to_transcription(corrected)

        self.assertEqual(corrected["metadata"]["tempo"], 120)
        self.assertEqual(corrected["constraints"]["time_signature"], "3/4")
        self.assertEqual(corrected["tracks"][0]["statistics"]["note_count"], 1)
        self.assertEqual(corrected["tracks"][0]["notes"][0]["fret"], 0)
        self.assertEqual(transcription["tempo"], 120)
        self.assertEqual(transcription["time_signature"], "3/4")
        self.assertEqual(transcription["capo_fret"], 2)


if __name__ == "__main__":
    unittest.main()
