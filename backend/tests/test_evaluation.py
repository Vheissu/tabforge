from __future__ import annotations

import unittest


class EvaluationTests(unittest.TestCase):
    def test_scores_matching_notes_by_track_pitch_and_timing(self) -> None:
        from app.services.evaluation import compare_drafts

        reference = {
            "tracks": [
                {
                    "name": "guitar",
                    "notes": [
                        {"pitch_midi": 64, "start_beat": 0},
                        {"pitch_midi": 67, "start_beat": 1},
                    ],
                }
            ]
        }
        candidate = {
            "tracks": [
                {
                    "name": "guitar",
                    "notes": [
                        {"pitch_midi": 64, "start_beat": 0.1},
                        {"pitch_midi": 69, "start_beat": 1},
                    ],
                }
            ]
        }

        result = compare_drafts(reference, candidate, tolerance_beats=0.25)

        self.assertEqual(result["tracks"]["guitar"]["matches"], 1)
        self.assertEqual(result["tracks"]["guitar"]["precision"], 0.5)
        self.assertEqual(result["tracks"]["guitar"]["recall"], 0.5)

    def test_scores_drum_notes_by_drum_type(self) -> None:
        from app.services.evaluation import compare_drafts

        reference = {"tracks": [{"name": "drums", "notes": [{"drum": "kick", "start_beat": 0}]}]}
        candidate = {"tracks": [{"name": "drums", "notes": [{"drum": "snare", "start_beat": 0}]}]}

        result = compare_drafts(reference, candidate)

        self.assertEqual(result["overall"]["matches"], 0)

    def test_matches_pitch_name_against_imported_midi_pitch(self) -> None:
        from app.services.evaluation import compare_drafts

        reference = {"tracks": [{"name": "guitar", "notes": [{"pitch_midi": 70, "start_beat": 0}]}]}
        candidate = {"tracks": [{"name": "guitar", "notes": [{"pitch": "Bb4", "start_beat": 0.05}]}]}

        result = compare_drafts(reference, candidate, tolerance_beats=0.25)

        self.assertEqual(result["overall"]["matches"], 1)

    def test_aggregates_duplicate_track_names(self) -> None:
        from app.services.evaluation import compare_drafts

        reference = {
            "tracks": [
                {"name": "guitar", "notes": [{"pitch_midi": 64, "start_beat": 0}]},
                {"name": "guitar", "notes": [{"pitch_midi": 67, "start_beat": 1}]},
            ]
        }
        candidate = {
            "tracks": [
                {"name": "guitar", "notes": [{"pitch_midi": 64, "start_beat": 0}]},
                {"name": "guitar", "notes": [{"pitch_midi": 67, "start_beat": 1}]},
            ]
        }

        result = compare_drafts(reference, candidate)

        self.assertEqual(result["tracks"]["guitar"]["matches"], 2)

    def test_strict_metrics_score_duration_position_technique_and_metadata(self) -> None:
        from app.services.evaluation import compare_drafts

        reference = {
            "metadata": {"tempo": 120},
            "constraints": {"time_signature": "4/4", "capo_fret": 2},
            "tuning": {"name": "standard"},
            "tracks": [
                {
                    "name": "guitar",
                    "notes": [
                        {
                            "pitch_midi": 64,
                            "start_beat": 0,
                            "duration": 1,
                            "string": 1,
                            "fret": 0,
                            "technique": "slide_up",
                        }
                    ],
                }
            ],
        }
        candidate = {
            "metadata": {"tempo": 120},
            "constraints": {"time_signature": "4/4", "capo_fret": 2},
            "tuning": {"name": "standard"},
            "tracks": [
                {
                    "name": "guitar",
                    "notes": [
                        {
                            "pitch_midi": 64,
                            "start_beat": 0.1,
                            "duration": 1.1,
                            "string": 1,
                            "fret": 0,
                            "technique": "slide_up",
                        }
                    ],
                }
            ],
        }

        result = compare_drafts(reference, candidate, strict=True)

        self.assertEqual(result["overall"]["f1"], 1)
        self.assertEqual(result["duration"]["accuracy"], 1)
        self.assertEqual(result["timing"]["max_delta_beats"], 0.1)
        self.assertEqual(result["position"]["accuracy"], 1)
        self.assertEqual(result["technique"]["accuracy"], 1)
        self.assertEqual(result["metadata"]["accuracy"], 1)
        self.assertEqual(result["strict_accuracy"], 1)

    def test_strict_accuracy_drops_for_wrong_fret(self) -> None:
        from app.services.evaluation import compare_drafts

        reference = {
            "tracks": [
                {"name": "guitar", "notes": [{"pitch_midi": 64, "start_beat": 0, "duration": 1, "string": 1, "fret": 0}]}
            ]
        }
        candidate = {
            "tracks": [
                {"name": "guitar", "notes": [{"pitch_midi": 64, "start_beat": 0, "duration": 1, "string": 2, "fret": 5}]}
            ]
        }

        result = compare_drafts(reference, candidate, strict=True)

        self.assertEqual(result["overall"]["f1"], 1)
        self.assertEqual(result["position"]["accuracy"], 0)
        self.assertEqual(result["diagnostics"]["position_mismatches"][0]["reference"]["fret"], 0)
        self.assertLess(result["strict_accuracy"], 0.9)

    def test_strict_diagnostics_include_missed_and_extra_events(self) -> None:
        from app.services.evaluation import compare_drafts

        reference = {
            "tracks": [
                {
                    "name": "guitar",
                    "notes": [
                        {"pitch_midi": 64, "start_beat": 0, "duration": 1},
                        {"pitch_midi": 67, "start_beat": 1, "duration": 1},
                    ],
                }
            ]
        }
        candidate = {
            "tracks": [
                {
                    "name": "guitar",
                    "notes": [
                        {"pitch_midi": 64, "start_beat": 0, "duration": 1},
                        {"pitch_midi": 69, "start_beat": 1, "duration": 1},
                    ],
                }
            ]
        }

        result = compare_drafts(reference, candidate, strict=True)

        self.assertEqual(result["tracks"]["guitar"]["matches"], 1)
        self.assertEqual(result["diagnostics"]["weakest_tracks"][0]["name"], "guitar")
        self.assertEqual(result["diagnostics"]["missed_reference"]["guitar"][0]["pitch_midi"], 67)
        self.assertEqual(result["diagnostics"]["extra_candidate"]["guitar"][0]["pitch_midi"], 69)


if __name__ == "__main__":
    unittest.main()
