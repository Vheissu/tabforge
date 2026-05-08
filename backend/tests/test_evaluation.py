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


if __name__ == "__main__":
    unittest.main()
