from __future__ import annotations

import unittest


class ConstraintTests(unittest.TestCase):
    def test_normalises_user_tempo_and_meter(self) -> None:
        from app.services.constraints import normalise_constraints

        result = normalise_constraints(
            {
                "first_bar_time_signature": "6/8",
                "first_bar_tempo_bpm": 96,
                "pickup_bar_beats": 1.5,
                "triplet_feel": True,
                "capo_fret": 2,
            }
        )

        self.assertEqual(result["time_signature"], "6/8")
        self.assertEqual(result["tempo_bpm"], 96)
        self.assertEqual(result["tempo_source"], "user")
        self.assertEqual(result["pickup_bar_beats"], 1.5)
        self.assertTrue(result["triplet_feel"])
        self.assertEqual(result["capo_fret"], 2)

    def test_pickup_time_signature_uses_musical_duration(self) -> None:
        from app.services.constraints import pickup_time_signature

        self.assertEqual(pickup_time_signature(1.0), (1, 4))
        self.assertEqual(pickup_time_signature(1.5), (3, 8))

    def test_triplet_feel_auto_and_straight_are_not_truthy(self) -> None:
        from app.services.constraints import normalise_constraints

        self.assertFalse(normalise_constraints({"triplet_feel": "auto"})["triplet_feel"])
        self.assertFalse(normalise_constraints({"triplet_feel": "straight"})["triplet_feel"])
        self.assertFalse(normalise_constraints({"triplet_feel": None})["triplet_feel"])
        self.assertTrue(normalise_constraints({"triplet_feel": "triplet"})["triplet_feel"])


if __name__ == "__main__":
    unittest.main()
