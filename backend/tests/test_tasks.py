from __future__ import annotations

import unittest
from pathlib import Path


class TaskHelperTests(unittest.TestCase):
    def test_tuning_detection_prefers_requested_pitched_stem(self) -> None:
        from app.services.stems import select_tuning_detection_stem

        stems = {
            "other": Path("other.wav"),
            "guitar": Path("guitar.wav"),
            "bass": Path("bass.wav"),
        }

        self.assertEqual(select_tuning_detection_stem(stems, ["guitar", "bass"], Path("mix.wav")), (Path("guitar.wav"), "guitar"))
        self.assertEqual(select_tuning_detection_stem(stems, ["bass"], Path("mix.wav")), (Path("bass.wav"), "bass"))

    def test_tuning_detection_falls_back_to_mix(self) -> None:
        from app.services.stems import select_tuning_detection_stem

        self.assertEqual(select_tuning_detection_stem({}, ["drums"], Path("mix.wav")), (Path("mix.wav"), "mix"))


if __name__ == "__main__":
    unittest.main()
