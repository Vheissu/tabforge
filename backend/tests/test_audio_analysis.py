from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(
    importlib.util.find_spec("librosa")
    and importlib.util.find_spec("numpy")
    and importlib.util.find_spec("soundfile"),
    "audio analysis dependencies are not installed",
)
class AudioAnalysisTests(unittest.TestCase):
    def test_detect_tuning_handles_silence(self) -> None:
        import numpy as np
        import soundfile as sf

        from app.services.audio import detect_tuning

        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "silence.wav"
            sf.write(audio_path, np.zeros(22050), 22050)

            result = detect_tuning(audio_path)

        self.assertEqual(result["tuning"], "standard")
        self.assertIn("offset_semitones", result)
        self.assertIn("low_freq", result)
        self.assertEqual(result["confidence"], 0.0)
        self.assertEqual(result["candidate_count"], 0)


if __name__ == "__main__":
    unittest.main()
