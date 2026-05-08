from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(
    importlib.util.find_spec("librosa")
    and importlib.util.find_spec("numpy")
    and importlib.util.find_spec("soundfile"),
    "drum fallback dependencies are not installed",
)
class DrumFallbackTests(unittest.TestCase):
    def test_librosa_fallback_detects_impulse_hits(self) -> None:
        import numpy as np
        import soundfile as sf

        from app.services.drums import _transcribe_with_librosa

        sample_rate = 22050
        audio = np.zeros(sample_rate, dtype=np.float32)
        for offset in (0.1, 0.35, 0.6, 0.85):
            start = int(offset * sample_rate)
            audio[start:start + 220] = np.hanning(220)

        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "drums.wav"
            sf.write(audio_path, audio, sample_rate)

            hits = _transcribe_with_librosa(str(audio_path), 120)

        self.assertGreaterEqual(len(hits), 2)


if __name__ == "__main__":
    unittest.main()
