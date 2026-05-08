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

    def test_librosa_fallback_classifies_low_and_high_hits(self) -> None:
        import numpy as np
        import soundfile as sf

        from app.services.drums import _transcribe_with_librosa

        sample_rate = 22050
        audio = np.zeros(sample_rate, dtype=np.float32)
        low_start = int(0.1 * sample_rate)
        low_times = np.linspace(0, 0.08, int(0.08 * sample_rate), endpoint=False)
        audio[low_start:low_start + low_times.size] += 0.8 * np.hanning(low_times.size) * np.sin(2 * np.pi * 70 * low_times)

        high_start = int(0.45 * sample_rate)
        high_times = np.linspace(0, 0.04, int(0.04 * sample_rate), endpoint=False)
        audio[high_start:high_start + high_times.size] += 0.5 * np.hanning(high_times.size) * np.sin(2 * np.pi * 6000 * high_times)

        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "classified-drums.wav"
            sf.write(audio_path, audio, sample_rate)

            hits = _transcribe_with_librosa(str(audio_path), 120)

        drums = {hit["drum"] for hit in hits}
        self.assertIn("kick", drums)
        self.assertIn("hihat_closed", drums)


class DrumActivationTests(unittest.TestCase):
    def test_madmom_activations_use_local_peaks(self) -> None:
        from app.services.drums import _local_peak_frames

        activations = [[0.0, 0.0, 0.0] for _ in range(8)]
        activations[2][0] = 0.6
        activations[3][0] = 0.7
        activations[4][0] = 0.65
        activations[6][1] = 0.8

        peaks = _local_peak_frames(activations, threshold=0.3, wait=2)

        self.assertEqual(peaks, [(3, 0, 0.7), (6, 1, 0.8)])


if __name__ == "__main__":
    unittest.main()
