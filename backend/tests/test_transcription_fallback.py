from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(
    importlib.util.find_spec("librosa")
    and importlib.util.find_spec("numpy")
    and importlib.util.find_spec("soundfile"),
    "librosa fallback dependencies are not installed",
)
class TranscriptionFallbackTests(unittest.TestCase):
    def test_librosa_fallback_detects_simple_guitar_note(self) -> None:
        import numpy as np
        import soundfile as sf

        from app.services.transcription import _audio_to_midi_with_librosa

        sample_rate = 22050
        duration = 0.5
        times = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        audio = 0.35 * np.sin(2 * np.pi * 440 * times)

        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "a4.wav"
            sf.write(audio_path, audio, sample_rate)

            notes = _audio_to_midi_with_librosa(audio_path, "guitar")

        self.assertTrue(notes)
        self.assertTrue(any(abs(note["pitch_midi"] - 69) <= 1 for note in notes))


if __name__ == "__main__":
    unittest.main()
