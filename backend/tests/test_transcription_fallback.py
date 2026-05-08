from __future__ import annotations

import importlib.util
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


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

    def test_spectral_fallback_detects_pulsed_bass_note(self) -> None:
        import numpy as np
        import soundfile as sf

        from app.services.transcription import _audio_to_midi_with_spectral_fallback

        sample_rate = 22050
        duration = 1.2
        times = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        envelope = ((times % 0.3) < 0.16).astype(float)
        audio = 0.35 * envelope * np.sin(2 * np.pi * 110 * times)

        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "bass-a2.wav"
            sf.write(audio_path, audio, sample_rate)

            notes = _audio_to_midi_with_spectral_fallback(audio_path, "bass")

        self.assertTrue(notes)
        self.assertTrue(any(abs(note["pitch_midi"] - 45) <= 1 for note in notes))


@unittest.skipUnless(importlib.util.find_spec("pydantic_settings"), "API dependencies are not installed")
class GeminiRefinementTests(unittest.TestCase):
    def test_parse_refinement_response_accepts_fenced_json(self) -> None:
        from app.services.transcription import _parse_refinement_response

        self.assertEqual(_parse_refinement_response("```json\n{\"ok\": true}\n```"), {"ok": True})

    def test_refinement_errors_do_not_fail_transcription(self) -> None:
        from app.services import transcription

        class FakeModels:
            def generate_content(self, **_kwargs):
                raise RuntimeError("model unavailable")

        class FakeClient:
            files = SimpleNamespace(upload=lambda **_kwargs: object())
            models = FakeModels()

            def __init__(self, **_kwargs):
                pass

            def close(self):
                pass

        fake_genai = types.ModuleType("google.genai")
        fake_genai.Client = FakeClient
        fake_google = types.ModuleType("google")
        fake_google.genai = fake_genai

        settings = SimpleNamespace(gemini_api_key="test-key", gemini_model="bad-model")
        with patch.dict(sys.modules, {"google": fake_google, "google.genai": fake_genai}):
            with patch.object(transcription, "settings", settings):
                result = transcription.refine_with_gemini(Path("missing.wav"), [], "guitar", 120)

        self.assertIsNone(result)

    def test_refinement_timeout_does_not_fail_transcription(self) -> None:
        from app.services import transcription

        class FakeModels:
            def generate_content(self, **_kwargs):
                return SimpleNamespace(text="{}")

        def slow_upload(_path):
            time.sleep(2)
            return object()

        class FakeClient:
            files = SimpleNamespace(upload=lambda **kwargs: slow_upload(kwargs["file"]))
            models = FakeModels()

            def __init__(self, **_kwargs):
                pass

            def close(self):
                pass

        fake_genai = types.ModuleType("google.genai")
        fake_genai.Client = FakeClient
        fake_google = types.ModuleType("google")
        fake_google.genai = fake_genai

        settings = SimpleNamespace(
            gemini_api_key="test-key",
            gemini_model="gemini-3-flash-preview",
            gemini_refinement_timeout_seconds=1,
        )
        with patch.dict(sys.modules, {"google": fake_google, "google.genai": fake_genai}):
            with patch.object(transcription, "settings", settings):
                started_at = time.monotonic()
                result = transcription.refine_with_gemini(Path("slow.wav"), [], "guitar", 120)

        self.assertIsNone(result)
        self.assertLess(time.monotonic() - started_at, 1.8)


@unittest.skipUnless(importlib.util.find_spec("pydantic_settings"), "API dependencies are not installed")
class BasicPitchNormalizationTests(unittest.TestCase):
    def test_tuple_events_are_converted_from_basic_pitch_output(self) -> None:
        from app.services.transcription import _normalize_basic_pitch_event

        event = (1.25, 1.75, 64, 0.5, [0, 1])

        self.assertEqual(
            _normalize_basic_pitch_event(event),
            {"start_time": 1.25, "end_time": 1.75, "pitch_midi": 64, "velocity": 64},
        )

    def test_dict_amplitude_is_converted_to_velocity(self) -> None:
        from app.services.transcription import _normalize_basic_pitch_event

        event = {"start_time": 0, "end_time": 1, "pitch_midi": 40, "amplitude": 0.75}

        self.assertEqual(_normalize_basic_pitch_event(event)["velocity"], 95)


if __name__ == "__main__":
    unittest.main()
