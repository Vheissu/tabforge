from __future__ import annotations

import tempfile
import unittest
import importlib.util
from pathlib import Path


@unittest.skipUnless(importlib.util.find_spec("pydantic_settings"), "API dependencies are not installed")
class SeparationTests(unittest.TestCase):
    def test_collects_optional_six_stem_demucs_outputs(self) -> None:
        from app.services.separation import _collect_demucs_cli_stems

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            for stem in ("drums", "bass", "other", "vocals", "guitar", "piano"):
                (output_dir / f"{stem}.wav").touch()

            stems = _collect_demucs_cli_stems(output_dir)

        self.assertEqual(set(stems), {"drums", "bass", "other", "vocals", "guitar", "piano"})
        self.assertEqual(stems["guitar"].name, "guitar.wav")

    def test_collect_demucs_outputs_requires_core_stems(self) -> None:
        from app.services.separation import _collect_demucs_cli_stems

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            (output_dir / "guitar.wav").touch()

            with self.assertRaisesRegex(RuntimeError, "bass"):
                _collect_demucs_cli_stems(output_dir)


if __name__ == "__main__":
    unittest.main()
