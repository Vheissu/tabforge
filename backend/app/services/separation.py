from __future__ import annotations

from pathlib import Path

import subprocess

import torch

from app.core.config import get_settings


def separate_stems(audio_path: Path, output_dir: Path) -> dict:
    settings = get_settings()
    if not settings.separation_enabled:
        return {
            "drums": audio_path,
            "bass": audio_path,
            "other": audio_path,
            "vocals": audio_path,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = settings.separation_model
    segment = settings.separation_segment_seconds if device == "cpu" else None
    try:
        from demucs.api import Separator

        separator = Separator(
            model=model_name,
            device=device,
            segment=segment,
        )

        _, separated = separator.separate_audio_file(str(audio_path))

        stems = {}
        for stem_name, stem_tensor in separated.items():
            stem_path = output_dir / f"{stem_name}.wav"
            separator.save_audio(stem_tensor, str(stem_path), samplerate=44100)
            stems[stem_name] = stem_path

        return stems
    except Exception:
        # Fallback to CLI if Python API is unavailable.
        cmd = ["demucs", "-n", model_name, "-o", str(output_dir)]
        if device == "cuda":
            cmd += ["-d", "cuda"]
        cmd.append(str(audio_path))

        subprocess.run(cmd, check=True)

        model_dir = output_dir / model_name / audio_path.stem
        stems = {
            "drums": model_dir / "drums.wav",
            "bass": model_dir / "bass.wav",
            "other": model_dir / "other.wav",
            "vocals": model_dir / "vocals.wav",
        }
        missing = [name for name, path in stems.items() if not path.exists()]
        if missing:
            raise RuntimeError(f"Demucs output missing stems: {', '.join(missing)}")
        return stems
