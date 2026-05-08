from __future__ import annotations

from pathlib import Path


def select_tuning_detection_stem(stems: dict[str, Path], instruments: list, audio_path: Path) -> tuple[Path, str]:
    requested = {str(instrument) for instrument in instruments}
    priority: list[str] = []
    if "guitar" in requested:
        priority.append("guitar")
    if "bass" in requested:
        priority.append("bass")
    priority.extend(["other", "guitar", "bass"])

    for stem_name in priority:
        stem_path = stems.get(stem_name)
        if stem_path is not None:
            return stem_path, stem_name
    return audio_path, "mix"
