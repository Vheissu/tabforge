from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

YOUTUBE_REGEX = re.compile(
    r"^(https?://)?(www\.)?(youtube\.com|youtu\.be)/.+$",
    re.IGNORECASE,
)


class YouTubeError(RuntimeError):
    pass


def validate_youtube_url(youtube_url: str, max_duration: int) -> int:
    if not YOUTUBE_REGEX.match(youtube_url):
        raise YouTubeError("Invalid YouTube URL")
    if "list=" in youtube_url:
        raise YouTubeError("Playlists are not supported")

    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--no-download",
        "--print-json",
        "--no-warnings",
        youtube_url,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)
    duration = int(info.get("duration") or 0)

    if duration <= 0:
        raise YouTubeError("Unable to determine video duration")
    if duration > max_duration:
        raise YouTubeError(f"Video exceeds maximum duration of {max_duration} seconds")

    return duration


def extract_audio(youtube_url: str, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(output_dir / "%(id)s.%(ext)s")

    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--extract-audio",
        "--audio-format",
        "wav",
        "--audio-quality",
        "0",
        "--output",
        output_template,
        "--print-json",
        "--no-warnings",
        youtube_url,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)

    audio_path = output_dir / f"{info['id']}.wav"
    if not audio_path.exists():
        raise YouTubeError("Audio extraction failed")

    return {
        "id": info["id"],
        "title": info.get("title", "Unknown"),
        "duration": info.get("duration", 0),
        "artist": info.get("artist") or info.get("uploader", "Unknown"),
        "audio_path": audio_path,
    }
