from __future__ import annotations

import unittest


class SchemaTests(unittest.TestCase):
    def test_transcription_request_accepts_reference_tab(self) -> None:
        try:
            import pydantic  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("pydantic is not installed in this lightweight local test environment")

        from app.schemas import TranscriptionRequest

        request = TranscriptionRequest(
            youtube_url="https://www.youtube.com/watch?v=lIR1BcDUKBk",
            reference_tab={
                "ascii_tab": """
e|---|
B|---|
G|-2-|
D|---|
A|---|
E|-0-|
""",
                "tempo_bpm": 152,
                "key": "A",
            },
        )

        self.assertEqual(request.reference_tab.tempo_bpm, 152)
        self.assertEqual(request.reference_tab.track_name.value, "guitar")
        self.assertEqual(request.reference_tab.to_worker_payload()["key"], "A")


if __name__ == "__main__":
    unittest.main()
