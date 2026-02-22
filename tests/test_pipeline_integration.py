"""Integration test that runs the full pipeline on a real video file.

Skipped by default — requires GPU, Whisper models, and a running Ollama instance.
Run with: pytest -m slow
"""

from __future__ import annotations

from pathlib import Path

import pytest

VIDEOS_DIR = Path(__file__).resolve().parent.parent / "raw"


def _find_test_video() -> Path | None:
    """Return the first .webm file in raw/, or None."""
    if not VIDEOS_DIR.is_dir():
        return None
    for f in sorted(VIDEOS_DIR.iterdir()):
        if f.suffix == ".webm":
            return f
    return None


@pytest.mark.slow
def test_full_pipeline(tmp_path: Path) -> None:
    """Run the complete debrief pipeline on a real video file."""
    video = _find_test_video()
    if video is None:
        pytest.skip("No test video found in raw/")

    from debrief.pipeline import run_pipeline  # noqa: PLC0415

    pdf_path = run_pipeline(
        input_path=video,
        speakers=["Speaker 1", "Speaker 2"],
        model="mistral",
        whisper_model="tiny",  # smallest model for faster test runs
        output_path=tmp_path / "integration_test.pdf",
        device="auto",
    )

    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 500
