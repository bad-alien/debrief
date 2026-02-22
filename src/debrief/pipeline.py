"""Pipeline orchestration for the debrief CLI tool.

Coordinates the full debrief workflow:
  1. Input validation
  2. Transcription and speaker diarization
  3. Screenshot extraction
  4. LLM analysis via Ollama
  5. PDF report generation
  6. Temp file cleanup
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from debrief.transcribe import AUDIO_EXTENSIONS, VIDEO_EXTENSIONS, transcribe
from debrief.screenshots import extract_screenshots
from debrief.analyze import analyze
from debrief.pdf import generate_pdf
from debrief.metadata import OllamaStats, RunMetadata, StepTiming
from debrief.refine import refine_transcript


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS: frozenset[str] = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS

DEFAULT_MODEL: str = "hf.co/Qwen/Qwen3-14B-GGUF:Q4_K_M"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _print_err(message: str) -> None:
    """Write a message to stderr."""
    print(message, file=sys.stderr, flush=True)


def _validate_input(input_path: Path) -> None:
    """Raise informative errors if the input file is unusable."""
    if not input_path.exists():
        raise FileNotFoundError(
            f"Recording not found: {input_path}\n"
            "Check that the path is correct and the file has not been moved."
        )

    suffix = input_path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(
            f"Unsupported file format '{suffix}'.\n"
            f"Supported formats: {supported}"
        )


def _resolve_output_path(input_path: Path, output_path: str | Path | None) -> Path:
    """Determine the final PDF output path."""
    if output_path is not None:
        return Path(output_path)

    debriefs_dir = Path.cwd() / "debriefs"
    debriefs_dir.mkdir(parents=True, exist_ok=True)
    return debriefs_dir / f"{input_path.stem}_debrief.pdf"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_pipeline(
    input_path: str | Path,
    speakers: list[str] | None = None,
    model: str = DEFAULT_MODEL,
    whisper_model: str = "large-v3",
    output_path: str | Path | None = None,
    device: str = "auto",
) -> tuple[Path, RunMetadata]:
    """Run the complete debrief pipeline and return the PDF path and metadata.

    Returns:
        A two-tuple of (pdf_path, run_metadata).

    Raises:
        FileNotFoundError: If the input file or ffmpeg/ffprobe cannot be found.
        ValueError: If the file extension is not supported.
        RuntimeError: If ffmpeg, Whisper, pyannote, or Ollama encounter an
            error during processing.
    """
    input_path = Path(input_path)
    temp_dir: Path | None = None
    started_at = datetime.now(timezone.utc).isoformat()
    wall_start = time.monotonic()
    steps: list[StepTiming] = []

    try:
        # Step 1: Validate input
        _validate_input(input_path)
        resolved_output = _resolve_output_path(input_path, output_path)

        # Step 2: Create temp directory
        temp_dir = Path(tempfile.mkdtemp(prefix="debrief_"))

        # Step 3: Transcribe
        t0 = time.monotonic()
        segments, duration = transcribe(
            input_path,
            speakers=speakers,
            model_size=whisper_model,
            device=device,
        )
        steps.append(StepTiming(name="Transcribe", duration_seconds=time.monotonic() - t0))

        # Step 4: Extract screenshots
        t0 = time.monotonic()
        screenshots = extract_screenshots(input_path, segments, temp_dir)
        steps.append(StepTiming(name="Screenshots", duration_seconds=time.monotonic() - t0))

        # Step 5: Analyse transcript
        t0 = time.monotonic()
        result = analyze(segments, model, duration)
        steps.append(StepTiming(name="Analyze", duration_seconds=time.monotonic() - t0))

        analysis = result.analysis

        # Step 6: Refine transcript
        t0 = time.monotonic()
        segments = refine_transcript(segments, analysis)
        steps.append(StepTiming(name="Refine", duration_seconds=time.monotonic() - t0))

        # Step 7: Generate PDF
        t0 = time.monotonic()
        pdf_path = generate_pdf(
            resolved_output,
            analysis.summary,
            analysis.tasks,
            analysis.concepts,
            segments,
            screenshots,
            duration,
            speakers or [],
            title=analysis.title,
        )
        steps.append(StepTiming(name="Generate PDF", duration_seconds=time.monotonic() - t0))

        total_wall = time.monotonic() - wall_start

        metadata = RunMetadata(
            started_at=started_at,
            total_wall_seconds=total_wall,
            whisper_model=whisper_model,
            llm_model=model,
            transcript_duration_seconds=duration,
            segment_count=len(segments),
            speaker_count=len({s.speaker for s in segments}),
            screenshot_count=len(screenshots),
            task_count=len(analysis.tasks),
            concept_count=len(analysis.concepts),
            json_parsed_cleanly=result.json_parsed_cleanly,
            chunk_count=result.chunk_count,
            ollama=OllamaStats(
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                eval_duration_seconds=result.eval_duration_seconds,
                total_duration_seconds=result.total_duration_seconds,
                load_duration_seconds=result.load_duration_seconds,
            ),
            steps=steps,
        )

        return pdf_path, metadata

    except FileNotFoundError as exc:
        _print_err(f"\nError: {exc}")
        if "ffmpeg" in str(exc).lower() or "ffprobe" in str(exc).lower():
            _print_err(
                "Tip: Install ffmpeg and make sure it is on your PATH.\n"
                "     macOS:   brew install ffmpeg\n"
                "     Ubuntu:  sudo apt install ffmpeg\n"
                "     Windows: https://ffmpeg.org/download.html"
            )
        raise

    except ValueError as exc:
        _print_err(f"\nError: {exc}")
        raise

    except ConnectionError as exc:
        _print_err(
            f"\nError: Could not connect to Ollama — {exc}\n"
            "Tip: Start Ollama with `ollama serve` and ensure the requested\n"
            f"     model is pulled: ollama pull {model}"
        )
        raise RuntimeError("Ollama connection failed") from exc

    except RuntimeError as exc:
        error_text = str(exc).lower()
        _print_err(f"\nError: {exc}")
        if "ffmpeg" in error_text:
            _print_err(
                "Tip: ffmpeg failed.  Verify the recording file is not corrupted\n"
                "     and that you have a recent ffmpeg version installed."
            )
        elif "ollama" in error_text or "model" in error_text:
            _print_err(
                f"Tip: Make sure the model is available: ollama pull {model}"
            )
        raise

    finally:
        if temp_dir is not None and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
