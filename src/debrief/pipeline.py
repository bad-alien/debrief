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
from pathlib import Path

from debrief.transcribe import AUDIO_EXTENSIONS, VIDEO_EXTENSIONS, transcribe
from debrief.screenshots import extract_screenshots
from debrief.analyze import analyze
from debrief.pdf import generate_pdf


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS: frozenset[str] = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _print_err(message: str) -> None:
    """Write a message to stderr."""
    print(message, file=sys.stderr, flush=True)


def _validate_input(input_path: Path) -> None:
    """Raise informative errors if the input file is unusable.

    Args:
        input_path: Path to the recording file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file extension is not supported.
    """
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
    """Determine the final PDF output path.

    If *output_path* is provided it is used as-is.  Otherwise the output is
    placed in a ``debriefs/`` directory relative to the current working
    directory with the input file's stem suffixed by ``_debrief`` and a
    ``.pdf`` extension.

    Args:
        input_path: The recording file path.
        output_path: Caller-supplied output path, or None.

    Returns:
        Resolved output :class:`Path`.
    """
    if output_path is not None:
        return Path(output_path)

    # Default: place output in debriefs/ relative to the current working directory
    debriefs_dir = Path.cwd() / "debriefs"
    debriefs_dir.mkdir(parents=True, exist_ok=True)
    return debriefs_dir / f"{input_path.stem}_debrief.pdf"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_pipeline(
    input_path: str | Path,
    speakers: list[str] | None = None,
    model: str = "mistral",
    whisper_model: str = "large-v3",
    output_path: str | Path | None = None,
    device: str = "auto",
) -> Path:
    """Run the complete debrief pipeline and return the PDF path.

    Steps:

    1. Validate the input file (exists, supported extension).
    2. Create a temporary directory for intermediate files.
    3. Transcribe the recording with Whisper + pyannote diarization.
    4. Extract representative screenshots from video frames.
    5. Analyse the transcript with the requested Ollama model.
    6. Render a PDF report and write it to *output_path*.
    7. Remove the temporary directory (WeasyPrint embeds images, so the
       temp dir is safe to delete after the PDF has been written).

    Args:
        input_path: Path to the audio or video recording.
        speakers: Optional ordered list of speaker display names.  Maps
            index-by-index onto pyannote's SPEAKER_00, SPEAKER_01, … labels.
        model: Ollama model identifier used for summarisation and task
            extraction (default: ``"mistral"``).
        whisper_model: faster-whisper model variant (default: ``"large-v3"``).
        output_path: Destination path for the generated PDF.  Defaults to
            ``debriefs/<input_stem>_debrief.pdf``.
        device: Compute device — ``"cpu"``, ``"cuda"``, or ``"auto"``.

    Returns:
        The :class:`Path` of the written PDF report.

    Raises:
        FileNotFoundError: If the input file or ffmpeg/ffprobe cannot be found.
        ValueError: If the file extension is not supported.
        RuntimeError: If ffmpeg, Whisper, pyannote, or Ollama encounter an
            error during processing.
    """
    input_path = Path(input_path)
    temp_dir: Path | None = None

    try:
        # ------------------------------------------------------------------
        # Step 1: Validate input
        # ------------------------------------------------------------------
        _validate_input(input_path)

        resolved_output = _resolve_output_path(input_path, output_path)

        # ------------------------------------------------------------------
        # Step 2: Create temp directory for intermediate files
        # ------------------------------------------------------------------
        temp_dir = Path(tempfile.mkdtemp(prefix="debrief_"))

        # ------------------------------------------------------------------
        # Step 3: Transcribe
        # ------------------------------------------------------------------
        _print_err("Step 1/4  Transcribing recording...")
        segments, duration = transcribe(
            input_path,
            speakers=speakers,
            model_size=whisper_model,
            device=device,
        )

        # ------------------------------------------------------------------
        # Step 4: Extract screenshots
        # ------------------------------------------------------------------
        _print_err("Step 2/4  Extracting screenshots...")
        screenshots = extract_screenshots(input_path, segments, temp_dir)

        # ------------------------------------------------------------------
        # Step 5: Analyse transcript
        # ------------------------------------------------------------------
        _print_err("Step 3/4  Analysing transcript with Ollama...")
        analysis = analyze(segments, model, duration)

        # ------------------------------------------------------------------
        # Step 6: Generate PDF
        # ------------------------------------------------------------------
        _print_err("Step 4/4  Generating PDF report...")
        pdf_path = generate_pdf(
            resolved_output,
            analysis.summary,
            analysis.tasks,
            analysis.concepts,
            segments,
            screenshots,
            duration,
            speakers or [],
        )

        _print_err(f"\nDone! Report saved to {pdf_path}")
        return pdf_path

    except FileNotFoundError as exc:
        # Covers missing input file and missing ffmpeg/ffprobe binaries.
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
        # Ollama is not running or refused the connection.
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
        # Clean up the temporary directory.  WeasyPrint embeds images directly
        # into the PDF, so no intermediate files need to survive past this point.
        if temp_dir is not None and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
