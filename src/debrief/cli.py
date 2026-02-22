"""CLI entry point for the debrief tool.

Registered as the ``debrief`` console script in ``pyproject.toml``.  Drives
the pipeline steps individually with a Rich progress UI.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from debrief.analyze import analyze
from debrief.metadata import OllamaStats, RunMetadata, StepTiming
from debrief.pdf import generate_pdf
from debrief.pipeline import DEFAULT_MODEL, SUPPORTED_EXTENSIONS, _resolve_output_path
from debrief.refine import refine_transcript
from debrief.screenshots import extract_screenshots
from debrief.transcribe import transcribe

# Rich console writing to stderr so stdout stays clean.
console = Console(stderr=True)


def _parse_speakers(speakers_str: str | None) -> list[str] | None:
    """Convert a comma-separated speaker string to an ordered list."""
    if speakers_str is None:
        return None

    names = [name.strip() for name in speakers_str.split(",")]
    filtered = [name for name in names if name]

    return filtered if filtered else None


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


def _fmt_duration(seconds: float) -> str:
    """Format seconds as a compact string like '135.2s'."""
    return f"{seconds:.1f}s"


@click.command()
@click.argument("recording", type=click.Path(exists=True))
@click.option(
    "--speakers",
    "-s",
    default=None,
    help="Comma-separated speaker names (e.g., 'Alice,Bob,Carol')",
)
@click.option(
    "--model",
    "-m",
    default=DEFAULT_MODEL,
    show_default=True,
    help="Ollama model for analysis",
)
@click.option(
    "--whisper-model",
    "-w",
    default="large-v3",
    show_default=True,
    help="Whisper model size",
)
@click.option(
    "--output",
    "-o",
    default=None,
    type=click.Path(),
    help="Output PDF path (default: debriefs/<recording>_debrief.pdf)",
)
@click.option(
    "--device",
    "-d",
    default="auto",
    show_default=True,
    type=click.Choice(["auto", "cpu", "cuda"]),
    help="Compute device",
)
def main(
    recording: str,
    speakers: str | None,
    model: str,
    whisper_model: str,
    output: str | None,
    device: str,
) -> None:
    """Generate a debrief report from a recording.

    RECORDING is the path to an audio or video file (.mp4, .mkv, .webm,
    .avi, .mov, .wav, .mp3, .ogg, .flac, .m4a).

    The generated PDF contains a transcript with speaker labels, an
    executive summary, action items, and key concepts — produced entirely
    locally using Whisper and Ollama.

    \b
    Project layout:
      raw/        Place your recordings here
      debriefs/   Generated PDF reports go here (auto-created)

    \b
    Examples:
      debrief raw/standup.mp4
      debrief raw/standup.mp4 --speakers "Alice,Bob,Carol"
      debrief raw/standup.mp4 -m llama3 -w medium -o debriefs/custom.pdf
      debrief raw/call.wav --device cuda
    """
    speaker_list = _parse_speakers(speakers)
    input_path = Path(recording)
    temp_dir: Path | None = None

    try:
        _validate_input(input_path)
        resolved_output = _resolve_output_path(input_path, output)
        temp_dir = Path(tempfile.mkdtemp(prefix="debrief_"))

        started_at = datetime.now(timezone.utc).isoformat()
        wall_start = time.monotonic()
        steps: list[StepTiming] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
            transient=False,
        ) as progress:
            # ---- Transcribe ------------------------------------------------
            task_id = progress.add_task("Transcribing recording...", total=None)
            t0 = time.monotonic()
            segments, duration = transcribe(
                input_path,
                speakers=speaker_list,
                model_size=whisper_model,
                device=device,
            )
            elapsed = time.monotonic() - t0
            steps.append(StepTiming(name="Transcribe", duration_seconds=elapsed))
            progress.update(
                task_id,
                description=f"[green]✓[/green] Transcribed ({_fmt_duration(elapsed)})",
                completed=True,
            )

            # ---- Screenshots -----------------------------------------------
            task_id = progress.add_task("Extracting screenshots...", total=None)
            t0 = time.monotonic()
            screenshots = extract_screenshots(input_path, segments, temp_dir)
            elapsed = time.monotonic() - t0
            steps.append(StepTiming(name="Screenshots", duration_seconds=elapsed))
            count_msg = f"{len(screenshots)} extracted, " if screenshots else ""
            progress.update(
                task_id,
                description=(
                    f"[green]✓[/green] Screenshots ({count_msg}{_fmt_duration(elapsed)})"
                ),
                completed=True,
            )

            # ---- Analyze ---------------------------------------------------
            task_id = progress.add_task("Analysing with Ollama...", total=None)
            t0 = time.monotonic()
            result = analyze(segments, model, duration)
            elapsed = time.monotonic() - t0
            steps.append(StepTiming(name="Analyze", duration_seconds=elapsed))
            progress.update(
                task_id,
                description=f"[green]✓[/green] Analysed ({_fmt_duration(elapsed)})",
                completed=True,
            )

            analysis = result.analysis

            # ---- Refine transcript -----------------------------------------
            task_id = progress.add_task("Refining transcript...", total=None)
            t0 = time.monotonic()
            segments = refine_transcript(segments, analysis)
            elapsed = time.monotonic() - t0
            steps.append(StepTiming(name="Refine", duration_seconds=elapsed))
            progress.update(
                task_id,
                description=f"[green]✓[/green] Refined ({_fmt_duration(elapsed)})",
                completed=True,
            )

            # ---- Generate PDF ----------------------------------------------
            task_id = progress.add_task("Generating PDF...", total=None)
            t0 = time.monotonic()
            pdf_path = generate_pdf(
                resolved_output,
                analysis.summary,
                analysis.tasks,
                analysis.concepts,
                segments,
                screenshots,
                duration,
                speaker_list or [],
                title=analysis.title,
            )
            elapsed = time.monotonic() - t0
            steps.append(StepTiming(name="Generate PDF", duration_seconds=elapsed))
            progress.update(
                task_id,
                description=f"[green]✓[/green] PDF written ({_fmt_duration(elapsed)})",
                completed=True,
            )

        # ---- Build metadata ------------------------------------------------
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

        json_path = metadata.write(pdf_path)

        # ---- Summary table -------------------------------------------------
        console.print()

        table = Table(title="Debrief Complete", show_header=True)
        table.add_column("Step", style="bold")
        table.add_column("Duration", justify="right")

        for step in steps:
            table.add_row(step.name, _fmt_duration(step.duration_seconds))

        table.add_section()
        table.add_row("[bold]Total[/bold]", f"[bold]{_fmt_duration(total_wall)}[/bold]")
        console.print(table)

        console.print()
        console.print(f" PDF:  {pdf_path}")
        console.print(f" JSON: {json_path}")

        if result.prompt_tokens or result.completion_tokens:
            console.print()
            console.print(
                f" Ollama — {result.prompt_tokens:,} prompt tokens, "
                f"{result.completion_tokens:,} completion tokens, "
                f"{result.eval_duration_seconds:.1f}s eval"
            )

        console.print()

    except KeyboardInterrupt:
        console.print("\n[red]Interrupted.[/red]")
        sys.exit(130)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        console.print(f"\n[red]Error:[/red] {exc}")
        sys.exit(1)
    finally:
        if temp_dir is not None and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
