"""CLI entry point for the debrief tool.

Registered as the ``debrief`` console script in ``pyproject.toml``.  Parses
user arguments and delegates all work to :func:`debrief.pipeline.run_pipeline`.
"""

from __future__ import annotations

import sys

import click

from debrief.pipeline import run_pipeline


def _parse_speakers(speakers_str: str | None) -> list[str] | None:
    """Convert a comma-separated speaker string to an ordered list.

    Empty names that result from trailing commas or double commas are dropped
    so that ``"Alice,,Bob,"`` is treated the same as ``"Alice,Bob"``.

    Args:
        speakers_str: Raw value from the ``--speakers`` option, or ``None``.

    Returns:
        A list of stripped speaker names, or ``None`` if no input was given.
    """
    if speakers_str is None:
        return None

    names = [name.strip() for name in speakers_str.split(",")]
    filtered = [name for name in names if name]

    return filtered if filtered else None


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
    default="mistral",
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

    try:
        run_pipeline(
            input_path=recording,
            speakers=speaker_list,
            model=model,
            whisper_model=whisper_model,
            output_path=output,
            device=device,
        )
    except KeyboardInterrupt:
        click.echo("\nInterrupted.", err=True)
        sys.exit(130)
    except (FileNotFoundError, ValueError, RuntimeError):
        # run_pipeline already printed a helpful message; just exit non-zero.
        sys.exit(1)
