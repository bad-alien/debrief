"""Transcription module for the debrief CLI tool.

Handles audio extraction from video files, speech-to-text transcription via
faster-whisper, and speaker diarization via pyannote.audio.  The public
interface is a single :func:`transcribe` function that returns a list of
:class:`Segment` dataclasses, one per speaker turn, with word-level
timestamps already merged in.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    """A single speaker-attributed transcript segment.

    Attributes:
        start: Segment start time in seconds.
        end: Segment end time in seconds.
        speaker: Human-readable speaker label, e.g. "Speaker 1" or "Alice".
        text: Transcribed text for this segment.
    """

    start: float
    end: float
    speaker: str
    text: str


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VIDEO_EXTENSIONS: frozenset[str] = frozenset(
    {".mp4", ".mkv", ".webm", ".avi", ".mov"}
)
AUDIO_EXTENSIONS: frozenset[str] = frozenset(
    {".wav", ".mp3", ".ogg", ".flac", ".m4a"}
)

# pyannote speaker label format produced by the 3.x pipeline
_PYANNOTE_SPEAKER_PREFIX = "SPEAKER_"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _print_progress(message: str) -> None:
    """Write a progress message to stderr."""
    print(message, file=sys.stderr, flush=True)


def _resolve_device(device: str) -> str:
    """Resolve "auto" to "cuda" or "cpu" based on torch availability.

    Args:
        device: "auto", "cuda", or "cpu".

    Returns:
        Concrete device string suitable for faster-whisper.
    """
    if device != "auto":
        return device

    try:
        import torch  # noqa: PLC0415

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _extract_audio(input_path: Path, dest_wav: Path) -> None:
    """Extract or convert audio from *input_path* and write a 16 kHz mono WAV.

    Uses ffmpeg via subprocess.  Raises :class:`RuntimeError` if ffmpeg exits
    with a non-zero status code.

    Args:
        input_path: Source file (video or audio).
        dest_wav: Destination WAV file path (will be created/overwritten).

    Raises:
        RuntimeError: If the ffmpeg subprocess fails.
        FileNotFoundError: If ffmpeg is not installed.
    """
    cmd = [
        "ffmpeg",
        "-y",                     # overwrite output without prompting
        "-i", str(input_path),
        "-vn",                    # drop video stream
        "-ac", "1",               # mono
        "-ar", "16000",           # 16 kHz — optimal for Whisper
        "-sample_fmt", "s16",     # 16-bit signed PCM
        str(dest_wav),
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if result.returncode != 0:
        error_output = result.stderr.decode(errors="replace")
        raise RuntimeError(
            f"ffmpeg failed with exit code {result.returncode}.\n{error_output}"
        )


def _get_audio_duration(wav_path: Path) -> float:
    """Return the duration of a WAV file in seconds using ffprobe.

    Args:
        wav_path: Path to the WAV file.

    Returns:
        Duration in seconds, or 0.0 if it cannot be determined.
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(wav_path),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        return float(result.stdout.decode().strip())
    except ValueError:
        return 0.0


def _label_for_speaker(
    pyannote_label: str,
    speakers: list[str] | None,
) -> str:
    """Map a raw pyannote speaker label to a human-readable name.

    Args:
        pyannote_label: Raw label such as "SPEAKER_00" or "SPEAKER_01".
        speakers: Optional ordered list of human-readable names.  Index 0
            corresponds to SPEAKER_00, index 1 to SPEAKER_01, etc.

    Returns:
        A display name like "Alice" or "Speaker 1".
    """
    index: int | None = None
    if pyannote_label.startswith(_PYANNOTE_SPEAKER_PREFIX):
        try:
            index = int(pyannote_label[len(_PYANNOTE_SPEAKER_PREFIX):])
        except ValueError:
            pass

    if index is not None and speakers and index < len(speakers):
        return speakers[index]

    if index is not None:
        return f"Speaker {index + 1}"

    # Fallback: return the raw label unchanged
    return pyannote_label


def _assign_speaker(
    seg_start: float,
    seg_end: float,
    diarization_turns: list[tuple[float, float, str]],
) -> str:
    """Find the speaker label with the greatest overlap with a whisper segment.

    Args:
        seg_start: Segment start in seconds.
        seg_end: Segment end in seconds.
        diarization_turns: List of (turn_start, turn_end, speaker_label) tuples
            produced by pyannote.

    Returns:
        The speaker label with the longest overlap, or "SPEAKER_00" if no
        overlap is found.
    """
    best_label = "SPEAKER_00"
    best_overlap = 0.0

    for turn_start, turn_end, label in diarization_turns:
        overlap = min(seg_end, turn_end) - max(seg_start, turn_start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_label = label

    return best_label


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def transcribe(
    input_path: str | Path,
    speakers: list[str] | None = None,
    model_size: str = "large-v3",
    device: str = "auto",
) -> tuple[list[Segment], float]:
    """Transcribe a recording and return labelled speaker segments.

    Workflow:

    1. If the input is a video file, extract audio to a temporary 16 kHz mono
       WAV via ffmpeg.  Audio files are converted to the same WAV format if
       they are not already a WAV (pyannote requires WAV input).
    2. Run faster-whisper transcription with word-level timestamps.
    3. Run pyannote speaker diarization on the same WAV.
    4. Merge the two outputs: each Whisper segment is assigned the speaker
       whose diarization turn overlaps it the most.
    5. Map raw pyannote labels ("SPEAKER_00") to display names from the
       optional *speakers* list, or fall back to "Speaker 1", "Speaker 2", …

    Args:
        input_path: Path to the audio or video file to transcribe.
        speakers: Optional ordered list of speaker names.  The first name is
            mapped to SPEAKER_00, the second to SPEAKER_01, and so on.
        model_size: Whisper model variant.  Defaults to "large-v3".
        device: Compute device — "cpu", "cuda", or "auto".  "auto" selects
            "cuda" when a CUDA-capable GPU is available, otherwise "cpu".

    Returns:
        A two-tuple of:
        - A list of :class:`Segment` instances in chronological order.
        - The total duration of the recording in seconds.

    Raises:
        FileNotFoundError: If *input_path* does not exist.
        ValueError: If the file extension is not a recognised audio or video
            format.
        RuntimeError: If ffmpeg fails during audio extraction.
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    suffix = input_path.suffix.lower()
    is_video = suffix in VIDEO_EXTENSIONS
    is_audio = suffix in AUDIO_EXTENSIONS

    if not is_video and not is_audio:
        raise ValueError(
            f"Unsupported file extension '{suffix}'.  "
            f"Supported video: {sorted(VIDEO_EXTENSIONS)}, "
            f"audio: {sorted(AUDIO_EXTENSIONS)}."
        )

    resolved_device = _resolve_device(device)
    compute_type = "float16" if resolved_device == "cuda" else "int8"

    # ------------------------------------------------------------------
    # Step 1: Obtain a 16 kHz mono WAV file
    # ------------------------------------------------------------------

    # We may need a temp file; keep a reference so we can clean it up.
    _tmp_wav_file: tempfile.NamedTemporaryFile | None = None
    wav_path: Path

    try:
        needs_conversion = is_video or suffix != ".wav"

        if needs_conversion:
            _tmp_wav_file = tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False
            )
            _tmp_wav_file.close()  # Close so ffmpeg can write to it on all platforms
            wav_path = Path(_tmp_wav_file.name)

            _print_progress("Extracting audio...")
            _extract_audio(input_path, wav_path)
        else:
            wav_path = input_path

        duration = _get_audio_duration(wav_path)

        # ------------------------------------------------------------------
        # Step 2: Transcription with faster-whisper
        # ------------------------------------------------------------------

        _print_progress(f"Transcribing with Whisper {model_size} on {resolved_device}...")

        from faster_whisper import WhisperModel  # noqa: PLC0415

        whisper_model = WhisperModel(
            model_size,
            device=resolved_device,
            compute_type=compute_type,
        )

        whisper_segments_iter, _info = whisper_model.transcribe(
            str(wav_path),
            word_timestamps=True,
        )

        # Materialise the lazy iterator into a plain list so we can iterate
        # it a second time for the merge step.
        whisper_segments: list[tuple[float, float, str]] = [
            (seg.start, seg.end, seg.text.strip())
            for seg in whisper_segments_iter
        ]

        # ------------------------------------------------------------------
        # Step 3: Speaker diarization with pyannote
        # ------------------------------------------------------------------

        _print_progress("Diarizing speakers...")

        from pyannote.audio import Pipeline  # noqa: PLC0415

        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")

        if resolved_device == "cuda":
            import torch  # noqa: PLC0415

            pipeline = pipeline.to(torch.device("cuda"))

        diarization = pipeline(str(wav_path))

        # Flatten the pyannote annotation into a simple list of turns.
        diarization_turns: list[tuple[float, float, str]] = [
            (turn.start, turn.end, label)
            for turn, _, label in diarization.itertracks(yield_label=True)
        ]

        # ------------------------------------------------------------------
        # Step 4 & 5: Merge transcription with diarization, apply names
        # ------------------------------------------------------------------

        _print_progress("Merging transcription and diarization...")

        segments: list[Segment] = []
        for seg_start, seg_end, text in whisper_segments:
            if not text:
                continue

            raw_label = _assign_speaker(seg_start, seg_end, diarization_turns)
            display_name = _label_for_speaker(raw_label, speakers)

            segments.append(
                Segment(
                    start=seg_start,
                    end=seg_end,
                    speaker=display_name,
                    text=text,
                )
            )

        _print_progress(
            f"Done. {len(segments)} segments, "
            f"{len({s.speaker for s in segments})} speaker(s), "
            f"{duration:.1f}s total."
        )

        return segments, duration

    finally:
        # Always clean up the temporary WAV file, even if an exception occurs.
        if _tmp_wav_file is not None:
            tmp_path = Path(_tmp_wav_file.name)
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
