"""Screenshot extraction module for debrief.

Scans transcript segments for NLP triggers that indicate a speaker is referencing
something visual, then extracts video frames at those timestamps using ffmpeg.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Screenshot:
    """A single extracted video frame tied to a visual-reference trigger.

    Attributes:
        timestamp: Seconds into the recording at which the frame was extracted.
        path: Absolute path to the saved JPEG image file.
        trigger_text: The transcript text that caused extraction (e.g. "look at this graph").
    """

    timestamp: float
    path: Path
    trigger_text: str


# ---------------------------------------------------------------------------
# Trigger pattern
# ---------------------------------------------------------------------------

# All phrases that indicate the speaker is referencing something on screen.
# Compiled once at module import time for efficiency.
_TRIGGER_PHRASES: list[str] = [
    # Directing attention
    r"look at this",
    r"look at that",
    r"look here",
    r"looking at",
    # Visibility cues
    r"as you can see",
    r"you can see",
    r"can you see",
    # Named visual artefacts
    r"this graph",
    r"this chart",
    r"this diagram",
    r"this slide",
    r"this screen",
    r"this page",
    r"the graph",
    r"the chart",
    r"the diagram",
    r"the slide",
    r"the screen",
    # Screen position indicators
    r"on screen",
    r"on the screen",
    r"shown here",
    r"shown above",
    # Design artefacts
    r"this design",
    r"the design",
    r"this mockup",
    r"the mockup",
    r"this wireframe",
    # Sharing / showing intent
    r"let me show",
    r"let me share",
    r"I'll share",
    r"sharing my screen",
    # Presentational openers
    r"here we have",
    r"here you see",
    r"here is",
    r"here are",
    # Tables and images
    r"this table",
    r"the table",
    r"this image",
    r"the image",
    # Positional deictics
    r"right here",
    r"over here",
]

_TRIGGER_RE: re.Pattern[str] = re.compile(
    r"(?:" + r"|".join(_TRIGGER_PHRASES) + r")",
    re.IGNORECASE,
)

# Minimum gap in seconds between two extracted frames (deduplication window).
_DEDUP_WINDOW_SECONDS: float = 3.0


# ---------------------------------------------------------------------------
# ffprobe helper
# ---------------------------------------------------------------------------


def _has_video_stream(path: Path) -> bool:
    """Return True if the media file at *path* contains at least one video stream.

    Uses ``ffprobe`` with JSON output to inspect stream metadata.  Any error
    (missing ffprobe, corrupt file, etc.) causes the function to return False
    so that the caller degrades gracefully.

    Args:
        path: Path to the media file to probe.

    Returns:
        True when a video stream is detected, False otherwise.
    """
    cmd: list[str] = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        str(path),
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

    try:
        probe_data: dict = json.loads(result.stdout)
    except json.JSONDecodeError:
        return False

    streams: list[dict] = probe_data.get("streams", [])
    return any(s.get("codec_type") == "video" for s in streams)


# ---------------------------------------------------------------------------
# Frame extraction helper
# ---------------------------------------------------------------------------


def _extract_frame(video_path: Path, timestamp: float, output_path: Path) -> None:
    """Extract a single JPEG frame from *video_path* at *timestamp* seconds.

    Runs ``ffmpeg -ss {timestamp} -i {video_path} -frames:v 1 -q:v 2 {output_path}``.
    The ``-ss`` flag is placed *before* ``-i`` (input seeking) so ffmpeg seeks
    efficiently without decoding the entire preceding stream.

    Args:
        video_path: Path to the source video file.
        timestamp: Time in seconds at which to capture the frame.
        output_path: Destination path for the JPEG image (created by ffmpeg).

    Raises:
        subprocess.CalledProcessError: If ffmpeg exits with a non-zero status.
        FileNotFoundError: If ffmpeg is not found on PATH.
    """
    cmd: list[str] = [
        "ffmpeg",
        "-ss", str(timestamp),
        "-i", str(video_path),
        "-frames:v", "1",
        "-q:v", "2",
        str(output_path),
        "-y",       # overwrite without prompting
        "-loglevel", "error",
    ]
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Deduplication helper
# ---------------------------------------------------------------------------


def _deduplicate_timestamps(
    candidates: list[tuple[float, str]],
    window: float = _DEDUP_WINDOW_SECONDS,
) -> list[tuple[float, str]]:
    """Filter *candidates* so that no two retained timestamps are within *window* seconds.

    The list is assumed to be sorted ascending by timestamp (as it comes from
    iterating transcript segments in order).  The first occurrence in each
    cluster wins.

    Args:
        candidates: Pairs of (timestamp, trigger_text) in chronological order.
        window: Minimum allowed gap in seconds between retained timestamps.

    Returns:
        A filtered list preserving the first representative of each cluster.
    """
    kept: list[tuple[float, str]] = []
    last_kept: float = -window - 1.0  # sentinel guarantees first item is always kept

    for ts, text in candidates:
        if ts - last_kept >= window:
            kept.append((ts, text))
            last_kept = ts

    return kept


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_screenshots(
    video_path: str | Path,
    segments: list,
    output_dir: str | Path,
) -> list[Screenshot]:
    """Extract video frames at timestamps where speakers reference visual content.

    Scans each transcript segment's ``.text`` for NLP trigger phrases (e.g.
    "look at this graph", "as you can see").  When a trigger is found the
    segment's ``.start`` timestamp is used to grab a single JPEG frame via
    ffmpeg.  Triggers within 3 seconds of a previously extracted frame are
    skipped to avoid near-duplicate frames.

    If the source file has no video stream (e.g. audio-only ``.m4a``) the
    function returns an empty list immediately without calling ffmpeg.

    Args:
        video_path: Path to the recording (video or audio file).
        segments: List of ``Segment`` objects from the transcription module.
                  Each segment must expose ``.start`` (float, seconds),
                  ``.end`` (float, seconds), ``.text`` (str), and
                  ``.speaker`` (str).
        output_dir: Directory in which extracted JPEG images are saved.
                    Created if it does not already exist.

    Returns:
        A list of :class:`Screenshot` instances in chronological order.
        Empty when no triggers are found or the file has no video stream.

    Example::

        screenshots = extract_screenshots(
            "recording.mp4",
            transcript_segments,
            "output/screenshots",
        )
        for shot in screenshots:
            print(f"{shot.timestamp:.1f}s  {shot.trigger_text}  -> {shot.path}")
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)

    # ------------------------------------------------------------------
    # Guard: audio-only input
    # ------------------------------------------------------------------
    print("debrief/screenshots: probing media for video streams...", file=sys.stderr)
    if not _has_video_stream(video_path):
        print(
            "debrief/screenshots: no video stream found — skipping screenshot extraction.",
            file=sys.stderr,
        )
        return []

    # ------------------------------------------------------------------
    # Collect trigger candidates from all segments
    # ------------------------------------------------------------------
    candidates: list[tuple[float, str]] = []

    for segment in segments:
        text: str = getattr(segment, "text", "") or ""
        match = _TRIGGER_RE.search(text)
        if match:
            timestamp: float = float(getattr(segment, "start", 0.0))
            trigger_text: str = match.group(0)
            candidates.append((timestamp, trigger_text))

    if not candidates:
        print("debrief/screenshots: no visual-reference triggers detected.", file=sys.stderr)
        return []

    print(
        f"debrief/screenshots: {len(candidates)} trigger(s) found before deduplication.",
        file=sys.stderr,
    )

    # ------------------------------------------------------------------
    # Deduplicate triggers within the 3-second window
    # ------------------------------------------------------------------
    unique_candidates = _deduplicate_timestamps(candidates)

    print(
        f"debrief/screenshots: {len(unique_candidates)} unique frame(s) to extract "
        f"after deduplication.",
        file=sys.stderr,
    )

    # ------------------------------------------------------------------
    # Ensure output directory exists
    # ------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Extract frames
    # ------------------------------------------------------------------
    screenshots: list[Screenshot] = []

    for index, (timestamp, trigger_text) in enumerate(unique_candidates):
        filename = f"screenshot_{index:03d}_{timestamp:.3f}s.jpg"
        output_path = output_dir / filename

        print(
            f"debrief/screenshots: [{index + 1}/{len(unique_candidates)}] "
            f"extracting frame at {timestamp:.3f}s  ({trigger_text!r}) -> {output_path}",
            file=sys.stderr,
        )

        try:
            _extract_frame(video_path, timestamp, output_path)
        except FileNotFoundError:
            print(
                "debrief/screenshots: ERROR — ffmpeg not found on PATH; "
                "install ffmpeg and retry.",
                file=sys.stderr,
            )
            raise
        except subprocess.CalledProcessError as exc:
            print(
                f"debrief/screenshots: WARNING — ffmpeg failed for timestamp "
                f"{timestamp:.3f}s (exit {exc.returncode}); skipping.",
                file=sys.stderr,
            )
            continue

        screenshots.append(
            Screenshot(
                timestamp=timestamp,
                path=output_path.resolve(),
                trigger_text=trigger_text,
            )
        )

    print(
        f"debrief/screenshots: done — {len(screenshots)} screenshot(s) saved to {output_dir}.",
        file=sys.stderr,
    )
    return screenshots
