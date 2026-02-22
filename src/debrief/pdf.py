"""PDF generation module for debrief reports.

Renders a Jinja2 HTML template and converts it to PDF via WeasyPrint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as _date
from pathlib import Path
from typing import Protocol, runtime_checkable
from urllib.request import pathname2url

from jinja2 import Environment, FileSystemLoader, select_autoescape
from weasyprint import HTML


# ---------------------------------------------------------------------------
# Protocols — structural typing for caller-supplied objects
# ---------------------------------------------------------------------------


@runtime_checkable
class Task(Protocol):
    """Any object with a task description and optional assignee."""

    description: str
    assignee: str | None


@runtime_checkable
class Segment(Protocol):
    """A single diarised transcript segment."""

    start: float  # seconds
    end: float    # seconds
    speaker: str
    text: str


@runtime_checkable
class Screenshot(Protocol):
    """A captured screenshot linked to a timestamp."""

    timestamp: float  # seconds from start
    path: Path
    trigger_text: str


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------


@dataclass
class TranscriptEntry:
    """A transcript segment enriched with formatted timestamp and screenshot.

    Attributes:
        timestamp: Formatted as ``HH:MM:SS``.
        speaker: Speaker label or name.
        text: Transcribed speech content.
        screenshot_path: ``file://`` URI string if a screenshot belongs at
            this point in the transcript, otherwise ``None``.
    """

    timestamp: str
    speaker: str
    text: str
    screenshot_path: str | None = field(default=None)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _seconds_to_hms(seconds: float) -> str:
    """Convert a floating-point second count to ``HH:MM:SS``.

    Args:
        seconds: Total seconds (non-negative).

    Returns:
        Zero-padded string in ``HH:MM:SS`` format.
    """
    total = max(0, int(seconds))
    h, remainder = divmod(total, 3600)
    m, s = divmod(remainder, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _format_duration(seconds: float) -> str:
    """Format a duration in seconds as a human-readable string.

    Examples:
        ``3661`` -> ``"1h 1m"``
        ``45``   -> ``"0h 45m"``

    Args:
        seconds: Total duration in seconds.

    Returns:
        String in ``Xh Ym`` format.
    """
    total = max(0, int(seconds))
    h, remainder = divmod(total, 3600)
    m = remainder // 60
    return f"{h}h {m}m"


def _path_to_file_uri(path: Path) -> str:
    """Convert an absolute ``Path`` to a ``file://`` URI.

    WeasyPrint requires ``file://`` URIs to load local images.

    Args:
        path: Absolute filesystem path to the image.

    Returns:
        URI string suitable for use in an ``<img src="...">`` attribute.
    """
    absolute = path.resolve()
    # pathname2url handles platform-specific separators
    return "file://" + pathname2url(str(absolute))


def _build_transcript_entries(
    segments: list[Segment],
    screenshots: list[Screenshot],
) -> list[TranscriptEntry]:
    """Merge transcript segments with screenshots.

    Each screenshot is attached to the segment whose time range
    ``[segment.start, segment.end)`` contains the screenshot's timestamp.
    If a screenshot falls between segments it is attached to the nearest
    preceding segment. Screenshots that precede all segments are attached
    to the first segment.

    Args:
        segments: Ordered list of diarised transcript segments.
        screenshots: List of screenshots with timestamps in seconds.

    Returns:
        Ordered list of :class:`TranscriptEntry` objects, one per segment.
    """
    # Index screenshots by the segment they belong to (0-based position).
    # A dict maps segment_index -> list[screenshot_path_uri].
    # We only attach the *first* screenshot per segment for simplicity;
    # if multiple screenshots land in the same segment the last one wins
    # (could be changed to a list if the template needs to render all).
    screenshot_map: dict[int, str] = {}

    if segments:
        for shot in screenshots:
            ts = shot.timestamp
            matched_idx = 0  # fallback: attach to first segment

            for idx, seg in enumerate(segments):
                if seg.start <= ts < seg.end:
                    matched_idx = idx
                    break
                if ts >= seg.start:
                    # Keep updating so we end up with the last segment whose
                    # start is <= the screenshot timestamp (nearest preceding).
                    matched_idx = idx

            uri = _path_to_file_uri(shot.path)
            screenshot_map[matched_idx] = uri

    entries: list[TranscriptEntry] = []
    for idx, seg in enumerate(segments):
        entries.append(
            TranscriptEntry(
                timestamp=_seconds_to_hms(seg.start),
                speaker=seg.speaker,
                text=seg.text.strip(),
                screenshot_path=screenshot_map.get(idx),
            )
        )

    return entries


def _split_paragraphs(text: str) -> list[str]:
    """Split a summary string into paragraph strings.

    Paragraphs are separated by one or more blank lines.

    Args:
        text: Raw summary text, possibly multi-paragraph.

    Returns:
        List of non-empty paragraph strings.
    """
    paragraphs: list[str] = []
    current_lines: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            current_lines.append(stripped)
        else:
            if current_lines:
                paragraphs.append(" ".join(current_lines))
                current_lines = []

    if current_lines:
        paragraphs.append(" ".join(current_lines))

    return paragraphs or [text.strip()]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_pdf(
    output_path: str | Path,
    summary: str,
    tasks: list[Task],
    concepts: list[str],
    segments: list[Segment],
    screenshots: list[Screenshot],
    duration: float,
    speakers: list[str],
    date: str | None = None,
    title: str | None = None,
) -> Path:
    """Render a debrief report and write it as a PDF file.

    The function builds :class:`TranscriptEntry` objects by merging *segments*
    with *screenshots*, renders the Jinja2 ``report.html`` template, and
    converts the resulting HTML to PDF via WeasyPrint.

    Args:
        output_path: Destination path for the generated PDF.
        summary: High-level summary (supports multiple paragraphs
            separated by blank lines).
        tasks: Sequence of task objects exposing ``description`` and
            ``assignee`` attributes.
        concepts: Ordered list of key concept strings.
        segments: Ordered list of transcript segments (each must expose
            ``start``, ``end``, ``speaker``, and ``text`` attributes).
        screenshots: List of screenshot objects (each must expose
            ``timestamp``, ``path``, and ``trigger_text`` attributes).
        duration: Total duration in seconds.
        speakers: Ordered list of speaker names for the header.
        date: ISO-format date (``YYYY-MM-DD``). Defaults to today.

    Returns:
        Resolved :class:`~pathlib.Path` pointing to the written PDF file.

    Raises:
        FileNotFoundError: If the HTML template cannot be located.
        weasyprint.urls.URLFetchingError: If WeasyPrint cannot load a
            referenced resource (e.g. a screenshot image).
    """
    output_path = Path(output_path).resolve()
    report_date = date or _date.today().isoformat()

    entries = _build_transcript_entries(segments, screenshots)

    # -- Logo ------------------------------------------------------------
    logo_path_uri = None
    logo_file = Path(__file__).parent / "static" / "logo.png"
    if logo_file.exists():
        logo_path_uri = _path_to_file_uri(logo_file)

    # -- Template --------------------------------------------------------
    templates_dir = Path(__file__).parent / "templates"
    if not templates_dir.is_dir():
        raise FileNotFoundError(
            f"Templates directory not found: {templates_dir}"
        )

    env = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template("report.html")

    report_title = title or "Debrief"

    html_content = template.render(
        title=report_title,
        logo_path=logo_path_uri,
        date=report_date,
        duration_formatted=_format_duration(duration),
        speakers=speakers,
        summary=summary,
        summary_paragraphs=_split_paragraphs(summary),
        tasks=tasks,
        concepts=concepts,
        entries=entries,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # base_url ensures relative paths inside the HTML resolve correctly;
    # we anchor it to the templates directory so any local assets there
    # (fonts, icons) are found automatically.
    HTML(
        string=html_content,
        base_url=str(templates_dir),
    ).write_pdf(str(output_path))

    return output_path
