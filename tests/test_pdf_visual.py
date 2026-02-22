"""Visual tests for PDF report generation.

Each test generates a real PDF via WeasyPrint (asserting correctness), then
renders the same HTML template with headless Chrome to capture a visual
snapshot under ``tests/snapshots/`` for human review.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from jinja2 import Environment, FileSystemLoader, select_autoescape

from debrief.pdf import (
    _build_transcript_entries,
    _format_duration,
    _split_paragraphs,
    generate_pdf,
)
from tests.conftest import MockScreenshot, MockSegment, MockTask
from tests.visual import VisualValidator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "src" / "debrief" / "templates"


def _assert_pdf_valid(pdf_path: Path) -> None:
    """Assert the PDF exists and has non-trivial content."""
    assert pdf_path.exists(), f"PDF was not created: {pdf_path}"
    size = pdf_path.stat().st_size
    assert size > 500, f"PDF suspiciously small ({size} bytes): {pdf_path}"


def _render_html(
    summary: str,
    tasks: list,
    concepts: list[str],
    segments: list,
    screenshots: list,
    duration: float,
    speakers: list[str],
    date: str = "2026-02-22",
) -> str:
    """Render the report HTML template with the given data (no PDF conversion)."""
    entries = _build_transcript_entries(segments, screenshots)
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template("report.html")
    return template.render(
        date=date,
        duration_formatted=_format_duration(duration),
        speakers=speakers,
        summary=summary,
        summary_paragraphs=_split_paragraphs(summary),
        tasks=tasks,
        concepts=concepts,
        entries=entries,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFullReport:
    """Generate a complete report with all sections populated."""

    def test_full_report_renders(
        self,
        tmp_path: Path,
        sample_segments: list[MockSegment],
        sample_tasks: list[MockTask],
        sample_concepts: list[str],
        sample_screenshots: list[MockScreenshot],
        visual_validator: VisualValidator,
    ) -> None:
        summary = (
            "This sprint review covered progress on the authentication module "
            "and the API redesign. Bob demonstrated the new OAuth login flow "
            "with rate limiting, and Carol presented the refactored REST endpoints.\n\n"
            "The team agreed to deploy the auth module to staging and update "
            "the client SDK to match the new API schema."
        )
        speakers = ["Alice", "Bob", "Carol"]
        pdf_path = tmp_path / "full_report.pdf"

        result = generate_pdf(
            output_path=pdf_path,
            summary=summary,
            tasks=sample_tasks,
            concepts=sample_concepts,
            segments=sample_segments,
            screenshots=sample_screenshots,
            duration=130.0,
            speakers=speakers,
            date="2026-02-22",
        )

        _assert_pdf_valid(result)

        # Capture HTML rendering for visual review
        html = _render_html(
            summary, sample_tasks, sample_concepts,
            sample_segments, sample_screenshots, 130.0, speakers,
        )
        snapshot = visual_validator.screenshot_html(html, "full_report")
        if snapshot:
            assert snapshot.stat().st_size > 0


class TestEmptySections:
    """Generate a report with no tasks, concepts, or screenshots."""

    def test_empty_sections(
        self,
        tmp_path: Path,
        sample_segments: list[MockSegment],
        visual_validator: VisualValidator,
    ) -> None:
        summary = "A brief meeting with no action items identified."
        speakers = ["Alice", "Bob", "Carol"]
        pdf_path = tmp_path / "empty_sections.pdf"

        result = generate_pdf(
            output_path=pdf_path,
            summary=summary,
            tasks=[],
            concepts=[],
            segments=sample_segments,
            screenshots=[],
            duration=130.0,
            speakers=speakers,
            date="2026-02-22",
        )

        _assert_pdf_valid(result)

        html = _render_html(summary, [], [], sample_segments, [], 130.0, speakers)
        snapshot = visual_validator.screenshot_html(html, "empty_sections")
        if snapshot:
            assert snapshot.stat().st_size > 0


class TestLongTranscript:
    """Generate a report with many segments to exercise pagination."""

    def test_long_transcript(
        self,
        tmp_path: Path,
        visual_validator: VisualValidator,
    ) -> None:
        speakers = ["Alice", "Bob", "Carol"]
        segments = []
        for i in range(100):
            start = i * 10.0
            segments.append(
                MockSegment(
                    start=start,
                    end=start + 9.5,
                    speaker=speakers[i % 3],
                    text=f"This is segment number {i + 1}. We are discussing item {i + 1} on the agenda, "
                    f"which involves reviewing the implementation details and gathering feedback from the team.",
                )
            )

        summary = "An extended meeting covering 100 agenda items."
        tasks = [MockTask("Follow up on all 100 items", "Alice")]
        concepts = ["Scalability", "Code review process"]
        pdf_path = tmp_path / "long_transcript.pdf"

        result = generate_pdf(
            output_path=pdf_path,
            summary=summary,
            tasks=tasks,
            concepts=concepts,
            segments=segments,
            screenshots=[],
            duration=1000.0,
            speakers=speakers,
            date="2026-02-22",
        )

        _assert_pdf_valid(result)
        assert result.stat().st_size > 5000

        html = _render_html(summary, tasks, concepts, segments, [], 1000.0, speakers)
        snapshot = visual_validator.screenshot_html(html, "long_transcript")
        if snapshot:
            assert snapshot.stat().st_size > 0


class TestWithScreenshots:
    """Generate a report that embeds screenshot images in the transcript."""

    def test_with_screenshots(
        self,
        tmp_path: Path,
        sample_segments: list[MockSegment],
        sample_screenshots: list[MockScreenshot],
        visual_validator: VisualValidator,
    ) -> None:
        summary = "Meeting with visual references captured as screenshots."
        speakers = ["Alice", "Bob", "Carol"]
        pdf_path = tmp_path / "with_screenshots.pdf"

        result = generate_pdf(
            output_path=pdf_path,
            summary=summary,
            tasks=[],
            concepts=[],
            segments=sample_segments,
            screenshots=sample_screenshots,
            duration=130.0,
            speakers=speakers,
            date="2026-02-22",
        )

        _assert_pdf_valid(result)

        html = _render_html(
            summary, [], [], sample_segments, sample_screenshots, 130.0, speakers,
        )
        snapshot = visual_validator.screenshot_html(html, "with_screenshots")
        if snapshot:
            assert snapshot.stat().st_size > 0
