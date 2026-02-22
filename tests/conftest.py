"""Shared fixtures for debrief tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.visual import VisualValidator


# ---------------------------------------------------------------------------
# Mock data classes (satisfy the Protocol interfaces in pdf.py)
# ---------------------------------------------------------------------------


@dataclass
class MockSegment:
    start: float
    end: float
    speaker: str
    text: str


@dataclass
class MockTask:
    description: str
    assignee: str | None = None


@dataclass
class MockScreenshot:
    timestamp: float
    path: Path
    trigger_text: str


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_segments() -> list[MockSegment]:
    """A handful of realistic transcript segments."""
    return [
        MockSegment(0.0, 12.5, "Alice", "Good morning everyone, let's get started with the sprint review."),
        MockSegment(12.5, 28.0, "Bob", "Sure. So last week I finished the authentication module and started on the dashboard."),
        MockSegment(28.0, 45.3, "Alice", "Great. Can you walk us through the changes? Let me share my screen so we can look at this together."),
        MockSegment(45.3, 62.1, "Bob", "As you can see, the login flow now supports OAuth and we added rate limiting."),
        MockSegment(62.1, 80.0, "Carol", "I have a question about the rate limiting. What happens when the limit is exceeded?"),
        MockSegment(80.0, 95.5, "Bob", "Good question. We return a 429 status code with a Retry-After header."),
        MockSegment(95.5, 110.0, "Alice", "Perfect. Let's move on to the next item. Carol, can you give us an update on the API redesign?"),
        MockSegment(110.0, 130.0, "Carol", "Yes. I've refactored the endpoints to follow REST conventions. Here is the new schema."),
    ]


@pytest.fixture()
def sample_tasks() -> list[MockTask]:
    return [
        MockTask("Deploy authentication module to staging", "Bob"),
        MockTask("Write integration tests for OAuth flow", "Bob"),
        MockTask("Review API schema changes", "Alice"),
        MockTask("Update client SDK for new endpoints", None),
    ]


@pytest.fixture()
def sample_concepts() -> list[str]:
    return [
        "OAuth 2.0 authentication",
        "Rate limiting (429 + Retry-After)",
        "REST API redesign",
        "Sprint review process",
        "Client SDK compatibility",
    ]


@pytest.fixture()
def sample_screenshots(tmp_path: Path) -> list[MockScreenshot]:
    """Create dummy screenshot images and return MockScreenshot objects."""
    shots: list[MockScreenshot] = []
    for i, (ts, trigger) in enumerate([
        (30.0, "look at this"),
        (46.0, "as you can see"),
    ]):
        img_path = tmp_path / f"shot_{i}.jpg"
        # Create a minimal valid JPEG (smallest possible: 2x1 pixel)
        img_path.write_bytes(
            b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
            b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t"
            b"\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a"
            b"\x1f\x1e\x1d\x1a\x1c\x1c $.\' \",#\x1c\x1c(7),01444\x1f\'9=82<.342"
            b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x02\x01\x01\x11\x00"
            b"\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b"
            b"\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05"
            b"\x04\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05\x12!1A\x06"
            b"\x13Qa\x07\"q\x142\x81\x91\xa1\x08#B\xb1\xc1\x15R\xd1\xf0$3br"
            b"\x82\t\n\x16\x17\x18\x19\x1a%&\'()*456789:CDEFGHIJSTUVWXYZcdef"
            b"ghijstuvwxyz\x83\x84\x85\x86\x87\x88\x89\x8a\x92\x93\x94\x95"
            b"\x96\x97\x98\x99\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xb2"
            b"\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xc2\xc3\xc4\xc5\xc6\xc7\xc8"
            b"\xc9\xca\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xe1\xe2\xe3\xe4"
            b"\xe5\xe6\xe7\xe8\xe9\xea\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa"
            b"\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xfb\xd2\x8a+\xff\xd9"
        )
        shots.append(MockScreenshot(timestamp=ts, path=img_path, trigger_text=trigger))
    return shots


@pytest.fixture()
def visual_validator() -> VisualValidator:
    """Provide a VisualValidator instance pointed at the snapshots directory."""
    snapshots_dir = Path(__file__).parent / "snapshots"
    return VisualValidator(snapshots_dir)
