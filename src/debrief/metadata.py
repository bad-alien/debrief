"""Run metadata for debrief pipeline executions.

Captures timing, token counts, model info, and quality indicators.
Written as a JSON file alongside each generated PDF.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class StepTiming:
    """Timing information for a single pipeline step."""

    name: str
    duration_seconds: float


@dataclass
class OllamaStats:
    """Token and timing statistics from the Ollama API."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    eval_duration_seconds: float | None = None
    total_duration_seconds: float | None = None
    load_duration_seconds: float | None = None


@dataclass
class RunMetadata:
    """Full metadata snapshot for a single pipeline run."""

    started_at: str  # ISO-8601 UTC
    total_wall_seconds: float
    whisper_model: str
    llm_model: str
    transcript_duration_seconds: float
    segment_count: int
    speaker_count: int
    screenshot_count: int
    task_count: int
    concept_count: int
    json_parsed_cleanly: bool
    chunk_count: int
    ollama: OllamaStats | None = None
    steps: list[StepTiming] = field(default_factory=list)

    def to_json(self) -> str:
        """Serialise to a pretty-printed JSON string."""
        data = asdict(self)
        return json.dumps(data, indent=2, ensure_ascii=False)

    def write(self, pdf_path: Path) -> Path:
        """Write JSON metadata next to *pdf_path* with a ``.json`` extension.

        Args:
            pdf_path: Path to the generated PDF report.

        Returns:
            Path to the written JSON file.
        """
        json_path = pdf_path.with_suffix(".json")
        json_path.write_text(self.to_json(), encoding="utf-8")
        return json_path
