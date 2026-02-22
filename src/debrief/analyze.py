"""LLM analysis module for debrief.

Sends transcript text to a local Ollama model and extracts a high-level
summary, action items with optional assignees, and key concepts discussed.

Chunking strategy
-----------------
- Estimate token count at ~0.75 tokens per word.
- If the full transcript fits under 6 000 estimated tokens, process in one
  shot.
- Otherwise split at Segment boundaries into chunks of ~6 000 estimated tokens
  each, request a partial analysis per chunk, then do a final merge pass that
  asks the LLM to synthesise the chunk results into a single cohesive report.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import ollama

# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------

TOKENS_PER_WORD: float = 0.75
CHUNK_TOKEN_LIMIT: int = 6_000
SINGLE_PASS_TOKEN_LIMIT: int = 6_000


@dataclass
class Task:
    """A single action item extracted from the transcript.

    Attributes:
        description: What needs to be done.
        assignee: Speaker name if identifiable, otherwise ``None``.
    """

    description: str
    assignee: str | None = None


@dataclass
class Analysis:
    """Full analysis result returned to callers.

    Attributes:
        summary: 2-3 paragraph high-level overview.
        tasks: Action items extracted from the transcript.
        concepts: Key topics and concepts discussed.
    """

    summary: str
    tasks: list[Task] = field(default_factory=list)
    concepts: list[str] = field(default_factory=list)


@dataclass
class _LLMCall:
    """Internal: captures an Ollama chat response with token/timing stats."""

    content: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    eval_duration_seconds: float | None = None
    total_duration_seconds: float | None = None
    load_duration_seconds: float | None = None


@dataclass
class AnalysisResult:
    """Public wrapper around :class:`Analysis` with pipeline metadata."""

    analysis: Analysis
    json_parsed_cleanly: bool
    chunk_count: int
    prompt_tokens: int
    completion_tokens: int
    eval_duration_seconds: float
    total_duration_seconds: float
    load_duration_seconds: float


# ---------------------------------------------------------------------------
# Structural type for transcript segments
# ---------------------------------------------------------------------------


@runtime_checkable
class SegmentLike(Protocol):
    """Structural protocol for transcript segment objects.

    Any object with ``start``, ``end``, ``speaker``, and ``text`` attributes
    satisfies this protocol — no explicit inheritance required.
    """

    start: float
    end: float
    speaker: str
    text: str


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_CHUNK_PROMPT = """\
Analyze this transcript and return a JSON object with:
- "summary": a 2-3 paragraph high-level summary
- "tasks": array of {{"description": "...", "assignee": "..." or null}}
- "concepts": array of strings (key topics/concepts discussed)

Transcript:
{transcript_text}"""

_MERGE_PROMPT = """\
Combine these analysis chunks into a single cohesive report. Return JSON with:
- "summary": unified 2-3 paragraph summary (not just concatenation)
- "tasks": deduplicated task list as {{"description": "...", "assignee": "..." or null}}
- "concepts": deduplicated list of key concepts

Chunk analyses:
{json_chunks}"""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _estimate_tokens(text: str) -> int:
    """Estimate token count for *text* using a words-to-tokens heuristic."""
    words = len(text.split())
    return int(words * TOKENS_PER_WORD)


def _format_transcript(segments: list[SegmentLike]) -> str:
    """Format transcript segments as ``[Speaker]: text`` lines."""
    lines: list[str] = []
    for seg in segments:
        speaker = seg.speaker or "Unknown"
        lines.append(f"[{speaker}]: {seg.text.strip()}")
    return "\n".join(lines)


def _split_into_chunks(
    segments: list[SegmentLike],
    token_limit: int = CHUNK_TOKEN_LIMIT,
) -> list[list[SegmentLike]]:
    """Split segments into chunks whose estimated token count stays under *token_limit*."""
    chunks: list[list[SegmentLike]] = []
    current_chunk: list[SegmentLike] = []
    current_tokens: int = 0

    for seg in segments:
        seg_tokens = _estimate_tokens(seg.text)
        if current_chunk and current_tokens + seg_tokens > token_limit:
            chunks.append(current_chunk)
            current_chunk = [seg]
            current_tokens = seg_tokens
        else:
            current_chunk.append(seg)
            current_tokens += seg_tokens

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences from *text*."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    return text.strip()


def _extract_json_from_text(text: str) -> str | None:
    """Attempt to locate a JSON object inside *text* using a regex."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else None


def _parse_llm_response(raw: str) -> tuple[dict[str, Any], bool]:
    """Parse a JSON object from a raw LLM response string.

    Returns:
        A tuple of (parsed_dict, json_parsed_cleanly).
    """
    # Strategy 1: strip fences and parse
    cleaned = _strip_code_fences(raw)
    try:
        return json.loads(cleaned), True
    except json.JSONDecodeError:
        pass

    # Strategy 2: regex extraction
    extracted = _extract_json_from_text(raw)
    if extracted:
        try:
            return json.loads(extracted), True
        except json.JSONDecodeError:
            pass

    # Strategy 3: fallback
    return {"summary": raw.strip(), "tasks": [], "concepts": []}, False


def _dict_to_analysis(data: dict[str, Any]) -> Analysis:
    """Convert a parsed LLM response dict into an :class:`Analysis` object."""
    summary = str(data.get("summary", "")).strip()

    tasks: list[Task] = []
    for item in data.get("tasks", []):
        if isinstance(item, dict):
            desc = str(item.get("description", "")).strip()
            assignee = item.get("assignee") or None
            if assignee is not None:
                assignee = str(assignee).strip() or None
            if desc:
                tasks.append(Task(description=desc, assignee=assignee))

    concepts: list[str] = []
    for concept in data.get("concepts", []):
        c = str(concept).strip()
        if c:
            concepts.append(c)

    return Analysis(summary=summary, tasks=tasks, concepts=concepts)


def _nanos_to_seconds(nanos: int | None) -> float | None:
    """Convert nanoseconds to seconds, returning None if input is None."""
    if nanos is None:
        return None
    return nanos / 1_000_000_000


def _call_ollama(model: str, prompt: str) -> _LLMCall:
    """Send *prompt* to the Ollama model and return an :class:`_LLMCall`."""
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return _LLMCall(
        content=response["message"]["content"],
        prompt_tokens=response.get("prompt_eval_count"),
        completion_tokens=response.get("eval_count"),
        eval_duration_seconds=_nanos_to_seconds(response.get("eval_duration")),
        total_duration_seconds=_nanos_to_seconds(response.get("total_duration")),
        load_duration_seconds=_nanos_to_seconds(response.get("load_duration")),
    )


def _accumulate_stats(calls: list[_LLMCall]) -> dict[str, float]:
    """Sum token and timing stats across multiple LLM calls."""
    prompt_tokens = 0
    completion_tokens = 0
    eval_duration = 0.0
    total_duration = 0.0
    load_duration = 0.0
    for call in calls:
        prompt_tokens += call.prompt_tokens or 0
        completion_tokens += call.completion_tokens or 0
        eval_duration += call.eval_duration_seconds or 0.0
        total_duration += call.total_duration_seconds or 0.0
        load_duration += call.load_duration_seconds or 0.0
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "eval_duration_seconds": eval_duration,
        "total_duration_seconds": total_duration,
        "load_duration_seconds": load_duration,
    }


def _analyze_chunk(
    segments: list[SegmentLike],
    model: str,
) -> tuple[dict[str, Any], bool, _LLMCall]:
    """Run a single-chunk analysis against the LLM.

    Returns:
        Tuple of (parsed_dict, json_parsed_cleanly, llm_call).
    """
    transcript_text = _format_transcript(segments)
    prompt = _CHUNK_PROMPT.format(transcript_text=transcript_text)
    call = _call_ollama(model, prompt)
    data, clean = _parse_llm_response(call.content)
    return data, clean, call


def _merge_chunks(
    chunk_results: list[dict[str, Any]],
    model: str,
) -> tuple[dict[str, Any], bool, _LLMCall]:
    """Ask the LLM to merge multiple chunk analyses into one cohesive result.

    Returns:
        Tuple of (merged_dict, json_parsed_cleanly, llm_call).
    """
    json_chunks = json.dumps(chunk_results, indent=2, ensure_ascii=False)
    prompt = _MERGE_PROMPT.format(json_chunks=json_chunks)
    call = _call_ollama(model, prompt)
    data, clean = _parse_llm_response(call.content)
    return data, clean, call


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze(
    segments: list,
    model: str = "hf.co/Qwen/Qwen3-14B-GGUF:Q4_K_M",
    duration: float = 0.0,  # noqa: ARG001 — reserved for future use
) -> AnalysisResult:
    """Analyse a transcript and return an :class:`AnalysisResult`.

    Args:
        segments: Ordered list of transcript segment objects.
        model: Name of the Ollama model to use.
        duration: Total duration in seconds (reserved for future use).

    Returns:
        An :class:`AnalysisResult` wrapping the :class:`Analysis` plus
        metadata about the LLM calls (token counts, timing, parse quality).

    Raises:
        ollama.ResponseError: If the Ollama API returns an error response.
    """
    if not segments:
        return AnalysisResult(
            analysis=Analysis(summary="No transcript segments provided.", tasks=[], concepts=[]),
            json_parsed_cleanly=True,
            chunk_count=0,
            prompt_tokens=0,
            completion_tokens=0,
            eval_duration_seconds=0.0,
            total_duration_seconds=0.0,
            load_duration_seconds=0.0,
        )

    full_transcript = _format_transcript(segments)
    total_tokens = _estimate_tokens(full_transcript)

    all_calls: list[_LLMCall] = []
    all_clean = True

    if total_tokens <= SINGLE_PASS_TOKEN_LIMIT:
        data, clean, call = _analyze_chunk(segments, model)
        all_calls.append(call)
        all_clean = clean
        chunk_count = 1
        analysis = _dict_to_analysis(data)
    else:
        # Multi-chunk path
        chunks = _split_into_chunks(segments, token_limit=CHUNK_TOKEN_LIMIT)
        chunk_count = len(chunks)

        chunk_results: list[dict[str, Any]] = []
        for chunk in chunks:
            data, clean, call = _analyze_chunk(chunk, model)
            all_calls.append(call)
            if not clean:
                all_clean = False
            chunk_results.append(data)

        if chunk_count == 1:
            analysis = _dict_to_analysis(chunk_results[0])
        else:
            merged, clean, call = _merge_chunks(chunk_results, model)
            all_calls.append(call)
            if not clean:
                all_clean = False
            analysis = _dict_to_analysis(merged)

    stats = _accumulate_stats(all_calls)
    return AnalysisResult(
        analysis=analysis,
        json_parsed_cleanly=all_clean,
        chunk_count=chunk_count,
        prompt_tokens=int(stats["prompt_tokens"]),
        completion_tokens=int(stats["completion_tokens"]),
        eval_duration_seconds=stats["eval_duration_seconds"],
        total_duration_seconds=stats["total_duration_seconds"],
        load_duration_seconds=stats["load_duration_seconds"],
    )
