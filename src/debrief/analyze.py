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
import sys
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
    """Estimate token count for *text* using a words-to-tokens heuristic.

    Args:
        text: Arbitrary text string.

    Returns:
        Estimated token count (int).
    """
    words = len(text.split())
    return int(words * TOKENS_PER_WORD)


def _format_transcript(segments: list[SegmentLike]) -> str:
    """Format transcript segments as ``[Speaker]: text`` lines.

    Args:
        segments: Ordered list of transcript segments.

    Returns:
        Multi-line string with one ``[Speaker]: text`` entry per segment.
    """
    lines: list[str] = []
    for seg in segments:
        speaker = seg.speaker or "Unknown"
        lines.append(f"[{speaker}]: {seg.text.strip()}")
    return "\n".join(lines)


def _split_into_chunks(
    segments: list[SegmentLike],
    token_limit: int = CHUNK_TOKEN_LIMIT,
) -> list[list[SegmentLike]]:
    """Split segments into chunks whose estimated token count stays under *token_limit*.

    Chunks are broken at segment boundaries (never mid-sentence). Each chunk
    contains at least one segment, even if a single segment exceeds the limit.

    Args:
        segments: Full ordered list of transcript segments.
        token_limit: Maximum estimated tokens per chunk.

    Returns:
        List of segment groups, each group being a list of segments.
    """
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
    """Remove markdown code fences from *text*.

    Handles both `` ```json `` and bare `` ``` `` variants.

    Args:
        text: Raw LLM response text.

    Returns:
        Text with leading/trailing code fences removed and stripped.
    """
    text = text.strip()
    # Remove opening fence (```json or ```)
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    # Remove closing fence
    text = re.sub(r"\s*```\s*$", "", text)
    return text.strip()


def _extract_json_from_text(text: str) -> str | None:
    """Attempt to locate a JSON object inside *text* using a regex.

    Args:
        text: Arbitrary string that may contain a JSON object.

    Returns:
        The matched JSON substring, or ``None`` if no object is found.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else None


def _parse_llm_response(raw: str) -> dict[str, Any]:
    """Parse a JSON object from a raw LLM response string.

    Attempts three strategies in order:
    1. Strip code fences and parse directly.
    2. Regex-extract a ``{...}`` block and parse.
    3. Return a fallback dict with the raw text as ``summary``.

    Args:
        raw: Raw string returned by the Ollama model.

    Returns:
        Parsed dict with at least ``summary``, ``tasks``, and ``concepts``
        keys (empty lists for tasks/concepts on fallback).
    """
    # Strategy 1: strip fences and parse
    cleaned = _strip_code_fences(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Strategy 2: regex extraction
    extracted = _extract_json_from_text(raw)
    if extracted:
        try:
            return json.loads(extracted)
        except json.JSONDecodeError:
            pass

    # Strategy 3: fallback
    return {"summary": raw.strip(), "tasks": [], "concepts": []}


def _dict_to_analysis(data: dict[str, Any]) -> Analysis:
    """Convert a parsed LLM response dict into an :class:`Analysis` object.

    Missing or malformed fields are replaced with safe defaults.

    Args:
        data: Dict with ``summary``, ``tasks``, and ``concepts`` keys.

    Returns:
        Populated :class:`Analysis` instance.
    """
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


def _call_ollama(model: str, prompt: str) -> str:
    """Send *prompt* to the Ollama model and return the response text.

    Args:
        model: Ollama model name (e.g. ``"mistral"``).
        prompt: Full prompt string to send.

    Returns:
        Response content string from the model.

    Raises:
        ollama.ResponseError: If the Ollama API returns an error.
    """
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]


def _analyze_chunk(
    segments: list[SegmentLike],
    model: str,
) -> dict[str, Any]:
    """Run a single-chunk analysis against the LLM.

    Args:
        segments: Segments that make up this chunk.
        model: Ollama model name.

    Returns:
        Parsed dict with ``summary``, ``tasks``, and ``concepts`` keys.
    """
    transcript_text = _format_transcript(segments)
    prompt = _CHUNK_PROMPT.format(transcript_text=transcript_text)
    raw = _call_ollama(model, prompt)
    return _parse_llm_response(raw)


def _merge_chunks(
    chunk_results: list[dict[str, Any]],
    model: str,
) -> dict[str, Any]:
    """Ask the LLM to merge multiple chunk analyses into one cohesive result.

    Args:
        chunk_results: List of per-chunk parsed dicts.
        model: Ollama model name.

    Returns:
        Merged dict with ``summary``, ``tasks``, and ``concepts`` keys.
    """
    json_chunks = json.dumps(chunk_results, indent=2, ensure_ascii=False)
    prompt = _MERGE_PROMPT.format(json_chunks=json_chunks)
    raw = _call_ollama(model, prompt)
    return _parse_llm_response(raw)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze(
    segments: list,
    model: str = "mistral",
    duration: float = 0.0,  # noqa: ARG001 — reserved for future use
) -> Analysis:
    """Analyse a transcript and return a structured :class:`Analysis`.

    Sends transcript text to a local Ollama model and extracts a high-level
    summary, action items with optional assignees, and key concepts discussed.

    For short transcripts (estimated tokens < 6 000) the analysis is performed
    in a single LLM call. For longer transcripts the transcript is split into
    chunks of ~6 000 estimated tokens each; each chunk is analysed separately
    and a final merge pass synthesises the results.

    Progress messages are printed to *stderr* so they do not pollute stdout.

    Args:
        segments: Ordered list of transcript segment objects. Each must expose
            ``start`` (float), ``end`` (float), ``speaker`` (str), and
            ``text`` (str) attributes — matching :class:`SegmentLike`.
        model: Name of the Ollama model to use (default ``"mistral"``).
        duration: Total duration in seconds. Currently reserved for future
            use (e.g. cost estimation or progress display).

    Returns:
        An :class:`Analysis` with ``summary``, ``tasks``, and ``concepts``
        populated from the LLM response. On JSON parse failure the raw
        response is placed in ``summary`` and the other fields are empty.

    Raises:
        ollama.ResponseError: If the Ollama API returns an error response.
    """
    if not segments:
        return Analysis(summary="No transcript segments provided.", tasks=[], concepts=[])

    full_transcript = _format_transcript(segments)
    total_tokens = _estimate_tokens(full_transcript)

    if total_tokens <= SINGLE_PASS_TOKEN_LIMIT:
        print("Analyzing transcript...", file=sys.stderr)
        data = _analyze_chunk(segments, model)
        return _dict_to_analysis(data)

    # Multi-chunk path
    chunks = _split_into_chunks(segments, token_limit=CHUNK_TOKEN_LIMIT)
    total_chunks = len(chunks)

    chunk_results: list[dict[str, Any]] = []
    for idx, chunk in enumerate(chunks, start=1):
        print(f"Analyzing chunk {idx}/{total_chunks}...", file=sys.stderr)
        result = _analyze_chunk(chunk, model)
        chunk_results.append(result)

    if total_chunks == 1:
        # No merge needed — single chunk that happened to split cleanly
        return _dict_to_analysis(chunk_results[0])

    print("Merging chunk analyses...", file=sys.stderr)
    merged = _merge_chunks(chunk_results, model)
    return _dict_to_analysis(merged)
