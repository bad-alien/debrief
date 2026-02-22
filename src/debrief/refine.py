"""Post-analysis transcript refinement.

Applies two transformations to the raw transcript after LLM analysis:

1. **Speaker renaming** — replaces generic labels like "Speaker 1" with real
   names discovered by the LLM (e.g. "Martin", "Miguel").
2. **Segment merging** — collapses consecutive segments from the same speaker
   into single, longer turns when the gap between them is small.
"""

from __future__ import annotations

from debrief.analyze import Analysis
from debrief.transcribe import Segment


def rename_speakers(
    segments: list[Segment],
    speaker_mapping: dict[str, str],
) -> list[Segment]:
    """Return a copy of *segments* with speaker labels replaced per *speaker_mapping*.

    Args:
        segments: Original transcript segments.
        speaker_mapping: Maps generic labels (e.g. ``"Speaker 1"``) to real
            names (e.g. ``"Martin"``).  Unmapped labels are left unchanged.

    Returns:
        New list of :class:`Segment` objects — originals are not mutated.
    """
    if not speaker_mapping:
        return list(segments)

    return [
        Segment(
            start=seg.start,
            end=seg.end,
            speaker=speaker_mapping.get(seg.speaker, seg.speaker),
            text=seg.text,
        )
        for seg in segments
    ]


def merge_consecutive_segments(
    segments: list[Segment],
    max_gap: float = 2.0,
) -> list[Segment]:
    """Merge consecutive segments from the same speaker when the gap is small.

    Args:
        segments: Transcript segments (assumed chronologically ordered).
        max_gap: Maximum silence gap in seconds between two segments for them
            to be merged.  Defaults to ``2.0``.

    Returns:
        New list of :class:`Segment` objects with adjacent same-speaker
        segments collapsed.
    """
    if not segments:
        return []

    merged: list[Segment] = []
    current = segments[0]

    for seg in segments[1:]:
        gap = seg.start - current.end
        if seg.speaker == current.speaker and gap <= max_gap:
            # Extend the current segment
            current = Segment(
                start=current.start,
                end=seg.end,
                speaker=current.speaker,
                text=current.text + " " + seg.text,
            )
        else:
            merged.append(current)
            current = seg

    merged.append(current)
    return merged


def refine_transcript(
    segments: list[Segment],
    analysis: Analysis,
) -> list[Segment]:
    """Apply speaker renaming and segment merging to *segments*.

    This is the public entry point — call it after LLM analysis and before
    PDF generation.

    Args:
        segments: Raw transcript segments from the transcription step.
        analysis: LLM analysis result containing ``speaker_mapping``.

    Returns:
        Refined list of :class:`Segment` objects.
    """
    renamed = rename_speakers(segments, analysis.speaker_mapping)
    return merge_consecutive_segments(renamed)
