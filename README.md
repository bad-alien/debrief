# debrief

CLI tool that turns meeting recordings into structured PDF reports — entirely offline using Whisper and Ollama.

**Input:** audio or video file
**Output:** PDF with transcript, speaker labels, executive summary, action items, key concepts, and auto-captured screenshots

## Prerequisites

- Python 3.11+
- [ffmpeg](https://ffmpeg.org/) on PATH
- [Ollama](https://ollama.com/) running locally with a model pulled (default: `mistral`)
- A [HuggingFace token](https://huggingface.co/settings/tokens) for pyannote speaker diarization

## Install

```bash
pip install -e .
```

## Usage

```
debrief <recording> [options]
```

Place recordings in `raw/`, reports are written to `debriefs/` by default.

### Examples

```bash
# Basic — uses default Whisper large-v3 and Mistral
debrief raw/standup.mp4

# Name the speakers
debrief raw/standup.mp4 --speakers "Alice,Bob,Carol"

# Use a different LLM and smaller Whisper model
debrief raw/standup.mp4 --model llama3 --whisper-model medium

# Custom output path
debrief raw/standup.mp4 --output debriefs/custom.pdf

# Force CPU
debrief raw/call.wav --device cpu
```

### Options

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--speakers` | `-s` | — | Comma-separated speaker names |
| `--model` | `-m` | `mistral` | Ollama model for analysis |
| `--whisper-model` | `-w` | `large-v3` | Whisper model size |
| `--output` | `-o` | `debriefs/<name>_debrief.pdf` | Output PDF path |
| `--device` | `-d` | `auto` | Compute device (`auto`, `cpu`, `cuda`) |

## Pipeline

1. **Transcribe** — extracts audio, runs Whisper with word timestamps
2. **Diarize** — identifies speakers via pyannote, maps to provided names
3. **Screenshot** — detects visual references in transcript ("look at this"), captures video frames
4. **Analyze** — sends transcript to Ollama for summary, action items, key concepts
5. **Render** — generates a styled PDF via Jinja2 + WeasyPrint

## Project layout

```
raw/           Recordings (not tracked)
debriefs/      Generated PDFs (not tracked)
src/debrief/   Source code
tests/         Test suite
```

## Supported formats

**Video:** `.mp4`, `.mkv`, `.webm`, `.avi`, `.mov`
**Audio:** `.wav`, `.mp3`, `.ogg`, `.flac`, `.m4a`
