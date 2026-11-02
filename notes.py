"""Offline meeting notes: a recording goes in, a transcript and summary come out. Nothing leaves this machine.

Usage: python3 notes.py RECORDING   (any format ffmpeg reads: webm, m4a, mp3, wav...)
Needs ffmpeg, whisper-cli (whisper.cpp) and a running Ollama.
"""
import json
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

WHISPER_MODEL = Path.home() / "models/whisper/ggml-small.en.bin"
LLM = "qwen3.5:4b"
NUM_CTX = 16384  # ponytail: about an hour of talk; Ollama silently drops anything past this, chunk longer meetings

PROMPT = """Summarize this meeting transcript. It has no speaker labels, so only use names that appear in it.
List every action item someone agreed to do. If the owner isn't named, use "not named". If there's no due date, use "none".

Transcript:
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "summary": {"type": "string"},
        "decisions": {"type": "array", "items": {"type": "string"}},
        "action_items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"task": {"type": "string"}, "owner": {"type": "string"}, "due": {"type": "string"}},
                "required": ["task", "owner", "due"],
            },
        },
    },
    "required": ["title", "summary", "decisions", "action_items"],
}


def run(*cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"{cmd[0]} failed:\n{result.stderr}")


def transcribe(recording):
    """Return [(start_seconds, text), ...]. The start times let the app jump to that point in the audio."""
    with tempfile.TemporaryDirectory() as tmp:
        wav, out = f"{tmp}/audio.wav", f"{tmp}/transcript"
        run("ffmpeg", "-y", "-loglevel", "error", "-i", str(recording),
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", wav)
        run("whisper-cli", "-np", "-m", str(WHISPER_MODEL), "-f", wav, "-oj", "-of", out)
        segments = json.loads(Path(f"{out}.json").read_text())["transcription"]
    return [(s["offsets"]["from"] / 1000, s["text"].strip()) for s in segments]


def summarize(transcript):
    """Return {"title", "summary", "decisions", "action_items"} from the local LLM."""
    body = {
        "model": LLM,
        "prompt": PROMPT + transcript,
        "format": SCHEMA,
        "stream": False,
        "think": False,  # thinking was just as accurate in testing, but 25x slower
        "options": {"temperature": 0, "num_ctx": NUM_CTX},
    }
    request = urllib.request.Request("http://localhost:11434/api/generate", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    return json.loads(json.load(urllib.request.urlopen(request))["response"])


def to_markdown(notes):
    lines = [f"# {notes['title']}", "", notes["summary"], "", "## Decisions"]
    lines += [f"- {d}" for d in notes["decisions"]]
    lines += ["", "## Action items"]
    lines += [f"- {a['task']} (owner: {a['owner']}, due: {a['due']})" for a in notes["action_items"]]
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    segments = transcribe(sys.argv[1])
    print(to_markdown(summarize("\n".join(text for _, text in segments))))
