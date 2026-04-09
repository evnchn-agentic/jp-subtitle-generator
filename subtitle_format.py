"""
Native subtitle format v1 — replaces SRT.
JSON-based, supports simultaneous speakers, colors, translations.

Format:
{
  "version": 1,
  "video_id": "...",
  "duration": 140.0,
  "generated_at": "2026-04-09T09:10:00",
  "engine": "gemini_coreml",
  "events": [
    {
      "time": 5.5,           # frame-accurate appearance time
      "lines": [             # ALL text visible at this moment
        {"text": "おおもり！", "color": "pink", "zh": "大份！"},
        {"text": "私はメガ盛！", "color": "blue", "zh": "我要特大份！"},
      ]
    },
    {
      "time": 7.0,           # next change — previous lines gone, new ones appear
      "lines": [
        {"text": "大盛とメガ盛と…", "color": "blue", "zh": "大份和特大份..."},
      ]
    },
    ...
  ]
}

Viewer logic:
  At time T, find the latest event where event.time <= T.
  Display ALL lines from that event simultaneously.
  When the next event arrives, replace ALL lines.
"""

import json
from pathlib import Path
from datetime import datetime

VERSION = 1

def create_subtitle_file(video_id, duration, events, engine="gemini_coreml"):
    """Create a native subtitle file from Gemini read results.
    
    events: list of (time_ms, [{text, color, zh}]) tuples, sorted by time.
    """
    data = {
        "version": VERSION,
        "video_id": video_id,
        "duration": duration,
        "generated_at": datetime.now().isoformat(),
        "engine": engine,
        "events": [
            {
                "time": round(t_ms / 1000, 3),
                "lines": [
                    {"text": s.get("text", ""), 
                     "color": s.get("color", "white"),
                     "zh": s.get("zh", "")}
                    for s in subs if s.get("text", "").strip()
                ]
            }
            for t_ms, subs in events
            if subs  # skip empty frames
        ]
    }
    
    path = Path(f"subs/{video_id}.json")
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    return path


def load_subtitle_file(path):
    """Load native subtitle file. Returns list of {time, lines} events."""
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if data.get("version", 0) < VERSION:
        print(f"Warning: old format v{data.get('version')}, regenerate recommended")
    return data


def get_active_lines(events, t):
    """Get all visible subtitle lines at time t.
    Returns list of {text, color, zh} dicts."""
    active_event = None
    for event in events:
        if event["time"] <= t:
            active_event = event
        else:
            break
    return active_event["lines"] if active_event else []
