"""
Full subtitle pipeline: OCR + Whisper + Fusion + Translation + Cross-check player.
Usage: python pipeline.py video.webm
"""

import cv2
import numpy as np
import os
import sys
import json
import time
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from difflib import SequenceMatcher
from pathlib import Path

os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'


@dataclass
class SubSegment:
    start: float
    end: float
    text: str
    source: str  # 'ocr', 'whisper', 'fused'
    confidence: float = 1.0
    translation: str = ''


# ── OCR ──────────────────────────────────────────────

def run_ocr(video_path, interval_ms=1000):
    """Run PP-OCR mobile on video zones. Returns list of SubSegment."""
    from paddleocr import PaddleOCR

    print("[OCR] Loading PP-OCR mobile...")
    ocr = PaddleOCR(
        text_detection_model_name='PP-OCRv5_mobile_det',
        text_recognition_model_name='PP-OCRv5_mobile_rec',
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    cap = cv2.VideoCapture(video_path)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)

    readings = []
    t0 = time.time()
    for t_ms in range(0, int(dur * 1000), interval_ms):
        cap.set(cv2.CAP_PROP_POS_MSEC, t_ms)
        ret, frame = cap.read()
        if not ret:
            break

        t_s = t_ms / 1000.0
        texts = []
        for zy1, zy2 in [(0, int(h * 0.28)), (int(h * 0.78), int(h * 0.96))]:
            zone = frame[zy1:zy2, :]
            tmp = '/tmp/_ocr_z.jpg'
            cv2.imwrite(tmp, zone)
            result = ocr.predict(tmp)
            for r in result:
                for box, text, score in zip(r['dt_polys'], r['rec_texts'], r['rec_scores']):
                    if score > 0.4 and len(text) > 1:
                        texts.append(text)

        combined = ' '.join(texts)
        readings.append((t_s, combined))

        n = t_ms // interval_ms + 1
        total = int(dur * 1000) // interval_ms
        if n % 10 == 0:
            print(f"[OCR] {n}/{total} frames...")

    cap.release()
    print(f"[OCR] Done in {time.time() - t0:.0f}s")

    # Deduplicate
    return _deduplicate(readings, interval_ms / 1000.0, 'ocr')


def _deduplicate(readings, interval_s, source):
    """Group consecutive similar readings into segments."""
    if not readings:
        return []

    groups = [[readings[0]]]
    for t, txt in readings[1:]:
        prev = groups[-1][-1][1]
        if _similar(txt, prev) > 0.5:
            groups[-1].append((t, txt))
        else:
            groups.append([(t, txt)])

    segments = []
    for g in groups:
        non_empty = [(t, tx) for t, tx in g if tx.strip()]
        if len(non_empty) < 2:
            continue
        start, end = g[0][0], g[-1][0] + interval_s
        if end - start < 0.8:
            continue
        counts = defaultdict(int)
        for _, tx in non_empty:
            counts[tx] += 1
        best = max(counts, key=counts.get)
        if len(best.strip()) < 2:
            continue
        segments.append(SubSegment(
            start=start, end=end, text=best,
            source=source, confidence=counts[best] / len(non_empty)
        ))

    return segments


def _similar(a, b):
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


# ── Whisper ──────────────────────────────────────────

def run_whisper(video_path, model_name='base'):
    """Run Whisper on video audio. Returns list of SubSegment."""
    import whisper

    print(f"[Whisper] Loading {model_name} model...")
    model = whisper.load_model(model_name)

    print("[Whisper] Transcribing...")
    t0 = time.time()
    result = model.transcribe(video_path, language='ja')
    print(f"[Whisper] Done in {time.time() - t0:.0f}s")

    segments = []
    for seg in result['segments']:
        text = seg['text'].strip()
        if not text or len(text) < 2:
            continue
        # Filter obvious hallucinations (English, Chinese)
        jp_chars = sum(1 for c in text if '\u3040' <= c <= '\u309f' or '\u30a0' <= c <= '\u30ff' or '\u4e00' <= c <= '\u9fff' or c in '！？。、')
        if jp_chars / max(len(text), 1) < 0.3:
            continue
        segments.append(SubSegment(
            start=seg['start'], end=seg['end'],
            text=text, source='whisper',
            confidence=seg.get('avg_logprob', -1)
        ))

    return segments


# ── Fusion ───────────────────────────────────────────

def fuse_segments(ocr_segs, whisper_segs):
    """Fuse OCR and Whisper segments by time overlap."""
    fused = []

    # Build timeline at 0.5s resolution
    all_times = set()
    for seg in ocr_segs + whisper_segs:
        t = seg.start
        while t < seg.end:
            all_times.add(round(t, 1))
            t += 0.5

    for seg in ocr_segs:
        # Find overlapping whisper segments
        overlapping = [w for w in whisper_segs
                       if w.start < seg.end and w.end > seg.start]
        whisper_text = ' '.join(w.text for w in overlapping) if overlapping else ''

        fused.append(SubSegment(
            start=seg.start, end=seg.end,
            text=seg.text,
            source='fused',
            confidence=seg.confidence,
        ))
        # Store whisper text in translation field temporarily for cross-check
        fused[-1].translation = f'[whisper] {whisper_text}' if whisper_text else ''

    # Add whisper-only segments (not covered by OCR)
    for w in whisper_segs:
        covered = any(o.start <= w.start and o.end >= w.end for o in ocr_segs)
        if not covered:
            has_overlap = any(o.start < w.end and o.end > w.start for o in ocr_segs)
            if not has_overlap:
                fused.append(SubSegment(
                    start=w.start, end=w.end,
                    text=w.text, source='whisper_only',
                    confidence=w.confidence,
                ))

    fused.sort(key=lambda s: s.start)
    return fused


# ── Translation ──────────────────────────────────────

def translate_segments(segments, target_lang='Traditional Chinese'):
    """Translate segments using claude -p CLI."""
    texts = [s.text for s in segments if s.text.strip()]
    if not texts:
        return segments

    # Batch all texts into one prompt
    numbered = '\n'.join(f'{i+1}. {t}' for i, t in enumerate(texts))
    prompt = f"""Translate each numbered Japanese line to {target_lang}.
Output ONLY the translations, one per line, with the same numbering.
Keep punctuation style. If a line is garbled/unclear, translate what you can and mark unclear parts with [?].

{numbered}"""

    print(f"[Translate] Sending {len(texts)} lines to claude...")
    try:
        result = subprocess.run(
            ['claude', '-p', prompt],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            translations = {}
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                # Parse "1. translation" format
                parts = line.split('. ', 1)
                if len(parts) == 2 and parts[0].isdigit():
                    idx = int(parts[0]) - 1
                    translations[idx] = parts[1]

            # Apply translations
            text_idx = 0
            for seg in segments:
                if seg.text.strip():
                    if text_idx in translations:
                        seg.translation = translations[text_idx]
                    text_idx += 1

            print(f"[Translate] Got {len(translations)} translations")
        else:
            print(f"[Translate] Error: {result.stderr[:200]}")
    except subprocess.TimeoutExpired:
        print("[Translate] Timeout")

    return segments


# ── SRT Output ───────────────────────────────────────

def segments_to_srt(segments, include_translation=True):
    lines = []
    for i, s in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(f"{_fmt(s.start)} --> {_fmt(s.end)}")
        if include_translation and s.translation and not s.translation.startswith('[whisper]'):
            lines.append(s.text)
            lines.append(s.translation)
        else:
            lines.append(s.text)
        lines.append('')
    return '\n'.join(lines)


def _fmt(s):
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = int(s % 60)
    ms = int((s % 1) * 1000)
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"


# ── Cross-check Player ──────────────────────────────

def run_player(video_path, segments):
    """NiceGUI video player with subtitle cross-check."""
    from nicegui import ui, app

    # Serve video file
    video_name = Path(video_path).name
    app.add_media_file(local_file=video_path, url_path=f'/media/{video_name}')

    # Build segment data for JS
    seg_data = [asdict(s) for s in segments]

    @ui.page('/')
    def page():
        ui.dark_mode(True)
        ui.label('Subtitle Cross-Check Player').classes('text-2xl font-bold mb-2')

        with ui.row().classes('w-full gap-4'):
            # Left: video
            with ui.column().classes('w-2/3'):
                video = ui.video(f'/media/{video_name}').classes('w-full')
                time_label = ui.label('0:00.0').classes('text-lg font-mono')

            # Right: subtitle panel
            with ui.column().classes('w-1/3'):
                ui.label('Current Subtitles').classes('text-lg font-bold')

                ocr_card = ui.card().classes('w-full')
                with ocr_card:
                    ui.label('OCR').classes('text-sm text-gray-400')
                    ocr_text = ui.label('—').classes('text-lg')

                whisper_card = ui.card().classes('w-full')
                with whisper_card:
                    ui.label('Whisper').classes('text-sm text-gray-400')
                    whisper_text = ui.label('—').classes('text-lg')

                trans_card = ui.card().classes('w-full')
                with trans_card:
                    ui.label('Translation').classes('text-sm text-gray-400')
                    trans_text = ui.label('—').classes('text-lg')

        # Subtitle timeline table
        ui.label('All Segments').classes('text-xl font-bold mt-4')
        columns = [
            {'name': 'time', 'label': 'Time', 'field': 'time', 'sortable': True},
            {'name': 'source', 'label': 'Source', 'field': 'source'},
            {'name': 'text', 'label': 'Japanese', 'field': 'text'},
            {'name': 'translation', 'label': 'Translation', 'field': 'translation'},
        ]
        rows = []
        for s in segments:
            rows.append({
                'time': f"{_fmt(s.start)} → {_fmt(s.end)}",
                'source': s.source,
                'text': s.text,
                'translation': s.translation if not s.translation.startswith('[whisper]') else s.translation[10:],
            })
        ui.table(columns=columns, rows=rows).classes('w-full')

        # Timer to update current subtitle display
        def update_display():
            # This would need JS integration for real-time video position
            pass

    ui.run(port=8088, title='Subtitle Cross-Check')


# ── Main ─────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('video', help='Video file path')
    parser.add_argument('--skip-ocr', action='store_true')
    parser.add_argument('--skip-whisper', action='store_true')
    parser.add_argument('--skip-translate', action='store_true')
    parser.add_argument('--target-lang', default='Traditional Chinese')
    parser.add_argument('--interval', type=int, default=1000)
    parser.add_argument('--whisper-model', default='base')
    parser.add_argument('--player', action='store_true', help='Launch cross-check player')
    args = parser.parse_args()

    video_path = args.video
    stem = Path(video_path).stem
    cache_dir = Path(video_path).parent / f'.{stem}_cache'
    cache_dir.mkdir(exist_ok=True)

    # Step 1: OCR
    ocr_cache = cache_dir / 'ocr.json'
    if args.skip_ocr and ocr_cache.exists():
        print("[OCR] Loading from cache...")
        ocr_segs = [SubSegment(**s) for s in json.loads(ocr_cache.read_text())]
    else:
        ocr_segs = run_ocr(video_path, args.interval)
        ocr_cache.write_text(json.dumps([asdict(s) for s in ocr_segs], ensure_ascii=False, indent=2))
        print(f"[OCR] {len(ocr_segs)} segments cached")

    # Step 2: Whisper
    whisper_cache = cache_dir / 'whisper.json'
    if args.skip_whisper and whisper_cache.exists():
        print("[Whisper] Loading from cache...")
        whisper_segs = [SubSegment(**s) for s in json.loads(whisper_cache.read_text())]
    else:
        whisper_segs = run_whisper(video_path, args.whisper_model)
        whisper_cache.write_text(json.dumps([asdict(s) for s in whisper_segs], ensure_ascii=False, indent=2))
        print(f"[Whisper] {len(whisper_segs)} segments cached")

    # Step 3: Fuse
    fused = fuse_segments(ocr_segs, whisper_segs)
    print(f"[Fuse] {len(fused)} segments ({sum(1 for f in fused if f.source == 'whisper_only')} whisper-only)")

    # Step 4: Translate
    if not args.skip_translate:
        fused = translate_segments(fused, args.target_lang)

    # Save outputs
    srt_path = Path(video_path).with_suffix('.srt')
    srt_path.write_text(segments_to_srt(fused), encoding='utf-8')
    print(f"\n[Output] SRT saved to {srt_path}")

    # Save full data
    full_path = cache_dir / 'fused.json'
    full_path.write_text(json.dumps([asdict(s) for s in fused], ensure_ascii=False, indent=2))

    # Print summary
    print("\n=== Results ===")
    for s in fused:
        src_tag = {'fused': 'OCR', 'whisper_only': 'WHI', 'ocr': 'OCR', 'whisper': 'WHI'}.get(s.source, s.source)
        trans = f" → {s.translation}" if s.translation and not s.translation.startswith('[whisper]') else ''
        print(f"  [{_fmt(s.start)}-{_fmt(s.end)}] ({src_tag}) {s.text}{trans}")

    # Step 5: Player
    if args.player:
        run_player(video_path, fused)


if __name__ in {"__main__", "__mp_main__"}:
    main()
