"""
Channel-specialized subtitle extraction engine.
Runs entirely local — no VLM/LLM API calls.

Color segmentation + state machine + de-style + PP-OCR.
"""

import cv2
import numpy as np
import json
import os
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from collections import defaultdict

os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'


@dataclass
class ChannelConfig:
    """Learned parameters for a specific channel's text style."""
    name: str = 'default'
    # Speaker colors in HSV (OpenCV: H=0-180, S=0-255, V=0-255)
    speaker_colors: dict = field(default_factory=lambda: {
        'A': {'h_range': (158, 174), 's_range': (50, 200), 'v_range': (100, 255), 'label': 'pink'},
        'B': {'h_range': (95, 115), 's_range': (40, 200), 'v_range': (100, 255), 'label': 'blue'},
    })
    # Text structure
    outline_color_v_max: int = 55  # black outline: V < this
    fill_color_v_min: int = 210    # white fill: V > this
    # Shadow offset (dx, dy in pixels at 1080p)
    shadow_dx: int = 9
    shadow_dy: int = 9
    # Subtitle zones (fraction of frame height)
    zones: list = field(default_factory=lambda: [
        {'name': 'top', 'y_start': 0.0, 'y_end': 0.32},
        {'name': 'bottom', 'y_start': 0.72, 'y_end': 1.0},
    ])
    # State machine params
    appear_frames: int = 2   # consecutive frames to confirm appearance
    disappear_frames: int = 2  # consecutive frames to confirm disappearance
    # Minimum text region area (pixels) at 1080p
    min_region_area: int = 3000


# Pre-built configs
BOKUWATA_CONFIG = ChannelConfig(
    name='bokuwata',
    speaker_colors={
        'A': {'h_range': (158, 174), 's_range': (50, 200), 'v_range': (100, 255), 'label': 'pink'},
        'B': {'h_range': (95, 115), 's_range': (40, 200), 'v_range': (100, 255), 'label': 'blue'},
    },
)


@dataclass
class SubtitleState:
    """Tracks one subtitle region through frames."""
    zone: str
    text: str = ''
    speaker: str = '?'
    bbox: tuple = (0, 0, 0, 0)
    first_seen: float = 0
    last_seen: float = 0
    stable_frames: int = 0
    gone_frames: int = 0
    confirmed: bool = False


def detect_speaker_regions(frame_bgr, config: ChannelConfig):
    """Detect text regions using color segmentation. Returns list of (bbox, speaker_label, mask)."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h_frame, w_frame = frame_bgr.shape[:2]

    regions = []
    for speaker_id, color_cfg in config.speaker_colors.items():
        h_lo, h_hi = color_cfg['h_range']
        s_lo, s_hi = color_cfg['s_range']
        v_lo, v_hi = color_cfg['v_range']

        # Create shadow color mask
        shadow_mask = ((hsv[:, :, 0] >= h_lo) & (hsv[:, :, 0] <= h_hi) &
                       (hsv[:, :, 1] >= s_lo) & (hsv[:, :, 1] <= s_hi) &
                       (hsv[:, :, 2] >= v_lo) & (hsv[:, :, 2] <= v_hi)).astype(np.uint8) * 255

        # Zone filter
        zone_mask = np.zeros_like(shadow_mask)
        for zone in config.zones:
            y1 = int(zone['y_start'] * h_frame)
            y2 = int(zone['y_end'] * h_frame)
            zone_mask[y1:y2, :] = 255
        shadow_mask = shadow_mask & zone_mask

        # Dilate to merge character shadows into text line blobs
        kernel = np.ones((15, 25), np.uint8)
        dilated = cv2.dilate(shadow_mask, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            if area > config.min_region_area and w > 80:
                # Expand bbox to capture full text (shadow is offset down-right)
                x1 = max(0, x - config.shadow_dx - 5)
                y1 = max(0, y - config.shadow_dy - 5)
                x2 = min(w_frame, x + w + 5)
                y2 = min(h_frame, y + h + 5)

                # Determine zone
                cy = (y1 + y2) / 2
                zone_name = 'unknown'
                for zone in config.zones:
                    if zone['y_start'] * h_frame <= cy <= zone['y_end'] * h_frame:
                        zone_name = zone['name']
                        break

                regions.append({
                    'bbox': (x1, y1, x2, y2),
                    'speaker': speaker_id,
                    'speaker_label': color_cfg['label'],
                    'zone': zone_name,
                    'area': area,
                })

    return regions


def destyle_region(frame_bgr, bbox, config: ChannelConfig):
    """Extract clean black-on-white text from a styled text region."""
    x1, y1, x2, y2 = bbox
    crop = frame_bgr[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # The text has white fill (>210) and black outline (<55)
    # Create a mask of the text shape: dark outline pixels
    _, text_mask = cv2.threshold(gray, config.outline_color_v_max, 255, cv2.THRESH_BINARY_INV)

    # Also include white fill pixels that are adjacent to dark outlines
    _, white_mask = cv2.threshold(gray, config.fill_color_v_min, 255, cv2.THRESH_BINARY)
    near_dark = cv2.dilate(text_mask, np.ones((5, 5), np.uint8), iterations=1)
    white_near_dark = white_mask & near_dark

    # Combine: text = outlines + white fill near outlines
    combined = text_mask | white_near_dark

    # Clean up with morphology
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    # Result: black text on white background
    result = 255 - combined
    return result


def ocr_region(ocr, destyled_img):
    """Run PP-OCR on a destyled text image. Returns text string."""
    tmp = '/tmp/_engine_ocr.jpg'
    cv2.imwrite(tmp, destyled_img)
    result = ocr.predict(tmp)
    texts = []
    for r in result:
        for box, text, score in zip(r['dt_polys'], r['rec_texts'], r['rec_scores']):
            if score > 0.3 and len(text) > 1:
                texts.append(text)
    return ' '.join(texts)


def extract_subtitles(video_path, config: ChannelConfig, interval_ms=500):
    """Full extraction pipeline with state machine tracking."""
    from paddleocr import PaddleOCR

    ocr = PaddleOCR(
        text_detection_model_name='PP-OCRv5_mobile_det',
        text_recognition_model_name='PP-OCRv5_mobile_rec',
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    frame_ms = 1000.0 / fps

    print(f"Video: {dur:.1f}s @ {fps:.0f}fps, {config.name} engine")

    # State machine per zone
    active_subs = {}  # zone -> SubtitleState
    completed_subs = []

    t_start = time.time()
    frame_count = 0

    for t_ms in range(0, int(dur * 1000), interval_ms):
        cap.set(cv2.CAP_PROP_POS_MSEC, t_ms)
        ret, frame = cap.read()
        if not ret:
            break

        t_s = t_ms / 1000.0
        regions = detect_speaker_regions(frame, config)

        # Track which zones have detections this frame
        zones_seen = set()

        for region in regions:
            zone = region['zone']
            zones_seen.add(zone)

            # De-style and OCR
            destyled = destyle_region(frame, region['bbox'], config)
            text = ocr_region(ocr, destyled)

            if not text or len(text) < 2:
                # Try raw zone OCR as fallback
                x1, y1, x2, y2 = region['bbox']
                crop = frame[y1:y2, x1:x2]
                tmp = '/tmp/_engine_raw.jpg'
                cv2.imwrite(tmp, crop)
                result = ocr.predict(tmp)
                texts = []
                for r in result:
                    for box, t, score in zip(r['dt_polys'], r['rec_texts'], r['rec_scores']):
                        if score > 0.3 and len(t) > 1:
                            texts.append(t)
                text = ' '.join(texts)

            if not text or len(text) < 2:
                continue

            if zone in active_subs:
                state = active_subs[zone]
                state.last_seen = t_s
                state.stable_frames += 1
                state.gone_frames = 0
                if state.stable_frames >= config.appear_frames:
                    state.confirmed = True
                # Update text if different (might be cleaner read)
                if len(text) > len(state.text):
                    state.text = text
                    state.speaker = region['speaker_label']
            else:
                active_subs[zone] = SubtitleState(
                    zone=zone, text=text,
                    speaker=region['speaker_label'],
                    bbox=region['bbox'],
                    first_seen=t_s, last_seen=t_s,
                    stable_frames=1,
                )

        # Check for disappeared subtitles
        for zone in list(active_subs.keys()):
            if zone not in zones_seen:
                state = active_subs[zone]
                state.gone_frames += 1
                if state.gone_frames >= config.disappear_frames:
                    if state.confirmed and state.text:
                        completed_subs.append({
                            'start': state.first_seen,
                            'end': state.last_seen + interval_ms / 1000,
                            'text': state.text,
                            'speaker': state.speaker,
                            'zone': state.zone,
                        })
                    del active_subs[zone]

        frame_count += 1
        if frame_count % 20 == 0:
            elapsed = time.time() - t_start
            print(f"  {t_s:.0f}s/{dur:.0f}s ({frame_count / elapsed:.1f} fps)")

    # Flush remaining active subs
    for zone, state in active_subs.items():
        if state.confirmed and state.text:
            completed_subs.append({
                'start': state.first_seen,
                'end': state.last_seen + interval_ms / 1000,
                'text': state.text,
                'speaker': state.speaker,
                'zone': state.zone,
            })

    cap.release()
    completed_subs.sort(key=lambda s: s['start'])
    elapsed = time.time() - t_start
    print(f"\nDone: {len(completed_subs)} subtitles in {elapsed:.0f}s")
    return completed_subs


def subs_to_srt(subs):
    def fmt(s):
        h = int(s // 3600); m = int((s % 3600) // 60); sec = int(s % 60); ms = int((s % 1) * 1000)
        return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"

    lines = []
    for i, s in enumerate(subs, 1):
        lines.append(str(i))
        lines.append(f"{fmt(s['start'])} --> {fmt(s['end'])}")
        speaker_tag = f"[{s['speaker']}] " if s.get('speaker', '?') != '?' else ''
        lines.append(f"{speaker_tag}{s['text']}")
        lines.append('')
    return '\n'.join(lines)


if __name__ == '__main__':
    import sys
    video = sys.argv[1] if len(sys.argv) > 1 else '停らないキッチンカー.webm'
    config_name = sys.argv[2] if len(sys.argv) > 2 else 'bokuwata'

    config = BOKUWATA_CONFIG if config_name == 'bokuwata' else ChannelConfig()

    subs = extract_subtitles(video, config, interval_ms=500)

    srt = subs_to_srt(subs)
    out = Path(video).with_suffix('.engine.srt')
    out.write_text(srt, encoding='utf-8')
    print(f"Saved to {out}")

    print("\n=== Subtitles ===")
    for s in subs:
        print(f"  [{s['start']:5.1f}-{s['end']:5.1f}] ({s['speaker']}) {s['text']}")
