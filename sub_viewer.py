"""
YouTube subtitle viewer — paste URL, auto-loads cached subs or generates new ones.
"""
from nicegui import ui, run
from pathlib import Path
import json, re, subprocess, asyncio, base64, requests, numpy as np
import cv2

SUBS_DIR = Path('subs')
SUBS_DIR.mkdir(exist_ok=True)


def parse_srt(path):
    entries = []
    text = Path(path).read_text(encoding='utf-8')
    for block in text.strip().split('\n\n'):
        lines = block.strip().split('\n')
        if len(lines) < 2:
            continue
        m = re.match(r'(\d{2}):(\d{2}):(\d{2}),(\d{3}) --> (\d{2}):(\d{2}):(\d{2}),(\d{3})', lines[1])
        if not m:
            continue
        g = [int(x) for x in m.groups()]
        start = g[0]*3600 + g[1]*60 + g[2] + g[3]/1000
        end = g[4]*3600 + g[5]*60 + g[6] + g[7]/1000

        # Parse speaker tag and translation
        text_line = lines[2] if len(lines) > 2 else ''
        speaker = '?'
        sm = re.match(r'\[(\w+)\]\s*(.*)', text_line)
        if sm:
            speaker = sm.group(1)
            text_line = sm.group(2)

        translation = ''
        if len(lines) > 3 and not re.match(r'\d+$', lines[3]) and '-->' not in lines[3]:
            translation = lines[3]

        entries.append({'start': start, 'end': end, 'text': text_line,
                        'speaker': speaker, 'translation': translation})
    return entries


def load_subs(path):
    """Load subs from JSON (native) or SRT (legacy). Returns list of events for viewer."""
    path = Path(path)
    if path.suffix == '.json':
        data = json.loads(path.read_text(encoding='utf-8'))
        return data.get('events', [])
    else:
        # Legacy SRT → convert to event format
        entries = parse_srt(path)
        # Group overlapping entries into events
        events = []
        for e in entries:
            if events and abs(e['start'] - events[-1]['time']) < 0.1:
                events[-1]['lines'].append({
                    'text': e['text'], 'color': e.get('speaker', 'white'),
                    'zh': e.get('translation', '')})
            else:
                events.append({
                    'time': e['start'],
                    'lines': [{'text': e['text'], 'color': e.get('speaker', 'white'),
                               'zh': e.get('translation', '')}]
                })
        return events


def extract_video_id(url):
    m = re.search(r'(?:v=|youtu\.be/)([\w-]{11})', url or '')
    return m.group(1) if m else None


def find_existing_sub(vid):
    """Check if we already have subs for this video ID. Prefer JSON over SRT."""
    json_path = SUBS_DIR / f'{vid}.json'
    if json_path.exists():
        return json_path
    for f in SUBS_DIR.glob('*.srt'):
        if vid in f.stem:
            return f
    return None


def _ocr_align(video_path, whisper_segs, progress_cb=None):
    """Align Whisper content to OCR-derived timestamps."""
    import os
    os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
    from paddleocr import PaddleOCR
    from difflib import SequenceMatcher
    from collections import defaultdict

    ocr = PaddleOCR(
        text_detection_model_name='PP-OCRv5_mobile_det',
        text_recognition_model_name='PP-OCRv5_mobile_rec',
        use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False,
    )

    cap = cv2.VideoCapture(str(video_path))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)

    # OCR every 2s
    ocr_timeline = []
    for t_ms in range(0, int(dur * 1000), 2000):
        cap.set(cv2.CAP_PROP_POS_MSEC, t_ms)
        ret, frame = cap.read()
        if not ret:
            break
        zone = frame[int(h * 0.70):, :]
        tmp = '/tmp/_ocr_align.jpg'
        cv2.imwrite(tmp, zone)
        result = ocr.predict(tmp)
        texts = []
        for r in result:
            for box, text, score in zip(r['dt_polys'], r['rec_texts'], r['rec_scores']):
                if score > 0.5 and len(text) > 2:
                    texts.append(text)
        ocr_timeline.append((t_ms / 1000, ' '.join(texts)))
    cap.release()

    if not ocr_timeline:
        return whisper_segs  # fallback

    # Deduplicate OCR into segments
    def sim(a, b):
        if not a or not b: return 0
        return SequenceMatcher(None, a, b).ratio()

    ocr_segs = []
    groups = [[ocr_timeline[0]]]
    for t, txt in ocr_timeline[1:]:
        if sim(txt, groups[-1][-1][1]) > 0.5:
            groups[-1].append((t, txt))
        else:
            groups.append([(t, txt)])
    for g in groups:
        non_empty = [(t, tx) for t, tx in g if tx.strip()]
        if not non_empty:
            continue
        counts = defaultdict(int)
        for _, tx in non_empty:
            counts[tx] += 1
        best = max(counts, key=counts.get)
        if best:
            ocr_segs.append({'start': g[0][0], 'end': g[-1][0] + 2, 'text': best})

    if not ocr_segs:
        return whisper_segs

    # Align: for each OCR segment, find best Whisper match
    aligned = []
    used = set()
    for os_ in ocr_segs:
        best_i, best_s = -1, 0.3
        for i, ws in enumerate(whisper_segs):
            if i in used:
                continue
            if abs(ws['start'] - os_['start']) > 15:
                continue
            s = sim(os_['text'], ws['text'])
            if s > best_s:
                best_s = s
                best_i = i
        if best_i >= 0:
            aligned.append({'start': os_['start'], 'end': os_['end'], 'text': whisper_segs[best_i]['text']})
            used.add(best_i)
        else:
            aligned.append({'start': os_['start'], 'end': os_['end'], 'text': os_['text']})

    if progress_cb:
        progress_cb(f'Aligned {len(aligned)} segments (OCR timing + Whisper text)', 0.65)

    return aligned


def _get_openrouter_key():
    """Fetch OpenRouter API key from homelab credentials."""
    try:
        r = subprocess.run(['curl', '-su', 'evnchn:insecure', 'http://192.168.50.1:8080/999-credentials.html'],
                           capture_output=True, text=True, timeout=5)
        for line in r.stdout.split('\n'):
            if 'OpenRouter' in line and 'sk-or-' in line:
                import re as _re
                m = _re.search(r'(sk-or-v1-[a-f0-9]+)', line)
                if m: return m.group(1)
    except: pass
    return None


def _gemini_video_call(api_key, vid, start_from=0):
    """Single Gemini video call, optionally from a timestamp."""
    extra = f" Start from {start_from:.0f} seconds — skip earlier content." if start_from > 5 else ""
    r = requests.post("https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": "google/gemini-2.5-flash",
              "messages": [{"role": "user", "content": [
                  {"type": "text", "text": f"Extract ALL burned-in subtitle text with timestamps from this video.{extra} Return JSON: [{{\"start_time\": number, \"end_time\": number, \"text\": \"...\", \"speaker\": \"pink\" or \"blue\"}}]. All times as NUMBERS (seconds). Continue to the VERY END."},
                  {"type": "video_url", "video_url": {"url": f"https://www.youtube.com/watch?v={vid}"}}]}],
              "response_format": {"type": "json_object"}},
        timeout=300)
    data = json.loads(r.json()['choices'][0]['message']['content'])
    if isinstance(data, dict): data = data.get('subtitles', data.get('results', []))
    return data


def generate_gemini_video_srt_sync(vid, progress_cb=None):
    """Gemini video with retry until full coverage."""
    from difflib import SequenceMatcher

    api_key = _get_openrouter_key()
    if not api_key:
        raise RuntimeError('No OpenRouter API key found')

    # Get video duration
    dur_out = subprocess.run(['yt-dlp', '--print', 'duration', f'https://www.youtube.com/watch?v={vid}'],
        capture_output=True, text=True, timeout=15)
    duration = float(dur_out.stdout.strip()) if dur_out.stdout.strip() else 120

    all_subs = []
    covered_until = 0
    calls = 0
    max_retries = 3

    for attempt in range(max_retries + 1):
        if progress_cb:
            progress_cb(f'Gemini call {attempt+1} (from {covered_until:.0f}s)...', 0.1 + 0.15 * attempt)

        subs = _gemini_video_call(api_key, vid, start_from=covered_until if covered_until > 5 else 0)
        calls += 1

        new_subs = [s for s in subs if float(s.get('start_time', 0)) >= covered_until - 2]
        all_subs.extend(new_subs)

        if subs:
            last_t = max(float(s.get('end_time', 0)) for s in subs)
            coverage = last_t / duration * 100
            if coverage >= 85:
                break
            covered_until = last_t - 5
        else:
            break

    # Deduplicate overlapping results
    def normalize(t): return t.replace('！','!').replace('！','!').strip()
    deduped = []
    for s in sorted(all_subs, key=lambda x: float(x.get('start_time', 0))):
        if deduped:
            prev = deduped[-1]
            if abs(float(s.get('start_time',0)) - float(prev.get('start_time',0))) < 1 and \
               SequenceMatcher(None, normalize(s.get('text','')), normalize(prev.get('text',''))).ratio() > 0.6:
                continue
        deduped.append(s)

    data = deduped
    if progress_cb: progress_cb(f'{len(data)} subs in {calls} calls, translating...', 0.7)

    if progress_cb: progress_cb(f'Translating {len(data)} lines...', 0.8)

    # Translate
    numbered = '\n'.join(f"{i+1}. {s.get('text','')}" for i, s in enumerate(data))
    try:
        tr = requests.post("https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": "google/gemini-2.5-flash",
                  "messages": [{"role": "user", "content": f"Translate each Japanese line to Traditional Chinese. Comedy anime. Natural, punchy. Output ONLY numbered translations.\n\n{numbered}"}]},
            timeout=60)
        translations = {}
        for line in tr.json()['choices'][0]['message']['content'].strip().split('\n'):
            parts = line.strip().split('. ', 1)
            if len(parts) == 2 and parts[0].isdigit():
                translations[int(parts[0])-1] = parts[1]
    except: translations = {}

    if progress_cb: progress_cb('Writing SRT...', 0.95)

    def fmt(s):
        hh=int(s//3600); m=int((s%3600)//60); sec=int(s%60); ms=int((s%1)*1000)
        return f"{hh:02d}:{m:02d}:{sec:02d},{ms:03d}"

    srt_lines = []
    for i, s in enumerate(data):
        start = s.get('start_time', 0)
        end = s.get('end_time', start + 1)
        text = s.get('text', '')
        speaker = s.get('speaker', '?')
        srt_lines.append(str(i+1))
        srt_lines.append(f"{fmt(start)} --> {fmt(end)}")
        srt_lines.append(f"[{speaker}] {text}")
        if i in translations:
            srt_lines.append(translations[i])
        srt_lines.append('')

    srt_path = SUBS_DIR / f'{vid}.srt'
    srt_path.write_text('\n'.join(srt_lines), encoding='utf-8')
    return srt_path


def _temporal_consistency(events):
    """Sequential pass to fix parallel Gemini inconsistencies.
    Aggressive dedup: Gemini reads same text differently each frame.
    """
    if not events:
        return events

    def norm(t):
        """Ultra-aggressive normalize: strip ALL non-kanji/kana."""
        return re.sub(r'[^ぁ-んァ-ヶ一-龥a-zA-Z]', '', t)

    def event_fingerprint(lines):
        """Fingerprint an event by sorted normalized text of all lines."""
        return tuple(sorted(norm(l['text']) for l in lines if norm(l['text'])))

    def events_similar(fp1, fp2):
        """Check if two event fingerprints are similar (>60% character overlap)."""
        if not fp1 or not fp2:
            return fp1 == fp2
        # Join all text, compare as character sets
        t1 = ''.join(fp1)
        t2 = ''.join(fp2)
        if not t1 or not t2:
            return False
        from difflib import SequenceMatcher
        return SequenceMatcher(None, t1, t2).ratio() > 0.6

    cleaned = []
    color_memory = {}

    for i, event in enumerate(events):
        lines = event.get('lines', [])
        if not lines:
            continue

        # Lock colors
        for line in lines:
            tn = norm(line['text'])
            if tn in color_memory:
                if line['color'] == 'white' and color_memory[tn] != 'white':
                    line['color'] = color_memory[tn]
            elif line['color'] != 'white':
                color_memory[tn] = line['color']

        # Compare to previous: skip if similar
        curr_fp = event_fingerprint(lines)
        if cleaned:
            prev_fp = event_fingerprint(cleaned[-1].get('lines', []))
            if events_similar(curr_fp, prev_fp):
                continue

        cleaned.append(event)

    return cleaned


def generate_gemini_srt_sync(vid, progress_cb=None):
    """Combined pipeline: binary search + Gemini reads at boundaries + translate."""
    # cv2/numpy imported at top level
    from difflib import SequenceMatcher

    api_key = _get_openrouter_key()
    if not api_key:
        raise RuntimeError('No OpenRouter API key found')

    video_path = SUBS_DIR / f'{vid}.webm'
    srt_path = SUBS_DIR / f'{vid}.srt'

    # Download
    if progress_cb: progress_cb('Downloading...', 0.05)
    if not video_path.exists():
        subprocess.run(['yt-dlp', '--no-playlist', '-o', str(video_path),
                        f'https://www.youtube.com/watch?v={vid}'], capture_output=True, timeout=120)
        for ext in ['webm', 'mp4', 'mkv']:
            p = SUBS_DIR / f'{vid}.{ext}'
            if p.exists(): video_path = p; break
    if not video_path.exists():
        raise RuntimeError('Download failed')

    cap = cv2.VideoCapture(str(video_path))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)

    def frame_at(ms):
        cap.set(cv2.CAP_PROP_POS_MSEC, ms); ret, f = cap.read(); return f if ret else None

    # CoreML detection fingerprint at 5fps
    if progress_cb: progress_cb('CoreML detection scan (5fps)...', 0.1)
    import onnxruntime
    det_path = str(Path(__file__).parent / 'venv/lib/python3.12/site-packages/rapidocr/models/ch_PP-OCRv5_mobile_det.onnx')
    try:
        det_session = onnxruntime.InferenceSession(det_path, providers=[
            ('CoreMLExecutionProvider', {'MLComputeUnits': 'ALL'}), 'CPUExecutionProvider'])
    except:
        det_session = onnxruntime.InferenceSession(det_path, providers=['CPUExecutionProvider'])
    det_input = det_session.get_inputs()[0].name

    heatmaps = []  # (t_ms, binary_heatmap)
    for t_ms in range(0, int(dur*1000), 200):  # 5fps
        frame = frame_at(t_ms)
        if frame is None: break
        scale = 640 / frame.shape[1]
        resized = cv2.resize(frame, (640, int(frame.shape[0] * scale)))
        pad_h = (32 - resized.shape[0] % 32) % 32
        padded = cv2.copyMakeBorder(resized, 0, pad_h, 0, 0, cv2.BORDER_CONSTANT, value=0)
        blob = (padded.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis]
        output = det_session.run(None, {det_input: blob})
        hm_bin = (output[0][0, 0] > 0.3).astype(np.uint8)
        heatmaps.append((t_ms, hm_bin))

    # Pixel-level IoU dedup: keep only frames where heatmap pattern changed
    def hm_iou(a, b):
        inter = (a & b).sum(); union = (a | b).sum()
        return inter / max(union, 1)

    deduped_hm = [heatmaps[0]]
    for i in range(1, len(heatmaps)):
        if hm_iou(heatmaps[i][1], deduped_hm[-1][1]) < 0.75:
            deduped_hm.append(heatmaps[i])
    deduped = [t_ms for t_ms, _ in deduped_hm]

    # Gemini reads — parallel (8 concurrent)
    if progress_cb: progress_cb(f'Gemini reading {len(deduped)} boundaries (parallel)...', 0.3)
    def normalize(t): return re.sub(r'[！!\-\.。、\s8]+', '', t)

    # Prepare frames
    frames_to_read = []
    for t_ms in deduped:
        frame = frame_at(t_ms)
        if frame is not None:
            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
            frames_to_read.append((t_ms, base64.b64encode(buf).decode()))

    def read_one(args):
        t_ms, b64 = args
        try:
            r = requests.post("https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": "google/gemini-2.5-flash",
                      "messages": [{"role": "user", "content": [
                          {"type": "text", "text": 'Read ALL text visible in this frame. Classify each line by priority:\n- "active": current main subtitle/dialogue (most important)\n- "persist": previous subtitle still visible on screen\n- "bg": background element (signs, blackboard, posters, dates)\nAlso pick closest color: pink, blue, red, yellow, green, orange, purple, cyan, white. Translate to Traditional Chinese.\nReturn JSON: [{"text": "...", "priority": "active/persist/bg", "color": "palette", "zh": "translation"}].'},
                          {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}],
                      "response_format": {"type": "json_object"}}, timeout=20)
            data = json.loads(r.json()['choices'][0]['message']['content'])
            if isinstance(data, dict): data = data.get('subtitles', data.get('results', []))
            return (t_ms, data if isinstance(data, list) else [])
        except: return (t_ms, [])

    from concurrent.futures import ThreadPoolExecutor, as_completed
    results_map = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(read_one, item): item[0] for item in frames_to_read}
        done = 0
        for future in as_completed(futures):
            t_ms, subs = future.result()
            results_map[t_ms] = subs
            done += 1
            if progress_cb and done % 10 == 0:
                progress_cb(f'Gemini: {done}/{len(frames_to_read)}...', 0.3 + 0.4 * done / len(frames_to_read))

    # Native format: each timestamp = ALL lines Gemini saw at that moment
    # Dedup consecutive identical reads
    events = []
    prev_key = None
    for t_ms in sorted(results_map.keys()):
        subs = results_map[t_ms]
        lines = [{'text': s.get('text','').strip(),
                  'color': s.get('color', s.get('speaker', 'white')),
                  'zh': s.get('zh', '')}
                 for s in subs if s.get('text','').strip() and len(s.get('text','').strip()) > 1]
        curr_key = tuple(normalize(l['text']) for l in lines)
        if curr_key == prev_key:
            continue
        prev_key = curr_key
        if lines:
            events.append({'time': round(t_ms / 1000, 3), 'lines': lines})

    cap.release()
    video_path.unlink(missing_ok=True)

    # Temporal consistency pass
    if progress_cb: progress_cb('Temporal consistency...', 0.9)
    events = _temporal_consistency(events)

    # Write native JSON
    if progress_cb: progress_cb('Saving...', 0.95)
    from datetime import datetime
    data = {
        'version': 1, 'video_id': vid, 'duration': dur,
        'generated_at': datetime.now().isoformat(),
        'engine': 'gemini_coreml', 'events': events,
    }
    json_path = SUBS_DIR / f'{vid}.json'
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    return json_path


def generate_srt_sync(vid, progress_cb=None, whisper_model='base'):
    """Download video, run Whisper, translate. Returns SRT path.

    Gen1 default: whisper medium (slower, used for initial development)
    Gen2 default: whisper base (84ms timing, 10x faster, near-zero drift)
    """
    import whisper
    import time

    video_path = SUBS_DIR / f'{vid}.webm'
    srt_path = SUBS_DIR / f'{vid}.srt'

    # Step 1: Download
    if progress_cb:
        progress_cb('Downloading video...', 0.1)
    subprocess.run([
        'yt-dlp', '--no-playlist', '-o', str(video_path),
        f'https://www.youtube.com/watch?v={vid}'
    ], capture_output=True, timeout=120)

    if not video_path.exists():
        # yt-dlp may have chosen a different extension
        for ext in ['webm', 'mp4', 'mkv']:
            p = SUBS_DIR / f'{vid}.{ext}'
            if p.exists():
                video_path = p
                break

    if not video_path.exists():
        raise RuntimeError('Download failed')

    # Step 2: Whisper
    if progress_cb:
        progress_cb(f'Transcribing with Whisper {whisper_model}...', 0.3)
    model = whisper.load_model(whisper_model)
    result = model.transcribe(str(video_path), language='ja')

    whisper_clean = []
    for s in result['segments']:
        t = s['text'].strip()
        if not t or len(t) < 2:
            continue
        jp = sum(1 for c in t if '\u3040' <= c <= '\u309f' or '\u30a0' <= c <= '\u30ff' or '\u4e00' <= c <= '\u9fff' or c in '！？。、!?')
        if jp / max(len(t), 1) < 0.4:
            continue
        whisper_clean.append({'start': s['start'], 'end': s['end'], 'text': t})

    clean = whisper_clean

    if progress_cb:
        progress_cb(f'Translating {len(clean)} lines...', 0.7)

    # Step 3: Translate
    numbered = '\n'.join(f"{i+1}. {s['text']}" for i, s in enumerate(clean))
    prompt = f"Translate each Japanese line to Traditional Chinese. Comedy animation. Output ONLY numbered translations.\n\n{numbered}"
    tr = subprocess.run(['claude', '-p', prompt], capture_output=True, text=True, timeout=180)
    translations = {}
    if tr.returncode == 0:
        for line in tr.stdout.strip().split('\n'):
            parts = line.strip().split('. ', 1)
            if len(parts) == 2 and parts[0].isdigit():
                translations[int(parts[0]) - 1] = parts[1]

    # Step 4: Write SRT
    if progress_cb:
        progress_cb('Writing SRT...', 0.95)

    def fmt(s):
        h = int(s // 3600); m = int((s % 3600) // 60); sec = int(s % 60); ms = int((s % 1) * 1000)
        return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"

    srt_lines = []
    for i, s in enumerate(clean):
        srt_lines.append(str(i + 1))
        srt_lines.append(f"{fmt(s['start'])} --> {fmt(s['end'])}")
        srt_lines.append(s['text'])
        if i in translations:
            srt_lines.append(translations[i])
        srt_lines.append('')

    srt_path.write_text('\n'.join(srt_lines), encoding='utf-8')

    # Cleanup video file
    video_path.unlink(missing_ok=True)

    return srt_path


@ui.page('/')
def index():
    ui.dark_mode(True)
    ui.label('JP Subtitle Generator').classes('text-2xl font-bold')

    with ui.row().classes('w-full items-end gap-4'):
        url_input = ui.input('YouTube URL', placeholder='https://www.youtube.com/watch?v=...').classes('w-96')
        go_btn = ui.button('Go', color='primary')
        regen_btn = ui.button('Regen', color='warning').props('outline')
        align_btn = ui.button('OCR Align', color='info').props('outline')
    with ui.row().classes('w-full items-end gap-4'):
        engine_select = ui.select(
            {'gemini': 'Gemini frame-by-frame (~$0.007)',
             'whisper': 'Whisper only (fast, free)',
             'gemini_video': 'Gemini video (experimental)'},
            value='gemini', label='Engine'
        ).classes('w-48')
        model_select = ui.select(
            {'base': 'base (best timing)', 'small': 'small', 'medium': 'medium'},
            value='base', label='Whisper model'
        ).classes('w-48')
        offset_slider = ui.slider(min=-5, max=5, step=0.1, value=0).props('label-always').classes('w-48')
        ui.label('Offset (s)').classes('text-xs')
    with ui.row().classes('w-full items-center gap-4'):
        show_persist = ui.switch('Show persisting subs', value=False).classes('text-sm')
        show_bg = ui.switch('Show background text', value=False).classes('text-sm')

    status_label = ui.label('').classes('text-sm text-gray-400')
    progress = ui.linear_progress(value=0, show_value=False).classes('w-full').style('display:none')

    player_container = ui.element('div').classes('w-full flex justify-center mt-4')

    # Multi-line colored subtitle container
    sub_container = ui.column().classes('w-full items-center gap-0').style(
        'min-height:100px; margin-top:8px;')

    _state = {'timer': None, 'subs': [], 'sub_elements': []}

    async def start_player(vid, subs):
        _state['subs'] = subs

        player_container.clear()
        with player_container:
            ui.html('<div id="ytplayer-host"></div>')

        await ui.run_javascript(f'''
            if (window._boku_player) {{
                try {{ window._boku_player.destroy(); }} catch(e) {{}}
            }}
            document.getElementById("ytplayer-host").innerHTML = '<div id="ytplayer"></div>';
            function createPlayer() {{
                window._boku_player = new YT.Player("ytplayer", {{
                    width: 960, height: 540,
                    videoId: "{vid}",
                    playerVars: {{ autoplay: 1 }},
                }});
            }}
            if (window.YT && window.YT.Player) {{
                createPlayer();
            }} else {{
                var tag = document.createElement("script");
                tag.src = "https://www.youtube.com/iframe_api";
                document.head.appendChild(tag);
                window.onYouTubeIframeAPIReady = createPlayer;
            }}
        ''', timeout=10)

        if _state['timer']:
            _state['timer'].deactivate()

        async def poll():
            try:
                t = await ui.run_javascript(
                    'window._boku_player && window._boku_player.getCurrentTime ? window._boku_player.getCurrentTime() : -1',
                    timeout=2)
                if t is not None and t >= 0:
                    t_adj = t + offset_slider.value

                    # Find the latest event at this time
                    active_event = None
                    for event in _state['subs']:
                        t_event = event.get('time', event.get('start', 0))
                        if t_event <= t_adj:
                            active_event = event
                        else:
                            break

                    # Get lines from the active event, filtered by priority switches
                    if active_event and 'lines' in active_event:
                        active = []
                        for line in active_event['lines']:
                            pri = line.get('priority', 'active')
                            if pri == 'active':
                                active.append(line)
                            elif pri == 'persist' and show_persist.value:
                                active.append(line)
                            elif pri == 'bg' and show_bg.value:
                                active.append(line)
                    elif active_event:
                        active = [active_event]
                    else:
                        active = []

                    # Build display — only update if changed
                    active_key = tuple((s.get('text',''), s.get('color', s.get('speaker',''))) for s in active)
                    prev_key = tuple((s.get('text',''), s.get('color', s.get('speaker',''))) for s in _state.get('prev_active', []))

                    if active_key != prev_key:
                        sub_container.clear()
                        with sub_container:
                            for s in active:
                                text = s.get('text', '')
                                speaker = s.get('color', s.get('speaker', '?'))
                                # Map color name to CSS
                                color_map = {
                                    'pink': '#FC8DC2', 'blue': '#8EC6FD',
                                    'red': '#FF6B6B', 'yellow': '#FFE066',
                                    'green': '#66CC66', 'orange': '#FFB347',
                                    'purple': '#CC99FF', 'cyan': '#66CCCC',
                                    'white': '#FFFFFF',
                                }
                                color = color_map.get(speaker, '#FFFFFF')

                                # Style by priority
                                pri = s.get('priority', 'active')
                                if pri == 'active':
                                    ui.label(text).classes('text-xl font-bold text-center').style(
                                        f'color: {color}; text-shadow: 1px 1px 2px black, -1px -1px 2px black;')
                                elif pri == 'persist':
                                    ui.label(text).classes('text-lg text-center').style(
                                        f'color: {color}; opacity: 0.6; text-shadow: 1px 1px 2px black;')
                                else:  # bg
                                    ui.label(text).classes('text-sm text-center').style(
                                        f'color: gray; opacity: 0.4; font-style: italic;')

                                translation = s.get('zh', s.get('translation', ''))
                                if translation:
                                    ui.label(translation).classes('text-sm text-center text-gray-300')

                        _state['prev_active'] = active
            except Exception:
                pass

        _state['timer'] = ui.timer(0.3, poll)

    async def on_go():
        vid = extract_video_id(url_input.value)
        if not vid:
            ui.notify('Invalid YouTube URL', type='negative')
            return

        go_btn.disable()

        # Check for existing SRT
        existing = find_existing_sub(vid)
        if existing:
            status_label.set_text(f'Found cached subtitles: {existing.name}')
            progress.style('display:none')
            subs = load_subs(existing)
            ui.notify(f'Loaded {len(subs)} cached subtitles')
            await start_player(vid, subs)
            go_btn.enable()
            return

        # Generate new subtitles
        progress.style('display:block')
        progress.set_value(0.05)
        status_label.set_text('Starting pipeline...')

        def update_progress(msg, val):
            status_label.set_text(msg)
            progress.set_value(val)

        try:
            if engine_select.value == 'gemini_video':
                srt_path = await run.io_bound(generate_gemini_video_srt_sync, vid, update_progress)
            elif engine_select.value == 'gemini':
                srt_path = await run.io_bound(generate_gemini_srt_sync, vid, update_progress)
            else:
                srt_path = await run.io_bound(generate_srt_sync, vid, update_progress, model_select.value)
            subs = load_subs(srt_path)
            status_label.set_text(f'Done! {len(subs)} subtitles generated.')
            progress.set_value(1.0)
            ui.notify(f'Generated {len(subs)} subtitles', type='positive')
            await start_player(vid, subs)
        except Exception as e:
            status_label.set_text(f'Error: {e}')
            ui.notify(f'Pipeline failed: {e}', type='negative')
        finally:
            go_btn.enable()
            await asyncio.sleep(2)
            progress.style('display:none')

    go_btn.on_click(on_go)

    async def on_regen():
        vid = extract_video_id(url_input.value)
        if not vid:
            ui.notify('Enter a URL first', type='negative')
            return
        existing = find_existing_sub(vid)
        if existing:
            existing.unlink()
            ui.notify(f'Deleted {existing.name}, regenerating...')
        await on_go()

    regen_btn.on_click(on_regen)

    async def on_align():
        """Re-run with OCR alignment on existing video."""
        vid = extract_video_id(url_input.value)
        if not vid:
            ui.notify('Enter a URL first', type='negative')
            return
        # Need the video file — download if not present
        video_path = SUBS_DIR / f'{vid}.webm'
        if not video_path.exists():
            status_label.set_text('Downloading for OCR alignment...')
            progress.style('display:block')
            progress.set_value(0.1)
            await run.io_bound(lambda: subprocess.run(
                ['yt-dlp', '--no-playlist', '-o', str(video_path),
                 f'https://www.youtube.com/watch?v={vid}'],
                capture_output=True, timeout=120))

        # Load existing Whisper subs
        existing = find_existing_sub(vid)
        if not existing:
            ui.notify('Run Go first to generate base subs', type='negative')
            return

        subs = load_subs(existing)
        whisper_segs = [{'start': s['start'], 'end': s['end'], 'text': s['text'].split('\n')[0]} for s in subs]

        align_btn.disable()
        status_label.set_text('Running OCR alignment...')
        progress.style('display:block')
        progress.set_value(0.2)

        try:
            aligned = await run.io_bound(_ocr_align, str(video_path), whisper_segs, None)

            # Re-translate aligned
            status_label.set_text(f'Translating {len(aligned)} aligned segments...')
            progress.set_value(0.8)
            numbered = '\n'.join(f"{i+1}. {a['text']}" for i, a in enumerate(aligned))
            prompt = f"Translate each Japanese line to Traditional Chinese. Comedy animation. Output ONLY numbered translations.\n\n{numbered}"
            tr = await run.io_bound(lambda: subprocess.run(['claude', '-p', prompt], capture_output=True, text=True, timeout=300))
            translations = {}
            if tr.returncode == 0:
                for line in tr.stdout.strip().split('\n'):
                    parts = line.strip().split('. ', 1)
                    if len(parts) == 2 and parts[0].isdigit():
                        translations[int(parts[0]) - 1] = parts[1]

            def fmt(s):
                h=int(s//3600); m=int((s%3600)//60); sec=int(s%60); ms=int((s%1)*1000)
                return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"

            srt_lines = []
            for i, a in enumerate(aligned):
                srt_lines.append(str(i+1))
                srt_lines.append(f"{fmt(a['start'])} --> {fmt(a['end'])}")
                srt_lines.append(a['text'])
                if i in translations:
                    srt_lines.append(translations[i])
                srt_lines.append('')

            srt_path = SUBS_DIR / f'{vid}.srt'
            srt_path.write_text('\n'.join(srt_lines), encoding='utf-8')

            # Cleanup video
            video_path.unlink(missing_ok=True)

            new_subs = load_subs(srt_path)
            status_label.set_text(f'OCR aligned! {len(new_subs)} subtitles')
            progress.set_value(1.0)
            await start_player(vid, new_subs)
        except Exception as e:
            status_label.set_text(f'Alignment failed: {e}')
            ui.notify(str(e), type='negative')
        finally:
            align_btn.enable()
            import asyncio
            await asyncio.sleep(2)
            progress.style('display:none')

    align_btn.on_click(on_align)

    # Show cached count
    cached = list(SUBS_DIR.glob('*.srt'))
    if cached:
        ui.label(f'{len(cached)} videos cached').classes('mt-4 text-sm text-gray-500')


if __name__ in {"__main__", "__mp_main__"}:
    ui.run(port=8089, title='Bokuwata Sub Viewer')
