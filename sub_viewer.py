"""
YouTube subtitle viewer — paste URL, auto-loads cached subs or generates new ones.
"""
from nicegui import ui, run
from pathlib import Path
import json, re, subprocess, asyncio, base64, requests

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
        entries.append({'start': start, 'end': end, 'text': '\n'.join(lines[2:])})
    return entries


def extract_video_id(url):
    m = re.search(r'(?:v=|youtu\.be/)([\w-]{11})', url or '')
    return m.group(1) if m else None


def find_existing_srt(vid):
    """Check if we already have an SRT for this video ID."""
    # Check subs/ dir first
    for f in SUBS_DIR.glob('*.srt'):
        if vid in f.stem:
            return f
    # Check root dir (legacy)
    for f in Path('.').glob('*.srt'):
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


def generate_gemini_video_srt_sync(vid, progress_cb=None):
    """ONE Gemini call with YouTube URL → full subtitle extraction."""
    api_key = _get_openrouter_key()
    if not api_key:
        raise RuntimeError('No OpenRouter API key found')

    if progress_cb: progress_cb('Sending video to Gemini...', 0.2)

    r = requests.post("https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": "google/gemini-2.5-flash",
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": """Watch this video and extract ALL burned-in subtitle text with timestamps.

For each subtitle appearance, return:
- start_time (seconds, precise)
- end_time (seconds, precise)
- text (the Japanese subtitle text)
- speaker: "pink" or "blue" (based on the colored drop shadow on the text)

The subtitles have a distinctive style: white text fill, black outline, with either a pink (#FC8DC2) or blue (#8EC6FD) drop shadow indicating the speaker.

Return as JSON array. Be precise with timestamps. Include ALL subtitle lines."""},
                {"type": "video_url", "video_url": {"url": f"https://www.youtube.com/watch?v={vid}"}}
            ]}],
            "response_format": {"type": "json_object"}
        },
        timeout=180
    )

    if progress_cb: progress_cb('Parsing response...', 0.7)

    content = r.json()['choices'][0]['message']['content']
    data = json.loads(content)
    if isinstance(data, dict):
        data = data.get('subtitles', data.get('results', data.get('lines', [])))

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


def generate_gemini_srt_sync(vid, progress_cb=None):
    """Combined pipeline: binary search + Gemini reads at boundaries + translate."""
    import cv2, numpy as np
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
    def zone_diff(f1, f2):
        g1, g2 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
        return max(cv2.absdiff(g1[:int(h*0.3),:], g2[:int(h*0.3),:]).mean(),
                   cv2.absdiff(g1[int(h*0.75):,:], g2[int(h*0.75):,:]).mean())

    # Binary search
    if progress_cb: progress_cb('Scanning for subtitle boundaries...', 0.1)
    changes = []; prev = frame_at(0)
    for t_ms in range(500, int(dur*1000), 500):
        curr = frame_at(t_ms)
        if curr is None: break
        if zone_diff(prev, curr) > 18: changes.append(t_ms)
        prev = curr

    exact = []
    for c in changes:
        lo, hi = c-500, c; f_lo = frame_at(lo)
        while hi-lo > 33:
            mid = (lo+hi)//2
            if zone_diff(f_lo, frame_at(mid)) > 12: hi = mid
            else: lo = mid
        exact.append(hi)

    deduped = [0]
    for t in exact:
        if t - deduped[-1] > 400: deduped.append(t)

    # Gemini reads
    if progress_cb: progress_cb(f'Gemini reading {len(deduped)} boundaries...', 0.3)
    events = []; prev_texts = set()
    prev_reading = "[]"  # temporal context: what Gemini read last frame
    def normalize(t): return re.sub(r'[！!\-\.。、\s8]+', '', t)

    for i, t_ms in enumerate(deduped):
        frame = frame_at(t_ms)
        if frame is None: continue
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
        b64 = base64.b64encode(buf).decode()
        # Build prompt with temporal context
        ctx = f"\nPrevious frame had: {prev_reading}\nText already showing is the SAME speaker. NEW text is likely the OTHER speaker." if prev_reading != "[]" else ""
        prompt = f"Read subtitle text in this animation frame. The text has colored drop shadows: pink (#FC8DC2) = Speaker A, blue (#8EC6FD) = Speaker B. Identify speaker by shadow color.{ctx}\nReturn JSON array: [{{\"text\": \"...\", \"speaker\": \"pink\" or \"blue\"}}]. If no subtitle, return []."
        try:
            r = requests.post("https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": "google/gemini-2.5-flash",
                      "messages": [{"role": "user", "content": [
                          {"type": "text", "text": prompt},
                          {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}],
                      "response_format": {"type": "json_object"}}, timeout=20)
            data = json.loads(r.json()['choices'][0]['message']['content'])
            if isinstance(data, dict): data = data.get('subtitles', data.get('results', []))
            subs = data if isinstance(data, list) else []
        except: subs = []

        next_t = deduped[i+1] if i+1 < len(deduped) else int(dur*1000)
        for s in subs:
            text = s.get('text','').strip(); speaker = s.get('speaker','?')
            if not text or len(text) < 2: continue
            is_new = not any(SequenceMatcher(None, normalize(text), normalize(pt)).ratio() > 0.6 for pt in prev_texts)
            if is_new:
                events.append({'start': t_ms/1000, 'end': next_t/1000, 'text': text, 'speaker': speaker})
        prev_texts = {s.get('text','') for s in subs if s.get('text','')}
        # Update temporal context for next frame
        if subs:
            prev_reading = json.dumps([{"text": s.get("text",""), "speaker": s.get("speaker","?")} for s in subs], ensure_ascii=False)
        else:
            prev_reading = "[]"

        if progress_cb and (i+1) % 10 == 0:
            progress_cb(f'Gemini: {i+1}/{len(deduped)} boundaries...', 0.3 + 0.4 * i / len(deduped))

    # Merge
    merged = []
    for e in events:
        if merged and SequenceMatcher(None, normalize(e['text']), normalize(merged[-1]['text'])).ratio() > 0.6:
            merged[-1]['end'] = e['end']
        else: merged.append(dict(e))
    merged = [m for m in merged if m['end'] - m['start'] >= 0.3]

    cap.release()
    video_path.unlink(missing_ok=True)

    # Translate
    if progress_cb: progress_cb(f'Translating {len(merged)} lines...', 0.8)
    numbered = '\n'.join(f"{i+1}. {m['text']}" for i, m in enumerate(merged))
    try:
        r = requests.post("https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": "google/gemini-2.5-flash",
                  "messages": [{"role": "user", "content": f"Translate each Japanese line to Traditional Chinese. Comedy anime. Output ONLY numbered translations.\n\n{numbered}"}]},
            timeout=60)
        translations = {}
        for line in r.json()['choices'][0]['message']['content'].strip().split('\n'):
            parts = line.strip().split('. ', 1)
            if len(parts) == 2 and parts[0].isdigit(): translations[int(parts[0])-1] = parts[1]
    except: translations = {}

    # Write SRT
    if progress_cb: progress_cb('Writing SRT...', 0.95)
    def fmt(s):
        hh=int(s//3600); m=int((s%3600)//60); sec=int(s%60); ms=int((s%1)*1000)
        return f"{hh:02d}:{m:02d}:{sec:02d},{ms:03d}"
    srt_lines = []
    for i, m in enumerate(merged):
        srt_lines.append(str(i+1))
        srt_lines.append(f"{fmt(m['start'])} --> {fmt(m['end'])}")
        srt_lines.append(f"[{m['speaker']}] {m['text']}")
        if i in translations: srt_lines.append(translations[i])
        srt_lines.append('')
    srt_path.write_text('\n'.join(srt_lines), encoding='utf-8')
    return srt_path


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
            {'whisper': 'Whisper only (fast, free)',
             'gemini_video': 'Gemini video (1 call, ~$0.01)',
             'gemini': 'Gemini frame-by-frame (legacy)'},
            value='gemini_video', label='Engine'
        ).classes('w-48')
        model_select = ui.select(
            {'base': 'base (best timing)', 'small': 'small', 'medium': 'medium'},
            value='base', label='Whisper model'
        ).classes('w-48')
        offset_slider = ui.slider(min=-5, max=5, step=0.1, value=0).props('label-always').classes('w-48')
        ui.label('Offset (s)').classes('text-xs')

    status_label = ui.label('').classes('text-sm text-gray-400')
    progress = ui.linear_progress(value=0, show_value=False).classes('w-full').style('display:none')

    player_container = ui.element('div').classes('w-full flex justify-center mt-4')
    sub_label = ui.label('').classes('text-2xl font-bold text-center w-full').style(
        'min-height:80px; white-space:pre-wrap; margin-top:8px;')

    _state = {'timer': None, 'subs': []}

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
                    t_adj = t + offset_slider.value  # apply user offset
                    current = ''
                    for s in _state['subs']:
                        if s['start'] <= t_adj <= s['end']:
                            current = s['text']
                            break
                    sub_label.set_text(current)
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
        existing = find_existing_srt(vid)
        if existing:
            status_label.set_text(f'Found cached subtitles: {existing.name}')
            progress.style('display:none')
            subs = parse_srt(existing)
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
            subs = parse_srt(srt_path)
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
        existing = find_existing_srt(vid)
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
        existing = find_existing_srt(vid)
        if not existing:
            ui.notify('Run Go first to generate base subs', type='negative')
            return

        subs = parse_srt(existing)
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

            new_subs = parse_srt(srt_path)
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
