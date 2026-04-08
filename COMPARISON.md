# Pipeline Comparison: Kitchen Car Video (71s)

## Methods

| Pipeline | Description | API calls | Cost |
|---|---|---|---|
| **Whisper base** | Audio transcription only | 0 | Free |
| **Whisper medium** | Audio transcription (gen1 default) | 0 | Free |
| **Visual YOLO+OCR** | Binary search → YOLO → PP-OCR per event | 0 | Free |
| **Gemini visual** | Binary search → Gemini reads at boundaries | ~70 | ~$0.007 |
| **Combined** | Whisper base timing + Gemini visual content + speaker tags | ~70 | ~$0.007 |

## Results (to be filled after overnight runs)

| Metric | Whisper base | Whisper medium | Visual YOLO+OCR | Gemini visual | Combined |
|---|---|---|---|---|---|
| Time | 3s | 70s | 44s | ~200s | TBD |
| Subtitle lines | 59 | 59 | 23 | TBD | TBD |
| Timing precision | 84ms | 383ms | 33ms | 33ms | 33ms |
| Drift | +0.02s | -0.22s | N/A | N/A | N/A |
| Content accuracy | ~95% | ~95% | ~70% | ~99%? | ~99%? |
| Speaker separation | No | No | Yes (color) | Yes (color) | Yes |
| Visual-only text | No | No | Yes | Yes | Yes |
| Cost | Free | Free | Free | ~$0.007 | ~$0.007 |

## Key Findings

### Whisper
- Base model: best timing (84ms), 10x faster, near-zero drift
- Medium: better content on edge cases but 4x worse timing
- Cannot read visual-only text (end cards, signs)
- No speaker separation

### Visual (YOLO + OCR)  
- Frame-perfect timing (33ms)
- Speaker colors detected automatically
- OCR accuracy limited by PP-OCR on stylized text (~70%)
- Fully local, no API cost

### Gemini Visual
- Perfect text reading (zero errors on test frames)
- Speaker color identification built-in
- Frame-perfect timing via binary search
- ~$0.007 per 1-min video (~20-70 calls)
- 3s latency per call limits throughput

### Best of All Worlds (proposed)
- Whisper base for fast initial transcription + rough timing
- Binary search for frame-perfect change boundaries  
- Gemini reads at boundaries for perfect text + speaker tags
- Merge: Whisper content confirmed/corrected by Gemini visual
- Total: ~$0.007, frame-perfect, speaker-tagged, bilingual
