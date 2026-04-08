# Pipeline Comparison: Kitchen Car Video (71s)

## Final Results

| Pipeline | Subs | Time | API calls | Cost | Timing | Speakers |
|---|---|---|---|---|---|---|
| Whisper base | 59 | 3s | 0 | Free | 84ms | No |
| Whisper medium | 59 | 70s | 0 | Free | 383ms | No |
| Visual YOLO+OCR | 23 | 44s | 0 | Free | 33ms | Yes |
| **Gemini visual** | **75** | **250s** | **64** | **$0.006** | **33ms** | **Yes** |
| Smart (Whisper+Gemini) | 56 | 132s | 28 | $0.003 | 33ms | Partial |

## Multi-AI Round Table

3 models read the same frames — all correct where PP-OCR failed:

| Model | Speed | Accuracy | Cost/call |
|---|---|---|---|
| Gemini Flash | 3s | 100% | ~$0.0001 |
| GPT-4.1 mini | 3s | 100% | ~$0.0002 |
| Llama Maverick | 2s | ~95% | ~$0.0001 |

## Key Discoveries

1. **Whisper base > medium for timing** (84ms vs 383ms, 10x faster)
2. **PP-OCR mobile > server on stylized text** (correct 辛 where server failed)
3. **Gemini reads styled Japanese perfectly** (zero errors, zero false positives)
4. **Binary search gives 33ms precision** on subtitle boundaries
5. **Speaker colors auto-detected** via pink/blue shadow on text
6. **Multi-AI consensus** confirms readings — redundancy at pennies

## Overnight Benchmarks

### Whisper Model Comparison (MariMariMarie GT, ms-precision)

| Model | Speed | Mean offset | p90 | Drift |
|---|---|---|---|---|
| **base** | **10x** | **84ms** | **140ms** | **0.02s** |
| small | 4x | 114ms | 270ms | -0.3s |
| medium | 1.7x | 383ms | 830ms | -0.2s |
| turbo | 2.5x | 227ms | 520ms | -0.02s |

Smaller = better timing. Counter-intuitive but consistent.

### Batch Processing (4 videos)

| Video | Duration | Subs | Gemini calls | Time |
|---|---|---|---|---|
| Kitchen car | 71s | 75 | 64 | 250s |
| Strike zone | 93s | 44 | 52 | 239s |
| Ghost story | 106s | 16 | 29 | 123s |
| Momotaro gacha | 85s | 48 | 57 | 220s |

All speaker-tagged, all translated to Traditional Chinese.
Total cost: ~$0.025 for 4 videos.
