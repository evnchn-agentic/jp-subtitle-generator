# Feasibility Assessment: JP Subtitle Generator as a Service

## Cost Analysis

### Per-video cost breakdown (1-min video)

| Component | Method | Cost | Time |
|---|---|---|---|
| Download | yt-dlp | Free | ~5s |
| Transcription | Whisper base (local) | Free (CPU) | ~3s |
| Timing alignment | Gemini Flash (20 calls) | ~$0.002 | ~60s |
| Translation | Claude claude -p (60 lines) | ~$0.01 | ~10s |
| **Total** | | **~$0.012** | **~78s** |

### Per-video cost (10-min video)

| Component | Method | Cost | Time |
|---|---|---|---|
| Download | yt-dlp | Free | ~15s |
| Transcription | Whisper base (local) | Free (CPU) | ~30s |
| Timing alignment | Gemini Flash (~50 calls) | ~$0.005 | ~150s |
| Translation | Claude (200 lines) | ~$0.03 | ~30s |
| **Total** | | **~$0.035** | **~225s** |

### At scale

| Videos/month | Cost/month | Revenue @ $0.50/video | Margin |
|---|---|---|---|
| 100 | $3.50 | $50 | 93% |
| 1,000 | $35 | $500 | 93% |
| 10,000 | $350 | $5,000 | 93% |

### Infrastructure

- **Compute**: M4 MacBook Air handles ~1x realtime. A $50/month cloud GPU would do 5-10x.
- **Storage**: SRT files are tiny (<50KB). Video not stored (download, process, delete).
- **Bandwidth**: yt-dlp downloads, no hosting needed.

## Pricing Models

### Option A: Pay-per-use
- $0.50 per video (short, <5 min)
- $1.00 per video (long, 5-30 min)
- Margin: >90%

### Option B: Subscription
- $5/month: 20 videos
- $15/month: 100 videos
- $30/month: unlimited

### Option C: Freemium
- 3 free videos/month (Whisper-only, no Gemini alignment)
- Paid: adds Gemini timing + speaker separation + priority

## Legal / Terms Considerations

### Must check:
- **YouTube ToS**: yt-dlp downloads for processing — gray area. Alternative: user provides video file directly
- **OpenRouter / Gemini API ToS**: commercial use of API outputs — generally allowed
- **Anthropic Claude API ToS**: commercial use of translations — allowed under standard API terms
- **Whisper license**: MIT — fully commercial-use compatible
- **PaddleOCR**: Apache 2.0 — commercial-use compatible
- **NiceGUI**: MIT — commercial-use compatible
- **Content copyright**: subtitles are derivative works of the original video content
  - Fair use argument: transformative (translation), educational (accessibility)
  - Risk: content owners could claim subtitles reproduce their dialogue
  - Mitigation: user-initiated processing (like a tool, not a library of pre-made subs)

### Safest model:
User provides URL → tool processes on-demand → user gets SRT file.
We never store/redistribute the subtitles or video. We're a tool, not a content provider.

## Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Whisper accuracy drops on unusual content | Medium | Medium | Fall back to Gemini for full visual read |
| yt-dlp breaks (YouTube changes) | High | High | Support direct file upload as alternative |
| API cost spikes | Low | Medium | Cache results, rate limit |
| Gemini API changes | Low | Medium | Abstract behind interface, swap models easily |

## Competitive Landscape

- **Rev.com**: $1.50/min for human transcription. We're $0.01/min.
- **Kapwing/VEED.io**: Auto-subtitle but English-focused, no JP hardcoded sub extraction
- **Subtitle Edit**: Desktop tool, manual, no translation
- **YouTube auto-captions**: Free but terrible on JP content

## Verdict

**Technically feasible and economically viable.** The per-video cost (~$0.01-0.04) leaves massive margin even at low price points. The main risk is legal (YouTube ToS, content copyright), not technical or financial.

**Recommended next step**: Build as a free open-source tool first (current state). If there's demand, add a hosted version with user accounts and payment. The tool-not-library framing is the safest legal position.
