# AI Persona

RAG-grounded AI representative built for Scaler's AI Engineer screening assignment.
Answers questions about my background, research, and GitHub repos. Books interviews via Cal.com.

**Live:** `https://YOUR_RENDER_URL` · **Phone:** `+1-XXX-XXX-XXXX` (Vapi number)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CALLER / USER                           │
└─────────────┬───────────────────────────────┬──────────────────┘
              │ Phone call                     │ HTTPS chat
              ▼                                ▼
┌─────────────────────┐             ┌──────────────────────────┐
│     VAPI             │             │   Chat UI (HTML/JS)      │
│  - Phone number      │             │   frontend/index.html    │
│  - STT: Deepgram     │             └──────────┬───────────────┘
│  - TTS: ElevenLabs   │                        │ POST /chat
│  - Barge-in support  │                        │
└──────────┬──────────┘                        │
           │ POST /vapi/webhook                 │
           │ (function calls)                   │
           ▼                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Server  (main.py)                    │
│                    Deployed on Render (free)                    │
└───────────────────────────┬─────────────────────────────────────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
   ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐
   │  RAG Pipeline │  │  Cal.com API  │  │  Vapi Webhook    │
   │  rag/         │  │  tools/       │  │  Handler         │
   │               │  │  calendar.py  │  │  vapi/handler.py │
   │ ChromaDB      │  │               │  └──────────────────┘
   │ (local/disk)  │  │ book_appt()   │
   │               │  │ check_avail() │
   │ Embeddings:   │  └──────────────┘
   │ all-MiniLM-L6 │
   │ (free, local) │
   └──────┬────────┘
          │ top-k chunks
          ▼
   ┌──────────────┐
   │ lama-3.1-8b-
      instant  │
   └──────────────┘

Data sources ingested into ChromaDB:
  ├── data/zidan_resume.pdf
  ├── github:zidan18za/research-paper-rag    (README + commits + file tree)
  ├── github:zidan18za/models-from-scratch   (README + commits + file tree)
  └── github:zidan18za/twitter-sentiment     (README + commits + file tree)
```

---

## Stack & Cost Breakdown

| Component | Service | Cost | Free Alternative |
|-----------|---------|------|-----------------|
| LLM inference | Groq (lama-3.1-8b-instant) | Free (6k TPM) | Ollama (fully local) |
| Embeddings | all-MiniLM-L6-v2 | Free (local) | — already free |
| Vector DB | ChromaDB | Free (local/disk) | FAISS |
| Voice agent | Vapi | ~$0.05/min, $10 free credit | Retell AI (similar free credit) |
| TTS voice | ElevenLabs | Free tier: 10k chars/month | Vapi built-in voices (free) |
| STT | Deepgram (via Vapi) | Included in Vapi pricing | — |
| Phone number | Vapi provisioned | ~$2/month | Twilio ($1/mo + usage) |
| Calendar | Cal.com | Free (open-source) | Calendly (paid for API) |
| Hosting | Render free tier | Free | Railway ($5 free/month) |
| Total per call | | ~$0.05–0.15 | |
| Total per chat | | <$0.001 | |

---

## Quickstart

### Prerequisites
- Python 3.11+
- Free accounts: [Groq](https://console.groq.com), [Cal.com](https://cal.com), [Vapi](https://vapi.ai), [Render](https://render.com)

### 1. Clone & Install
```bash
git clone https://github.com/zidan18za/zidan-ai-persona
cd zidan-ai-persona
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure env
```bash
cp .env.example .env
# Fill in GROQ_API_KEY (required), and optional GITHUB_TOKEN, CAL_API_KEY
```

### 3. Add your resume
```bash
mkdir -p data
cp /path/to/zidan_resume.pdf data/
```

### 4. Build vector store
```bash
python rag/ingest.py
# For fresh rebuild: python rag/ingest.py --rebuild
```

### 5. Run locally
```bash
uvicorn main:app --reload --port 8000
# Open http://localhost:8000
```

### 6. Run evals
```bash
python eval/runner.py
# Outputs eval_results.json + summary table
```

---

## Deployment (Render free tier)

Connect your GitHub repo to Render — it auto-deploys on push using `render.yaml`.

Add env vars in the Render dashboard under the Environment tab.

---

## Vapi Voice Agent Setup

1. Sign up at [vapi.ai](https://vapi.ai) (free $10 credit)
2. Go to Assistants → Create → upload `vapi/assistant_config.json`
3. Change `serverUrl` in the JSON to your deployed Render URL
4. Go to Phone Numbers → Buy a number (~$2/month)
5. Assign the assistant to the phone number
6. In Vapi dashboard → Providers → add your `GROQ_API_KEY` for the Groq provider

---

## Cal.com Setup

1. Sign up at [cal.com](https://cal.com) (free)
2. Create a new event type: "30-min Interview"
3. Settings → API Keys → create one
4. Get your event type ID:
   ```bash
   curl https://api.cal.com/v2/event-types \
     -H "Authorization: Bearer YOUR_CAL_KEY" \
     -H "cal-api-version: 2024-09-04"
   ```
5. Copy the ID into your `.env` as `CAL_EVENT_TYPE_ID`

---

## Failure Modes & Fixes

| Failure | Root Cause | Fix |
|---------|-----------|-----|
| Hallucinated metrics | LLM fills gaps when context is empty | Keyword fact injection + explicit refusal in prompt |
| Cold start latency >2s | Render free tier spins down after 15 min | Pre-warm cron job pinging /health every 14 min |
| GitHub fetch fails | Rate limit without token | Set GITHUB_TOKEN — raises limit from 60 to 5000 req/hr |

---

## Repo Structure

```
zidan-ai-persona/
├── main.py                   # FastAPI entrypoint
├── rag/
│   ├── ingest.py             # PDF + GitHub → ChromaDB
│   ├── retriever.py          # Semantic search + Groq generation
│   └── prompts.py            # Persona system prompts
├── tools/
│   └── calendar.py           # Cal.com v2 booking
├── vapi/
│   ├── handler.py            # Webhook handler
│   └── assistant_config.json # Vapi assistant definition
├── eval/
│   └── runner.py             # Hallucination + latency evals
├── frontend/
│   └── index.html            # Chat UI
├── data/                     # Put zidan_resume.pdf here (gitignored)
├── render.yaml
├── requirements.txt
└── .env.example
```
