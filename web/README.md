# Prism — web UI

Semantic search over everything the pipeline has ingested. Type what you
remember seeing or hearing; results come back ranked by meaning rather than by
filename.

## Running it

Two processes. Both must be up.

```bash
# 1. API  (from the repo root, not from web/)
venv/bin/python3 -m uvicorn web.api.main:app --reload --port 8000

# 2. UI
cd web/ui && npm install && npm run dev      # http://localhost:5173
```

Postgres must be running (`brew services start postgresql@17`) and ollama must
be up if you want captions (`ollama serve`). The API applies `db/schema.sql` at
startup, so a fresh database needs no extra step.

The first search loads CLIP and MiniLM (a few seconds). The API warms both in
a background thread at startup so that cost is usually paid before you type.

## How search works

Two vector spaces, filled during enrichment, answer different questions:

| Space | Column | Answers |
|---|---|---|
| CLIP (512-d) | `clip_embedding` | what a frame *looks like* |
| MiniLM (384-d) | `text_embedding` | what was *said or written* |

CLIP puts images and text in one space, so a typed phrase is compared directly
against a video frame. That is what makes **"mountains" find mountain footage
in a video nobody ever captioned, tagged, or named that way.**

The three modes map onto that:

- **Everything** — both spaces, merged and weighted (text 0.55, visual 0.45).
  A node found in both scores higher than one found in either alone.
- **What it looks like** — CLIP only.
- **What was said** — MiniLM only, over transcript + caption + OCR + page text.

### Why weak matches are hidden

Every image scores *something* against every phrase, so a naive nearest-
neighbour search returns your whole library sorted by noise — searching
"mountains" in a library with no mountains would still return four confident
looking results. Two filters prevent that (`web/api/search.py`):

1. an absolute cosine floor per space, and
2. a **relative gate** — a hit must stay within a fraction of the best hit.

So a query with no real match returns nothing rather than the library's own
noise. Tuning both is a recall/precision trade: lower them to see more
marginal matches, raise them to see only confident ones.

## The reasoning rail

The left column shows *how* the pipeline reached its conclusions, not just
that it finished. Each upload becomes a vertical chain:

```
▾ meeting_architecture.mp4
  12:01 ✓ Extracting media
        → 5 segment(s) extracted
  12:02 ✓ Running models
        → 5 node(s) analyzed
        ▸ Described: "Architecture diagram with API Gateway…"     ← findings
        ▸ Read on screen: "API Gateway", "Read Replica"
        ▸ Audio events: Speech (0.98), Conversation (0.87)
        │ loading whisper(large-v3-turbo)…                        ← real logs
        │ transcribed meeting.wav: 45 segment(s), language=en
  12:03 ✓ Indexing for search
        → 5 visual + 5 text vector(s) across 5 node(s)
```

Two kinds of line, deliberately styled differently:

- **Findings** (highlighted) — what the models *concluded*: the caption a
  vision model wrote, the words OCR read off a frame, the objects detected,
  the audio events classified. This is the chain that explains why a file
  later matches a given search.
- **Logs** (monospace) — the pipeline's own output, verbatim.

Those log lines are not re-implemented for the UI. `ingestion` and
`enrichment` already narrate their work, so `JobLogRouter` in
`web/api/jobs.py` attaches a handler to those loggers and files each record
under the job whose worker thread emitted it. Routing by thread means
concurrent uploads never mix, and the pipeline packages stay unaware that an
HTTP layer exists — they keep owning sequencing and nothing else.

The rail hydrates from `GET /api/jobs` on load, so reloading the page mid-run
still shows work in flight. Collapse it with the toggle in its header.

## Pipeline integration

Uploading runs the real phases, in order, on a worker thread:

```
upload → ingestion (hash, scan, extract segments/frames/pages)
       → enrichment (Whisper, CLIP, YOLO, PaddleOCR, Qwen captions, AST)
       → searchable
```

The request returns `202` with a job id immediately; the UI polls
`/api/jobs/{id}` while it runs. One video is minutes of model inference, well
past any browser timeout.

Re-uploading a file whose bytes are unchanged is a no-op — the pipeline keeps
the existing nodes and their enrichment rather than rebuilding them.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/search?q=&mode=&collection_id=` | ranked semantic search |
| GET | `/api/search/status` | encoder availability + index coverage |
| GET | `/api/collections` | collections with file/node counts |
| POST | `/api/collections` | create one |
| DELETE | `/api/collections/{id}` | remove it (uploaded files stay on disk) |
| POST | `/api/collections/{id}/upload` | upload → `202` + job id |
| GET | `/api/jobs/{id}` | processing progress |
| GET | `/api/files/{id}/nodes` | segments extracted from a file |
| GET | `/api/nodes/{id}` | one node in full |
| GET | `/api/nodes/{id}/thumbnail` | still image for a node |
| GET | `/api/files/{id}/media` | original file, for playback |

### Serving media safely

The API never accepts a filesystem path from the client. A request names a
node id; `web/api/media.py` looks up that node's stored path and proves it
resolves inside an allowed root using the same `resolve_within` the scanner
uses. Path traversal and planted symlinks are rejected at the HTTP boundary,
and only the media types the pipeline itself produces are served.
