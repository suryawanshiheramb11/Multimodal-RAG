# Evidence Ingestion Pipeline

Multi-modal forensic evidence ingestion: takes a case folder of mixed media,
registers every file with a SHA256 hash, and extracts it into a queryable
evidence graph in Postgres + pgvector.

See [CLAUDE.md](CLAUDE.md) for architecture, design rules, and the security model.

## Quick start

```bash
# Postgres (once)
brew services start postgresql@17
createdb -U "$(whoami)" -O postgres evidence_db
psql -U postgres -d evidence_db -c 'CREATE EXTENSION IF NOT EXISTS vector'

# Credentials (the password is never read from config.yaml)
cp .env.example .env && set -a && source .env && set +a

# Run
venv/bin/python3 -m ingestion ingest     # scan + ingest the configured case
venv/bin/python3 -m ingestion verify     # what is currently stored
venv/bin/python3 -m pytest tests/        # 64 tests, ~1s
```

## What it does

Reads `config.yaml`, applies `db/schema.sql`, walks the case folder, and for
each file: validates it is safely contained, sniffs its real type from content,
hashes it, registers it in `source_file`, then extracts it into `evidence_node`
rows.

| Media | Extraction |
|---|---|
| video | 16 kHz mono WAV (ffmpeg) + PySceneDetect segments (5s fixed-window fallback) + 1 frame/sec/segment (PyAV) |
| audio | normalised to 16 kHz mono WAV |
| pdf | per-page text + rendered page image (PyMuPDF) |
| image | decoded, dimensions recorded |

Re-running is idempotent — a file's nodes are replaced, not duplicated.

## Security

The case folder is treated as untrusted input. Symlink escapes, path traversal,
extension masquerading, decompression bombs, oversized files, runaway videos,
and SQL injection are all blocked, and each control has a test that tries to
break it. Credentials come from the environment only. Derived artefacts are
written `0600` in `0700` directories. A file whose content changed since the
last ingest is flagged rather than silently re-added.

Full table of threats and controls: [CLAUDE.md](CLAUDE.md#security-model).

## Layout

```
config.yaml         case metadata, paths, processing knobs, resource limits
db/schema.sql       9 tables; vector dims 384 (text) / 512 (CLIP) / 1024 (audio)
ingestion/
  config.py         pydantic config; DB settings from env
  models.py         domain types, no I/O
  security.py       containment, type sniffing, hashing, permissions
  workspace.py      derived-artefact paths (named from SHA256)
  scanner.py        folder -> validated, hashed files
  media/            probe, scenes, frames, audio
  processors/       one class per media type + registry
  repositories.py   all SQL
  pipeline.py       sequencing + error policy
  cli.py            typer commands
sample_case/        one video, audio, image, pdf for smoke testing
data/               extracted audio/frames/pages (gitignored)
```

## Next phase

`evidence_node` embedding columns (`text_embedding`, `clip_embedding`,
`audio_embedding`) are left NULL for the enrichment phase: transcription
(faster-whisper), OCR (paddleocr), object detection (ultralytics), embeddings
(sentence-transformers / CLIP), then face and voice clustering into
`identity`.
