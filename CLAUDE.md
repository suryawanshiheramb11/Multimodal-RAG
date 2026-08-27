# Evidence Ingestion Pipeline — project context

Multi-modal forensic evidence ingestion. Takes a case folder of mixed media
(video, audio, image, PDF), registers every file with a cryptographic hash, and
extracts it into a queryable evidence graph in Postgres + pgvector.

**This file is the reference for all work on this repo. Read it before adding a
feature; update it when you change an invariant.**

Phase 1 (ingestion) is complete and tested. Phases 2+ (embeddings, face/voice
clustering, entity extraction, identity resolution) build on `evidence_node`.

---

## Commands

```bash
venv/bin/python3 -m ingestion ingest          # scan + ingest the configured case
venv/bin/python3 -m ingestion ingest -v       # with debug logging
venv/bin/python3 -m ingestion verify          # counts currently stored
venv/bin/python3 -m pytest tests/ -q          # 319 tests, ~4s

# Web API. Note the absent --reload: see "Running the web API" below.
venv/bin/python3 -m uvicorn web.api.main:app --port 8000
```

Postgres runs as a brew service: `brew services start postgresql@17`.
The DB password comes from `EVIDENCE_DB_PASSWORD` (see `.env.example`), never
from `config.yaml`.

---

## Architecture

Data flows in one direction, and each stage only knows the stage's contract:

```
config.yaml ──> AppConfig (pydantic)
                    │
CaseScanner ────────┴──> [ScannedFile]        walk, contain, sniff, hash
                              │
ProcessorRegistry ────────────┴──> [EvidenceNodeDraft]   decode & extract
                                        │
Repositories ───────────────────────────┴──> Postgres
```

| Module | Responsibility |
|---|---|
| `config.py` | Load + validate YAML; database settings from env |
| `models.py` | Domain types (`ScannedFile`, `EvidenceNodeDraft`, `MediaType`) — no I/O |
| `security.py` | Path containment, type sniffing, hashing, permissions |
| `workspace.py` | Where derived artefacts go (names from SHA256, not filenames) |
| `scanner.py` | Folder walk → validated, hashed `ScannedFile` list. **No DB.** |
| `media/` | `probe` (PyAV), `scenes` (PySceneDetect), `frames` (PyAV), `audio` (ffmpeg) |
| `processors/` | One class per media type, returning drafts. **No DB.** |
| `repositories.py` | All SQL. Nothing else in the codebase writes SQL. |
| `pipeline.py` | Sequencing + error policy only |
| `cli.py` | Typer commands, Rich output |

### Design rules

These are the ones that will bite you if broken:

1. **Processors never touch the database.** They return `EvidenceNodeDraft`
   objects; the pipeline persists them. This is what lets processor tests run
   without Postgres.
2. **All SQL lives in `repositories.py`**, with fixed column lists and bound
   parameters. No identifier or value is ever interpolated into a query string.
3. **Adding a media type is pure addition**: write a `FileProcessor` subclass,
   register it in `processors/__init__.py::build_registry`. Nothing else
   changes — `pipeline.py` dispatches through the registry.
4. **Collaborators are constructor-injected.** `build_registry` and
   `build_pipeline` are the only places that name concrete classes.
5. **One bad file must never abort a run.** Per-file failures become a
   `FileReport(status="failed")`; only config/connection/schema errors are fatal.
6. **Re-running must be idempotent.** `replace_for_source` deletes a file's
   existing nodes before inserting. (An earlier version did not, and silently
   doubled every node on the second run.)
7. **An unchanged file is never re-extracted.** `replace_for_source` deletes
   before inserting, so rebuilding a file also destroys the *enrichment* phase 2
   wrote onto its nodes — embeddings, transcripts, captions that cost hours of
   inference and that ingestion cannot regenerate. `IngestionPipeline` therefore
   compares the stored hash first and reports `status="unchanged"`, preserving
   the rows. `ingest --reprocess` forces the rebuild (and warns per file about
   the enrichment it discards) when extraction logic itself has changed.
   `verify` reports `N enriched` so a stale case is visible before `build` runs
   against empty embeddings.

---

## Security model

The case folder is **untrusted input**. Evidence arrives from seized devices,
third parties, and opposing counsel. Every guard below has a test in
`tests/test_security.py` or `tests/test_repositories.py`.

| Threat | Control | Where |
|---|---|---|
| Symlink escaping the case folder | Resolve both sides, assert containment; refuse symlinks outright | `security.resolve_within`, `assert_regular_file` |
| Directory traversal via config | `FileEntry.path` rejects absolute paths and `..` | `config.FileEntry` |
| Symlinked directory loops | `rglob(recurse_symlinks=False)` | `scanner._walk` |
| FIFOs/devices blocking on read | `assert_regular_file` | `security.py` |
| Extension/type masquerade | Content sniffing (puremagic) beats declaration **and** extension; conflict stored in `source_file.type_mismatch` | `security.classify` |
| Decompression bombs (image) | Explicit pixel check + Pillow's `MAX_IMAGE_PIXELS` | `processors/image.py` |
| Decompression bombs (PDF) | Page cap; render zoom clamped to a pixel budget | `processors/pdf.py` |
| Oversized files | `max_file_bytes` before any decode | `scanner._inspect` |
| Runaway video | `max_video_seconds`, global `max_frames_per_file` | `media/scenes.py`, `media/frames.py` |
| ffmpeg hanging forever | `timeout`, `-nostdin`, `stdin=DEVNULL` | `media/audio.py` |
| Argument injection into ffmpeg (`-i.mp4`) | Absolute paths only — they cannot parse as flags | `security.assert_safe_external_path` |
| Filename injection into output paths | Derived names come from the SHA256 prefix | `workspace.py` |
| SQL injection via filename/metadata/text | Bound parameters everywhere | `repositories.py` |
| Credential in version control | Password from env only; `password:` in YAML is dropped | `config.DatabaseSettings.build` |
| Credential in logs | `SecretStr` + `safe_dsn()` | `config.py`, `db.py` |
| Sensitive derivatives world-readable | Dirs `0700`, files `0600` | `security.ensure_private_dir`, `harden_file` |
| Evidence tampering between runs | Same path + different hash is flagged | `repositories.SourceFileRepository.register` |

**Rules for new code:**
- Never pass a user-influenced path to a subprocess without
  `assert_safe_external_path`.
- Never build SQL with f-strings or `.format()`.
- Any new decoder gets a resource limit in `config.Limits` before it ships.
- Originals are read-only. Derived artefacts go in `data/`, never beside the
  evidence.

---

## Database

Postgres 17 + pgvector 0.8.6, database `evidence_db`. Schema in
`db/schema.sql`, applied automatically on every run (all statements are
`IF NOT EXISTS`, so it doubles as the migration path).

Nine tables: `case`, `source_file`, `evidence_node`, `entity`, `face_cluster`,
`voice_cluster`, `identity`, `relationship`, `source_alignment`.

Vector dimensions on `evidence_node` — **these must match the models that fill
them**:

| Column | Dim | Model |
|---|---|---|
| `text_embedding` | 384 | sentence-transformers MiniLM |
| `clip_embedding` | 512 | CLIP ViT-B/32 |
| `audio_embedding` | 1024 | speaker embedding |

HNSW cosine indexes exist on the text and CLIP columns.

`CREATE EXTENSION` needs superuser. For a least-privilege role, create the
extensions once as superuser and run with `ingest --skip-schema`.

### What phase 1 writes

| Media | `node_type` | Contents |
|---|---|---|
| video | `scene_segment` | one per segment; `metadata.frames[]` lists frame paths + timestamps; `file_path` is the extracted WAV |
| audio | `audio_track` | whole recording, normalised WAV |
| pdf | `page` | per-page text + rendered PNG |
| image | `image` | dimensions; `file_path` points at the original |

Embedding columns are left NULL for phase 2 to fill.

---

## Library choices

Prefer an existing library over hand-rolled code. Already resolved:

| Need | Use | Not |
|---|---|---|
| Config validation | pydantic / pydantic-settings | hand-written getters |
| File hashing | `hashlib.file_digest` (3.11+) | manual chunk loop |
| Content type detection | puremagic | extension maps alone |
| Video probe + frame decode | PyAV | OpenCV `CAP_PROP_POS_FRAMES` seeking — slow and inaccurate on B-frames |
| Audio transcode | ffmpeg subprocess | hand-rolled libswresample |
| Scene detection | PySceneDetect `ContentDetector` | frame differencing by hand |
| Date parsing | pydantic / `dateutil` | `strptime` chains |
| CLI + output | Typer + Rich | `argparse` + `print` |
| Bulk insert | `psycopg2.extras.execute_values` | per-row `execute` |

Installed and waiting for phase 2: faster-whisper, transformers, torch,
torchvision, ultralytics, paddleocr, sentence-transformers, ollama, librosa,
scikit-learn.

---

## Running the web API

**Never start the server with a bare `--reload` while a job can be running.**
`watchfiles` is not installed, so uvicorn falls back to `StatReload`, which
`rglob("*.py")`s the whole working directory — 25k files, almost all of them in
`venv/` — four times a second, and SIGKILLs the app process when any of them
changes. Jobs live in an in-process registry (`web/api/jobs.py`) on a worker
thread, so a restart destroys a running enrichment mid-node: the console keeps
the last line the dead process printed and the job never completes or fails.
Installing a package is enough to trigger it.

```bash
venv/bin/python3 -m uvicorn web.api.main:app --port 8000            # jobs are safe
venv/bin/pip install watchfiles                                     # if you want reload
venv/bin/python3 -m uvicorn web.api.main:app --port 8000 \
  --reload --reload-dir web/api --reload-exclude 'data/*'
```

---

## Enrichment performance

Feature extraction is the expensive phase, and the costs are not where they
look. Measured on a 65s screen recording, 10 scene segments, Apple Silicon:

| | Before | After |
|---|---|---|
| Model load (once per run) | 30.4s | 10.0s |
| First node (carries ASR of the whole track) | 69.5s | 20.5s |
| Each later node | 40.1s | 6.9s |

What actually mattered, in order:

1. **OCR was 60% of a node.** PaddleOCR's cost tracks the number of text
   regions it detects, so a 3600px frame cost 26s. Capping the longest side
   (`ocr_max_side`, default 2400) and naming the mobile checkpoints
   (`ocr_det_model` / `ocr_rec_model` — PaddleOCR otherwise picks the slower
   `medium` pair) took it to ~10s *and* found more lines, not fewer.
2. **Captioning looked like 12s and was really a 10s model reload.** ollama
   evicts a model after 5 minutes by default; `ollama_keep_alive` pins it, and
   `_build` warms it during the availability pass. Generation is ~24ms/token,
   so the prompt and `caption_max_tokens` are the remaining levers — hence a
   one-sentence `DEFAULT_CAPTION_PROMPT`.
3. **The slow stages are not in this interpreter.** OCR is a round-trip to the
   worker process and captioning is an HTTP call, so `VisualExtractor` runs
   both on a thread pool while CLIP and YOLO run on the calling thread
   (`parallel_stages`). torch stays on one thread — MPS dislikes concurrent
   callers. `ModelRegistry.availability()` loads all seven models the same way.
4. **Whisper was using 4 of 12 cores.** `asr_cpu_threads` (0 = all) plus the
   batched pipeline (`asr_batch_size`) took a 65s track from 18.1s to 10.4s.
   Batching picks slightly coarser segment boundaries; set it to 1 to opt out.

Two invariants this adds:

- **Anything the offloaded stages write is guarded.** `VisualFeatures.lock`
  covers `metadata` and the shared `EnrichmentResult`; `LazyModel.load` is
  locked so a model is built exactly once under contention. For `OcrReader` a
  double build would mean two spawned processes and one leaked.
- **The OCR worker must survive its parent dying badly.** `daemon=True` only
  covers a clean exit. The worker polls `req_queue` with a timeout and exits
  when `os.getppid()` changes, and replies carry the id of the request they
  answer so a timed-out call cannot hand its late result to the next frame.

---

## Environment notes

- **Python 3.13**, not 3.14 — torch and paddleocr have no 3.14 wheels.
- **Postgres 17**, not 16 — brew's `pgvector` bottle only ships extension
  binaries for 17 and 18.
- Homebrew's post-install cleanup once removed `python@3.13` mid-install and
  corrupted the venv. Set `HOMEBREW_NO_INSTALL_CLEANUP=1` when brew-installing
  during a session.
- `opencv-python` is installed (ultralytics needs it) but the ingestion code
  no longer uses it for decoding.

---

## Testing

319 tests, no network, ~4s. `tests/test_repositories.py` needs Postgres and
skips cleanly without it; `tests/test_video.py` needs ffmpeg and builds its own
fixtures with it.

When adding a feature, add tests in the matching file:
`test_security.py` (guards), `test_config.py` (validation/secrets),
`test_processors.py` (extraction + limits), `test_pipeline.py` (orchestration,
using the fake repositories already defined there), `test_video.py` (real
decode), `test_repositories.py` (SQL).

A security control without a test that tries to break it is not a control.
