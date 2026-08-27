# Prism: Multimodal Forensic Evidence Intelligence Platform

<<<<<<< HEAD
Prism is an end-to-end multimodal forensic intelligence system. It takes raw case folders containing mixed media (bodycam footage, surveillance video, 911 calls, audio interviews, scanned PDF documents, and crime scene photos), extracts them into a tamper-evident schema in **PostgreSQL + pgvector**, enriches every segment with localized neural networks, builds a cross-modal **Evidence Graph**, synchronizes independent timelines across multiple recording sources, and exposes everything through a **FastAPI backend**, **CLI**, and a **React web interface**.

---

## Architecture Overview

```
                          ┌────────────────────────┐
                          │   Raw Case Evidence    │
                          │ (Video, Audio, PDF, Img)│
                          └───────────┬────────────┘
                                      │
                                      ▼
                   ┌──────────────────────────────────────┐
                   │ Phase 1: Ingestion & Segmentation    │
                   │ • SHA-256 Checksum & Type Sniffing   │
                   │ • PySceneDetect (Video Cuts & Frames)│
                   │ • FFmpeg 16kHz Mono Normalization    │
                   │ • PyMuPDF Page & Text Extraction     │
                   └──────────────────┬───────────────────┘
                                      │
                                      ▼
                   ┌──────────────────────────────────────┐
                   │ Phase 2: Multimodal Neural Enrichment│
                   │ • Faster-Whisper (ASR Transcription) │
                   │ • AST (Audio Spectrogram Classifier) │
                   │ • CLIP ViT-B/32 (Visual Embedding)   │
                   │ • YOLOv8s (Weapon & Object Detection)│
                   │ • PaddleOCR (Spawn Process + SharedM)│
                   │ • Qwen2.5-VL 7B (Vision-Language LLM)│
                   │ • all-MiniLM-L6-v2 (Text Embeddings) │
                   └──────────────────┬───────────────────┘
                                      │
                                      ▼
                   ┌──────────────────────────────────────┐
                   │ Phase 3 & 4: Evidence Graph & Links  │
                   │ • Entity Extraction (LLM + NER)      │
                   │ • Cross-Modal Edges: MENTIONS,       │
                   │   REFERENCES, DESCRIBES, SAME_EVENT  │
                   └──────────────────┬───────────────────┘
                                      │
                                      ▼
                   ┌──────────────────────────────────────┐
                   │ Phase 5: Contradiction & Claims      │
                   │ • Claim Distillation                 │
                   │ • Pairwise LLM Contradiction Judge   │
                   │ • Corroboration & Conflict Edges     │
                   └──────────────────┬───────────────────┘
                                      │
                                      ▼
                   ┌──────────────────────────────────────┐
                   │ Phase 6: Multi-Source Timeline Sync  │
                   │ • AST Audio Fingerprinting (>0.9)    │
                   │ • CLIP Visual Consistency Matching   │
                   │ • Face Cluster Co-Occurrence         │
                   │ • Median Offset Estimation           │
                   │ • `case_time` Unified Alignment      │
                   └──────────────────┬───────────────────┘
                                      │
                                      ▼
                   ┌──────────────────────────────────────┐
                   │ Phase 7: Biometrics & Identity Fusion│
                   │ • InsightFace (Face Clustering)      │
                   │ • Pyannote.audio (Voice Diarization) │
                   │ • Multimodal Identity Resolution     │
                   └──────────────────┬───────────────────┘
                                      │
                ┌─────────────────────┴─────────────────────┐
                ▼                                           ▼
   ┌───────────────────────────┐               ┌───────────────────────────┐
   │  PostgreSQL 17 + pgvector │               │   Interactive Web UI      │
   │  • HNSW Cosine Vectors    │◄─────────────►│   • Unified Timeline View │
   │  • Relational Graph Tables│   FastAPI     │   • Synced Modalities Grid│
   │  • Source Offsets & Graph │   REST API    │   • Complete Transcripts  │
   │  • Full Text Search       │               │   • Hybrid Semantic Search│
   └───────────────────────────┘               └───────────────────────────┘
```

---

## Technology Stack & Models

| Capability | Framework / Model | Execution Device | Architecture Notes |
|---|---|---|---|
| **Speech Recognition** | `faster-whisper` (`large-v3-turbo`) | CPU / MPS | Audio track transcribed once & cached; segmented per scene |
| **Audio Event Tagging** | `MIT/ast-finetuned-audioset` | CPU / MPS | 527 AudioSet classes (gunshots, sirens, screams) |
| **Visual Embeddings** | `openai/clip-vit-base-patch32` | MPS (Metal) / CUDA | 512-dim normalized vectors for image-to-text search |
| **Object Detection** | `yolov8s.pt` (Ultralytics) | MPS (Metal) / CUDA | Fast weapon & object detection with bbox coordinates |
| **OCR Text Extraction** | `PaddleOCR` (PP-OCRv6) | Isolated Subprocess | **`spawn` process isolation** + **POSIX `shared_memory`** (prevents OpenMP clashes) |
| **Vision-Language LLM** | `qwen2.5vl:7b` (via Ollama) | MPS (Metal) / CUDA | Image normalization (HEIC/PNG $\to$ RGB JPEG) + capped token generation |
| **Text Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | MPS (Metal) / CUDA | 384-dim dense vectors over fused multimodal text |
| **Biometrics** | `insightface` (`buffalo_l`) & `pyannote.audio` | CPU / MPS | 512-d face vectors & voice diarization clustered into canonical identities |
| **Database** | PostgreSQL 17 + `pgvector` | Local Server | HNSW indexing for instantaneous vector similarity queries |
| **Backend & API** | FastAPI + Uvicorn + Psycopg2 | Python 3.13 | Non-blocking async endpoints, background job worker thread |
| **Frontend** | React 19 + Vite + Phosphor Icons | Browser (SPA) | Pure Vanilla CSS (custom design system), live reasoning rail |

---

## Prerequisites & Installation

### 1. System Requirements
- **macOS** (Apple Silicon M1/M2/M3/M4 recommended) or **Linux x86_64/ARM64**
- **Python 3.13**
- **Node.js 18+** and `npm`
- **PostgreSQL 17** with `pgvector`
- **Ollama**

---

### 2. Step-by-Step Setup

#### A. Database Setup
```bash
# Start PostgreSQL
brew services start postgresql@17

# Create the database and enable pgvector
createdb -U "$(whoami)" -O postgres evidence_db
psql -U postgres -d evidence_db -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

#### B. Ollama & Vision-Language Model
```bash
# Start Ollama service in background
ollama serve &

# Pull the Vision-Language Model (Qwen2.5-VL 7B)
ollama pull qwen2.5vl:7b
```

#### C. Python Virtual Environment & Backend Setup
```bash
# Create and activate Python 3.13 virtual environment
python3.13 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
```

Ensure `.env` contains:
```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=evidence_db
DB_USER=postgres
DB_PASSWORD=postgres
OLLAMA_HOST=http://localhost:11434
# Optional: for pyannote speaker diarization
# HF_TOKEN=your_huggingface_token
```

#### D. Web UI Setup
```bash
cd web/ui
npm install
cd ../..
```

---

## Running the Platform

### Option 1: Full Web Platform (API + UI)

1. **Start the FastAPI Backend** (runs on port 8000):
   ```bash
   venv/bin/python3 -m uvicorn web.api.main:app --reload --port 8000
   ```

2. **Start the Web UI** (runs on port 5173):
   ```bash
   cd web/ui
   npm run dev
   ```

3. Open **`http://localhost:5173`** in your browser.

---

### Option 2: Command Line Interface (CLI)

The CLI provides complete control over every pipeline phase:
=======
Multi-modal forensic evidence ingestion. Takes a case folder of mixed media (video, audio, image, PDF), registers every file with a cryptographic hash, and extracts it into a queryable evidence graph in Postgres + pgvector.

Phase 1 (ingestion) is complete and tested. Phases 2+ (embeddings, face/voice clustering, entity extraction, identity resolution) build on `evidence_node`.

---

## Commands & Quick Start
>>>>>>> 2849ee06d2658fdd600fa596c3c947cd70a8c64f

```bash
# 1. Ingest media from the configured case folder (config.yaml)
venv/bin/python3 -m ingestion.cli ingest

# 2. Enrich all pending nodes (Whisper, CLIP, YOLO, OCR, Qwen)
venv/bin/python3 -m enrichment.cli enrich

<<<<<<< HEAD
# 3. Build Evidence Graph, Contradictions, and Identity Clustering
venv/bin/python3 -m graph.cli build

# 4. Synchronize Multi-Source Timelines
venv/bin/python3 -m graph.cli sync

# 5. Display Unified Timeline across all sources
venv/bin/python3 -m graph.cli timeline

# 6. Ask Natural Language Questions against the Case Graph
venv/bin/python3 -m graph.cli ask "Who was holding the weapon at the scene?"
=======
# Run
venv/bin/python3 -m ingestion ingest          # scan + ingest the configured case
venv/bin/python3 -m ingestion ingest -v       # with debug logging
venv/bin/python3 -m ingestion verify          # counts currently stored
venv/bin/python3 -m pytest tests/ -q          # 218 tests, ~3s
>>>>>>> 2849ee06d2658fdd600fa596c3c947cd70a8c64f
```

---

<<<<<<< HEAD
## Key Features & How It Works

### 1. Multi-Source Timeline Synchronization (Phase 6)
When an incident is captured across multiple angles (e.g. Bodycam A, Bodycam B, and CCTV), their recording clocks rarely match. Prism aligns them automatically:
- **Audio AST Fingerprinting**: Compares AST embedding vectors across tracks. High-similarity pairs ($>0.9$) are clustered into anchor events (gunshots, sirens).
- **Visual Temporal Matching**: Computes CLIP frame-to-frame similarity across video sources and verifies temporal offset consistency.
- **Identity Co-Occurrence**: Correlates appearances of the same face cluster across cameras.
- **Median Offset Estimation**: Computes the median offset ($\Delta t = t_A - t_B$) with a confidence score based on residual agreement ($\pm 1.0\text{s}$).
- **`case_time` Normalization**: Automatically updates all `evidence_node.case_time` columns so all evidence sits on a unified timeline.

---

### 2. Synchronized Modalities & Complete Transcript Display
- **Segment-Level Synchronization**: In the UI, each video segment displays **Transcript**, **Visual Description**, and **On-Screen Text (OCR)** in parallel columns aligned to the exact same moment.
- **Clickable Timestamp Seeking**: Clicking any segment timestamp (e.g. `[01:24]`) directly seeks and plays the video/audio player at that precise second.
- **Full File Transcript**: The `GET /api/files/{id}/transcript` endpoint stitches all timestamped Whisper segments across the entire file into a seamless, searchable transcript drawer.

---

### 3. Strict Process Isolation Architecture (PaddleOCR + PyTorch)
On macOS (Apple Silicon) and Linux, PyTorch and PaddlePaddle both manage C++ OpenMP thread pools and native memory allocators. Loading both in the same process or relying on `fork` causes deadlocks and memory corruption.

Prism implements an **Isolated Worker Subprocess** ([`enrichment/models/ocr.py`](enrichment/models/ocr.py)):
- Spawns a dedicated PaddleOCR worker via `multiprocessing.get_context('spawn')`.
- Heavy image tensors use **POSIX `shared_memory`** (`SharedMemory`) for **zero-copy transfer** without queue pickling bottlenecks.
- IPC Queues are reserved strictly for lightweight command metadata.
- Preprocessors (such as document unwarping) are disabled on video keyframes for rapid sub-second inference.

---

### 4. Semantic Hybrid Search
Prism provides 3 search modes powered by pgvector HNSW indexing:
- **Everything (Hybrid)**: Combines CLIP visual similarity + MiniLM text embeddings ($0.45 \cdot \text{visual} + 0.55 \cdot \text{text}$).
- **What it looks like (Visual)**: Natural language query matched directly against video frames via CLIP (`ViT-B/32`).
- **What was said (Text)**: Dense vector search over spoken transcripts, visual captions, OCR text, and PDF document text.

---

## Directory Structure

```
.
├── config.yaml               # Pipeline configurations & thresholds
├── requirements.txt          # Python package requirements
├── db/
│   └── schema.sql            # PostgreSQL schema + pgvector table definitions
├── ingestion/                # Phase 1: Media ingestion, hashing, scene splitting
│   ├── scanner.py            # Untrusted folder traversal & SHA-256 sniffing
│   ├── media/                # FFmpeg audio, PySceneDetect scenes, PyAV frames
│   └── processors/           # Video, Audio, Image, and PDF processors
├── enrichment/               # Phase 2: Multimodal neural network feature extraction
│   ├── registry.py           # Unified model loader & device manager (MPS/CUDA)
│   ├── models/               # Faster-Whisper, AST, CLIP, YOLOv8s, PaddleOCR, Qwen2.5-VL
│   └── analyzers/            # Video segment, visual extractor, text fusion
├── graph/                    # Phases 3-7: Knowledge graph, sync, and reasoning
│   ├── pipeline.py           # Orchestration root for graph & timeline sync
│   ├── timeline_sync/        # Audio AST, CLIP visual, and identity alignment
│   ├── contradictions.py     # Claim extraction & pairwise contradiction judging
│   ├── clustering.py         # Face and voice clustering
│   ├── qa.py                 # Natural language graph QA engine
│   └── repository.py         # All graph SQL queries and pgvector operations
├── web/
│   ├── api/                  # FastAPI backend
│   │   ├── main.py           # REST routes (/api/collections, /api/timeline, /api/ask)
│   │   ├── search.py         # Vector similarity search engine
│   │   └── jobs.py           # Background job state & log router
│   └── ui/                   # React 19 SPA
│       ├── src/
│       │   ├── App.jsx       # Main layout & navigation
│       │   ├── components/   # SearchView, TimelineView, DetailModal, LibraryView, AskView
│       │   └── index.css     # Bespoke dark-mode design system
└── tests/                    # 312 automated unit & integration tests
```

---

## REST API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/collections` | List all investigation cases / collections |
| `POST` | `/api/collections` | Create a new case collection |
| `POST` | `/api/collections/{id}/upload` | Upload video/audio/image/PDF and trigger background enrichment |
| `GET` | `/api/collections/{id}/timeline` | Get unified timeline across all sources sorted by `case_time` |
| `GET` | `/api/collections/{id}/sync-status`| Get source-to-source alignment offsets, confidence, and anchor counts |
| `GET` | `/api/files/{id}/transcript` | Get complete stitched transcript with start/end timestamps |
| `GET` | `/api/nodes/{id}` | Get full node detail with transcript segments, OCR, caption, and detections |
| `GET` | `/api/search?q=...&mode=...` | Hybrid / Visual / Text semantic search against pgvector |
| `POST` | `/api/ask` | Natural language QA reasoning against the evidence graph |
| `GET` | `/api/jobs/{id}` | Poll background processing progress and live activity reasoning log |

---

## Testing & Quality Assurance

The codebase includes **312 automated tests** covering security containment, media extraction, model inference, graph operations, timeline synchronization, and API contracts.

```bash
# Run the entire test suite
venv/bin/python3 -m pytest tests/ -v

# Run timeline synchronization tests
venv/bin/python3 -m pytest tests/test_timeline_sync.py -v

# Run web API integration tests
venv/bin/python3 -m pytest tests/test_web_api.py -v
```

---

## License

Internal Hackathon Project — Designed for High-Assurance Multimodal Forensic Analysis.


