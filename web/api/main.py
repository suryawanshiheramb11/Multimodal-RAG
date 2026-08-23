"""HTTP API for the media search UI.

The pipeline's phases are exposed here in the order a user actually meets
them: create a collection, drop files into it, watch them get processed, then
search across everything that came out. The heavy lifting all lives in the
`ingestion` / `enrichment` / `graph` packages — this layer only sequences
them, serves results, and keeps long work off the request thread.
"""
from __future__ import annotations

import logging
import shutil
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from psycopg2.extras import RealDictCursor

# Import the pipeline packages from the repo root.
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from enrichment.config import EnrichmentSettings  # noqa: E402
from enrichment.models.captioning import Captioner  # noqa: E402
from enrichment.pipeline import build_enrichment_pipeline  # noqa: E402
from graph.config import GraphSettings  # noqa: E402
from graph.pipeline import build_graph_pipeline  # noqa: E402
from graph.qa import answer_question  # noqa: E402
from graph.repository import GraphRepository  # noqa: E402
from ingestion.config import load_config  # noqa: E402
from ingestion.db import apply_schema, connect  # noqa: E402
from ingestion.errors import IngestionError  # noqa: E402
from ingestion.pipeline import build_pipeline  # noqa: E402

from .jobs import JobLogRouter, JobRegistry  # noqa: E402
from .media import MediaResolver  # noqa: E402
from .search import SemanticSearch  # noqa: E402

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

config = load_config()
ROOT = Path(__file__).resolve().parent.parent.parent
UPLOAD_ROOT = config.resolve(config.paths.data_dir) / "uploads"

jobs = JobRegistry()
# Listens to what the pipelines already log and files each line under the job
# whose worker thread emitted it, so the UI can show real reasoning rather
# than a spinner.
job_logs = JobLogRouter(jobs)
job_logs.install()
searcher = SemanticSearch()
# Media may live under the derived-artefact tree (frames, page renders,
# extracted audio, uploads) or in the original case folder. Nothing outside
# these two roots is ever served.
media = MediaResolver([config.resolve(config.paths.data_dir), config.case_folder])

# One Captioner shared by the /api/ask endpoint: like `searcher`, built once
# so the ollama reachability check is paid at most once per process instead
# of once per question.
qa_settings = GraphSettings()
qa_captioner = Captioner(
    qa_settings.entity_model, qa_settings.ollama_host, qa_settings.ollama_timeout_sec
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Apply the schema once, then warm the search encoders off-thread.

    Warming matters: the first CLIP load is several seconds, and paying it
    inside a user's first search makes the feature feel broken.
    """
    try:
        with connect(config.database) as conn:
            apply_schema(conn)
        log.info("schema applied")
    except IngestionError:
        log.exception("could not apply schema at startup")

    threading.Thread(target=searcher.warm, name="warm-encoders", daemon=True).start()
    yield


app = FastAPI(title="Semantic Media Search", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Collections
# ---------------------------------------------------------------------------

class CollectionCreate(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    title: str | None = None
    description: str | None = None


@app.get("/api/collections")
def list_collections():
    """Every collection, with live counts so the UI needs one round trip."""
    with connect(config.database) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT c.id, c.case_number, c.title, c.description, c.created_at,
                   count(DISTINCT f.id) AS file_count,
                   count(n.id)          AS node_count,
                   count(n.enriched_at) AS enriched_count
            FROM "case" c
            LEFT JOIN source_file f ON f.case_id = c.id
            LEFT JOIN evidence_node n ON n.source_file_id = f.id
            GROUP BY c.id
            ORDER BY c.created_at DESC
            """
        )
        return [dict(row) for row in cur.fetchall()]


@app.post("/api/collections", status_code=201)
def create_collection(payload: CollectionCreate):
    """Create a collection. Re-using an existing name returns that one."""
    with connect(config.database) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            INSERT INTO "case" (case_number, title, description)
            VALUES (%s, %s, %s)
            ON CONFLICT (case_number) DO UPDATE
                SET title = COALESCE(EXCLUDED.title, "case".title),
                    description = COALESCE(EXCLUDED.description, "case".description)
            RETURNING id, case_number, title, description, created_at
            """,
            (payload.name, payload.title, payload.description),
        )
        row = dict(cur.fetchone())
        conn.commit()
    return row


@app.delete("/api/collections/{collection_id}", status_code=204)
def delete_collection(collection_id: str):
    """Remove a collection and everything derived from it.

    Only the database rows go: the uploaded originals stay on disk, because
    deleting a view of the data should never destroy the data itself.
    """
    with connect(config.database) as conn, conn.cursor() as cur:
        cur.execute('DELETE FROM "case" WHERE id = %s::uuid', (collection_id,))
        deleted = cur.rowcount
        conn.commit()
    if not deleted:
        raise HTTPException(status_code=404, detail="Collection not found")


@app.get("/api/collections/{collection_id}/files")
def list_files(collection_id: str):
    with connect(config.database) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT f.id, f.file_name, f.file_type, f.size_bytes, f.duration_sec,
                   f.page_count, f.registered_at, f.type_mismatch,
                   count(n.id)          AS node_count,
                   count(n.enriched_at) AS enriched_count
            FROM source_file f
            LEFT JOIN evidence_node n ON n.source_file_id = f.id
            WHERE f.case_id = %s::uuid
            GROUP BY f.id
            ORDER BY f.registered_at DESC
            """,
            (collection_id,),
        )
        return [dict(row) for row in cur.fetchall()]


@app.get("/api/files/{file_id}/nodes")
def list_nodes(file_id: str):
    with connect(config.database) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT id, node_type, start_time, end_time, page_number, text_content,
                   metadata, created_at, enriched_at,
                   (clip_embedding IS NOT NULL) AS has_visual_vector,
                   (text_embedding IS NOT NULL) AS has_text_vector
            FROM evidence_node
            WHERE source_file_id = %s::uuid
            ORDER BY page_number NULLS LAST, start_time NULLS LAST, created_at
            """,
            (file_id,),
        )
        return [_node_json(dict(row)) for row in cur.fetchall()]


@app.get("/api/nodes/{node_id}")
def node_detail(node_id: str):
    """One node in full.

    Search results carry a trimmed snippet so the grid stays light; the detail
    view fetches the whole thing, which is where a long transcript or the full
    document text actually belongs.
    """
    with connect(config.database) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT n.id, n.node_type, n.start_time, n.end_time, n.page_number,
                   n.text_content, n.metadata, n.created_at, n.enriched_at,
                   n.source_file_id, f.file_name, f.file_type, f.case_id,
                   (n.clip_embedding IS NOT NULL) AS has_visual_vector,
                   (n.text_embedding IS NOT NULL) AS has_text_vector
            FROM evidence_node n
            JOIN source_file f ON f.id = n.source_file_id
            WHERE n.id = %s::uuid
            """,
            (node_id,),
        )
        row = cur.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Node not found")

    row = dict(row)
    metadata = row.get("metadata") or {}
    detail = _node_json(row)
    detail.update({
        "source_file_id": str(row["source_file_id"]),
        "file_name": row["file_name"],
        "file_type": row["file_type"],
        "case_id": str(row["case_id"]),
        "duration_sec": metadata.get("duration_sec"),
        "violence": (metadata.get("violence") or {}).get("score"),
        "frame_count": len(metadata.get("frames") or []),
    })
    return detail


def _node_json(row: dict) -> dict:
    """Shape a node row for the UI, dropping the bulky metadata blob.

    `metadata` carries every detection box and transcript segment; the list
    view needs only whether a thumbnail exists, so the blob is replaced by the
    handful of derived fields that actually get rendered.
    """
    metadata = row.get("metadata") or {}
    return {
        "id": str(row["id"]),
        "node_type": row["node_type"],
        "start_time": row["start_time"],
        "end_time": row["end_time"],
        "page_number": row["page_number"],
        "text_content": row.get("text_content"),
        "created_at": row["created_at"].isoformat() if row.get("created_at") else None,
        "enriched": row.get("enriched_at") is not None,
        "has_visual_vector": row.get("has_visual_vector", False),
        "has_text_vector": row.get("has_text_vector", False),
        "has_thumbnail": MediaResolver.thumbnail_path(row) is not None,
        "caption": metadata.get("caption"),
        "detections": sorted((metadata.get("detections") or {}).get("labels") or {}),
    }


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

@app.get("/api/search")
def search(
    q: str = Query(..., min_length=1, description="Natural-language query."),
    mode: str = Query("hybrid", pattern="^(hybrid|visual|text)$"),
    collection_id: str | None = Query(None),
    limit: int = Query(40, ge=1, le=100),
):
    """Rank media by meaning, not filename.

    'mountains' matches footage of mountains even when nothing in the file is
    named or captioned that way, because the query is compared against CLIP's
    view of each frame.
    """
    with connect(config.database) as conn:
        hits = searcher.search(conn, q, mode=mode, case_id=collection_id, limit=limit)

    return {
        "query": q,
        "mode": mode,
        "encoders": searcher.status,
        "count": len(hits),
        "results": [hit.to_json() for hit in hits],
    }


@app.get("/api/search/status")
def search_status():
    """Whether each search space is usable, so the UI can explain a gap."""
    with connect(config.database) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT count(*) AS total,
                   count(clip_embedding) AS visual_indexed,
                   count(text_embedding) AS text_indexed
            FROM evidence_node
            """
        )
        coverage = dict(cur.fetchone())
    return {"encoders": searcher.status, "coverage": coverage}


# ---------------------------------------------------------------------------
# Question answering
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    collection_id: str = Field(min_length=1)


@app.post("/api/ask")
def ask(payload: AskRequest):
    """Answer a question about one collection.

    Classification and retrieval (graph/qa.py) are plain code and SQL against
    the graph `build` already constructed — no model call. The LLM, if
    reachable, only turns the retrieved facts into one readable sentence at
    the very end; when it isn't reachable the endpoint still answers, just
    with a plainer templated sentence instead of a written one.
    """
    with connect(config.database) as conn:
        answer = answer_question(
            GraphRepository(conn), searcher.text_encoder, qa_captioner,
            payload.collection_id, payload.question, qa_settings,
        )

    return {
        "question": answer.question,
        "intent": answer.intent,
        "answer": answer.text,
        "used_llm": answer.used_llm,
        "facts": [
            {"label": f.label, "detail": f.detail, "node_id": f.node_id} for f in answer.facts
        ],
        "source_node_ids": answer.source_node_ids,
    }


# ---------------------------------------------------------------------------
# Media
# ---------------------------------------------------------------------------

@app.get("/api/nodes/{node_id}/thumbnail")
def node_thumbnail(node_id: str):
    with connect(config.database) as conn:
        resolved = media.node_thumbnail(conn, node_id)
    if resolved is None:
        raise HTTPException(status_code=404, detail="No thumbnail for this node")
    return FileResponse(resolved.path, media_type=resolved.content_type)


@app.get("/api/nodes/{node_id}/media")
def node_media(node_id: str):
    with connect(config.database) as conn:
        resolved = media.node_media(conn, node_id)
    if resolved is None:
        raise HTTPException(status_code=404, detail="No media for this node")
    return FileResponse(resolved.path, media_type=resolved.content_type)


@app.get("/api/files/{file_id}/media")
def file_media(file_id: str):
    """The original file, so a hit can be played at its timestamp."""
    with connect(config.database) as conn:
        resolved = media.source_file(conn, file_id)
    if resolved is None:
        raise HTTPException(status_code=404, detail="Original file is unavailable")
    return FileResponse(resolved.path, media_type=resolved.content_type)


# ---------------------------------------------------------------------------
# Upload + processing
# ---------------------------------------------------------------------------

@app.post("/api/collections/{collection_id}/upload", status_code=202)
def upload(collection_id: str, file: UploadFile = File(...)):
    """Store an upload and process it on a worker thread.

    Returns 202 with a job id rather than blocking: extraction plus model
    inference on one video runs for minutes, well past any browser timeout.
    """
    with connect(config.database) as conn, conn.cursor() as cur:
        cur.execute(
            'SELECT case_number, title, description FROM "case" WHERE id = %s::uuid',
            (collection_id,),
        )
        row = cur.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Collection not found")
    case_number, case_title, case_description = row

    safe_name = Path(file.filename or "upload").name  # strip any client-supplied path
    if not safe_name or safe_name in (".", ".."):
        raise HTTPException(status_code=400, detail="Invalid file name")

    upload_dir = UPLOAD_ROOT / case_number
    upload_dir.mkdir(parents=True, exist_ok=True)
    destination = upload_dir / safe_name
    try:
        with destination.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    finally:
        file.file.close()

    if destination.stat().st_size == 0:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    job = jobs.create(file_name=safe_name, case_id=collection_id)
    threading.Thread(
        target=_process_upload,
        args=(job.id, collection_id, case_number, case_title, case_description, upload_dir),
        name=f"process-{job.id[:8]}",
        daemon=True,
    ).start()
    return {"job_id": job.id, "status": job.status, "file_name": safe_name}


def _process_upload(
    job_id: str, collection_id: str, case_number: str,
    case_title: str | None, case_description: str | None, upload_dir: Path,
) -> None:
    """Ingest then enrich, narrating each stage into the job registry.

    The whole collection folder is rescanned rather than just the new file —
    that is what the scanner works on — but files already ingested and
    unchanged are left untouched by the pipeline's own hash check, so the cost
    is one hash per existing file and no enrichment is lost.
    """
    jobs.update(job_id, status="running", stage="extract")
    job_logs.bind(job_id)  # route this thread's pipeline logs into the job
    try:
        upload_config = load_config()
        upload_config.case.case_number = case_number
        if case_title:
            upload_config.case.title = case_title
        if case_description:
            upload_config.case.description = case_description
        upload_config.paths.case_folder = upload_dir

        with connect(upload_config.database) as conn:
            # -- stage 1: split the file into evidence nodes ----------------
            jobs.begin_stage(job_id, "extract", "Extracting media")
            report = build_pipeline(upload_config, conn).run()
            extracted = sum(f.node_count for f in report.ok)
            unchanged = sum(f.node_count for f in report.unchanged)

            failures = [f for f in report.files if f.status == "failed"]
            if failures and not report.ok and not report.unchanged:
                jobs.finish_stage(
                    job_id, "extract", "failed", failures[0].detail or "extraction failed"
                )
                raise RuntimeError(failures[0].detail or "extraction failed")

            jobs.finish_stage(
                job_id, "extract", "ok",
                detail=(
                    f"{extracted} segment(s) extracted"
                    + (f", {unchanged} already present" if unchanged else "")
                ),
                findings=[
                    f"{f.file_name}: {f.node_count} segment(s)"
                    for f in report.ok + report.unchanged
                ],
            )
            jobs.update(job_id, nodes_extracted=extracted)

            # -- stage 2: run the models over each node ---------------------
            jobs.begin_stage(job_id, "analyze", "Running models")
            enrichment = build_enrichment_pipeline(conn, EnrichmentSettings())
            enrichment_report = enrichment.run(collection_id, only_pending=True)
            enriched = len(enrichment_report.ok)
            jobs.finish_stage(
                job_id, "analyze", "ok" if enriched or not enrichment_report.failed else "failed",
                detail=(
                    f"{enriched} node(s) analyzed"
                    + (f", {len(enrichment_report.failed)} failed" if enrichment_report.failed else "")
                ),
                findings=_model_findings(conn, collection_id),
            )

            # -- stage 3: confirm what became searchable --------------------
            jobs.begin_stage(job_id, "index", "Indexing for search")
            coverage = _index_coverage(conn, collection_id)
            jobs.finish_stage(
                job_id, "index", "ok",
                detail=(
                    f"{coverage['visual']} visual + {coverage['text']} text vector(s) "
                    f"across {coverage['total']} node(s)"
                ),
                findings=[
                    "Visual vectors let a typed phrase match what a frame looks like",
                    "Text vectors match transcripts, captions, OCR and document text",
                ],
            )

            # -- stage 4: entities, relationships and identities -------------
            # This is what /api/ask reads from: without it, questions have
            # nothing to retrieve. Every step inside build_graph_pipeline is
            # already independently isolated (one LLM step failing does not
            # abort the others), so this is safe to always attempt — a case
            # with ollama unreachable just gets fewer of the graph steps.
            jobs.begin_stage(job_id, "graph", "Building the evidence graph")
            graph_report = build_graph_pipeline(conn, GraphSettings()).run(collection_id)
            graph_failures = [
                name for name, status in graph_report.step_status.items()
                if status.startswith("failed")
            ]
            jobs.finish_stage(
                job_id, "graph", "ok",
                detail=(
                    f"{graph_report.entities} entit(y/ies), {graph_report.contradicts} "
                    f"contradiction(s), {graph_report.identities} identity(ies)"
                    + (f" — {len(graph_failures)} step(s) skipped/failed" if graph_failures else "")
                ),
                findings=[
                    f"{graph_report.mentions} entity mention(s) across the case",
                    f"{graph_report.timeline_events} timeline event(s) grouped",
                    *(
                        [f"{graph_report.contradicts} contradiction(s) flagged for review"]
                        if graph_report.contradicts else []
                    ),
                    *(
                        [f"{graph_report.identities} identit(y/ies) fused from face + voice"]
                        if graph_report.identities else []
                    ),
                ],
            )

        jobs.update(
            job_id, status="done", stage="ready", nodes_enriched=enriched,
            detail=(
                f"{extracted} segment(s) extracted, {enriched} analyzed and searchable"
                if extracted or enriched
                else "already processed; nothing new to do"
            ),
        )
        log.info("job %s finished: %d extracted, %d enriched", job_id[:8], extracted, enriched)
    except Exception as exc:  # noqa: BLE001 - a failed upload must not kill the worker
        log.exception("job %s failed", job_id[:8])
        jobs.update(job_id, status="failed", stage="failed", error=str(exc))
    finally:
        job_logs.unbind()


def _model_findings(conn, collection_id: str, limit: int = 8) -> list[str]:
    """What the models actually concluded, as short human-readable lines.

    This is the "why will this match a search later" evidence: the caption a
    vision model wrote, the words OCR read off a frame, the objects a detector
    boxed. Read back from the stored metadata rather than threaded out of the
    enrichment run, so it reflects what was really persisted.
    """
    findings: list[str] = []
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT n.node_type, n.metadata
            FROM evidence_node n
            JOIN source_file f ON f.id = n.source_file_id
            WHERE f.case_id = %s::uuid AND n.enriched_at IS NOT NULL
            ORDER BY n.enriched_at DESC
            LIMIT %s
            """,
            (collection_id, limit),
        )
        rows = cur.fetchall()

    for row in rows:
        metadata = row["metadata"] or {}
        caption = metadata.get("caption")
        if caption:
            findings.append(f"Described: “{_clip_text(caption, 110)}”")

        ocr_text = (metadata.get("ocr") or {}).get("text")
        if ocr_text and ocr_text.strip():
            findings.append(f"Read on screen: “{_clip_text(ocr_text, 90)}”")

        labels = sorted((metadata.get("detections") or {}).get("labels") or {})
        if labels:
            findings.append(f"Objects detected: {', '.join(labels[:6])}")

        transcript = (metadata.get("transcript") or {}).get("text")
        if transcript and transcript.strip():
            findings.append(f"Transcribed: “{_clip_text(transcript, 110)}”")

        events = metadata.get("audio_events") or []
        if events:
            top = ", ".join(
                f"{e.get('label')} ({e.get('probability'):.2f})"
                for e in events[:3]
                if e.get("label") and e.get("probability") is not None
            )
            if top:
                findings.append(f"Audio events: {top}")

    # De-duplicate while keeping order: several segments of one video often
    # yield the same caption, and repeating it teaches the reader nothing.
    seen: set[str] = set()
    unique = [f for f in findings if not (f in seen or seen.add(f))]
    return unique[:12]


def _clip_text(text: str, limit: int) -> str:
    cleaned = " ".join(text.split())
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 1].rstrip() + "…"


def _index_coverage(conn, collection_id: str) -> dict:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT count(*) AS total,
                   count(n.clip_embedding) AS visual,
                   count(n.text_embedding) AS text
            FROM evidence_node n
            JOIN source_file f ON f.id = n.source_file_id
            WHERE f.case_id = %s::uuid
            """,
            (collection_id,),
        )
        return dict(cur.fetchone())


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.to_json()


@app.get("/api/jobs")
def recent_jobs():
    return [job.to_json() for job in jobs.recent()]


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@app.get("/api/stats")
def stats():
    with connect(config.database) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT
                (SELECT count(*) FROM "case")          AS collections,
                (SELECT count(*) FROM source_file)     AS files,
                (SELECT count(*) FROM evidence_node)   AS nodes,
                (SELECT count(*) FROM evidence_node
                  WHERE clip_embedding IS NOT NULL OR text_embedding IS NOT NULL)
                                                       AS searchable
            """
        )
        return dict(cur.fetchone())
