"""Background processing jobs, with a per-stage reasoning trail the UI can poll.

Ingesting and enriching a single video is minutes of model inference. Doing
that inside the upload request means a browser timeout on anything real, so
the upload returns a job id immediately and the work continues on a worker
thread.

Each job records not just "running/done" but *how it got there*: the stages it
passed through, the real log lines each stage emitted, and the findings that
came out (what was transcribed, what was read off the screen, what objects
were detected). That trail is the answer to "why does this file now match that
search" — without it the pipeline is a black box that either works or doesn't.

State is deliberately in-process: this is a local single-process tool, and a
job that dies with the server should not be resurrected as "running" by a
restart. Anything durable already lives in Postgres — the job record only
tracks the progress of work whose *results* are committed as they happen.
"""
from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

log = logging.getLogger(__name__)

#: Finished jobs are kept so the UI can show a result after the fact, but not
#: forever — a long session would otherwise accumulate them without bound.
_MAX_RETAINED = 30

#: Per-stage cap on captured log lines. Enrichment logs per node, so a large
#: upload would otherwise pin an unbounded transcript in memory.
_MAX_LOGS_PER_STAGE = 150

#: Loggers whose output counts as pipeline reasoning worth showing.
_PIPELINE_LOGGERS = ("ingestion", "enrichment", "graph")


@dataclass
class JobStage:
    """One phase of processing, with the evidence of what it did."""

    key: str
    label: str
    status: str = "running"  # running | ok | skipped | failed
    detail: str | None = None
    #: Real log lines emitted by the pipeline while this stage ran.
    logs: list[str] = field(default_factory=list)
    #: Concrete things the stage produced — a caption, OCR text, a detection.
    findings: list[str] = field(default_factory=list)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None

    def to_json(self) -> dict:
        return {
            "key": self.key,
            "label": self.label,
            "status": self.status,
            "detail": self.detail,
            "logs": self.logs,
            "findings": self.findings,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
        }


@dataclass
class Job:
    id: str
    file_name: str
    case_id: str
    status: str = "queued"  # queued | running | done | failed
    stage: str = "waiting"
    detail: str | None = None
    nodes_extracted: int = 0
    nodes_enriched: int = 0
    error: str | None = None
    stages: list[JobStage] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "file_name": self.file_name,
            "case_id": self.case_id,
            "status": self.status,
            "stage": self.stage,
            "detail": self.detail,
            "nodes_extracted": self.nodes_extracted,
            "nodes_enriched": self.nodes_enriched,
            "error": self.error,
            "stages": [s.to_json() for s in self.stages],
            "created_at": self.created_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
        }


class JobRegistry:
    """Thread-safe store of job state.

    Every mutation goes through the lock because the worker thread writes
    while the polling request reads; without it the UI can observe a job
    half-updated (a "done" status beside a stale stage).
    """

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    # -- lifecycle ----------------------------------------------------------

    def create(self, file_name: str, case_id: str) -> Job:
        job = Job(id=str(uuid.uuid4()), file_name=file_name, case_id=case_id)
        with self._lock:
            self._jobs[job.id] = job
            self._evict_finished()
        return job

    def update(self, job_id: str, **changes) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            for key, value in changes.items():
                setattr(job, key, value)
            if job.status in ("done", "failed") and job.finished_at is None:
                job.finished_at = datetime.now(timezone.utc)
                # A crash mid-stage would otherwise leave that stage spinning
                # in the UI forever.
                for stage in job.stages:
                    if stage.status == "running":
                        stage.status = "failed" if job.status == "failed" else "ok"
                        stage.finished_at = job.finished_at

    # -- stages -------------------------------------------------------------

    def begin_stage(self, job_id: str, key: str, label: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.stages.append(JobStage(key=key, label=label))
            job.stage = key

    def finish_stage(
        self, job_id: str, key: str, status: str = "ok",
        detail: str | None = None, findings: list[str] | None = None,
    ) -> None:
        with self._lock:
            stage = self._current_stage(job_id, key)
            if stage is None:
                return
            stage.status = status
            stage.detail = detail
            stage.finished_at = datetime.now(timezone.utc)
            if findings:
                stage.findings.extend(findings)

    def add_log(self, job_id: str, message: str) -> None:
        """Append a pipeline log line to whichever stage is currently open."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or not job.stages:
                return
            stage = job.stages[-1]
            if len(stage.logs) >= _MAX_LOGS_PER_STAGE:
                # Keep the newest lines; a truncation marker is more honest
                # than silently dropping the tail.
                stage.logs[0] = "… earlier lines trimmed …"
                del stage.logs[1]
            stage.logs.append(message)

    def _current_stage(self, job_id: str, key: str) -> JobStage | None:
        """Caller must hold the lock."""
        job = self._jobs.get(job_id)
        if job is None:
            return None
        for stage in reversed(job.stages):
            if stage.key == key:
                return stage
        return None

    # -- reads --------------------------------------------------------------

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def recent(self, limit: int = 20) -> list[Job]:
        with self._lock:
            ordered = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
            return ordered[:limit]

    def _evict_finished(self) -> None:
        """Caller must hold the lock. Drops the oldest finished jobs only —
        a running job is never evicted, however old it is."""
        if len(self._jobs) <= _MAX_RETAINED:
            return
        finished = sorted(
            (j for j in self._jobs.values() if j.status in ("done", "failed")),
            key=lambda j: j.finished_at or j.created_at,
        )
        for job in finished[: len(self._jobs) - _MAX_RETAINED]:
            del self._jobs[job.id]


class JobLogRouter(logging.Handler):
    """Captures pipeline log output and files it under the right job.

    The pipelines already narrate their own work ("video x.mp4: 12.0s, 5
    segments, 272 frames", "transcribed x.wav: 45 segment(s), language=en").
    Rather than making `ingestion` and `enrichment` aware of HTTP jobs — which
    would break the rule that a pipeline owns sequencing and nothing else —
    this handler listens to what they already say.

    Routing is by thread: each job runs on its own worker thread, so a record's
    originating thread identifies the job unambiguously even when several
    uploads process concurrently.
    """

    def __init__(self, registry: JobRegistry) -> None:
        super().__init__(level=logging.INFO)
        self._registry = registry
        self._threads: dict[int, str] = {}
        self._map_lock = threading.Lock()

    def bind(self, job_id: str) -> None:
        """Attribute this thread's pipeline logs to `job_id`."""
        with self._map_lock:
            self._threads[threading.get_ident()] = job_id

    def unbind(self) -> None:
        with self._map_lock:
            self._threads.pop(threading.get_ident(), None)

    def install(self) -> None:
        for name in _PIPELINE_LOGGERS:
            logger = logging.getLogger(name)
            if self not in logger.handlers:
                logger.addHandler(self)
            logger.setLevel(logging.INFO)

    def emit(self, record: logging.LogRecord) -> None:
        with self._map_lock:
            job_id = self._threads.get(record.thread)
        if job_id is None:
            return
        try:
            self._registry.add_log(job_id, record.getMessage())
        except Exception:  # noqa: BLE001 - logging must never break the run
            self.handleError(record)
