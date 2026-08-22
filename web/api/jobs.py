"""Background processing jobs, with progress the UI can poll.

Ingesting and enriching a single video is minutes of model inference. Doing
that inside the upload request means a browser timeout on anything real, so
the upload returns a job id immediately and the work continues on a worker
thread.

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
_MAX_RETAINED = 50


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

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def active(self) -> list[Job]:
        with self._lock:
            return [j for j in self._jobs.values() if j.status in ("queued", "running")]

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
