"""Ingestion pipeline orchestrator: scan case folder, register files, and
run per-type extraction (video/audio/pdf/image), writing evidence_node rows."""
from __future__ import annotations

import logging

import psycopg2.extras

from . import db
from .audio import process_audio
from .config import Config, load_config
from .image import process_image
from .pdf import process_pdf
from .scanner import FileObject, scan_case_folder
from .video import process_video

log = logging.getLogger("ingestion")


def _insert_evidence_node(conn, source_file_id, node_type, **kwargs):
    fields = ["source_file_id", "node_type"] + list(kwargs.keys())
    values = [source_file_id, node_type] + list(kwargs.values())
    placeholders = ", ".join(["%s"] * len(fields))
    with conn.cursor() as cur:
        cur.execute(
            f"INSERT INTO evidence_node ({', '.join(fields)}) VALUES ({placeholders}) RETURNING id",
            values,
        )
        return cur.fetchone()[0]


def process_file(conn, file_obj: FileObject, cfg: Config) -> int:
    """Dispatch to the right per-type processor and persist evidence_node rows.
    Returns the number of evidence_node rows created."""
    count = 0

    if file_obj.file_type == "video":
        result = process_video(file_obj, cfg)
        for seg in result["segments"]:
            _insert_evidence_node(
                conn, file_obj.id, "scene_segment",
                start_time=seg.start_time, end_time=seg.end_time,
                file_path=result["audio_path"],
                metadata=psycopg2.extras.Json({"frame_paths": seg.frame_paths}),
            )
            count += 1
        log.info(
            "video %s: audio=%s segments=%d",
            file_obj.file_name, result["audio_path"], len(result["segments"]),
        )

    elif file_obj.file_type == "audio":
        result = process_audio(file_obj, cfg)
        _insert_evidence_node(
            conn, file_obj.id, "audio_full",
            file_path=result["audio_path"],
        )
        count += 1
        log.info("audio %s: normalized=%s", file_obj.file_name, result["audio_path"])

    elif file_obj.file_type == "pdf":
        pages = process_pdf(file_obj, cfg)
        for page in pages:
            _insert_evidence_node(
                conn, file_obj.id, "page",
                page_number=page.page_number, text_content=page.text,
                file_path=page.image_path,
            )
            count += 1
        log.info("pdf %s: pages=%d", file_obj.file_name, len(pages))

    elif file_obj.file_type == "image":
        info = process_image(file_obj)
        _insert_evidence_node(
            conn, file_obj.id, "image",
            file_path=info.path,
            metadata=psycopg2.extras.Json(
                {"width": info.width, "height": info.height, "mode": info.mode}
            ),
        )
        count += 1
        log.info("image %s: %dx%d", file_obj.file_name, info.width, info.height)

    else:
        log.warning("no processor for file_type=%s (%s)", file_obj.file_type, file_obj.file_name)

    conn.commit()
    return count


def run(config_path: str = "config.yaml") -> dict:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(config_path)
    cfg.ensure_dirs()

    conn = db.connect(cfg)
    db.init_schema(conn)
    case_id = db.get_or_create_case(conn, cfg)

    files = scan_case_folder(conn, cfg, case_id)
    log.info("registered %d files for case %s", len(files), cfg.case_number)

    total_nodes = 0
    for f in files:
        total_nodes += process_file(conn, f, cfg)

    conn.close()
    return {"case_id": case_id, "files": len(files), "evidence_nodes": total_nodes}


if __name__ == "__main__":
    summary = run()
    print(summary)
