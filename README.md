# Evidence Ingestion Pipeline

## Setup (already done in this environment)
```bash
python3.13 -m venv venv
venv/bin/pip install -r requirements.txt

brew install postgresql@17 pgvector
brew services start postgresql@17
createdb -U $(whoami) -O postgres evidence_db
psql -U postgres -d evidence_db -f db/schema.sql   # also run automatically by the pipeline
```

## Run ingestion
```bash
venv/bin/python3 -m ingestion.pipeline
```
Reads `config.yaml`, connects to Postgres, applies `db/schema.sql`, scans
`sample_case/` (or whatever `paths.case_folder` points to), hashes + registers
every file in `source_file`, and runs per-type extraction:

- **video**: ffmpeg -> 16kHz mono WAV in `data/audio/`, PySceneDetect scene
  split (fixed 5s windows fallback), 1 frame/sec/segment saved to `data/frames/`
- **audio**: normalized to 16kHz mono WAV in `data/audio/`
- **pdf**: per-page text + rendered PNG in `data/pages/` (PyMuPDF)
- **image**: loaded, dimensions recorded

Each extracted unit becomes a row in `evidence_node` (vector columns for
embeddings are populated in later pipeline phases).

## Verify
```bash
psql -U postgres -d evidence_db -c "SELECT file_name, file_type FROM source_file;"
psql -U postgres -d evidence_db -c "SELECT node_type, count(*) FROM evidence_node GROUP BY 1;"
ls data/audio data/frames data/pages
```

## Layout
- `config.yaml` — case metadata, DB connection, file->type/metadata map
- `db/schema.sql` — case, source_file, evidence_node, entity, face_cluster,
  voice_cluster, identity, relationship, source_alignment
- `ingestion/` — scanner (hashing/registration), video/audio/pdf/image
  processors, pipeline orchestrator
- `sample_case/` — one video, audio, image, pdf for smoke testing
- `data/` — extracted audio/frames/page images (gitignored)
