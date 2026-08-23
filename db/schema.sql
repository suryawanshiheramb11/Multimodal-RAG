-- Multi-modal evidence graph schema
-- Vector dims, each matching the model that fills it:
--   text_embedding  384  sentence-transformers all-MiniLM-L6-v2
--   clip_embedding  512  CLIP ViT-B/32 (joint image/text space)
--   audio_embedding 768  AST pooled hidden state (see the phase 2 migration
--                        below; the original 1024 did not match any model)
--   voice_segment   256  pyannote WeSpeaker ResNet34 speaker embedding (see
--                        the phase 7 migration below; voice_cluster's 1024
--                        below has the same "never verified" problem
--                        audio_embedding had, fixed the same way)

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS "case" (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    case_number     TEXT UNIQUE NOT NULL,
    title           TEXT,
    description     TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS source_file (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    case_id         UUID NOT NULL REFERENCES "case"(id) ON DELETE CASCADE,
    file_path       TEXT NOT NULL,
    file_name       TEXT NOT NULL,
    file_type       TEXT NOT NULL CHECK (file_type IN ('video', 'audio', 'image', 'pdf', 'doc')),
    sha256          TEXT NOT NULL,
    hash_algorithm  TEXT NOT NULL DEFAULT 'sha256',
    size_bytes      BIGINT,
    duration_sec    DOUBLE PRECISION,
    page_count      INTEGER,
    author          TEXT,
    created_date    TIMESTAMPTZ,
    -- Chain-of-custody fields: what the config claimed the file was, what its
    -- bytes actually say it is, and whether those two disagree. A mismatch is
    -- a finding in its own right, so it is stored rather than only logged.
    declared_type   TEXT,
    detected_mime   TEXT,
    type_mismatch   BOOLEAN NOT NULL DEFAULT FALSE,
    metadata        JSONB DEFAULT '{}'::jsonb,
    registered_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (case_id, sha256)
);

-- Idempotent upgrade path for databases created before these columns existed.
ALTER TABLE source_file ADD COLUMN IF NOT EXISTS hash_algorithm TEXT NOT NULL DEFAULT 'sha256';
ALTER TABLE source_file ADD COLUMN IF NOT EXISTS declared_type TEXT;
ALTER TABLE source_file ADD COLUMN IF NOT EXISTS detected_mime TEXT;
ALTER TABLE source_file ADD COLUMN IF NOT EXISTS type_mismatch BOOLEAN NOT NULL DEFAULT FALSE;

CREATE INDEX IF NOT EXISTS idx_source_file_case ON source_file(case_id);
CREATE INDEX IF NOT EXISTS idx_source_file_type ON source_file(file_type);
CREATE INDEX IF NOT EXISTS idx_source_file_sha256 ON source_file(sha256);
-- Supports the "has this path changed since last ingest?" tamper check.
CREATE INDEX IF NOT EXISTS idx_source_file_case_path ON source_file(case_id, file_path);

-- A generic node of extracted evidence: a video segment, a page, an image,
-- a transcript chunk, an OCR block, a detected object, etc.
CREATE TABLE IF NOT EXISTS evidence_node (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_file_id  UUID NOT NULL REFERENCES source_file(id) ON DELETE CASCADE,
    node_type       TEXT NOT NULL, -- e.g. 'scene_segment','frame','page','transcript_chunk','ocr_block','detection'
    start_time      DOUBLE PRECISION, -- seconds, for audio/video
    end_time        DOUBLE PRECISION,
    page_number     INTEGER,        -- for pdf/doc
    text_content    TEXT,
    file_path       TEXT,           -- path to extracted frame/page image/audio clip
    text_embedding  VECTOR(384),
    clip_embedding  VECTOR(512),
    audio_embedding VECTOR(1024),
    metadata        JSONB DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_evidence_node_source ON evidence_node(source_file_id);
CREATE INDEX IF NOT EXISTS idx_evidence_node_type ON evidence_node(node_type);
CREATE INDEX IF NOT EXISTS idx_evidence_node_text_embedding ON evidence_node
    USING hnsw (text_embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_evidence_node_clip_embedding ON evidence_node
    USING hnsw (clip_embedding vector_cosine_ops);

-- ============================================================================
-- Phase 6: Multi-source timeline synchronization
-- ============================================================================

-- Unified timeline: case_time = start_time + offset_from_reference for each source.
-- Enables questions like "what happened before the arrest?" across multiple sources.
ALTER TABLE evidence_node ADD COLUMN IF NOT EXISTS case_time DOUBLE PRECISION;
CREATE INDEX IF NOT EXISTS idx_evidence_node_case_time ON evidence_node(case_time);

-- Source-to-source offsets: offset_seconds = time_in_source_b - time_in_source_a
-- (negative if source_b started before source_a). Includes method name (audio/visual/identity)
-- and confidence (fraction of anchor points that agreed within ±1 second).
CREATE TABLE IF NOT EXISTS source_offset (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    case_id         UUID NOT NULL REFERENCES "case"(id) ON DELETE CASCADE,
    source_a_id     UUID NOT NULL REFERENCES source_file(id) ON DELETE CASCADE,
    source_b_id     UUID NOT NULL REFERENCES source_file(id) ON DELETE CASCADE,
    offset_seconds  DOUBLE PRECISION NOT NULL,
    confidence      DOUBLE PRECISION, -- 0.0-1.0: fraction of anchors within ±1 second
    method          TEXT NOT NULL, -- 'audio_fingerprinting','visual_matching','identity_cooccurrence'
    anchor_count    INTEGER DEFAULT 0,
    metadata        JSONB DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (case_id, source_a_id, source_b_id)
);

CREATE INDEX IF NOT EXISTS idx_source_offset_case ON source_offset(case_id);
CREATE INDEX IF NOT EXISTS idx_source_offset_pair ON source_offset(source_a_id, source_b_id);

-- Named entities extracted from text (NER) or objects (vision)
CREATE TABLE IF NOT EXISTS entity (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    evidence_node_id UUID NOT NULL REFERENCES evidence_node(id) ON DELETE CASCADE,
    entity_type     TEXT NOT NULL, -- 'PERSON','ORG','LOCATION','OBJECT','DATE', etc.
    value           TEXT NOT NULL,
    confidence      DOUBLE PRECISION,
    bbox            JSONB, -- {x,y,w,h} for vision detections
    metadata        JSONB DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- No indexes declared here: entity is reshaped into a canonical table by the
-- phase 3 migration below (columns renamed/dropped), which declares the
-- indexes that match its final shape. An index tied to a column this file
-- later drops or renames would break re-applying this script against an
-- already-migrated database — CREATE INDEX IF NOT EXISTS only skips by name,
-- not by whether the referenced column still exists.

-- Clusters of detected faces across evidence, prior to identity resolution
CREATE TABLE IF NOT EXISTS face_cluster (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    case_id         UUID NOT NULL REFERENCES "case"(id) ON DELETE CASCADE,
    representative_embedding VECTOR(512),
    face_count      INTEGER DEFAULT 0,
    identity_id     UUID, -- FK added after identity table exists
    metadata        JSONB DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Clusters of detected voices across evidence, prior to identity resolution
CREATE TABLE IF NOT EXISTS voice_cluster (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    case_id         UUID NOT NULL REFERENCES "case"(id) ON DELETE CASCADE,
    representative_embedding VECTOR(1024),
    segment_count   INTEGER DEFAULT 0,
    identity_id     UUID,
    metadata        JSONB DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Resolved real-world identity, merging one or more face/voice clusters
CREATE TABLE IF NOT EXISTS identity (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    case_id         UUID NOT NULL REFERENCES "case"(id) ON DELETE CASCADE,
    display_name    TEXT,
    aliases         TEXT[],
    notes           TEXT,
    metadata        JSONB DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE face_cluster
    DROP CONSTRAINT IF EXISTS fk_face_cluster_identity,
    ADD CONSTRAINT fk_face_cluster_identity FOREIGN KEY (identity_id)
        REFERENCES identity(id) ON DELETE SET NULL;

ALTER TABLE voice_cluster
    DROP CONSTRAINT IF EXISTS fk_voice_cluster_identity,
    ADD CONSTRAINT fk_voice_cluster_identity FOREIGN KEY (identity_id)
        REFERENCES identity(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_face_cluster_case ON face_cluster(case_id);
CREATE INDEX IF NOT EXISTS idx_voice_cluster_case ON voice_cluster(case_id);
CREATE INDEX IF NOT EXISTS idx_identity_case ON identity(case_id);

-- Relationships between identities/entities (e.g. co-occurrence, communication)
CREATE TABLE IF NOT EXISTS relationship (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    case_id         UUID NOT NULL REFERENCES "case"(id) ON DELETE CASCADE,
    subject_identity_id UUID REFERENCES identity(id) ON DELETE CASCADE,
    object_identity_id  UUID REFERENCES identity(id) ON DELETE CASCADE,
    relationship_type   TEXT NOT NULL, -- 'co_occurs_with','communicates_with','mentions', etc.
    evidence_node_id     UUID REFERENCES evidence_node(id) ON DELETE SET NULL,
    confidence      DOUBLE PRECISION,
    metadata        JSONB DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_relationship_case ON relationship(case_id);
CREATE INDEX IF NOT EXISTS idx_relationship_subject ON relationship(subject_identity_id);
CREATE INDEX IF NOT EXISTS idx_relationship_object ON relationship(object_identity_id);

-- Cross-source alignment: links evidence nodes across different source files
-- that refer to the same real-world event/moment (e.g. video frame <-> pdf page mention)
CREATE TABLE IF NOT EXISTS source_alignment (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    case_id         UUID NOT NULL REFERENCES "case"(id) ON DELETE CASCADE,
    node_a_id       UUID NOT NULL REFERENCES evidence_node(id) ON DELETE CASCADE,
    node_b_id       UUID NOT NULL REFERENCES evidence_node(id) ON DELETE CASCADE,
    alignment_type  TEXT NOT NULL, -- 'temporal','semantic','identity','manual'
    score           DOUBLE PRECISION,
    metadata        JSONB DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_source_alignment_case ON source_alignment(case_id);
CREATE INDEX IF NOT EXISTS idx_source_alignment_a ON source_alignment(node_a_id);
CREATE INDEX IF NOT EXISTS idx_source_alignment_b ON source_alignment(node_b_id);

-- ============================================================================
-- Phase 2: feature extraction
-- ============================================================================

-- Enrichment bookkeeping on evidence_node. `enriched_at` lets a re-run resume
-- instead of repeating hours of model inference; `enrichment_error` keeps a
-- failed node visible rather than silently empty.
ALTER TABLE evidence_node ADD COLUMN IF NOT EXISTS enriched_at TIMESTAMPTZ;
ALTER TABLE evidence_node ADD COLUMN IF NOT EXISTS enrichment_error TEXT;

CREATE INDEX IF NOT EXISTS idx_evidence_node_enriched ON evidence_node(enriched_at);

-- AST (MIT/ast-finetuned-audioset) pools to 768 dimensions, not the 1024 the
-- original outline assumed. A vector column has to match the model that fills
-- it, so the column follows the model.
-- pgvector keeps a column's dimension in atttypmod, which information_schema
-- does not expose, so the current width is read via format_type.
DO $$
DECLARE
    current_type TEXT;
BEGIN
    SELECT format_type(a.atttypid, a.atttypmod) INTO current_type
    FROM pg_attribute a
    WHERE a.attrelid = 'evidence_node'::regclass
      AND a.attname = 'audio_embedding'
      AND NOT a.attisdropped;

    IF current_type IS NOT NULL AND current_type <> 'vector(768)' THEN
        -- Nothing ever populated this column at the old width, so dropping and
        -- re-adding loses no data.
        ALTER TABLE evidence_node DROP COLUMN audio_embedding;
        ALTER TABLE evidence_node ADD COLUMN audio_embedding VECTOR(768);
        RAISE NOTICE 'audio_embedding migrated from % to vector(768)', current_type;
    END IF;
END
$$;

-- One row per enrichment run: which models were actually live. Without this a
-- NULL embedding is indistinguishable from a model that was missing that day.
CREATE TABLE IF NOT EXISTS enrichment_run (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    case_id             UUID NOT NULL REFERENCES "case"(id) ON DELETE CASCADE,
    model_availability  JSONB NOT NULL DEFAULT '{}'::jsonb,
    settings            JSONB NOT NULL DEFAULT '{}'::jsonb,
    started_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_enrichment_run_case ON enrichment_run(case_id);

-- ============================================================================
-- Phase 3: structured storage and graph construction
-- ============================================================================

-- `entity` was originally shaped as one row per mention (evidence_node_id was
-- NOT NULL), which cannot represent "the same knife mentioned in 5 nodes" as
-- one entity. It becomes canonical here: one row per unique (case, type,
-- normalized name); the `mention` table below carries the per-node edges that
-- evidence_node_id used to.
ALTER TABLE entity ADD COLUMN IF NOT EXISTS case_id UUID REFERENCES "case"(id) ON DELETE CASCADE;
-- DROP COLUMN removes the NOT NULL constraint along with the column, so no
-- separate DROP NOT NULL step is needed.
ALTER TABLE entity DROP COLUMN IF EXISTS evidence_node_id;
ALTER TABLE entity DROP COLUMN IF EXISTS bbox;

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns
               WHERE table_name = 'entity' AND column_name = 'value') THEN
        ALTER TABLE entity RENAME COLUMN value TO canonical_name;
    END IF;
END
$$;

ALTER TABLE entity ADD COLUMN IF NOT EXISTS canonical_name TEXT;
ALTER TABLE entity ALTER COLUMN canonical_name SET NOT NULL;
-- Matching key: trimmed/lowercased canonical_name. A plain column (rather than
-- an expression index) keeps ON CONFLICT a straightforward column list.
ALTER TABLE entity ADD COLUMN IF NOT EXISTS normalized_name TEXT;
ALTER TABLE entity ADD COLUMN IF NOT EXISTS embedding VECTOR(384);
ALTER TABLE entity ADD COLUMN IF NOT EXISTS mention_count INTEGER NOT NULL DEFAULT 0;
ALTER TABLE entity ALTER COLUMN case_id SET NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS uq_entity_case_type_normalized
    ON entity(case_id, entity_type, normalized_name);
CREATE INDEX IF NOT EXISTS idx_entity_case ON entity(case_id);
CREATE INDEX IF NOT EXISTS idx_entity_type_canonical ON entity(entity_type, canonical_name);
CREATE INDEX IF NOT EXISTS idx_entity_embedding ON entity
    USING hnsw (embedding vector_cosine_ops);

-- MENTIONS edges: which nodes reference which canonical entity, and how the
-- mention was found (LLM text extraction vs. an object detector).
CREATE TABLE IF NOT EXISTS mention (
    id                UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    entity_id         UUID NOT NULL REFERENCES entity(id) ON DELETE CASCADE,
    evidence_node_id  UUID NOT NULL REFERENCES evidence_node(id) ON DELETE CASCADE,
    mention_text      TEXT,
    source            TEXT NOT NULL DEFAULT 'llm_extraction',
    confidence        DOUBLE PRECISION,
    metadata          JSONB DEFAULT '{}'::jsonb,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (entity_id, evidence_node_id)
);

CREATE INDEX IF NOT EXISTS idx_mention_entity ON mention(entity_id);
CREATE INDEX IF NOT EXISTS idx_mention_node ON mention(evidence_node_id);

-- ALIGNS_WITH and SIMILAR_TO edges both connect two evidence nodes and already
-- fit source_alignment's shape (node_a_id, node_b_id, alignment_type, score);
-- `relationship` stays reserved for identity-level edges, which face
-- clustering will populate in a later phase.
CREATE INDEX IF NOT EXISTS idx_source_alignment_type ON source_alignment(alignment_type);

-- One row per detected face. `case_id` is denormalized from the node's source
-- file so per-case clustering does not need a join across every detection.
CREATE TABLE IF NOT EXISTS face_detection (
    id                UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    case_id           UUID NOT NULL REFERENCES "case"(id) ON DELETE CASCADE,
    evidence_node_id  UUID NOT NULL REFERENCES evidence_node(id) ON DELETE CASCADE,
    frame_path        TEXT NOT NULL,
    bbox              JSONB NOT NULL,
    confidence        DOUBLE PRECISION,
    embedding         VECTOR(512) NOT NULL,
    face_cluster_id   UUID REFERENCES face_cluster(id) ON DELETE SET NULL,
    metadata          JSONB DEFAULT '{}'::jsonb,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_face_detection_case ON face_detection(case_id);
CREATE INDEX IF NOT EXISTS idx_face_detection_node ON face_detection(evidence_node_id);
CREATE INDEX IF NOT EXISTS idx_face_detection_cluster ON face_detection(face_cluster_id);
CREATE INDEX IF NOT EXISTS idx_face_detection_embedding ON face_detection
    USING hnsw (embedding vector_cosine_ops);

-- ============================================================================
-- Phase 4: cross-modal linking (semantic + temporal refinement)
-- ============================================================================

-- source_alignment.alignment_type now also carries:
--   'REFERENCES' — a pdf page's rendered image and a video segment's
--                  representative frame are the same visual (CLIP similarity)
--   'DESCRIBES'  — a transcript's CLIP text embedding matches a frame's CLIP
--                  image embedding (spoken description <-> visual evidence)
-- Both reuse the existing node_a_id/node_b_id/score/metadata shape, so no
-- column changes are needed — only the index below, for the new query pattern
-- of "give me every edge touching this node".
CREATE INDEX IF NOT EXISTS idx_source_alignment_node_a ON source_alignment(node_a_id);
CREATE INDEX IF NOT EXISTS idx_source_alignment_node_b ON source_alignment(node_b_id);

-- A timeline event groups evidence nodes (possibly from different source
-- files) that occurred within a short time window and share an entity or
-- high text similarity. `node_ids` is denormalized for cheap reads (e.g. a
-- graph UI listing an event's evidence in one query); `timeline_event_link`
-- below is the authoritative SAME_EVENT edge and is what FK integrity and
-- per-node traversal actually rely on.
CREATE TABLE IF NOT EXISTS timeline_event (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    case_id         UUID NOT NULL REFERENCES "case"(id) ON DELETE CASCADE,
    description     TEXT,
    start_time      DOUBLE PRECISION,
    end_time        DOUBLE PRECISION,
    node_ids        JSONB NOT NULL DEFAULT '[]'::jsonb,
    metadata        JSONB DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_timeline_event_case ON timeline_event(case_id);

-- SAME_EVENT edges: which evidence nodes belong to which timeline event. A
-- node can only belong to a given event once, mirroring `mention`'s
-- one-row-per-(entity, node) shape for MENTIONS.
CREATE TABLE IF NOT EXISTS timeline_event_link (
    id                UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timeline_event_id UUID NOT NULL REFERENCES timeline_event(id) ON DELETE CASCADE,
    evidence_node_id  UUID NOT NULL REFERENCES evidence_node(id) ON DELETE CASCADE,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (timeline_event_id, evidence_node_id)
);

CREATE INDEX IF NOT EXISTS idx_timeline_event_link_event ON timeline_event_link(timeline_event_id);
CREATE INDEX IF NOT EXISTS idx_timeline_event_link_node ON timeline_event_link(evidence_node_id);

-- ============================================================================
-- Phase 5: contradiction and corroboration detection
-- ============================================================================

-- The one-sentence factual claim the LLM distilled from a node's text.
-- `claim_extracted_at` marks that extraction was *attempted*, so a re-run
-- resumes instead of re-prompting for every node and a node with no factual
-- claim is not retried forever (same role `enriched_at` plays for phase 2).
ALTER TABLE evidence_node ADD COLUMN IF NOT EXISTS claim TEXT;
ALTER TABLE evidence_node ADD COLUMN IF NOT EXISTS claim_extracted_at TIMESTAMPTZ;

CREATE INDEX IF NOT EXISTS idx_evidence_node_claim_extracted ON evidence_node(claim_extracted_at);

-- CONTRADICTS / CORROBORATES relate two evidence *nodes* — "the statement in
-- this PDF page conflicts with what this video segment shows" — not two
-- identities, which is all `relationship` could express before. Both pairs of
-- columns stay nullable: an identity edge fills the identity columns, a claim
-- edge fills the node columns.
ALTER TABLE relationship ADD COLUMN IF NOT EXISTS subject_node_id UUID
    REFERENCES evidence_node(id) ON DELETE CASCADE;
ALTER TABLE relationship ADD COLUMN IF NOT EXISTS object_node_id UUID
    REFERENCES evidence_node(id) ON DELETE CASCADE;
-- Why the LLM ruled the way it did. Stored rather than logged: a flagged
-- disagreement a reviewer cannot interrogate is not usable as evidence.
ALTER TABLE relationship ADD COLUMN IF NOT EXISTS explanation TEXT;

-- Partial index: identity-level rows leave both node columns NULL, and a
-- plain unique index over nullable columns would neither constrain them nor
-- (in Postgres) collide, so the constraint is scoped to node-level rows only.
CREATE UNIQUE INDEX IF NOT EXISTS uq_relationship_node_pair
    ON relationship(case_id, subject_node_id, object_node_id, relationship_type)
    WHERE subject_node_id IS NOT NULL AND object_node_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_relationship_subject_node ON relationship(subject_node_id);
CREATE INDEX IF NOT EXISTS idx_relationship_object_node ON relationship(object_node_id);
CREATE INDEX IF NOT EXISTS idx_relationship_type ON relationship(relationship_type);

-- ============================================================================
-- Phase 7: cross-modal identity fusion (face + voice)
-- ============================================================================

-- voice_cluster.representative_embedding was declared VECTOR(1024) on day one
-- as a placeholder ("reserved for speaker embeddings in a later phase") and,
-- like the original audio_embedding guess, was never checked against a real
-- model. pyannote's community diarization pipeline embeds with WeSpeaker
-- ResNet34, which is 256-dimensional — empirically confirmed by loading it,
-- not assumed. Same fix as the phase-2 audio_embedding migration: drop and
-- re-add, safe because nothing has populated this column at the old width.
DO $$
DECLARE
    current_type TEXT;
BEGIN
    SELECT format_type(a.atttypid, a.atttypmod) INTO current_type
    FROM pg_attribute a
    WHERE a.attrelid = 'voice_cluster'::regclass
      AND a.attname = 'representative_embedding'
      AND NOT a.attisdropped;

    IF current_type IS NOT NULL AND current_type <> 'vector(256)' THEN
        ALTER TABLE voice_cluster DROP COLUMN representative_embedding;
        ALTER TABLE voice_cluster ADD COLUMN representative_embedding VECTOR(256);
        RAISE NOTICE 'voice_cluster.representative_embedding migrated from % to vector(256)',
            current_type;
    END IF;
END
$$;

-- One row per diarized speaker turn, mirroring face_detection's shape:
-- detection rows first, clustering assigns them to a *_cluster afterwards.
-- Turns are tied to source_file rather than evidence_node — a turn's
-- boundaries come from the diarizer, not from how the video was cut into
-- scene_segment nodes, and the two frequently disagree (a turn can span two
-- segments, or a segment can contain several turns). Which evidence nodes a
-- turn is relevant to is answered later, by a time-overlap query, not by a
-- foreign key here.
CREATE TABLE IF NOT EXISTS voice_segment (
    id                UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    case_id           UUID NOT NULL REFERENCES "case"(id) ON DELETE CASCADE,
    source_file_id    UUID NOT NULL REFERENCES source_file(id) ON DELETE CASCADE,
    start_time        DOUBLE PRECISION NOT NULL,
    end_time          DOUBLE PRECISION NOT NULL,
    -- Diarizer-assigned label local to this one file (e.g. 'SPEAKER_00');
    -- meaningless across files until voice_cluster ties turns together.
    speaker_label     TEXT NOT NULL,
    embedding         VECTOR(256) NOT NULL,
    voice_cluster_id  UUID REFERENCES voice_cluster(id) ON DELETE SET NULL,
    metadata          JSONB DEFAULT '{}'::jsonb,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_voice_segment_case ON voice_segment(case_id);
CREATE INDEX IF NOT EXISTS idx_voice_segment_source ON voice_segment(source_file_id);
CREATE INDEX IF NOT EXISTS idx_voice_segment_cluster ON voice_segment(voice_cluster_id);
CREATE INDEX IF NOT EXISTS idx_voice_segment_embedding ON voice_segment
    USING hnsw (embedding vector_cosine_ops);

-- IDENTITY_LINK edges: an evidence node showing a face or carrying a voice
-- that resolved to an identity. Reuses `relationship`'s existing
-- subject_node_id (added in phase 5) paired with its original
-- object_identity_id column (present since phase 3, unused until now) rather
-- than adding new columns — the shape a node-to-identity edge needs already
-- exists, just never had a matching unique constraint.
CREATE UNIQUE INDEX IF NOT EXISTS uq_relationship_identity_link
    ON relationship(case_id, subject_node_id, object_identity_id, relationship_type)
    WHERE subject_node_id IS NOT NULL AND object_identity_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_relationship_object_identity ON relationship(object_identity_id);
