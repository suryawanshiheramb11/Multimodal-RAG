-- Multi-modal evidence graph schema
-- Vector dims: text embeddings = 384 (sentence-transformers MiniLM),
-- CLIP embeddings = 512, audio/voice embeddings = 1024

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
    size_bytes      BIGINT,
    duration_sec    DOUBLE PRECISION,
    page_count      INTEGER,
    author          TEXT,
    created_date    TIMESTAMPTZ,
    metadata        JSONB DEFAULT '{}'::jsonb,
    registered_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (case_id, sha256)
);

CREATE INDEX IF NOT EXISTS idx_source_file_case ON source_file(case_id);
CREATE INDEX IF NOT EXISTS idx_source_file_type ON source_file(file_type);

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

CREATE INDEX IF NOT EXISTS idx_entity_node ON entity(evidence_node_id);
CREATE INDEX IF NOT EXISTS idx_entity_type_value ON entity(entity_type, value);

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
