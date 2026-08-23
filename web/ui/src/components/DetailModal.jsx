import { useCallback, useEffect, useRef, useState } from 'react';
import { X, DownloadSimple as Download, ChatText, Eye, TextT, SpeakerHigh, CaretDown, CaretUp } from '@phosphor-icons/react';
import { api } from '../api';
import { fileIcon, timecode } from '../lib/format';

/**
 * Splits the fused text back into its labelled parts.
 *
 * Enrichment stores one string with "Transcript:", "Visual description:",
 * "On-screen text:" and "Document text:" headings. Showing that raw is a wall
 * of prose; showing each source separately lets a reader tell what was *said*
 * from what was *seen*, which is the distinction the headings exist for.
 */
function splitSections(text) {
  if (!text) return [];
  const known = ['Transcript', 'Visual description', 'On-screen text', 'Document text'];
  const pattern = new RegExp(`^(${known.join('|')}):\\s*`, 'gm');

  const sections = [];
  const matches = [...text.matchAll(pattern)];
  if (matches.length === 0) return [{ label: 'Extracted text', body: text.trim() }];

  matches.forEach((match, i) => {
    const start = match.index + match[0].length;
    const end = i + 1 < matches.length ? matches[i + 1].index : text.length;
    const body = text.slice(start, end).trim();
    if (body) sections.push({ label: match[1], body });
  });
  return sections;
}

/** MM:SS from seconds, for transcript line timestamps. */
function formatTimestamp(sec) {
  if (sec === null || sec === undefined) return '';
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${String(s).padStart(2, '0')}`;
}

export default function DetailModal({ hit, onClose }) {
  const [mediaFailed, setMediaFailed] = useState(false);
  const [full, setFull] = useState(null);
  const [fullTranscript, setFullTranscript] = useState(null);
  const [showFullTranscript, setShowFullTranscript] = useState(false);
  const [loadingTranscript, setLoadingTranscript] = useState(false);
  const closeRef = useRef(null);
  const videoRef = useRef(null);
  const audioRef = useRef(null);

  useEffect(() => {
    let cancelled = false;
    setFull(null);
    api.node(hit.node_id)
      .then((data) => { if (!cancelled) setFull(data); })
      .catch(() => { /* fall back to the snippet already in hand */ });
    return () => { cancelled = true; };
  }, [hit.node_id]);

  // Escape closes, and focus moves into the dialog so keyboard users are not
  // left tabbing through the page behind the overlay.
  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape') onClose(); };
    document.addEventListener('keydown', onKey);
    closeRef.current?.focus();
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => {
      document.removeEventListener('keydown', onKey);
      document.body.style.overflow = previousOverflow;
    };
  }, [onClose]);

  /** Fetch the full file transcript when toggled on. */
  const loadFullTranscript = useCallback(() => {
    if (fullTranscript) {
      setShowFullTranscript((v) => !v);
      return;
    }
    const fileId = full?.source_file_id || hit.source_file_id;
    if (!fileId) return;

    setLoadingTranscript(true);
    api.fileTranscript(fileId)
      .then((data) => {
        setFullTranscript(data);
        setShowFullTranscript(true);
      })
      .catch(() => { /* silently fail, segment transcript still visible */ })
      .finally(() => setLoadingTranscript(false));
  }, [full, hit, fullTranscript]);

  /** Seek the video/audio player to a specific timestamp. */
  const seekTo = useCallback((seconds) => {
    const player = videoRef.current || audioRef.current;
    if (player) {
      player.currentTime = seconds;
      player.play().catch(() => {});
    }
  }, []);

  if (!hit) return null;

  const detail = { ...hit, ...(full || {}) };
  const sections = splitSections(detail.text_content ?? hit.snippet);
  const isVideo = hit.file_type === 'video';
  const isAudio = hit.file_type === 'audio' || hit.node_type === 'audio_track';
  const start = hit.start_time ?? 0;

  // Extract modality data from the detail endpoint
  const transcript = detail.transcript;
  const transcriptSegments = transcript?.segments || [];
  const transcriptText = transcript?.text;
  const caption = detail.caption;
  const ocrData = detail.ocr;
  const ocrText = typeof ocrData === 'object' ? ocrData?.text : ocrData;
  const audioEvents = detail.audio_events;

  // Decide whether we have rich modality data (from the fixed API) or only the fused text
  const hasRichData = transcriptSegments.length > 0 || caption || ocrText;

  return (
    <div className="overlay" onClick={onClose} role="dialog" aria-modal="true">
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          {fileIcon(hit.file_type, 17)}
          <h3 title={hit.file_name}>{hit.file_name}</h3>
          <a
            className="btn sm"
            href={api.fileMediaUrl(hit.source_file_id)}
            target="_blank"
            rel="noreferrer"
          >
            <Download size={14} /> Open original
          </a>
          <button className="btn ghost" onClick={onClose} ref={closeRef} aria-label="Close">
            <X size={18} />
          </button>
        </div>

        <div className="modal-body">
          <div className="media-frame">
            {isVideo && !mediaFailed ? (
              <video
                ref={videoRef}
                src={`${api.fileMediaUrl(hit.source_file_id)}#t=${start}`}
                controls
                preload="metadata"
                onError={() => setMediaFailed(true)}
              />
            ) : isAudio && !mediaFailed ? (
              <audio
                ref={audioRef}
                src={api.nodeMediaUrl(hit.node_id)}
                controls
                onError={() => setMediaFailed(true)}
              />
            ) : (
              <img
                src={api.thumbnailUrl(hit.node_id)}
                alt={hit.file_name}
                onError={(e) => { e.currentTarget.style.display = 'none'; }}
              />
            )}
          </div>

          <div className="meta-grid">
            <div className="meta-cell">
              <span>Type</span>
              <b>{hit.node_type.replace('_', ' ')}</b>
            </div>
            {hit.start_time !== null && hit.start_time !== undefined && (
              <div className="meta-cell">
                <span>Timestamp</span>
                <b>{timecode(hit.start_time)} – {timecode(hit.end_time)}</b>
              </div>
            )}
            {hit.page_number !== null && hit.page_number !== undefined && (
              <div className="meta-cell">
                <span>Page</span>
                <b>{hit.page_number}</b>
              </div>
            )}
            {hit.score !== undefined && (
              <div className="meta-cell">
                <span>Relevance</span>
                <b>{Math.round(hit.score * 100)}%</b>
              </div>
            )}
            {hit.matched_on?.length > 0 && (
              <div className="meta-cell">
                <span>Matched on</span>
                <b>{hit.matched_on.map((m) => (m === 'visual' ? 'appearance' : 'text')).join(' + ')}</b>
              </div>
            )}
            {transcript?.language && (
              <div className="meta-cell">
                <span>Language</span>
                <b>{transcript.language}</b>
              </div>
            )}
          </div>

          {/* ---- Synced modality columns ---- */}
          {hasRichData ? (
            <div className="modality-grid">
              {/* Transcript column */}
              {(transcriptSegments.length > 0 || transcriptText) && (
                <div className="modality-col">
                  <div className="modality-header">
                    <ChatText size={15} weight="bold" />
                    <span>Transcript</span>
                  </div>
                  {transcriptSegments.length > 0 ? (
                    <div className="transcript-segments">
                      {transcriptSegments.map((seg, i) => (
                        <div className="transcript-line" key={i}>
                          <button
                            className="ts-stamp"
                            onClick={() => seekTo(seg.start)}
                            title={`Seek to ${formatTimestamp(seg.start)}`}
                          >
                            {formatTimestamp(seg.start)}
                          </button>
                          <span className="ts-text">{seg.text}</span>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="transcript">{transcriptText}</div>
                  )}
                </div>
              )}

              {/* Visual description column */}
              {caption && (
                <div className="modality-col">
                  <div className="modality-header">
                    <Eye size={15} weight="bold" />
                    <span>Visual description</span>
                  </div>
                  <div className="transcript">{caption}</div>
                </div>
              )}

              {/* On-screen text column */}
              {ocrText && ocrText.trim() && (
                <div className="modality-col">
                  <div className="modality-header">
                    <TextT size={15} weight="bold" />
                    <span>On-screen text</span>
                  </div>
                  <div className="transcript">{ocrText}</div>
                </div>
              )}

              {/* Audio events */}
              {audioEvents?.length > 0 && (
                <div className="modality-col">
                  <div className="modality-header">
                    <SpeakerHigh size={15} weight="bold" />
                    <span>Audio events</span>
                  </div>
                  <div className="tag-row">
                    {audioEvents.map((e, i) => (
                      <span key={i} className="tag">
                        {e.label} {e.probability != null && <small>({Math.round(e.probability * 100)}%)</small>}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ) : (
            /* Fallback: show the fused text split by headings (legacy) */
            sections.length > 0 ? (
              sections.map((section) => (
                <div key={section.label}>
                  <div className="section-label">{section.label}</div>
                  <div className="transcript">{section.body}</div>
                </div>
              ))
            ) : (
              <div>
                <div className="section-label">Extracted text</div>
                <div className="transcript" style={{ color: 'var(--text-faint)' }}>
                  Nothing was transcribed or read from this segment.
                </div>
              </div>
            )
          )}

          {detail.detections?.length > 0 && !audioEvents?.length && (
            <div>
              <div className="section-label">Objects detected</div>
              <div className="tag-row">
                {detail.detections.map((d) => <span key={d} className="tag">{d}</span>)}
              </div>
            </div>
          )}

          {/* ---- Full file transcript toggle ---- */}
          {(isVideo || isAudio) && (
            <div className="full-transcript-section">
              <button
                className="btn sm full-transcript-toggle"
                onClick={loadFullTranscript}
                disabled={loadingTranscript}
              >
                {loadingTranscript ? 'Loading…' : (
                  <>
                    {showFullTranscript ? <CaretUp size={14} /> : <CaretDown size={14} />}
                    {showFullTranscript ? 'Hide full transcript' : 'View full transcript'}
                  </>
                )}
              </button>

              {showFullTranscript && fullTranscript && (
                <div className="full-transcript-panel">
                  <div className="full-transcript-meta">
                    <span><b>{fullTranscript.file_name}</b></span>
                    {fullTranscript.language && <span>Language: {fullTranscript.language}</span>}
                    {fullTranscript.segments?.length > 0 && (
                      <span>{fullTranscript.segments.length} segment(s)</span>
                    )}
                  </div>

                  {fullTranscript.segments?.length > 0 ? (
                    <div className="transcript-segments full">
                      {fullTranscript.segments.map((seg, i) => (
                        <div className="transcript-line" key={i}>
                          <button
                            className="ts-stamp"
                            onClick={() => seekTo(seg.start)}
                            title={`Seek to ${formatTimestamp(seg.start)}`}
                          >
                            {formatTimestamp(seg.start)}
                          </button>
                          <span className="ts-text">{seg.text}</span>
                        </div>
                      ))}
                    </div>
                  ) : fullTranscript.full_text ? (
                    <div className="transcript">{fullTranscript.full_text}</div>
                  ) : (
                    <div className="transcript" style={{ color: 'var(--text-faint)' }}>
                      No transcript available for this file.
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
