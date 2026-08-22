import { useEffect, useRef, useState } from 'react';
import { X, Download } from 'lucide-react';
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

export default function DetailModal({ hit, onClose }) {
  const [mediaFailed, setMediaFailed] = useState(false);
  // Search hands over a trimmed snippet to keep the grid light; the full text
  // and metadata are fetched here so the detail view shows the whole record.
  const [full, setFull] = useState(null);
  const closeRef = useRef(null);

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

  if (!hit) return null;

  const detail = { ...hit, ...(full || {}) };
  const sections = splitSections(detail.text_content ?? hit.snippet);
  const isVideo = hit.file_type === 'video';
  const isAudio = hit.file_type === 'audio' || hit.node_type === 'audio_track';
  const start = hit.start_time ?? 0;

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
              // #t= seeks straight to the moment that matched, so the hit is
              // verifiable rather than just asserted.
              <video
                src={`${api.fileMediaUrl(hit.source_file_id)}#t=${start}`}
                controls
                preload="metadata"
                onError={() => setMediaFailed(true)}
              />
            ) : isAudio && !mediaFailed ? (
              <audio
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
          </div>

          {sections.length > 0 ? (
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
          )}

          {detail.detections?.length > 0 && (
            <div>
              <div className="section-label">Objects detected</div>
              <div className="tag-row">
                {detail.detections.map((d) => <span key={d} className="tag">{d}</span>)}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
