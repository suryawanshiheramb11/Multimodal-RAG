import { useState } from 'react';
import { Stack as Layers } from '@phosphor-icons/react';
import { api } from '../api';
import { fileIcon, locationLabel } from '../lib/format';

export default function ResultCard({ hit, onOpen }) {
  // A node can legitimately have no still (an audio track), and a thumbnail
  // can 404 if its frame was cleaned up — both fall back to the icon rather
  // than leaving a broken image in the grid.
  const [thumbFailed, setThumbFailed] = useState(false);
  const showThumb = !thumbFailed && hit.node_type !== 'audio_track';
  const where = locationLabel(hit);

  return (
    <button className="card" onClick={() => onOpen(hit)}>
      <div className="card-thumb">
        {showThumb ? (
          <img
            src={api.thumbnailUrl(hit.node_id)}
            alt={hit.snippet || hit.file_name}
            loading="lazy"
            onError={() => setThumbFailed(true)}
          />
        ) : (
          <div className="placeholder">
            {fileIcon(hit.file_type, 26)}
            <small>{hit.node_type.replace('_', ' ')}</small>
          </div>
        )}
        <span className="badge tl">{hit.node_type.replace('_', ' ')}</span>
        {hit.score !== undefined && (
          <span className="badge tr">{Math.round(hit.score * 100)}%</span>
        )}
        {where && <span className="badge bl">{where}</span>}
      </div>

      <div className="card-body">
        <div className="card-file">
          {fileIcon(hit.file_type)}
          <span title={hit.file_name}>{hit.file_name}</span>
        </div>
        {hit.snippet ? (
          <p className="card-snippet">{hit.snippet}</p>
        ) : (
          <p className="card-snippet" style={{ color: 'var(--text-faint)' }}>
            No text extracted — matched on appearance.
          </p>
        )}
        {hit.matched_on?.length > 0 && (
          <div className="match-tags">
            {hit.matched_on.map((m) => (
              <span key={m} className={`match-tag ${m}`}>
                {m === 'visual' ? 'looks like' : 'mentions'}
              </span>
            ))}
          </div>
        )}
      </div>
    </button>
  );
}

export function SkeletonGrid({ count = 8 }) {
  return (
    <div className="grid">
      {Array.from({ length: count }, (_, i) => (
        <div key={i} className="skeleton-card">
          <div className="sk-thumb" />
          <div className="sk-line" style={{ width: '70%' }} />
          <div className="sk-line" style={{ width: '90%' }} />
        </div>
      ))}
    </div>
  );
}

export function EmptyState({ icon, title, children, variant }) {
  return (
    <div className={`state ${variant || ''}`}>
      <div className="state-icon">{icon || <Layers size={24} />}</div>
      <h3>{title}</h3>
      <p>{children}</p>
    </div>
  );
}
