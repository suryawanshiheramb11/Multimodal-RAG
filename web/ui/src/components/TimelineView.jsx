import { useCallback, useEffect, useState } from 'react';
import { Clock, ChatText, Eye, TextT, SpeakerHigh, CircleNotch as Loader2, Stack as Layers, ArrowsClockwise } from '@phosphor-icons/react';
import { api } from '../api';
import { timecode } from '../lib/format';

/** Group timeline entries by display_time into "moments" within a window. */
function groupByMoment(timeline, windowSec = 3) {
  if (!timeline.length) return [];
  const moments = [];
  let current = { time: timeline[0].display_time, nodes: [timeline[0]] };

  for (let i = 1; i < timeline.length; i++) {
    const node = timeline[i];
    if (node.display_time != null && current.time != null &&
        Math.abs(node.display_time - current.time) <= windowSec) {
      current.nodes.push(node);
    } else {
      moments.push(current);
      current = { time: node.display_time, nodes: [node] };
    }
  }
  moments.push(current);
  return moments;
}

/** Assign a consistent color per source file. */
const SOURCE_COLORS = [
  'var(--accent)',
  '#f472b6',
  '#34d399',
  '#fbbf24',
  '#a78bfa',
  '#fb923c',
  '#38bdf8',
];
function sourceColor(index) {
  return SOURCE_COLORS[index % SOURCE_COLORS.length];
}

export default function TimelineView({ activeCollection, onOpen }) {
  const [timeline, setTimeline] = useState([]);
  const [syncStatus, setSyncStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const load = useCallback(async () => {
    if (!activeCollection) return;
    setLoading(true);
    setError(null);
    try {
      const [tl, sync] = await Promise.all([
        api.timeline(activeCollection.id),
        api.syncStatus(activeCollection.id).catch(() => null),
      ]);
      setTimeline(tl.timeline || []);
      setSyncStatus(sync);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [activeCollection]);

  useEffect(() => { load(); }, [load]);

  if (!activeCollection) {
    return (
      <div className="state">
        <div className="state-icon"><Clock size={24} /></div>
        <h3>Select a collection</h3>
        <p>Choose a collection from the dropdown to see its unified timeline.</p>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="state">
        <div className="state-icon"><Loader2 size={24} className="spin" /></div>
        <h3>Loading timeline…</h3>
      </div>
    );
  }

  if (error) {
    return (
      <div className="banner err">
        <div>{error}</div>
      </div>
    );
  }

  if (timeline.length === 0) {
    return (
      <div className="state">
        <div className="state-icon"><Clock size={24} /></div>
        <h3>No timeline data</h3>
        <p>Upload media and run the pipeline to generate the timeline. Nodes need timestamps to appear here.</p>
      </div>
    );
  }

  // Build source-file index for color coding
  const sourceFiles = [...new Set(timeline.map((n) => n.file_name))];
  const sourceColorMap = {};
  sourceFiles.forEach((name, i) => { sourceColorMap[name] = sourceColor(i); });

  const moments = groupByMoment(timeline);

  return (
    <div className="timeline-view">
      {/* Sync status banner */}
      {syncStatus && (
        <div className="timeline-sync-banner">
          <ArrowsClockwise size={15} />
          {syncStatus.synced ? (
            <span>
              Sources synchronised
              {syncStatus.offsets?.map((o, i) => (
                <span key={i} className="sync-detail">
                  {' '}{o.source_a.name} → {o.source_b.name}: {o.offset_seconds >= 0 ? '+' : ''}{o.offset_seconds.toFixed(1)}s
                  ({Math.round(o.confidence * 100)}%)
                </span>
              ))}
            </span>
          ) : (
            <span>Sources not yet synchronised — upload ≥2 media files and run the pipeline</span>
          )}
        </div>
      )}

      {/* Source legend */}
      <div className="timeline-legend">
        {sourceFiles.map((name) => (
          <span key={name} className="legend-item">
            <span className="legend-dot" style={{ background: sourceColorMap[name] }} />
            {name}
          </span>
        ))}
      </div>

      {/* Timeline track */}
      <div className="timeline-track">
        {moments.map((moment, mi) => (
          <div className="timeline-moment" key={mi}>
            <div className="timeline-time">
              <Clock size={13} />
              {moment.time != null ? timecode(moment.time) : '—'}
            </div>
            <div className="timeline-nodes">
              {moment.nodes.map((node) => (
                <button
                  className="timeline-node"
                  key={node.node_id}
                  onClick={() => onOpen?.({
                    node_id: node.node_id,
                    source_file_id: node.source_file_id,
                    file_name: node.file_name,
                    file_type: node.file_type,
                    node_type: node.node_type,
                    start_time: node.start_time,
                    end_time: node.end_time,
                  })}
                  style={{ borderLeftColor: sourceColorMap[node.file_name] }}
                >
                  <div className="tl-node-head">
                    <span className="badge tl-badge">{node.node_type.replace('_', ' ')}</span>
                    <span className="tl-source" style={{ color: sourceColorMap[node.file_name] }}>
                      {node.file_name}
                    </span>
                  </div>

                  <div className="tl-modalities">
                    {node.transcript_text && (
                      <div className="tl-mod">
                        <ChatText size={12} />
                        <span>{truncate(node.transcript_text, 120)}</span>
                      </div>
                    )}
                    {node.caption && (
                      <div className="tl-mod">
                        <Eye size={12} />
                        <span>{truncate(node.caption, 120)}</span>
                      </div>
                    )}
                    {node.ocr_text && (
                      <div className="tl-mod">
                        <TextT size={12} />
                        <span>{truncate(node.ocr_text, 80)}</span>
                      </div>
                    )}
                    {node.audio_events?.length > 0 && (
                      <div className="tl-mod">
                        <SpeakerHigh size={12} />
                        <span>{node.audio_events.slice(0, 3).map((e) => e.label).join(', ')}</span>
                      </div>
                    )}
                    {!node.transcript_text && !node.caption && !node.ocr_text && (
                      <div className="tl-mod faint">No text extracted</div>
                    )}
                  </div>
                </button>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function truncate(text, limit) {
  if (!text) return '';
  const cleaned = text.replace(/^"|"$/g, '').trim();
  return cleaned.length <= limit ? cleaned : cleaned.slice(0, limit - 1).trimEnd() + '…';
}
