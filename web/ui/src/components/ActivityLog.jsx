import { useEffect, useRef, useState } from 'react';
import {
  Check, X, CircleNotch as Loader2, CaretRight as ChevronRight, CaretDown as ChevronDown, TerminalWindow as Terminal, SidebarSimple as PanelLeftClose,
  SidebarSimple as PanelLeftOpen, CircleDashed,
} from '@phosphor-icons/react';

/**
 * The reasoning rail.
 *
 * A pipeline that reports only "done" is a black box: you cannot tell why a
 * file became searchable, or why it didn't. This shows the chain — each stage,
 * the pipeline's own log lines, and the conclusions the models reached
 * (what was transcribed, read off the screen, detected) — so a result in the
 * grid can be traced back to the step that produced it.
 */

function StatusIcon({ status }) {
  if (status === 'ok' || status === 'done') return <Check size={13} className="st ok" />;
  if (status === 'failed') return <X size={13} className="st err" />;
  if (status === 'skipped') return <CircleDashed size={13} className="st dim" />;
  return <Loader2 size={13} className="st busy spin" />;
}

function clockOf(iso) {
  if (!iso) return '';
  return new Date(iso).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

function Stage({ stage }) {
  // Open the stage that is running so progress is visible without a click;
  // collapse it once it succeeds so a finished job stays scannable. A failed
  // stage stays open — that is the one you need to read.
  const [open, setOpen] = useState(stage.status === 'running' || stage.status === 'failed');
  const wasRunning = useRef(stage.status === 'running');

  useEffect(() => {
    if (wasRunning.current && stage.status === 'ok') setOpen(false);
    if (stage.status === 'failed') setOpen(true);
    wasRunning.current = stage.status === 'running';
  }, [stage.status]);

  const hasBody = stage.logs.length > 0 || stage.findings.length > 0;

  return (
    <div className={`stage ${stage.status}`}>
      <button className="stage-head" onClick={() => hasBody && setOpen(!open)} disabled={!hasBody}>
        <span className="stage-time">{clockOf(stage.started_at)}</span>
        <StatusIcon status={stage.status} />
        <span className="stage-label">{stage.label}</span>
        {hasBody && (open ? <ChevronDown size={12} /> : <ChevronRight size={12} />)}
      </button>

      {stage.detail && <div className="stage-detail">→ {stage.detail}</div>}

      {open && hasBody && (
        <div className="stage-body">
          {stage.findings.map((finding, i) => (
            <div key={`f${i}`} className="finding">{finding}</div>
          ))}
          {stage.logs.length > 0 && (
            <pre className="stage-logs">{stage.logs.join('\n')}</pre>
          )}
        </div>
      )}
    </div>
  );
}

function JobEntry({ job, defaultOpen }) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <div className="job-entry">
      <button className="job-entry-head" onClick={() => setOpen(!open)}>
        {open ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
        <StatusIcon status={job.status} />
        <span className="job-name" title={job.file_name}>{job.file_name}</span>
      </button>

      {open && (
        <div className="job-stages">
          {job.stages.length === 0 && (
            <div className="stage-detail">Queued — waiting for a worker…</div>
          )}
          {job.stages.map((stage, i) => <Stage key={`${stage.key}-${i}`} stage={stage} />)}

          {job.status === 'failed' && job.error && (
            <div className="stage-detail err">✕ {job.error}</div>
          )}
          {job.status === 'done' && (
            <div className="job-summary">
              <Check size={12} /> {job.detail}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default function ActivityLog({ jobs, collapsed, onToggle }) {
  const scrollRef = useRef(null);
  const active = jobs.filter((j) => j.status === 'queued' || j.status === 'running');

  // Follow the newest output while work is in flight, but never yank the view
  // out from under someone reading a finished job.
  useEffect(() => {
    if (active.length > 0 && scrollRef.current) {
      scrollRef.current.scrollTop = 0;
    }
  }, [active.length]);

  if (collapsed) {
    return (
      <button className="rail-collapsed" onClick={onToggle} title="Show processing log">
        <PanelLeftOpen size={17} />
        {active.length > 0 && <span className="rail-badge">{active.length}</span>}
        <span className="rail-vertical">Reasoning log</span>
      </button>
    );
  }

  return (
    <aside className="rail">
      <div className="rail-head">
        <Terminal size={14} />
        <h3>Reasoning log</h3>
        {active.length > 0 && <span className="pill busy">{active.length} running</span>}
        <button className="btn ghost" onClick={onToggle} title="Hide log" aria-label="Hide log">
          <PanelLeftClose size={15} />
        </button>
      </div>

      <div className="rail-body" ref={scrollRef}>
        {jobs.length === 0 ? (
          <div className="rail-empty">
            <p>
              Nothing processed yet this session. Upload something in
              <b> Library</b> and each step the pipeline takes — and what it
              concluded — appears here.
            </p>
          </div>
        ) : (
          jobs.map((job, i) => (
            <JobEntry key={job.id} job={job} defaultOpen={i === 0} />
          ))
        )}
      </div>
    </aside>
  );
}
