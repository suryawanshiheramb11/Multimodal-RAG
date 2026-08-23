import { useCallback, useEffect, useRef, useState } from 'react';
import {
  Plus, CloudArrowUp as UploadCloud, Trash, CircleNotch as Loader2, CheckCircle, XCircle, FolderOpen, Tray as Inbox,
} from '@phosphor-icons/react';
import { api } from '../api';
import { EmptyState } from './ResultCard';
import { fileIcon, formatSize } from '../lib/format';

/** How far a file got through the pipeline, as one honest label. */
function fileStatus(file) {
  if (file.node_count === 0) return { cls: 'err', text: 'not extracted' };
  if (file.enriched_count === 0) return { cls: 'warn', text: `${file.node_count} segments · not analyzed` };
  if (file.enriched_count < file.node_count) {
    return { cls: 'warn', text: `${file.enriched_count}/${file.node_count} analyzed` };
  }
  return { cls: 'ok', text: `${file.node_count} searchable` };
}

export default function LibraryView({
  collections, activeCollection, onSelect, onCollectionsChanged, jobs, onJobStarted,
}) {
  const [files, setFiles] = useState([]);
  const [newName, setNewName] = useState('');
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef(null);

  const refreshFiles = useCallback(async () => {
    if (!activeCollection) { setFiles([]); return; }
    try {
      setFiles(await api.listFiles(activeCollection.id));
    } catch (e) {
      setError(e.message);
    }
  }, [activeCollection]);

  useEffect(() => { refreshFiles(); }, [refreshFiles]);

  // Re-read the file list when a job finishes, so counts and status pills
  // catch up with what the worker just committed.
  const finishedCount = jobs.filter((j) => j.status === 'done' || j.status === 'failed').length;
  useEffect(() => { refreshFiles(); }, [finishedCount, refreshFiles]);

  const createCollection = async (e) => {
    e.preventDefault();
    const name = newName.trim();
    if (!name) return;
    setCreating(true);
    setError(null);
    try {
      const created = await api.createCollection(name);
      setNewName('');
      await onCollectionsChanged();
      onSelect(created);
    } catch (err) {
      setError(err.message);
    } finally {
      setCreating(false);
    }
  };

  const removeCollection = async (collection, event) => {
    event.stopPropagation();
    if (!window.confirm(`Delete "${collection.case_number}" and everything indexed from it?\n\nUploaded files stay on disk.`)) return;
    try {
      await api.deleteCollection(collection.id);
      await onCollectionsChanged();
      if (activeCollection?.id === collection.id) onSelect(null);
    } catch (err) {
      setError(err.message);
    }
  };

  const uploadFiles = async (fileList) => {
    if (!activeCollection || fileList.length === 0) return;
    setError(null);
    for (const file of fileList) {
      try {
        const { job_id } = await api.upload(activeCollection.id, file);
        await onJobStarted(job_id);
      } catch (err) {
        setError(`${file.name}: ${err.message}`);
      }
    }
    if (inputRef.current) inputRef.current.value = '';
  };

  const onDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    uploadFiles([...e.dataTransfer.files]);
  };

  return (
    <div className="lib-layout">
      <aside className="panel">
        <div className="panel-head"><h3>Collections</h3></div>

        {collections.length === 0 && (
          <p style={{ fontSize: '0.82rem', color: 'var(--text-faint)', marginBottom: '0.75rem', lineHeight: 1.5 }}>
            Create one to start uploading media.
          </p>
        )}

        {collections.map((collection) => (
          <div
            key={collection.id}
            className={`collection-item ${activeCollection?.id === collection.id ? 'active' : ''}`}
            onClick={() => onSelect(collection)}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => { if (e.key === 'Enter') onSelect(collection); }}
            style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}
          >
            <div style={{ flex: 1, minWidth: 0 }}>
              <div className="name">{collection.case_number}</div>
              <div className="meta">
                {collection.file_count} file{collection.file_count === 1 ? '' : 's'} ·{' '}
                {collection.enriched_count}/{collection.node_count} searchable
              </div>
            </div>
            <button
              className="btn ghost"
              onClick={(e) => removeCollection(collection, e)}
              aria-label={`Delete ${collection.case_number}`}
              title="Delete collection"
            >
              <Trash size={14} />
            </button>
          </div>
        ))}

        <form className="field" onSubmit={createCollection}>
          <input
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            placeholder="New collection name"
            aria-label="New collection name"
          />
          <button className="btn primary" type="submit" disabled={!newName.trim() || creating}>
            {creating ? <Loader2 size={15} className="spin" /> : <Plus size={15} />}
          </button>
        </form>
      </aside>

      <main style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', minWidth: 0 }}>
        {error && (
          <div className="banner err" style={{ marginBottom: 0 }}>
            <XCircle size={16} /><div>{error}</div>
          </div>
        )}

        {!activeCollection ? (
          <EmptyState icon={<FolderOpen size={24} />} title="Pick a collection">
            Choose a collection on the left, or create one, then drop in video, audio, images
            or PDFs. Each upload is split into segments, analyzed by the models, and indexed
            for search.
          </EmptyState>
        ) : (
          <>
            <div className="panel">
              <div className="panel-head">
                <h3>Add to {activeCollection.case_number}</h3>
              </div>
              <label
                className={`dropzone ${dragOver ? 'over' : ''}`}
                onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                onDragLeave={() => setDragOver(false)}
                onDrop={onDrop}
              >
                <input
                  ref={inputRef}
                  type="file"
                  multiple
                  hidden
                  onChange={(e) => uploadFiles([...e.target.files])}
                  accept="video/*,audio/*,image/*,.pdf"
                />
                <UploadCloud size={26} />
                <strong>Drop files here, or click to choose</strong>
                <small>Video, audio, images and PDFs · processing runs in the background</small>
              </label>
            </div>

            {jobs.length > 0 && (
              <div className="panel">
                <div className="panel-head"><h3>Processing</h3></div>
                {jobs.map((job) => (
                  <div key={job.id} className="job-row">
                    {job.status === 'done' ? <CheckCircle size={16} color="var(--ok)" />
                      : job.status === 'failed' ? <XCircle size={16} color="var(--err)" />
                      : <Loader2 size={16} className="spin" color="var(--accent)" />}
                    <div className="info">
                      <b>{job.file_name}</b>
                      <small>{job.error || job.detail || job.stage}</small>
                    </div>
                    <span className={`pill ${job.status === 'done' ? 'ok' : job.status === 'failed' ? 'err' : 'busy'}`}>
                      {job.status === 'running' ? job.stage : job.status}
                    </span>
                  </div>
                ))}
              </div>
            )}

            <div className="panel">
              <div className="panel-head">
                <h3>Files</h3>
                <span style={{ fontSize: '0.78rem', color: 'var(--text-faint)' }}>
                  {files.length} total
                </span>
              </div>
              {files.length === 0 ? (
                <div style={{ textAlign: 'center', padding: '2rem 1rem', color: 'var(--text-faint)' }}>
                  <Inbox size={22} style={{ marginBottom: '0.6rem' }} />
                  <p style={{ fontSize: '0.85rem' }}>Nothing uploaded yet.</p>
                </div>
              ) : (
                files.map((file) => {
                  const status = fileStatus(file);
                  return (
                    <div key={file.id} className="file-row">
                      <div className="icon">{fileIcon(file.file_type, 17)}</div>
                      <div className="info">
                        <b title={file.file_name}>{file.file_name}</b>
                        <small>
                          {formatSize(file.size_bytes)} · {file.file_type}
                          {file.type_mismatch ? ' · content/extension mismatch' : ''}
                        </small>
                      </div>
                      <span className={`pill ${status.cls}`}>{status.text}</span>
                    </div>
                  );
                })
              )}
            </div>
          </>
        )}
      </main>
    </div>
  );
}
