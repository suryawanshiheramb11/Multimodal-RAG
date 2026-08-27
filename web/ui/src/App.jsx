import { useCallback, useEffect, useState } from 'react';
import { MagnifyingGlass as SearchIcon, FolderOpen as Library, Aperture, Question as MessageCircleQuestion, Clock, ShareNetwork } from '@phosphor-icons/react';
import { api } from './api';
import SearchView from './components/SearchView';
import LibraryView from './components/LibraryView';
import AskView from './components/AskView';
import TimelineView from './components/TimelineView';
import GraphView from './components/GraphView';
import DetailModal from './components/DetailModal';
import ActivityLog from './components/ActivityLog';
import { useJobs } from './lib/useJobs';
import './index.css';

export default function App() {
  const [tab, setTab] = useState('search');
  const [stats, setStats] = useState(null);
  const [coverage, setCoverage] = useState(null);

  const [collections, setCollections] = useState([]);
  const [activeCollection, setActiveCollection] = useState(null);

  const [query, setQuery] = useState('');
  const [mode, setMode] = useState('hybrid');
  const [results, setResults] = useState([]);
  const [searching, setSearching] = useState(false);
  const [searched, setSearched] = useState(null);
  const [error, setError] = useState(null);

  const [openHit, setOpenHit] = useState(null);
  const [railCollapsed, setRailCollapsed] = useState(false);

  const refreshStats = useCallback(async () => {
    try {
      const [s, status] = await Promise.all([api.stats(), api.searchStatus()]);
      setStats(s);
      setCoverage(status.coverage);
    } catch {
      // The header is decoration; a stats failure must not blank the app.
    }
  }, []);

  const refreshCollections = useCallback(async () => {
    try {
      const list = await api.listCollections();
      setCollections(list);
      // Keep the selected collection's counts fresh without losing the
      // selection, and drop it if it no longer exists.
      setActiveCollection((current) =>
        current ? list.find((c) => c.id === current.id) ?? null : null,
      );
      return list;
    } catch (e) {
      setError(e.message);
      return [];
    }
  }, []);

  // A finished job changed the database, so pull the new counts in.
  const onJobFinished = useCallback(() => {
    refreshStats();
    refreshCollections();
  }, [refreshStats, refreshCollections]);

  const { jobs, track } = useJobs(onJobFinished);

  useEffect(() => {
    refreshStats();
    refreshCollections();
  }, [refreshStats, refreshCollections]);

  const runSearch = useCallback(
    async (overrideQuery, overrideMode) => {
      const q = (overrideQuery ?? query).trim();
      const m = overrideMode ?? mode;
      if (!q) return;

      setSearching(true);
      setError(null);
      try {
        const data = await api.search(q, m, activeCollection?.id);
        setResults(data.results);
        setSearched(q);
        if (!data.encoders.visual && !data.encoders.text) {
          setError('No search model could be loaded — check that the API host can reach its model files.');
        } else if (m === 'visual' && !data.encoders.visual) {
          setError('The visual model (CLIP) is unavailable, so appearance search is disabled.');
        } else if (m === 'text' && !data.encoders.text) {
          setError('The text model is unavailable, so transcript search is disabled.');
        }
      } catch (e) {
        setError(e.message);
        setResults([]);
        setSearched(q);
      } finally {
        setSearching(false);
      }
    },
    [query, mode, activeCollection],
  );

  const scope = activeCollection ? activeCollection.case_number : 'all collections';

  return (
    <div className="shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark"><Aperture size={17} /></span>
          <span>
            Prism<br />
            <span className="brand-sub">semantic media search</span>
          </span>
        </div>

        <nav className="nav-tabs">
          <button
            className={`nav-tab ${tab === 'search' ? 'active' : ''}`}
            onClick={() => setTab('search')}
          >
            <SearchIcon size={15} /> Search
          </button>
          <button
            className={`nav-tab ${tab === 'ask' ? 'active' : ''}`}
            onClick={() => setTab('ask')}
          >
            <MessageCircleQuestion size={15} /> Ask
          </button>
          <button
            className={`nav-tab ${tab === 'library' ? 'active' : ''}`}
            onClick={() => { setTab('library'); refreshCollections(); }}
          >
            <Library size={15} /> Library
          </button>
          <button
            className={`nav-tab ${tab === 'timeline' ? 'active' : ''}`}
            onClick={() => setTab('timeline')}
          >
            <Clock size={15} /> Timeline
          </button>
          <button
            className={`nav-tab ${tab === 'graph' ? 'active' : ''}`}
            onClick={() => setTab('graph')}
          >
            <ShareNetwork size={15} /> Graph
          </button>
        </nav>

        <div className="topbar-right">
          <select
            className="mode-pill"
            value={activeCollection?.id || ''}
            onChange={(e) => {
              const found = collections.find((c) => c.id === e.target.value);
              setActiveCollection(found || null);
            }}
            style={{ background: 'var(--bg-input)', padding: '0.45rem 0.8rem' }}
            aria-label="Limit search to a collection"
          >
            <option value="">All collections</option>
            {collections.map((c) => (
              <option key={c.id} value={c.id}>{c.case_number}</option>
            ))}
          </select>

          <div className="mini-stats">
            <div className="mini-stat"><b>{stats?.files ?? '–'}</b><span>files</span></div>
            <div className="mini-stat"><b>{stats?.searchable ?? '–'}</b><span>indexed</span></div>
          </div>
        </div>
      </header>

      <div className={`workspace ${railCollapsed ? 'rail-hidden' : ''}`}>
        <div className="content">
          {tab === 'search' ? (
            <SearchView
              query={query}
              setQuery={setQuery}
              mode={mode}
              setMode={setMode}
              results={results}
              searching={searching}
              searched={searched}
              error={error}
              onSearch={runSearch}
              onOpen={setOpenHit}
              scope={scope}
              coverage={coverage}
            />
          ) : tab === 'ask' ? (
            <AskView activeCollection={activeCollection} />
          ) : tab === 'timeline' ? (
            <TimelineView activeCollection={activeCollection} onOpen={setOpenHit} />
          ) : tab === 'graph' ? (
            <GraphView activeCollection={activeCollection} onOpen={setOpenHit} onJobStarted={track} />
          ) : (
            <LibraryView
              collections={collections}
              activeCollection={activeCollection}
              onSelect={setActiveCollection}
              onCollectionsChanged={refreshCollections}
              jobs={jobs}
              onJobStarted={track}
            />
          )}
        </div>

        <ActivityLog
          jobs={jobs}
          collapsed={railCollapsed}
          onToggle={() => setRailCollapsed((c) => !c)}
        />
      </div>

      {openHit && <DetailModal hit={openHit} onClose={() => setOpenHit(null)} />}
    </div>
  );
}
