import { useState } from 'react';
import { MagnifyingGlass as Search, ArrowRight, MagnifyingGlassMinus as SearchX, Sparkle as Sparkles, Warning as AlertTriangle, CircleNotch as Loader2 } from '@phosphor-icons/react';
import ResultCard, { SkeletonGrid, EmptyState } from './ResultCard';

const MODES = [
  { id: 'hybrid', label: 'Everything', hint: 'appearance and words together' },
  { id: 'visual', label: 'What it looks like', hint: 'CLIP image matching' },
  { id: 'text', label: 'What was said', hint: 'transcripts, captions, documents' },
];

const EXAMPLES = ['mountains', 'a person talking', 'text on a screen', 'someone outdoors'];

export default function SearchView({
  query, setQuery, mode, setMode, results, searching, searched, error,
  onSearch, onOpen, scope, coverage,
}) {
  const [touched, setTouched] = useState(false);

  const submit = (e) => {
    e?.preventDefault();
    setTouched(true);
    onSearch();
  };

  const runExample = (example) => {
    setQuery(example);
    setTouched(true);
    onSearch(example);
  };

  const nothingIndexed = coverage && coverage.total > 0 &&
    coverage.visual_indexed === 0 && coverage.text_indexed === 0;

  return (
    <>
      <div className={`hero ${searched ? 'compact' : ''}`}>
        {!searched && (
          <>
            <h1>Search what's <Sparkles size={38} className="text-accent" style={{ verticalAlign: 'text-bottom', filter: 'drop-shadow(0 0 12px var(--accent))', margin: '0 4px' }} /> <em>inside</em> your media</h1>
            <p>
              Type what you remember seeing or hearing. Every frame, transcript and page
              is indexed by meaning — so "mountains" finds mountain footage even when no
              filename, caption or tag ever said the word.
            </p>
          </>
        )}

        <form className="searchbar" onSubmit={submit}>
          <Search size={19} />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="mountains at sunset, a red car, someone shouting…"
            autoFocus
            aria-label="Search your media"
          />
          <button className="search-go" type="submit" disabled={!query.trim() || searching}>
            {searching ? <Loader2 size={17} className="spin" /> : <ArrowRight size={17} />}
          </button>
        </form>

        <div className="mode-row">
          {MODES.map((m) => (
            <button
              key={m.id}
              className={`mode-pill ${mode === m.id ? 'active' : ''}`}
              onClick={() => { setMode(m.id); if (touched && query.trim()) onSearch(query, m.id); }}
              title={m.hint}
              type="button"
            >
              {m.label}
            </button>
          ))}
        </div>

        {!searched && (
          <div className="suggestions">
            <span>Try</span>
            {EXAMPLES.map((example) => (
              <button key={example} className="chip" onClick={() => runExample(example)} type="button">
                {example}
              </button>
            ))}
          </div>
        )}

        <div className="scope-note">
          Searching <b>{scope}</b>
          {coverage ? ` · ${coverage.visual_indexed} visual, ${coverage.text_indexed} text vectors indexed` : ''}
        </div>
      </div>

      {nothingIndexed && (
        <div className="banner warn">
          <AlertTriangle size={16} />
          <div>
            Your media is uploaded but nothing is indexed yet, so search has nothing to match
            against. Open <b>Library</b> and re-upload, or run <code>enrich</code> from the CLI.
          </div>
        </div>
      )}

      {error && (
        <div className="banner err">
          <AlertTriangle size={16} />
          <div>{error}</div>
        </div>
      )}

      {searching && (
        <>
          <div className="results-head"><h2>Searching…</h2></div>
          <SkeletonGrid />
        </>
      )}

      {!searching && searched && (
        <>
          <div className="results-head">
            <h2>{results.length > 0 ? `${results.length} match${results.length === 1 ? '' : 'es'}` : 'No matches'}</h2>
            <span className="muted">for "{searched}"</span>
          </div>

          {results.length > 0 ? (
            <div className="grid">
              {results.map((hit) => (
                <ResultCard key={hit.node_id} hit={hit} onOpen={onOpen} />
              ))}
            </div>
          ) : (
            <EmptyState icon={<SearchX size={24} />} title="Nothing matched that">
              No segment in this collection resembles "{searched}" closely enough to be a real
              match. Weak look-alikes are deliberately hidden rather than padded into the results —
              try different wording, or switch mode above.
            </EmptyState>
          )}
        </>
      )}

      {!searching && !searched && (
        <EmptyState icon={<Sparkles size={24} />} title="Ready when you are">
          Results appear here. Search runs against every frame, spoken word and page of text
          the pipeline has analyzed.
        </EmptyState>
      )}
    </>
  );
}
