import { useRef, useState } from 'react';
import {
  MessageCircleQuestion, ArrowRight, Loader2, Sparkles, ListTree, Cpu, FolderSearch,
} from 'lucide-react';
import { api } from '../api';
import { EmptyState } from './ResultCard';

const EXAMPLES = [
  'What does the video show?',
  'Are there any contradictions?',
  'Who is present in this evidence?',
  'What happened before the meeting?',
];

const INTENT_LABEL = {
  entity: 'about a topic',
  contradiction: 'checking for disagreements',
  identity: 'about a person',
  co_occurrence: 'about who else is present',
  timeline: 'about a point in time',
  general: 'closest match',
};

/**
 * One question, one answer, shown as a small exchange rather than a full
 * chat log — this asks the graph a question, it doesn't converse. Each
 * answer names its own intent and whether an LLM actually wrote it, because
 * "the model wrote this" and "this is a plain listing of what was found" are
 * different enough claims that the reader should not have to guess which one
 * they're looking at.
 */
function Exchange({ exchange }) {
  const [showFacts, setShowFacts] = useState(false);
  const { question, answer, error } = exchange;

  return (
    <div className="panel" style={{ marginBottom: '1rem' }}>
      <div className="ask-question">
        <MessageCircleQuestion size={15} />
        <span>{question}</span>
      </div>

      {error ? (
        <div className="banner err" style={{ margin: '0.75rem 0 0' }}>{error}</div>
      ) : answer ? (
        <>
          <div className="ask-answer">{answer.answer}</div>
          <div className="ask-meta">
            <span className="tag">
              <ListTree size={12} /> {INTENT_LABEL[answer.intent] || answer.intent}
            </span>
            <span className="tag">
              <Cpu size={12} /> {answer.used_llm ? 'written by the model' : 'listed, no model used'}
            </span>
            {answer.facts.length > 0 && (
              <button className="btn ghost sm" onClick={() => setShowFacts((s) => !s)}>
                {showFacts ? 'Hide' : 'Show'} {answer.facts.length} supporting fact
                {answer.facts.length === 1 ? '' : 's'}
              </button>
            )}
          </div>

          {showFacts && (
            <div className="ask-facts">
              {answer.facts.map((fact, i) => (
                <div key={fact.node_id || i} className="ask-fact">
                  <b>{fact.label}</b>
                  {fact.detail && <p>{fact.detail}</p>}
                </div>
              ))}
            </div>
          )}
        </>
      ) : (
        <div className="ask-answer" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--text-faint)' }}>
          <Loader2 size={14} className="spin" /> Retrieving evidence…
        </div>
      )}
    </div>
  );
}

export default function AskView({ activeCollection }) {
  const [question, setQuestion] = useState('');
  const [exchanges, setExchanges] = useState([]);
  const [asking, setAsking] = useState(false);
  const inputRef = useRef(null);

  const ask = async (overrideQuestion) => {
    const q = (overrideQuestion ?? question).trim();
    if (!q || !activeCollection || asking) return;

    setQuestion('');
    setAsking(true);
    const placeholder = { question: q, answer: null, error: null };
    setExchanges((current) => [placeholder, ...current]);

    try {
      const answer = await api.ask(q, activeCollection.id);
      setExchanges((current) =>
        current.map((ex) => (ex === placeholder ? { ...ex, answer } : ex)),
      );
    } catch (e) {
      setExchanges((current) =>
        current.map((ex) => (ex === placeholder ? { ...ex, error: e.message } : ex)),
      );
    } finally {
      setAsking(false);
      inputRef.current?.focus();
    }
  };

  if (!activeCollection) {
    return (
      <EmptyState icon={<FolderSearch size={24} />} title="Pick a collection first">
        Questions are answered against one collection's evidence graph — choose one from the
        selector at the top right, or create one in Library.
      </EmptyState>
    );
  }

  return (
    <>
      <div className={`hero ${exchanges.length ? 'compact' : ''}`}>
        {!exchanges.length && (
          <>
            <h1>Ask about <em>{activeCollection.case_number}</em></h1>
            <p>
              Retrieval is plain SQL against the evidence graph — entities, contradictions,
              identities, timelines. A model only writes the final sentence from what was
              actually found, so an answer never states something the evidence doesn't support.
            </p>
          </>
        )}

        <form
          className="searchbar"
          onSubmit={(e) => { e.preventDefault(); ask(); }}
        >
          <MessageCircleQuestion size={19} />
          <input
            ref={inputRef}
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Ask a question about this collection…"
            autoFocus
            aria-label="Ask a question"
          />
          <button className="search-go" type="submit" disabled={!question.trim() || asking}>
            {asking ? <Loader2 size={17} className="spin" /> : <ArrowRight size={17} />}
          </button>
        </form>

        {!exchanges.length && (
          <div className="suggestions">
            <span>Try</span>
            {EXAMPLES.map((example) => (
              <button key={example} className="chip" onClick={() => ask(example)} type="button">
                {example}
              </button>
            ))}
          </div>
        )}
      </div>

      {exchanges.length > 0 ? (
        <div style={{ maxWidth: '48rem', margin: '0 auto' }}>
          {exchanges.map((exchange, i) => <Exchange key={i} exchange={exchange} />)}
        </div>
      ) : (
        <EmptyState icon={<Sparkles size={24} />} title="Ask anything about this collection">
          Questions about entities, contradictions, timing, or people are answered from the
          evidence graph directly. Anything else falls back to a semantic search over every
          transcript, caption and document in the collection.
        </EmptyState>
      )}
    </>
  );
}
