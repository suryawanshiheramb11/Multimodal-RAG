import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  forceCenter, forceCollide, forceLink, forceManyBody, forceSimulation, forceX, forceY,
} from 'd3-force';
import { select } from 'd3-selection';
import { drag } from 'd3-drag';
import { zoom, zoomIdentity } from 'd3-zoom';
import {
  ShareNetwork, CircleNotch as Loader2, MagnifyingGlass, Plus, Minus,
  ArrowsOut, ArrowsIn, Crosshair,
} from '@phosphor-icons/react';
import { api } from '../api';

/** One color per node kind. Entity types match graph/config.py's
 * ENTITY_TYPES; identity, event, and evidence are the other kinds
 * `/graph` returns. */
const NODE_COLORS = {
  person: '#00e5a0',
  organization: '#fbbf24',
  location: '#38bdf8',
  phone: '#a78bfa',
  vehicle: '#fb923c',
  weapon: '#f87171',
  identity: '#f472b6',
  event: '#facc15',
  evidence: '#9aa1b4',
  other: '#646b80',
};

/** Which filter tab a node type falls under — collapses the six entity
 * sub-types into one "Entity" bucket, matching how the graph is actually
 * organized (three phases: extraction, timeline grouping, evidence). */
const CATEGORY_OF = {
  person: 'entity', organization: 'entity', location: 'entity', phone: 'entity',
  vehicle: 'entity', weapon: 'entity', identity: 'entity',
  event: 'event', evidence: 'evidence',
};

const FILTERS = [
  { key: 'all', label: 'All' },
  { key: 'entity', label: 'Entity' },
  { key: 'event', label: 'Event' },
  { key: 'evidence', label: 'Evidence' },
];

const EDGE_STYLE = {
  co_occurs_with: { stroke: 'rgba(255,255,255,0.28)', dash: null, label: 'co-occurs with' },
  identity_link: { stroke: '#f472b6', dash: '3,3', label: 'identity link' },
  belongs_to: { stroke: '#facc15', dash: '2,3', label: 'belongs to' },
  contradicts: { stroke: '#f87171', dash: null, label: 'contradicts' },
  corroborates: { stroke: '#00e5a0', dash: null, label: 'corroborates' },
};

const MINI_W = 168;
const MINI_H = 112;

function nodeColor(type) {
  return NODE_COLORS[type] || NODE_COLORS.other;
}

function nodeRadius(node) {
  if (node.type === 'evidence') return 6;
  if (node.type === 'event') return 8;
  return Math.min(6 + Math.sqrt(node.weight || 1) * 2.5, 22);
}

export default function GraphView({ activeCollection, onOpen, onJobStarted }) {
  const [graph, setGraph] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selected, setSelected] = useState(null);
  const [previewError, setPreviewError] = useState(null);
  const [search, setSearch] = useState('');
  const [filterType, setFilterType] = useState('all');
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [building, setBuilding] = useState(false);
  const [buildStatus, setBuildStatus] = useState(null);
  const [buildError, setBuildError] = useState(null);

  const svgRef = useRef(null);
  const containerRef = useRef(null);
  const miniRef = useRef(null);
  const buildGuardRef = useRef(null);

  // Mutable handles the toolbar and the search-highlight effect read: these
  // change every tick or every rebuild, and neither should cause a React
  // re-render — nothing on screen needs to react to them except through the
  // d3 selections themselves.
  const zoomBehaviorRef = useRef(null);
  const nodeSelRef = useRef(null);
  const linkSelRef = useRef(null);
  const nodesDataRef = useRef([]);
  const transformRef = useRef(zoomIdentity);
  const miniNodeSelRef = useRef(null);
  const miniViewportSelRef = useRef(null);

  const load = useCallback(async () => {
    if (!activeCollection) return;
    setLoading(true);
    setError(null);
    setSelected(null);
    setSearch('');
    setFilterType('all');
    try {
      const data = await api.graph(activeCollection.id);
      setGraph(data);
      setLastUpdated(new Date());
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [activeCollection]);

  useEffect(() => { load(); }, [load]);

  // Cancels an in-flight build-graph poll when the collection actually
  // changes from under it (switching tabs mid-poll) or the view unmounts —
  // not when the poll's own success path calls `load()` to pick up what it
  // just built, which must be allowed to complete normally.
  useEffect(() => () => {
    if (buildGuardRef.current) buildGuardRef.current.cancelled = true;
  }, [activeCollection]);

  useEffect(() => {
    const onChange = () => setIsFullscreen(document.fullscreenElement === containerRef.current);
    document.addEventListener('fullscreenchange', onChange);
    return () => document.removeEventListener('fullscreenchange', onChange);
  }, []);

  // Runs graph construction (entities, identities, timeline events,
  // contradictions) for a collection that has already been ingested and
  // enriched but never got this phase — it failed, ollama was down, or the
  // case predates the graph feature. Polls the same job the upload pipeline
  // reports through, so progress shows in the activity rail too, then
  // reloads once it finishes.
  const buildGraph = useCallback(async () => {
    if (!activeCollection) return;
    const guard = { cancelled: false };
    buildGuardRef.current = guard;
    setBuilding(true);
    setBuildStatus('Starting…');
    setBuildError(null);
    try {
      const { job_id } = await api.buildGraph(activeCollection.id);
      onJobStarted?.(job_id);
      while (!guard.cancelled) {
        await new Promise((resolve) => { setTimeout(resolve, 2000); });
        if (guard.cancelled) break;
        const job = await api.job(job_id);
        if (guard.cancelled) break;
        setBuildStatus(job.detail || job.stage);
        if (job.status === 'done') {
          await load();
          break;
        }
        if (job.status === 'failed') {
          setBuildError(job.error || 'Graph build failed');
          break;
        }
      }
    } catch (e) {
      if (!guard.cancelled) setBuildError(e.message);
    } finally {
      if (!guard.cancelled) {
        setBuilding(false);
        setBuildStatus(null);
      }
    }
  }, [activeCollection, onJobStarted, load]);

  // Only "evidence" nodes are backed by one actual file/frame/page — an
  // entity or identity can be mentioned across dozens of nodes, so there is
  // no single thing to preview. Reuses the same DetailModal every other view
  // opens, so a node's preview is exactly what upload → search → detail
  // already renders for that piece of evidence, not a second bespoke viewer.
  const selectNode = useCallback((d) => {
    setSelected(d);
    setPreviewError(null);
    if (d.type !== 'evidence' || !d.node_id || !onOpen) return;
    api.node(d.node_id)
      .then((detail) => onOpen({ ...detail, node_id: detail.id }))
      .catch((e) => setPreviewError(e.message));
  }, [onOpen]);

  const filteredGraph = useMemo(() => {
    if (!graph) return null;
    const nodes = filterType === 'all'
      ? graph.nodes
      : graph.nodes.filter((n) => CATEGORY_OF[n.type] === filterType);
    const ids = new Set(nodes.map((n) => n.id));
    const edges = graph.edges.filter((e) => ids.has(e.source) && ids.has(e.target));
    return { nodes, edges };
  }, [graph, filterType]);

  // The simulation drives the DOM directly through d3 selections rather than
  // React state: a few hundred nodes ticking through React on every frame
  // would repaint the whole tree dozens of times a second for no benefit —
  // nothing here needs to be reactive except which node is selected.
  useEffect(() => {
    if (!filteredGraph || !svgRef.current || !containerRef.current) return;

    const svg = select(svgRef.current);
    svg.selectAll('*').remove();
    if (miniRef.current) select(miniRef.current).selectAll('*').remove();
    if (filteredGraph.nodes.length === 0) return;

    const width = containerRef.current.clientWidth || 800;
    const height = containerRef.current.clientHeight || 600;

    const byId = new Map(filteredGraph.nodes.map((n) => [n.id, { ...n }]));
    const links = filteredGraph.edges.map((e) => ({ ...e }));
    const nodes = Array.from(byId.values());
    nodesDataRef.current = nodes;

    const root = svg.append('g').attr('class', 'graph-root');
    const linkLayer = root.append('g').attr('class', 'graph-links');
    const nodeLayer = root.append('g').attr('class', 'graph-nodes');

    const linkSel = linkLayer.selectAll('line')
      .data(links)
      .join('line')
      .attr('stroke', (d) => (EDGE_STYLE[d.type] || EDGE_STYLE.co_occurs_with).stroke)
      .attr('stroke-dasharray', (d) => (EDGE_STYLE[d.type] || {}).dash || null)
      .attr('stroke-width', (d) => Math.min(1 + Math.sqrt(d.weight || 1), 5));
    linkSelRef.current = linkSel;

    const nodeSel = nodeLayer.selectAll('g.graph-node')
      .data(nodes)
      .join('g')
      .attr('class', 'graph-node')
      .style('cursor', 'pointer')
      .on('click', (_event, d) => selectNode(d));
    nodeSelRef.current = nodeSel;

    nodeSel.append('circle')
      .attr('r', nodeRadius)
      .attr('fill', (d) => nodeColor(d.type))
      .attr('fill-opacity', (d) => (d.type === 'evidence' ? 0.5 : 0.85))
      .attr('stroke', 'rgba(255,255,255,0.35)')
      .attr('stroke-width', 1);

    nodeSel.append('text')
      .text((d) => (d.label.length > 22 ? `${d.label.slice(0, 21)}…` : d.label))
      .attr('x', (d) => nodeRadius(d) + 5)
      .attr('y', 4)
      .attr('font-size', 11)
      .attr('fill', 'var(--text-dim)')
      .style('pointer-events', 'none');

    // Minimap: a scaled-down copy of the same node positions, plus a
    // rectangle tracing the main canvas's current zoom/pan.
    const miniSvg = select(miniRef.current);
    const miniDots = miniSvg.append('g').selectAll('circle')
      .data(nodes)
      .join('circle')
      .attr('r', 1.5)
      .attr('fill', (d) => nodeColor(d.type));
    miniNodeSelRef.current = miniDots;
    miniViewportSelRef.current = miniSvg.append('rect')
      .attr('fill', 'none')
      .attr('stroke', 'var(--accent)')
      .attr('stroke-width', 1);

    const updateMinimap = () => {
      const xs = nodes.map((n) => n.x ?? width / 2);
      const ys = nodes.map((n) => n.y ?? height / 2);
      const minX = Math.min(...xs);
      const maxX = Math.max(...xs);
      const minY = Math.min(...ys);
      const maxY = Math.max(...ys);
      const spanX = Math.max(maxX - minX, 1);
      const spanY = Math.max(maxY - minY, 1);
      const scale = Math.min((MINI_W - 12) / spanX, (MINI_H - 12) / spanY);
      const ox = 6 - minX * scale;
      const oy = 6 - minY * scale;

      miniNodeSelRef.current
        ?.attr('cx', (d) => (d.x ?? width / 2) * scale + ox)
        .attr('cy', (d) => (d.y ?? height / 2) * scale + oy);

      const t = transformRef.current;
      miniViewportSelRef.current
        ?.attr('x', (-t.x / t.k) * scale + ox)
        .attr('y', (-t.y / t.k) * scale + oy)
        .attr('width', Math.min((width / t.k) * scale, MINI_W))
        .attr('height', Math.min((height / t.k) * scale, MINI_H));
    };

    const simulation = forceSimulation(nodes)
      .force('link', forceLink(links).id((d) => d.id).distance(70).strength(0.4))
      .force('charge', forceManyBody().strength(-160))
      .force('center', forceCenter(width / 2, height / 2))
      .force('collide', forceCollide().radius((d) => nodeRadius(d) + 14))
      // `forceCenter` only re-centers the whole layout's average position; it
      // does nothing to stop disconnected clusters (an entity graph is rarely
      // one connected component) from drifting apart from each other under
      // mutual charge repulsion with no link pulling them back. A gentle pull
      // on every node individually is what actually makes the layout settle.
      .force('x', forceX(width / 2).strength(0.03))
      .force('y', forceY(height / 2).strength(0.03));

    simulation.on('tick', () => {
      linkSel
        .attr('x1', (d) => d.source.x)
        .attr('y1', (d) => d.source.y)
        .attr('x2', (d) => d.target.x)
        .attr('y2', (d) => d.target.y);
      nodeSel.attr('transform', (d) => `translate(${d.x},${d.y})`);
      updateMinimap();
    });

    nodeSel.call(
      drag()
        .on('start', (event, d) => {
          if (!event.active) simulation.alphaTarget(0.2).restart();
          d.fx = d.x;
          d.fy = d.y;
        })
        .on('drag', (event, d) => {
          d.fx = event.x;
          d.fy = event.y;
        })
        .on('end', (event, d) => {
          if (!event.active) simulation.alphaTarget(0);
          d.fx = null;
          d.fy = null;
        }),
    );

    const zoomBehavior = zoom()
      .scaleExtent([0.15, 4])
      .on('zoom', (event) => {
        root.attr('transform', event.transform);
        transformRef.current = event.transform;
        updateMinimap();
      });
    zoomBehaviorRef.current = zoomBehavior;
    svg.call(zoomBehavior);
    svg.call(zoomBehavior.transform, zoomIdentity);
    transformRef.current = zoomIdentity;

    return () => simulation.stop();
  }, [filteredGraph, selectNode]);

  // Highlighting a search match is a pure style toggle on the selections the
  // effect above already built — it must not restart the simulation, or
  // every keystroke would re-scatter the whole layout.
  useEffect(() => {
    const term = search.trim().toLowerCase();
    nodeSelRef.current?.style('opacity', (d) => (
      !term || d.label.toLowerCase().includes(term) ? 1 : 0.15
    ));
    linkSelRef.current?.style('opacity', (d) => {
      if (!term) return 1;
      const matches = (n) => n && typeof n === 'object' && n.label.toLowerCase().includes(term);
      return matches(d.source) || matches(d.target) ? 1 : 0.08;
    });
  }, [search, filteredGraph]);

  const zoomBy = (factor) => {
    if (!svgRef.current || !zoomBehaviorRef.current) return;
    select(svgRef.current).transition().duration(200).call(zoomBehaviorRef.current.scaleBy, factor);
  };

  const fitToView = () => {
    const nodes = nodesDataRef.current;
    if (!svgRef.current || !zoomBehaviorRef.current || !containerRef.current || !nodes.length) return;
    const width = containerRef.current.clientWidth;
    const height = containerRef.current.clientHeight;
    const xs = nodes.map((n) => n.x ?? width / 2);
    const ys = nodes.map((n) => n.y ?? height / 2);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const scale = Math.min(
      (0.9 * width) / Math.max(maxX - minX, 1),
      (0.9 * height) / Math.max(maxY - minY, 1),
      4,
    );
    const tx = width / 2 - scale * (minX + maxX) / 2;
    const ty = height / 2 - scale * (minY + maxY) / 2;
    select(svgRef.current).transition().duration(300)
      .call(zoomBehaviorRef.current.transform, zoomIdentity.translate(tx, ty).scale(scale));
  };

  const toggleFullscreen = () => {
    if (!containerRef.current) return;
    if (document.fullscreenElement) document.exitFullscreen();
    else containerRef.current.requestFullscreen();
  };

  if (!activeCollection) {
    return (
      <div className="state">
        <div className="state-icon"><ShareNetwork size={24} /></div>
        <h3>Select a collection</h3>
        <p>Choose a collection from the dropdown to see its knowledge graph.</p>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="state">
        <div className="state-icon"><Loader2 size={24} className="spin" /></div>
        <h3>Loading knowledge graph…</h3>
      </div>
    );
  }

  if (error) {
    return (
      <div className="state error">
        <div className="state-icon"><ShareNetwork size={24} /></div>
        <h3>Couldn't load the graph</h3>
        <p>{error}</p>
      </div>
    );
  }

  if (!graph || graph.nodes.length === 0) {
    return (
      <div className="state">
        <div className="state-icon">
          {building ? <Loader2 size={24} className="spin" /> : <ShareNetwork size={24} />}
        </div>
        <h3>{building ? 'Building the knowledge graph…' : 'Nothing to graph yet'}</h3>
        <p>
          {building
            ? (buildStatus || 'This runs entity extraction, timeline grouping, and contradiction detection — it can take a few minutes.')
            : 'No entities, identities, timeline events, or contradiction links have been built for this collection yet.'}
        </p>
        {!building && (
          <button className="btn sm" onClick={buildGraph} style={{ marginTop: '0.75rem' }}>
            <ShareNetwork size={14} /> Build knowledge graph
          </button>
        )}
        {buildError && <p className="text-muted" style={{ marginTop: '0.5rem' }}>{buildError}</p>}
      </div>
    );
  }

  return (
    <div className="graph-view">
      <div className="graph-toolbar">
        <div className="graph-search">
          <MagnifyingGlass size={14} />
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search nodes…"
          />
        </div>

        <div className="graph-filters">
          {FILTERS.map((f) => (
            <button
              key={f.key}
              className={`graph-filter-btn ${filterType === f.key ? 'active' : ''}`}
              onClick={() => setFilterType(f.key)}
            >
              {f.label}
            </button>
          ))}
        </div>

        <div className="graph-zoom-controls">
          <button title="Zoom out" onClick={() => zoomBy(1 / 1.4)}><Minus size={14} /></button>
          <button title="Fit to view" onClick={fitToView}><Crosshair size={14} /></button>
          <button title="Zoom in" onClick={() => zoomBy(1.4)}><Plus size={14} /></button>
          <button title={isFullscreen ? 'Exit fullscreen' : 'Fullscreen'} onClick={toggleFullscreen}>
            {isFullscreen ? <ArrowsIn size={14} /> : <ArrowsOut size={14} />}
          </button>
        </div>

        <div className="graph-stats">
          <span><b>{filteredGraph?.nodes.length ?? 0}</b> nodes</span>
          <span><b>{filteredGraph?.edges.length ?? 0}</b> edges</span>
          {lastUpdated && <span className="text-muted">updated {lastUpdated.toLocaleTimeString()}</span>}
        </div>
      </div>

      <div className="graph-body">
        <div className="graph-canvas" ref={containerRef}>
          <svg ref={svgRef} width="100%" height="100%" />

          <div className="graph-legend graph-legend-nodes">
            {Object.entries(NODE_COLORS).filter(([k]) => k !== 'other').map(([type, color]) => (
              <span key={type} className="graph-legend-item">
                <i style={{ background: color }} />{type.replace('_', ' ')}
              </span>
            ))}
          </div>

          <div className="graph-legend graph-legend-edges">
            {Object.entries(EDGE_STYLE).map(([type, style]) => (
              <span key={type} className="graph-legend-item">
                <i className="graph-legend-line" style={{
                  background: style.dash
                    ? `repeating-linear-gradient(90deg, ${style.stroke} 0 3px, transparent 3px 6px)`
                    : style.stroke,
                }} />
                {style.label}
              </span>
            ))}
          </div>

          <svg ref={miniRef} className="graph-minimap" width={MINI_W} height={MINI_H} />
        </div>

        <aside className="graph-detail panel">
          <div className="panel-head"><h3>Node</h3></div>
          {selected ? (
            <div className="graph-detail-body">
              <div className="graph-detail-label" style={{ color: nodeColor(selected.type) }}>
                {selected.type}
              </div>
              <h4>{selected.label}</h4>
              {selected.detail && <p>{selected.detail}</p>}
              {selected.type !== 'evidence' && selected.type !== 'event' && (
                <p className="text-muted">{selected.weight} mention(s)</p>
              )}
              {previewError && <p className="text-muted">Couldn't load a preview: {previewError}</p>}
            </div>
          ) : (
            <p className="rail-empty">
              Click a node to see its details — an evidence node opens its preview.
              Drag to reposition, scroll to zoom.
            </p>
          )}
        </aside>
      </div>
    </div>
  );
}
