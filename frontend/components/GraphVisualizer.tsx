import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import dynamic from 'next/dynamic';

const ForceGraph2D = dynamic(() => import('react-force-graph-2d'), { ssr: false });

import {
  ZoomIn, ZoomOut, Maximize2, RefreshCw, Database, Search,
  Download, X, ChevronDown, ChevronUp, Eye, EyeOff, Info,
} from 'lucide-react';
import { api } from '@/lib/api';

// ─────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────

const SOURCE_PALETTE = [
  '#60a5fa', '#f87171', '#34d399', '#c084fc', '#fb923c',
  '#22d3ee', '#facc15', '#f472b6', '#a3e635', '#818cf8',
  '#2dd4bf', '#e879f9',
];

// ─────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────

interface GraphVisualizerProps {
  onClose?: () => void;
  isFullscreen?: boolean;
}

// ─────────────────────────────────────────────────
// Component
// ─────────────────────────────────────────────────

export function GraphVisualizer({ onClose, isFullscreen = false }: GraphVisualizerProps) {
  // === Core State ===
  const [rawData, setRawData] = useState<{ nodes: any[]; links: any[] }>({ nodes: [], links: [] });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });

  // === Feature State ===
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedNode, setSelectedNode] = useState<any | null>(null);
  const [hoveredNode, setHoveredNode] = useState<any | null>(null);
  const [activeSources, setActiveSources] = useState<Set<string>>(new Set());
  const [activeEdgeTypes, setActiveEdgeTypes] = useState<Set<string>>(new Set());
  const [showPanel, setShowPanel] = useState(true);
  const [showEdgeFilters, setShowEdgeFilters] = useState(false);

  // === Refs ===
  const graphRef = useRef<any>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const searchInputRef = useRef<HTMLInputElement>(null);

  // ─── Resize Observer ───────────────────────────
  useEffect(() => {
    if (!containerRef.current) return;
    const observer = new ResizeObserver(entries => {
      const { width, height } = entries[0].contentRect;
      setDimensions({ width, height });
    });
    observer.observe(containerRef.current);
    const rect = containerRef.current.getBoundingClientRect();
    setDimensions({ width: rect.width, height: rect.height });
    return () => observer.disconnect();
  }, []);

  // ─── Fetch Graph (no limits) ───────────────────
  const fetchGraph = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await api.getGraphData();
      setRawData(data);
    } catch (err) {
      console.error('Failed to load graph:', err);
      setError('Failed to load knowledge graph data.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchGraph(); }, [fetchGraph]);

  // ─── Initialise filters when data loads ────────
  useEffect(() => {
    if (rawData.nodes.length === 0) return;
    setActiveSources(new Set(rawData.nodes.map((n: any) => n.source || 'unknown')));
    setActiveEdgeTypes(new Set(rawData.links.map((l: any) => l.label || 'unknown')));
  }, [rawData]);

  // ─── Derived: colour map ───────────────────────
  const sourceColorMap = useMemo(() => {
    const sources = [...new Set(rawData.nodes.map((n: any) => n.source || 'unknown'))];
    const map: Record<string, string> = {};
    sources.forEach((s, i) => { map[s] = SOURCE_PALETTE[i % SOURCE_PALETTE.length]; });
    return map;
  }, [rawData.nodes]);

  // ─── Derived: all edge types ───────────────────
  const allEdgeTypes = useMemo(
    () => [...new Set(rawData.links.map((l: any) => l.label || 'unknown'))],
    [rawData.links],
  );

  // ─── Derived: filtered graph data ──────────────
  const filteredData = useMemo(() => {
    const nodes = rawData.nodes.filter((n: any) => activeSources.has(n.source || 'unknown'));
    const nodeIdSet = new Set(nodes.map((n: any) => n.id));
    const links = rawData.links.filter((l: any) =>
      activeEdgeTypes.has(l.label || 'unknown') &&
      nodeIdSet.has(l.source) &&
      nodeIdSet.has(l.target),
    );
    return { nodes, links };
  }, [rawData, activeSources, activeEdgeTypes]);

  const isLargeGraph = filteredData.links.length > 5000;

  // ─── Derived: node degree map ──────────────────
  const nodeDegreeMap = useMemo(() => {
    const deg: Record<string, number> = {};
    filteredData.links.forEach((l: any) => {
      const s = typeof l.source === 'object' ? l.source.id : l.source;
      const t = typeof l.target === 'object' ? l.target.id : l.target;
      deg[s] = (deg[s] || 0) + 1;
      deg[t] = (deg[t] || 0) + 1;
    });
    return deg;
  }, [filteredData.links]);

  // ─── Derived: search matches ───────────────────
  const searchMatches = useMemo(() => {
    if (!searchQuery.trim()) return new Set<string>();
    const q = searchQuery.toLowerCase();
    return new Set(
      filteredData.nodes
        .filter((n: any) =>
          (n.name || '').toLowerCase().includes(q) ||
          (n.label || '').toLowerCase().includes(q),
        )
        .map((n: any) => n.id),
    );
  }, [searchQuery, filteredData.nodes]);

  // ─── Derived: hover neighbourhood ──────────────
  const { highlightNodes, highlightLinks } = useMemo(() => {
    const hNodes = new Set<string>();
    const hLinks = new Set<string>();
    if (!hoveredNode) return { highlightNodes: hNodes, highlightLinks: hLinks };

    hNodes.add(hoveredNode.id);
    filteredData.links.forEach((l: any) => {
      const s = typeof l.source === 'object' ? l.source.id : l.source;
      const t = typeof l.target === 'object' ? l.target.id : l.target;
      if (s === hoveredNode.id || t === hoveredNode.id) {
        hNodes.add(s);
        hNodes.add(t);
        hLinks.add(l.id);
      }
    });
    return { highlightNodes: hNodes, highlightLinks: hLinks };
  }, [hoveredNode, filteredData.links]);

  // ─── Derived: selected node connections ────────
  const selectedConnections = useMemo(() => {
    if (!selectedNode) return [];
    const conns: Array<{ node: any; edgeType: string; weight: number }> = [];
    filteredData.links.forEach((l: any) => {
      const s = typeof l.source === 'object' ? l.source.id : l.source;
      const t = typeof l.target === 'object' ? l.target.id : l.target;
      if (s !== selectedNode.id && t !== selectedNode.id) return;
      const otherId = s === selectedNode.id ? t : s;
      const otherNode = filteredData.nodes.find((n: any) => n.id === otherId);
      if (otherNode) conns.push({ node: otherNode, edgeType: l.label, weight: l.weight || 0 });
    });
    return conns.sort((a, b) => b.weight - a.weight).slice(0, 25);
  }, [selectedNode, filteredData]);

  // ─── Handlers ──────────────────────────────────

  const handleZoomIn = useCallback(() => {
    graphRef.current?.zoom(graphRef.current.zoom() * 1.5, 400);
  }, []);

  const handleZoomOut = useCallback(() => {
    graphRef.current?.zoom(graphRef.current.zoom() / 1.5, 400);
  }, []);

  const handleCenter = useCallback(() => {
    graphRef.current?.zoomToFit(400, 50);
  }, []);

  const handleExportPNG = useCallback(() => {
    const canvas = containerRef.current?.querySelector('canvas');
    if (!canvas) return;
    const a = document.createElement('a');
    a.download = 'knowledge-graph.png';
    a.href = canvas.toDataURL('image/png');
    a.click();
  }, []);

  const toggleSource = useCallback((source: string) => {
    setActiveSources(prev => {
      const next = new Set(prev);
      if (next.has(source)) next.delete(source); else next.add(source);
      return next;
    });
  }, []);

  const toggleEdgeType = useCallback((et: string) => {
    setActiveEdgeTypes(prev => {
      const next = new Set(prev);
      if (next.has(et)) next.delete(et); else next.add(et);
      return next;
    });
  }, []);

  const handleNodeClick = useCallback((node: any) => {
    setSelectedNode(node);
    graphRef.current?.centerAt(node.x, node.y, 500);
    graphRef.current?.zoom(3, 500);
  }, []);

  const handleNodeHover = useCallback((node: any | null) => {
    setHoveredNode(node || null);
  }, []);

  const focusSearchResult = useCallback(() => {
    if (searchMatches.size === 0) return;
    const firstId = [...searchMatches][0];
    const node = filteredData.nodes.find((n: any) => n.id === firstId);
    if (node?.x !== undefined) {
      graphRef.current?.centerAt(node.x, node.y, 500);
      graphRef.current?.zoom(3, 500);
    }
  }, [searchMatches, filteredData.nodes]);

  // ─── Node sizing ───────────────────────────────
  const getNodeSize = useCallback((node: any) => {
    const degree = nodeDegreeMap[node.id] || 0;
    return Math.max(3, Math.min(14, 3 + Math.sqrt(degree) * 1.2));
  }, [nodeDegreeMap]);

  const avgDegree = filteredData.nodes.length > 0
    ? ((2 * filteredData.links.length) / filteredData.nodes.length).toFixed(1)
    : '0';

  // ─────────────────────────────────────────────────
  // Render
  // ─────────────────────────────────────────────────

  return (
    <div
      ref={containerRef}
      className="relative w-full h-full flex bg-[#0a0a0a] overflow-hidden rounded-xl border border-white/10"
    >
      {/* ═══════════ LEFT PANEL ═══════════ */}
      {showPanel && (
        <div
          className="absolute top-4 left-4 z-10 w-[220px] bg-black/70 backdrop-blur-xl border border-white/10 rounded-2xl shadow-2xl flex flex-col overflow-hidden"
          style={{ maxHeight: 'calc(100% - 32px)' }}
        >
          {/* Stats */}
          <div className="p-3 border-b border-white/10">
            <div className="flex items-center gap-2 mb-2">
              <Database className="w-4 h-4 text-emerald-400" />
              <span className="text-xs font-bold tracking-widest uppercase text-white">
                Knowledge Graph
              </span>
            </div>
            <div className="grid grid-cols-3 gap-1 text-center">
              <div>
                <div className="text-[10px] text-neutral-500 font-bold uppercase">Nodes</div>
                <div className="text-sm font-black text-emerald-400">
                  {filteredData.nodes.length}
                </div>
              </div>
              <div>
                <div className="text-[10px] text-neutral-500 font-bold uppercase">Edges</div>
                <div className="text-sm font-black text-blue-400">
                  {filteredData.links.length.toLocaleString()}
                </div>
              </div>
              <div>
                <div className="text-[10px] text-neutral-500 font-bold uppercase">Avg Deg</div>
                <div className="text-sm font-black text-purple-400">{avgDegree}</div>
              </div>
            </div>
          </div>

          {/* Search */}
          <div className="p-2 border-b border-white/10">
            <div className="relative">
              <Search className="absolute left-2 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-neutral-500" />
              <input
                ref={searchInputRef}
                type="text"
                value={searchQuery}
                onChange={e => setSearchQuery(e.target.value)}
                onKeyDown={e => { if (e.key === 'Enter') focusSearchResult(); }}
                placeholder="Search nodes..."
                className="w-full bg-white/5 border border-white/10 rounded-lg pl-8 pr-8 py-1.5 text-xs text-white placeholder-neutral-600 focus:outline-none focus:border-emerald-500/50"
              />
              {searchQuery && (
                <button
                  onClick={() => setSearchQuery('')}
                  className="absolute right-2 top-1/2 -translate-y-1/2"
                >
                  <X className="w-3 h-3 text-neutral-500 hover:text-white" />
                </button>
              )}
            </div>
            {searchQuery && (
              <div className="text-[10px] text-neutral-500 mt-1 px-1">
                {searchMatches.size} match{searchMatches.size !== 1 ? 'es' : ''} — Enter to
                focus
              </div>
            )}
          </div>

          {/* Scrollable filter area */}
          <div className="flex-1 overflow-y-auto">
            {/* Source Filters */}
            <div className="p-2 border-b border-white/10">
              <div className="flex items-center justify-between mb-1">
                <span className="text-[10px] text-neutral-500 font-bold uppercase tracking-widest">
                  Sources
                </span>
                <div className="flex gap-1">
                  <button
                    onClick={() => setActiveSources(new Set(Object.keys(sourceColorMap)))}
                    className="text-[9px] text-neutral-500 hover:text-emerald-400 px-1"
                  >
                    All
                  </button>
                  <button
                    onClick={() => setActiveSources(new Set())}
                    className="text-[9px] text-neutral-500 hover:text-red-400 px-1"
                  >
                    None
                  </button>
                </div>
              </div>
              {Object.entries(sourceColorMap).map(([src, color]) => {
                const active = activeSources.has(src);
                return (
                  <button
                    key={src}
                    onClick={() => toggleSource(src)}
                    className="flex items-center gap-2 w-full px-1.5 py-1 rounded-md hover:bg-white/5 transition-colors"
                    title={src}
                  >
                    <div
                      className="w-2.5 h-2.5 rounded-full flex-shrink-0 transition-opacity"
                      style={{
                        background: color,
                        boxShadow: active ? `0 0 6px ${color}66` : 'none',
                        opacity: active ? 1 : 0.2,
                      }}
                    />
                    <span
                      className={`text-[11px] truncate transition-opacity ${
                        active ? 'text-neutral-300' : 'text-neutral-600 line-through'
                      }`}
                    >
                      {src.length > 24 ? src.slice(0, 22) + '...' : src}
                    </span>
                  </button>
                );
              })}
            </div>

            {/* Edge Type Filters */}
            <div className="p-2">
              <button
                onClick={() => setShowEdgeFilters(v => !v)}
                className="flex items-center justify-between w-full mb-1"
              >
                <span className="text-[10px] text-neutral-500 font-bold uppercase tracking-widest">
                  Edge Types
                </span>
                {showEdgeFilters ? (
                  <ChevronUp className="w-3 h-3 text-neutral-500" />
                ) : (
                  <ChevronDown className="w-3 h-3 text-neutral-500" />
                )}
              </button>

              {showEdgeFilters && (
                <>
                  <div className="flex gap-1 mb-1">
                    <button
                      onClick={() => setActiveEdgeTypes(new Set(allEdgeTypes))}
                      className="text-[9px] text-neutral-500 hover:text-emerald-400 px-1"
                    >
                      All
                    </button>
                    <button
                      onClick={() => setActiveEdgeTypes(new Set())}
                      className="text-[9px] text-neutral-500 hover:text-red-400 px-1"
                    >
                      None
                    </button>
                  </div>
                  {allEdgeTypes.map(et => {
                    const active = activeEdgeTypes.has(et);
                    return (
                      <button
                        key={et}
                        onClick={() => toggleEdgeType(et)}
                        className="flex items-center gap-2 w-full px-1.5 py-1 rounded-md hover:bg-white/5 transition-colors"
                      >
                        <div
                          className={`w-2 h-2 rounded-sm border transition-colors ${
                            active
                              ? 'bg-blue-500 border-blue-400'
                              : 'bg-transparent border-neutral-600'
                          }`}
                        />
                        <span
                          className={`text-[11px] ${
                            active ? 'text-neutral-300' : 'text-neutral-600 line-through'
                          }`}
                        >
                          {et.replace(/_/g, ' ')}
                        </span>
                      </button>
                    );
                  })}
                </>
              )}
            </div>
          </div>
        </div>
      )}

      {/* ═══════════ RIGHT PANEL: Node Details ═══════════ */}
      {selectedNode && (
        <div
          className="absolute top-4 right-4 z-10 w-[280px] bg-black/80 backdrop-blur-xl border border-white/10 rounded-2xl shadow-2xl flex flex-col overflow-hidden"
          style={{ maxHeight: 'calc(100% - 32px)' }}
        >
          {/* Header */}
          <div className="p-3 border-b border-white/10 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Info className="w-4 h-4 text-blue-400" />
              <span className="text-xs font-bold text-white uppercase tracking-wider">
                Node Details
              </span>
            </div>
            <button
              onClick={() => setSelectedNode(null)}
              className="text-neutral-500 hover:text-white"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          {/* Body */}
          <div className="flex-1 overflow-y-auto p-3 space-y-3">
            {/* Source */}
            <div>
              <div className="text-[10px] text-neutral-500 uppercase tracking-wider mb-0.5">
                Source
              </div>
              <div className="flex items-center gap-2">
                <div
                  className="w-2.5 h-2.5 rounded-full flex-shrink-0"
                  style={{
                    background:
                      sourceColorMap[selectedNode.source || 'unknown'] || '#94a3b8',
                  }}
                />
                <span
                  className="text-xs text-white font-medium truncate"
                  title={selectedNode.source || selectedNode.name}
                >
                  {(selectedNode.source || selectedNode.name || 'Unknown').slice(0, 40)}
                </span>
              </div>
            </div>

            {/* Content */}
            <div>
              <div className="text-[10px] text-neutral-500 uppercase tracking-wider mb-0.5">
                Content
              </div>
              <div className="text-xs text-neutral-300 leading-relaxed bg-white/5 rounded-lg p-2 max-h-[120px] overflow-y-auto">
                {selectedNode.label || 'No content.'}
              </div>
            </div>

            {/* Meta badges */}
            <div className="flex gap-2">
              <div className="flex-1 bg-white/5 rounded-lg p-2 text-center">
                <div className="text-[9px] text-neutral-500 uppercase">Type</div>
                <div className="text-[11px] text-white font-bold">
                  {selectedNode.group || 'text'}
                </div>
              </div>
              <div className="flex-1 bg-white/5 rounded-lg p-2 text-center">
                <div className="text-[9px] text-neutral-500 uppercase">Resolution</div>
                <div className="text-[11px] text-white font-bold">
                  {selectedNode.resolution || 'unknown'}
                </div>
              </div>
              <div className="flex-1 bg-white/5 rounded-lg p-2 text-center">
                <div className="text-[9px] text-neutral-500 uppercase">Degree</div>
                <div className="text-[11px] text-emerald-400 font-bold">
                  {nodeDegreeMap[selectedNode.id] || 0}
                </div>
              </div>
            </div>

            {/* Connections list */}
            {selectedConnections.length > 0 && (
              <div>
                <div className="text-[10px] text-neutral-500 uppercase tracking-wider mb-1">
                  Top Connections ({selectedConnections.length})
                </div>
                <div className="space-y-0.5">
                  {selectedConnections.map((conn, i) => (
                    <button
                      key={i}
                      onClick={() => {
                        setSelectedNode(conn.node);
                        if (conn.node.x !== undefined)
                          graphRef.current?.centerAt(conn.node.x, conn.node.y, 500);
                      }}
                      className="w-full flex items-center gap-2 px-2 py-1 rounded-md hover:bg-white/10 transition-colors text-left"
                    >
                      <div
                        className="w-2 h-2 rounded-full flex-shrink-0"
                        style={{
                          background:
                            sourceColorMap[conn.node.source || 'unknown'] || '#94a3b8',
                        }}
                      />
                      <div className="flex-1 min-w-0">
                        <div className="text-[10px] text-neutral-400 truncate">
                          {(conn.node.label || conn.node.name || 'Node').slice(0, 40)}
                        </div>
                      </div>
                      <span className="text-[9px] text-neutral-600 flex-shrink-0">
                        {conn.edgeType?.replace(/_/g, ' ')}
                      </span>
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ═══════════ BOTTOM CONTROLS ═══════════ */}
      <div className="absolute bottom-4 right-4 z-10 flex items-center gap-1 p-1 bg-black/60 backdrop-blur-md border border-white/10 rounded-xl shadow-2xl">
        <button
          onClick={handleZoomIn}
          className="p-2 hover:bg-white/10 rounded-lg text-neutral-400 hover:text-white transition-colors"
          title="Zoom In"
        >
          <ZoomIn className="w-4 h-4" />
        </button>
        <button
          onClick={handleZoomOut}
          className="p-2 hover:bg-white/10 rounded-lg text-neutral-400 hover:text-white transition-colors"
          title="Zoom Out"
        >
          <ZoomOut className="w-4 h-4" />
        </button>
        <div className="w-px h-4 bg-white/10" />
        <button
          onClick={handleCenter}
          className="p-2 hover:bg-white/10 rounded-lg text-neutral-400 hover:text-white transition-colors"
          title="Fit to View"
        >
          <Maximize2 className="w-4 h-4" />
        </button>
        <div className="w-px h-4 bg-white/10" />
        <button
          onClick={handleExportPNG}
          className="p-2 hover:bg-white/10 rounded-lg text-neutral-400 hover:text-white transition-colors"
          title="Export as PNG"
        >
          <Download className="w-4 h-4" />
        </button>
        <div className="w-px h-4 bg-white/10" />
        <button
          onClick={() => setShowPanel(p => !p)}
          className="p-2 hover:bg-white/10 rounded-lg text-neutral-400 hover:text-white transition-colors"
          title={showPanel ? 'Hide Panel' : 'Show Panel'}
        >
          {showPanel ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
        </button>
        <div className="w-px h-4 bg-white/10" />
        <button
          onClick={fetchGraph}
          className="p-2 hover:bg-white/10 rounded-lg text-neutral-400 hover:text-white transition-colors"
          title="Reload Graph"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {/* ═══════════ LOADING ═══════════ */}
      {loading && (
        <div className="absolute inset-0 z-20 flex flex-col items-center justify-center bg-black/40 backdrop-blur-xl">
          <RefreshCw className="w-8 h-8 text-neutral-400 animate-spin mb-4" />
          <p className="text-sm font-bold tracking-widest uppercase text-neutral-400 animate-pulse">
            Computing Force Physics Engine...
          </p>
        </div>
      )}

      {/* ═══════════ ERROR ═══════════ */}
      {error && !loading && (
        <div className="absolute inset-0 z-20 flex flex-col items-center justify-center bg-[#0a0a0a]">
          <p className="text-red-400 text-sm">{error}</p>
          <button
            onClick={fetchGraph}
            className="mt-4 px-4 py-2 bg-white/5 hover:bg-white/10 rounded-lg text-sm text-white transition-colors border border-white/10"
          >
            Try Again
          </button>
        </div>
      )}

      {/* ═══════════ GRAPH CANVAS ═══════════ */}
      <div className="flex-1 w-full h-full cursor-grab active:cursor-grabbing">
        {!loading && !error && dimensions.width > 0 && dimensions.height > 0 && (
          <ForceGraph2D
            ref={graphRef}
            width={dimensions.width}
            height={dimensions.height}
            graphData={filteredData}
            /* ── Performance tuning ── */
            cooldownTicks={150}
            warmupTicks={isLargeGraph ? 50 : 0}
            d3AlphaDecay={0.02}
            d3VelocityDecay={0.3}
            /* ── Tooltips ── */
            nodeLabel={(node: any) => `
              <div style="background:rgba(10,10,10,.95);padding:12px;border-radius:8px;border:1px solid rgba(255,255,255,.1);max-width:320px;color:#fff;box-shadow:0 10px 25px rgba(0,0,0,.5)">
                <div style="font-weight:700;font-size:13px;color:${sourceColorMap[node.source || 'unknown'] || '#94a3b8'};margin-bottom:6px;padding-bottom:6px;border-bottom:1px solid rgba(255,255,255,.1);text-transform:uppercase;letter-spacing:.5px">
                  ${node.name || 'Knowledge Node'}
                </div>
                <div style="font-size:13px;line-height:1.5;word-wrap:break-word;white-space:pre-wrap;color:rgba(255,255,255,.85)">
                  ${node.label || 'No content available.'}
                </div>
              </div>
            `}
            linkLabel={(link: any) =>
              `<div style="background:rgba(10,10,10,.95);padding:8px;border-radius:6px;border:1px solid rgba(255,255,255,.1);color:#fff;font-size:12px;font-weight:bold;text-transform:uppercase">${link.label || 'Connected'}</div>`
            }
            /* ── Link styling ── */
            linkColor={(link: any) => {
              if (hoveredNode && highlightLinks.size > 0)
                return highlightLinks.has(link.id)
                  ? 'rgba(255,255,255,0.4)'
                  : 'rgba(255,255,255,0.02)';
              return 'rgba(255,255,255,0.08)';
            }}
            linkWidth={(link: any) =>
              hoveredNode && highlightLinks.has(link.id) ? 2 : 0.5
            }
            linkDirectionalParticles={isLargeGraph ? 0 : 1}
            linkDirectionalParticleSpeed={0.004}
            linkCanvasObjectMode={isLargeGraph ? undefined : (() => 'after' as const)}
            linkCanvasObject={
              isLargeGraph
                ? undefined
                : (link: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
                    if (globalScale < 3.5 || !link.label) return;
                    const start = link.source;
                    const end = link.target;
                    if (typeof start !== 'object' || typeof end !== 'object') return;
                    const mx = (start.x + end.x) / 2;
                    const my = (start.y + end.y) / 2;
                    const angle = Math.atan2(end.y - start.y, end.x - start.x);
                    ctx.save();
                    ctx.translate(mx, my);
                    ctx.rotate(angle);
                    const fs = 10 / globalScale;
                    ctx.font = `bold ${fs}px Inter,sans-serif`;
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';
                    const tw = ctx.measureText(link.label).width;
                    ctx.fillStyle = 'rgba(0,0,0,0.6)';
                    ctx.fillRect(
                      -tw / 2 - 2 / globalScale,
                      -fs / 2 - 2 / globalScale,
                      tw + 4 / globalScale,
                      fs + 4 / globalScale,
                    );
                    ctx.fillStyle = 'rgba(255,255,255,0.7)';
                    ctx.fillText(link.label, 0, 0);
                    ctx.restore();
                  }
            }
            /* ── Node rendering ── */
            nodeCanvasObject={(
              node: any,
              ctx: CanvasRenderingContext2D,
              globalScale: number,
            ) => {
              const nodeId = node.id;
              const size = getNodeSize(node);
              const color =
                sourceColorMap[node.source || 'unknown'] || '#94a3b8';

              // Decide opacity — dim non-matches during search or hover
              let opacity = 1;
              const isSearchActive =
                searchQuery.trim() !== '' && searchMatches.size > 0;
              const isHoverActive =
                hoveredNode !== null && highlightNodes.size > 0;

              if (isSearchActive && !searchMatches.has(nodeId)) opacity = 0.06;
              else if (isHoverActive && !highlightNodes.has(nodeId)) opacity = 0.06;

              // Glow ring for highlighted / searched nodes
              const isHighlighted =
                (isSearchActive && searchMatches.has(nodeId)) ||
                (isHoverActive && highlightNodes.has(nodeId));
              if (isHighlighted) {
                ctx.beginPath();
                ctx.arc(node.x!, node.y!, size + 4, 0, 2 * Math.PI);
                ctx.fillStyle = color + '33';
                ctx.fill();
              }

              // Main circle
              ctx.globalAlpha = opacity;
              ctx.beginPath();
              ctx.arc(node.x!, node.y!, size, 0, 2 * Math.PI);
              ctx.fillStyle = color;
              ctx.fill();

              // Text label when zoomed in
              if (globalScale > 2) {
                const label = (node.name || 'Node') as string;
                const fs = Math.max(10 / globalScale, 1.5);
                ctx.font = `${fs}px Inter,sans-serif`;
                ctx.textAlign = 'center';
                ctx.textBaseline = 'top';
                const tw = ctx.measureText(label).width;
                const pad = 2 / globalScale;
                ctx.fillStyle = 'rgba(0,0,0,0.8)';
                ctx.fillRect(
                  node.x! - tw / 2 - pad,
                  node.y! + size + pad,
                  tw + pad * 2,
                  fs + pad * 2,
                );
                ctx.fillStyle = 'rgba(255,255,255,0.9)';
                ctx.fillText(label, node.x!, node.y! + size + pad * 2);
              }

              ctx.globalAlpha = 1;
            }}
            nodeRelSize={6}
            /* ── Interactions ── */
            onNodeClick={handleNodeClick}
            onNodeHover={handleNodeHover}
            onBackgroundClick={() => setSelectedNode(null)}
            onEngineStop={() => graphRef.current?.zoomToFit(400, 50)}
          />
        )}
      </div>
    </div>
  );
}
