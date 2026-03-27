import React, { useState, useEffect, useRef, useCallback } from 'react';
import dynamic from 'next/dynamic';

const ForceGraph2D = dynamic(() => import('react-force-graph-2d'), { ssr: false });
import { Maximize2, Minimize2, RefreshCw, ZoomIn, ZoomOut, Database } from 'lucide-react';
import { api } from '@/lib/api';

interface GraphVisualizerProps {
  onClose?: () => void;
  isFullscreen?: boolean;
}

export function GraphVisualizer({ onClose, isFullscreen = false }: GraphVisualizerProps) {
  const [data, setData] = useState<{ nodes: any[]; links: any[] }>({ nodes: [], links: [] });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const graphRef = useRef<any>(null);
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
  const containerRef = useRef<HTMLDivElement>(null);

  // Auto-resize canvas
  useEffect(() => {
    if (!containerRef.current) return;
    const observer = new ResizeObserver(entries => {
      const { width, height } = entries[0].contentRect;
      setDimensions({ width, height });
    });
    observer.observe(containerRef.current);
    
    // Initial size
    const rect = containerRef.current.getBoundingClientRect();
    setDimensions({ width: rect.width, height: rect.height });
    
    return () => observer.disconnect();
  }, []);

  const fetchGraph = async () => {
    try {
      setLoading(true);
      setError(null);
      const graphData = await api.getGraphData(1500);
      setData(graphData);
    } catch (err) {
      console.error('Failed to load graph:', err);
      setError('Failed to load knowledge graph data. Check system logs.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchGraph();
  }, []);

  const handleZoomIn = useCallback(() => {
    if (graphRef.current) {
      const currentZoom = graphRef.current.zoom();
      graphRef.current.zoom(currentZoom * 1.5, 400);
    }
  }, []);

  const handleZoomOut = useCallback(() => {
    if (graphRef.current) {
      const currentZoom = graphRef.current.zoom();
      graphRef.current.zoom(currentZoom / 1.5, 400);
    }
  }, []);

  const handleCenter = useCallback(() => {
    if (graphRef.current) {
      graphRef.current.zoomToFit(400, 50);
    }
  }, []);

  // Node Styling logic
  const getNodeColor = (node: any) => {
    const group = (node.group || '').toLowerCase();
    if (group === 'text') return '#60a5fa'; // blue-400
    if (group === 'image') return '#818cf8'; // indigo-400
    if (group === 'audio') return '#34d399'; // emerald-400
    if (group === 'video') return '#f87171'; // red-400 
    if (group === 'unknown') return '#a3a3a3'; // neutral-400
    return '#facc15'; // yellow-400
  };

  return (
    <div className="relative w-full h-full flex flex-col bg-[#0a0a0a] overflow-hidden rounded-xl border border-white/10" ref={containerRef}>
      
      {/* Floating Toolbar Overlay */}
      <div className="absolute top-4 left-4 z-10 p-3 bg-black/60 backdrop-blur-md border border-white/10 rounded-2xl flex flex-col gap-1 shadow-2xl">
        <div className="flex items-center gap-2 mb-2 px-2 pb-2 border-b border-white/10">
          <Database className="w-4 h-4 text-emerald-400" />
          <span className="text-xs font-bold tracking-widest uppercase text-white">Knowledge Graph</span>
        </div>
        
        <div className="flex flex-col gap-1">
          <div className="flex items-center justify-between text-xs px-2 py-1">
            <span className="text-neutral-500 font-bold uppercase tracking-wider">Nodes</span>
            <span className="text-emerald-400 font-black">{data.nodes.length}</span>
          </div>
          <div className="flex items-center justify-between text-xs px-2 py-1">
            <span className="text-neutral-500 font-bold uppercase tracking-wider">Edges</span>
            <span className="text-blue-400 font-black">{data.links.length}</span>
          </div>
        </div>
      </div>

      <div className="absolute bottom-4 right-4 z-10 flex items-center gap-2 p-1 bg-black/60 backdrop-blur-md border border-white/10 rounded-xl shadow-2xl">
        <button onClick={handleZoomIn} className="p-2 hover:bg-white/10 rounded-lg text-neutral-400 hover:text-white transition-colors" title="Zoom In">
          <ZoomIn className="w-4 h-4" />
        </button>
        <button onClick={handleZoomOut} className="p-2 hover:bg-white/10 rounded-lg text-neutral-400 hover:text-white transition-colors" title="Zoom Out">
          <ZoomOut className="w-4 h-4" />
        </button>
        <div className="w-px h-4 bg-white/10 mx-1" />
        <button onClick={handleCenter} className="p-2 hover:bg-white/10 rounded-lg text-neutral-400 hover:text-white transition-colors" title="Center View">
          <Maximize2 className="w-4 h-4" />
        </button>
        <div className="w-px h-4 bg-white/10 mx-1" />
        <button onClick={fetchGraph} className="p-2 hover:bg-white/10 rounded-lg text-neutral-400 hover:text-white transition-colors" title="Reload Graph">
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {loading && (
        <div className="absolute inset-0 z-20 flex flex-col items-center justify-center bg-black/40 backdrop-blur-xl">
          <RefreshCw className="w-8 h-8 text-neutral-400 animate-spin mb-4" />
          <p className="text-sm font-bold tracking-widest uppercase text-neutral-400 animate-pulse">Computing Force Physics Engine...</p>
        </div>
      )}

      {error && !loading && (
        <div className="absolute inset-0 z-20 flex flex-col items-center justify-center bg-[#0a0a0a]">
          <p className="text-red-400 text-sm">{error}</p>
          <button onClick={fetchGraph} className="mt-4 px-4 py-2 bg-white/5 hover:bg-white/10 rounded-lg text-sm text-white transition-colors border border-white/10">Try Again</button>
        </div>
      )}

      {/* Force Graph Canvas */}
      <div className="flex-1 w-full h-full cursor-grab active:cursor-grabbing">
        {!loading && !error && dimensions.width > 0 && dimensions.height > 0 && (
          <ForceGraph2D
            ref={graphRef}
            width={dimensions.width}
            height={dimensions.height}
            graphData={data}
            nodeLabel={(node: any) => `
              <div style="background: rgba(10,10,10,0.95); padding: 12px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1); max-width: 320px; color: white; box-shadow: 0 10px 25px rgba(0,0,0,0.5);">
                <div style="font-weight: 700; font-size: 13px; color: ${getNodeColor(node)}; margin-bottom: 6px; padding-bottom: 6px; border-bottom: 1px solid rgba(255,255,255,0.1); text-transform: uppercase; letter-spacing: 0.5px;">
                  ${node.name || 'Knowledge Node'}
                </div>
                <div style="font-size: 13px; line-height: 1.5; word-wrap: break-word; white-space: pre-wrap; color: rgba(255,255,255,0.85);">
                  ${node.label || 'No content available.'}
                </div>
              </div>
            `}
            nodeColor={getNodeColor}
            nodeRelSize={6}
            linkLabel={(link: any) => `<div style="background: rgba(10,10,10,0.95); padding: 8px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.1); color: white; font-size: 12px; font-weight: bold; text-transform: uppercase;">${link.label || 'Connected'}</div>`}
            linkColor={() => '#ffffff1a'} // text-white/10 edge color
            linkWidth={1.5}
            linkDirectionalParticles={2}
            linkDirectionalParticleSpeed={d => (d as any).weight ? (d as any).weight * 0.005 : 0.005}
            linkCanvasObjectMode={() => 'after'}
            linkCanvasObject={(link: any, ctx, globalScale) => {
              // Only render link labels when zoomed deeply in
              if (globalScale < 3.5 || !link.label) return;
              
              const start = link.source;
              const end = link.target;
              
              // Prevent crashes during transition states
              if (typeof start !== 'object' || typeof end !== 'object') return;
              
              const textPos = {
                x: start.x + (end.x - start.x) / 2,
                y: start.y + (end.y - start.y) / 2
              };
              
              const relLink = { x: end.x - start.x, y: end.y - start.y };
              const textAngle = Math.atan2(relLink.y, relLink.x);
              
              ctx.save();
              ctx.translate(textPos.x, textPos.y);
              ctx.rotate(textAngle);
              
              const fontSize = 10 / globalScale;
              ctx.font = `bold ${fontSize}px Inter, sans-serif`;
              ctx.fillStyle = 'rgba(255, 255, 255, 0.4)'; // subtle text
              ctx.textAlign = 'center';
              ctx.textBaseline = 'middle';
              
              // Add a slight dark background bounding box for text clarity
              const textWidth = ctx.measureText(link.label).width;
              ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
              ctx.fillRect(-textWidth/2 - (2/globalScale), -fontSize/2 - (2/globalScale), textWidth + (4/globalScale), fontSize + (4/globalScale));
              
              ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
              ctx.fillText(link.label, 0, 0);
              
              ctx.restore();
            }}
            // Glow effect for nodes based on links
            nodeCanvasObject={(node, ctx, globalScale) => {
              const displayName = node.name as string || 'Node';
              const fontSize = 12 / globalScale;
              
              // Draw node glow
              ctx.beginPath();
              ctx.arc(node.x!, node.y!, 5.5, 0, 2 * Math.PI, false);
              ctx.fillStyle = getNodeColor(node);
              ctx.fill();

              // Only show text if zoomed in
              if (globalScale > 2) {
                ctx.font = `${fontSize}px Inter, sans-serif`;
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                const textWidth = ctx.measureText(displayName).width;
                const bckgDimensions = [textWidth, fontSize].map(n => n + fontSize * 0.2); 

                ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
                ctx.fillRect(node.x! - bckgDimensions[0] / 2, node.y! + 9, bckgDimensions[0], bckgDimensions[1]);

                ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
                ctx.fillText(displayName, node.x!, node.y! + 9 + bckgDimensions[1]/2);
              }
            }}
            cooldownTicks={100}
            onEngineStop={() => graphRef.current?.zoomToFit(400, 50)}
          />
        )}
      </div>

    </div>
  );
}
