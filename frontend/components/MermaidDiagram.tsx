import React, { useEffect, useRef, useState } from 'react';
import mermaid from 'mermaid';
import { Loader2 } from 'lucide-react';

mermaid.initialize({
  startOnLoad: false,
  theme: 'dark',
  securityLevel: 'loose',
});

interface MermaidProps {
  chart: string;
}

export default function MermaidDiagram({ chart }: MermaidProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [svgId] = useState(() => `mermaid-${Math.random().toString(36).substring(2, 9)}`);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let isMounted = true;
    setLoading(true);
    
    const renderDiagram = async () => {
      try {
        if (containerRef.current) {
          // Clear previous render to avoid visual stacking if it gets called twice initially
          containerRef.current.innerHTML = '';
          const { svg } = await mermaid.render(svgId, chart);
          if (isMounted) {
            containerRef.current.innerHTML = svg;
            setError(null);
          }
        }
      } catch (err: any) {
        console.error('Mermaid rendering failed', err);
        if (isMounted) {
          setError(err?.message || 'Failed to render diagram');
        }
      } finally {
        if (isMounted) {
          setLoading(false);
        }
      }
    };
    
    // Slight delay to ensure DOM is ready and prevent rapid render flashing
    const timeout = setTimeout(renderDiagram, 100);
    return () => {
      isMounted = false;
      clearTimeout(timeout);
    };
  }, [chart, svgId]);

  return (
    <div className="relative p-6 bg-[#0d0d0d] border border-white/10 rounded-xl flex flex-col items-center justify-center my-4 w-full min-h-[200px] shadow-lg group">
      <div className="absolute left-6 top-6 z-10 pointer-events-none">
        <span className="text-[10px] uppercase tracking-widest text-white/40 font-bold">Diagram Visualization</span>
      </div>
      
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center bg-[#0d0d0d]/80 backdrop-blur-sm z-10 rounded-xl">
          <Loader2 className="w-5 h-5 text-primary animate-spin" />
        </div>
      )}
      
      {error ? (
         <div className="p-4 bg-red-500/10 border border-red-500/30 rounded-xl text-red-500 text-sm mt-6 w-full">
           <div className="font-bold flex items-center gap-2 mb-2">
             ⚠️ Invalid Mermaid Syntax
           </div>
           <pre className="text-xs overflow-auto font-mono text-red-400/80 bg-black/20 p-2 rounded">{error}</pre>
         </div>
      ) : (
        <div ref={containerRef} className="mermaid diagram-container w-full flex justify-center mt-4 [&>svg]:max-w-full [&>svg]:h-auto" />
      )}
    </div>
  );
}
