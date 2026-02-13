import React, { useState, useRef, useEffect } from 'react';
import { Send, Paperclip, Bot, User, Trash2, StopCircle, Zap, Settings2, Clock, ChevronDown, ChevronUp, Activity, X, Globe, FileText } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { api, Message, RetrievalConfig, QueryMetrics } from '@/lib/api';
import { cn } from '@/lib/utils';
import { motion, AnimatePresence } from 'framer-motion';
import { useNotification } from '@/context/NotificationContext';

interface ChatInterfaceProps {
  chatId: string | null;
  onChatUpdated?: (chatId: string, updates: Partial<{ title: string }>) => void;
}

export function ChatInterface({ chatId, onChatUpdated }: ChatInterfaceProps) {
  const [input, setInput] = useState('');
  const [history, setHistory] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  
  // Retrieval mode state
  const [retrievalMode, setRetrievalMode] = useState<'fast' | 'advanced'>('advanced');
  const [showAdvancedOptions, setShowAdvancedOptions] = useState(false);
  const [algorithms, setAlgorithms] = useState({
    use_vector: true,
    use_graph: true,
    use_pagerank: true,
    use_bm25: true,
    use_structural: true,
    use_mmr: true
  });
  const [lastMetrics, setLastMetrics] = useState<QueryMetrics | null>(null);
  const [showMetricsOverlay, setShowMetricsOverlay] = useState(false);

  useEffect(() => {
    const loadHistory = async () => {
      if (!chatId) {
          setHistory([]);
          return;
      }
      try {
        setLoading(true);
        const savedHistory = await api.getChatHistory(chatId);
        setHistory(savedHistory || []);
      } catch (err) {
        console.error('Failed to load history:', err);
      } finally {
        setLoading(false);
      }
    };
    loadHistory();
  }, [chatId]);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [history, loading]);

  const { showNotification } = useNotification();

  const handleSubmit = async (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!input.trim() || loading) return;

    const userMsg: Message = { role: 'user', content: input };
    setHistory(prev => [...prev, userMsg]);
    setInput('');
    setLoading(true);

    if (!chatId) return; // Should not happen if UI is correct

    try {
      // Build config for API
      const config: RetrievalConfig = {
        mode: retrievalMode,
        ...(retrievalMode === 'advanced' ? algorithms : {})
      };

      const response = await api.query(userMsg.content, chatId, history, config);
      
      const assistantMsg: Message = { 
        role: 'assistant', 
        content: response.answer || "Sorry, I couldn't generate a response.",
        sources: response.sources,
        metrics: response.metrics
      };
      
      if (response.metrics) {
        setLastMetrics(response.metrics);
      }

      if (response.chat_title && chatId && onChatUpdated) {
        onChatUpdated(chatId, { title: response.chat_title });
      }
      
      setHistory(prev => [...prev, assistantMsg]);
    } catch (err) {
      console.error('Query error', err);
      showNotification({
        type: 'error',
        title: 'Query Failed',
        message: 'There was an issue connecting to the Albot engine. please check the backend status.'
      });
      const errorMsg: Message = { role: 'assistant', content: `**Error**: ${err instanceof Error ? err.message : 'Unknown error'}` };
      setHistory(prev => [...prev, errorMsg]);
    } finally {
      setLoading(false);
    }
  };

  const clearChat = () => {
    if (!chatId) return;
    
    showNotification({
      type: 'confirm',
      title: 'Clear Conversation',
      message: 'Are you sure you want to clear the entire chat history? This cannot be undone.',
      confirmText: 'Clear',
      onConfirm: async () => {
        try {
          await api.clearChatHistory(chatId);
          setHistory([]);
          showNotification({
            type: 'success',
            title: 'History Cleared',
            message: 'Conversation history has been reset.'
          });
        } catch (err) {
           console.error('Failed to clear history:', err);
           showNotification({
            type: 'error',
            title: 'Error',
            message: 'Failed to clear history on backend.'
          });
        }
      }
    });
  };

  return (
    <div className="flex-1 flex flex-col h-full relative">
      <div className="absolute top-4 right-8 z-20 flex gap-2">
        {history.some(m => m.metrics) && (
          <button 
            type="button" 
            onClick={() => setShowMetricsOverlay(true)}
            className="px-3 py-1.5 bg-white/5 hover:bg-primary/10 text-neutral-400 hover:text-primary rounded-lg transition-all border border-white/5 text-[10px] uppercase tracking-widest font-bold flex items-center gap-2 group/btn backdrop-blur-md"
          >
            <Activity className="w-3.5 h-3.5" />
          </button>
        )}
        {history.length > 0 && (
          <button 
            type="button" 
            onClick={clearChat}
            className="px-3 py-1.5 bg-white/5 hover:bg-red-500/10 text-neutral-400 hover:text-red-400 rounded-lg transition-all border border-white/5 text-[10px] uppercase tracking-widest font-bold flex items-center gap-2 group/btn backdrop-blur-md"
          >
            <Trash2 className="w-3.5 h-3.5" />
            Clear Chat
          </button>
        )}
      </div>

      <div className="flex-1 overflow-y-auto p-4 md:p-8 space-y-6" ref={scrollRef}>
        {history.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-neutral-500 space-y-4">
             <div className="w-24 h-24 rounded-2xl bg-white/5 flex items-center justify-center shadow-[0_0_30px_rgba(147,51,234,0.1)]">
                <Bot className="w-12 h-12 text-primary/50" />
             </div>
             <p className="text-lg font-medium">Hello! I'm Albot. How can I assist you today?</p>
          </div>
        )}
        
        <AnimatePresence mode="popLayout">
          {history.map((msg, idx) => (
            <motion.div 
              key={idx} 
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className={`flex gap-3 ${msg.role === 'user' ? 'flex-row-reverse' : 'flex-row'}`}
            >
              <div className={cn(
                "w-8 h-8 rounded-lg flex items-center justify-center shrink-0 mt-1",
                msg.role === 'user' ? "bg-primary/20 border border-primary/30" : "bg-white/5 border border-white/10"
              )}>
                {msg.role === 'user' ? <User className="w-4 h-4 text-primary" /> : <Bot className="w-4 h-4 text-primary" />}
              </div>
              <div className={`
                max-w-[92%] rounded-2xl px-5 py-4 text-[15px] leading-relaxed shadow-lg transition-all
                ${msg.role === 'user' 
                  ? 'bg-primary text-white rounded-tr-none' 
                  : 'bg-white/5 border border-white/10 text-neutral-200 rounded-tl-none'}
              `}>
                {msg.role === 'assistant' ? (
                  <div className="flex flex-col gap-3">
                    <div className="prose prose-invert prose-base max-w-none prose-p:leading-relaxed prose-pre:my-4 prose-p:my-3 prose-headings:mb-4 prose-headings:mt-6 first:prose-headings:mt-0">
                      <ReactMarkdown>{msg.content}</ReactMarkdown>
                    </div>
                    {msg.sources && msg.sources.length > 0 && (
                      <div className="pt-3 border-t border-white/10 mt-1">
                        <p className="text-[10px] uppercase tracking-wider font-bold text-neutral-500 mb-2">Sources</p>
                        <div className="flex flex-wrap gap-2">
                          {msg.sources.map((src, i) => {
                            const isUrl = src.startsWith('http');
                            let displayName = src;
                            if (isUrl) {
                              try {
                                const url = new URL(src);
                                displayName = url.hostname.replace('www.', '');
                              } catch (e) {
                                displayName = 'Link';
                              }
                            }
                            return (
                              <a 
                                key={i} 
                                href={isUrl ? src : undefined}
                                target="_blank"
                                rel="noopener noreferrer"
                                className={cn(
                                  "px-2 py-0.5 rounded-full bg-white/5 border border-white/10 text-[10px] text-neutral-400 flex items-center gap-1 transition-colors",
                                  isUrl ? "hover:bg-primary/10 hover:border-primary/30 hover:text-primary pointer-events-auto" : "pointer-events-none"
                                )}
                              >
                                {isUrl ? <Globe className="w-2.5 h-2.5" /> : <FileText className="w-2.5 h-2.5" />}
                                {displayName}
                              </a>
                            );
                          })}
                        </div>
                      </div>
                    )}
                    {msg.metrics && (
                      <div className="pt-2 flex justify-end">
                        <button 
                          onClick={() => {
                            setLastMetrics(msg.metrics || null);
                            setShowMetricsOverlay(true);
                          }}
                          className="text-[9px] uppercase tracking-widest font-bold text-primary hover:bg-primary/10 transition-colors flex items-center gap-1 bg-primary/5 px-2 py-1 rounded-lg border border-primary/10"
                        >
                          <Zap className="w-2.5 h-2.5" />
                          Technical Report
                        </button>
                      </div>
                    )}
                  </div>
                ) : (
                  <ReactMarkdown>{msg.content}</ReactMarkdown>
                )}
              </div>
            </motion.div>
          ))}
          
          {loading && (
             <motion.div 
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex gap-3 justify-start"
            >
               <div className="w-8 h-8 rounded-lg bg-white/5 border border-white/10 flex items-center justify-center shrink-0 mt-1">
                 <Bot className="w-4 h-4 text-primary animate-pulse" />
               </div>
               <div className="bg-white/5 border border-white/10 rounded-2xl px-5 py-4 rounded-tl-none flex flex-col gap-2 min-w-[200px]">
                 <div className="flex gap-1.5">
                   <span className="w-1.5 h-1.5 rounded-full bg-primary animate-bounce shadow-[0_0_8px_rgba(147,51,234,0.5)]" style={{ animationDelay: '0s' }}></span>
                   <span className="w-1.5 h-1.5 rounded-full bg-primary animate-bounce shadow-[0_0_8px_rgba(147,51,234,0.5)]" style={{ animationDelay: '0.2s' }}></span>
                   <span className="w-1.5 h-1.5 rounded-full bg-primary animate-bounce shadow-[0_0_8px_rgba(147,51,234,0.5)]" style={{ animationDelay: '0.4s' }}></span>
                 </div>
                 <p className="text-[10px] uppercase tracking-[0.2em] font-black text-neutral-500 animate-pulse">
                    Thinking & Retrieving Evidence...
                 </p>
               </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <div className="p-4 md:p-6 w-full max-w-4xl mx-auto space-y-4">
        {/* RAG Mode & Metrics Panel */}
        <div className="flex flex-col gap-3">
          <div className="flex items-center justify-between px-2">
            <div className="flex items-center gap-4">
              <div className="flex bg-white/5 p-1 rounded-xl border border-white/10">
                <button
                  onClick={() => setRetrievalMode('fast')}
                  className={cn(
                    "flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium transition-all",
                    retrievalMode === 'fast' 
                      ? "bg-primary text-white shadow-lg shadow-primary/20" 
                      : "text-neutral-400 hover:text-neutral-200"
                  )}
                >
                  <Zap className="w-3.5 h-3.5" />
                  Fast RAG
                </button>
                <button
                  onClick={() => setRetrievalMode('advanced')}
                  className={cn(
                    "flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium transition-all",
                    retrievalMode === 'advanced' 
                      ? "bg-primary text-white shadow-lg shadow-primary/20" 
                      : "text-neutral-400 hover:text-neutral-200"
                  )}
                >
                  <Settings2 className="w-3.5 h-3.5" />
                  Advanced RAG
                </button>
              </div>
            </div>

            {retrievalMode === 'advanced' && (
              <button
                type="button"
                onClick={() => setShowAdvancedOptions(!showAdvancedOptions)}
                className={cn(
                  "flex items-center gap-2 px-3 py-1.5 rounded-lg text-[10px] uppercase tracking-widest font-black transition-all border",
                  showAdvancedOptions 
                    ? "bg-primary/20 border-primary/30 text-primary" 
                    : "bg-white/5 border-white/10 text-neutral-500 hover:text-neutral-300 shadow-lg"
                )}
              >
                <Settings2 className="w-3 h-3" />
                {showAdvancedOptions ? "Hide Algos" : "Configure Algos"}
                {showAdvancedOptions ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
              </button>
            )}
          </div>

          {retrievalMode === 'advanced' && (
            <AnimatePresence>
              {showAdvancedOptions && (
                <motion.div 
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  className="overflow-hidden mt-2"
                >
                  <div className="bg-white/5 rounded-2xl border border-white/10 p-4 grid grid-cols-2 lg:grid-cols-6 gap-4">
                    {Object.entries(algorithms).map(([key, enabled]) => (
                      <div 
                        key={key} 
                        onClick={() => setAlgorithms(prev => ({ ...prev, [key]: !enabled }))}
                        className="flex items-center gap-2 cursor-pointer group select-none"
                      >
                        <div 
                          className={cn(
                            "w-4 h-4 rounded border flex items-center justify-center transition-all duration-200",
                            enabled 
                              ? "bg-primary border-primary shadow-[0_0_8px_rgba(147,51,234,0.4)]" 
                              : "border-white/20 group-hover:border-white/40"
                          )}
                        >
                          {enabled && <div className="w-1.5 h-1.5 bg-white rounded-full" />}
                        </div>
                        <span className={cn(
                          "text-[10px] uppercase tracking-wider font-bold transition-colors",
                          enabled ? "text-neutral-200" : "text-neutral-500 group-hover:text-neutral-400"
                        )}>
                          {key.replace('use_', '').replace('_', ' ')}
                        </span>
                      </div>
                    ))}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          )}
        </div>

        <form onSubmit={handleSubmit} className="relative group">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={`Ask Albot anything... (${retrievalMode} RAG enabled)`}
            className="w-full input-glass rounded-2xl py-4 pl-5 pr-14 text-sm focus:outline-none transition-all placeholder:text-neutral-500 text-white shadow-2xl"
          />
          <button 
            type="submit"
            disabled={!input.trim() || loading}
            className="absolute right-2 top-2 p-2 bg-white/10 hover:bg-primary text-white rounded-xl transition-colors disabled:opacity-50 disabled:hover:bg-white/10 group-hover:scale-105 active:scale-95 duration-200"
          >
             {loading ? <StopCircle className="w-5 h-5 animate-pulse" /> : <Send className="w-5 h-5" />}
          </button>
        </form>
        <p className="text-center text-[10px] text-neutral-600 uppercase tracking-widest font-medium">
           Albot Multi-Modal Intelligence Engine v1.0
        </p>
      </div>

      {/* Analytics Overlay */}
      <AnimatePresence>
        {showMetricsOverlay && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center p-4 md:p-8 backdrop-blur-xl bg-black/40"
          >
            <motion.div 
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              className="w-full max-w-5xl bg-[#0a0a0a] border border-white/10 rounded-3xl overflow-hidden shadow-[0_0_100px_rgba(147,51,234,0.15)] flex flex-col max-h-[90vh]"
            >
              <div className="p-6 border-b border-white/5 flex justify-between items-center bg-white/5">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-xl bg-primary/20 flex items-center justify-center border border-primary/30">
                    <Activity className="w-5 h-5 text-primary" />
                  </div>
                  <div>
                    <h2 className="text-lg font-bold text-white tracking-tight uppercase">Technical Intelligence Report</h2>
                    <p className="text-[10px] text-neutral-500 uppercase tracking-widest font-bold">Deep Retrieval Analytics</p>
                  </div>
                </div>
                <button 
                  onClick={() => setShowMetricsOverlay(false)}
                  className="p-2 hover:bg-white/5 rounded-xl transition-colors text-neutral-400"
                >
                  <X className="w-6 h-6" />
                </button>
              </div>

              <div className="flex-1 overflow-y-auto p-6 space-y-8">
                 {lastMetrics ? (
                   <section className="space-y-4">
                      <div className="flex items-center gap-2 px-1">
                        <Zap className="w-4 h-4 text-primary" />
                        <h3 className="text-xs font-black uppercase tracking-[0.2em] text-neutral-400">Current Turn Analysis</h3>
                      </div>
                      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                        {[
                          { label: 'Vector Search', value: lastMetrics.vector_time_ms, color: 'text-white' },
                          { label: 'Graph Traversal', value: lastMetrics.graph_time_ms, color: 'text-white' },
                          { label: 'BM25 / Algos', value: lastMetrics.bm25_time_ms, color: 'text-white' },
                          { label: 'AI Synthesis', value: lastMetrics.synthesis_time_ms, color: 'text-primary' }
                        ].map((stat, i) => (
                          <div key={i} className="bg-white/5 p-4 rounded-2xl border border-white/5 flex flex-col gap-1">
                            <span className="text-[9px] uppercase tracking-widest text-neutral-500 font-black">{stat.label}</span>
                            <span className={cn("text-xl font-bold tracking-tight", stat.color)}>{stat.value.toFixed(1)}<span className="text-xs ml-0.5 opacity-50">ms</span></span>
                          </div>
                        ))}
                      </div>
                      
                      {lastMetrics.web_search_used && (
                        <div className="mt-4 grid grid-cols-2 lg:grid-cols-4 gap-4">
                           <div className="bg-white/5 p-4 rounded-2xl border border-white/5 flex flex-col gap-1">
                             <span className="text-[9px] uppercase tracking-widest text-neutral-500 font-black">Web Search</span>
                             <span className="text-xl font-bold tracking-tight text-white flex items-center gap-1">
                                <Globe className="w-4 h-4 text-primary" />
                                {lastMetrics.web_search_time_ms?.toFixed(1)}<span className="text-xs ml-0.5 opacity-50">ms</span>
                             </span>
                           </div>
                           <div className="bg-white/5 p-4 rounded-2xl border border-white/5 flex flex-col gap-1 lg:col-span-3">
                             <span className="text-[9px] uppercase tracking-widest text-neutral-500 font-black">Providers Consulted</span>
                             <div className="flex flex-wrap gap-2 mt-1">
                                {Object.entries(lastMetrics.web_providers_used || {}).map(([key, count]) => (
                                  <span key={key} className="bg-white/10 px-2 py-1 rounded-lg text-[10px] font-bold text-neutral-300 border border-white/10">
                                    {key}: {count} results
                                  </span>
                                ))}
                             </div>
                           </div>
                        </div>
                      )}
                   </section>
                 ) : (
                   <div className="flex flex-col items-center justify-center h-48 text-neutral-500 gap-4">
                      <Activity className="w-12 h-12 opacity-20" />
                      <p className="text-sm font-bold uppercase tracking-widest">No metrics recorded yet</p>
                   </div>
                 )}

                 {history.filter(m => m.role === 'assistant' && m.metrics).length > 0 && (
                   <section className="space-y-4">
                      <div className="flex items-center gap-2 px-1">
                        <Clock className="w-4 h-4 text-neutral-500" />
                        <h3 className="text-xs font-black uppercase tracking-[0.2em] text-neutral-400">Conversation History</h3>
                      </div>
                      <div className="space-y-2">
                         {history.filter(m => m.role === 'assistant' && m.metrics).map((m, i) => (
                           <div 
                            key={i} 
                            className={cn(
                              "flex items-center justify-between p-4 rounded-2xl border transition-all cursor-pointer group",
                              m.metrics === lastMetrics 
                                ? "bg-primary/10 border-primary/30" 
                                : "bg-white/5 border-white/5 hover:border-white/10"
                            )}
                            onClick={() => setLastMetrics(m.metrics || null)}
                           >
                              <div className="flex items-center gap-4">
                                <span className="w-8 h-8 rounded-lg bg-white/5 flex items-center justify-center text-[10px] font-black text-neutral-500 border border-white/10 group-hover:text-white transition-colors">T{i+1}</span>
                                <div className="flex flex-col">
                                  <span className="text-sm font-bold text-neutral-200">{m.metrics?.total_time_ms.toFixed(0)}ms Total</span>
                                  <span className="text-[9px] uppercase tracking-widest text-neutral-500 font-bold">{m.content.slice(0, 80).replace(/\*\*/g, '')}...</span>
                                </div>
                              </div>
                              <div className="flex gap-4">
                                 <div className="text-right flex flex-col">
                                   <span className="text-[9px] uppercase tracking-widest text-neutral-600 font-black">Mode</span>
                                   <span className="text-[10px] font-bold text-primary">{m.metrics?.mode}</span>
                                 </div>
                                 <Zap className={cn("w-4 h-4 mt-1 transition-colors", m.metrics === lastMetrics ? "text-primary" : "text-neutral-700")} />
                              </div>
                           </div>
                         ))}
                      </div>
                   </section>
                 )}
              </div>

              <div className="p-6 bg-white/5 border-t border-white/5 text-center">
                 <p className="text-[10px] text-neutral-500 uppercase tracking-[0.3em] font-black">
                   Advanced Retrieval Metrics System v1.0
                 </p>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
