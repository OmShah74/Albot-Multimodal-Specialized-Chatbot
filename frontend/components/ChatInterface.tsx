import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Send, Paperclip, Bot, User, Trash2, StopCircle, Zap, Settings2, Clock, ChevronDown, ChevronUp, Activity, X, Globe, FileText, BookOpen, Brain, MessageSquare, Database, Search, Mic, MicOff } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import axios from 'axios';
import { api, Message, RetrievalConfig, QueryMetrics } from '@/lib/api';
import { cn } from '@/lib/utils';
import { motion, AnimatePresence } from 'framer-motion';
import { useNotification } from '@/context/NotificationContext';

interface ChatInterfaceProps {
  chatId: string | null;
  onChatUpdated?: (chatId: string, updates: Partial<{ title: string }>) => void;
  onCreateSession?: () => Promise<string | null>;
}

export function ChatInterface({ chatId, onChatUpdated, onCreateSession }: ChatInterfaceProps) {
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

  // Memory viewer state
  const [showMemoryViewer, setShowMemoryViewer] = useState(false);
  const [memoryDump, setMemoryDump] = useState<any>(null);
  const [memoryLoading, setMemoryLoading] = useState(false);
  const [memoryTab, setMemoryTab] = useState<'conversation' | 'fragments' | 'web' | 'traces'>('conversation');

  // Search mode: web_search (with fallback) or knowledge_base (local only)
  const [searchMode, setSearchMode] = useState<'web_search' | 'knowledge_base'>('knowledge_base');

  // Abort controller for stopping queries
  const abortControllerRef = useRef<AbortController | null>(null);

  // Speech recognition state
  const [isListening, setIsListening] = useState(false);
  const [interimText, setInterimText] = useState('');
  const [recordingTime, setRecordingTime] = useState(0);
  const [speechSupported, setSpeechSupported] = useState(false);
  const recognitionRef = useRef<any>(null);
  const recordingTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Initialize speech recognition
  useEffect(() => {
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SpeechRecognition) {
      setSpeechSupported(false);
      return;
    }
    setSpeechSupported(true);

    const recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = 'en-US';
    recognition.maxAlternatives = 1;

    recognition.onresult = (event: any) => {
      let finalTranscript = '';
      let interimTranscript = '';

      for (let i = event.resultIndex; i < event.results.length; i++) {
        const transcript = event.results[i][0].transcript;
        if (event.results[i].isFinal) {
          finalTranscript += transcript;
        } else {
          interimTranscript += transcript;
        }
      }

      if (finalTranscript) {
        setInput(prev => {
          const spacer = prev && !prev.endsWith(' ') ? ' ' : '';
          return prev + spacer + finalTranscript.trim();
        });
        setInterimText('');
      } else {
        setInterimText(interimTranscript);
      }
    };

    recognition.onerror = (event: any) => {
      console.error('Speech recognition error:', event.error);
      if (event.error !== 'no-speech') {
        setIsListening(false);
        setInterimText('');
      }
    };

    recognition.onend = () => {
      // If we're still supposed to be listening, restart (handles auto-stop)
      if (recognitionRef.current?._shouldRestart) {
        try {
          recognition.start();
        } catch (e) {
          setIsListening(false);
        }
      } else {
        setIsListening(false);
        setInterimText('');
      }
    };

    recognitionRef.current = recognition;

    return () => {
      recognition.abort();
      if (recordingTimerRef.current) clearInterval(recordingTimerRef.current);
    };
  }, []);

  const toggleListening = useCallback(() => {
    const recognition = recognitionRef.current;
    if (!recognition) return;

    if (isListening) {
      // Stop listening
      recognition._shouldRestart = false;
      recognition.stop();
      setIsListening(false);
      setInterimText('');
      setRecordingTime(0);
      if (recordingTimerRef.current) {
        clearInterval(recordingTimerRef.current);
        recordingTimerRef.current = null;
      }
    } else {
      // Start listening
      try {
        recognition._shouldRestart = true;
        recognition.start();
        setIsListening(true);
        setRecordingTime(0);
        recordingTimerRef.current = setInterval(() => {
          setRecordingTime(prev => prev + 1);
        }, 1000);
      } catch (e) {
        console.error('Failed to start speech recognition:', e);
      }
    }
  }, [isListening]);

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

    // Stop speech recognition if active
    if (isListening) {
      const recognition = recognitionRef.current;
      if (recognition) {
        recognition._shouldRestart = false;
        recognition.stop();
      }
      setIsListening(false);
      setInterimText('');
      setRecordingTime(0);
      if (recordingTimerRef.current) {
        clearInterval(recordingTimerRef.current);
        recordingTimerRef.current = null;
      }
    }

    const userMsg: Message = { role: 'user', content: input };
    setHistory(prev => [...prev, userMsg]);
    setInput('');
    setLoading(true);

    // If no chat ID (New Chat state), create a session first
    let activeChatId = chatId;
    if (!activeChatId) {
      if (onCreateSession) {
        try {
          const newId = await onCreateSession();
          if (newId) {
            activeChatId = newId;
          } else {
             throw new Error("Failed to create new chat session");
          }
        } catch (e) {
          console.error("Auto-creation failed", e);
          setLoading(false);
          return;
        }
      } else {
        // No create handler, cannot proceed
        setLoading(false);
        return;
      }
    }
    
    // Create a new AbortController for this request
    const controller = new AbortController();
    abortControllerRef.current = controller;

    try {
      // Build config for API with search mode
      const config: RetrievalConfig = {
        mode: retrievalMode,
        ...(retrievalMode === 'advanced' ? algorithms : {}),
        search_mode: searchMode
      };

      const response = await api.query(userMsg.content, activeChatId!, history, config, controller.signal);
      
      const assistantMsg: Message = { 
        role: 'assistant', 
        content: response.answer || "Sorry, I couldn't generate a response.",
        sources: response.sources,
        metrics: response.metrics
      };
      
      if (response.metrics) {
        setLastMetrics(response.metrics);
      }

      if (response.chat_title && activeChatId && onChatUpdated) {
        onChatUpdated(activeChatId, { title: response.chat_title });
      }
      
      setHistory(prev => [...prev, assistantMsg]);
    } catch (err) {
      // Check if this was an intentional abort (stop button)
      if (axios.isCancel(err) || (err instanceof DOMException && err.name === 'AbortError')) {
        // User clicked stop - append stopped message
        const stoppedMsg: Message = { role: 'assistant', content: '⏹ Generation stopped by user.' };
        setHistory(prev => [...prev, stoppedMsg]);
      } else {
        console.error('Query error', err);
        showNotification({
          type: 'error',
          title: 'Query Failed',
          message: 'There was an issue connecting to the Albot engine. please check the backend status.'
        });
        const errorMsg: Message = { role: 'assistant', content: `**Error**: ${err instanceof Error ? err.message : 'Unknown error'}` };
        setHistory(prev => [...prev, errorMsg]);
      }
    } finally {
      setLoading(false);
      abortControllerRef.current = null;
    }
  };

  const handleStop = async () => {
    // Abort the in-flight HTTP request
    abortControllerRef.current?.abort();

    // Also notify the backend to cancel server-side processing
    // Also notify the backend to cancel server-side processing
    if (chatId) {
      try {
        await api.cancelQuery(chatId);
      } catch (e) {
        // Best-effort — the abort above is the primary mechanism
        console.warn('Backend cancel failed:', e);
      }
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
        {chatId && (
          <button 
            type="button" 
            onClick={async () => {
              setShowMemoryViewer(true);
              setMemoryLoading(true);
              try {
                const dump = await api.getMemoryDump(chatId);
                setMemoryDump(dump);
              } catch (err) {
                console.error('Failed to load memory dump:', err);
              } finally {
                setMemoryLoading(false);
              }
            }}
            className="px-3 py-1.5 bg-white/5 hover:bg-purple-500/10 text-neutral-400 hover:text-purple-400 rounded-lg transition-all border border-white/5 text-[10px] uppercase tracking-widest font-bold flex items-center gap-2 group/btn backdrop-blur-md"
          >
            <Brain className="w-3.5 h-3.5" />
            Memory
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
        {/* Search Mode & RAG Mode Panel */}
        <div className="flex flex-col gap-3">
          <div className="flex items-center justify-between px-2 flex-wrap gap-2">
            <div className="flex items-center gap-3">
              {/* Search Mode Toggle */}
              <div className="flex bg-white/5 p-1 rounded-xl border border-white/10">
                <button
                  onClick={() => setSearchMode('knowledge_base')}
                  className={cn(
                    "flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium transition-all",
                    searchMode === 'knowledge_base' 
                      ? "bg-purple-600 text-white shadow-lg shadow-purple-600/20" 
                      : "text-neutral-400 hover:text-neutral-200"
                  )}
                >
                  <BookOpen className="w-3.5 h-3.5" />
                  Knowledge Base
                </button>
                <button
                  onClick={() => setSearchMode('web_search')}
                  className={cn(
                    "flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium transition-all",
                    searchMode === 'web_search' 
                      ? "bg-purple-600 text-white shadow-lg shadow-purple-600/20" 
                      : "text-neutral-400 hover:text-neutral-200"
                  )}
                >
                  <Globe className="w-3.5 h-3.5" />
                  Web Search
                </button>
              </div>

              {/* Retrieval Mode Toggle */}
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
            placeholder={isListening ? 'Listening...' : `Ask Albot anything... (${searchMode === 'knowledge_base' ? '📚 KB' : '🌐 Web'} · ${retrievalMode} RAG)`}
            className={cn(
              "w-full input-glass rounded-2xl py-4 pl-5 text-sm focus:outline-none transition-all placeholder:text-neutral-500 text-white shadow-2xl",
              isListening ? "pr-28 border-red-500/30 shadow-[0_0_20px_rgba(239,68,68,0.1)]" : "pr-24"
            )}
            disabled={loading}
          />
          <div className="absolute right-2 top-2 flex items-center gap-1">
            {/* Mic Button */}
            {speechSupported && !loading && (
              <button 
                type="button"
                onClick={toggleListening}
                className={cn(
                  "p-2 rounded-xl transition-all duration-300 active:scale-95 relative",
                  isListening 
                    ? "bg-red-500/20 text-red-400 hover:bg-red-500/30 border border-red-500/40 shadow-[0_0_15px_rgba(239,68,68,0.2)]" 
                    : "bg-white/10 hover:bg-white/20 text-neutral-400 hover:text-white"
                )}
                title={isListening ? 'Stop recording' : 'Voice input'}
              >
                {isListening ? (
                  <>
                    <MicOff className="w-5 h-5" />
                    {/* Pulse rings */}
                    <span className="absolute inset-0 rounded-xl border-2 border-red-500/50 animate-ping" />
                    <span className="absolute -top-1 -right-1 w-2.5 h-2.5 bg-red-500 rounded-full animate-pulse shadow-[0_0_8px_rgba(239,68,68,0.8)]" />
                  </>
                ) : (
                  <Mic className="w-5 h-5" />
                )}
              </button>
            )}
            {/* Send / Stop Button */}
            {loading ? (
              <button 
                type="button"
                onClick={handleStop}
                className="p-2 bg-red-500/20 hover:bg-red-500/40 text-red-400 hover:text-red-300 rounded-xl transition-colors active:scale-95 duration-200 border border-red-500/30"
                title="Stop generation"
              >
                <StopCircle className="w-5 h-5" />
              </button>
            ) : (
              <button 
                type="submit"
                disabled={!input.trim()}
                className="p-2 bg-white/10 hover:bg-primary text-white rounded-xl transition-colors disabled:opacity-50 disabled:hover:bg-white/10 group-hover:scale-105 active:scale-95 duration-200"
              >
                <Send className="w-5 h-5" />
              </button>
            )}
          </div>
        </form>

        {/* Voice Recording Indicator */}
        <AnimatePresence>
          {isListening && (
            <motion.div
              initial={{ opacity: 0, y: -8, height: 0 }}
              animate={{ opacity: 1, y: 0, height: 'auto' }}
              exit={{ opacity: 0, y: -8, height: 0 }}
              className="overflow-hidden"
            >
              <div className="flex items-center gap-3 px-4 py-2.5 bg-red-500/10 border border-red-500/20 rounded-xl mt-2">
                {/* Animated waveform bars */}
                <div className="flex items-center gap-0.5 h-4">
                  {[0, 1, 2, 3, 4].map(i => (
                    <div
                      key={i}
                      className="w-1 bg-red-400 rounded-full"
                      style={{
                        animation: `speechWave 0.8s ease-in-out ${i * 0.12}s infinite alternate`,
                        height: '4px',
                      }}
                    />
                  ))}
                </div>
                <span className="text-[10px] uppercase tracking-widest font-black text-red-400">
                  Recording
                </span>
                <span className="text-[10px] font-mono text-red-400/60">
                  {Math.floor(recordingTime / 60)}:{String(recordingTime % 60).padStart(2, '0')}
                </span>
                {interimText && (
                  <span className="text-xs text-neutral-400 italic truncate flex-1 ml-2">
                    {interimText}
                  </span>
                )}
                <button
                  type="button"
                  onClick={toggleListening}
                  className="text-[9px] uppercase tracking-widest font-black text-red-400 hover:text-red-300 transition-colors px-2 py-1 bg-red-500/10 rounded-lg border border-red-500/20 hover:border-red-500/30 ml-auto"
                >
                  Stop
                </button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
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
      {/* Memory Viewer Overlay */}
      <AnimatePresence>
        {showMemoryViewer && (
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
              {/* Header */}
              <div className="p-6 border-b border-white/5 flex justify-between items-center bg-white/5">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-xl bg-purple-500/20 flex items-center justify-center border border-purple-500/30">
                    <Brain className="w-5 h-5 text-purple-400" />
                  </div>
                  <div>
                    <h2 className="text-lg font-bold text-white tracking-tight uppercase">Session Memory</h2>
                    <p className="text-[10px] text-neutral-500 uppercase tracking-widest font-bold">
                      {memoryDump?.stats ? `${memoryDump.stats.total_messages} messages · ${memoryDump.stats.total_fragments} fragments · ${memoryDump.stats.total_traces} traces` : 'Loading...'}
                    </p>
                  </div>
                </div>
                <button 
                  onClick={() => setShowMemoryViewer(false)}
                  className="p-2 hover:bg-white/5 rounded-xl transition-colors text-neutral-400"
                >
                  <X className="w-6 h-6" />
                </button>
              </div>

              {/* Tabs */}
              <div className="flex border-b border-white/5 bg-white/[0.02]">
                {([
                  { key: 'conversation' as const, label: 'Conversation', icon: MessageSquare },
                  { key: 'fragments' as const, label: 'Fragments', icon: Database },
                  { key: 'web' as const, label: 'Web History', icon: Globe },
                  { key: 'traces' as const, label: 'Traces', icon: Search },
                ]).map(tab => (
                  <button
                    key={tab.key}
                    onClick={() => setMemoryTab(tab.key)}
                    className={cn(
                      "flex-1 flex items-center justify-center gap-2 px-4 py-3 text-[10px] uppercase tracking-widest font-black transition-all border-b-2",
                      memoryTab === tab.key
                        ? "border-purple-500 text-purple-400 bg-purple-500/5"
                        : "border-transparent text-neutral-500 hover:text-neutral-300 hover:bg-white/5"
                    )}
                  >
                    <tab.icon className="w-3.5 h-3.5" />
                    {tab.label}
                    {memoryDump && (
                      <span className="ml-1 text-[8px] opacity-60">
                        {tab.key === 'conversation' && memoryDump.stats?.total_messages}
                        {tab.key === 'fragments' && memoryDump.stats?.total_fragments}
                        {tab.key === 'web' && memoryDump.stats?.total_web_searches}
                        {tab.key === 'traces' && memoryDump.stats?.total_traces}
                      </span>
                    )}
                  </button>
                ))}
              </div>

              {/* Tab Content */}
              <div className="flex-1 overflow-y-auto p-6 space-y-4">
                {memoryLoading ? (
                  <div className="flex flex-col items-center justify-center h-48 text-neutral-500 gap-4">
                    <Brain className="w-12 h-12 opacity-20 animate-pulse" />
                    <p className="text-sm font-bold uppercase tracking-widest">Loading memory...</p>
                  </div>
                ) : !memoryDump ? (
                  <div className="flex flex-col items-center justify-center h-48 text-neutral-500 gap-4">
                    <Brain className="w-12 h-12 opacity-20" />
                    <p className="text-sm font-bold uppercase tracking-widest">Failed to load memory</p>
                  </div>
                ) : (
                  <>
                    {/* Conversation Tab */}
                    {memoryTab === 'conversation' && (
                      <div className="space-y-3">
                        {memoryDump.conversation_log.length === 0 ? (
                          <p className="text-center text-neutral-500 text-sm py-8">No messages in this session yet.</p>
                        ) : (
                          memoryDump.conversation_log.map((msg: any, i: number) => (
                            <div key={i} className={cn(
                              "p-4 rounded-2xl border",
                              msg.role === 'user' 
                                ? "bg-primary/5 border-primary/20" 
                                : "bg-white/5 border-white/5"
                            )}>
                              <div className="flex items-center gap-2 mb-2">
                                <span className={cn(
                                  "text-[9px] uppercase tracking-widest font-black px-2 py-0.5 rounded-md",
                                  msg.role === 'user' ? "bg-primary/20 text-primary" : "bg-white/10 text-neutral-400"
                                )}>
                                  {msg.role}
                                </span>
                                {msg.timestamp && (
                                  <span className="text-[9px] text-neutral-600">{new Date(msg.timestamp).toLocaleString()}</span>
                                )}
                              </div>
                              <p className="text-sm text-neutral-300 whitespace-pre-wrap leading-relaxed">{msg.content.length > 500 ? msg.content.slice(0, 500) + '...' : msg.content}</p>
                            </div>
                          ))
                        )}
                      </div>
                    )}

                    {/* Fragments Tab */}
                    {memoryTab === 'fragments' && (
                      <div className="space-y-3">
                        {memoryDump.fragments.length === 0 ? (
                          <p className="text-center text-neutral-500 text-sm py-8">No memory fragments extracted yet.</p>
                        ) : (
                          memoryDump.fragments.map((frag: any, i: number) => (
                            <div key={i} className="p-4 rounded-2xl bg-white/5 border border-white/5 space-y-2">
                              <div className="flex items-center gap-2 flex-wrap">
                                <span className="text-[9px] uppercase tracking-widest font-black bg-purple-500/20 text-purple-400 px-2 py-0.5 rounded-md">
                                  {frag.fragment_type}
                                </span>
                                <span className="text-[9px] text-neutral-500">Score: {(frag.importance_score || 0).toFixed(2)}</span>
                                <span className="text-[9px] text-neutral-600">Accesses: {frag.access_count || 0}</span>
                                {frag.tags && frag.tags.length > 0 && (
                                  <div className="flex gap-1">
                                    {frag.tags.map((tag: string, j: number) => (
                                      <span key={j} className="text-[8px] bg-white/10 text-neutral-400 px-1.5 py-0.5 rounded">{tag}</span>
                                    ))}
                                  </div>
                                )}
                              </div>
                              <p className="text-sm text-neutral-300 leading-relaxed">{frag.content}</p>
                              {frag.created_at && (
                                <p className="text-[9px] text-neutral-600">{new Date(frag.created_at).toLocaleString()}</p>
                              )}
                            </div>
                          ))
                        )}
                      </div>
                    )}

                    {/* Web History Tab */}
                    {memoryTab === 'web' && (
                      <div className="space-y-3">
                        {memoryDump.web_history.length === 0 ? (
                          <p className="text-center text-neutral-500 text-sm py-8">No web searches performed in this session.</p>
                        ) : (
                          memoryDump.web_history.map((log: any, i: number) => (
                            <div key={i} className="p-4 rounded-2xl bg-white/5 border border-white/5 space-y-1">
                              <div className="flex items-center gap-2">
                                <Globe className="w-3 h-3 text-primary" />
                                <span className="text-[9px] uppercase tracking-widest font-black bg-blue-500/20 text-blue-400 px-2 py-0.5 rounded-md">
                                  {log.provider}
                                </span>
                                <span className="text-[9px] text-neutral-500">Turn {log.turn_index}</span>
                                {log.relevance_score > 0 && (
                                  <span className="text-[9px] text-neutral-500">Score: {log.relevance_score.toFixed(2)}</span>
                                )}
                              </div>
                              <p className="text-xs text-neutral-400">Query: &quot;{log.query_sent}&quot;</p>
                              {log.title && <p className="text-sm text-neutral-200 font-medium">{log.title}</p>}
                              {log.url && (
                                <a href={log.url} target="_blank" rel="noopener noreferrer" className="text-xs text-primary hover:underline break-all">{log.url}</a>
                              )}
                              {log.snippet && <p className="text-xs text-neutral-400 leading-relaxed">{log.snippet}</p>}
                            </div>
                          ))
                        )}
                      </div>
                    )}

                    {/* Traces Tab */}
                    {memoryTab === 'traces' && (
                      <div className="space-y-3">
                        {memoryDump.reasoning_traces.length === 0 ? (
                          <p className="text-center text-neutral-500 text-sm py-8">No reasoning traces recorded yet.</p>
                        ) : (
                          memoryDump.reasoning_traces.map((trace: any, i: number) => (
                            <div key={i} className="p-4 rounded-2xl bg-white/5 border border-white/5 space-y-3">
                              <div className="flex items-center justify-between">
                                <div className="flex items-center gap-2">
                                  <span className="w-8 h-8 rounded-lg bg-white/5 flex items-center justify-center text-[10px] font-black text-neutral-500 border border-white/10">T{trace.turn_index + 1}</span>
                                  <span className="text-sm font-bold text-neutral-200">{trace.total_time_ms?.toFixed(0)}ms</span>
                                  {trace.web_search_triggered && <Globe className="w-3 h-3 text-blue-400" />}
                                </div>
                                <span className="text-[9px] text-neutral-600">{trace.created_at ? new Date(trace.created_at).toLocaleString() : ''}</span>
                              </div>
                              <div className="space-y-1">
                                <p className="text-xs text-neutral-400"><span className="font-bold text-neutral-300">Query:</span> {trace.user_query}</p>
                                {trace.reformulated_query && trace.reformulated_query !== trace.user_query && (
                                  <p className="text-xs text-neutral-500"><span className="font-bold text-neutral-400">Reformulated:</span> {trace.reformulated_query}</p>
                                )}
                                {trace.algorithms_used && trace.algorithms_used.length > 0 && (
                                  <div className="flex flex-wrap gap-1 mt-1">
                                    {trace.algorithms_used.map((algo: string, j: number) => (
                                      <span key={j} className="text-[8px] bg-primary/10 text-primary px-1.5 py-0.5 rounded border border-primary/20 font-bold uppercase">{algo}</span>
                                    ))}
                                  </div>
                                )}
                                {trace.answer_summary && (
                                  <p className="text-xs text-neutral-400 mt-1"><span className="font-bold text-neutral-300">Summary:</span> {trace.answer_summary.slice(0, 200)}{trace.answer_summary.length > 200 ? '...' : ''}</p>
                                )}
                              </div>
                            </div>
                          ))
                        )}
                      </div>
                    )}
                  </>
                )}
              </div>

              {/* Footer */}
              <div className="p-6 bg-white/5 border-t border-white/5 text-center">
                <p className="text-[10px] text-neutral-500 uppercase tracking-[0.3em] font-black">
                  Memory Architecture System v1.0
                </p>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
