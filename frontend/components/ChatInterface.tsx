import React, { useState, useRef, useEffect } from 'react';
import { Send,  Paperclip, Bot, User, Trash2, StopCircle } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { api, Message } from '@/lib/api';
import { motion, AnimatePresence } from 'framer-motion';

export function ChatInterface() {
  const [input, setInput] = useState('');
  const [history, setHistory] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [history, loading]);

  const handleSubmit = async (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!input.trim() || loading) return;

    const userMsg: Message = { role: 'user', content: input };
    setHistory(prev => [...prev, userMsg]);
    setInput('');
    setLoading(true);

    try {
      // In a real streaming setup we'd use a stream, but for now we wait for full response
      // mimicking the app.py behavior
      const response = await api.query(userMsg.content, history);
      
      const assistantMsg: Message = { 
        role: 'assistant', 
        content: response.answer || "Sorry, I couldn't generate a response." 
      };
      
      setHistory(prev => [...prev, assistantMsg]);
    } catch (err) {
      const errorMsg: Message = { role: 'assistant', content: `**Error**: ${err instanceof Error ? err.message : 'Unknown error'}` };
      setHistory(prev => [...prev, errorMsg]);
    } finally {
      setLoading(false);
    }
  };

  const clearChat = () => setHistory([]);

  return (
    <div className="flex-1 flex flex-col h-full relative">
      <div className="flex-1 overflow-y-auto p-4 md:p-8 space-y-6" ref={scrollRef}>
        {history.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-neutral-500 space-y-4">
             <div className="w-24 h-24 rounded-2xl bg-white/5 flex items-center justify-center shadow-[0_0_30px_rgba(147,51,234,0.1)]">
                <Bot className="w-12 h-12 text-primary/50" />
             </div>
             <p className="text-lg font-medium">How can I help you today?</p>
          </div>
        )}
        
        <AnimatePresence mode="popLayout">
          {history.map((msg, idx) => (
            <motion.div 
              key={idx} 
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className={`flex gap-4 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div className={`
                max-w-[80%] rounded-2xl px-5 py-3.5 text-sm leading-relaxed shadow-sm
                ${msg.role === 'user' 
                  ? 'bg-primary text-white rounded-br-none' 
                  : 'bg-white/5 border border-white/5 text-neutral-200 rounded-bl-none'}
              `}>
                {msg.role === 'assistant' ? (
                  <div className="prose prose-invert prose-sm max-w-none">
                    <ReactMarkdown>{msg.content}</ReactMarkdown>
                  </div>
                ) : (
                  msg.content
                )}
              </div>
            </motion.div>
          ))}
          
          {loading && (
             <motion.div 
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex gap-4 justify-start"
            >
               <div className="bg-white/5 border border-white/5 rounded-2xl px-5 py-4 rounded-bl-none flex gap-1">
                 <span className="w-2 h-2 rounded-full bg-primary/40 animate-bounce" style={{ animationDelay: '0s' }}></span>
                 <span className="w-2 h-2 rounded-full bg-primary/40 animate-bounce" style={{ animationDelay: '0.2s' }}></span>
                 <span className="w-2 h-2 rounded-full bg-primary/40 animate-bounce" style={{ animationDelay: '0.4s' }}></span>
               </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <div className="p-4 md:p-6 w-full max-w-4xl mx-auto">
        <form onSubmit={handleSubmit} className="relative group">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Send a message..."
            className="w-full input-glass rounded-2xl py-4 pl-5 pr-14 text-sm focus:outline-none transition-all placeholder:text-neutral-500 text-white"
          />
          <button 
            type="submit"
            disabled={!input.trim() || loading}
            className="absolute right-2 top-2 p-2 bg-white/10 hover:bg-primary text-white rounded-xl transition-colors disabled:opacity-50 disabled:hover:bg-white/10"
          >
             {loading ? <StopCircle className="w-5 h-5 animate-pulse" /> : <Send className="w-5 h-5" />}
          </button>
          
          <div className="absolute left-0 -top-12 flex gap-2">
            {history.length > 0 && (
                <button 
                  type="button" 
                  onClick={clearChat}
                  className="p-2 bg-white/5 hover:bg-red-500/20 text-neutral-400 hover:text-red-400 rounded-xl transition-colors border border-white/5"
                  title="Clear Chat"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
            )}
           </div>
        </form>
        <p className="text-center text-xs text-neutral-600 mt-2">
           AI can make mistakes. Please verify important information.
        </p>
      </div>
    </div>
  );
}
