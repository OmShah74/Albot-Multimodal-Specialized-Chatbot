import React, { useState } from 'react';
import { X, Upload, FileUp, Link, Loader2, CheckCircle } from 'lucide-react';
import { api } from '@/lib/api';
import { cn } from '@/lib/utils';
import { motion } from 'framer-motion';

interface IngestModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export function IngestModal({ isOpen, onClose }: IngestModalProps) {
  const [activeTab, setActiveTab] = useState<'file' | 'url'>('file');
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState<string>('');
  const [loading, setLoading] = useState(false);

  if (!isOpen) return null;

  const handleFileUpload = async () => {
    if (!file) return;
    setLoading(true);
    setStatus('Uploading and processing...');
    try {
      const res = await api.uploadFile(file);
      setStatus(typeof res === 'string' ? res : JSON.stringify(res));
    } catch (e) {
      setStatus('Error uploading file');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm">
      <motion.div 
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.95 }}
        className="w-full max-w-md bg-[#0a0a0a] border border-white/10 rounded-2xl shadow-2xl overflow-hidden"
      >
         <div className="p-4 border-b border-white/10 flex justify-between items-center bg-white/5">
          <h2 className="font-semibold flex items-center gap-2">
            <Upload className="w-4 h-4 text-primary" />
            Ingest Knowledge
          </h2>
          <button onClick={onClose} className="p-1 hover:bg-white/10 rounded-lg transition-colors">
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="p-1 mx-6 mt-6 bg-white/5 rounded-xl flex">
          <button 
            onClick={() => setActiveTab('file')}
            className={cn(
              "flex-1 py-2 text-sm font-medium rounded-lg transition-all",
              activeTab === 'file' ? "bg-white/10 text-white shadow-sm" : "text-neutral-500 hover:text-neutral-300"
            )}
          >
            Upload File
          </button>
          <button 
             onClick={() => setActiveTab('url')}
             className={cn(
              "flex-1 py-2 text-sm font-medium rounded-lg transition-all",
              activeTab === 'url' ? "bg-white/10 text-white shadow-sm" : "text-neutral-500 hover:text-neutral-300"
            )}
          >
            Ingest URL
          </button>
        </div>

        <div className="p-6">
          {activeTab === 'file' ? (
            <div className="space-y-4">
              <div className="border-2 border-dashed border-white/10 rounded-2xl p-8 hover:border-primary/50 transition-colors bg-white/5 flex flex-col items-center justify-center gap-3 group relative cursor-pointer">
                <input 
                  type="file" 
                  className="absolute inset-0 opacity-0 cursor-pointer"
                  onChange={(e) => setFile(e.target.files?.[0] || null)}
                />
                <div className="w-12 h-12 rounded-full bg-white/5 flex items-center justify-center group-hover:scale-110 transition-transform">
                  <FileUp className="w-6 h-6 text-primary" />
                </div>
                <div className="text-center">
                  <p className="font-medium text-sm">{file ? file.name : "Click to select or drag file"}</p>
                  <p className="text-xs text-neutral-500 mt-1">PDF, DOCX, TXT, MD</p>
                </div>
              </div>

              <button 
                onClick={handleFileUpload}
                disabled={!file || loading}
                className="w-full bg-primary hover:bg-primary-hover text-white py-3 rounded-xl font-medium transition-all shadow-lg shadow-purple-500/20 flex items-center justify-center gap-2 disabled:opacity-50"
              >
                {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Upload className="w-4 h-4" />}
                Process File
              </button>
            </div>
          ) : (
             <div className="space-y-4">
               <div className="space-y-2">
                 <label className="text-xs font-medium text-neutral-400 uppercase">Data Source URL</label>
                 <div className="relative">
                   <Link className="absolute left-3 top-3.5 w-4 h-4 text-neutral-500" />
                   <input 
                     type="url" 
                     placeholder="https://example.com/guide"
                     className="w-full bg-black/50 border border-white/10 rounded-xl pl-10 pr-4 py-3 text-sm focus:border-primary focus:ring-1 focus:ring-primary outline-none transition-all"
                   />
                 </div>
               </div>
               <button 
                disabled
                className="w-full bg-white/10 text-neutral-400 py-3 rounded-xl font-medium cursor-not-allowed flex items-center justify-center gap-2"
              >
                URL Ingestion Coming Soon
              </button>
             </div>
          )}

          {status && (
            <div className="mt-4 p-3 bg-white/5 rounded-xl border border-white/10 text-sm text-neutral-300 flex items-start gap-3">
              <div className="shrink-0 mt-0.5">
                 {loading ? <Loader2 className="w-4 h-4 animate-spin text-primary" /> : <CheckCircle className="w-4 h-4 text-green-400" />}
              </div>
              <span className="break-all">{status}</span>
            </div>
          )}
        </div>
      </motion.div>
    </div>
  );
}
