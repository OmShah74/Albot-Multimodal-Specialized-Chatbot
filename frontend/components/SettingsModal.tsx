import React, { useState, useEffect } from 'react';
// Native confirm/alert replaced with custom NotificationModal
import { X, Database, Trash2, RefreshCw, FileText, Music, Video, Table, Link as LinkIcon, Image, FileJson } from 'lucide-react';
import { api } from '@/lib/api';
import { cn } from '@/lib/utils';
import { motion } from 'framer-motion';
import { useNotification } from '@/context/NotificationContext';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export function SettingsModal({ isOpen, onClose }: SettingsModalProps) {
  const [sources, setSources] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [stats, setStats] = useState<any>(null);

  const fetchData = async () => {
    try {
      setLoading(true);
      const [sourcesData, statsData] = await Promise.all([
        api.listSources(),
        api.getStats()
      ]);
      setSources(sourcesData || []);
      setStats(statsData);
    } catch (e) {
      console.error('Failed to fetch data', e);
      // Fallback if statistics fails but sources might work
      try {
        const sourcesData = await api.listSources();
        setSources(sourcesData || []);
      } catch (innerE) {
        console.error('Failed to fetch sources too', innerE);
      }
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (isOpen) {
      fetchData();
    }
  }, [isOpen]);

  const { showNotification } = useNotification();

  const handleDeleteSource = async (name: string) => {
    showNotification({
      type: 'confirm',
      title: 'Delete Source',
      message: `Are you sure you want to delete "${name}" and all its associated data? This action cannot be undone.`,
      confirmText: 'Delete',
      onConfirm: async () => {
        try {
          await api.deleteSource(name);
          showNotification({
            type: 'success',
            title: 'Source Deleted',
            message: `"${name}" has been successfully removed from the knowledge base.`
          });
          fetchData();
        } catch (err) {
          console.error('Failed to delete source', err);
          showNotification({
            type: 'error',
            title: 'Deletion Failed',
            message: `Could not delete "${name}". Please check the system logs.`
          });
        }
      }
    });
  };

  const getFileIcon = (name: string) => {
    const ext = name.split('.').pop()?.toLowerCase();
    if (['jpg', 'jpeg', 'png', 'gif', 'webp'].includes(ext || '')) return <Image className="w-4 h-4 text-blue-400" />;
    if (['mp3', 'wav', 'm4a', 'ogg'].includes(ext || '')) return <Music className="w-4 h-4 text-green-400" />;
    if (['mp4', 'avi', 'mov', 'mkv'].includes(ext || '')) return <Video className="w-4 h-4 text-red-400" />;
    if (['csv', 'tsv', 'xlsx', 'xls'].includes(ext || '')) return <Table className="w-4 h-4 text-yellow-400" />;
    if (['json'].includes(ext || '')) return <FileJson className="w-4 h-4 text-orange-400" />;
    if (name.startsWith('http')) return <LinkIcon className="w-4 h-4 text-purple-400" />;
    return <FileText className="w-4 h-4 text-neutral-400" />;
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm">
      <motion.div 
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.95 }}
        className="w-full max-w-3xl bg-[#0a0a0a] border border-white/10 rounded-2xl shadow-2xl overflow-hidden flex flex-col max-h-[90vh]"
      >
        <div className="p-4 border-b border-white/10 flex justify-between items-center bg-white/5 shrink-0">
          <h2 className="font-semibold flex items-center gap-2 text-white">
            <Database className="w-4 h-4 text-primary" />
            Knowledge Base & Settings
          </h2>
          <div className="flex items-center gap-4">
            <button onClick={fetchData} className="text-neutral-400 hover:text-white transition-colors">
              <RefreshCw className={cn("w-4 h-4", loading && "animate-spin")} />
            </button>
            <button onClick={onClose} className="p-1 hover:bg-white/10 rounded-lg transition-colors text-neutral-400">
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-6 space-y-8">
          {/* Stats section */}
          {stats && (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="bg-white/5 border border-white/5 p-4 rounded-xl">
                 <p className="text-[10px] uppercase tracking-wider text-neutral-500 font-bold mb-1">Total Atoms</p>
                 <p className="text-2xl font-bold text-white">{stats.database.total_nodes}</p>
              </div>
              <div className="bg-white/5 border border-white/5 p-4 rounded-xl">
                 <p className="text-[10px] uppercase tracking-wider text-neutral-500 font-bold mb-1">Total Edges</p>
                 <p className="text-2xl font-bold text-white">{stats.database.total_edges}</p>
              </div>
            </div>
          )}

          <div className="space-y-4">
            <h3 className="text-sm font-medium text-white/70">Ingested Media (Documents, Audio, Video, Datasets)</h3>
            <div className="grid gap-2">
              {sources.length === 0 ? (
                <div className="text-center py-12 bg-white/5 border border-dashed border-white/10 rounded-2xl text-neutral-600 italic text-sm">
                  No media ingested yet.
                </div>
              ) : (
                sources.map((source) => (
                  <div key={source} className="flex items-center justify-between p-3 bg-white/5 rounded-xl border border-white/5 group hover:border-white/10 transition-colors">
                    <div className="flex items-center gap-3 overflow-hidden">
                      <div className="w-8 h-8 rounded-lg bg-black/40 flex items-center justify-center shrink-0">
                        {getFileIcon(source)}
                      </div>
                      <span className="text-sm text-neutral-300 truncate" title={source}>{source}</span>
                    </div>
                    <button 
                      onClick={() => handleDeleteSource(source)}
                      className="p-2 text-neutral-600 hover:text-red-400 transition-colors opacity-0 group-hover:opacity-100"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      </motion.div>
    </div>
  );
}
