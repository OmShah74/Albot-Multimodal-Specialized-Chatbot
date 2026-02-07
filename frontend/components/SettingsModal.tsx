import React, { useState } from 'react';
import { X, Key, Check, Loader2 } from 'lucide-react';
import { api } from '@/lib/api';
import { cn } from '@/lib/utils';
import { motion } from 'framer-motion';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export function SettingsModal({ isOpen, onClose }: SettingsModalProps) {
  const [provider, setProvider] = useState('OpenAI');
  const [name, setName] = useState('');
  const [key, setKey] = useState('');
  const [modelName, setModelName] = useState('');
  const [status, setStatus] = useState<{msg: string, type: 'success'|'error'} | null>(null);
  const [loading, setLoading] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [connectionError, setConnectionError] = useState('');
  const [apiUrl, setApiUrl] = useState('');
  
  // Define type for keys
  type ApiKeyData = {name: string, key: string, model_name?: string};
  const [existingKeys, setExistingKeys] = useState<Record<string, Array<ApiKeyData>>>({});

  // Fetch keys on open
  React.useEffect(() => {
    if (isOpen) {
      setApiUrl(process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000');
      checkConnection();
      fetchKeys();
    }
  }, [isOpen]);

  const checkConnection = async () => {
    try {
      await api.getStats(); // Simple health check
      setIsConnected(true);
      setConnectionError('');
    } catch (e: any) {
      setIsConnected(false);
      setConnectionError(e.message || 'Failed to connect');
    }
  };

  const fetchKeys = async () => {
    try {
      const keys = await api.getApiKeys();
      setExistingKeys(keys);
    } catch (e) {
      // already handled by checkConnection mainly, but silent fail here
    }
  };

  const handleSave = async () => {
    if (!name || !key) {
      setStatus({ msg: 'Name and Key are required', type: 'error' });
      return;
    }
    
    setLoading(true);
    setStatus(null);
    try {
      const res = await api.addApiKey(provider, name, key, modelName);
      if (typeof res === 'string' && res.startsWith('Error')) {
        setStatus({ msg: res, type: 'error' });
      } else {
        setStatus({ msg: 'Key added successfully', type: 'success' });
        // Clear fields on success
        setName('');
        setKey('');
        setModelName('');
        fetchKeys(); // Refresh list
      }
    } catch (e: any) {
      setStatus({ msg: e.response?.data?.detail || 'Failed to add key', type: 'error' });
    } finally {
      setLoading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm">
      <motion.div 
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.95 }}
        className="w-full max-w-2xl bg-[#0a0a0a] border border-white/10 rounded-2xl shadow-2xl overflow-hidden flex flex-col max-h-[90vh]"
      >
        <div className="p-4 border-b border-white/10 flex justify-between items-center bg-white/5 shrink-0">
          <h2 className="font-semibold flex items-center gap-2">
            <Key className="w-4 h-4 text-primary" />
            API Settings
          </h2>
          <div className="flex items-center gap-2">
            <div className={cn("w-2 h-2 rounded-full", isConnected ? "bg-green-500" : "bg-red-500")} title={isConnected ? "Connected" : `Disconnected: ${connectionError}`} />
             <span className="text-xs text-neutral-500 font-mono">{apiUrl.replace('http://', '')}</span>
            <button onClick={onClose} className="p-1 hover:bg-white/10 rounded-lg transition-colors">
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        {connectionError && (
            <div className="px-6 pt-6 pb-0">
                <div className="p-3 bg-red-500/10 border border-red-500/20 rounded-xl text-red-200 text-xs">
                    <strong>Connection Error:</strong> {connectionError}. <br/>
                    Ensure backend is running at <code>{apiUrl}</code>.
                </div>
            </div>
        )}

        <div className="flex-1 overflow-y-auto p-6 space-y-8">
          {/* Add New Key Section */}
          <div className="space-y-4">
            <h3 className="text-sm font-medium text-white/70">Add New Key</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <label className="text-xs font-medium text-neutral-400 uppercase">Provider</label>
                <select 
                  value={provider}
                  onChange={(e) => setProvider(e.target.value)}
                  className="w-full bg-black/50 border border-white/10 rounded-xl px-4 py-3 text-sm focus:border-primary focus:ring-1 focus:ring-primary outline-none transition-all"
                >
                  {['OpenAI', 'Anthropic', 'Groq', 'Gemini', 'OpenRouter'].map(p => (
                    <option key={p} value={p}>{p}</option>
                  ))}
                </select>
              </div>

              <div className="space-y-2">
                <label className="text-xs font-medium text-neutral-400 uppercase">Key Name</label>
                <input 
                  type="text" 
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="My Primary Key"
                  className="w-full bg-black/50 border border-white/10 rounded-xl px-4 py-3 text-sm focus:border-primary focus:ring-1 focus:ring-primary outline-none transition-all"
                />
              </div>

              <div className="space-y-2">
                <label className="text-xs font-medium text-neutral-400 uppercase">Model Name (Optional)</label>
                <input 
                  type="text" 
                  value={modelName}
                  onChange={(e) => setModelName(e.target.value)}
                  placeholder="e.g. gpt-4-turbo"
                  className="w-full bg-black/50 border border-white/10 rounded-xl px-4 py-3 text-sm focus:border-primary focus:ring-1 focus:ring-primary outline-none transition-all"
                />
              </div>

              <div className="space-y-2">
                <label className="text-xs font-medium text-neutral-400 uppercase">API Key</label>
                <input 
                  type="password" 
                  value={key}
                  onChange={(e) => setKey(e.target.value)}
                  placeholder="sk-..."
                  className="w-full bg-black/50 border border-white/10 rounded-xl px-4 py-3 text-sm focus:border-primary focus:ring-1 focus:ring-primary outline-none transition-all"
                />
              </div>
            </div>

            {status && (
              <div className={cn(
                "text-sm p-3 rounded-lg border",
                status.type === 'success' ? "bg-green-500/10 border-green-500/20 text-green-400" : "bg-red-500/10 border-red-500/20 text-red-400"
              )}>
                {status.msg}
              </div>
            )}

            <button 
              onClick={handleSave}
              disabled={loading}
              className="w-full bg-primary hover:bg-primary-hover text-white py-3 rounded-xl font-medium transition-all shadow-lg shadow-purple-500/20 flex items-center justify-center gap-2 disabled:opacity-50"
            >
              {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Check className="w-4 h-4" />}
              Save Configuration
            </button>
          </div>

          {/* Existing Keys Section */}
          <div className="space-y-4 pt-4 border-t border-white/10">
            <h3 className="text-sm font-medium text-white/70">Configured Keys</h3>
            <div className="grid gap-3">
              {Object.entries(existingKeys).flatMap(([prov, keys]) => 
                keys.map((k, i) => (
                  <div key={`${prov}-${i}`} className="flex items-center justify-between p-3 bg-white/5 rounded-xl border border-white/5">
                    <div className="flex items-center gap-3">
                      <div className="w-8 h-8 rounded-lg bg-white/5 flex items-center justify-center">
                        <Key className="w-4 h-4 text-neutral-400" />
                      </div>
                      <div>
                        <div className="font-medium text-sm text-white">{k.name}</div>
                        <div className="text-xs text-neutral-500 flex gap-2">
                           <span className="uppercase">{prov}</span>
                           <span>•</span>
                           <span className="font-mono">{k.key}</span>
                           {k.model_name && (
                             <>
                               <span>•</span>
                               <span className="text-primary/70">{k.model_name}</span>
                             </>
                           )}
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                       <span className="px-2 py-1 rounded-md bg-green-500/10 text-green-400 text-[10px] uppercase font-bold tracking-wider">
                         Active
                       </span>
                    </div>
                  </div>
                ))
              )}
              {Object.values(existingKeys).every(arr => arr.length === 0) && (
                <div className="text-center py-8 text-neutral-600 text-sm">
                  No API keys configured yet.
                </div>
              )}
            </div>
          </div>
        </div>
      </motion.div>
    </div>
  );
}
