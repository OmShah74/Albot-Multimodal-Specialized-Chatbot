import React, { useState, useEffect } from 'react';
import { X, Key, Check, Loader2, Trash2, Zap, CheckCircle } from 'lucide-react';
import { api } from '@/lib/api';
import { cn } from '@/lib/utils';
import { motion } from 'framer-motion';
import { useNotification } from '@/context/NotificationContext';

interface ApiKeyModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export function ApiKeyModal({ isOpen, onClose }: ApiKeyModalProps) {
  const [provider, setProvider] = useState('OpenAI');
  const [name, setName] = useState('');
  const [key, setKey] = useState('');
  const [modelName, setModelName] = useState('');
  const [status, setStatus] = useState<{msg: string, type: 'success'|'error'} | null>(null);
  const [loading, setLoading] = useState(false);
  const [testStatuses, setTestStatuses] = useState<Record<string, 'testing' | 'success' | 'error'>>({});
  const [existingKeys, setExistingKeys] = useState<Record<string, Array<{name: string, key: string, model_name?: string}>>>({});

  useEffect(() => {
    if (isOpen) {
      fetchKeys();
    }
  }, [isOpen]);

  const fetchKeys = async () => {
    try {
      const keys = await api.getApiKeys();
      setExistingKeys(keys);
    } catch (e) {
      console.error('Failed to fetch keys', e);
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
      await api.addApiKey(provider, name, key, modelName);
      setStatus({ msg: 'Key saved persistently', type: 'success' });
      setName('');
      setKey('');
      setModelName('');
      fetchKeys();
      fetchKeys();
    } catch (e: any) {
      setStatus({ msg: e.response?.data?.detail || 'Failed to add key', type: 'error' });
    } finally {
      setLoading(false);
    }
  };

  const handleTest = async (prov: string, keyName: string) => {
    const keyId = `${prov}-${keyName}`;
    setTestStatuses(prev => ({ ...prev, [keyId]: 'testing' }));
    
    try {
      const res = await api.testApiKey(prov, keyName);
      if (res.status === 'success') {
        setTestStatuses(prev => ({ ...prev, [keyId]: 'success' }));
      } else {
        setTestStatuses(prev => ({ ...prev, [keyId]: 'error' }));
      }
    } catch (e) {
      console.error('Failed to test key', e);
      setTestStatuses(prev => ({ ...prev, [keyId]: 'error' }));
      showNotification({
          type: 'error',
          title: 'Validation Failed',
          message: `The API key "${keyName}" for ${prov} failed validation. Please check your credentials.`
      });
    }
  };

  const { showNotification } = useNotification();

  const handleDelete = async (prov: string, keyName: string) => {
    showNotification({
      type: 'confirm',
      title: 'Delete API Key',
      message: `Are you sure you want to delete the API key "${keyName}" for ${prov}? Actions that require this key may fail.`,
      confirmText: 'Delete',
      onConfirm: async () => {
        try {
          await api.deleteApiKey(prov, keyName);
          showNotification({
            type: 'success',
            title: 'Key Deleted',
            message: `API key "${keyName}" has been removed.`
          });
          fetchKeys();
        } catch (e) {
          console.error('Failed to delete key', e);
          showNotification({
            type: 'error',
            title: 'Error',
            message: 'Failed to delete the API key. Please try again.'
          });
        }
      }
    });
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
          <h2 className="font-semibold flex items-center gap-2 text-white">
            <Key className="w-4 h-4 text-primary" />
            API Keys
          </h2>
          <button onClick={onClose} className="p-1 hover:bg-white/10 rounded-lg transition-colors text-neutral-400">
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-6 space-y-8">
          <div className="space-y-4">
            <h3 className="text-sm font-medium text-white/70">Add Persistent Key</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <label className="text-xs font-medium text-neutral-400 uppercase">Provider</label>
                <select 
                  value={provider}
                  onChange={(e) => setProvider(e.target.value)}
                  className="w-full bg-black/50 border border-white/10 rounded-xl px-4 py-3 text-sm focus:border-primary focus:ring-1 focus:ring-primary outline-none transition-all text-white"
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
                  className="w-full bg-black/50 border border-white/10 rounded-xl px-4 py-3 text-sm focus:border-primary focus:ring-1 focus:ring-primary outline-none transition-all text-white"
                />
              </div>
              <div className="space-y-2">
                <label className="text-xs font-medium text-neutral-400 uppercase">API Key</label>
                <input 
                  type="password" 
                  value={key}
                  onChange={(e) => setKey(e.target.value)}
                  placeholder="sk-..."
                  className="w-full bg-black/50 border border-white/10 rounded-xl px-4 py-3 text-sm focus:border-primary focus:ring-1 focus:ring-primary outline-none transition-all text-white"
                />
              </div>
              <div className="space-y-2">
                <label className="text-xs font-medium text-neutral-400 uppercase">Model Name (Optional)</label>
                <input 
                  type="text" 
                  value={modelName}
                  onChange={(e) => setModelName(e.target.value)}
                  placeholder="gpt-4"
                  className="w-full bg-black/50 border border-white/10 rounded-xl px-4 py-3 text-sm focus:border-primary focus:ring-1 focus:ring-primary outline-none transition-all text-white"
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
              className="w-full bg-primary hover:bg-primary-hover text-white py-3 rounded-xl font-medium transition-all flex items-center justify-center gap-2 disabled:opacity-50"
            >
              {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Check className="w-4 h-4" />}
              Save Key
            </button>
          </div>

          <div className="space-y-4 pt-4 border-t border-white/10">
            <h3 className="text-sm font-medium text-white/70">Stored Keys</h3>
            <div className="grid gap-3">
              {Object.entries(existingKeys).flatMap(([prov, keys]) => 
                keys.map((k, i) => {
                  const keyId = `${prov}-${k.name}`;
                  const testStatus = testStatuses[keyId];
                  return (
                  <div key={`${prov}-${i}`} className={cn(
                    "flex items-center justify-between p-3 rounded-xl border transition-colors",
                    testStatus === 'success' ? "bg-green-500/10 border-green-500/30 shadow-[0_0_10px_rgba(34,197,94,0.1)]" : "bg-white/5 border-white/5"
                  )}>
                    <div className="flex items-center gap-3">
                      <div className={cn(
                        "w-8 h-8 rounded-lg flex items-center justify-center transition-colors",
                        testStatus === 'success' ? "bg-green-500/20 text-green-400" : "bg-white/5 text-neutral-400"
                      )}>
                        {testStatus === 'success' ? <CheckCircle className="w-4 h-4" /> : <Key className="w-4 h-4" />}
                      </div>
                      <div>
                        <div className="font-medium text-sm text-white flex items-center gap-2">
                          {k.name}
                          {testStatus === 'error' && <span className="text-[10px] bg-red-500/20 text-red-400 px-1.5 py-0.5 rounded font-bold tracking-wider">INVALID</span>}
                        </div>
                        <div className="text-xs text-neutral-500 uppercase">{prov} • {k.key}</div>
                      </div>
                    </div>
                    <div className="flex items-center gap-1">
                      <button 
                        onClick={() => handleTest(prov, k.name)} 
                        disabled={testStatus === 'testing'}
                        className={cn(
                          "p-2 rounded transition-colors flex items-center justify-center",
                          testStatus === 'testing' ? "text-primary opacity-50 cursor-not-allowed" : "text-neutral-500 hover:text-primary hover:bg-white/5"
                        )}
                        title="Test API Key"
                      >
                         {testStatus === 'testing' ? <Loader2 className="w-4 h-4 animate-spin" /> : <Zap className="w-4 h-4" />}
                      </button>
                      <button onClick={() => handleDelete(prov, k.name)} className="p-2 text-neutral-600 hover:text-red-400 hover:bg-red-500/10 rounded transition-colors" title="Delete API Key">
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                )})
              )}
            </div>
          </div>
        </div>
      </motion.div>
    </div>
  );
}
