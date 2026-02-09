import React, { useEffect, useState } from 'react';
import { Settings, Upload, MessageSquare, Plus, Database, Cpu, Trash2, RefreshCw, Key } from 'lucide-react';
import { cn } from '@/lib/utils';

interface SidebarProps {
  onOpenSettings: () => void;
  onOpenIngest: () => void;
  onOpenApiKeys: () => void;
}

export function Sidebar({ onOpenSettings, onOpenIngest, onOpenApiKeys }: SidebarProps) {
  return (
    <div className="w-64 h-screen border-r border-white/10 flex flex-col glass backdrop-blur-xl shrink-0">
      <div className="p-4 flex items-center gap-2 border-b border-white/10">
        <div className="w-8 h-8 rounded-lg bg-primary flex items-center justify-center shadow-[0_0_15px_rgba(147,51,234,0.5)]">
          <Database className="w-5 h-5 text-white" />
        </div>
        <h1 className="font-bold text-lg tracking-tight">Albot</h1>
      </div>

      <div className="p-4">
        <button 
          onClick={() => window.location.reload()} 
          className="w-full flex items-center gap-2 px-4 py-3 rounded-xl border border-white/10 hover:bg-white/5 transition-all text-sm font-medium hover:border-primary/50 group"
        >
          <Plus className="w-4 h-4 text-primary group-hover:text-primary-glow" />
          New Chat
        </button>
      </div>

      <div className="flex-1 overflow-y-auto px-2 py-2">
        <div className="px-4 py-2">
          <p className="text-xs font-medium text-neutral-500 uppercase tracking-wider">Recent</p>
          <div className="mt-4 text-center py-4 text-xs text-neutral-600 italic">
            No recent chats
          </div>
        </div>
      </div>

      <div className="p-4 border-t border-white/10 flex flex-col gap-1">
        <button 
          onClick={onOpenIngest}
          className="flex items-center gap-4 px-3 py-2.5 rounded-lg text-sm text-neutral-400 hover:text-white hover:bg-white/5 transition-colors group"
        >
          <Upload className="w-4 h-4 group-hover:text-primary transition-colors" />
          Ingest Knowledge
        </button>
        <button 
          onClick={onOpenApiKeys}
          className="flex items-center gap-4 px-3 py-2.5 rounded-lg text-sm text-neutral-400 hover:text-white hover:bg-white/5 transition-colors group"
        >
          <Key className="w-4 h-4 group-hover:text-primary transition-colors" />
          API Keys
        </button>
        <button 
          onClick={onOpenSettings}
          className="flex items-center gap-4 px-3 py-2.5 rounded-lg text-sm text-neutral-400 hover:text-white hover:bg-white/5 transition-colors group"
        >
          <Settings className="w-4 h-4 group-hover:text-primary transition-colors" />
          General Settings
        </button>

        <div className="flex items-center gap-3 px-3 py-2 mt-4 rounded-lg text-xs text-neutral-600 border-t border-white/5 pt-4">
           <Cpu className="w-3 h-3" />
           <span>Albot Core: v1.0.0</span>
        </div>
      </div>
    </div>
  );
}
