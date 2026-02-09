import React, { useEffect, useState } from 'react';
import { Settings, Upload, MessageSquare, Plus, Database, Cpu, Trash2, RefreshCw, Key, PanelLeft } from 'lucide-react';
import { cn } from '@/lib/utils';

interface SidebarProps {
  onOpenSettings: () => void;
  onOpenIngest: () => void;
  onOpenApiKeys: () => void;
  isCollapsed: boolean;
  onToggle: () => void;
}

export function Sidebar({ onOpenSettings, onOpenIngest, onOpenApiKeys, isCollapsed, onToggle }: SidebarProps) {
  return (
    <div className={cn(
      "h-screen border-r border-white/10 flex flex-col glass backdrop-blur-xl shrink-0 transition-all duration-300 relative",
      isCollapsed ? "w-20" : "w-64"
    )}>
      <div className="p-4 flex items-center justify-between border-b border-white/10">
        {!isCollapsed && (
          <div className="flex items-center gap-2 overflow-hidden">
            <div className="w-8 h-8 rounded-lg bg-primary flex items-center justify-center shadow-[0_0_15px_rgba(147,51,234,0.5)]">
              <Database className="w-5 h-5 text-white" />
            </div>
            <h1 className="font-bold text-lg tracking-tight">Albot</h1>
          </div>
        )}
        {isCollapsed && (
          <div className="mx-auto w-8 h-8 rounded-lg bg-primary flex items-center justify-center shadow-[0_0_15px_rgba(147,51,234,0.5)]">
            <Database className="w-5 h-5 text-white" />
          </div>
        )}
        <button 
          onClick={onToggle}
          className={cn(
            "p-1.5 hover:bg-white/5 rounded-lg transition-colors text-neutral-500 hover:text-white",
            isCollapsed && "absolute -right-3 top-20 bg-[#0a0a0a] border border-white/10 shadow-xl z-50 rounded-full"
          )}
        >
          <PanelLeft className={cn("w-4 h-4", isCollapsed && "rotate-180")} />
        </button>
      </div>

      <div className="p-4">
        <button 
          onClick={() => window.location.reload()} 
          className={cn(
            "flex items-center gap-2 rounded-xl border border-white/10 hover:bg-white/5 transition-all text-sm font-medium hover:border-primary/50 group",
            isCollapsed ? "p-3 justify-center" : "px-4 py-3 w-full"
          )}
        >
          <Plus className="w-4 h-4 text-primary group-hover:text-primary-glow" />
          {!isCollapsed && <span>New Chat</span>}
        </button>
      </div>

      <div className="flex-1 overflow-y-auto px-2 py-2">
        <div className={cn("px-4 py-2", isCollapsed && "px-2 text-center")}>
          {!isCollapsed && <p className="text-xs font-medium text-neutral-500 uppercase tracking-wider">Recent</p>}
          <div className="mt-4 text-center py-4 text-xs text-neutral-600 italic">
            {isCollapsed ? "..." : "No recent chats"}
          </div>
        </div>
      </div>

      <div className="p-4 border-t border-white/10 flex flex-col gap-1">
        <button 
          onClick={onOpenIngest}
          className={cn("flex items-center gap-4 px-3 py-2.5 rounded-lg text-sm text-neutral-400 hover:text-white hover:bg-white/5 transition-colors group", isCollapsed && "justify-center")}
        >
          <Upload className="w-4 h-4 group-hover:text-primary transition-colors" />
          {!isCollapsed && <span>Ingest Knowledge</span>}
        </button>
        <button 
          onClick={onOpenApiKeys}
          className={cn("flex items-center gap-4 px-3 py-2.5 rounded-lg text-sm text-neutral-400 hover:text-white hover:bg-white/5 transition-colors group", isCollapsed && "justify-center")}
        >
          <Key className="w-4 h-4 group-hover:text-primary transition-colors" />
          {!isCollapsed && <span>API Keys</span>}
        </button>
        <button 
          onClick={onOpenSettings}
          className={cn("flex items-center gap-4 px-3 py-2.5 rounded-lg text-sm text-neutral-400 hover:text-white hover:bg-white/5 transition-colors group", isCollapsed && "justify-center")}
        >
          <Settings className="w-4 h-4 group-hover:text-primary transition-colors" />
          {!isCollapsed && <span>General Settings</span>}
        </button>

        <div className={cn("flex items-center gap-3 px-3 py-2 mt-4 rounded-lg text-xs text-neutral-600 border-t border-white/5 pt-4", isCollapsed && "justify-center")}>
           <Cpu className="w-3 h-3" />
           {!isCollapsed && <span>Albot Core: v1.0</span>}
        </div>
      </div>
    </div>
  );
}
