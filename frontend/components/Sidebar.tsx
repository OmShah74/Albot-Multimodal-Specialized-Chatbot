import React, { useState, useRef, useEffect } from 'react';
import { Settings, Upload, MessageSquare, Plus, Database, Cpu, Trash2, PanelLeft, Edit2, Check, X, Key } from 'lucide-react';
import { cn } from '@/lib/utils';
import { ChatSession } from '@/lib/api';

interface SidebarProps {
  onOpenSettings: () => void;
  onOpenIngest: () => void;
  onOpenApiKeys: () => void;
  isCollapsed: boolean;
  onToggle: () => void;
  
  // Chat Management
  chats: ChatSession[];
  currentChatId: string | null;
  onSelectChat: (id: string) => void;
  onCreateChat: () => void;
  onDeleteChat: (id: string) => void;
  onRenameChat: (id: string, newTitle: string) => void;
}

export function Sidebar({ 
  onOpenSettings, 
  onOpenIngest, 
  onOpenApiKeys,
  isCollapsed, 
  onToggle,
  chats,
  currentChatId,
  onSelectChat,
  onCreateChat,
  onDeleteChat,
  onRenameChat
}: SidebarProps) {
  
  const [editingChatId, setEditingChatId] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState("");
  const editInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (editingChatId && editInputRef.current) {
      editInputRef.current.focus();
    }
  }, [editingChatId]);

  const startEditing = (e: React.MouseEvent, chat: ChatSession) => {
    e.stopPropagation();
    setEditingChatId(chat.id);
    setEditTitle(chat.title);
  };

  const saveEdit = (e?: React.FormEvent) => {
    e?.preventDefault();
    if (editingChatId && editTitle.trim()) {
      onRenameChat(editingChatId, editTitle.trim());
      setEditingChatId(null);
    }
  };

  const cancelEdit = (e: React.MouseEvent) => {
    e.stopPropagation();
    setEditingChatId(null);
  };

  const handleDelete = (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    onDeleteChat(id);
  };

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
          onClick={onCreateChat} 
          className={cn(
            "flex items-center gap-2 rounded-xl border border-white/10 hover:bg-white/5 transition-all text-sm font-medium hover:border-primary/50 group bg-primary/5",
            isCollapsed ? "p-3 justify-center" : "px-4 py-3 w-full"
          )}
        >
          <Plus className="w-4 h-4 text-primary group-hover:text-primary-glow" />
          {!isCollapsed && <span>New Chat</span>}
        </button>
      </div>

      <div className="flex-1 overflow-y-auto px-2 py-2 CustomScrollbar">
        <div className={cn("px-2", isCollapsed && "text-center")}>
          {!isCollapsed && <p className="text-xs font-medium text-neutral-500 uppercase tracking-wider px-2 mb-2">Recent</p>}
          
          <div className="space-y-1">
            {chats.length === 0 ? (
               <div className="text-center py-4 text-xs text-neutral-600 italic">
                 {isCollapsed ? "..." : "No recent chats"}
               </div>
            ) : (
              chats.map(chat => (
                <div 
                  key={chat.id}
                  onClick={() => onSelectChat(chat.id)}
                  className={cn(
                    "flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-all cursor-pointer group relative",
                    currentChatId === chat.id 
                      ? "bg-white/10 text-white shadow-lg border border-white/5" 
                      : "text-neutral-400 hover:text-white hover:bg-white/5"
                  )}
                >
                  <MessageSquare className={cn(
                    "w-4 h-4 shrink-0 transition-colors", 
                    currentChatId === chat.id ? "text-primary" : "text-neutral-500 group-hover:text-primary/70"
                  )} />
                  
                  {!isCollapsed && (
                    <div className="flex-1 overflow-hidden">
                      {editingChatId === chat.id ? (
                        <form onSubmit={saveEdit} className="flex items-center gap-1" onClick={e => e.stopPropagation()}>
                          <input 
                            ref={editInputRef}
                            type="text" 
                            value={editTitle}
                            onChange={e => setEditTitle(e.target.value)}
                            className="bg-black/50 border border-primary/50 rounded px-1 py-0.5 text-xs w-full focus:outline-none text-white"
                            onBlur={() => setEditingChatId(null)} 
                            onKeyDown={e => {
                                if (e.key === 'Escape') setEditingChatId(null);
                                if (e.key === 'Enter') saveEdit();
                            }}
                          />
                        </form>
                      ) : (
                        <span className="truncate block">{chat.title}</span>
                      )}
                    </div>
                  )}

                  {/* Hover Actions: Only show if not editing and not collapsed */}
                  {!isCollapsed && editingChatId !== chat.id && (
                    <div className="absolute right-2 opacity-0 group-hover:opacity-100 flex items-center gap-1 transition-opacity bg-[#0a0a0a]/80 backdrop-blur-sm rounded-lg p-0.5 shadow-xl">
                      <button 
                        onClick={(e) => startEditing(e, chat)}
                        className="p-1 hover:bg-white/10 rounded text-neutral-400 hover:text-white transition-colors"
                        title="Rename"
                      >
                        <Edit2 className="w-3 h-3" />
                      </button>
                      <button 
                        onClick={(e) => handleDelete(e, chat.id)}
                        className="p-1 hover:bg-red-500/20 rounded text-neutral-400 hover:text-red-400 transition-colors"
                        title="Delete"
                      >
                         <Trash2 className="w-3 h-3" />
                      </button>
                    </div>
                  )}
                </div>
              ))
            )}
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
