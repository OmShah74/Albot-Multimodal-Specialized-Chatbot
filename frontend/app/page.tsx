"use client";

import React, { useState, useEffect } from 'react';
import { Sidebar } from '@/components/Sidebar';
import { ChatInterface } from '@/components/ChatInterface';
import { api, ChatSession } from '@/lib/api';

import { SettingsModal } from '@/components/SettingsModal';
import { IngestModal } from '@/components/IngestModal';
import { ApiKeyModal } from '@/components/ApiKeyModal';
import { AnimatePresence } from 'framer-motion';
import { NotificationProvider, useNotification } from '@/context/NotificationContext';

export default function Home() {
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isIngestOpen, setIsIngestOpen] = useState(false);
  const [isApiKeyOpen, setIsApiKeyOpen] = useState(false);
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);
  
  // Chat State
  const [chats, setChats] = useState<ChatSession[]>([]);
  const [currentChatId, setCurrentChatId] = useState<string | null>(null);

  useEffect(() => {
    loadChats();
  }, []);

  const loadChats = async () => {
    try {
      const chatList = await api.getChats();
      setChats(chatList);
      
      // Do not auto-select recent chat. Begin with new chat interface.
      // If chats exist, they are loaded in sidebar, but main view is empty state.
    } catch (err) {
      console.error("Failed to load chats:", err);
    }
  };

  const handleCreateChat = async () => {
    try {
      const newChat = await api.createChat();
      setChats(prev => [newChat, ...prev]);
      setCurrentChatId(newChat.id);
      return newChat.id;
    } catch (err) {
      console.error("Failed to create chat:", err);
      return null;
    }
  };



  const handleRenameChat = async (id: string, newTitle: string) => {
    try {
      await api.renameChat(id, newTitle);
      setChats(prev => prev.map(c => c.id === id ? { ...c, title: newTitle } : c));
    } catch (err) {
      console.error("Failed to rename chat:", err);
    }
  };

  return (
    <NotificationProvider>
      <HomeContent 
          isSettingsOpen={isSettingsOpen} setIsSettingsOpen={setIsSettingsOpen}
          isIngestOpen={isIngestOpen} setIsIngestOpen={setIsIngestOpen}
          isApiKeyOpen={isApiKeyOpen} setIsApiKeyOpen={setIsApiKeyOpen}
          isSidebarCollapsed={isSidebarCollapsed} setIsSidebarCollapsed={setIsSidebarCollapsed}
          chats={chats} setChats={setChats}
          currentChatId={currentChatId} setCurrentChatId={setCurrentChatId}
          onRenameChat={handleRenameChat}
          onCreateChat={handleCreateChat}
          loadChats={loadChats}
      />
    </NotificationProvider>
  );
}

// Sub-component to consume Context
function HomeContent({
    isSettingsOpen, setIsSettingsOpen,
    isIngestOpen, setIsIngestOpen,
    isApiKeyOpen, setIsApiKeyOpen,
    isSidebarCollapsed, setIsSidebarCollapsed,
    chats, setChats,
    currentChatId, setCurrentChatId,
    onRenameChat, onCreateChat, loadChats
}: any) {
    const { showNotification } = useNotification();

    const handleDeleteChat = async (id: string) => {
        showNotification({
            type: 'confirm',
            title: 'Delete Chat',
            message: 'Are you sure you want to delete this chat? This action cannot be undone.',
            confirmText: 'Delete',
            onConfirm: async () => {
                try {
                    await api.deleteChat(id);
                    setChats((prev: ChatSession[]) => prev.filter(c => c.id !== id));
                    
                    if (currentChatId === id) {
                        setCurrentChatId(null);
                        // We can verify if we need to loadChats or just pick one locally
                        // But calling loadChats from parent is safe
                        loadChats();
                    }
                    
                    showNotification({
                        type: 'success',
                        title: 'Chat Deleted',
                        message: 'The chat session has been successfully removed.'
                    });
                } catch (err) {
                    console.error("Failed to delete chat:", err);
                    showNotification({
                        type: 'error',
                        title: 'Deletion Failed',
                        message: 'Could not delete the chat. Please try again.'
                    });
                }
            }
        });
    };

    const handleChatUpdated = (id: string, updates: any) => {
        setChats((prev: ChatSession[]) => prev.map(c => c.id === id ? { ...c, ...updates } : c));
    };

    return (
      <main className="flex h-screen bg-black text-white overflow-hidden relative selection:bg-primary/30 selection:text-white">
        {/* Background Gradient */}
        <div className="absolute inset-0 z-0 bg-purple-glow opacity-30 pointer-events-none" />
        
        {/* Sidebar */}
        <div className="z-10 relative">
           <Sidebar 
              isCollapsed={isSidebarCollapsed}
              onToggle={() => setIsSidebarCollapsed(!isSidebarCollapsed)}
              onOpenSettings={() => setIsSettingsOpen(true)}
              onOpenIngest={() => setIsIngestOpen(true)}
              onOpenApiKeys={() => setIsApiKeyOpen(true)}
              
              chats={chats}
              currentChatId={currentChatId}
              onSelectChat={setCurrentChatId}
              onCreateChat={onCreateChat}
              onDeleteChat={handleDeleteChat}
              onRenameChat={onRenameChat}
           />
        </div>

        {/* Main Chat Area */}
        <div className="flex-1 z-10 relative flex flex-col h-full bg-[#050505]/50 backdrop-blur-sm">
           <ChatInterface 
              chatId={currentChatId} 
              onChatUpdated={handleChatUpdated}
              onCreateSession={onCreateChat}
           />
        </div>

        {/* Modals */}
        <AnimatePresence>
          {isSettingsOpen && (
            <SettingsModal 
              isOpen={isSettingsOpen} 
              onClose={() => setIsSettingsOpen(false)} 
            />
          )}
          {isIngestOpen && (
            <IngestModal 
              isOpen={isIngestOpen} 
              onClose={() => setIsIngestOpen(false)} 
            />
          )}
          {isApiKeyOpen && (
            <ApiKeyModal 
              isOpen={isApiKeyOpen} 
              onClose={() => setIsApiKeyOpen(false)} 
            />
          )}
        </AnimatePresence>
      </main>
    );
}
