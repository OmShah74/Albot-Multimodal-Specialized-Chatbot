"use client";

import React, { useState } from 'react';
import { Sidebar } from '@/components/Sidebar';
import { ChatInterface } from '@/components/ChatInterface';
import { SettingsModal } from '@/components/SettingsModal';
import { IngestModal } from '@/components/IngestModal';
import { AnimatePresence } from 'framer-motion';

export default function Home() {
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isIngestOpen, setIsIngestOpen] = useState(false);

  return (
    <main className="flex h-screen bg-black text-white overflow-hidden relative selection:bg-primary/30 selection:text-white">
      {/* Background Gradient */}
      <div className="absolute inset-0 z-0 bg-purple-glow opacity-30 pointer-events-none" />
      
      {/* Sidebar */}
      <div className="z-10 relative">
         <Sidebar 
            onOpenSettings={() => setIsSettingsOpen(true)}
            onOpenIngest={() => setIsIngestOpen(true)}
         />
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 z-10 relative flex flex-col h-full bg-[#050505]/50 backdrop-blur-sm">
         <ChatInterface />
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
      </AnimatePresence>
    </main>
  );
}
