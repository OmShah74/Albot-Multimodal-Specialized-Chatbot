"use client";

import React, { createContext, useContext, useState, useCallback, ReactNode } from 'react';
import { NotificationModal, NotificationType } from '@/components/NotificationModal';
import { AnimatePresence } from 'framer-motion';

interface NotificationOptions {
  type: NotificationType;
  title: string;
  message: string;
  confirmText?: string;
  cancelText?: string;
  onConfirm?: () => void;
}

interface NotificationContextType {
  showNotification: (options: NotificationOptions) => void;
}

const NotificationContext = createContext<NotificationContextType | undefined>(undefined);

export function NotificationProvider({ children }: { children: ReactNode }) {
  const [notification, setNotification] = useState<NotificationOptions | null>(null);
  const [isOpen, setIsOpen] = useState(false);

  const showNotification = useCallback((options: NotificationOptions) => {
    setNotification(options);
    setIsOpen(true);
  }, []);

  const handleClose = useCallback(() => {
    setIsOpen(false);
  }, []);

  return (
    <NotificationContext.Provider value={{ showNotification }}>
      {children}
      <AnimatePresence>
        {isOpen && notification && (
          <NotificationModal
            isOpen={isOpen}
            type={notification.type}
            title={notification.title}
            message={notification.message}
            confirmText={notification.confirmText}
            cancelText={notification.cancelText}
            onClose={handleClose}
            onConfirm={notification.onConfirm}
          />
        )}
      </AnimatePresence>
    </NotificationContext.Provider>
  );
}

export function useNotification() {
  const context = useContext(NotificationContext);
  if (context === undefined) {
    throw new Error('useNotification must be used within a NotificationProvider');
  }
  return context;
}
