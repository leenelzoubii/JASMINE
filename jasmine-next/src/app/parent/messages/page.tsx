'use client';

import { useState, useEffect, useRef } from 'react';
import { Send, Phone, Mail } from 'lucide-react';
import { motion } from 'framer-motion';
import { getCurrentUser } from '@/lib/auth';
import { getUserConnections } from '@/lib/parent-requests';
import {
  sendMessage,
  subscribeToMessages,
  markConversationAsRead,
  markMessagesDelivered,
  Message,
} from '@/lib/messages';
import { addNotification } from '@/lib/notifications';

function StatusTicks({ status }: { status?: string }) {
  if (!status || status === 'sent') {
    return (
      <span className="inline-flex items-center" style={{ color: 'var(--text-muted)' }}>
        <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <polyline points="20 6 9 17 4 12" />
        </svg>
      </span>
    );
  }
  if (status === 'delivered') {
    return (
      <span className="inline-flex items-center" style={{ color: 'var(--text-muted)' }}>
        <svg className="w-4 h-4 -mr-1" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <polyline points="20 6 9 17 4 12" />
        </svg>
        <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <polyline points="20 6 9 17 4 12" />
        </svg>
      </span>
    );
  }
  return (
    <span className="inline-flex items-center" style={{ color: '#3b82f6' }}>
      <svg className="w-4 h-4 -mr-1" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <polyline points="20 6 9 17 4 12" />
      </svg>
      <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <polyline points="20 6 9 17 4 12" />
      </svg>
    </span>
  );
}

export default function ParentMessagesPage() {
  const [selectedChat, setSelectedChat] = useState<string | null>(null);
  const [newMessage, setNewMessage] = useState('');
  const [mounted, setMounted] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [connections, setConnections] = useState<any[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const currentUser = mounted ? getCurrentUser() : null;

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    if (!currentUser) return;
    getUserConnections(currentUser.id)
      .then(setConnections)
      .catch(console.error);
  }, [currentUser]);

  // Clear all unread counts when visiting messages page
  useEffect(() => {
    if (!currentUser || connections.length === 0) return;
    connections.forEach(async (conn) => {
      const otherUserId = conn.professionalId || conn.parentId;
      if (otherUserId && otherUserId !== currentUser.id) {
        await markConversationAsRead(currentUser.id, otherUserId);
      }
    });
  }, [currentUser, connections]);

  useEffect(() => {
    if (!selectedChat || !currentUser) return;

    markMessagesDelivered(currentUser.id, selectedChat);
    markConversationAsRead(currentUser.id, selectedChat);

    const unsub = subscribeToMessages(currentUser.id, selectedChat, setMessages);
    return () => unsub();
  }, [selectedChat, currentUser]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const activeConnection = connections.find(
    (c) => c.parentId === currentUser?.id && c.professionalId === selectedChat
  );

  const handleSend = async () => {
    if (!newMessage.trim() || !currentUser || !selectedChat) return;
    try {
      await sendMessage(currentUser.id, selectedChat, newMessage.trim());
      await addNotification({
        userId: selectedChat,
        type: 'message',
        title: 'New Message',
        message: `${currentUser.name} sent you a message.`,
        link: '/professional/messages',
      });
      setNewMessage('');
    } catch (err) {
      console.error('Failed to send:', err);
    }
  };

  const handleSelectChat = async (otherUserId: string) => {
    setSelectedChat(otherUserId);
    if (currentUser) {
      await markConversationAsRead(currentUser.id, otherUserId);
    }
  };

  if (!mounted) {
    return (
      <div className="h-[calc(100vh-8rem)]">
        <div className="flex h-full rounded-2xl border animate-pulse" style={{ backgroundColor: 'var(--background)', borderColor: 'var(--border)' }} />
      </div>
    );
  }

  return (
    <div className="h-[calc(100vh-8rem)]">
      <div className="flex h-full rounded-2xl border overflow-hidden" style={{ backgroundColor: 'var(--background)', borderColor: 'var(--border)' }}>
        <div className="w-80 flex flex-col" style={{ borderRight: '1px solid var(--border)' }}>
          <div className="p-4" style={{ borderBottom: '1px solid var(--border)' }}>
            <h2 className="text-lg font-semibold" style={{ color: 'var(--foreground)' }}>Messages</h2>
            <p className="text-sm" style={{ color: 'var(--text-muted)' }}>Contact your healthcare provider</p>
          </div>
          <div className="flex-1 overflow-y-auto">
            {connections.length === 0 ? (
              <div className="py-8 text-center px-4">
                <p className="text-sm" style={{ color: 'var(--text-muted)' }}>No conversations yet.</p>
                <p className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>Accept a connection request from a professional to start messaging.</p>
              </div>
            ) : (
              connections.map((conn) => {
                const otherId = conn.professionalId;
                const otherName = conn.professionalName || 'Professional';
                const initials = otherName.split(' ').map((n: string) => n[0]).join('').toUpperCase();
                const isActive = selectedChat === otherId;
                return (
                  <button key={conn.id} onClick={() => handleSelectChat(otherId)}
                    className="w-full p-4 flex items-center gap-3 text-left transition-colors"
                    style={{ backgroundColor: isActive ? 'rgba(74, 155, 184, 0.08)' : 'transparent' }}>
                    <div className="w-10 h-10 rounded-full flex items-center justify-center text-white font-semibold text-sm" style={{ backgroundColor: 'var(--primary)' }}>
                      {initials}
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="font-medium truncate" style={{ color: 'var(--foreground)' }}>{otherName}</p>
                      <p className="text-sm truncate" style={{ color: 'var(--text-muted)' }}>{conn.patientName ? `Your child: ${conn.patientName}` : 'Professional'}</p>
                    </div>
                  </button>
                );
              })
            )}
          </div>
        </div>

        <div className="flex-1 flex flex-col">
          {!selectedChat ? (
            <div className="flex-1 flex items-center justify-center">
              <p style={{ color: 'var(--text-muted)' }}>Select a conversation to start messaging</p>
            </div>
          ) : (
            <>
              <div className="p-4 flex items-center justify-between" style={{ borderBottom: '1px solid var(--border)' }}>
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-full flex items-center justify-center text-white font-semibold" style={{ backgroundColor: 'var(--primary)' }}>
                    {activeConnection?.professionalName?.charAt(0) || 'D'}
                  </div>
                  <div>
                    <p className="font-medium" style={{ color: 'var(--foreground)' }}>{activeConnection?.professionalName || 'Professional'}</p>
                    <p className="text-sm" style={{ color: 'var(--text-muted)' }}>Professional</p>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <a href={`tel:${activeConnection?.phone || ''}`} className="p-2 rounded-lg" style={{ backgroundColor: '#15803d', color: 'white' }}><Phone className="w-5 h-5" /></a>
                  <a href={`mailto:${activeConnection?.email || ''}`} className="p-2 rounded-lg" style={{ backgroundColor: '#1e40af', color: 'white' }}><Mail className="w-5 h-5" /></a>
                </div>
              </div>

              <div className="flex-1 overflow-y-auto p-4 space-y-4">
                {messages.length === 0 ? (
                  <div className="flex items-center justify-center h-full">
                    <p className="text-sm" style={{ color: 'var(--text-muted)' }}>No messages yet. Start a conversation!</p>
                  </div>
                ) : (
                  messages.map((msg) => (
                    <motion.div key={msg.id} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
                      className={`flex ${msg.senderId === currentUser?.id ? 'justify-end' : 'justify-start'}`}>
                      <div className="max-w-md px-4 py-3 rounded-2xl space-y-1"
                        style={{
                          backgroundColor: msg.senderId === currentUser?.id ? 'var(--primary)' : 'var(--background-alt)',
                          color: msg.senderId === currentUser?.id ? 'white' : 'var(--foreground)',
                          borderRadius: msg.senderId === currentUser?.id ? '18px 18px 4px 18px' : '18px 18px 18px 4px',
                        }}>
                        <p className="text-sm">{msg.text}</p>
                        <div className={`flex items-center justify-end gap-1 ${msg.senderId === currentUser?.id ? '' : 'hidden'}`}>
                          <StatusTicks status={msg.status} />
                        </div>
                      </div>
                      {msg.senderId !== currentUser?.id && msg.status === 'read' && (
                        <div className="flex items-center gap-1 mt-1 justify-end text-xs" style={{ color: '#3b82f6' }}>
                          <StatusTicks status="read" />
                          <span>Seen</span>
                        </div>
                      )}
                    </motion.div>
                  ))
                )}
                <div ref={messagesEndRef} />
              </div>

              <div className="p-4" style={{ borderTop: '1px solid var(--border)' }}>
                <div className="flex items-center gap-3">
                  <input type="text" value={newMessage} onChange={(e) => setNewMessage(e.target.value)}
                    placeholder="Type a message..." className="flex-1 px-4 py-3 rounded-xl"
                    style={{ backgroundColor: 'var(--background-alt)', border: '1px solid var(--border)', color: 'var(--foreground)' }}
                    onKeyDown={(e) => e.key === 'Enter' && handleSend()} />
                  <button onClick={handleSend} disabled={!newMessage.trim()}
                    className="p-3 text-white rounded-xl transition-all disabled:opacity-50" style={{ backgroundColor: 'var(--primary)' }}>
                    <Send className="w-5 h-5" />
                  </button>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
