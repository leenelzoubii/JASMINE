'use client';

import { useState, useEffect, useRef } from 'react';
import { Send, Search, MessageSquare } from 'lucide-react';
import { motion } from 'framer-motion';
import { getCurrentUser, User } from '@/lib/auth';
import { getUserConnections } from '@/lib/parent-requests';
import { sendMessage, subscribeToMessages, markConversationAsRead, markMessagesDelivered, Message } from '@/lib/messages';
import { addNotification } from '@/lib/notifications';

export default function ProfessionalMessagesPage() {
  const [selectedChat, setSelectedChat] = useState<string | null>(null);
  const [newMessage, setNewMessage] = useState('');
  const [currentUser, setCurrentUser] = useState<User | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [connections, setConnections] = useState<any[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => { setCurrentUser(getCurrentUser()); }, []);
  useEffect(() => {
    if (!currentUser) return;
    getUserConnections(currentUser.id).then(setConnections).catch(console.error);
  }, [currentUser]);
  useEffect(() => {
    if (!selectedChat || !currentUser) return;
    markMessagesDelivered(currentUser.id, selectedChat);
    markConversationAsRead(currentUser.id, selectedChat);
    const unsub = subscribeToMessages(currentUser.id, selectedChat, setMessages);
    return () => unsub();
  }, [selectedChat, currentUser]);
  useEffect(() => { messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages]);

  const handleSend = async () => {
    if (!newMessage.trim() || !currentUser || !selectedChat) return;
    try {
      await sendMessage(currentUser.id, selectedChat, newMessage.trim());
      await addNotification({ userId: selectedChat, type: 'message', title: 'New Message', message: `${currentUser.name} sent you a message.`, link: '/parent/messages' });
      setNewMessage('');
    } catch (err) { console.error(err); }
  };

  if (!currentUser) return <div className="h-[calc(100vh-8rem)]"><div className="flex h-full rounded-2xl premium-card"><div className="flex-1" /></div></div>;

  const activeConnection = connections.find(c => c.parentId === selectedChat);

  return (
    <div className="h-[calc(100vh-8rem)]">
      <div className="flex h-full premium-card overflow-hidden">
        <div className="w-72 flex flex-col shrink-0" style={{ borderRight: '1px solid var(--border-light)' }}>
          <div className="p-4 border-b" style={{ borderColor: 'var(--border-light)' }}>
            <h2 className="text-lg font-semibold" style={{ color: 'var(--foreground)' }}>Messages</h2>
            <div className="relative mt-3">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4" style={{ color: 'var(--text-dim)' }} />
              <input type="text" placeholder="Search..." className="premium-input py-2 pl-10 text-sm" />
            </div>
          </div>
          <div className="flex-1 overflow-y-auto">
            {connections.length === 0 ? (
              <div className="py-12 text-center px-4">
                <MessageSquare className="w-8 h-8 mx-auto mb-2" style={{ color: 'var(--text-dim)' }} />
                <p className="text-sm" style={{ color: 'var(--text-muted)' }}>No conversations yet.</p>
              </div>
            ) : connections.map(conn => {
              const userId = conn.parentId;
              const name = conn.parentName || 'Parent';
              const active = selectedChat === userId;
              return (
                <motion.button key={conn.id} whileHover={{ x: 2 }}
                  onClick={() => setSelectedChat(userId)}
                  className="w-full p-4 flex items-center gap-3 text-left transition-colors"
                  style={{ backgroundColor: active ? 'var(--primary-light)' : 'transparent' }}>
                  <div className="w-10 h-10 rounded-full flex items-center justify-center text-white font-semibold text-sm shrink-0" style={{ background: 'var(--gradient-primary)' }}>
                    {name.charAt(0)}
                  </div>
                  <div className="min-w-0">
                    <p className="font-medium text-sm truncate" style={{ color: 'var(--foreground)' }}>{name}</p>
                    <p className="text-xs truncate" style={{ color: 'var(--text-dim)' }}>{conn.patientName}</p>
                  </div>
                </motion.button>
              );
            })}
          </div>
        </div>

        <div className="flex-1 flex flex-col">
          {!selectedChat ? (
            <div className="flex-1 flex items-center justify-center">
              <div className="text-center">
                <MessageSquare className="w-12 h-12 mx-auto mb-3" style={{ color: 'var(--text-dim)' }} />
                <p style={{ color: 'var(--text-muted)' }}>Select a conversation to start messaging</p>
              </div>
            </div>
          ) : (
            <>
              <div className="p-4 border-b" style={{ borderColor: 'var(--border-light)' }}>
                <p className="font-medium" style={{ color: 'var(--foreground)' }}>{activeConnection?.parentName || 'Parent'}</p>
                <p className="text-sm" style={{ color: 'var(--text-dim)' }}>Patient: {activeConnection?.patientName || '—'}</p>
              </div>

              <div className="flex-1 overflow-y-auto p-4 space-y-3">
                {messages.length === 0 ? (
                  <div className="flex items-center justify-center h-full"><p className="text-sm" style={{ color: 'var(--text-muted)' }}>No messages yet. Send a message to start the conversation.</p></div>
                ) : messages.map(msg => {
                  const isMine = msg.senderId === currentUser?.id;
                  return (
                    <motion.div key={msg.id} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
                      className={`flex ${isMine ? 'justify-end' : 'justify-start'}`}>
                      <div className="max-w-md px-4 py-2.5 rounded-2xl space-y-0.5 shadow-sm"
                        style={{
                          backgroundColor: isMine ? 'var(--primary)' : 'var(--background-alt)',
                          color: isMine ? 'white' : 'var(--foreground)',
                          borderRadius: isMine ? '18px 18px 4px 18px' : '18px 18px 18px 4px',
                        }}>
                        <p className="text-sm">{msg.text}</p>
                        <p className="text-[11px] opacity-60 text-right">
                          {isMine ? (msg.status === 'read' ? 'Seen' : msg.status === 'delivered' ? 'Delivered' : 'Sent') : ''}
                        </p>
                      </div>
                    </motion.div>
                  );
                })}
                <div ref={messagesEndRef} />
              </div>

              <div className="p-4 border-t" style={{ borderColor: 'var(--border-light)' }}>
                <div className="flex items-center gap-3">
                  <input type="text" value={newMessage} onChange={e => setNewMessage(e.target.value)}
                    placeholder="Type a message..." className="premium-input py-3"
                    onKeyDown={e => e.key === 'Enter' && handleSend()} />
                  <motion.button whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}
                    onClick={handleSend} disabled={!newMessage.trim()}
                    className="p-3.5 text-white rounded-xl transition-all disabled:opacity-50 shrink-0"
                    style={{ background: 'var(--gradient-primary)' }}>
                    <Send className="w-5 h-5" />
                  </motion.button>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
