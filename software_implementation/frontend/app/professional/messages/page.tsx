'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { Send, Search, Paperclip, Image as ImageIcon, X } from 'lucide-react';
import { motion } from 'framer-motion';
import { getCurrentUser } from '@/lib/auth';
import { getUserConnections } from '@/lib/parent-requests';
import { isDemoUser, getDemoConnections, getDemoMessages } from '@/lib/demo-data';
import {
  sendMessage,
  subscribeToMessages,
  markConversationAsRead,
  markMessagesDelivered,
  setTypingStatus,
  subscribeTypingStatus,
  Message,
} from '@/lib/messages';
import { addNotification } from '@/lib/notifications';
import { storage } from '@/lib/firebase';
import { ref, uploadBytes, getDownloadURL } from 'firebase/storage';

export default function ProfessionalMessagesPage() {
  const [selectedChat, setSelectedChat] = useState<string | null>(null);
  const [newMessage, setNewMessage] = useState('');
  const [mounted, setMounted] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [connections, setConnections] = useState<any[]>([]);
  const [isTyping, setIsTyping] = useState(false);
  const [uploadingImage, setUploadingImage] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const currentUser = mounted ? getCurrentUser() : null;

  useEffect(() => { setMounted(true); }, []);

  useEffect(() => {
    if (!currentUser) return;
    if (isDemoUser(currentUser.id)) {
      setConnections(getDemoConnections(currentUser.id));
    } else {
      getUserConnections(currentUser.id).then(setConnections).catch(console.error);
    }
  }, [currentUser]);

  useEffect(() => {
    if (!selectedChat || !currentUser) return;
    markMessagesDelivered(currentUser.id, selectedChat);
    markConversationAsRead(currentUser.id, selectedChat);
    const conversationId = [currentUser.id, selectedChat].sort().join("_");

    // Check demo messages
    const demoMsgs = getDemoMessages(conversationId);
    if (isDemoUser(currentUser.id) && demoMsgs.length > 0) {
      setMessages(demoMsgs as any);
    } else {
      const unsub = subscribeToMessages(currentUser.id, selectedChat, setMessages);
      return () => unsub();
    }
  }, [selectedChat, currentUser]);

  // Typing indicator subscription
  useEffect(() => {
    if (!selectedChat || !currentUser) return;
    const conversationId = [currentUser.id, selectedChat].sort().join("_");
    const unsub = subscribeTypingStatus(conversationId, selectedChat, setIsTyping);
    return () => unsub();
  }, [selectedChat, currentUser]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleTyping = useCallback(() => {
    if (!currentUser || !selectedChat) return;
    const conversationId = [currentUser.id, selectedChat].sort().join("_");
    setTypingStatus(currentUser.id, conversationId, true);
  }, [currentUser, selectedChat]);

  const handleSend = async () => {
    if (!(newMessage.trim() || uploadingImage) || !currentUser || !selectedChat) return;
    try {
      const conversationId = [currentUser.id, selectedChat].sort().join("_");
      setTypingStatus(currentUser.id, conversationId, false).catch(() => {});
      await sendMessage(currentUser.id, selectedChat, newMessage.trim());
      await addNotification({
        userId: selectedChat,
        type: 'message',
        title: 'New Message',
        message: `${currentUser.name} sent you a message.`,
        link: '/parent/messages',
      });
      setNewMessage('');
    } catch (err) { console.error(err); }
  };

  const handleImageUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || !currentUser || !selectedChat) return;
    if (!file.type.startsWith('image/')) {
      alert('Please select an image file.');
      return;
    }
    if (file.size > 5 * 1024 * 1024) {
      alert('Image must be under 5MB.');
      return;
    }
    setUploadingImage(true);
    try {
      const storagePath = `chat-images/${currentUser.id}_${selectedChat}/${Date.now()}_${file.name}`;
      const storageRef = ref(storage, storagePath);
      const snapshot = await uploadBytes(storageRef, file);
      const downloadUrl = await getDownloadURL(snapshot.ref);
      const conversationId = [currentUser.id, selectedChat].sort().join("_");
      setTypingStatus(currentUser.id, conversationId, false).catch(() => {});
      await sendMessage(currentUser.id, selectedChat, '', downloadUrl);
      await addNotification({
        userId: selectedChat,
        type: 'message',
        title: 'New Photo',
        message: `${currentUser.name} shared a photo.`,
        link: '/parent/messages',
      });
    } catch (err) {
      console.error('Upload failed:', err);
      alert('Failed to upload image.');
    } finally {
      setUploadingImage(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const handleSelectChat = (id: string) => {
    setSelectedChat(id);
    if (currentUser) markConversationAsRead(currentUser.id, id);
  };

  if (!mounted) {
    return <div className="h-[calc(100vh-8rem)]"><div className="flex h-full rounded-2xl border animate-pulse" style={{ backgroundColor: 'var(--background)', borderColor: 'var(--border)' }} /></div>;
  }

  const activeConnection = connections.find(c => c.parentId === selectedChat);
  const filteredConnections = connections.filter(c =>
    c.parentName?.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="h-[calc(100vh-8rem)]">
      <div className="flex h-full rounded-2xl border overflow-hidden" style={{ backgroundColor: 'var(--background)', borderColor: 'var(--border)' }}>
        {/* Contact list */}
        <div className="w-72 flex-col shrink-0 hidden sm:flex" style={{ borderRight: '1px solid var(--border)' }}>
          <div className="p-4" style={{ borderBottom: '1px solid var(--border)' }}>
            <h2 className="text-lg font-semibold" style={{ color: 'var(--foreground)' }}>Messages</h2>
            <div className="relative mt-3">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4" style={{ color: 'var(--text-muted)' }} />
              <input type="text" placeholder="Search..." value={searchQuery} onChange={e => setSearchQuery(e.target.value)}
                className="w-full pl-10 pr-4 py-2 rounded-lg text-sm"
                style={{ backgroundColor: 'var(--background-alt)', border: '1px solid var(--border)', color: 'var(--foreground)' }} />
            </div>
          </div>
          <div className="flex-1 overflow-y-auto">
            {filteredConnections.length === 0 ? (
              <div className="py-8 text-center px-4">
                <p className="text-sm" style={{ color: 'var(--text-muted)' }}>No conversations yet.</p>
              </div>
            ) : filteredConnections.map(conn => {
              const userId = conn.parentId;
              const name = conn.parentName || 'Parent';
              const active = selectedChat === userId;
              return (
                <button key={conn.id} onClick={() => handleSelectChat(userId)}
                  className="w-full p-4 flex items-center gap-3 text-left transition-colors"
                  style={{ backgroundColor: active ? 'rgba(74,155,184,0.08)' : 'transparent' }}>
                  <div className="w-10 h-10 rounded-full flex items-center justify-center text-white font-semibold text-sm shrink-0" style={{ backgroundColor: 'var(--primary)' }}>
                    {name.charAt(0)}
                  </div>
                  <div className="min-w-0">
                    <p className="font-medium truncate" style={{ color: 'var(--foreground)' }}>{name}</p>
                    <p className="text-sm truncate" style={{ color: 'var(--text-muted)' }}>{conn.patientName}</p>
                  </div>
                </button>
              );
            })}
          </div>
        </div>

        {/* Chat */}
        <div className="flex-1 flex flex-col min-w-0">
          {!selectedChat ? (
            <div className="flex-1 flex items-center justify-center"><p style={{ color: 'var(--text-muted)' }}>Select a conversation</p></div>
          ) : (
            <>
              <div className="p-4 flex items-center justify-between shrink-0" style={{ borderBottom: '1px solid var(--border)' }}>
                <div>
                  <p className="font-medium" style={{ color: 'var(--foreground)' }}>{activeConnection?.parentName || 'Parent'}</p>
                  <p className="text-sm" style={{ color: 'var(--text-muted)' }}>Patient: {activeConnection?.patientName || '—'}</p>
                </div>
              </div>

              <div className="flex-1 overflow-y-auto p-4 space-y-3">
                {messages.length === 0 ? (
                  <div className="flex items-center justify-center h-full"><p className="text-sm" style={{ color: 'var(--text-muted)' }}>No messages yet</p></div>
                ) : messages.map(msg => {
                  const isMine = msg.senderId === currentUser?.id;
                  return (
                    <motion.div key={msg.id} initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                      className={`flex ${isMine ? 'justify-end' : 'justify-start'}`}>
                      <div className="max-w-md px-4 py-2.5 rounded-2xl space-y-1"
                        style={{
                          backgroundColor: isMine ? 'var(--primary)' : 'var(--background-alt)',
                          color: isMine ? 'white' : 'var(--foreground)',
                          borderRadius: isMine ? '18px 18px 4px 18px' : '18px 18px 18px 4px',
                        }}>
                        {msg.imageUrl && (
                          <img src={msg.imageUrl} alt="Shared image" className="max-w-full rounded-lg mb-1 max-h-48 object-cover cursor-pointer"
                            onClick={() => window.open(msg.imageUrl, '_blank')} />
                        )}
                        {msg.text && <p className="text-sm">{msg.text}</p>}
                        {isMine && (
                          <p className="text-[11px] opacity-60 text-right">
                            {msg.status === 'read' ? 'Seen' : msg.status === 'delivered' ? 'Delivered' : 'Sent'}
                          </p>
                        )}
                      </div>
                    </motion.div>
                  );
                })}
                {/* Typing indicator */}
                {isTyping && (
                  <div className="flex justify-start">
                    <div className="px-4 py-2.5 rounded-2xl" style={{ backgroundColor: 'var(--background-alt)' }}>
                      <div className="flex gap-1">
                        <span className="w-2 h-2 rounded-full animate-bounce" style={{ backgroundColor: 'var(--text-muted)', animationDelay: '0ms' }} />
                        <span className="w-2 h-2 rounded-full animate-bounce" style={{ backgroundColor: 'var(--text-muted)', animationDelay: '150ms' }} />
                        <span className="w-2 h-2 rounded-full animate-bounce" style={{ backgroundColor: 'var(--text-muted)', animationDelay: '300ms' }} />
                      </div>
                    </div>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>

              <div className="p-4 shrink-0" style={{ borderTop: '1px solid var(--border)' }}>
                <div className="flex items-center gap-3">
                  <input
                    type="file" ref={fileInputRef} accept="image/*" className="hidden"
                    onChange={handleImageUpload}
                  />
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    disabled={uploadingImage}
                    className="p-3 rounded-xl transition-all shrink-0 disabled:opacity-50"
                    style={{ backgroundColor: 'var(--background-alt)' }}
                    title="Attach image"
                  >
                    {uploadingImage ? (
                      <div className="w-5 h-5 border-2 border-primary/30 border-t-primary rounded-full animate-spin" />
                    ) : (
                      <Paperclip className="w-5 h-5" style={{ color: 'var(--text-muted)' }} />
                    )}
                  </button>
                  <input type="text" value={newMessage} onChange={e => { setNewMessage(e.target.value); handleTyping(); }}
                    placeholder="Type a message..." className="flex-1 px-4 py-3 rounded-xl"
                    style={{ backgroundColor: 'var(--background-alt)', border: '1px solid var(--border)', color: 'var(--foreground)' }}
                    onKeyDown={e => { if (e.key === 'Enter') handleSend(); }} />
                  <button onClick={handleSend} disabled={!newMessage.trim() && !uploadingImage}
                    className="p-3 text-white rounded-xl transition-all disabled:opacity-50 shrink-0" style={{ backgroundColor: 'var(--primary)' }}>
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
