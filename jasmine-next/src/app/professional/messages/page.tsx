'use client';

import { useState, useEffect } from 'react';
import { Send, Search, MoreVertical } from 'lucide-react';
import { motion } from 'framer-motion';

const initialConversations = [
  { id: 1, name: 'John Thompson', lastMessage: 'Thank you for the update...', time: '2h ago', unread: 2, avatar: 'JT' },
  { id: 2, name: 'Sarah Johnson', lastMessage: 'When is the next appointment?', time: '1d ago', unread: 0, avatar: 'SJ' },
  { id: 3, name: 'Mike Williams', lastMessage: 'Got it, thank you!', time: '3d ago', unread: 0, avatar: 'MW' },
];

const initialMessages = [
  { id: 1, sender: 'parent', text: 'Hello Dr. Jasmine, I wanted to ask about Emma\'s assessment results.', time: '10:30 AM' },
  { id: 2, sender: 'professional', text: 'Hi John! The results show some areas we should monitor. I\'ll send the full report shortly.', time: '10:32 AM' },
  { id: 3, sender: 'parent', text: 'Thank you for the update. Should we schedule a follow-up?', time: '10:35 AM' },
  { id: 4, sender: 'professional', text: 'Yes, let\'s schedule one for next week. I\'ll send some available times.', time: '10:40 AM' },
  { id: 5, sender: 'parent', text: 'Thank you for the update. Should we schedule a follow-up?', time: '10:35 AM' },
  { id: 6, sender: 'professional', text: 'Yes, let\'s schedule one for next week. I\'ll send some available times.', time: '10:40 AM' },
];

export default function ProfessionalMessagesPage() {
  const [selectedChat, setSelectedChat] = useState(1);
  const [newMessage, setNewMessage] = useState('');
  const [mounted, setMounted] = useState(false);
  const [conversations, setConversations] = useState(initialConversations);
  const [messages, setMessages] = useState(initialMessages);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return (
      <div className="h-[calc(100vh-8rem)]">
        <div className="flex h-full bg-white dark:bg-dark-surface rounded-2xl border border-gray-200 dark:border-dark-deep animate-pulse" />
      </div>
    );
  }

  const activeConversation = conversations.find(c => c.id === selectedChat);

  const handleSend = () => {
    if (newMessage.trim()) {
      const now = new Date();
      const timeString = now.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' });
      const newMsg = { id: Date.now(), sender: 'professional', text: newMessage.trim(), time: timeString };
      setMessages(prev => [...prev, newMsg]);
      setConversations(prev => prev.map(conv => 
        conv.id === selectedChat 
          ? { ...conv, lastMessage: newMessage.trim(), time: 'Just now' }
          : conv
      ));
      setNewMessage('');
    }
  };

  return (
    <div className="h-[calc(100vh-8rem)]">
      <div className="flex h-full bg-white dark:bg-dark-surface rounded-2xl border border-gray-200 dark:border-dark-deep overflow-hidden">
        {/* Conversations List */}
        <div className="w-80 border-r border-gray-200 dark:border-dark-deep flex flex-col">
          <div className="p-4 border-b border-gray-100 dark:border-dark-deep">
            <h2 className="text-lg font-semibold text-gray-900 dark:text-white">Messages</h2>
            <div className="relative mt-3">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
              <input
                type="text"
                placeholder="Search messages..."
                className="w-full pl-10 pr-4 py-2 bg-gray-50 dark:bg-dark-bg border border-gray-200 dark:border-dark-deep rounded-lg text-sm focus:outline-none focus:ring-1 focus:ring-primary"
              />
            </div>
          </div>
          <div className="flex-1 overflow-y-auto">
            {conversations.map((conv) => (
              <button
                key={conv.id}
                onClick={() => setSelectedChat(conv.id)}
                className={`w-full p-4 flex items-center gap-3 text-left hover:bg-gray-50 dark:hover:bg-dark-deep transition-colors ${
                  selectedChat === conv.id ? 'bg-primary-light/30 dark:bg-primary-dark/20' : ''
                }`}
              >
                <div className="w-10 h-10 rounded-full bg-primary flex items-center justify-center text-white font-semibold text-sm">
                  {conv.avatar}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between">
                    <p className="font-medium text-gray-900 dark:text-white truncate">{conv.name}</p>
                    <span className="text-xs text-gray-400">{conv.time}</span>
                  </div>
                  <p className="text-sm text-gray-500 dark:text-gray-400 truncate">{conv.lastMessage}</p>
                </div>
                {conv.unread > 0 && (
                  <span className="w-5 h-5 rounded-full bg-primary text-white text-xs flex items-center justify-center">
                    {conv.unread}
                  </span>
                )}
              </button>
            ))}
          </div>
        </div>

        {/* Chat Area */}
        <div className="flex-1 flex flex-col">
          {/* Chat Header */}
          <div className="p-4 border-b border-gray-100 dark:border-dark-deep flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-full bg-primary flex items-center justify-center text-white font-semibold">
                {activeConversation?.avatar}
              </div>
              <div>
                <p className="font-medium text-gray-900 dark:text-white">{activeConversation?.name}</p>
                <p className="text-sm text-gray-500 dark:text-gray-400">Parent of patient</p>
              </div>
            </div>
            <button className="p-2 bg-primary/10 text-primary hover:bg-primary/20 rounded-lg transition-colors">
              <MoreVertical className="w-5 h-5" />
            </button>
          </div>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {messages.map((msg) => (
              <motion.div
                key={msg.id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className={`flex ${msg.sender === 'professional' ? 'justify-end' : 'justify-start'}`}
              >
                <div className={`max-w-md px-4 py-3 rounded-2xl ${
                  msg.sender === 'professional'
                    ? 'bg-primary text-white rounded-br-md'
                    : 'bg-gray-100 dark:bg-dark-deep text-gray-900 dark:text-white rounded-bl-md'
                }`}>
                  <p className="text-sm">{msg.text}</p>
                  <p className={`text-xs mt-1 ${msg.sender === 'professional' ? 'text-white/70' : 'text-gray-400'}`}>
                    {msg.time}
                  </p>
                </div>
              </motion.div>
            ))}
          </div>

          {/* Input */}
          <div className="p-4 border-t border-gray-100 dark:border-dark-deep">
            <div className="flex items-center gap-3">
              <input
                type="text"
                value={newMessage}
                onChange={(e) => setNewMessage(e.target.value)}
                placeholder="Type a message..."
                className="flex-1 px-4 py-3 bg-gray-50 dark:bg-dark-bg border border-gray-200 dark:border-dark-deep rounded-xl focus:outline-none focus:ring-2 focus:ring-primary"
                onKeyDown={(e) => e.key === 'Enter' && handleSend()}
              />
              <button
                onClick={handleSend}
                disabled={!newMessage.trim()}
                className="p-3 bg-primary hover:bg-primary-dark text-white rounded-xl transition-all disabled:opacity-50"
              >
                <Send className="w-5 h-5" />
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}