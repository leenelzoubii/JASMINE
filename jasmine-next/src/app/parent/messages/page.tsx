'use client';

import { useState } from 'react';
import { Send, MoreVertical, Phone, Mail } from 'lucide-react';
import { motion } from 'framer-motion';

const conversations = [
  { id: 1, name: 'Dr. Jasmine', lastMessage: 'The assessment results are ready...', time: '2h ago', unread: 1, avatar: 'DJ' },
];

const messages = [
  { id: 1, sender: 'professional', text: 'Hello! I wanted to let you know that Emma\'s latest screening results are ready.', time: '10:30 AM' },
  { id: 2, sender: 'parent', text: 'That\'s great news! What do the results show?', time: '10:32 AM' },
  { id: 3, sender: 'professional', text: 'There are some areas we should monitor. I\'ve attached the full report. Based on the assessment, I recommend scheduling a follow-up appointment to discuss in detail.', time: '10:35 AM' },
  { id: 4, sender: 'professional', text: 'Please find the report attached. Let me know if you have any questions.', time: '10:36 AM' },
];

export default function ParentMessagesPage() {
  const [newMessage, setNewMessage] = useState('');

  const handleSend = () => {
    if (newMessage.trim()) {
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
            <p className="text-sm text-gray-500 dark:text-gray-400">Contact your healthcare provider</p>
          </div>
          <div className="flex-1 overflow-y-auto">
            {conversations.map((conv) => (
              <button
                key={conv.id}
                className="w-full p-4 flex items-center gap-3 text-left bg-primary-light/30 dark:bg-primary-dark/20"
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
                DJ
              </div>
              <div>
                <p className="font-medium text-gray-900 dark:text-white">Dr. Jasmine</p>
                <p className="text-sm text-gray-500 dark:text-gray-400">Professional</p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <button className="p-2 text-gray-400 hover:text-primary rounded-lg hover:bg-gray-100 dark:hover:bg-dark-deep">
                <Phone className="w-5 h-5" />
              </button>
              <button className="p-2 text-gray-400 hover:text-primary rounded-lg hover:bg-gray-100 dark:hover:bg-dark-deep">
                <Mail className="w-5 h-5" />
              </button>
              <button className="p-2 text-gray-400 hover:text-gray-600 rounded-lg hover:bg-gray-100 dark:hover:bg-dark-deep">
                <MoreVertical className="w-5 h-5" />
              </button>
            </div>
          </div>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {messages.map((msg) => (
              <motion.div
                key={msg.id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className={`flex ${msg.sender === 'parent' ? 'justify-end' : 'justify-start'}`}
              >
                <div className={`max-w-md px-4 py-3 rounded-2xl ${
                  msg.sender === 'parent'
                    ? 'bg-primary text-white rounded-br-md'
                    : 'bg-gray-100 dark:bg-dark-deep text-gray-900 dark:text-white rounded-bl-md'
                }`}>
                  <p className="text-sm">{msg.text}</p>
                  <p className={`text-xs mt-1 ${msg.sender === 'parent' ? 'text-white/70' : 'text-gray-400'}`}>
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
                onKeyPress={(e) => e.key === 'Enter' && handleSend()}
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