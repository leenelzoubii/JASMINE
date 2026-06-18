'use client';

import { useState, useEffect, useRef } from 'react';
import { useRouter } from 'next/navigation';
import { Bell, CheckCheck } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { getCurrentUser } from '@/lib/auth';
import { subscribeToNotifications, subscribeToUnreadCount, markNotificationRead, markAllNotificationsRead } from '@/lib/notifications';
import type { Notification } from '@/lib/notifications';

function timeAgo(createdAt: unknown): string {
  if (!createdAt) return '';
  const now = Date.now();
  const ts =
    typeof createdAt === 'object' && createdAt !== null
      ? ('toMillis' in createdAt && typeof (createdAt as { toMillis: () => number }).toMillis === 'function'
          ? (createdAt as { toMillis: () => number }).toMillis()
          : 'seconds' in createdAt && typeof (createdAt as { seconds: number }).seconds === 'number'
            ? (createdAt as { seconds: number }).seconds * 1000
            : 0)
      : typeof createdAt === 'number'
        ? createdAt
        : 0;
  const diff = now - ts;
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return 'Just now';
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

export function NotificationBell() {
  const router = useRouter();
  const [mounted, setMounted] = useState(false);
  const [userId, setUserId] = useState<string | null>(null);
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const [unreadCount, setUnreadCount] = useState(0);
  const [showDropdown, setShowDropdown] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setMounted(true);
    const user = getCurrentUser();
    if (user) setUserId(user.id);
  }, []);

  useEffect(() => {
    if (!userId) return;
    const unsubNotifs = subscribeToNotifications(userId, setNotifications);
    const unsubCount = subscribeToUnreadCount(userId, setUnreadCount);
    return () => {
      unsubNotifs();
      unsubCount();
    };
  }, [userId]);

  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setShowDropdown(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleNotificationClick = async (n: Notification) => {
    if (!n.read) await markNotificationRead(n.id);
    if (n.link) router.push(n.link);
  };

  const handleMarkAllRead = async () => {
    if (userId) await markAllNotificationsRead(userId);
  };

  if (!mounted) return <div className="w-9 h-9 rounded-xl skeleton" />;

  return (
    <div className="relative" ref={dropdownRef}>
      <motion.button
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        onClick={() => setShowDropdown(!showDropdown)}
        className="relative p-2.5 rounded-xl transition-all"
        style={{ backgroundColor: 'var(--background-alt)' }}
        aria-label={`Notifications${unreadCount > 0 ? ` (${unreadCount} unread)` : ''}`}
      >
        <Bell className="w-5 h-5" style={{ color: 'var(--text-secondary)' }} />
        {unreadCount > 0 && (
          <span className="absolute -top-0.5 -right-0.5 min-w-[18px] h-[18px] rounded-full flex items-center justify-center text-[10px] font-bold text-white px-1"
            style={{ backgroundColor: 'var(--risk-high)' }}>
            {unreadCount > 99 ? '99+' : unreadCount}
          </span>
        )}
      </motion.button>

      <AnimatePresence>
        {showDropdown && (
          <motion.div
            initial={{ opacity: 0, y: 8, scale: 0.96 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 8, scale: 0.96 }}
            transition={{ duration: 0.15 }}
            className="absolute right-0 mt-2 w-80 rounded-2xl overflow-hidden z-50 glass-card shadow-lg"
            style={{ border: '1px solid var(--border-light)' }}
          >
            <div className="flex items-center justify-between p-4 border-b" style={{ borderColor: 'var(--border-light)' }}>
              <p className="font-semibold text-sm" style={{ color: 'var(--foreground)' }}>Notifications</p>
              {unreadCount > 0 && (
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={handleMarkAllRead}
                  className="flex items-center gap-1 px-2 py-1 rounded-lg text-xs font-medium transition-colors"
                  style={{ backgroundColor: 'var(--background-alt)', color: 'var(--primary)' }}
                >
                  <CheckCheck className="w-3.5 h-3.5" /> Mark all read
                </motion.button>
              )}
            </div>
            <div className="max-h-80 overflow-y-auto">
              {notifications.length === 0 ? (
                <div className="p-6 text-center">
                  <Bell className="w-8 h-8 mx-auto mb-2" style={{ color: 'var(--text-dim)' }} />
                  <p className="text-sm" style={{ color: 'var(--text-muted)' }}>No new notifications</p>
                </div>
              ) : (
                notifications.map((n) => (
                  <motion.button
                    key={n.id}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    onClick={() => handleNotificationClick(n)}
                    className="w-full text-left px-4 py-3 transition-colors hover:bg-[var(--background-alt)] flex items-start gap-3 border-b"
                    style={{ borderColor: 'var(--border-light)', opacity: n.read ? 0.6 : 1 }}
                  >
                    {!n.read && (
                      <span className="w-2 h-2 rounded-full flex-shrink-0 mt-1.5" style={{ backgroundColor: 'var(--primary)' }} />
                    )}
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium truncate" style={{ color: 'var(--foreground)' }}>
                        {n.title}
                      </p>
                      <p className="text-xs mt-0.5 line-clamp-2" style={{ color: 'var(--text-muted)' }}>
                        {n.message}
                      </p>
                      <p className="text-[10px] mt-1" style={{ color: 'var(--text-dim)' }}>
                        {timeAgo(n.createdAt)}
                      </p>
                    </div>
                  </motion.button>
                ))
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
