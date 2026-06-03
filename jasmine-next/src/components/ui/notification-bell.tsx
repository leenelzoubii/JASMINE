'use client';

import { useState, useEffect, useRef } from 'react';
import { Bell } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { getCurrentUser } from '@/lib/auth';

export function NotificationBell() {
  const [mounted, setMounted] = useState(false);
  const [companyName, setCompanyName] = useState('Dr. Sarah Chen');
  const [showDropdown, setShowDropdown] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setMounted(true);
    const user = getCurrentUser();
    if (user) setCompanyName(user.name);
  }, []);

  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setShowDropdown(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  if (!mounted) return <div className="w-9 h-9 rounded-xl skeleton" />;

  return (
    <div className="relative" ref={dropdownRef}>
      <motion.button
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        onClick={() => setShowDropdown(!showDropdown)}
        className="relative p-2.5 rounded-xl transition-all"
        style={{ backgroundColor: 'var(--background-alt)' }}
        aria-label="Notifications"
      >
        <Bell className="w-5 h-5" style={{ color: 'var(--text-secondary)' }} />
        <span className="absolute top-1.5 right-1.5 w-2 h-2 rounded-full animate-pulse" style={{ backgroundColor: 'var(--risk-high)' }} />
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
            <div className="p-4 border-b" style={{ borderColor: 'var(--border-light)' }}>
              <p className="font-semibold text-sm" style={{ color: 'var(--foreground)' }}>Notifications</p>
            </div>
            <div className="p-6 text-center">
              <Bell className="w-8 h-8 mx-auto mb-2" style={{ color: 'var(--text-dim)' }} />
              <p className="text-sm" style={{ color: 'var(--text-muted)' }}>No new notifications</p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
