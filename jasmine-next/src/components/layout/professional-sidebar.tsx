'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { ThemeToggle } from '@/components/ui/theme-toggle';
import {
  Brain, LayoutDashboard, Users, FileText, MessageSquare,
  User, LogOut, Menu, X, UserPlus, Activity
} from 'lucide-react';
import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { logoutUser, getCurrentUser } from '@/lib/auth';
import { useUnreadMessages } from '@/lib/use-unread-messages';

const professionalLinks = [
  { href: '/professional', label: 'Dashboard', icon: LayoutDashboard },
  { href: '/professional/patients', label: 'Patients', icon: Users },
  { href: '/professional/assessments', label: 'Assessments', icon: FileText },
  { href: '/professional/messages', label: 'Messages', icon: MessageSquare },
  { href: '/professional/requests', label: 'Parent Requests', icon: UserPlus },
  { href: '/professional/profile', label: 'Profile', icon: User },
];

const sidebarVariants = {
  open: { x: 0, transition: { type: 'spring', stiffness: 300, damping: 30 } },
  closed: { x: '-100%', transition: { type: 'spring', stiffness: 300, damping: 30 } },
};

export function ProfessionalSidebar() {
  const pathname = usePathname();
  const [isOpen, setIsOpen] = useState(false);
  const [mounted, setMounted] = useState(false);

  useEffect(() => { setMounted(true); }, []);

  const user = mounted ? getCurrentUser() : null;
  const unreadMessages = useUnreadMessages(user?.id || null);

  const handleLogout = async () => {
    await logoutUser();
    window.location.href = '/login?loggedout=true';
  };

  const sidebarContent = (
    <div className="flex flex-col h-full overflow-hidden">
      <div className="p-5 border-b flex-shrink-0" style={{ borderColor: 'var(--border-light)' }}>
        <div className="flex items-center gap-3">
          <motion.div
            whileHover={{ scale: 1.05, rotate: -3 }}
            className="w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0"
            style={{ background: 'var(--gradient-primary)' }}
          >
            <Brain className="w-6 h-6 text-white" />
          </motion.div>
          <div>
            <Link href="/professional" className="text-lg font-bold block" style={{ color: 'var(--foreground)' }}>
              JASMINE
            </Link>
            <p className="text-xs font-medium" style={{ color: 'var(--text-dim)' }}>
              <Activity className="w-3 h-3 inline mr-1" />
              Professional Portal
            </p>
          </div>
        </div>
      </div>

      <nav className="flex-1 min-h-0 p-3 space-y-0.5 overflow-y-auto">
        {professionalLinks.map((link) => {
          const isActive = pathname === link.href;
          return (
            <Link
              key={link.href}
              href={link.href}
              onClick={() => setIsOpen(false)}
              className="flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-200 group"
              style={{
                backgroundColor: isActive ? 'var(--primary-light)' : 'transparent',
                color: isActive ? 'var(--primary)' : 'var(--text-secondary)',
              }}
            >
              <motion.div
                whileHover={{ scale: 1.1 }}
                className="flex items-center gap-3 flex-1"
              >
                <link.icon className="w-5 h-5 flex-shrink-0" />
                <span className="font-medium text-sm">{link.label}</span>
              </motion.div>
              {link.href.includes('/messages') && unreadMessages > 0 && (
                <span
                  className="px-2 py-0.5 rounded-full text-[11px] font-bold text-white flex-shrink-0"
                  style={{ backgroundColor: 'var(--risk-high)' }}
                >
                  {unreadMessages > 99 ? '99+' : unreadMessages}
                </span>
              )}
              {isActive && (
                <motion.div
                  layoutId="sidebar-active"
                  className="absolute left-0 w-1 h-8 rounded-r-full"
                  style={{ background: 'var(--gradient-primary)' }}
                />
              )}
            </Link>
          );
        })}
      </nav>

      <div className="p-4 border-t flex-shrink-0" style={{ borderColor: 'var(--border-light)' }}>
        <div className="flex items-center justify-between mb-3 px-1">
          <ThemeToggle />
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={handleLogout}
            className="flex items-center gap-2 px-3 py-2 text-sm rounded-lg transition-colors"
            style={{ color: 'var(--risk-high)' }}
          >
            <LogOut className="w-4 h-4" />
            Logout
          </motion.button>
        </div>
        <div
          className="flex items-center gap-3 p-3 rounded-xl"
          style={{ backgroundColor: 'var(--background-alt)' }}
        >
          {user ? (
            <>
              <div
                className="w-10 h-10 rounded-full flex items-center justify-center text-white font-bold flex-shrink-0"
                style={{ background: 'var(--gradient-primary)' }}
              >
                {user.name.charAt(0).toUpperCase()}
              </div>
              <div className="min-w-0">
                <p className="text-sm font-medium truncate" style={{ color: 'var(--foreground)' }}>
                  {user.name}
                </p>
                <p className="text-xs truncate" style={{ color: 'var(--text-dim)' }}>
                  {user.specialty || 'Professional'}
                </p>
              </div>
            </>
          ) : (
            <div className="flex items-center gap-3 w-full">
              <div className="w-10 h-10 rounded-full skeleton" />
              <div className="flex-1 space-y-2">
                <div className="h-3 w-24 skeleton" />
                <div className="h-2.5 w-16 skeleton" />
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );

  return (
    <>
      <motion.button
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        className="lg:hidden fixed top-3 left-3 z-50 p-2.5 rounded-xl glass-card shadow-md"
        onClick={() => setIsOpen(!isOpen)}
        aria-label="Toggle sidebar"
      >
        {isOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
      </motion.button>

      {/* Desktop Sidebar */}
      <aside
        className="hidden lg:flex sticky top-0 w-64 flex-col flex-shrink-0"
        style={{
          backgroundColor: 'var(--background)',
          borderRight: '1px solid var(--border-light)',
          height: '100vh',
        }}
      >
        {sidebarContent}
      </aside>

      {/* Mobile Sidebar */}
      <AnimatePresence>
        {isOpen && (
          <>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="lg:hidden fixed inset-0 bg-black/40 backdrop-blur-sm z-40"
              onClick={() => setIsOpen(false)}
            />
            <motion.aside
              variants={sidebarVariants}
              initial="closed"
              animate="open"
              exit="closed"
              className="lg:hidden fixed top-0 left-0 h-full w-72 z-50"
              style={{
                backgroundColor: 'var(--background)',
                borderRight: '1px solid var(--border-light)',
                boxShadow: 'var(--shadow-lg)',
              }}
            >
              {sidebarContent}
            </motion.aside>
          </>
        )}
      </AnimatePresence>
    </>
  );
}
