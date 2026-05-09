'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { ThemeToggle } from '@/components/ui/theme-toggle';
import { Brain, LayoutDashboard, Baby, FileText, MessageSquare, User, LogOut, Menu, X } from 'lucide-react';
import { useState, useEffect } from 'react';
import { logoutUser, getCurrentUser } from '@/lib/auth';

const parentLinks = [
  { href: '/parent', label: 'Dashboard', icon: LayoutDashboard },
  { href: '/parent/results', label: 'Results', icon: FileText },
  { href: '/parent/messages', label: 'Messages', icon: MessageSquare },
  { href: '/parent/profile', label: 'Profile', icon: User },
];

export function ParentSidebar() {
  const pathname = usePathname();
  const [isOpen, setIsOpen] = useState(false);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  const user = mounted ? getCurrentUser() : null;

  const handleLogout = async () => {
    await logoutUser();
    window.location.href = '/login?loggedout=true';
  };

  return (
    <>
      <button
        className="lg:hidden fixed top-4 left-4 z-50 p-2 rounded-lg bg-white dark:bg-dark-surface shadow-lg"
        onClick={() => setIsOpen(!isOpen)}
      >
        {isOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
      </button>

      <aside className={`fixed top-0 left-0 h-full w-64 bg-white dark:bg-dark-surface border-r border-gray-200 dark:border-dark-deep transform transition-transform lg:translate-x-0 ${isOpen ? 'translate-x-0' : '-translate-x-full'} z-40`}>
        <div className="flex flex-col h-full">
          {/* Logo */}
          <div className="p-6 border-b border-gray-200 dark:border-dark-deep">
            <Link href="/parent" className="flex items-center gap-2">
              <div className="w-10 h-10 rounded-xl bg-primary flex items-center justify-center">
                <Brain className="w-6 h-6 text-white" />
              </div>
              <span className="text-lg font-bold text-primary">JASMINE</span>
            </Link>
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">Parent Portal</p>
          </div>

          {/* Nav Links */}
          <nav className="flex-1 p-4 space-y-1">
            {parentLinks.map((link) => {
              const isActive = pathname === link.href;
              return (
                <Link
                  key={link.href}
                  href={link.href}
                  onClick={() => setIsOpen(false)}
                  className={`flex items-center gap-3 px-4 py-3 rounded-xl transition-all ${
                    isActive
                      ? 'bg-primary-light text-primary dark:bg-primary-dark/20 dark:text-primary-light'
                      : 'text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-dark-deep'
                  }`}
                >
                  <link.icon className="w-5 h-5" />
                  <span className="font-medium">{link.label}</span>
                </Link>
              );
            })}
          </nav>

          {/* Bottom Section */}
          <div className="p-4 border-t border-gray-200 dark:border-dark-deep">
            <div className="flex items-center justify-between mb-4">
              <ThemeToggle />
              <button
                onClick={handleLogout}
                className="flex items-center gap-2 px-3 py-2 text-sm text-gray-500 hover:text-red-500 rounded-lg hover:bg-red-50 dark:hover:bg-red-900/20"
              >
                <LogOut className="w-4 h-4" />
                Logout
              </button>
            </div>
            <div className="flex items-center gap-3 p-3 bg-gray-50 dark:bg-dark-bg rounded-xl">
              {user ? (
                <>
                  <div className="w-10 h-10 rounded-full bg-primary flex items-center justify-center text-white font-bold">
                    {user.name.charAt(0).toUpperCase()}
                  </div>
                  <div>
                    <p className="text-sm font-medium text-gray-900 dark:text-white">{user.name}</p>
                    <p className="text-xs text-gray-500 dark:text-gray-400">Parent</p>
                  </div>
                </>
              ) : (
                <div className="w-10 h-10 rounded-full bg-gray-200 dark:bg-dark-deep animate-pulse" />
              )}
            </div>
          </div>
        </div>
      </aside>

      {isOpen && (
        <div className="lg:hidden fixed inset-0 bg-black/50 z-30" onClick={() => setIsOpen(false)} />
      )}
    </>
  );
}