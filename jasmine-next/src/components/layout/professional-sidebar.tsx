'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { ThemeToggle } from '@/components/ui/theme-toggle';
import { Brain, LayoutDashboard, Users, FileText, MessageSquare, User, LogOut, Menu, X, Link2, UserPlus } from 'lucide-react';
import { useState, useEffect } from 'react';
import { logoutUser, getCurrentUser } from '@/lib/auth';

const professionalLinks = [
  { href: '/professional', label: 'Dashboard', icon: LayoutDashboard },
  { href: '/professional/patients', label: 'Patients', icon: Users },
  { href: '/professional/assessments', label: 'Assessments', icon: FileText },
  { href: '/professional/messages', label: 'Messages', icon: MessageSquare },
  { href: '/professional/requests', label: 'Parent Requests', icon: UserPlus },
  { href: '/professional/profile', label: 'Profile', icon: User },
];

export function ProfessionalSidebar() {
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
        className="lg:hidden fixed top-4 left-4 z-50 p-2 rounded-lg"
        style={{ backgroundColor: 'var(--background)', border: '1px solid var(--border)' }}
        onClick={() => setIsOpen(!isOpen)}
      >
        {isOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
      </button>

      <aside className={`fixed top-0 left-0 h-full w-64 flex flex-col z-40 ${isOpen ? 'translate-x-0' : '-translate-x-full'} lg:translate-x-0`} style={{ backgroundColor: 'var(--background)', borderRight: '1px solid var(--border)' }}>
        {/* Logo */}
        <div className="p-6" style={{ borderBottom: '1px solid var(--border)' }}>
          <Link href="/professional" className="flex items-center gap-2">
            <div className="w-10 h-10 rounded-xl flex items-center justify-center" style={{ backgroundColor: 'var(--primary)' }}>
              <Brain className="w-6 h-6 text-white" />
            </div>
            <span className="text-lg font-bold" style={{ color: 'var(--primary)' }}>JASMINE</span>
          </Link>
          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Professional Portal</p>
        </div>

        {/* Nav Links */}
        <nav className="flex-1 p-4 space-y-1">
          {professionalLinks.map((link) => {
            const isActive = pathname === link.href;
            return (
              <Link
                key={link.href}
                href={link.href}
                onClick={() => setIsOpen(false)}
                className="flex items-center gap-3 px-4 py-3 rounded-xl transition-all"
                style={{
                  backgroundColor: isActive ? 'var(--primary-light)' : 'transparent',
                  color: isActive ? 'var(--primary)' : 'var(--foreground)'
                }}
              >
                <link.icon className="w-5 h-5" />
                <span className="font-medium">{link.label}</span>
              </Link>
            );
          })}
        </nav>

        {/* Bottom Section */}
        <div className="p-4" style={{ borderTop: '1px solid var(--border)' }}>
          <div className="flex items-center justify-between mb-4">
            <ThemeToggle />
            <button
              onClick={handleLogout}
              className="flex items-center gap-2 px-3 py-2 text-sm rounded-lg"
              style={{ color: '#dc2626' }}
            >
              <LogOut className="w-4 h-4" />
              Logout
            </button>
          </div>
          <div className="flex items-center gap-3 p-3 rounded-xl" style={{ backgroundColor: 'var(--background-alt)' }}>
            {user ? (
              <>
                <div className="w-10 h-10 rounded-full flex items-center justify-center text-white font-bold" style={{ backgroundColor: 'var(--primary)' }}>
                  {user.name.charAt(0).toUpperCase()}
                </div>
                <div>
                  <p className="text-sm font-medium" style={{ color: 'var(--foreground)' }}>{user.name}</p>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{user.specialty || 'Professional'}</p>
                </div>
              </>
            ) : (
              <div className="w-10 h-10 rounded-full bg-gray-200 dark:bg-dark-deep animate-pulse" />
            )}
          </div>
        </div>
      </aside>

      {isOpen && (
        <div className="lg:hidden fixed inset-0 bg-black/50 z-30" onClick={() => setIsOpen(false)} />
      )}
    </>
  );
}