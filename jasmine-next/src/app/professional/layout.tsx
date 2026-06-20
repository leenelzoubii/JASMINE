'use client';

import { useEffect, useState } from 'react';
import { useRouter, usePathname } from 'next/navigation';
import { ProfessionalSidebar } from '@/components/layout/professional-sidebar';
import { getCurrentUser } from '@/lib/auth';
import { NotificationBell } from '@/components/ui/notification-bell';
import { ToastContainer } from '@/components/ui/toast';
import { motion } from 'framer-motion';

export default function ProfessionalLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const router = useRouter();
  const pathname = usePathname();
  const [mounted, setMounted] = useState(false);
  const [checked, setChecked] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    if (!mounted || checked) return;
    setChecked(true);
    const user = getCurrentUser();
    if (!user) {
      router.push(`/login?returnUrl=${encodeURIComponent(pathname)}`);
    } else if (user.role !== 'professional') {
      router.push('/parent');
    }
  }, [mounted, pathname, checked, router]);

  if (!mounted) {
    return (
      <div className="min-h-screen flex items-center justify-center" style={{ backgroundColor: 'var(--background-alt)' }}>
        <div className="flex flex-col items-center gap-3">
          <div className="w-10 h-10 border-2 rounded-full animate-spin" style={{ borderColor: 'var(--primary-light)', borderTopColor: 'var(--primary)' }} />
          <p className="text-sm animate-pulse" style={{ color: 'var(--text-dim)' }}>Loading...</p>
        </div>
      </div>
    );
  }

  const user = getCurrentUser();
  if (!user) {
    return (
      <div className="min-h-screen flex items-center justify-center" style={{ backgroundColor: 'var(--background-alt)' }}>
        <div className="flex flex-col items-center gap-3">
          <div className="w-10 h-10 border-2 rounded-full animate-spin" style={{ borderColor: 'var(--primary-light)', borderTopColor: 'var(--primary)' }} />
          <p className="text-sm animate-pulse" style={{ color: 'var(--text-dim)' }}>Redirecting...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex" style={{ backgroundColor: 'var(--background-alt)' }}>
      <ToastContainer />
      <ProfessionalSidebar />
      <main className="flex-1 min-h-screen">
        <div
          className="sticky top-0 z-30 glass-card flex items-center justify-between px-6 py-3"
          style={{ borderBottom: '1px solid var(--border-light)' }}
        >
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full animate-pulse" style={{ backgroundColor: 'var(--risk-low)' }} />
            <span className="text-xs font-medium" style={{ color: 'var(--text-dim)' }}>System Online</span>
          </div>
          <NotificationBell />
        </div>
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
          className="p-6 lg:p-8"
        >
          {children}
        </motion.div>
      </main>
    </div>
  );
}
