'use client';

import { useEffect, useState } from 'react';
import { useRouter, usePathname } from 'next/navigation';
import { ProfessionalSidebar } from '@/components/layout/professional-sidebar';
import { NotificationBell } from '@/components/ui/notification-bell';
import { getCurrentUser } from '@/lib/auth';

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
    }
  }, [mounted, pathname, checked, router]);

  if (!mounted) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="w-8 h-8 border-2 border-primary/30 border-t-primary rounded-full animate-spin" />
      </div>
    );
  }

  const user = getCurrentUser();
  if (!user) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="w-8 h-8 border-2 border-primary/30 border-t-primary rounded-full animate-spin" />
      </div>
    );
  }

  return (
    <div className="min-h-screen" style={{ backgroundColor: 'var(--background-alt)' }}>
      <ProfessionalSidebar />
      <main className="lg:pl-64 min-h-screen">
        {/* Top bar with notification bell */}
        <div className="sticky top-0 z-30 flex items-center justify-end px-6 py-3" style={{ backgroundColor: 'var(--background)', borderBottom: '1px solid var(--border)' }}>
          <NotificationBell />
        </div>
        <div className="p-6 lg:p-8">{children}</div>
      </main>
    </div>
  );
}