'use client';

import { useTheme } from 'next-themes';
import { Sun, Moon } from 'lucide-react';
import { useEffect, useState } from 'react';

export function ThemeToggle() {
  const { theme, setTheme, resolvedTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return (
      <div className="w-10 h-10 rounded-lg bg-primary-light/30 animate-pulse" />
    );
  }

  const isDark = resolvedTheme === 'dark';
  const nextTheme = isDark ? 'light' : 'dark';

  return (
    <button
      onClick={() => setTheme(nextTheme)}
      className="p-2.5 rounded-xl bg-primary-light/50 dark:bg-dark-surface hover:bg-primary-light dark:hover:bg-dark-deep transition-all duration-200 group"
      aria-label={`Switch to ${nextTheme} mode`}
    >
      {isDark ? (
        <Sun className="w-5 h-5 text-primary-light group-hover:rotate-45 transition-transform duration-300" />
      ) : (
        <Moon className="w-5 h-5 text-primary-dark group-hover:-rotate-12 transition-transform duration-300" />
      )}
    </button>
  );
}