'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { ThemeToggle } from '@/components/ui/theme-toggle';
import { Brain, Menu, X } from 'lucide-react';
import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const navLinks = [
  { href: '/', label: 'Home' },
  { href: '/about', label: 'About' },
  { href: '/about-creators', label: 'The Creators' },
  { href: '/login', label: 'Login' },
  { href: '/register', label: 'Register' },
];

export function Navbar() {
  const pathname = usePathname();
  const [isOpen, setIsOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 20);
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  return (
    <nav className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${scrolled ? 'glass-nav shadow-sm' : 'bg-transparent'}`}>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16 md:h-20">
          <Link href="/" className="flex items-center gap-3 group">
            <motion.div
              whileHover={{ scale: 1.05, rotate: -3 }}
              className="w-10 h-10 rounded-xl flex items-center justify-center"
              style={{ background: 'var(--gradient-primary)' }}
            >
              <Brain className="w-6 h-6 text-white" />
            </motion.div>
            <span className="text-xl font-bold" style={{ color: 'var(--primary)' }}>
              JASMINE
            </span>
          </Link>

          <div className="hidden md:flex items-center gap-8">
            {navLinks.map((link) => (
              <Link
                key={link.href}
                href={link.href}
                className="relative text-sm font-medium transition-colors py-1"
                style={{
                  color: pathname === link.href ? 'var(--primary)' : 'var(--text-muted)',
                }}
              >
                {link.label}
                {pathname === link.href && (
                  <motion.span
                    layoutId="nav-underline"
                    className="absolute -bottom-1 left-0 right-0 h-0.5 rounded-full"
                    style={{ background: 'var(--gradient-primary)' }}
                  />
                )}
              </Link>
            ))}
            <div className="pl-2 border-l" style={{ borderColor: 'var(--border)' }}>
              <ThemeToggle />
            </div>
          </div>

          <button
            className="md:hidden p-2.5 rounded-xl"
            style={{ backgroundColor: 'var(--background-alt)' }}
            onClick={() => setIsOpen(!isOpen)}
            aria-label="Toggle menu"
          >
            <motion.div
              animate={{ rotate: isOpen ? 90 : 0 }}
              transition={{ duration: 0.2 }}
            >
              {isOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </motion.div>
          </button>
        </div>
      </div>

      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="md:hidden overflow-hidden border-t"
            style={{ borderColor: 'var(--border)' }}
          >
            <div className="px-4 py-4 space-y-1" style={{ backgroundColor: 'var(--background)' }}>
              {navLinks.map((link) => (
                <Link
                  key={link.href}
                  href={link.href}
                  onClick={() => setIsOpen(false)}
                  className="flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-medium transition-all"
                  style={{
                    backgroundColor: pathname === link.href ? 'var(--primary-light)' : 'transparent',
                    color: pathname === link.href ? 'var(--primary)' : 'var(--foreground)',
                  }}
                >
                  {link.label}
                </Link>
              ))}
              <div className="pt-2 flex justify-center">
                <ThemeToggle />
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </nav>
  );
}
