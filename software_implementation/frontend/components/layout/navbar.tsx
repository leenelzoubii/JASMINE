'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { ThemeToggle } from '@/components/ui/theme-toggle';
import { Brain, Menu, X } from 'lucide-react';
import { useState } from 'react';

const navLinks = [
  { href: '/', label: 'Home' },
  { href: '/about', label: 'About' },
  { href: '/login', label: 'Login' },
  { href: '/register', label: 'Register' },
];

export function Navbar() {
  const pathname = usePathname();
  const [isOpen, setIsOpen] = useState(false);

  return (
    <nav className="fixed top-0 left-0 right-0 z-50" style={{ backgroundColor: 'var(--background)', borderBottom: '1px solid var(--border)' }}>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link href="/" className="flex items-center gap-2 group">
            <div className="w-10 h-10 rounded-xl flex items-center justify-center" style={{ backgroundColor: 'var(--primary)' }}>
              <Brain className="w-6 h-6 text-white" />
            </div>
            <span className="text-xl font-bold" style={{ color: 'var(--primary)' }}>
              JASMINE
            </span>
          </Link>

          {/* Desktop Nav */}
          <div className="hidden md:flex items-center gap-6">
            {navLinks.map((link) => (
              <Link
                key={link.href}
                href={link.href}
                className="text-sm font-medium transition-colors"
                style={{ 
                  color: pathname === link.href ? 'var(--primary)' : 'var(--text-muted)'
                }}
              >
                {link.label}
              </Link>
            ))}
            <ThemeToggle />
          </div>

          {/* Mobile Menu Button */}
          <button
            className="md:hidden p-2 rounded-lg"
            style={{ backgroundColor: 'var(--background-alt)' }}
            onClick={() => setIsOpen(!isOpen)}
          >
            {isOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          </button>
        </div>

        {/* Mobile Nav */}
        {isOpen && (
          <div className="md:hidden py-4 border-t" style={{ borderColor: 'var(--border)' }}>
            <div className="flex flex-col gap-2">
              {navLinks.map((link) => (
                <Link
                  key={link.href}
                  href={link.href}
                  onClick={() => setIsOpen(false)}
                  className="py-2 px-3 rounded-lg text-sm font-medium"
                  style={{ 
                    backgroundColor: pathname === link.href ? 'var(--primary-light)' : 'transparent',
                    color: pathname === link.href ? 'var(--primary)' : 'var(--foreground)'
                  }}
                >
                  {link.label}
                </Link>
              ))}
              <div className="pt-2 flex justify-center">
                <ThemeToggle />
              </div>
            </div>
          </div>
        )}
      </div>
    </nav>
  );
}