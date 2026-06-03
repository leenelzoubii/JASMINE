'use client';

import Link from 'next/link';
import { Suspense, useState, useEffect } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { Brain, Mail, Lock, Eye, EyeOff, AlertCircle, Sparkles, Shield, Activity } from 'lucide-react';
import { motion } from 'framer-motion';
import { authenticateUser } from '@/lib/auth';

function LoginForm() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [loggedOut, setLoggedOut] = useState(false);
  const router = useRouter();
  const searchParams = useSearchParams();
  const returnUrl = searchParams.get('returnUrl') || null;

  useEffect(() => {
    if (searchParams.get('loggedout') === 'true') {
      setLoggedOut(true);
    }
  }, [searchParams]);
  const validatePassword = (password: string) => {
    if (password.length <= 12) {
      return 'Password must be more than 12 characters.';
    }

    if (!/[A-Z]/.test(password)) {
      return 'Password must contain at least one uppercase letter.';
    }

    return '';
  };
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    const passwordError = validatePassword(password);
    if (passwordError) {
      setError(passwordError);
      return;
    }
    setLoading(true);

    try {
      const user = await authenticateUser(email, password);
      if (!user) {
        setError('Invalid email or password');
        return;
      }
      if (returnUrl && (returnUrl.startsWith('/professional') || returnUrl.startsWith('/parent'))) {
        router.push(returnUrl);
      } else if (user.role === 'parent') {
        router.push('/parent');
      } else {
        router.push('/professional');
      }
    } catch (error: unknown) {
      const err = error as { code?: string; message?: string };
      if (err.code === 'auth/invalid-credential' || err.code === 'auth/user-not-found' || err.code === 'auth/wrong-password') {
        setError('Invalid email or password. Try the demo accounts below.');
      } else {
        setError(err.message || 'Something went wrong.');
      }
    } finally {
      setLoading(false);
    }
  };

  const quickLogin = async (role: 'parent' | 'professional') => {
    const creds = role === 'parent'
      ? { email: 'parent@demo.com', password: 'demo123' }
      : { email: 'doctor@demo.com', password: 'demo123' };
    setEmail(creds.email);
    setPassword(creds.password);
    setError('');
    setLoading(true);
    try {
      const user = await authenticateUser(creds.email, creds.password);
      if (user) router.push(role === 'parent' ? '/parent' : '/professional');
    } catch {
      router.push(role === 'parent' ? '/parent' : '/professional');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex">
      {/* Left - Brand Panel */}
      <div className="hidden lg:flex lg:w-1/2 relative overflow-hidden items-center justify-center p-12"
        style={{ background: 'var(--gradient-hero)' }}>
        <div className="absolute inset-0 opacity-[0.04]">
          <div className="absolute top-1/4 left-1/4 w-96 h-96 rounded-full bg-blue-400 blur-[120px]" />
          <div className="absolute bottom-1/4 right-1/4 w-80 h-80 rounded-full bg-emerald-400 blur-[100px]" />
        </div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="relative z-10 text-center max-w-md"
        >
          <motion.div
            animate={{ rotate: [0, -5, 0] }}
            transition={{ repeat: Infinity, duration: 6, ease: 'easeInOut' }}
            className="w-20 h-20 rounded-2xl flex items-center justify-center mx-auto mb-8"
            style={{ background: 'var(--gradient-primary)' }}
          >
            <Brain className="w-10 h-10 text-white" />
          </motion.div>
          <h2 className="text-4xl font-bold mb-4" style={{ color: 'var(--text-primary)' }}>Welcome Back</h2>
          <p className="text-lg mb-8" style={{ color: 'var(--text-dim)' }}>
            Sign in to continue your autism screening workflow
          </p>
          <div className="space-y-4 text-left">
            {[
              { icon: Sparkles, text: 'AI-powered ensemble predictions' },
              { icon: Shield, text: 'Privacy-preserving analysis' },
              { icon: Activity, text: 'Real-time pipeline streaming' },
            ].map((item, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.3 + i * 0.1 }}
                className="flex items-center gap-3"
              >
                <div className="w-8 h-8 rounded-lg flex items-center justify-center" style={{ background: 'var(--primary-light)' }}>
                  <item.icon className="w-4 h-4" style={{ color: 'var(--primary)' }} />
                </div>
                <span style={{ color: 'var(--text-dim)' }}>{item.text}</span>
              </motion.div>
            ))}
          </div>
        </motion.div>
      </div>

      {/* Right - Form */}
      <div className="flex-1 flex items-center justify-center px-4 py-12" style={{ backgroundColor: 'var(--background-alt)' }}>
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="w-full max-w-md"
        >
          <div className="premium-card p-8">
            <div className="text-center mb-8">
              <Link href="/" className="inline-flex items-center gap-2 mb-4">
                <motion.div
                  whileHover={{ scale: 1.05, rotate: -3 }}
                  className="w-12 h-12 rounded-xl flex items-center justify-center"
                  style={{ background: 'var(--gradient-primary)' }}
                >
                  <Brain className="w-7 h-7 text-white" />
                </motion.div>
              </Link>
              <h1 className="text-2xl font-bold" style={{ color: 'var(--foreground)' }}>Sign In</h1>
              <p style={{ color: 'var(--text-muted)' }}>Access your JASMINE account</p>
            </div>

            {error && (
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                className="mb-6 p-4 rounded-xl flex items-start gap-3"
                style={{ backgroundColor: 'var(--risk-high-bg)', border: '1px solid var(--risk-high)' }}
              >
                <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5" style={{ color: 'var(--risk-high)' }} />
                <span className="text-sm" style={{ color: 'var(--risk-high)' }}>{error}</span>
              </motion.div>
            )}

            {loggedOut && (
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                className="mb-6 p-4 rounded-xl"
                style={{ backgroundColor: 'var(--risk-low-bg)', border: '1px solid var(--risk-low)', color: 'var(--risk-low)' }}
              >
                You have been logged out successfully.
              </motion.div>
            )}

            <form onSubmit={handleSubmit} className="space-y-5">
              <div>
                <label className="block text-sm font-medium mb-1.5" style={{ color: 'var(--foreground)' }}>
                  Email
                </label>
                <div className="relative">
                  <Mail className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4" style={{ color: 'var(--text-dim)' }} />
                  <input
                    type="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    placeholder="doctor@example.com"
                    className="premium-input pl-11"
                  />
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium mb-1.5" style={{ color: 'var(--foreground)' }}>
                  Password
                </label>
                <div className="relative">
                  <Lock className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4" style={{ color: 'var(--text-dim)' }} />
                  <input
                    type={showPassword ? 'text' : 'password'}
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    placeholder="Enter your password"
                    className="premium-input pl-11 pr-11"
                    minLength={13}
                    pattern="(?=.*[A-Z]).{13,}"
                    title="Password must be more than 12 characters and contain at least one uppercase letter."
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    className="absolute right-3 top-1/2 -translate-y-1/2 p-1.5 rounded-lg transition-colors"
                    style={{ color: 'var(--text-dim)' }}
                  >
                    {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                  </button>
                </div>
              </div>

              <motion.button
                type="submit"
                disabled={loading}
                whileHover={{ scale: 1.01 }}
                whileTap={{ scale: 0.99 }}
                className="premium-btn premium-btn-primary w-full py-3.5"
              >
                {loading ? (
                  <span className="flex items-center gap-2">
                    <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    Signing in...
                  </span>
                ) : 'Sign In'}
              </motion.button>
            </form>

            <div className="mt-6 pt-6 border-t" style={{ borderColor: 'var(--border-light)' }}>
              <p className="text-sm font-medium text-center mb-3" style={{ color: 'var(--text-muted)' }}>
                Demo Quick Access
              </p>
              <div className="grid grid-cols-2 gap-3">
                <motion.button
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={() => quickLogin('parent')}
                  className="py-2.5 px-4 rounded-xl text-sm font-medium transition-all"
                  style={{ backgroundColor: 'var(--primary-light)', color: 'var(--primary)' }}
                >
                  Parent Demo
                </motion.button>
                <motion.button
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={() => quickLogin('professional')}
                  className="py-2.5 px-4 rounded-xl text-sm font-medium transition-all"
                  style={{ backgroundColor: 'var(--primary-light)', color: 'var(--primary)' }}
                >
                  Doctor Demo
                </motion.button>
              </div>
            </div>

            <div className="mt-6 text-center space-y-2">
              <p style={{ color: 'var(--text-muted)' }}>
                Don&apos;t have an account?{' '}
                <Link href="/register" className="font-semibold hover:underline" style={{ color: 'var(--primary)' }}>
                  Create one
                </Link>
              </p>
              <Link href="/forgot-password" className="block text-sm transition-colors" style={{ color: 'var(--text-dim)' }}>
                Forgot your password?
              </Link>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
}

export default function LoginPage() {
  return (
    <Suspense fallback={
      <div className="min-h-screen flex items-center justify-center" style={{ backgroundColor: 'var(--background-alt)' }}>
        <div className="w-8 h-8 border-2 rounded-full animate-spin" style={{ borderColor: 'var(--primary-light)', borderTopColor: 'var(--primary)' }} />
      </div>
    }>
      <LoginForm />
    </Suspense>
  );
}
