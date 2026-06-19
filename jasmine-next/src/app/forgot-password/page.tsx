'use client';

import { useState } from 'react';
import Link from 'next/link';
import { Mail, ArrowLeft, CheckCircle, Loader2, Brain } from 'lucide-react';
import { motion } from 'framer-motion';
import { sendPasswordResetEmail } from 'firebase/auth';
import { auth } from '@/lib/firebase';

export default function ForgotPasswordPage() {
  const [email, setEmail] = useState('');
  const [loading, setLoading] = useState(false);
  const [sent, setSent] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      await sendPasswordResetEmail(auth, email.trim());
      setSent(true);
    } catch (err: unknown) {
      const firebaseErr = err as { code?: string, message?: string };
      console.error('Forgot password error:', firebaseErr.code, firebaseErr.message);
      if (firebaseErr.code === 'auth/user-not-found') setError('No account found with this email address.');
      else if (firebaseErr.code === 'auth/invalid-email') setError('Please enter a valid email address.');
      else if (firebaseErr.code === 'auth/too-many-requests') setError('Too many requests. Please try again later.');
      else setError(`Something went wrong. Please try again. (${firebaseErr.code || 'unknown'})`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center p-4" style={{ backgroundColor: 'var(--background-alt)' }}>
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="w-full max-w-md">
        <div className="premium-card p-8">
          {sent ? (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="text-center">
              <div className="w-16 h-16 mx-auto mb-4 rounded-full flex items-center justify-center" style={{ backgroundColor: 'var(--risk-low-bg)' }}>
                <CheckCircle className="w-8 h-8" style={{ color: 'var(--risk-low)' }} />
              </div>
              <h2 className="text-2xl font-bold mb-2" style={{ color: 'var(--foreground)' }}>Check your email</h2>
              <p className="mb-6" style={{ color: 'var(--text-muted)' }}>
                We sent a password reset email to <strong>{email}</strong>.
              </p>
              <Link href="/login" className="premium-btn premium-btn-primary">
                Back to Login
              </Link>
            </motion.div>
          ) : (
            <>
              <div className="text-center mb-6">
                <Link href="/login" className="inline-flex items-center gap-2 text-sm mb-6" style={{ color: 'var(--text-muted)' }}>
                  <ArrowLeft className="w-4 h-4" />
                  Back to login
                </Link>
                <div className="w-16 h-16 mx-auto mb-4 rounded-full flex items-center justify-center" style={{ background: 'var(--gradient-primary-subtle)' }}>
                  <Mail className="w-8 h-8" style={{ color: 'var(--primary)' }} />
                </div>
                <h2 className="text-2xl font-bold mb-2" style={{ color: 'var(--foreground)' }}>Reset Password</h2>
                <p style={{ color: 'var(--text-muted)' }}>Enter your email and we&apos;ll send you a reset link.</p>
              </div>

              {error && (
                <div className="mb-4 p-3 rounded-xl text-sm" style={{ backgroundColor: 'var(--risk-high-bg)', color: 'var(--risk-high)' }}>
                  {error}
                </div>
              )}

              <form onSubmit={handleSubmit} className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-1.5" style={{ color: 'var(--foreground)' }}>Email address</label>
                  <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} placeholder="you@example.com" required className="premium-input" />
                </div>
                <button type="submit" disabled={loading} className="premium-btn premium-btn-primary w-full py-3.5">
                  {loading ? <><Loader2 className="w-5 h-5 animate-spin" /> Sending...</> : 'Send Reset Email'}
                </button>
              </form>
            </>
          )}
        </div>
      </motion.div>
    </div>
  );
}
