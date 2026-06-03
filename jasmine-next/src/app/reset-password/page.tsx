'use client';

import { useState, Suspense } from 'react';
import { useSearchParams } from 'next/navigation';
import Link from 'next/link';
import { Lock, Eye, EyeOff, CheckCircle, Loader2, AlertCircle } from 'lucide-react';
import { motion } from 'framer-motion';

function ResetPasswordForm() {
  const searchParams = useSearchParams();
  const token = searchParams.get('token') || '';
  const userId = searchParams.get('userId') || '';
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    if (!token || !userId) { setError('Invalid reset link. Please request a new one.'); return; }
    if (newPassword.length < 8) { setError('Password must be at least 8 characters.'); return; }
    if (newPassword !== confirmPassword) { setError('Passwords do not match.'); return; }
    setLoading(true);
    try {
      const res = await fetch('/api/password/reset-confirm', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ userId, token, newPassword }),
      });
      const data = await res.json();
      if (!res.ok) { setError(data.error || 'Failed to reset password.'); return; }
      setSuccess(true);
    } catch { setError('Something went wrong.'); }
    finally { setLoading(false); }
  };

  if (!token || !userId) {
    return (
      <div className="min-h-screen flex items-center justify-center p-4" style={{ backgroundColor: 'var(--background-alt)' }}>
        <div className="premium-card p-8 max-w-md w-full text-center">
          <div className="w-16 h-16 mx-auto mb-4 rounded-full flex items-center justify-center" style={{ backgroundColor: 'var(--risk-high-bg)' }}>
            <AlertCircle className="w-8 h-8" style={{ color: 'var(--risk-high)' }} />
          </div>
          <h2 className="text-2xl font-bold mb-2" style={{ color: 'var(--foreground)' }}>Invalid Reset Link</h2>
          <p className="mb-6" style={{ color: 'var(--text-muted)' }}>This link is invalid or has expired.</p>
          <Link href="/forgot-password" className="premium-btn premium-btn-primary">Request New Link</Link>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex items-center justify-center p-4" style={{ backgroundColor: 'var(--background-alt)' }}>
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="w-full max-w-md">
        <div className="premium-card p-8">
          {success ? (
            <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} className="text-center">
              <div className="w-16 h-16 mx-auto mb-4 rounded-full flex items-center justify-center" style={{ backgroundColor: 'var(--risk-low-bg)' }}>
                <CheckCircle className="w-8 h-8" style={{ color: 'var(--risk-low)' }} />
              </div>
              <h2 className="text-2xl font-bold mb-2" style={{ color: 'var(--foreground)' }}>Password Reset!</h2>
              <p className="mb-6" style={{ color: 'var(--text-muted)' }}>You can now log in with your new password.</p>
              <Link href="/login" className="premium-btn premium-btn-primary">Log In</Link>
            </motion.div>
          ) : (
            <>
              <div className="text-center mb-6">
                <div className="w-16 h-16 mx-auto mb-4 rounded-full flex items-center justify-center" style={{ background: 'var(--gradient-primary-subtle)' }}>
                  <Lock className="w-8 h-8" style={{ color: 'var(--primary)' }} />
                </div>
                <h2 className="text-2xl font-bold mb-2" style={{ color: 'var(--foreground)' }}>Set New Password</h2>
                <p style={{ color: 'var(--text-muted)' }}>Enter your new password below.</p>
              </div>

              {error && (
                <div className="mb-4 p-3 rounded-xl text-sm" style={{ backgroundColor: 'var(--risk-high-bg)', color: 'var(--risk-high)' }}>{error}</div>
              )}

              <form onSubmit={handleSubmit} className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-1.5" style={{ color: 'var(--foreground)' }}>New Password</label>
                  <div className="relative">
                    <Lock className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4" style={{ color: 'var(--text-dim)' }} />
                    <input type={showPassword ? 'text' : 'password'} value={newPassword} onChange={(e) => setNewPassword(e.target.value)} placeholder="Enter new password" required className="premium-input pl-11 pr-11" />
                    <button type="button" onClick={() => setShowPassword(!showPassword)} className="absolute right-3 top-1/2 -translate-y-1/2 p-1 rounded-md" style={{ color: 'var(--text-dim)' }}>
                      {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                    </button>
                  </div>
                  <p className="mt-1 text-xs" style={{ color: 'var(--text-dim)' }}>Must be at least 8 characters</p>
                </div>
                <div>
                  <label className="block text-sm font-medium mb-1.5" style={{ color: 'var(--foreground)' }}>Confirm New Password</label>
                  <div className="relative">
                    <Lock className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4" style={{ color: 'var(--text-dim)' }} />
                    <input type={showPassword ? 'text' : 'password'} value={confirmPassword} onChange={(e) => setConfirmPassword(e.target.value)} placeholder="Confirm new password" required className="premium-input pl-11" />
                  </div>
                </div>
                <button type="submit" disabled={loading} className="premium-btn premium-btn-primary w-full py-3.5">
                  {loading ? <><Loader2 className="w-5 h-5 animate-spin" /> Resetting...</> : 'Reset Password'}
                </button>
              </form>
            </>
          )}
        </div>
      </motion.div>
    </div>
  );
}

export default function ResetPasswordPage() {
  return (
    <Suspense fallback={
      <div className="min-h-screen flex items-center justify-center" style={{ backgroundColor: 'var(--background-alt)' }}>
        <div className="w-8 h-8 border-2 rounded-full animate-spin" style={{ borderColor: 'var(--primary-light)', borderTopColor: 'var(--primary)' }} />
      </div>
    }>
      <ResetPasswordForm />
    </Suspense>
  );
}
