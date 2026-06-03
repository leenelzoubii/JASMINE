'use client';

import { useSearchParams, useRouter } from 'next/navigation';
import { Suspense, useState } from 'react';
import { Mail, CheckCircle, AlertCircle } from 'lucide-react';
import { motion } from 'framer-motion';
import { sendFirebaseResetPasswordEmail } from '@/lib/auth';

function ConfirmResetContent() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const email = searchParams.get('email') || '';
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');

  const handleYes = async () => {
    if (!email) { setError('Email is missing.'); return; }
    setLoading(true);
    setError(''); setMessage('');
    try {
      await sendFirebaseResetPasswordEmail(email);
      setMessage('Password reset email has been sent.');
      setTimeout(() => router.push('/login'), 3000);
    } catch {
      setError('Could not send reset email. Please try again.');
    } finally { setLoading(false); }
  };

  return (
    <div className="min-h-screen flex items-center justify-center px-4 py-12" style={{ backgroundColor: 'var(--background-alt)' }}>
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="w-full max-w-md">
        <div className="premium-card p-8">
          <div className="text-center mb-8">
            <div className="w-14 h-14 mx-auto rounded-xl flex items-center justify-center mb-4" style={{ background: 'var(--gradient-primary)' }}>
              <Mail className="w-7 h-7 text-white" />
            </div>
            <h1 className="text-2xl font-bold" style={{ color: 'var(--foreground)' }}>Password Reset Request</h1>
            <p className="mt-2" style={{ color: 'var(--text-muted)' }}>Did you request to reset the password for this account?</p>
            <p className="font-medium mt-3" style={{ color: 'var(--primary)' }}>{email}</p>
          </div>

          {message && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="mb-4 p-4 rounded-xl flex items-center gap-3" style={{ backgroundColor: 'var(--risk-low-bg)' }}>
              <CheckCircle className="w-5 h-5" style={{ color: 'var(--risk-low)' }} />
              <span className="text-sm" style={{ color: 'var(--risk-low)' }}>{message}</span>
            </motion.div>
          )}

          {error && (
            <div className="mb-4 p-4 rounded-xl flex items-center gap-3" style={{ backgroundColor: 'var(--risk-high-bg)' }}>
              <AlertCircle className="w-5 h-5" style={{ color: 'var(--risk-high)' }} />
              <span className="text-sm" style={{ color: 'var(--risk-high)' }}>{error}</span>
            </div>
          )}

          <div className="space-y-3">
            <button onClick={handleYes} disabled={loading} className="premium-btn premium-btn-primary w-full py-3.5">
              {loading ? 'Sending...' : 'Yes, this is me'}
            </button>
            <button onClick={() => router.push('/login')} disabled={loading} className="premium-btn premium-btn-ghost w-full py-3.5">
              No, go back to login
            </button>
          </div>
        </div>
      </motion.div>
    </div>
  );
}

export default function ConfirmResetPage() {
  return (
    <Suspense fallback={
      <div className="min-h-screen flex items-center justify-center" style={{ backgroundColor: 'var(--background-alt)' }}>
        <div className="w-8 h-8 border-2 rounded-full animate-spin" style={{ borderColor: 'var(--primary-light)', borderTopColor: 'var(--primary)' }} />
      </div>
    }>
      <ConfirmResetContent />
    </Suspense>
  );
}
