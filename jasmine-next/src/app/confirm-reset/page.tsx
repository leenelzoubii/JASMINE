'use client';

import { useSearchParams, useRouter } from 'next/navigation';
import { Suspense, useState } from 'react';
import { Mail, CheckCircle, AlertCircle } from 'lucide-react';
import { sendFirebaseResetPasswordEmail } from '@/lib/auth';

function ConfirmResetContent() {
  const searchParams = useSearchParams();
  const router = useRouter();

  const email = searchParams.get('email') || '';

  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');

  const handleYes = async () => {
    if (!email) {
      setError('Email is missing.');
      return;
    }

    setLoading(true);
    setError('');
    setMessage('');

    try {
      await sendFirebaseResetPasswordEmail(email);

      setMessage('Password reset email has been sent. Please check your inbox.');
      setTimeout(() => {
        router.push('/login');
      }, 3000);
    } catch (err) {
      console.error(err);
      setError('Could not send reset email. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleNo = () => {
    router.push('/login');
  };

  return (
    <div className="min-h-screen flex items-center justify-center px-4 py-12" style={{ backgroundColor: 'var(--background)' }}>
      <div className="w-full max-w-md rounded-2xl shadow-xl p-8" style={{ backgroundColor: 'var(--background)', border: '1px solid var(--border)' }}>
        <div className="text-center mb-8">
          <div className="w-14 h-14 mx-auto rounded-xl bg-primary flex items-center justify-center mb-4">
            <Mail className="w-7 h-7 text-white" />
          </div>

          <h1 className="text-2xl font-bold" style={{ color: 'var(--foreground)' }}>
            Password Reset Request
          </h1>

          <p className="mt-2" style={{ color: 'var(--text-muted)' }}>
            Did you request to reset the password for this account?
          </p>

          <p className="text-primary dark:text-primary-light font-medium mt-3">
            {email}
          </p>
        </div>

        {message && (
          <div className="mb-4 p-4 rounded-xl flex items-center gap-2" style={{ backgroundColor: 'var(--color-green-50)', color: 'var(--color-green-700)' }}>
            <CheckCircle className="w-5 h-5" />
            <span className="text-sm">{message}</span>
          </div>
        )}

        {error && (
          <div className="mb-4 p-4 rounded-xl flex items-center gap-2" style={{ backgroundColor: 'rgba(220, 38, 38, 0.1)', color: '#dc2626' }}>
            <AlertCircle className="w-5 h-5" />
            <span className="text-sm">{error}</span>
          </div>
        )}

        <div className="space-y-3">
          <button
            onClick={handleYes}
            disabled={loading}
            className="w-full py-3 rounded-xl bg-primary hover:bg-primary-dark text-white font-medium transition-all disabled:opacity-50"
          >
            {loading ? 'Sending reset email...' : 'Yes, this is me'}
          </button>

          <button
            onClick={handleNo}
            disabled={loading}
            className="w-full py-3 rounded-xl font-medium transition-all disabled:opacity-50 hover:bg-gray-100 dark:hover:bg-dark-deep"
            style={{ backgroundColor: 'transparent', border: '1px solid var(--border)', color: 'var(--foreground)' }}
          >
            No, go back to login
          </button>
        </div>
      </div>
    </div>
  );
}

export default function ConfirmResetPage() {
  return (
    <Suspense fallback={<div className="min-h-screen bg-dark-bg" />}>
      <ConfirmResetContent />
    </Suspense>
  );
  
}