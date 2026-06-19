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
    <div className="min-h-screen flex items-center justify-center px-4 py-12 bg-dark-bg">
      <div className="w-full max-w-md bg-dark-surface rounded-2xl shadow-xl p-8 border border-dark-deep">
        <div className="text-center mb-8">
          <div className="w-14 h-14 mx-auto rounded-xl bg-primary flex items-center justify-center mb-4">
            <Mail className="w-7 h-7 text-white" />
          </div>

          <h1 className="text-2xl font-bold text-white">
            Password Reset Request
          </h1>

          <p className="text-gray-400 mt-2">
            Did you request to reset the password for this account?
          </p>

          <p className="text-primary font-medium mt-3">
            {email}
          </p>
        </div>

        {message && (
          <div className="mb-4 p-4 rounded-xl bg-green-900/20 text-green-400 flex items-center gap-2">
            <CheckCircle className="w-5 h-5" />
            <span className="text-sm">{message}</span>
          </div>
        )}

        {error && (
          <div className="mb-4 p-4 rounded-xl bg-red-900/20 text-red-400 flex items-center gap-2">
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
            className="w-full py-3 rounded-xl border border-gray-600 text-gray-300 hover:bg-white/5 font-medium transition-all"
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