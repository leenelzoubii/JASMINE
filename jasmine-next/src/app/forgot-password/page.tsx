'use client';

import { useState } from 'react';
import Link from 'next/link';
import { Mail, ArrowLeft, CheckCircle, Loader2 } from 'lucide-react';
import { motion } from 'framer-motion';
import { getAuth, sendPasswordResetEmail } from 'firebase/auth';

const FIREBASE_PROJECT_ID = process.env.NEXT_PUBLIC_FIREBASE_PROJECT_ID || "jasmine-4671c";

async function checkUserExistsInFirestore(email: string): Promise<boolean> {
  try {
    const url = `https://firestore.googleapis.com/v1/projects/${FIREBASE_PROJECT_ID}/databases/(default)/documents:runQuery`;
    const queryBody = {
      structuredQuery: {
        from: [{ collectionId: "users" }],
        where: {
          fieldFilter: {
            field: { fieldPath: "email" },
            op: "EQUAL",
            value: { stringValue: email.toLowerCase().trim() },
          },
        },
        limit: 1,
      },
    };

    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(queryBody),
    });

    const data = await res.json();
    return Array.isArray(data) && data.length > 0 && data[0].document;
  } catch {
    // If check fails, proceed anyway - Firebase Auth will validate
    return true;
  }
}

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
      // Step 1: Check if user exists in Firestore
      const exists = await checkUserExistsInFirestore(email);

      if (!exists) {
        setError('No account found with this email address.');
        setLoading(false);
        return;
      }

      // Step 2: Send Firebase password reset email (works out of box)
      const auth = getAuth();
      await sendPasswordResetEmail(auth, email.trim());

      setSent(true);
    } catch (err: unknown) {
      const firebaseErr = err as { code?: string };
      if (firebaseErr.code === 'auth/user-not-found') {
        setError('No account found with this email address.');
      } else if (firebaseErr.code === 'auth/invalid-email') {
        setError('Please enter a valid email address.');
      } else if (firebaseErr.code === 'auth/too-many-requests') {
        setError('Too many requests. Please try again later.');
      } else {
        setError('Something went wrong. Please try again.');
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center p-4" style={{ backgroundColor: 'var(--background)' }}>
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="w-full max-w-md"
      >
        <div className="text-center mb-8">
          <Link href="/login" className="inline-flex items-center gap-2 text-sm mb-6" style={{ color: 'var(--text-muted)' }}>
            <ArrowLeft className="w-4 h-4" />
            Back to login
          </Link>
        </div>

        <div className="p-8 rounded-2xl border shadow-xl" style={{ backgroundColor: 'var(--background)', borderColor: 'var(--border)' }}>
          {sent ? (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="text-center">
              <div className="w-16 h-16 mx-auto mb-4 rounded-full flex items-center justify-center" style={{ backgroundColor: 'rgba(22, 163, 74, 0.1)' }}>
                <CheckCircle className="w-8 h-8" style={{ color: '#16a34a' }} />
              </div>
              <h2 className="text-2xl font-bold mb-2" style={{ color: 'var(--foreground)' }}>Check your email</h2>
              <p className="mb-6" style={{ color: 'var(--text-muted)' }}>
                We sent a password reset email to <strong>{email}</strong>. Check your spam folder if you don't see it.
              </p>
              <Link href="/login" className="inline-flex items-center gap-2 px-6 py-3 text-white font-medium rounded-xl transition-all" style={{ backgroundColor: 'var(--primary)' }}>
                Back to Login
              </Link>
            </motion.div>
          ) : (
            <>
              <div className="text-center mb-6">
                <div className="w-16 h-16 mx-auto mb-4 rounded-full flex items-center justify-center" style={{ backgroundColor: 'var(--primary-light)' }}>
                  <Mail className="w-8 h-8" style={{ color: 'var(--primary)' }} />
                </div>
                <h2 className="text-2xl font-bold mb-2" style={{ color: 'var(--foreground)' }}>Reset Password</h2>
                <p style={{ color: 'var(--text-muted)' }}>
                  Enter your email and we&apos;ll send you a reset link.
                </p>
              </div>

              {error && (
                <div className="mb-4 p-3 rounded-xl text-sm" style={{ backgroundColor: 'rgba(220, 38, 38, 0.1)', color: '#dc2626' }}>
                  {error}
                </div>
              )}

              <form onSubmit={handleSubmit} className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-2" style={{ color: 'var(--foreground)' }}>
                    Email address
                  </label>
                  <input
                    type="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    placeholder="you@example.com"
                    required
                    className="w-full px-4 py-3 rounded-xl focus:outline-none focus:ring-2"
                    style={{ backgroundColor: 'var(--background)', border: '1px solid var(--border)', color: 'var(--foreground)' }}
                  />
                </div>

                <button
                  type="submit"
                  disabled={loading}
                  className="w-full px-4 py-3 text-white font-medium rounded-xl transition-all flex items-center justify-center gap-2 disabled:opacity-50"
                  style={{ backgroundColor: 'var(--primary)' }}
                >
                  {loading ? (
                    <>
                      <Loader2 className="w-5 h-5 animate-spin" />
                      Sending...
                    </>
                  ) : (
                    'Send Reset Email'
                  )}
                </button>
              </form>
            </>
          )}
        </div>
      </motion.div>
    </div>
  );
}
