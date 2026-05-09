'use client';

import Link from 'next/link';
import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { Brain, Mail, Lock, User, Building2, AlertCircle, Check } from 'lucide-react';
import { motion } from 'framer-motion';
import { registerUser } from "@/lib/auth";

const roles = [
  { id: 'professional', title: 'Healthcare Professional', icon: '🏥', description: 'Doctors, therapists, and specialists' },
  { id: 'parent', title: 'Parent/Guardian', icon: '👨‍👩‍👧', description: 'Parents monitoring their child' },
];

export default function RegisterPage() {
  const [step, setStep] = useState(1);
  const [selectedRole, setSelectedRole] = useState('');
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [specialty, setSpecialty] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const router = useRouter();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");

    if (step === 1) {
      if (!selectedRole) {
        setError("Please select an account type");
        return;
      }
      setStep(2);
      return;
    }

    if (!name || !email || !password) {
      setError("Please fill in all required fields");
      return;
    }

    setLoading(true);

    try {
      const user = await registerUser(
        name,
        email,
        password,
        selectedRole as "parent" | "professional"
      );

      if (user.role === "parent") {
        router.push("/parent");
      } else {
        router.push("/professional");
      }
    } catch (error) {
      console.error(error);
      setError("Registration failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center px-4 py-12" style={{ backgroundColor: 'var(--background-alt)' }}>
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="w-full max-w-md"
      >
        <div className="rounded-2xl shadow-xl p-8" style={{ backgroundColor: 'var(--background)', border: '1px solid var(--border)' }}>
          <div className="text-center mb-8">
            <Link href="/" className="inline-flex items-center gap-2 mb-4">
              <div className="w-12 h-12 rounded-xl flex items-center justify-center" style={{ backgroundColor: 'var(--primary)' }}>
                <Brain className="w-7 h-7 text-white" />
              </div>
            </Link>
            <h1 className="text-2xl font-bold" style={{ color: 'var(--foreground)' }}>
              {step === 1 ? 'Create Account' : 'Complete Registration'}
            </h1>
            <p style={{ color: 'var(--text-muted)' }}>
              {step === 1 ? 'Choose your account type' : 'Fill in your details'}
            </p>
          </div>

          {error && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              className="mb-6 p-4 rounded-xl flex items-center gap-2"
              style={{ backgroundColor: 'rgba(220, 38, 38, 0.1)', border: '1px solid #dc2626', color: '#dc2626' }}
            >
              <AlertCircle className="w-5 h-5 flex-shrink-0" />
              <span className="text-sm">{error}</span>
            </motion.div>
          )}

          {step === 1 ? (
            <div className="space-y-4">
              {roles.map((role) => (
                <button
                  key={role.id}
                  onClick={() => setSelectedRole(role.id)}
                  className={`w-full p-5 rounded-xl border-2 transition-all text-left ${selectedRole === role.id ? 'border-[var(--primary)]' : 'border-[var(--border)]'}`}
                  style={{
                    backgroundColor: selectedRole === role.id ? 'var(--primary-light)' : 'var(--background)'
                  }}
                >
                  <div className="flex items-center gap-4">
                    <span className="text-3xl">{role.icon}</span>
                    <div>
                      <h3 className="font-semibold" style={{ color: 'var(--foreground)' }}>{role.title}</h3>
                      <p className="text-sm" style={{ color: 'var(--text-muted)' }}>{role.description}</p>
                    </div>
                    {selectedRole === role.id && (
                      <Check className="w-6 h-6 ml-auto" style={{ color: 'var(--primary)' }} />
                    )}
                  </div>
                </button>
              ))}
            </div>
          ) : (
            <form onSubmit={handleSubmit} className="space-y-5">
              <div>
                <label className="block text-sm font-medium mb-2" style={{ color: 'var(--foreground)' }}>
                  Full Name
                </label>
                <div className="relative">
                  <User className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5" style={{ color: 'var(--text-muted)' }} />
                  <input
                    type="text"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    placeholder="Your name"
                    className="w-full pl-12 pr-4 py-3 rounded-xl focus:outline-none focus:ring-2"
                    style={{
                      backgroundColor: 'var(--background)',
                      border: '1px solid var(--border)',
                      color: 'var(--foreground)'
                    }}
                  />
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium mb-2" style={{ color: 'var(--foreground)' }}>
                  Email
                </label>
                <div className="relative">
                  <Mail className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5" style={{ color: 'var(--text-muted)' }} />
                  <input
                    type="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    placeholder="you@example.com"
                    className="w-full pl-12 pr-4 py-3 rounded-xl focus:outline-none focus:ring-2"
                    style={{
                      backgroundColor: 'var(--background)',
                      border: '1px solid var(--border)',
                      color: 'var(--foreground)'
                    }}
                  />
                </div>
              </div>

              {selectedRole === 'professional' && (
                <div>
                  <label className="block text-sm font-medium mb-2" style={{ color: 'var(--foreground)' }}>
                    Specialty (Optional)
                  </label>
                  <div className="relative">
                    <Building2 className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5" style={{ color: 'var(--text-muted)' }} />
                    <input
                      type="text"
                      value={specialty}
                      onChange={(e) => setSpecialty(e.target.value)}
                      placeholder="e.g., Pediatric Neurology"
                      className="w-full pl-12 pr-4 py-3 rounded-xl focus:outline-none focus:ring-2"
                      style={{
                        backgroundColor: 'var(--background)',
                        border: '1px solid var(--border)',
                        color: 'var(--foreground)'
                      }}
                    />
                  </div>
                </div>
              )}

              <div>
                <label className="block text-sm font-medium mb-2" style={{ color: 'var(--foreground)' }}>
                  Password
                </label>
                <div className="relative">
                  <Lock className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5" style={{ color: 'var(--text-muted)' }} />
                  <input
                    type="password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    placeholder="••••••••"
                    className="w-full pl-12 pr-4 py-3 rounded-xl focus:outline-none focus:ring-2"
                    style={{
                      backgroundColor: 'var(--background)',
                      border: '1px solid var(--border)',
                      color: 'var(--foreground)'
                    }}
                  />
                </div>
              </div>

              <button
                type="button"
                onClick={() => setStep(1)}
                className="text-sm"
                style={{ color: 'var(--text-muted)' }}
              >
                Back
              </button>
            </form>
          )}

          <button
            onClick={handleSubmit}
            disabled={loading}
            className="w-full mt-6 py-3 font-medium rounded-xl transition-all"
            style={{ backgroundColor: 'var(--primary)', color: 'var(--text-primary)' }}
          >
            {loading ? 'Creating...' : step === 1 ? 'Continue' : 'Create Account'}
          </button>

          <p className="mt-6 text-center" style={{ color: 'var(--text-muted)' }}>
            Already have an account?{' '}
            <Link href="/login" className="font-medium" style={{ color: 'var(--primary)' }}>
              Sign in
            </Link>
          </p>
        </div>
      </motion.div>
    </div>
  );
}